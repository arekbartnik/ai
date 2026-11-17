package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/arekbartnik/ai/aitrace"
	"github.com/arekbartnik/ai/internal/httpapi"
)

const (
	DefaultBaseURL = "https://generativelanguage.googleapis.com"

	Flex           = "flex"
	flexRetryDelay = 2 * time.Second
)

// Config is set by the caller. An empty BaseURL is DefaultBaseURL, and an
// empty Model leaves the model to each Request.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Store       bool
	ServiceTier string
}

// APIError is a non-200 reply from Gemini.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("Gemini returned %d: %s", e.Status, e.Body)
}

type Client struct {
	api         httpapi.Endpoint
	model       string
	store       bool
	serviceTier string
}

func New(c Config) *Client {
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	c.BaseURL = strings.TrimSuffix(c.BaseURL, "/")
	return &Client{model: c.Model, store: c.Store, serviceTier: c.ServiceTier, api: httpapi.Endpoint{
		Provider:   "Gemini",
		URL:        c.BaseURL + "/v1beta/interactions",
		AuthHeader: "x-goog-api-key",
		AuthValue:  c.APIKey,
		APIError:   func(status int, body string) error { return &APIError{Status: status, Body: body} },
	}}
}

// Request is one interaction. Input is the user prompt, Instructions the
// system prompt. A nil Schema returns free text.
// PreviousInteractionID continues an earlier stored interaction's
// conversation and requires Config.Store.
// An empty ThinkingLevel leaves the model's default. Not every model
// accepts every level.
type Request struct {
	Model                 string
	Instructions          string
	Input                 string
	PreviousInteractionID string
	Tools                 []Tool
	Schema                json.RawMessage
	ThinkingLevel         ThinkingLevel
	Temperature           *float64
	MaxOutputTokens       int
}

type ThinkingLevel string

var ThinkingLevels = []ThinkingLevel{"minimal", "low", "medium", "high"}

type Response struct {
	ID          string
	Model       string
	ServiceTier string
	Text        string
	// Sources lists the URLs the model cited, in first-cited order. Search
	// citations are Google redirect URLs, not the pages themselves.
	Sources []string
	Usage   Usage
	Raw     json.RawMessage
}

type Usage struct {
	InputTokens     int `json:"total_input_tokens"`
	OutputTokens    int `json:"total_output_tokens"`
	TotalTokens     int `json:"total_tokens"`
	CachedTokens    int `json:"total_cached_tokens"`
	ReasoningTokens int `json:"total_thought_tokens"`
	WebSearchCalls  int `json:"-"`
}

func (r Response) Decode(v any) error {
	if err := json.Unmarshal([]byte(r.Text), v); err != nil {
		return fmt.Errorf("decode Gemini structured output: %w", err)
	}
	return nil
}

var errNoModel = errors.New("Gemini request has no model: set Request.Model or Config.Model")

// Model is the model a Request without its own Model uses. It is empty
// when Config named none.
func (c *Client) Model() string { return c.model }

// Generate retries a flex request on the standard tier when flex has no
// capacity.
func (c *Client) Generate(ctx context.Context, req Request) (Response, error) {
	w := c.wire(req)
	if w.Model == "" {
		return Response{}, errNoModel
	}
	r, err := c.generate(ctx, w)
	var apiErr *APIError
	if w.ServiceTier != Flex || !errors.As(err, &apiErr) || apiErr.Status != http.StatusTooManyRequests && apiErr.Status != http.StatusServiceUnavailable {
		return r, err
	}
	select {
	case <-ctx.Done():
		return Response{}, ctx.Err()
	case <-time.After(flexRetryDelay):
	}
	w.ServiceTier = ""
	return c.generate(ctx, w)
}

func (c *Client) generate(ctx context.Context, w wireRequest) (r Response, err error) {
	ctx, gen := aitrace.Start(ctx, w.call())
	defer func() { gen.End(r.trace(), err) }()
	body, err := c.api.Post(ctx, w)
	if err != nil {
		return Response{}, err
	}
	var interaction wireInteraction
	if err := json.Unmarshal(body, &interaction); err != nil {
		return Response{}, fmt.Errorf("decode Gemini response: %w", err)
	}
	return interaction.response(body)
}

// Stream is Generate with each text delta passed to onDelta as it arrives.
// The returned Response is the same complete result Generate would give,
// except Raw, which lacks the steps because Google's final stream event
// leaves them out.
// A non-nil error from onDelta stops the stream and is returned.
func (c *Client) Stream(ctx context.Context, req Request, onDelta func(string) error) (r Response, err error) {
	w := c.wire(req)
	if w.Model == "" {
		return Response{}, errNoModel
	}
	w.Stream = true
	ctx, gen := aitrace.Start(ctx, w.call())
	defer func() { gen.End(r.trace(), err) }()
	return readStream(c.api.Stream(ctx, w), onDelta)
}

func readStream(lines iter.Seq2[[]byte, error], onDelta func(string) error) (Response, error) {
	var steps []wireStep
	for line, err := range lines {
		if err != nil {
			return Response{}, err
		}
		// Google can end a started stream with a bare JSON error line
		// instead of an error event, for example on a recitation block.
		line, ok := bytes.CutPrefix(line, []byte("data: "))
		if !ok && !bytes.HasPrefix(line, []byte("{")) || string(line) == "[DONE]" {
			continue
		}
		var ev struct {
			EventType   string          `json:"event_type"`
			Index       int             `json:"index"`
			Step        wireStep        `json:"step"`
			Delta       wireContent     `json:"delta"`
			Interaction json.RawMessage `json:"interaction"`
			Error       *wireError      `json:"error"`
		}
		if err := json.Unmarshal(line, &ev); err != nil {
			return Response{}, fmt.Errorf("decode Gemini stream event: %w", err)
		}
		if ev.Error != nil {
			return Response{}, ev.Error
		}
		switch ev.EventType {
		case "step.start":
			for len(steps) <= ev.Index {
				steps = append(steps, wireStep{})
			}
			steps[ev.Index] = wireStep{Type: ev.Step.Type, Content: []wireContent{{Type: "text"}}}
		case "step.delta":
			if ev.Index >= len(steps) || steps[ev.Index].Type != "model_output" {
				continue
			}
			c := &steps[ev.Index].Content[0]
			switch ev.Delta.Type {
			case "text":
				c.Text += ev.Delta.Text
				if err := onDelta(ev.Delta.Text); err != nil {
					return Response{}, err
				}
			case "text_annotation_delta":
				c.Annotations = append(c.Annotations, ev.Delta.Annotations...)
			}
		case "interaction.completed":
			var w wireInteraction
			if err := json.Unmarshal(ev.Interaction, &w); err != nil {
				return Response{}, fmt.Errorf("decode Gemini response: %w", err)
			}
			w.Steps = steps
			return w.response(ev.Interaction)
		}
	}
	return Response{}, errors.New("Gemini stream ended without a completed interaction")
}

func (c *Client) wire(req Request) wireRequest {
	w := wireRequest{
		Model:                 req.Model,
		SystemInstruction:     req.Instructions,
		Input:                 req.Input,
		PreviousInteractionID: req.PreviousInteractionID,
		Store:                 c.store,
		ServiceTier:           c.serviceTier,
		GenerationConfig: &wireGenerationConfig{
			ThinkingLevel:   req.ThinkingLevel,
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxOutputTokens,
		},
	}
	if w.Model == "" {
		w.Model = c.model
	}
	if *w.GenerationConfig == (wireGenerationConfig{}) {
		w.GenerationConfig = nil
	}
	for _, t := range req.Tools {
		w.Tools = append(w.Tools, t.tool())
	}
	if req.Schema != nil {
		w.ResponseFormat = &wireResponseFormat{Type: "text", MimeType: "application/json", Schema: req.Schema}
	}
	return w
}

type wireRequest struct {
	Model                 string                `json:"model"`
	SystemInstruction     string                `json:"system_instruction,omitempty"`
	Input                 string                `json:"input"`
	PreviousInteractionID string                `json:"previous_interaction_id,omitempty"`
	Store                 bool                  `json:"store"`
	Tools                 []wireTool            `json:"tools,omitempty"`
	ResponseFormat        *wireResponseFormat   `json:"response_format,omitempty"`
	GenerationConfig      *wireGenerationConfig `json:"generation_config,omitempty"`
	ServiceTier           string                `json:"service_tier,omitempty"`
	Stream                bool                  `json:"stream,omitempty"`
}

func (w wireRequest) call() aitrace.Call {
	return aitrace.Call{Provider: aitrace.Google, Model: w.Model, Instructions: w.SystemInstruction, Input: w.Input, ServiceTier: w.ServiceTier, Stream: w.Stream}
}

// trace counts thought tokens as output, which Google leaves out of
// total_output_tokens but bills at the output rate.
func (r Response) trace() aitrace.Result {
	u := r.Usage
	return aitrace.Result{Text: r.Text, ServiceTier: r.ServiceTier, Usage: aitrace.Usage{
		Input: u.InputTokens, Cached: u.CachedTokens, Output: u.OutputTokens + u.ReasoningTokens, Reasoning: u.ReasoningTokens, WebSearchCalls: u.WebSearchCalls,
	}}
}

type wireGenerationConfig struct {
	ThinkingLevel   ThinkingLevel `json:"thinking_level,omitempty"`
	Temperature     *float64      `json:"temperature,omitempty"`
	MaxOutputTokens int           `json:"max_output_tokens,omitempty"`
}

type wireResponseFormat struct {
	Type     string          `json:"type"`
	MimeType string          `json:"mime_type"`
	Schema   json.RawMessage `json:"schema"`
}

type wireError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *wireError) Error() string {
	return fmt.Sprintf("Gemini error %s: %s", e.Code, e.Message)
}

type wireInteraction struct {
	ID          string      `json:"id"`
	Model       string      `json:"model"`
	ServiceTier string      `json:"service_tier"`
	Status      string      `json:"status"`
	Errors      []wireError `json:"errors"`
	Steps       []wireStep  `json:"steps"`
	Usage       struct {
		Usage
		GroundingToolCount []struct {
			Type  string `json:"type"`
			Count int    `json:"count"`
		} `json:"grounding_tool_count"`
	} `json:"usage"`
}

type wireStep struct {
	Type    string        `json:"type"`
	Content []wireContent `json:"content"`
}

type wireContent struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []struct {
		Type string `json:"type"`
		URL  string `json:"url"`
	} `json:"annotations"`
}

func (w wireInteraction) response(raw []byte) (Response, error) {
	if len(w.Errors) > 0 {
		return Response{}, &w.Errors[0]
	}
	if w.Status != "completed" {
		// Google gives no reason for an incomplete interaction.
		return Response{}, fmt.Errorf("Gemini interaction %s", w.Status)
	}
	r := Response{ID: w.ID, Model: w.Model, ServiceTier: w.ServiceTier, Usage: w.Usage.Usage, Raw: raw}
	for _, g := range w.Usage.GroundingToolCount {
		if g.Type == "google_search" {
			r.Usage.WebSearchCalls += g.Count
		}
	}
	seen := map[string]bool{}
	var text strings.Builder
	for _, step := range w.Steps {
		if step.Type != "model_output" {
			continue
		}
		for _, part := range step.Content {
			if part.Type != "text" {
				continue
			}
			text.WriteString(part.Text)
			for _, a := range part.Annotations {
				if a.Type == "url_citation" && !seen[a.URL] {
					seen[a.URL] = true
					r.Sources = append(r.Sources, a.URL)
				}
			}
		}
	}
	r.Text = text.String()
	if r.Text == "" {
		return Response{}, errors.New("Gemini response has no output text")
	}
	return r, nil
}
