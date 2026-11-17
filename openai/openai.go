package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/arekbartnik/ai/aitrace"
	"github.com/arekbartnik/ai/internal/httpapi"
)

const (
	DefaultBaseURL = "https://api.openai.com"

	Flex           = "flex"
	flexRetryDelay = 2 * time.Second
)

type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	Store       bool
	ServiceTier string
}

func ConfigFromEnv() (Config, error) {
	c := Config{
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		BaseURL: strings.TrimSuffix(os.Getenv("OPENAI_BASE_URL"), "/"),
		Model:   os.Getenv("OPENAI_MODEL"),
	}
	if c.APIKey == "" {
		return c, errors.New("missing OpenAI configuration: OPENAI_API_KEY")
	}
	return c, nil
}

// APIError is a non-200 reply from OpenAI.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("OpenAI returned %d: %s", e.Status, e.Body)
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
	return &Client{model: c.Model, store: c.Store, serviceTier: c.ServiceTier, api: httpapi.Endpoint{
		Provider:   "OpenAI",
		URL:        c.BaseURL + "/v1/responses",
		AuthHeader: "Authorization",
		AuthValue:  "Bearer " + c.APIKey,
		APIError:   func(status int, body string) error { return &APIError{Status: status, Body: body} },
	}}
}

// Request is one generation. Input is the user prompt, Instructions the
// system prompt. A nil Schema returns free text.
// PreviousResponseID continues an earlier stored response's conversation
// and requires Config.Store.
// An empty ReasoningEffort leaves the model's default. Reasoning models
// reject Temperature unless the effort is none.
type Request struct {
	Model              string
	Instructions       string
	Input              string
	PreviousResponseID string
	Tools              []Tool
	Schema             *Schema
	ReasoningEffort    ReasoningEffort
	Temperature        *float64
	MaxOutputTokens    int
}

type ReasoningEffort string

var ReasoningEfforts = []ReasoningEffort{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

type Schema struct {
	Name   string
	Schema json.RawMessage
}

type Response struct {
	ID          string
	Model       string
	ServiceTier string
	Text        string
	// Sources lists the URLs the model cited, in first-cited order.
	Sources []string
	Usage   Usage
	Raw     json.RawMessage
}

type Usage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	TotalTokens     int `json:"total_tokens"`
	CachedTokens    int
	ReasoningTokens int
	WebSearchCalls  int
}

func (r Response) Decode(v any) error {
	if err := json.Unmarshal([]byte(r.Text), v); err != nil {
		return fmt.Errorf("decode OpenAI structured output: %w", err)
	}
	return nil
}

var errNoModel = errors.New("OpenAI request has no model: set Request.Model or Config.Model")

// Model is the model a Request without its own Model uses. It is empty
// when Config named none.
func (c *Client) Model() string { return c.model }

// Generate retries a flex request on the default tier when flex has no
// capacity.
func (c *Client) Generate(ctx context.Context, req Request) (Response, error) {
	w := c.wire(req)
	if w.Model == "" {
		return Response{}, errNoModel
	}
	r, err := c.generate(ctx, w)
	var apiErr *APIError
	if w.ServiceTier != Flex || !errors.As(err, &apiErr) || apiErr.Status != http.StatusTooManyRequests {
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
	started := time.Now()
	defer func() { record(ctx, w, started, r, err) }()
	ctx, gen := aitrace.Start(ctx, w.call())
	defer func() { gen.End(r.trace(), err) }()
	body, err := c.api.Post(ctx, w)
	if err != nil {
		return Response{}, err
	}
	return parse(body)
}

// Stream is Generate with each text delta passed to onDelta as it arrives.
// The returned Response is the same complete result Generate would give.
// A non-nil error from onDelta stops the stream and is returned.
func (c *Client) Stream(ctx context.Context, req Request, onDelta func(string) error) (r Response, err error) {
	w := c.wire(req)
	if w.Model == "" {
		return Response{}, errNoModel
	}
	w.Stream = true
	started := time.Now()
	defer func() { record(ctx, w, started, r, err) }()
	ctx, gen := aitrace.Start(ctx, w.call())
	defer func() { gen.End(r.trace(), err) }()
	return readStream(c.api.Stream(ctx, w), onDelta)
}

func readStream(lines iter.Seq2[[]byte, error], onDelta func(string) error) (Response, error) {
	for line, err := range lines {
		if err != nil {
			return Response{}, err
		}
		data, ok := bytes.CutPrefix(line, []byte("data: "))
		if !ok {
			continue
		}
		var ev struct {
			Type     string          `json:"type"`
			Delta    string          `json:"delta"`
			Response json.RawMessage `json:"response"`
			Code     string          `json:"code"`
			Message  string          `json:"message"`
		}
		if err := json.Unmarshal(data, &ev); err != nil {
			return Response{}, fmt.Errorf("decode OpenAI stream event: %w", err)
		}
		switch ev.Type {
		case "response.output_text.delta":
			if err := onDelta(ev.Delta); err != nil {
				return Response{}, err
			}
		case "response.completed", "response.incomplete", "response.failed":
			return parse(ev.Response)
		case "error":
			return Response{}, fmt.Errorf("OpenAI error %s: %s", ev.Code, ev.Message)
		}
	}
	return Response{}, errors.New("OpenAI stream ended without a completed response")
}

func (c *Client) wire(req Request) wireRequest {
	w := wireRequest{
		Model:              req.Model,
		Instructions:       req.Instructions,
		Input:              req.Input,
		PreviousResponseID: req.PreviousResponseID,
		Temperature:        req.Temperature,
		MaxOutputTokens:    req.MaxOutputTokens,
		Store:              c.store,
		ServiceTier:        c.serviceTier,
	}
	if w.Model == "" {
		w.Model = c.model
	}
	if req.ReasoningEffort != "" {
		w.Reasoning = &wireReasoning{Effort: req.ReasoningEffort}
	}
	for _, t := range req.Tools {
		w.Tools = append(w.Tools, t.tool())
	}
	if s := req.Schema; s != nil {
		w.Text = &wireText{Format: wireFormat{Type: "json_schema", Name: s.Name, Schema: s.Schema, Strict: true}}
	}
	return w
}

type wireRequest struct {
	Model              string         `json:"model"`
	Instructions       string         `json:"instructions,omitempty"`
	Input              string         `json:"input"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
	Store              bool           `json:"store"`
	Tools              []wireTool     `json:"tools,omitempty"`
	Text               *wireText      `json:"text,omitempty"`
	Reasoning          *wireReasoning `json:"reasoning,omitempty"`
	Temperature        *float64       `json:"temperature,omitempty"`
	MaxOutputTokens    int            `json:"max_output_tokens,omitempty"`
	ServiceTier        string         `json:"service_tier,omitempty"`
	Stream             bool           `json:"stream,omitempty"`
}

func (w wireRequest) call() aitrace.Call {
	return aitrace.Call{Provider: aitrace.OpenAI, Model: w.Model, Instructions: w.Instructions, Input: w.Input, ServiceTier: w.ServiceTier, Stream: w.Stream}
}

func (r Response) trace() aitrace.Result {
	u := r.Usage
	return aitrace.Result{Text: r.Text, ServiceTier: r.ServiceTier, Usage: aitrace.Usage{
		Input: u.InputTokens, Cached: u.CachedTokens, Output: u.OutputTokens, Reasoning: u.ReasoningTokens, WebSearchCalls: u.WebSearchCalls,
	}}
}

type wireReasoning struct {
	Effort ReasoningEffort `json:"effort"`
}

type wireText struct {
	Format wireFormat `json:"format"`
}

type wireFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type wireResponse struct {
	ID          string `json:"id"`
	Model       string `json:"model"`
	ServiceTier string `json:"service_tier"`
	Status      string `json:"status"`
	Error       *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Output []struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Refusal     string `json:"refusal"`
			Annotations []struct {
				Type string `json:"type"`
				URL  string `json:"url"`
			} `json:"annotations"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		Usage
		InputTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}

func parse(body []byte) (Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return Response{}, fmt.Errorf("decode OpenAI response: %w", err)
	}
	if w.Error != nil {
		return Response{}, fmt.Errorf("OpenAI error %s: %s", w.Error.Code, w.Error.Message)
	}
	if w.Status != "completed" {
		reason := ""
		if w.IncompleteDetails != nil {
			reason = ": " + w.IncompleteDetails.Reason
		}
		return Response{}, fmt.Errorf("OpenAI response %s%s", w.Status, reason)
	}
	r := Response{ID: w.ID, Model: w.Model, ServiceTier: w.ServiceTier, Usage: w.Usage.Usage, Raw: body}
	r.Usage.CachedTokens = w.Usage.InputTokensDetails.CachedTokens
	r.Usage.ReasoningTokens = w.Usage.OutputTokensDetails.ReasoningTokens
	seen := map[string]bool{}
	var text strings.Builder
	for _, item := range w.Output {
		// OpenAI reports no web search count in usage, only one output item per call.
		if item.Type == "web_search_call" {
			r.Usage.WebSearchCalls++
		}
		if item.Type != "message" {
			continue
		}
		for _, part := range item.Content {
			if part.Type == "refusal" {
				return Response{}, fmt.Errorf("OpenAI refused: %s", part.Refusal)
			}
			if part.Type != "output_text" {
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
		return Response{}, errors.New("OpenAI response has no output text")
	}
	return r, nil
}
