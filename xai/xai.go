package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/arekbartnik/ai/aitrace"
	"github.com/arekbartnik/ai/internal/httpapi"
)

const (
	DefaultBaseURL = "https://api.x.ai"
)

// Config is set by the caller. An empty BaseURL is DefaultBaseURL, and an
// empty Model leaves the model to each Request.
type Config struct {
	APIKey  string
	BaseURL string
	Model   string
	Store   bool
}

// APIError is a non-200 reply from xAI.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("xAI returned %d: %s", e.Status, e.Body)
}

type Client struct {
	api   httpapi.Endpoint
	model string
	store bool
}

func New(c Config) *Client {
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	c.BaseURL = strings.TrimSuffix(c.BaseURL, "/")
	return &Client{model: c.Model, store: c.Store, api: httpapi.Endpoint{
		Provider:   "xAI",
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
type Request struct {
	Model              string
	Instructions       string
	Input              string
	PreviousResponseID string
	Tools              []Tool
	Schema             *Schema
	ReasoningEffort    string
	Temperature        *float64
	MaxOutputTokens    int
}

type Schema struct {
	Name   string
	Schema json.RawMessage
}

type Response struct {
	ID    string
	Model string
	Text  string
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
		return fmt.Errorf("decode xAI structured output: %w", err)
	}
	return nil
}

var errNoModel = errors.New("xAI request has no model: set Request.Model or Config.Model")

// Model is the model a Request without its own Model uses. It is empty
// when Config named none.
func (c *Client) Model() string { return c.model }

func (c *Client) Generate(ctx context.Context, req Request) (r Response, err error) {
	w := c.wire(req)
	if w.Model == "" {
		return Response{}, errNoModel
	}
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
			Error    *struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(data, &ev); err != nil {
			return Response{}, fmt.Errorf("decode xAI stream event: %w", err)
		}
		switch ev.Type {
		case "response.output_text.delta":
			if err := onDelta(ev.Delta); err != nil {
				return Response{}, err
			}
		case "response.completed", "response.incomplete", "response.failed":
			return parse(ev.Response)
		case "error":
			if ev.Error != nil {
				return Response{}, fmt.Errorf("xAI error %s: %s", ev.Error.Code, ev.Error.Message)
			}
			return Response{}, errors.New("xAI stream error")
		}
	}
	return Response{}, errors.New("xAI stream ended without a completed response")
}

func (c *Client) wire(req Request) wireRequest {
	w := wireRequest{
		Model:              req.Model,
		Instructions:       req.Instructions,
		Input:              req.Input,
		PreviousResponseID: req.PreviousResponseID,
		Temperature:        req.Temperature,
		MaxOutputTokens:    req.MaxOutputTokens,
		// By default xAI writes inline citation markup into output text.
		// Sources come from annotations.
		Include: []string{"no_inline_citations"},
		Store:   c.store,
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
	Include            []string       `json:"include"`
	Stream             bool           `json:"stream,omitempty"`
}

func (w wireRequest) call() aitrace.Call {
	return aitrace.Call{Provider: aitrace.XAI, Model: w.Model, Instructions: w.Instructions, Input: w.Input, Stream: w.Stream}
}

func (r Response) trace() aitrace.Result {
	u := r.Usage
	return aitrace.Result{Text: r.Text, Usage: aitrace.Usage{
		Input: u.InputTokens, Cached: u.CachedTokens, Output: u.OutputTokens, Reasoning: u.ReasoningTokens, WebSearchCalls: u.WebSearchCalls,
	}}
}

type wireReasoning struct {
	Effort string `json:"effort"`
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
	ID     string `json:"id"`
	Model  string `json:"model"`
	Status string `json:"status"`
	Error  *struct {
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
		ServerSideToolUsageDetails struct {
			WebSearchCalls int `json:"web_search_calls"`
		} `json:"server_side_tool_usage_details"`
	} `json:"usage"`
}

func parse(body []byte) (Response, error) {
	var w wireResponse
	if err := json.Unmarshal(body, &w); err != nil {
		return Response{}, fmt.Errorf("decode xAI response: %w", err)
	}
	if w.Error != nil {
		return Response{}, fmt.Errorf("xAI error %s: %s", w.Error.Code, w.Error.Message)
	}
	if w.Status != "completed" {
		reason := ""
		if w.IncompleteDetails != nil {
			reason = ": " + w.IncompleteDetails.Reason
		}
		return Response{}, fmt.Errorf("xAI response %s%s", w.Status, reason)
	}
	r := Response{ID: w.ID, Model: w.Model, Usage: w.Usage.Usage, Raw: body}
	r.Usage.CachedTokens = w.Usage.InputTokensDetails.CachedTokens
	r.Usage.ReasoningTokens = w.Usage.OutputTokensDetails.ReasoningTokens
	r.Usage.WebSearchCalls = w.Usage.ServerSideToolUsageDetails.WebSearchCalls
	// With server-side tools xAI can write a message, search, and write
	// another. Each message is a whole answer (a whole JSON value under a
	// Schema), so Text is the last non-empty one, never their concatenation.
	seen := map[string]bool{}
	for _, item := range w.Output {
		if item.Type != "message" {
			continue
		}
		var text strings.Builder
		for _, part := range item.Content {
			if part.Type == "refusal" {
				return Response{}, fmt.Errorf("xAI refused: %s", part.Refusal)
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
		if text.Len() > 0 {
			r.Text = text.String()
		}
	}
	if r.Text == "" {
		return Response{}, errors.New("xAI response has no output text")
	}
	return r, nil
}
