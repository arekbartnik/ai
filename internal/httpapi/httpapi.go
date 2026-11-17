// Package httpapi is the HTTP plumbing the provider clients share: one JSON
// POST per generation, read whole or line by line as a server-sent stream.
package httpapi

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"
)

const (
	timeout      = 5 * time.Minute
	maxReply     = 16 << 20
	maxErrorBody = 8 << 10
)

var client = &http.Client{Timeout: timeout}

// Endpoint is the URL a provider takes generations on.
type Endpoint struct {
	// Provider names the API in errors, as in "call OpenAI".
	Provider string
	URL      string
	// AuthHeader carries AuthValue, the API key in the provider's form.
	AuthHeader string
	AuthValue  string
	// APIError builds the provider's error for a reply other than 200.
	APIError func(status int, body string) error
}

// Post sends payload as JSON and returns the reply body.
func (e Endpoint) Post(ctx context.Context, payload any) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	resp, err := e.send(ctx, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxReply))
	if err != nil {
		return nil, fmt.Errorf("read %s response: %w", e.Provider, err)
	}
	return body, nil
}

// Stream sends payload as JSON and yields each line of the reply. The line
// is valid until the next one. An error ends the sequence.
func (e Endpoint) Stream(ctx context.Context, payload any) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		resp, err := e.send(ctx, payload)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64<<10), maxReply)
		for scanner.Scan() {
			if !yield(scanner.Bytes(), nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("read %s stream: %w", e.Provider, err))
		}
	}
}

func (e Endpoint) send(ctx context.Context, payload any) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set(e.AuthHeader, e.AuthValue)
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call %s: %w", e.Provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		reply, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		cut := ""
		if len(reply) > maxErrorBody {
			reply, cut = reply[:maxErrorBody], "…"
		}
		// A span with invalid UTF-8 fails the export of its whole batch.
		return nil, e.APIError(resp.StatusCode, strings.ToValidUTF8(string(reply), "")+cut)
	}
	return resp, nil
}
