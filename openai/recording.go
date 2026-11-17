package openai

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// Call is one request to OpenAI as it was sent, and how it ended. A retry or a
// fallback to another tier is a Call of its own.
type Call struct {
	Model      string          `json:"model"`
	StartedAt  time.Time       `json:"startedAt"`
	DurationMs int64           `json:"durationMs"`
	Request    json.RawMessage `json:"request"`
	Output     string          `json:"output,omitempty"`
	Error      string          `json:"error,omitempty"`
}

type Recording struct {
	mu    sync.Mutex
	calls []Call
}

type recordingKey struct{}

func Record(ctx context.Context) (context.Context, *Recording) {
	r := &Recording{}
	return context.WithValue(ctx, recordingKey{}, r), r
}

// Calls returns the calls in the order they ended.
func (r *Recording) Calls() []Call {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Call(nil), r.calls...)
}

func record(ctx context.Context, w wireRequest, started time.Time, r Response, err error) {
	recording, ok := ctx.Value(recordingKey{}).(*Recording)
	if !ok {
		return
	}
	request, marshalErr := json.Marshal(w)
	if marshalErr != nil {
		request = nil
	}
	call := Call{Model: w.Model, StartedAt: started, DurationMs: time.Since(started).Milliseconds(), Request: request, Output: r.Text}
	if err != nil {
		call.Error = err.Error()
	}
	recording.mu.Lock()
	defer recording.mu.Unlock()
	recording.calls = append(recording.calls, call)
}
