package xai

import (
	"slices"
	"testing"
)

func TestParseKeepsLastMessage(t *testing.T) {
	body := []byte(`{
		"id": "resp_1", "model": "grok", "status": "completed",
		"output": [
			{"type": "message", "role": "assistant", "content": [
				{"type": "output_text", "text": "{\"notes\":\"first\"}", "annotations": [{"type": "url_citation", "url": "https://a.example"}]}
			]},
			{"type": "web_search_call"},
			{"type": "message", "role": "assistant", "content": [
				{"type": "output_text", "text": "{\"notes\":", "annotations": [{"type": "url_citation", "url": "https://b.example"}]},
				{"type": "output_text", "text": "\"second\"}"}
			]},
			{"type": "message", "role": "assistant", "content": []}
		]
	}`)
	r, err := parse(body)
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Notes string `json:"notes"`
	}
	if err := r.Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Notes != "second" {
		t.Errorf("notes = %q, want second", out.Notes)
	}
	if want := []string{"https://a.example", "https://b.example"}; !slices.Equal(r.Sources, want) {
		t.Errorf("sources = %v, want %v", r.Sources, want)
	}
}
