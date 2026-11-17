package gemini

// Tool is a server-side tool the model may call during Generate. The set is
// sealed: Google runs these itself, so Generate never has to hand a call back
// to the caller.
type Tool interface {
	tool() wireTool
}

type GoogleSearch struct{}

func (GoogleSearch) tool() wireTool { return wireTool{Type: "google_search"} }

// URLContext lets the model fetch the URLs named in the prompt.
type URLContext struct{}

func (URLContext) tool() wireTool { return wireTool{Type: "url_context"} }

// CodeExecution runs Python in a sandbox Google creates for the request.
type CodeExecution struct{}

func (CodeExecution) tool() wireTool { return wireTool{Type: "code_execution"} }

type wireTool struct {
	Type string `json:"type"`
}
