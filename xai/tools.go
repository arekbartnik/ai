package xai

import "time"

// Tool is a server-side tool the model may call during Generate. The set is
// sealed: xAI runs these itself, so Generate never has to hand a call back
// to the caller.
type Tool interface {
	tool() wireTool
}

// WebSearch's AllowedDomains holds at most five entries.
type WebSearch struct {
	AllowedDomains     []string
	ImageUnderstanding bool
}

func (t WebSearch) tool() wireTool {
	w := wireTool{Type: "web_search", EnableImageUnderstanding: t.ImageUnderstanding}
	if len(t.AllowedDomains) > 0 {
		w.Filters = &wireFilters{AllowedDomains: t.AllowedDomains}
	}
	return w
}

// XSearch searches posts on X. AllowedHandles holds at most twenty entries.
// Zero From and To mean no bound.
type XSearch struct {
	AllowedHandles     []string
	From, To           time.Time
	ImageUnderstanding bool
	VideoUnderstanding bool
}

func (t XSearch) tool() wireTool {
	return wireTool{
		Type:                     "x_search",
		AllowedXHandles:          t.AllowedHandles,
		FromDate:                 day(t.From),
		ToDate:                   day(t.To),
		EnableImageUnderstanding: t.ImageUnderstanding,
		EnableVideoUnderstanding: t.VideoUnderstanding,
	}
}

type CodeInterpreter struct{}

func (CodeInterpreter) tool() wireTool { return wireTool{Type: "code_interpreter"} }

// MCP exposes a remote MCP server's tools to the model. Label and ServerURL
// are required.
type MCP struct {
	Label         string
	ServerURL     string
	Description   string
	AllowedTools  []string
	Authorization string
	Headers       map[string]string
}

func (t MCP) tool() wireTool {
	return wireTool{
		Type:              "mcp",
		ServerLabel:       t.Label,
		ServerURL:         t.ServerURL,
		ServerDescription: t.Description,
		AllowedTools:      t.AllowedTools,
		Authorization:     t.Authorization,
		Headers:           t.Headers,
	}
}

func day(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.Format("2006-01-02")
}

type wireTool struct {
	Type                     string            `json:"type"`
	Filters                  *wireFilters      `json:"filters,omitempty"`
	EnableImageUnderstanding bool              `json:"enable_image_understanding,omitempty"`
	AllowedXHandles          []string          `json:"allowed_x_handles,omitempty"`
	FromDate                 string            `json:"from_date,omitempty"`
	ToDate                   string            `json:"to_date,omitempty"`
	EnableVideoUnderstanding bool              `json:"enable_video_understanding,omitempty"`
	ServerLabel              string            `json:"server_label,omitempty"`
	ServerURL                string            `json:"server_url,omitempty"`
	ServerDescription        string            `json:"server_description,omitempty"`
	AllowedTools             []string          `json:"allowed_tools,omitempty"`
	Authorization            string            `json:"authorization,omitempty"`
	Headers                  map[string]string `json:"headers,omitempty"`
}

type wireFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}
