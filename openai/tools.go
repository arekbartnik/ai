package openai

// Tool is a server-side tool the model may call during Generate. The set is
// sealed: OpenAI runs these itself, so Generate never has to hand a call back
// to the caller.
type Tool interface {
	tool() wireTool
}

// WebSearch's AllowedDomains also admit their subdomains. Country is a
// two-letter ISO code that localizes results.
type WebSearch struct {
	AllowedDomains []string
	Country        string
}

func (t WebSearch) tool() wireTool {
	w := wireTool{Type: "web_search"}
	if len(t.AllowedDomains) > 0 {
		w.Filters = &wireFilters{AllowedDomains: t.AllowedDomains}
	}
	if t.Country != "" {
		w.UserLocation = &wireLocation{Type: "approximate", Country: t.Country}
	}
	return w
}

// CodeInterpreter runs Python in a container OpenAI creates for the request.
type CodeInterpreter struct{}

func (CodeInterpreter) tool() wireTool {
	return wireTool{Type: "code_interpreter", Container: &wireContainer{Type: "auto"}}
}

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
		// OpenAI otherwise stops at an approval request for each MCP call,
		// which would hand the call back to the caller.
		RequireApproval: "never",
	}
}

type wireTool struct {
	Type              string            `json:"type"`
	Filters           *wireFilters      `json:"filters,omitempty"`
	UserLocation      *wireLocation     `json:"user_location,omitempty"`
	Container         *wireContainer    `json:"container,omitempty"`
	ServerLabel       string            `json:"server_label,omitempty"`
	ServerURL         string            `json:"server_url,omitempty"`
	ServerDescription string            `json:"server_description,omitempty"`
	AllowedTools      []string          `json:"allowed_tools,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	RequireApproval   string            `json:"require_approval,omitempty"`
}

type wireFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

type wireLocation struct {
	Type    string `json:"type"`
	Country string `json:"country,omitempty"`
}

type wireContainer struct {
	Type string `json:"type"`
}
