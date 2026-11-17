# ai

Small Go clients for OpenAI, Gemini and xAI, with Langfuse tracing built in.

Each client sends one prompt and returns the text, the cited sources and the token usage. The provider runs the tools itself, so there is no tool loop to write. Every call becomes a Langfuse generation once tracing is installed.

```sh
go get github.com/arekbartnik/ai
```

## Packages

| Package | API | Tools |
| --- | --- | --- |
| `openai` | Responses, `/v1/responses` | `WebSearch`, `CodeInterpreter`, `MCP` |
| `gemini` | Interactions, `/v1beta/interactions` | `GoogleSearch`, `URLContext`, `CodeExecution` |
| `xai` | Responses, `/v1/responses` | `WebSearch`, `XSearch`, `CodeInterpreter`, `MCP` |
| `aitrace` | OpenTelemetry spans over OTLP to Langfuse | |

## Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/arekbartnik/ai/aitrace"
	"github.com/arekbartnik/ai/openai"
)

func main() {
	tracing := aitrace.ConfigFromEnv()
	tracing.ServiceName = "my-service"
	tracing.Environment = "production"
	flush, err := aitrace.Install(tracing)
	if err != nil {
		log.Fatal(err)
	}
	defer flush(context.Background())

	config, err := openai.ConfigFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	client := openai.New(config)

	ctx, end := aitrace.Root(context.Background(), "ai.example", aitrace.User("someone@example.com"))
	resp, err := client.Generate(ctx, openai.Request{
		Model:        "gpt-6-luna",
		Instructions: "Answer in one sentence.",
		Input:        "What changed in Go this year?",
		Tools:        []openai.Tool{openai.WebSearch{Country: "US"}},
	})
	end(err)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text, resp.Sources)
}
```

The library names no model and reads none from the environment. Set `Model` on the request, or set `Config.Model` once for every request that leaves it empty. A request with neither fails before it is sent.

`Stream` takes the same request and passes each text delta to a callback. A `Schema` on the request returns JSON that `Response.Decode` unmarshals.

## Tracing

`aitrace.Install` sets the global OpenTelemetry tracer provider. Without it the clients still work and record nothing.

`ConfigFromEnv` reads only the Langfuse host and keys. The application sets `ServiceName` and `Environment` itself. An empty one is not sent.

`aitrace.Root` groups the calls made under its context into one trace. The trace starts with the first call, so a root without calls exports nothing. `Session`, `User` and `Metadata` build the attributes Langfuse reads.

The library holds no prices. Set `Config.Cost` to price a call from its model, service tier and usage. Without it, or when it reports a model as unknown, the generation carries no cost and Langfuse falls back to its own prices.

```go
tracing.Cost = func(model, serviceTier string, u aitrace.Usage) (total, webSearch float64, ok bool) {
	if model != "my-model" {
		return 0, 0, false
	}
	webSearch = float64(u.WebSearchCalls) * 0.01
	return float64(u.Input)*1e-6 + float64(u.Output)*4e-6 + webSearch, webSearch, true
}
```

`openai.Record` keeps every request sent under its context, with the wire payload and the outcome, for callers that store their own audit trail.

## Environment

| Variable | Used by | Required |
| --- | --- | --- |
| `OPENAI_API_KEY` | `openai.ConfigFromEnv` | yes |
| `OPENAI_BASE_URL` | `openai.ConfigFromEnv` | no |
| `GOOGLE_GENERATIVE_AI_API_KEY` | `gemini.ConfigFromEnv` | yes |
| `GOOGLE_GENERATIVE_AI_BASE_URL` | `gemini.ConfigFromEnv` | no |
| `XAI_API_KEY` | `xai.ConfigFromEnv` | yes |
| `XAI_BASE_URL` | `xai.ConfigFromEnv` | no |
| `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `aitrace.ConfigFromEnv` | tracing is off without all three |

## License

MIT
