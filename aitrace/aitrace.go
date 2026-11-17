// Package aitrace records AI requests as OpenTelemetry spans that Langfuse
// reads as generations, named like the Vercel AI SDK's ai.* spans.
package aitrace

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	"go.opentelemetry.io/otel/trace"
)

// Config sends traces to Langfuse. Tracing stays off without Host and both
// keys. The caller sets Environment and ServiceName. An empty one is not
// sent, which leaves it to OTEL_RESOURCE_ATTRIBUTES and OTEL_SERVICE_NAME.
type Config struct {
	Host        string
	PublicKey   string
	SecretKey   string
	Environment string
	ServiceName string
	Cost        Cost
}

// Cost prices one call in USD: its total, and the web search part of that
// total. With a nil Cost, or ok false, a call reports no cost and Langfuse
// falls back to its own prices.
type Cost func(model, serviceTier string, u Usage) (total, webSearch float64, ok bool)

var cost Cost

// Install sets the global tracer provider. The returned func flushes the
// spans not yet exported.
func Install(c Config) (func(context.Context) error, error) {
	c.Host = strings.TrimSuffix(c.Host, "/")
	if c.Host == "" || c.PublicKey == "" || c.SecretKey == "" {
		slog.Info("AI tracing off", "reason", "Host, PublicKey or SecretKey is empty")
		return func(context.Context) error { return nil }, nil
	}
	exporter, err := otlptracehttp.New(context.Background(),
		otlptracehttp.WithEndpointURL(c.Host+"/api/public/otel/v1/traces"),
		otlptracehttp.WithHeaders(map[string]string{
			"Authorization":                "Basic " + base64.StdEncoding.EncodeToString([]byte(c.PublicKey+":"+c.SecretKey)),
			"x-langfuse-ingestion-version": "4",
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("create trace exporter: %w", err)
	}
	attrs := []attribute.KeyValue{semconv.ServiceVersion(version())}
	if c.Environment != "" {
		attrs = append(attrs, semconv.DeploymentEnvironmentNameKey.String(c.Environment))
	}
	if c.ServiceName != "" {
		attrs = append(attrs, semconv.ServiceName(c.ServiceName))
	}
	provider := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter), sdktrace.WithResource(resource.NewWithAttributes(semconv.SchemaURL, attrs...)))
	cost = c.Cost
	otel.SetTracerProvider(provider)
	slog.Info("AI tracing on", "host", c.Host, "environment", c.Environment)
	return provider.Shutdown, nil
}

func version() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" {
				return s.Value
			}
		}
	}
	return "dev"
}

var tracer = otel.Tracer("github.com/arekbartnik/ai/aitrace")

type rootKey struct{}

type root struct {
	name  string
	attrs []attribute.KeyValue
	start time.Time
	mu    sync.Mutex
	span  trace.Span
}

func (r *root) started(ctx context.Context) trace.Span {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.span == nil {
		_, r.span = tracer.Start(ctx, r.name, trace.WithTimestamp(r.start), trace.WithAttributes(r.attrs...))
	}
	return r.span
}

// Root makes the AI calls under the returned context one trace named name,
// with attrs on its root span. The trace starts with the first call, so a
// root without calls exports nothing. end closes the trace with err.
func Root(ctx context.Context, name string, attrs ...attribute.KeyValue) (_ context.Context, end func(err error)) {
	r := &root{name: name, attrs: attrs, start: time.Now()}
	return context.WithValue(ctx, rootKey{}, r), func(err error) {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.span == nil {
			return
		}
		fail(r.span, err)
		r.span.End()
	}
}

func Session(id string) attribute.KeyValue { return attribute.String("langfuse.session.id", id) }

func User(id string) attribute.KeyValue { return attribute.String("langfuse.user.id", id) }

func Metadata(key, value string) attribute.KeyValue {
	return attribute.String("langfuse.trace.metadata."+key, value)
}

type Provider string

const (
	OpenAI Provider = "openai"
	XAI    Provider = "xai"
	Google Provider = "google"
)

type Call struct {
	Provider     Provider
	Model        string
	Instructions string
	Input        string
	ServiceTier  string
	Stream       bool
}

// Usage counts tokens the way OpenAI does: Cached is part of Input and
// Reasoning part of Output.
type Usage struct {
	Input, Cached, Output, Reasoning, WebSearchCalls int
}

type Result struct {
	Text        string
	ServiceTier string
	Usage       Usage
}

type Generation struct {
	span trace.Span
	call Call
}

// Start opens a generation span, under the context's Root when it has one.
func Start(ctx context.Context, c Call) (context.Context, Generation) {
	if r, ok := ctx.Value(rootKey{}).(*root); ok {
		ctx = trace.ContextWithSpan(ctx, r.started(ctx))
	}
	name := "ai.generateText.doGenerate"
	if c.Stream {
		name = "ai.streamText.doStream"
	}
	ctx, span := tracer.Start(ctx, name)
	if span.IsRecording() {
		type message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}
		var messages []message
		if c.Instructions != "" {
			messages = append(messages, message{"system", c.Instructions})
		}
		input, _ := json.Marshal(append(messages, message{"user", c.Input}))
		span.SetAttributes(
			attribute.String("langfuse.observation.type", "generation"),
			attribute.String("langfuse.observation.model.name", c.Model),
			attribute.String("langfuse.observation.input", string(input)),
			attribute.String("ai.provider", string(c.Provider)),
			attribute.String("ai.model.id", c.Model),
		)
	}
	return ctx, Generation{span: span, call: c}
}

func (g Generation) End(r Result, err error) {
	defer g.span.End()
	if !g.span.IsRecording() {
		return
	}
	if err != nil {
		fail(g.span, err)
		return
	}
	u := r.Usage
	usage := map[string]int{"input": u.Input - u.Cached, "output": u.Output - u.Reasoning, "total": u.Input + u.Output}
	if u.Cached > 0 {
		usage["input_cached_tokens"] = u.Cached
	}
	if u.Reasoning > 0 {
		usage["output_reasoning_tokens"] = u.Reasoning
	}
	if u.WebSearchCalls > 0 {
		usage["web_search_requests"] = u.WebSearchCalls
	}
	usageJSON, _ := json.Marshal(usage)
	g.span.SetAttributes(
		attribute.String("langfuse.observation.output", r.Text),
		attribute.String("langfuse.observation.usage_details", string(usageJSON)),
	)
	tier := cmp.Or(r.ServiceTier, g.call.ServiceTier)
	if tier != "" {
		g.span.SetAttributes(attribute.String("ai.service_tier", tier))
	}
	if cost == nil {
		return
	}
	if total, search, ok := cost(g.call.Model, tier, u); ok {
		costs := map[string]float64{"total": total}
		if search > 0 {
			costs["web_search_requests"] = search
		}
		costJSON, _ := json.Marshal(costs)
		g.span.SetAttributes(attribute.String("langfuse.observation.cost_details", string(costJSON)))
	}
}

func fail(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
}
