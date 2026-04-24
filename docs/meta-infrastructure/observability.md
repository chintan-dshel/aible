---
sidebar_position: 3
title: Observability
description: Traces, spans, prompt logging, and latency tracking for AI systems — what to instrument and why.
---

# Observability

## What it is

Observability for AI systems is the practice of capturing enough data about every request — the prompt sent, the model called, the response received, latency at each step, token counts, and errors — that you can understand what your system did, why it did it, and where it went wrong.

Traditional software observability (metrics, logs, traces) applies, but AI systems have unique instrumentation needs: the "input" is often multi-thousand-token prompts, the "output" is unstructured text, decisions are made by a probabilistic model rather than deterministic code, and failure modes (hallucination, prompt injection, slow generation) are invisible without AI-specific tooling.

## The problem it solves

Without observability, you cannot answer the most basic operational questions:

- Which prompts are slowest? Which are most expensive?
- When a user reports "the bot gave wrong information," what exact prompt and context was used?
- Did a model upgrade actually improve response quality, or just change the style?
- Are there prompt patterns that consistently produce bad outputs?
- Is latency degrading over time as conversations grow longer?

In a debugging session for a traditional service, you have logs. In an AI system without observability, you have a model that did something you can't reproduce.

## How it works under the hood

### The four signals

**Traces** — a trace captures the full lifecycle of a single request. For a RAG pipeline, this means: the original user query, the retrieved chunks, the assembled prompt, the model call and response, and the total end-to-end latency. A trace is a tree of spans.

**Spans** — a span captures one unit of work within a trace: embedding the query (with duration and vector dimensions), the ANN retrieval (with the number of results and scores), the rerank step, the LLM call (with model, token counts, latency). Each span has a start time, end time, and metadata.

**Metrics** — aggregated statistics over many requests: P50/P95/P99 latency, token counts (input, output, total), error rates, cost per request, cache hit rate.

**Logs** — structured log entries for events: errors, warnings, retries, fallbacks, guardrail triggers.

### What to capture for every LLM call

```python
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMCallTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Request
    model: str = ""
    system_prompt: str = ""
    messages: list = field(default_factory=list)
    max_tokens: int = 0
    temperature: float = 1.0

    # Response
    output: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Derived
    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def cost_usd(self) -> float:
        # Approximate: update with current pricing
        input_cost = self.input_tokens * 3e-6   # $3/MTok for Sonnet
        output_cost = self.output_tokens * 15e-6  # $15/MTok
        return input_cost + output_cost
```

### Instrumenting an LLM call

```python
import anthropic
import json

client = anthropic.Anthropic()

def traced_llm_call(
    messages: list,
    system: str = "",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    session_id: str = None,
) -> tuple[str, LLMCallTrace]:
    trace = LLMCallTrace(
        session_id=session_id,
        model=model,
        system_prompt=system[:200],  # truncate for storage
        messages=messages,
        max_tokens=max_tokens,
    )

    trace.start_time = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        trace.output = response.content[0].text
        trace.stop_reason = response.stop_reason
        trace.input_tokens = response.usage.input_tokens
        trace.output_tokens = response.usage.output_tokens
    finally:
        trace.end_time = time.time()

    # Emit to your observability backend
    emit_trace(trace)

    return trace.output, trace

def scrub_pii_simple(text: str) -> str:
    import re
    patterns = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text

def emit_trace(trace: LLMCallTrace):
    # Scrub PII from any content fields before sending to observability backend.
    # Use presidio or scrubadub in production for more thorough detection.
    safe_output = scrub_pii_simple(trace.output) if trace.output else ""

    # In production: send to Langfuse, LangSmith, Datadog, etc.
    print(json.dumps({
        "trace_id": trace.trace_id,
        "model": trace.model,
        "latency_ms": round(trace.latency_ms, 1),
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "cost_usd": round(trace.cost_usd, 6),
        "output_scrubbed": safe_output[:500],  # truncate; never log full output at scale
    }))
```

### Prompt logging and PII

Prompt logs are powerful but carry risk: they contain everything the user said, including personal information. Before logging prompts, decide:

- **What to redact**: PII (names, emails, SSNs, credit card numbers) should be scrubbed before storage. Use a PII detection library (`presidio`, `scrubadub`) or a regex pass over known patterns.
- **What to truncate**: full prompt text at 200K tokens is expensive to store. Log a truncated version for debugging and a hash of the full prompt for deduplication.
- **Retention policy**: how long do you keep logs? Most compliance frameworks (GDPR, HIPAA, SOC 2) have specific requirements. Default to the minimum retention you need for debugging: 30 days is usually sufficient.

### Distributed tracing for multi-step pipelines

For a RAG pipeline or multi-agent system, a single user request triggers multiple operations. A distributed trace ties them together:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_exporter(ConsoleSpanExporter())
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag_pipeline")

def traced_rag_query(query: str, context_docs: list[str]) -> str:
    with tracer.start_as_current_span("rag_query") as root_span:
        root_span.set_attribute("query", query[:200])

        with tracer.start_as_current_span("embed_query"):
            query_embedding = embed(query)  # your embed function

        with tracer.start_as_current_span("retrieve") as span:
            chunks = retrieve(query_embedding, k=10)
            span.set_attribute("chunks_retrieved", len(chunks))

        with tracer.start_as_current_span("rerank") as span:
            top_chunks = rerank(query, chunks)[:5]
            span.set_attribute("chunks_after_rerank", len(top_chunks))

        with tracer.start_as_current_span("generate") as span:
            answer = generate(query, top_chunks)
            span.set_attribute("output_length", len(answer))

        return answer
```

## Concrete example

A minimal observability setup using Langfuse:

```python
from langfuse import Langfuse
import anthropic

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"
)
client = anthropic.Anthropic()

def rag_with_tracing(user_query: str, session_id: str) -> str:
    # Create a trace for this user request
    trace = langfuse.trace(
        name="rag_query",
        session_id=session_id,
        input=user_query,
    )

    # Retrieval span
    retrieval_span = trace.span(name="retrieval")
    chunks = retrieve_chunks(user_query)  # your retrieval function
    retrieval_span.end(output={"num_chunks": len(chunks)})

    # Generation span
    gen_span = trace.span(name="generation")
    context = "\n\n".join(chunks[:4])
    prompt = f"Answer from context:\n{context}\n\nQuestion: {user_query}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.content[0].text

    gen_span.end(
        output=answer,
        usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
    )

    trace.update(output=answer)
    return answer
```

In the Langfuse dashboard you can now see: which queries were slowest, which returned low-quality answers (if you add a score), and the exact prompt sent to the model for any specific request.

## When to use it / when not to

#### Instrument from day one

Observability is not something you add after problems appear — it's how you detect problems in the first place. The cost of instrumenting from the start is low; the cost of trying to debug a production incident without traces is very high.

Minimum viable observability: log the model, the input token count, the output token count, latency, and a session ID on every LLM call. This alone enables cost tracking and latency debugging.

#### What to add as you scale

| Stage | Add this |
|---|---|
| Prototype | Structured logging (JSON), session IDs, token counts |
| Beta | LLM observability platform (Langfuse, LangSmith), prompt versioning |
| Production | Distributed tracing, cost attribution per feature/user, alerting |
| Scale | Sampling (log 10% of prompts, 100% of errors), PII scrubbing, log retention policy |

:::tip[My take]

The first time you need to debug a user complaint ("the bot told me the wrong price") and you don't have prompt logs, you'll wish you'd added observability from the start. The exact prompt, the exact context, and the exact model response are irreproducible after the fact unless you logged them. Ten minutes of setup on day one saves hours of guessing later.

The second lesson: aggregate metrics lie. A mean latency of 400ms looks fine until you look at P95 (2.1 seconds) and realize 5% of your users are waiting over 2 seconds. Always instrument percentiles, not just averages.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| Langfuse | Open-source LLM observability; traces, evals, prompt versioning; self-hostable |
| LangSmith | LangChain-integrated; strong for teams already on LangChain |
| Weights & Biases Weave | AI experiment tracking + traces; good for research teams |
| Helicone | Proxy-based; works with any provider; minimal code changes |
| OpenTelemetry | Standard distributed tracing; vendor-neutral; good for multi-service systems |
| Datadog LLM Observability | Enterprise; integrates with existing Datadog stack |
| `presidio` | PII detection and redaction for log scrubbing |

## Common failure modes and gotchas

**1. Logging full prompts at 100% volume.** At scale, storing full 10K-token prompts for every request is expensive and may violate privacy policies. Implement sampling (log 10% at full fidelity, 100% at summary fidelity) and PII scrubbing from the start.

**2. No session ID.** Without a session ID, you can't reconstruct a multi-turn conversation from logs. Every request looks independent. Add a session ID on the first request and pass it through the entire conversation.

**3. Measuring mean latency only.** Mean latency hides tail latency problems. A P99 latency of 8 seconds is invisible in a mean of 600ms. Always instrument P50, P95, and P99.

**4. Not correlating traces to feedback.** If users can thumbs-up/thumbs-down responses, attach that signal to the trace. This turns user feedback into labeled training data and helps you identify which prompts or contexts produce bad outputs.

**5. Logging the system prompt once.** System prompts change over time. If you only log a hash of the system prompt, you can't see what the prompt said three months ago. Store a mapping from prompt hash to full prompt text with timestamps.

**6. Instrumenting only the LLM call, not the pipeline.** If retrieval takes 800ms and generation takes 200ms, your LLM observability looks fine but your system is slow. Instrument every meaningful step: embedding, retrieval, reranking, generation, and any external API calls.

## Project ideas

**1. Observability dashboard** — Build a simple RAG pipeline and instrument it with Langfuse or LangSmith. Add traces for retrieval, reranking, and generation. Run 50 queries and use the dashboard to identify: the 3 slowest queries, the most expensive queries (by token count), and any queries that returned errors. Then fix the slowest one.

**2. Cost attribution by feature** — Add a `feature_name` tag to your traces (e.g., "search", "summarize", "generate"). After 100 requests, compute cost breakdown by feature. This immediately shows which features are expensive relative to their usage frequency.

**3. PII scrubbing pipeline** — Use `presidio` to scan your prompt logs for PII before writing to storage. Measure: what fraction of queries contain detected PII? What false positive rate does presidio have on your domain? Tune the detection thresholds and document the decisions.

**4. Latency profiler** — Instrument a multi-step pipeline with OpenTelemetry spans (embed, retrieve, rerank, generate). Run 100 queries. Export the spans and plot the latency distribution at each step. Identify the bottleneck. Then try one optimization (e.g., async embedding, smaller reranker, lower k) and measure the improvement.

## Going deeper

#### Foundational reading

- OpenTelemetry specification (opentelemetry.io) — the standard distributed tracing model; vendor-neutral and widely supported.
- Kleppmann, *Designing Data-Intensive Applications*, Chapter 10 — on logging and observability in distributed systems; the principles carry directly to AI pipelines.

#### Tools

- Langfuse documentation (langfuse.com/docs) — the most practical reference for LLM-specific observability; includes Python SDK with LangChain and direct Anthropic integrations.
- OpenLLMetry (GitHub: `traceloop/openllmetry`) — OpenTelemetry-based LLM instrumentation; integrates with existing observability infrastructure (Datadog, Jaeger, Grafana).
