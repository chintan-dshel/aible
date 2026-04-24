---
sidebar_position: 12
title: Latency Optimization
description: Streaming, batching, speculative decoding, quantization — how to make AI responses feel fast.
---

# Latency Optimization

## What it is

Latency optimization for AI systems is the set of techniques that reduce the time between a user's request and a useful response — either by making the underlying computation faster, or by changing what the user perceives as "response time" through streaming and progressive delivery.

AI latency has two distinct components with different characteristics:
- **Time to first token (TTFT)**: how long until the first token appears. Determines perceived responsiveness.
- **Time to last token (TTLT)**: total end-to-end generation time. Determines when the response is complete.

Users tolerate high TTLT better than high TTFT. A response that starts streaming in 300ms and takes 3 seconds total feels faster than one that waits 2 seconds then delivers all at once.

## The problem it solves

LLM inference is slow relative to user expectations. Users experience latency as friction; latency above 2 seconds for first token hurts engagement. The latency problem is structural: autoregressive generation produces one token at a time, and each token requires a full forward pass through a large model.

Compounding factors:
- **Context length grows latency**: processing 50K input tokens takes significantly longer than processing 1K tokens — even with KV caching, prefill time scales with input length.
- **Output length is unpredictable**: you can't know in advance how many tokens the model will generate.
- **Network adds overhead**: API calls add round-trip latency; each model call adds a separate network hop.
- **Guardrail chains multiply latency**: input classifier + model call + output classifier = 3 sequential network hops.

## How it works under the hood

### Streaming

The highest-impact single change: stream the model's output token-by-token rather than waiting for the complete response. Users see text appearing immediately after TTFT, making a 3-second response feel much faster than a 3-second wait followed by a full response.

```python
import anthropic

client = anthropic.Anthropic()

def stream_response(prompt: str) -> None:
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # newline after stream completes

stream_response("Explain transformer attention in three paragraphs.")
```

For web applications, stream tokens over Server-Sent Events (SSE) or WebSockets. The UI renders tokens as they arrive. Most LLM API providers support streaming; it's always worth enabling.

**What streaming doesn't help:** tasks where you need the complete response before acting on it (structured extraction, function calling, validation). For those, streaming adds complexity without benefit.

### Parallelizing independent calls

If your pipeline makes multiple independent LLM calls, run them concurrently:

```python
import asyncio
import anthropic

client = anthropic.Anthropic()

async def async_call(prompt: str, label: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
    )
    return label, response.content[0].text

async def parallel_pipeline(query: str) -> dict:
    tasks = [
        async_call(f"Is this query safe? {query}", "safety"),
        async_call(f"What topic is this query about? {query}", "topic"),
        async_call(f"What language is this query in? {query}", "language"),
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)

output = asyncio.run(parallel_pipeline("What's your return policy?"))
```

Sequential calls add latency linearly. If you have 3 independent classifier calls each taking 200ms, sequential execution = 600ms; parallel = 200ms.

### Model selection for latency

Smaller models are dramatically faster and cheaper. Use the smallest model that meets your quality bar for each task:

```python
def classify_topic(query: str) -> str:
    # Haiku: < 100ms for classification tasks
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": f"Topic (one word): {query}"}]
    )
    return response.content[0].text.strip()

def generate_full_response(query: str, context: str) -> str:
    # Sonnet: higher quality for the actual user-facing response
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Context: {context}\n\n{query}"}]
    )
    return response.content[0].text
```

Haiku is typically 3–5× faster than Sonnet for the same task. For guardrails, classifiers, and routing decisions, always default to Haiku.

### Reducing input tokens

Longer prompts take longer to process (prefill time scales with input length). Techniques to reduce input tokens:

- **Truncate retrieved context**: pass only the top-3 retrieved chunks, not top-10. Often similar quality at 60% fewer context tokens.
- **Compress conversation history**: summarize older turns into a compact representation rather than keeping the full transcript.
- **Remove redundant instructions**: "please", "you should", "I would like you to" add tokens without adding instruction clarity.
- **Use structured formats**: XML tags and brief instructions are often more efficient than lengthy prose instructions.

### max_tokens discipline

Setting `max_tokens` too high doesn't cause slowness by itself (you pay only for tokens generated), but it prevents early stopping. Set `max_tokens` to the maximum reasonable length for your task — not the model's maximum.

For classification tasks where the answer is one word, `max_tokens=16` prevents the model from generating a lengthy explanation. For summaries, `max_tokens=256` prevents run-on responses.

### Async guardrail chains

If your pipeline runs input guardrails → model call → output guardrails sequentially, this is the baseline latency. Optimize by:

- Running cheap checks in parallel with the model call (if the check doesn't affect whether you call the model)
- Deferring expensive output checks to async workers (log and alert on violation, but don't block the user response for low-risk cases)

```
Sequential:  [input guard 200ms] → [model 600ms] → [output guard 200ms] = 1000ms
Parallel:    [model 600ms] (while input guard 200ms runs in background)
             + [output guard 200ms] = ~800ms
Async guard: [model 600ms] + async [output guard] → response returned, violations logged = 600ms
```

**Which output checks must stay synchronous (blocking):** safety and moderation checks (harmful content, jailbreak detection), accuracy checks on factual claims with high stakes, and any compliance check in regulated domains (financial advice, medical information, legal guidance). For these, the latency cost is non-negotiable — users receiving a harmful response is worse than a 200ms delay. Only defer checks where a false negative (violation undetected) has low consequence, such as formatting validators or metadata classifiers.

## Concrete example

A latency-instrumented pipeline showing where time is spent:

```python
import anthropic
import time
from dataclasses import dataclass, field

client = anthropic.Anthropic()

@dataclass
class LatencyTrace:
    stages: dict = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float):
        self.stages[stage] = round(duration_ms, 1)

    def report(self):
        total = sum(self.stages.values())
        for stage, ms in self.stages.items():
            print(f"  {stage}: {ms}ms ({ms/total:.0%})")
        print(f"  TOTAL: {total}ms")

def timed(label: str, trace: LatencyTrace):
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            trace.record(label, (time.time() - self.start) * 1000)
    return Timer()

def instrumented_pipeline(query: str) -> tuple[str, LatencyTrace]:
    trace = LatencyTrace()

    with timed("input_guard", trace):
        import re
        blocked = bool(re.search(r"(ignore|disregard)\s+(previous|prior)\s+instructions", query, re.I))
        if blocked:
            return "Blocked.", trace

    with timed("retrieval", trace):
        time.sleep(0.05)  # simulate 50ms retrieval
        context = "Our return policy is 30 days for all products."

    with timed("generation", trace):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": f"Context: {context}\n\n{query}"}]
        )
        answer = response.content[0].text

    with timed("output_guard", trace):
        time.sleep(0.03)  # simulate 30ms output check
        pass

    return answer, trace

answer, trace = instrumented_pipeline("What is the return policy?")
print(f"Answer: {answer[:100]}")
print("Latency breakdown:")
trace.report()
```

Running this shows exactly where time is spent — often revealing that retrieval or guardrails dominate, not model generation.

## When to use it / when not to

#### Always do

- **Streaming**: for any user-facing generation. No downside; dramatic perceived latency improvement.
- **Model selection**: use Haiku for classifiers, guardrails, and routing. Reserve Sonnet for user-facing responses.
- **Latency instrumentation**: always record per-stage timing. You can't optimize what you don't measure.

#### Do when latency is a constraint

- **Parallel calls**: when you have ≥ 2 independent model calls in a pipeline
- **Input token reduction**: when your context blocks exceed 8K tokens and latency is above target
- **max_tokens discipline**: when your task has a natural output length ceiling

#### Tradeoffs to consider carefully

- **Async output guardrails**: reduces latency but allows bad responses to reach users before violation detection. Only appropriate for low-risk output categories.
- **Smaller models for generation**: Haiku is faster but lower quality. Benchmark quality before downgrading.

:::tip[My take]

Streaming is the single highest-impact change for user-facing applications. It costs you nothing in quality or money, and it fundamentally changes how users perceive your system's speed. Do this first.

After streaming, instrument before optimizing. The bottleneck is almost never where you expect. I've seen pipelines where the retrieval step dominated latency at 800ms while the model call was only 200ms — the obvious target was wrong. Time everything, then optimize the actual bottleneck.

The guardrail latency problem is real. Running three model calls (input classifier, main generation, output classifier) sequentially often makes the guardrails cost more in latency than the actual generation. Run cheap checks in parallel and expensive checks asynchronously where the risk profile allows.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| Anthropic streaming API | Native streaming with `client.messages.stream()` |
| `asyncio` + `httpx` | Concurrent async API calls in Python |
| LiteLLM | Model routing with built-in latency tracking and fallbacks |
| vLLM | High-throughput self-hosted inference with PagedAttention |
| TensorRT-LLM | NVIDIA-optimized inference; significant throughput gains on GPU |
| Quantized models (GGUF, AWQ) | Lower-memory, faster local inference at some quality cost |

## Common failure modes and gotchas

**1. Optimizing the wrong bottleneck.** Common assumption: the model call is the bottleneck. Often wrong. Database queries, embedding calls, and retrieval steps frequently dominate. Instrument everything before optimizing anything.

**2. Streaming when the client can't render it.** Streaming tokens into a pipeline that buffers the full response before acting (function calling, validators, structured extraction) adds complexity with no benefit. Only stream when the output renders progressively.

**3. Parallel calls without error handling.** `asyncio.gather()` raises the first exception by default, cancelling all other tasks. Use `return_exceptions=True` for resilience, then handle exceptions per-task.

**4. Reducing max_tokens too aggressively.** Setting `max_tokens=64` for a task that occasionally requires 200 tokens causes truncation. The model stops mid-sentence. Always set `max_tokens` to the 99th percentile of expected output length for that task.

**5. Not measuring TTFT separately from TTLT.** P95 total latency looks fine while P95 TTFT is 2 seconds — users see a blank screen for 2 seconds before anything appears. Track both metrics.

**6. Model downgrade without quality benchmark.** Switching from Sonnet to Haiku cuts latency and cost but may significantly degrade quality on complex tasks. Always run your eval set on both models before downgrading in production.

## Project ideas

**1. Latency profiler** — Instrument a 3-stage pipeline (retrieve, classify, generate) with per-stage timing. Run 50 queries. Plot the distribution of each stage's latency contribution. Identify the bottleneck. Then apply one optimization (parallel calls, model downgrade for a stage, reduced context) and measure the improvement.

**2. Streaming UX study** — Build the same Q&A interface in two versions: streaming (tokens appear progressively) and batch (full response appears after N seconds). Show both to 5+ users. Ask which feels faster. Document the perception gap versus actual latency — this makes the UX value of streaming concrete.

**3. TTFT vs TTLT analysis** — Run 100 requests through a streaming pipeline. Record TTFT and TTLT for each. Plot both distributions (P50/P95/P99). Compute the correlation between input length and TTFT. This shows how context length affects perceived responsiveness.

**4. Model tier latency benchmark** — On the same set of 50 classification tasks, measure Haiku vs Sonnet: latency (P50/P95), quality (LLM-as-judge score), and cost. Find the cost-latency-quality Pareto frontier. Decide which tier is appropriate for which pipeline stage.

## Going deeper

#### Foundational reading

- Pope et al., "Efficiently Scaling Transformer Inference" (2022) — analysis of memory and compute bottlenecks in transformer inference; explains why KV cache memory becomes the constraint at long context.
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023) — the paper behind speculative decoding; explains the draft-verify loop and acceptance rate.

#### Tools

- vLLM documentation (docs.vllm.ai) — the reference for high-throughput self-hosted inference; explains PagedAttention and continuous batching.
- Anthropic streaming documentation — how to use `client.messages.stream()` and handle streaming responses correctly.
