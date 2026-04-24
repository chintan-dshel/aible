---
sidebar_position: 13
title: Model Routing
description: Capability-based routing, cost routing, and fallback chains — sending the right request to the right model.
---

# Model Routing

## What it is

Model routing is the practice of dynamically selecting which model (or model configuration) handles a given request, based on the request's characteristics — its complexity, the latency budget, the cost constraint, or the required capabilities.

Rather than sending every request to the same model, a routing layer inspects each request and directs it to the most appropriate option: a fast/cheap model for simple queries, a more capable model for complex ones, a specialized fine-tuned model for domain-specific tasks, or a fallback model when the primary is unavailable.

## The problem it solves

A single model choice is always a compromise. A fast, cheap model handles simple queries well but underperforms on complex ones. A highly capable model handles everything well but is overkill (and overpriced) for simple queries. Real-world traffic is heterogeneous — some queries are trivial, some are nuanced — and routing matches resources to requirements.

The cost difference is significant: Claude Haiku costs roughly 4–5× less per token than Sonnet. If 70% of your queries could be handled by Haiku, routing that 70% reduces your model API bill substantially — often by 60–70% on the routed fraction.

## How it works under the hood

### Complexity-based routing

Classify each request by complexity before dispatching it to a model. Use a lightweight classifier (fast model, heuristics, or an embedding-based classifier) that runs in < 50ms and returns a routing decision.

```python
import anthropic
import re

client = anthropic.Anthropic()

SIMPLE_PATTERNS = [
    r"^(what|when|where|who|how|is|are|can|does)\s+\w+",  # short factual questions
    r"^(yes|no|thanks|hello|bye)",                          # greetings/simple responses
    r"^.{0,50}$",                                           # very short queries
]

COMPLEX_SIGNALS = [
    r"(analyze|compare|evaluate|synthesize|design|critique)",
    r"(step\s*by\s*step|explain\s+why|reason\s+through)",
    r"(code|implement|algorithm|architecture)",
]

def classify_complexity(query: str) -> str:
    query_lower = query.lower().strip()

    for pattern in SIMPLE_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return "simple"

    for pattern in COMPLEX_SIGNALS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "complex"

    # Heuristic: query length as a proxy for complexity
    word_count = len(query.split())
    return "simple" if word_count < 20 else "complex"

def route_and_call(query: str, system: str = "") -> dict:
    complexity = classify_complexity(query)

    if complexity == "simple":
        model = "claude-haiku-4-5-20251001"
    else:
        model = "claude-sonnet-4-6"

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    return {
        "response": response.content[0].text,
        "model_used": model,
        "complexity": complexity,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
```

### LLM-as-router

Use a fast model to make the routing decision by classifying the request against a routing schema:

```python
import json
import anthropic

client = anthropic.Anthropic()

ROUTING_PROMPT = """Classify this user request into one category.

Categories:
- simple_factual: Single-fact questions, yes/no questions, greetings
- medium_reasoning: Multi-step questions, comparisons, summaries
- complex_generation: Code writing, analysis, creative tasks, long-form content
- domain_legal: Legal questions requiring precise language
- domain_medical: Medical questions requiring clinical accuracy

Request: {query}

Respond with JSON only: {{"category": "...", "confidence": 0.0-1.0}}"""

def llm_router(query: str) -> str:
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",  # always use fast model for routing
        max_tokens=64,
        messages=[{"role": "user", "content": ROUTING_PROMPT.format(query=query[:300])}]
    )
    try:
        routing = json.loads(result.content[0].text)
        return routing["category"]
    except (json.JSONDecodeError, KeyError):
        return "medium_reasoning"  # default fallback

MODEL_MAP = {
    "simple_factual":      "claude-haiku-4-5-20251001",
    "medium_reasoning":    "claude-haiku-4-5-20251001",
    "complex_generation":  "claude-sonnet-4-6",
    "domain_legal":        "claude-sonnet-4-6",
    "domain_medical":      "claude-sonnet-4-6",
}

def routed_call(query: str) -> dict:
    category = llm_router(query)
    model = MODEL_MAP.get(category, "claude-sonnet-4-6")
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    )
    return {"category": category, "model": model, "response": response.content[0].text}
```

### Fallback chains

Route to a primary model and fall back to an alternative if the primary fails, times out, or returns an error:

```python
import time
import anthropic

client = anthropic.Anthropic()

FALLBACK_CHAIN = [
    "claude-sonnet-4-6",       # primary: best quality
    "claude-haiku-4-5-20251001",  # fallback: faster, cheaper
]

def call_with_fallback(
    messages: list,
    system: str = "",
    max_tokens: int = 512,
    timeout_seconds: float = 10.0,
) -> dict:
    last_error = None

    for model in FALLBACK_CHAIN:
        try:
            start = time.time()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                timeout=timeout_seconds,
            )
            elapsed = time.time() - start
            return {
                "response": response.content[0].text,
                "model": model,
                "latency_ms": round(elapsed * 1000, 1),
                "fallback_used": model != FALLBACK_CHAIN[0],
            }
        except Exception as e:
            last_error = e
            # Log the failure and try the next model in the chain
            print(f"Model {model} failed: {e}. Trying next...")
            continue

    return {"response": "Service unavailable. Please try again.", "error": str(last_error)}
```

### Load-based routing

In high-throughput systems, route based on current load or quota status — send overflow to secondary providers or model tiers:

```python
import threading

class TokenBudgetRouter:
    def __init__(self, hourly_budget: int = 1_000_000):
        self.budget_remaining = hourly_budget
        self._lock = threading.Lock()

    def get_model(self, estimated_tokens: int) -> str:
        with self._lock:
            if self.budget_remaining > estimated_tokens * 2:  # comfortable headroom
                self.budget_remaining -= estimated_tokens
                return "claude-sonnet-4-6"
            elif self.budget_remaining > estimated_tokens:
                self.budget_remaining -= estimated_tokens
                return "claude-haiku-4-5-20251001"
            else:
                return None  # budget exhausted, queue or reject

router = TokenBudgetRouter(hourly_budget=500_000)
```

## Concrete example

A multi-tier customer support router that balances cost and quality:

```python
import anthropic
import json
import re
import time

client = anthropic.Anthropic()

class SupportRouter:
    def __init__(self):
        self.stats = {"simple": 0, "complex": 0, "escalated": 0}

    def _complexity_score(self, query: str) -> int:
        score = 0
        score += min(len(query.split()) // 10, 3)  # length (0-3)
        if re.search(r"(why|how|explain|analyze|compare)", query, re.I):
            score += 2
        if re.search(r"(legal|medical|financial|complaint|refund)", query, re.I):
            score += 2
        return score

    def route(self, query: str) -> str:
        score = self._complexity_score(query)
        if score <= 2:
            self.stats["simple"] += 1
            return "claude-haiku-4-5-20251001"
        elif score <= 5:
            self.stats["complex"] += 1
            return "claude-sonnet-4-6"
        else:
            self.stats["escalated"] += 1
            return "claude-sonnet-4-6"  # highest available; flag for human review

    def answer(self, query: str, system: str) -> dict:
        model = self.route(query)
        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": query}]
        )
        return {
            "response": response.content[0].text,
            "model": model,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

SYSTEM = "You are a helpful customer support agent. Answer only questions about our products and services."
router = SupportRouter()

test_queries = [
    "What are your business hours?",
    "I've been a customer for 3 years and I'm extremely frustrated with the billing error on my account. I need this resolved immediately and want to understand why this happened and what you're doing to prevent it in the future.",
    "Can I return this?",
]

for q in test_queries:
    result = router.answer(q, SYSTEM)
    print(f"Model: {result['model']} | Latency: {result['latency_ms']}ms | Tokens: {result['input_tokens']}+{result['output_tokens']}")
    print(f"Query: {q[:60]}...")
    print()

print("Routing distribution:", router.stats)
```

## When to use it / when not to

#### Implement routing when

- You have heterogeneous query complexity (some simple, some complex)
- Cost is a meaningful constraint — you're processing thousands of queries per day
- Different query types have different quality requirements (FAQ vs. complex analysis)
- You need fallback resilience (production systems where uptime matters)

#### Skip routing when

- All queries have similar complexity — routing overhead isn't worth it
- You're at low volume — a single model choice is simpler and routing complexity isn't justified
- Your model budget isn't a binding constraint

#### The practical test

Route 100 real queries through a complexity classifier. Manually review the routing decisions. If > 80% of simple-routed queries look genuinely simple and > 80% of complex-routed queries look genuinely complex, your router is working. If the routing is wrong on > 25% of queries, either your classifier is miscalibrated or your complexity distinction isn't meaningful for your use case.

:::tip[My take]

The biggest mistake with routing is over-engineering the classifier. A regex heuristic that takes 1ms and gets 80% accuracy often beats a 200ms LLM-based classifier that gets 85% accuracy — the latency gain from routing to Haiku is partially eaten by the classifier's overhead. Start with length + keyword heuristics. Only add an LLM router when you can show it meaningfully improves routing accuracy on your query distribution.

The fallback chain is not optional for production systems. Models have occasional outages, rate limits, and timeout events. A simple try/except that falls back to Haiku when Sonnet fails adds 10 lines of code and prevents complete outages from partial provider issues.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| LiteLLM | Unified API for multiple providers; built-in fallback, retry, and cost tracking |
| OpenRouter | Managed routing across multiple model providers via a single API |
| LangChain RouterChain | High-level routing abstraction; useful if already on LangChain |
| Custom classifier | Any fast model or heuristic; highest control, lowest latency |
| RouteLLM (lm-sys) | Trained routing classifier that predicts which model quality tier a query needs |

## Common failure modes and gotchas

**1. Routing cost exceeds routing savings.** If your LLM-based router takes 200ms and costs $0.001 per call, but only routes 20% of traffic to Haiku saving $0.001 per call, the math doesn't work. Measure the cost and latency of the router itself before assuming it's beneficial.

**2. Quality regression in the "simple" tier.** Queries classified as simple and routed to Haiku may actually require Sonnet-level reasoning on some inputs. Measure quality separately for simple-routed and complex-routed queries. If simple-routed quality is significantly worse, either your classifier is miscalibrated or your quality bar is wrong.

**3. Feedback loop blind spots.** If you only evaluate outputs from the complex-routed model, you may miss quality degradation in the simple-routed tier. Eval your routing tiers separately.

**4. No monitoring on routing distribution.** Track what fraction of traffic routes to each tier over time. If the distribution shifts (e.g., 30% → 60% going to Sonnet), your query distribution changed and your classifier may need recalibration.

**5. Fallback masking systematic failures.** If your primary model fails and falls back silently, you may not notice a real problem. Log every fallback event separately from normal traffic. Alert when fallback rate exceeds threshold.

**6. Routing doesn't account for context length.** A query with a 50K-token retrieved context routed to a cheap model may fail if that model has a shorter context window. Check context length compatibility before routing.

## Project ideas

**1. Routing accuracy benchmark** — Take 100 real queries from a domain and manually label each as simple/medium/complex. Run three routers: (a) length heuristic, (b) keyword heuristic, (c) LLM classifier. Measure accuracy of each against your manual labels. Compare routing latency and cost overhead. Find the best cost-accuracy router for your domain.

**2. Cost reduction analysis** — Log a week of production queries. Retroactively classify them by complexity. Compute: what would total model cost have been with and without routing? This quantifies the ROI of routing for your actual traffic pattern.

**3. Fallback resilience test** — Build a fallback chain. Simulate primary model failure by raising an exception in the primary call. Verify the fallback fires correctly, the latency increase is within bounds, and the fallback event is logged. Measure how many retries happen before a successful response.

**4. RouteLLM integration** — Use the open-source RouteLLM classifier from lm-sys. Compare its routing decisions on your query set against your hand-labeled complexity scores. Measure its routing latency. Decide whether it outperforms your heuristic classifier enough to justify the added dependency.

## Going deeper

#### Foundational reading

- Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data" (2024) — trains a routing classifier on human preference data to predict which model tier will produce good enough results at minimum cost.
- Anthropic API documentation — model capabilities and pricing reference for building routing logic.

#### Tools

- LiteLLM documentation (docs.litellm.ai) — the most practical guide to multi-model routing; supports Anthropic, OpenAI, and many others with unified syntax.
- OpenRouter (openrouter.ai) — managed routing as a service; useful for prototyping before building custom routing logic.
