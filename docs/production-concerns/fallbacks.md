---
sidebar_position: 5
title: Fallbacks
description: Model fallback chains, deterministic fallback content, and when to give up gracefully.
---

# Fallbacks

A fallback is what the system does when the primary path fails. For LLM applications, this means defining an ordered sequence of alternative models, providers, and responses — so that a failure in the primary model doesn't become a failure visible to the user.

## The problem it solves

Any single model or provider will occasionally be unavailable: rate limits hit, API downtime, context windows exceeded, content policy rejections. Without a fallback chain, every one of these events is a user-facing error. With a well-designed fallback chain, the system degrades in controlled steps rather than failing outright.

The problem is more subtle than simple retry logic ([[Reliability]] handles brief, transient failures within a single provider — usually resolved in seconds). Fallbacks address a different timescale: the primary model is rate-limited for the next 60 seconds or more, the request exceeds the primary's context window, or a faster/cheaper secondary is preferable for certain request types. The same error type (e.g., a rate limit 429) can belong to both layers: retry logic handles a momentary spike; a fallback chain handles sustained quota exhaustion.

## How it works under the hood

**Ordered fallback chain.** Define models in priority order. When the primary raises a specific exception class (rate limit, timeout, context-length exceeded), move to the next. When the last model fails, serve deterministic fallback content.

**Exception-typed routing.** Not all failures warrant the same fallback. A `RateLimitError` → try secondary model. A `ContextWindowExceeded` → try a model with a larger context window. A content policy rejection → return a static message, not another model (re-trying won't change the policy outcome).

**Capability matching.** The fallback model must be capable of the task. A 4K-token secondary is not a valid fallback for a request that already used 3K tokens of context. Track token counts before routing.

**Response stitching for streaming.** If the primary started streaming before failing mid-response, the fallback needs to either restart from scratch or continue from the partial output. Restarting is simpler and safer.

## Concrete example

```python
import anthropic
from dataclasses import dataclass
from typing import Callable

client = anthropic.Anthropic()


@dataclass
class ModelConfig:
    model_id: str
    max_context: int     # tokens
    cost_per_mtok: float # USD per million tokens


FALLBACK_CHAIN: list[ModelConfig] = [
    ModelConfig("claude-opus-4-7", max_context=200_000, cost_per_mtok=15.0),
    ModelConfig("claude-sonnet-4-6", max_context=200_000, cost_per_mtok=3.0),
    ModelConfig("claude-haiku-4-5-20251001", max_context=200_000, cost_per_mtok=0.25),
]

STATIC_FALLBACK = (
    "I'm temporarily unable to process this request. "
    "Please try again in a few minutes or contact support."
)

# Errors worth trying the next model for
RETRIABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIStatusError,  # 5xx server errors
    anthropic.APITimeoutError,
)

# Errors that won't be helped by retrying a different model
TERMINAL_EXCEPTIONS = (
    anthropic.AuthenticationError,
    anthropic.BadRequestError,  # malformed request
)


def estimate_tokens(messages: list[dict], system: str = "") -> int:
    """Rough token estimate — 4 chars ≈ 1 token."""
    total_chars = sum(len(m.get("content", "")) for m in messages) + len(system)
    return total_chars // 4


def call_with_fallback(
    messages: list[dict],
    system: str = "",
    max_output_tokens: int = 1024,
    on_fallback: Callable[[str, Exception], None] | None = None,
) -> str:
    estimated_input = estimate_tokens(messages, system)

    for config in FALLBACK_CHAIN:
        # Skip models whose context window can't fit this request
        if estimated_input + max_output_tokens > config.max_context:
            continue

        try:
            kwargs: dict = {
                "model": config.model_id,
                "max_tokens": max_output_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            return response.content[0].text

        except TERMINAL_EXCEPTIONS:
            # No point trying other models — the error is in the request itself
            raise

        except RETRIABLE_EXCEPTIONS as e:
            if on_fallback:
                on_fallback(config.model_id, e)
            # Try the next model in the chain

        except Exception as e:
            if on_fallback:
                on_fallback(config.model_id, e)

    # Entire chain exhausted
    return STATIC_FALLBACK


def log_fallback(failed_model: str, error: Exception) -> None:
    # Truncate — API exception messages may echo request content
    print(f"[FALLBACK] {failed_model} failed: {type(error).__name__}: {str(error)[:200]}")


# Usage
result = call_with_fallback(
    messages=[{"role": "user", "content": "Summarize this document..."}],
    system="You are a helpful assistant.",
    on_fallback=log_fallback,
)
```

`call_with_fallback` tries each model in order, skipping those that can't handle the request size, and returns static fallback content only if all models in the chain fail.

## When to use it / when not to

**Use when:**
- You have a hard availability SLA — acceptable downtime is measured in seconds, not minutes
- Your primary model is expensive and you want automatic cost downgrade during high load
- Requests vary widely in complexity (route simple requests to Haiku, complex to Opus, fall back in either direction)

**Don't use when:**
- Task correctness requires a specific model's capabilities — falling back to a weaker model on a reasoning task may return a wrong answer silently rather than failing loudly
- You're in a latency-sensitive path — each fallback attempt adds latency. Set tight timeouts per attempt.

**Design the chain so each step degrades gracefully**, not just differently. If the secondary model produces worse outputs in a way that isn't detectable, a fallback chain hides quality regression rather than surfacing it.

## Main tools and libraries

| Tool | Role |
|------|------|
| LiteLLM | Built-in `fallbacks` and `allowed_fails` parameters — wraps multiple providers with one interface |
| LangChain `with_fallbacks()` | Decorator pattern for chaining any `Runnable` with fallback alternatives |
| Martian | LLM router with automatic fallback and cost optimization |
| Braintrust | Eval platform that can A/B test fallback chains against golden datasets |

LiteLLM is the most practical choice if you're mixing providers (Anthropic + OpenAI + open-weight). For single-provider chains (different Claude models), the pattern above is simpler than adding a dependency.

## Common failure modes and gotchas

**Silent quality degradation.** The fallback succeeds technically (returns text) but the weaker model produces a worse answer. The monitoring system sees a 200 OK and the user sees a bad response — with no error to trigger an alert, quality can degrade for days before anyone notices. Log which model was actually used and track per-model quality metrics separately (see [[Monitoring]] for how to set this up).

**Fallback to a model that also hits the rate limit.** If Opus and Sonnet share the same API key and account-level rate limit, falling back from Opus to Sonnet during a rate limit event may fail for the same reason immediately. Use provider-level fallbacks (Anthropic → OpenAI) for true rate-limit resilience.

**Cost spike during sustained primary outage.** If Haiku is the primary and Opus is the fallback, a 30-minute Haiku outage at scale can generate an unexpectedly large Opus bill. Budget and alert on per-model spend, not just total spend.

**Infinite loops.** If the fallback chain's last entry also fails with a retriable error, the code must return static content rather than looping. Always have a terminal static fallback.

**System prompt differences.** Different models interpret the same system prompt differently. A system prompt tuned for Claude Sonnet may produce different (worse) behavior on a different model. Test fallback behavior explicitly, don't assume prompts transfer.

## Project ideas

- **Fallback rate dashboard**: track what % of requests hit each fallback level. A spike in secondary-model usage signals primary model issues before users start complaining.
- **Fallback quality monitor**: for a sample of fallback-served responses, run an LLM judge to compare quality against primary-model responses on similar prompts. Alert when fallback quality drops below a threshold.
- **Cost-aware routing**: instead of a fixed chain, route based on request complexity score — simple requests always use the cheapest model, only route to expensive models when needed.

## Going deeper

- [LiteLLM fallbacks documentation](https://docs.litellm.ai/docs/routing) — `allowed_fails`, `fallbacks`, `context_window_fallbacks`
- [LangChain `with_fallbacks()`](https://python.langchain.com/docs/how_to/fallbacks/) — runnable-level fallback composition
- [[Reliability]] — retry logic for transient failures within a single model
- [[Graceful Degradation]] — what to do when the entire fallback chain is exhausted
- [[Model Routing]] — proactive routing (before failure) vs. reactive fallback (after failure)
