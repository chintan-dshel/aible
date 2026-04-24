---
sidebar_position: 2
title: Reliability
description: Retry logic, idempotency, and circuit breakers — building LLM calls that don't silently fail.
---

# Reliability

LLM APIs fail in ways that traditional services don't. Rate limits, server timeouts, and truncated outputs on 200 responses are all normal. A reliability layer handles these gracefully instead of letting them propagate.

## The problem it solves

HTTP 200 doesn't mean the call succeeded. An LLM API can return a 200 with an empty `choices` array, a finish reason of `"length"` (output was cut off), or a content-filtered response that looks like success. At the same time, transient failures — rate limits (429), server errors (500, 529), and network timeouts — need retry logic that doesn't hammer the API during an outage.

Without explicit handling, these failures silently corrupt downstream data or crash request handlers in unpredictable ways.

## How it works under the hood

**Retry with exponential backoff and jitter.** On transient errors, wait `base * 2^attempt + random_jitter` seconds before retrying. The jitter prevents retry storms: without it, hundreds of clients hitting the same outage retry at the same moment, creating a synchronized surge that can overwhelm a service that was just starting to recover.

**Circuit breaker.** Track the rolling failure rate over a time window. When failures exceed a threshold, "open" the circuit — reject calls immediately rather than waiting for timeouts. After a cooldown, enter "half-open" state and allow one test request. If it succeeds, close the circuit.

**Idempotency.** An operation is idempotent if repeating it produces the same result as doing it once. For write operations (storing completions, triggering workflows), use an idempotency key that deduplicates at the handler — otherwise a retry that follows a succeeded-but-timed-out request creates a duplicate.

**Output validation.** After a successful API response, check that the output is actually usable: non-empty, finish reason is `"end_turn"` or `"stop_sequence"`, not blocked by content filters. A `"max_tokens"` stop reason means the output was truncated mid-generation — decide whether to continue with a follow-up call or reject the response.

## Concrete example

```python
import time
import random
import anthropic
from dataclasses import dataclass, field
from enum import Enum

client = anthropic.Anthropic()


class CircuitState(Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # rejecting calls
    HALF_OPEN = "half_open"  # testing recovery


@dataclass
class CircuitBreaker:
    failure_threshold: float = 0.5   # open at 50% failure rate
    window_seconds: int = 60
    cooldown_seconds: int = 30
    min_calls: int = 5                # need this many calls before evaluating

    state: CircuitState = CircuitState.CLOSED
    calls: list = field(default_factory=list)   # (timestamp, success)
    opened_at: float = 0.0

    def record(self, success: bool) -> None:
        now = time.time()
        self.calls = [(t, s) for t, s in self.calls if now - t < self.window_seconds]
        self.calls.append((now, success))

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED if success else CircuitState.OPEN
            if not success:
                self.opened_at = now
            return

        if len(self.calls) >= self.min_calls:
            failure_rate = sum(1 for _, s in self.calls if not s) / len(self.calls)
            if failure_rate >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = now

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.opened_at >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one test request


_circuit = CircuitBreaker()


def call_with_reliability(
    prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    if not _circuit.allow_request():
        raise RuntimeError("Circuit open — upstream service unavailable")

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Validate the output is actually usable
            if not response.content:
                raise ValueError("Empty response from API")
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                raise ValueError(f"Unexpected stop reason: {response.stop_reason}")

            text = response.content[0].text
            if not text.strip():
                raise ValueError("Blank completion")

            _circuit.record(success=True)
            return text

        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            _circuit.record(success=False)
            last_error = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0, base_delay)
                delay = base_delay * (2 ** attempt) + jitter
                time.sleep(delay)

        except (ValueError, anthropic.APIConnectionError) as e:
            _circuit.record(success=False)
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay)

    raise RuntimeError(f"All retries exhausted") from last_error
```

On a rate limit error, the first retry waits ~1s, the second ~2s, the third ~4s, each with random jitter. After 5 calls with a 50%+ failure rate, the circuit opens and callers get an immediate error instead of waiting through timeouts.

## When to use it / when not to

**Use for**: every production LLM call — this is table stakes, not optional.

**Tailor the parameters** to your SLA: a customer-facing chat app needs aggressive timeouts (5s) and quick circuit opens; an overnight batch job can tolerate longer waits and more retries.

**Don't retry**: non-transient errors like authentication failures (401), invalid request formats (400), or content policy rejections. Retrying these wastes quota and delays error surfacing.

## Main tools and libraries

| Tool | Role |
|------|------|
| `tenacity` | Python retry library — cleaner than manual loops, supports `retry_if_exception_type`, `wait_exponential`, `wait_random` |
| Anthropic SDK | Has built-in `max_retries` and `timeout` parameters — use these as the first layer |
| `circuitbreaker` (PyPI) | Decorator-based circuit breaker for Python |
| LiteLLM | Handles retries, fallbacks, and circuit breaking across multiple providers |
| `httpx` | Async-first HTTP client with timeout and retry support for custom integrations |

The Anthropic SDK's built-in retries (`anthropic.Anthropic(max_retries=3)`) are a good first layer. The circuit breaker pattern above is the second layer for sustained outages.

## Common failure modes and gotchas

**Silent truncation.** The API returns 200 and partial output when `max_tokens` is hit. Check `stop_reason == "max_tokens"` and decide whether to retry with continuation or reject the response.

**Retry amplification.** If a downstream service is already overloaded, retrying 3 times triples the load. Exponential backoff + jitter is non-negotiable.

**Idempotency gaps.** Retrying a request that triggers a side effect (sending an email, charging a card, writing a record) can cause duplicates. Use idempotency keys at the effect layer, not just at the LLM call layer.

**Timeout blindspot.** LLM calls can legitimately take 30–60 seconds for long outputs. Setting a 5-second timeout causes false failures. Set timeouts based on expected completion length, or use streaming with per-chunk timeouts.

**Circuit breaker too aggressive.** If the threshold is too low (e.g., open on 1 failure), transient errors permanently block traffic. Require `min_calls` before evaluating and use a rolling window, not a lifetime counter.

**Global circuit breaker in multi-tenant deployments.** The `_circuit` singleton above is shared across all callers. In a multi-tenant service, one tenant's failures — or deliberate quota exhaustion — can open the circuit for every other tenant. Consider per-tenant or per-API-key circuit breakers, or exclude quota-exhaustion errors from the shared circuit's failure accounting.

## Project ideas

- **Retry budget dashboard**: track per-model retry rates and circuit state in a Prometheus/Grafana setup. Alert when retry rate exceeds 5%.
- **Adaptive timeout**: measure rolling p95 latency per model and set timeouts dynamically instead of hardcoding.
- **Chaos test harness**: inject random API errors (mock the Anthropic client) and verify the circuit breaker and retry logic behave correctly.

## Going deeper

- [Anthropic SDK — error handling](https://docs.anthropic.com/en/api/errors) — official error type taxonomy
- [AWS Well-Architected: Reliability Pillar](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html) — retry and circuit breaker patterns at scale
- [Martin Fowler — Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html) — canonical definition
- [tenacity docs](https://tenacity.readthedocs.io/) — Python retry library with wait strategies
