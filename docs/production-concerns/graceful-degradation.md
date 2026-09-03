---
sidebar_position: 3
title: Graceful Degradation
description: Fallback content, confidence thresholds, and human escalation — what happens when the model shouldn't be trusted.
---

# Graceful Degradation

When an LLM can't be trusted for a given request, you have three options: crash loudly, return a wrong answer silently, or degrade gracefully. Only the third is acceptable in production.

## The problem it solves

"Fail loud" means surfacing an unhandled exception to the user — bad UX but at least honest. "Fail silent" means returning a hallucinated or low-confidence answer as if it were reliable — the worst outcome. Graceful degradation is the third path: the system detects it can't produce a reliable answer and responds in a controlled, safe way — a fallback message, a reduced-feature mode, or a handoff to a human.

This matters especially in regulated domains (medical, legal, financial), high-stakes actions (booking, purchasing, deleting data), and anywhere the cost of a wrong answer exceeds the cost of a non-answer.

## How it works under the hood

**Confidence threshold routing.** Estimate confidence in the model's output — via logprobs (the model's own raw per-word probability scores, when the API exposes them), self-consistency (asking the model the same question several times and checking how often it agrees with itself), or an LLM judge (a second model call that grades the first model's output); see [[Confidence Estimation]] for how each technique works. If confidence falls below a threshold, route to a fallback instead of serving the output.

**Fallback content hierarchy.** Define an ordered sequence of increasingly conservative responses:
1. Try the primary model
2. Try a secondary model or simplified prompt
3. Serve static / pre-approved fallback content
4. Acknowledge receipt and escalate to a human queue

**Feature-level degradation.** Disable only the AI-powered feature, not the entire application. A search bar without semantic ranking is worse than one with it, but still usable. This requires building AI features as optional enhancements, not load-bearing dependencies.

**Human escalation queue.** Route low-confidence or high-stakes requests to a review queue. The user gets an acknowledgment ("We're reviewing this manually and will respond within 24h") instead of a wrong answer.

## Concrete example

```python
import anthropic
from dataclasses import dataclass
from enum import Enum

client = anthropic.Anthropic()


class DegradationLevel(Enum):
    FULL = "full"           # normal AI response
    REDUCED = "reduced"     # simplified model / prompt
    STATIC = "static"       # pre-approved fallback content
    ESCALATED = "escalated" # human review queue


@dataclass
class DegradedResponse:
    content: str
    level: DegradationLevel
    reason: str


STATIC_FALLBACKS = {
    "billing": "For billing questions, contact support@example.com or call 1-800-555-0100.",
    "legal": "We can't provide legal advice. Please consult a qualified attorney.",
    "medical": "Please consult a healthcare professional for medical guidance.",
    "default": "I'm unable to answer this confidently right now. A team member will follow up.",
}


def estimate_confidence(response_text: str, prompt: str) -> float:
    """Use a lightweight LLM-as-judge to score response confidence (0.0–1.0)."""
    # User content is interpolated here — a crafted prompt can manipulate the judge's
    # score (prompt injection). Treat this score as a soft signal, not a hard gate.
    judge_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Rate from 0.0 to 1.0 how confident and accurate this response is.\n"
                f"Question: {prompt}\nResponse: {response_text}\n"
                f"Reply with only a number between 0.0 and 1.0."
            ),
        }],
    )
    try:
        score = float(judge_response.content[0].text.strip())
        return max(0.0, min(1.0, score))  # clamp — out-of-range value bypasses routing threshold
    except ValueError:
        return 0.0  # unparseable → treat as low confidence


def enqueue_for_human_review(prompt: str, context: dict) -> str:
    """Submit to human review queue. Returns a ticket ID."""
    # In production: insert to DB, Slack webhook, Zendesk, etc.
    ticket_id = f"ESC-{hash(prompt) % 100000:05d}"
    # Never log prompt content to stdout — use a secured audit log for sensitive content
    print(f"[ESCALATION] Ticket {ticket_id} queued (topic: {context.get('topic', 'unknown')})")
    return ticket_id


def respond_with_degradation(
    prompt: str,
    topic: str = "default",
    confidence_threshold: float = 0.7,
    allow_escalation: bool = True,
) -> DegradedResponse:
    # Level 1: try primary model
    try:
        primary = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = primary.content[0].text

        confidence = estimate_confidence(text, prompt)

        if confidence >= confidence_threshold:
            return DegradedResponse(text, DegradationLevel.FULL, "high confidence")

        # Level 2: try simplified prompt with reduced scope
        simplified = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system="Answer only if you are highly certain. Otherwise say 'I don't know.'",
            messages=[{"role": "user", "content": prompt}],
        )
        reduced_text = simplified.content[0].text
        if "i don't know" not in reduced_text.lower():
            return DegradedResponse(reduced_text, DegradationLevel.REDUCED, "low primary confidence")

    except Exception:
        pass  # fall through to static/escalation

    # Level 3: static fallback
    fallback = STATIC_FALLBACKS.get(topic, STATIC_FALLBACKS["default"])

    if allow_escalation:
        # Level 4: escalate and acknowledge
        ticket_id = enqueue_for_human_review(prompt, {"topic": topic})
        return DegradedResponse(
            f"{fallback}\n\nYour request has been escalated (ref: {ticket_id}).",
            DegradationLevel.ESCALATED,
            "below confidence threshold — human review queued",
        )

    return DegradedResponse(fallback, DegradationLevel.STATIC, "below confidence threshold")
```

The system tries the primary model, checks confidence, attempts a more conservative simplified prompt, then falls back to static content and human escalation. The caller always gets a `DegradedResponse` — never an unhandled exception.

## When to use it / when not to

**Use when:**
- Wrong AI output is worse than no AI output (medical, legal, financial, high-stakes actions)
- The system touches external state (bookings, payments, deletions) — never let a low-confidence output trigger an irreversible action
- You have an SLA that requires graceful handling rather than 500 errors

**Don't over-index on it when:**
- The feature is purely informational and low-stakes — a wrong suggestion in a recipe recommender doesn't need a human escalation queue
- Degradation adds more latency than users will tolerate — time the fallback path

## Main tools and libraries

| Tool | Role |
|------|------|
| LangChain `FallbackRunnable` | Wraps a chain with an ordered fallback sequence |
| LiteLLM | Provider-level fallbacks (if Anthropic fails, try OpenAI) |
| Celery / Redis | Background task queue for human escalation — don't block the request thread |
| Zendesk / Linear / Slack webhooks | Human review routing destination |

## Common failure modes and gotchas

**Stale static fallback content.** A pre-approved static response can become wrong over time (outdated pricing, deprecated product). Treat static fallbacks as content — give them version control and review cycles.

**Escalation queue overflow.** If the confidence threshold is too strict, most requests escalate and humans can't keep up. The queue becomes a black hole. Monitor queue depth and escalation rate as first-class metrics.

**Threshold miscalibration.** A threshold of 0.7 that sounds reasonable might cause 90% escalation in practice. Tune thresholds against held-out labeled data — a set of examples you already know the right answer to, kept aside for testing rather than used during development — not intuition.

**Degradation hiding model failure.** If the primary model starts consistently producing low-confidence outputs due to a regression, graceful degradation masks it. Alert when degradation rate spikes — don't let it run silently at 80%.

**Feature coupling.** If the AI feature is entangled with the core data pipeline (e.g., classification controls which records are shown), "disable AI" breaks the whole product. Build AI as an optional enrichment layer from the start.

## Project ideas

- **Degradation rate dashboard**: track what % of requests hit each level (FULL / REDUCED / STATIC / ESCALATED) over time. Spike in ESCALATED = model regression signal.
- **Escalation inbox**: a simple web UI showing the human review queue, with one-click "approve", "edit", and "reject" actions that feed back into a training dataset.
- **Threshold calibration tool**: given a labeled validation set, find the confidence threshold that minimizes `cost_of_wrong_answer * false_positives + cost_of_escalation * false_negatives` — in plain terms, the threshold where the combined cost of wrongly-confident answers and unnecessarily-escalated ones is lowest; e.g., if a wrong answer costs $50 and an unneeded escalation costs $2, the formula will favor a stricter threshold that escalates more often, because the $2 mistakes are far cheaper than the $50 ones.

## Going deeper

- [Google SRE Book — Handling Overload](https://sre.google/sre-book/handling-overload/) — load shedding and degradation patterns from distributed systems
- [Martin Fowler — Graceful Degradation](https://martinfowler.com/articles/patterns-of-distributed-systems/graceful-degradation.html)
- [[Confidence Estimation]] — how to produce the confidence score that gates degradation
- [[Fallbacks]] — model-level fallback chains (the layer above this one)
