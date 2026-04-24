---
sidebar_position: 6
title: Monitoring
description: Drift detection, quality regression, and alerting — keeping tabs on a system that outputs free text.
---

# Monitoring

Traditional application monitoring tracks uptime, error rates, and latency. LLM systems need all of that plus something harder: detecting when the outputs are getting worse without a clear error signal.

## The problem it solves

A 99.9% success rate means almost nothing for an LLM system if "success" is defined as "returned HTTP 200". The model can return a confident, fluent, completely wrong answer and every monitoring dashboard stays green. Meanwhile, quality has degraded — the model started refusing more requests, hallucinating product names, or subtly shifting tone — and no alert fired.

LLM monitoring is hard because the output space is free text and ground truth labels are expensive to produce at scale. The challenge is detecting quality regression quickly without requiring a human to read every response.

## How it works under the hood

**Sampling-based LLM-as-judge.** Evaluate a random sample of production responses (typically 1–5%) using a separate judge model. The judge scores each response on task-specific dimensions (accuracy, relevance, safety, tone). Track average judge scores over time. A sustained drop = quality regression signal.

**Input distribution monitoring.** Embed incoming prompts and track the distribution against a baseline. A shift in embedding distribution (cosine distance from baseline centroid, or increases in distribution entropy) signals that users are sending different kinds of requests — which may break prompts tuned for the original distribution.

**Structural metrics.** Track response-level signals that don't require a judge: average response length, refusal rate (% of responses containing "I can't help with"), completion length relative to `max_tokens` (if many responses hit `max_tokens`, the model is being asked to do more than it can in the token budget), and finish reason distribution.

**Latency and cost.** Standard metrics — but with LLM-specific dimensions: p50/p95/p99 latency per model, tokens per request (input + output separately), cost per request, and cache hit rate (if prompt caching is active).

**Alert hierarchy.** Define tiers: immediate page (error rate > 5%, latency p99 > 10s), async alert (judge score drop > 10% over 24h, refusal rate spike), weekly digest (trend reports, cost drift, distribution shift).

## Concrete example

:::caution[Privacy: ResponseRecord contains raw user prompts]
`ResponseRecord.prompt` stores the user's input verbatim. Before writing records to any persistent store (database, BigQuery, S3), apply PII scrubbing or tokenization. Truncating to 500 characters is a cost control measure, not a privacy control. In user-facing applications, treat every prompt as potentially containing names, emails, medical details, or credentials.
:::

```python
import random
import time
from dataclasses import dataclass, field
from collections import deque
import anthropic

client = anthropic.Anthropic()


@dataclass
class ResponseRecord:
    prompt: str
    response: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityScore:
    overall: float          # 0.0–1.0
    accuracy: float
    relevance: float
    safety: float
    reasoning: str


JUDGE_PROMPT = """You are evaluating an AI assistant response for production quality monitoring.

Question: {prompt}
Response: {response}

Score each dimension from 0.0 to 1.0:
- accuracy: Is the information factually correct?
- relevance: Does the response address what was asked?
- safety: Is the response safe and appropriate?

Reply in this exact format:
accuracy: <score>
relevance: <score>
safety: <score>
reasoning: <one sentence>"""


def judge_response(record: ResponseRecord) -> QualityScore:
    judge = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                # User content interpolated here — a crafted prompt can manipulate the
                # judge score. Validate output is in [0.0, 1.0] after parsing.
                prompt=record.prompt[:500],
                response=record.response[:500],
            ),
        }],
    )
    text = judge.content[0].text
    scores = {}
    reasoning = ""
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key in ("accuracy", "relevance", "safety"):
                try:
                    scores[key] = float(val)
                except ValueError:
                    scores[key] = 0.5
            elif key == "reasoning":
                reasoning = val

    return QualityScore(
        overall=sum(scores.values()) / max(len(scores), 1),
        accuracy=scores.get("accuracy", 0.5),
        relevance=scores.get("relevance", 0.5),
        safety=scores.get("safety", 0.5),
        reasoning=reasoning,
    )


class LLMMonitor:
    def __init__(self, sample_rate: float = 0.05, window_size: int = 100):
        self.sample_rate = sample_rate
        self.window: deque[ResponseRecord] = deque(maxlen=window_size)
        self.quality_scores: deque[float] = deque(maxlen=window_size)
        self.refusal_keywords = ["i can't help", "i'm unable to", "i cannot assist"]

    def record(self, record: ResponseRecord) -> None:
        self.window.append(record)

        if random.random() < self.sample_rate:
            score = judge_response(record)
            self.quality_scores.append(score.overall)

    def is_refusal(self, text: str) -> bool:
        lowered = text.lower()
        return any(kw in lowered for kw in self.refusal_keywords)

    def metrics(self) -> dict:
        if not self.window:
            return {}

        records = list(self.window)
        avg_latency = sum(r.latency_ms for r in records) / len(records)
        refusal_rate = sum(1 for r in records if self.is_refusal(r.response)) / len(records)
        avg_output_tokens = sum(r.output_tokens for r in records) / len(records)
        truncation_rate = sum(
            1 for r in records if r.finish_reason == "max_tokens"
        ) / len(records)
        avg_quality = (
            sum(self.quality_scores) / len(self.quality_scores)
            if self.quality_scores else None
        )

        return {
            "window_size": len(records),
            "avg_latency_ms": round(avg_latency, 1),
            "refusal_rate": round(refusal_rate, 3),
            "avg_output_tokens": round(avg_output_tokens, 1),
            "truncation_rate": round(truncation_rate, 3),
            "avg_quality_score": round(avg_quality, 3) if avg_quality is not None else None,
            "quality_sample_count": len(self.quality_scores),
        }

    def check_alerts(self) -> list[str]:
        m = self.metrics()
        alerts = []
        if m.get("refusal_rate", 0) > 0.1:
            alerts.append(f"HIGH REFUSAL RATE: {m['refusal_rate']:.1%}")
        if m.get("truncation_rate", 0) > 0.2:
            alerts.append(f"HIGH TRUNCATION RATE: {m['truncation_rate']:.1%} — increase max_tokens or reduce input")
        if m.get("avg_quality_score") is not None and m["avg_quality_score"] < 0.6:
            alerts.append(f"LOW QUALITY SCORE: {m['avg_quality_score']:.2f} — investigate model or prompt")
        return alerts


monitor = LLMMonitor(sample_rate=0.05)


def monitored_call(prompt: str) -> str:
    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.time() - start) * 1000
    text = response.content[0].text

    monitor.record(ResponseRecord(
        prompt=prompt,
        response=text,
        model="claude-sonnet-4-6",
        latency_ms=latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        finish_reason=response.stop_reason,
    ))

    return text
```

The monitor records every response and judges 5% of them. `check_alerts()` can be called on a schedule (e.g., every 5 minutes via a cron job) to detect regressions.

## When to use it / when not to

**Use in production always.** Monitoring is not optional. The only question is what to monitor.

**Tune the sample rate** to your traffic volume. At 100 requests/day, 5% = 5 judge calls per day (fine). At 100,000 requests/day, 5% = 5,000 judge calls per day — you may want to reduce to 0.5% and increase sample rate only when an alert fires.

**Don't monitor everything.** Define 3–5 metrics that actually matter for your use case and set thresholds for them. A dashboard with 30 metrics and no thresholds is surveillance theater.

## Main tools and libraries

| Tool | Role |
|------|------|
| Langfuse | Open-source LLM observability — traces, scores, prompt versions, cost tracking |
| Braintrust | Eval and monitoring platform — golden dataset evals + production sampling |
| Helicone | Proxy-based logging — captures every LLM call without code changes |
| Arize / Phoenix | ML observability with LLM-specific embedding drift detection |
| Prometheus + Grafana | Standard metrics infrastructure — works well for latency/cost/error rate dashboards |
| OpenTelemetry | Standard tracing instrumentation — `opentelemetry-instrumentation-anthropic` for automatic traces |

Langfuse or Helicone for tracing + Prometheus/Grafana for operational metrics is a common low-cost stack for production LLM apps.

## Common failure modes and gotchas

**Judge same model as target.** Using claude-sonnet-4-6 to judge claude-sonnet-4-6 outputs means the judge shares the same failure modes. Use a different model family, or at minimum a different model version, as the judge.

**Alert fatigue from noisy metrics.** Response length is noisy — it varies with prompt length. Refusal rate is noisy on small sample windows. Set alert thresholds against a baseline measured over at least a week of production traffic, not intuition.

**Sampling bias.** If you sample only short responses (faster to judge), you miss quality issues in long-form outputs. Sample uniformly across response length buckets.

**Monitoring infra failing silently.** If the monitoring pipeline itself is broken — judge calls failing, metrics not writing, alerts not firing — you're flying blind and don't know it. Monitor your monitoring: alert if no quality scores have been recorded in the last 6 hours.

**Drift without regression.** Input distribution shift isn't always a problem — it may mean users discovered a new use case. Don't auto-alert on distribution shift alone; investigate whether it's correlated with quality degradation.

## Project ideas

- **Quality regression detector**: maintain a rolling 7-day quality score baseline and alert if the current 24h average drops more than 2 standard deviations below the baseline.
- **Prompt version tracker**: tag each LLM call with the prompt version that generated it. When you update a prompt, compare quality scores before and after across matched request types.
- **Cost anomaly detector**: track daily token spend per model. Alert if any model's spend increases more than 30% day-over-day (indicates traffic spike, runaway loop, or prompt regression that generates long outputs).

## Going deeper

- [Langfuse documentation](https://langfuse.com/docs) — open-source LLM observability platform
- [Arize Phoenix](https://phoenix.arize.com/) — embedding drift detection and LLM tracing
- [[Evals]] — offline evaluation that complements production monitoring
- [[Observability]] — distributed tracing and prompt logging (the underlying infrastructure)
- [[Cost Tracking]] — per-request token accounting and budget guardrails
