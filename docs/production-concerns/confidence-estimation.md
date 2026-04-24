---
sidebar_position: 4
title: Confidence Estimation
description: Logprobs, self-consistency, and calibration — knowing when to trust the model.
---

# Confidence Estimation

LLMs produce fluent, confident-sounding text regardless of whether they actually know the answer. Confidence estimation is the set of techniques for producing a numeric signal that correlates with whether the output is actually correct.

## The problem it solves

A model that says "The capital of France is Paris" and "The Eiffel Tower was built in 1887" sounds equally certain. The first is right; the second is wrong by 2 years. Without an external confidence signal, downstream systems can't distinguish reliable outputs from hallucinations — and can't decide when to route to a fallback, trigger human review, or abstain.

Calibrated confidence enables: routing (high confidence → serve, low → escalate), uncertainty-aware UX (show confidence alongside the answer), and quality filtering in data pipelines.

## How it works under the hood

**Logprobs.** Some APIs return the log-probability of each output token. The average or minimum logprob across the completion is a proxy for model confidence. Low logprob = the model was "surprised" by its own output — a sign of uncertainty. Limitations: logprobs are available on OpenAI and some open-weight models but not on Anthropic's API as of mid-2025.

**Self-consistency.** Sample N completions at temperature > 0. Measure agreement: if 9 of 10 answers say "1847" and 1 says "1887", the majority answer has 90% self-consistency. Higher agreement = higher confidence. Works without logprob access.

**Verbalized confidence.** Ask the model to rate its own confidence ("On a scale of 0–10, how confident are you?"). Cheap and available everywhere, but unreliable — models are systematically overconfident and sycophantic in their self-assessments.

**Calibration.** A model is *calibrated* if when it says it's 80% confident, it's right about 80% of the time. Calibration is measured on held-out data using the Expected Calibration Error (ECE) metric and visualized as a reliability diagram. Calibration doesn't produce a per-request confidence score — it tells you how to interpret the scores you're already generating. This is the operational point: if your scores are uncalibrated, a score of 0.7 might correspond to only 40% actual accuracy in practice, making any threshold you set meaningless without empirical validation on labeled data.

## Concrete example

```python
import anthropic
from collections import Counter

client = anthropic.Anthropic()


def self_consistency_confidence(
    prompt: str,
    n_samples: int = 10,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """
    Sample n_samples completions and return the majority answer with its
    agreement rate as a confidence score.
    """
    responses = []
    for _ in range(n_samples):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        responses.append(response.content[0].text.strip())

    counts = Counter(responses)
    majority_answer, majority_count = counts.most_common(1)[0]
    confidence = majority_count / n_samples

    return majority_answer, confidence


def verbalized_confidence(prompt: str, answer: str) -> float:
    """
    Ask the model to self-rate confidence. Treat as a weak signal only —
    models are systematically overconfident. Use for ranking, not absolute thresholds.
    """
    # User content interpolated here — a crafted prompt can inflate the self-rating.
    # Never use verbalized confidence as the sole routing gate in adversarial contexts.
    rating_prompt = (
        f"You answered the following question:\n\n"
        f"Question: {prompt}\n"
        f"Your answer: {answer}\n\n"
        f"On a scale from 0.0 to 1.0, how confident are you this answer is correct? "
        f"Reply with only a decimal number."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": rating_prompt}],
    )
    try:
        score = float(response.content[0].text.strip())
        return max(0.0, min(1.0, score))  # clamp — out-of-range value bypasses routing thresholds
    except ValueError:
        return 0.5  # default to uncertain on parse failure


def confidence_gated_response(
    prompt: str,
    high_threshold: float = 0.8,
    low_threshold: float = 0.4,
    n_samples: int = 7,
) -> dict:
    """
    Returns the answer along with routing recommendation based on confidence.
    """
    answer, confidence = self_consistency_confidence(prompt, n_samples=n_samples)

    if confidence >= high_threshold:
        routing = "serve"
    elif confidence >= low_threshold:
        routing = "review"   # show to user but flag for spot-check
    else:
        routing = "escalate" # don't serve — route to human or fallback

    return {
        "answer": answer,
        "confidence": confidence,
        "routing": routing,
        "samples": n_samples,
    }


# Calibration measurement (offline, on a labeled validation set)
def expected_calibration_error(
    predictions: list[tuple[float, bool]],  # (confidence, correct)
    n_bins: int = 10,
) -> float:
    """Compute ECE — lower is better calibrated. 0.0 = perfect calibration."""
    bins = [[] for _ in range(n_bins)]
    for confidence, correct in predictions:
        bin_idx = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_idx].append((confidence, correct))

    ece = 0.0
    n_total = len(predictions)
    for bin_data in bins:
        if not bin_data:
            continue
        avg_confidence = sum(c for c, _ in bin_data) / len(bin_data)
        accuracy = sum(1 for _, correct in bin_data if correct) / len(bin_data)
        ece += (len(bin_data) / n_total) * abs(avg_confidence - accuracy)

    return ece
```

For factual QA, self-consistency across 7 samples gives a meaningful confidence signal. An answer that 6 of 7 samples agree on (confidence = 0.86) should be served. An answer only 2 of 7 agree on (confidence = 0.29) should escalate. Note: 7 samples means 7 API calls — factor the N× cost into your routing threshold decisions. Running self-consistency on every request may be unacceptable for latency-sensitive applications; consider reserving it for high-stakes queries or running it offline to build a calibration dataset.

## When to use it / when not to

**Use when:**
- You need to gate actions on reliability (only trigger a booking if confidence ≥ 0.85)
- Building a human-in-the-loop pipeline that should escalate uncertain outputs
- Filtering synthetic data before using it for fine-tuning
- Providing uncertainty-aware UX ("I'm fairly confident, but double-check this")

**Don't rely on it for:**
- Open-ended creative tasks — "confidence" isn't meaningful when there's no ground truth
- Single-sample scenarios where self-consistency would be too expensive (n=10 calls = 10x cost)
- Security decisions — even high-confidence output can be adversarially manipulated

**Self-consistency cost.** 10 samples = 10x the API cost and latency. Cache results for repeated queries. Consider using it offline (batch eval) and training a lightweight confidence classifier instead.

## Main tools and libraries

| Tool | Role |
|------|------|
| OpenAI API `logprobs` | Token-level probabilities — useful for open-weight and GPT models |
| LangChain `SelfConsistencyChain` | Samples multiple completions and votes |
| `litellm` | Unified interface that surfaces logprobs where available |
| `sklearn.calibration` | `calibration_curve()` for reliability diagrams, `CalibratedClassifierCV` for post-hoc calibration |
| `uncertainty-toolbox` (PyPI) | Calibration metrics and plots for regression uncertainty |

## Common failure modes and gotchas

**Logprobs ≠ calibration.** A high average logprob means the model was fluent, not that it was correct. Fluent hallucinations have high logprobs. Calibration on held-out data is the only way to know if a score means anything.

**Self-consistency with one model = correlated errors.** If the model is wrong, it often agrees with itself across samples. Self-consistency underestimates uncertainty on systematic biases. Ensemble across multiple models for more independent error modes.

**Verbalized confidence is sycophantic.** Models rate themselves highly especially after giving a detailed-sounding answer. Treat verbalized confidence as a ranking signal (this answer > that answer) not an absolute threshold.

**Threshold choice without calibration data.** Setting a confidence threshold of 0.7 without knowing what accuracy that corresponds to is a guess. Always compute ECE on a representative labeled set before deploying a threshold in production.

**Self-consistency for long-form is ambiguous.** Agreement on a one-word answer is clear. Agreement on a 500-word essay is not — two essays can be semantically identical but textually different. Use a judge model to check semantic agreement, not string equality.

## Project ideas

- **Calibration dashboard**: run the model on a labeled validation set monthly; plot the reliability diagram and track ECE over time. Alert when ECE degrades (model update changed calibration).
- **Confidence-gated human review queue**: automatically route low-confidence outputs to a Slack channel for spot review. Track what % of escalated items were actually wrong vs. correctly uncertain.
- **Cheap confidence classifier**: generate 1,000 prompts with self-consistency labels, then fine-tune a small classifier (logistic regression on embeddings) that approximates self-consistency at single-sample cost.

## Going deeper

- [Kadavath et al., "Language Models (Mostly) Know What They Know"](https://arxiv.org/abs/2207.05221) — seminal paper on verbalized confidence and calibration in LLMs
- [Xiong et al., "Can LLMs Express Their Uncertainty?"](https://arxiv.org/abs/2306.13063) — comprehensive survey of LLM uncertainty methods
- [Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171) — original self-consistency paper
- [Guo et al., "On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) — temperature scaling for post-hoc calibration
- [[Graceful Degradation]] — using confidence scores to trigger fallback behavior
