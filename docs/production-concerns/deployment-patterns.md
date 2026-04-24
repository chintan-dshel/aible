---
sidebar_position: 7
title: Deployment Patterns
description: Shadow mode, A/B testing, canary, and blue-green deployments adapted for AI systems.
---

# Deployment Patterns

You can't unit test a model or prompt change. The only way to know whether it regresses on real traffic is to run it on real traffic — carefully, with controls, and with a fast rollback path.

## The problem it solves

Model updates, prompt changes, and system prompt rewrites all change behavior in ways that are hard to predict from offline evals alone. A prompt that scores better on your golden dataset might perform worse on the long tail of production queries. Traditional deployment patterns (canary, blue-green, A/B) apply, but need to be adapted for the specific properties of LLM systems: non-deterministic outputs, subjective quality metrics, and prompt-coupled behavior (where the prompt and model output format are entangled with how downstream systems parse the response — change the prompt and you may silently break a parser without any code error).

## How it works under the hood

**Shadow mode.** Run the new model/prompt in parallel with the current production version. Serve the current version's response to the user; log the new version's response to a comparison store. No user impact. Use this to collect comparison data before any live traffic exposure.

**Canary deployment.** Route a small percentage (1–5%) of live traffic to the new version. Monitor quality metrics for that slice. Gradually increase the percentage if metrics hold. Roll back immediately if they degrade.

**A/B testing.** Split traffic into two groups: control (current version) and treatment (new version). Collect outcome metrics for both groups, run a significance test, and make a ship/no-ship decision. Requires a defined outcome metric and enough traffic to reach statistical significance. Note: LLM outputs are non-deterministic, so the same user can get meaningfully different responses within the same variant — this inflates within-group variance and means LLM A/B tests require more traffic than equivalent deterministic software tests to detect the same effect size.

**Blue-green deployment.** Maintain two complete environments (blue = current, green = new). Switch traffic entirely from blue to green in one step. Keep blue running for instant rollback. Higher infrastructure cost than canary but simpler mentally — no partial state.

**Feature flags.** Toggle the new model or prompt at runtime without redeployment. Enables instant rollback, per-user or per-tenant targeting, and gradual rollout by cohort.

## Concrete example

```python
import hashlib
import random
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()


@dataclass
class PromptConfig:
    version: str
    system_prompt: str
    model: str


CURRENT = PromptConfig(
    version="v1.2",
    system_prompt="You are a helpful assistant. Answer clearly and concisely.",
    model="claude-sonnet-4-6",
)

CANDIDATE = PromptConfig(
    version="v1.3",
    system_prompt=(
        "You are a helpful assistant. Answer clearly and concisely. "
        "When you're uncertain, say so rather than guessing."
    ),
    model="claude-sonnet-4-6",
)


def route_traffic(user_id: str, canary_pct: float = 0.05) -> PromptConfig:
    """
    Deterministic routing: same user always gets the same version.
    Uses a hash so routing is stable across requests without storing state.
    """
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = (hash_val % 10000) / 10000  # 0.0000–0.9999
    return CANDIDATE if bucket < canary_pct else CURRENT


def shadow_call(
    prompt: str,
    config: PromptConfig,
) -> tuple[str, dict]:
    response = client.messages.create(
        model=config.model,
        max_tokens=512,
        system=config.system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    metadata = {
        "version": config.version,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
    }
    return text, metadata


def call_with_shadow(prompt: str, user_id: str) -> str:
    """
    Serve the current version to the user, but also run the candidate
    version and log both responses for offline comparison.

    Privacy note: shadow logs contain user prompts, partial responses, and user IDs —
    a secondary data processing activity not covered by the primary request. Before
    enabling in production: apply the same PII scrubbing, retention limits, and access
    controls as your primary data store. Consider hashing user_id before logging.
    """
    # Production response — served to user
    production_text, prod_meta = shadow_call(prompt, CURRENT)

    # Shadow response — logged only, never served
    shadow_text, shadow_meta = shadow_call(prompt, CANDIDATE)

    # In production: write to a comparison store (DB, BigQuery, etc.)
    log_shadow_comparison({
        "prompt": prompt[:200],
        "user_id": user_id,
        "production": {"text": production_text[:500], **prod_meta},
        "shadow": {"text": shadow_text[:500], **shadow_meta},
    })

    return production_text


def log_shadow_comparison(record: dict) -> None:
    # Replace with actual write to your analytics store
    print(f"[SHADOW] {record['production']['version']} vs {record['shadow']['version']}")


def call_with_canary(prompt: str, user_id: str) -> str:
    """Route a fraction of users to the candidate version."""
    config = route_traffic(user_id, canary_pct=0.05)

    response = client.messages.create(
        model=config.model,
        max_tokens=512,
        system=config.system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    # Tag the response with its version for monitoring
    text = response.content[0].text
    log_production_call(user_id, config.version, response)
    return text


def log_production_call(user_id: str, version: str, response) -> None:
    # Use a pseudonymous token instead of raw user_id — HMAC(user_id, server_secret)
    # prevents re-identification if logs are accessed by unauthorized parties.
    print(f"[PRODUCTION] user={user_id} version={version} "
          f"tokens={response.usage.output_tokens}")
```

`call_with_shadow` lets you collect comparison data with zero user impact before enabling a canary. `call_with_canary` uses deterministic hash-based routing so a user always gets the same version — avoiding confusing mixed experiences across a session.

## When to use it / when not to

**Shadow mode first** for any significant change: new model version, system prompt rewrite, new tool added to an agent. Collect 1–3 days of shadow data before enabling any canary.

**Canary at 1–5% initially** for changes that have passed shadow validation. Increase to 10%, 25%, 50%, 100% on 24-hour intervals if metrics hold.

**A/B test when you have a measurable outcome** — user thumbs-up rate, task completion rate, conversation length. Without a real outcome metric, A/B testing just tells you which version sounds different, not which one is better.

**Blue-green for stateless prompt services** where switching traffic is simple. Avoid it when model output format is coupled to downstream processing — flipping from a version that returns JSON to one that returns prose will break the parser before you notice.

**Feature flags** are the operational primitive: canary and A/B are deployment strategies that feature flags enable. Add a flag to every significant model or prompt change from day one.

## Main tools and libraries

| Tool | Role |
|------|------|
| LaunchDarkly / GrowthBook | Feature flags with percentage rollouts and user targeting |
| Langfuse | Prompt versioning + production trace tagging — compare traces by prompt version |
| Statsig | A/B testing with built-in significance testing and guardrail metrics |
| LiteLLM | Router-level model versioning and traffic splitting |
| Braintrust | Golden dataset evals + production sampling to compare versions |

GrowthBook is open-source and self-hostable — good for teams that want A/B testing infrastructure without a SaaS dependency.

## Common failure modes and gotchas

**Shadow mode doesn't catch latency regressions.** The shadow call is off the critical path — it adds no latency from the user's perspective. If the candidate model is 3x slower, you won't see it until you go live. Run a dedicated latency benchmark before promoting from shadow to canary.

**A/B test without a stopping rule.** Running an A/B test until it "looks significant" is p-hacking. Define the minimum detectable effect, required power (80%), and significance level (0.05) before the test starts. Use a stopping rule — don't peek daily.

**Rollback that can't actually roll back.** If the candidate model produces responses in a new format that downstream systems have started consuming, rolling back the model doesn't un-break the data. Keep schema-breaking changes decoupled from model changes.

**User assignment drift.** If you assign users to variants at request time with a random function (not a hash), the same user gets different versions across requests. Deterministic hash-based assignment (or storing the variant) is required for any test where session coherence matters.

**Prompt leak across versions.** During a canary, some users get the old prompt and some get the new. If users share screenshots or outputs, the different behavior may surface before you've reached a go/no-go decision. This is a comms issue, not a technical one — but worth flagging for high-visibility product changes.

## Project ideas

- **Version comparison viewer**: a lightweight internal tool that shows side-by-side shadow comparisons with an LLM judge score for each, plus a thumbs up/down for human review.
- **Automated canary gating**: integrate with your monitoring metrics to auto-pause a canary if quality score drops more than 10% relative to the current version. Resume manually.
- **Prompt change diff log**: track all prompt version changes in a structured log (version, diff, author, deploy date, rollout %) — make rollback a one-click operation to any previous version.

## Going deeper

- [Accelerate: The Science of Lean Software and DevOps](https://itrevolution.com/product/accelerate/) — deployment frequency and DORA metrics; the principles apply to LLM deployments
- [GrowthBook documentation](https://docs.growthbook.io/) — open-source feature flags and A/B testing
- [Langfuse prompt versioning](https://langfuse.com/docs/prompts/get-started) — tracking prompt versions in production
- [[Monitoring]] — the quality metrics that drive canary go/no-go decisions
- [[Evals]] — offline golden dataset evals that precede any shadow/canary rollout
