---
sidebar_position: 4
title: Guardrails
description: Input/output filtering and policy enforcement — how to constrain model behavior without breaking it.
---

# Guardrails

## What it is

Guardrails are the layer that intercepts inputs before they reach the model and outputs before they reach the user, applying rules, classifiers (something that automatically sorts input into categories, like "safe" or "off-topic"), or secondary model calls to enforce policy. They are the operational implementation of "this system should not do X."

The name comes from highway guardrails — they don't prevent cars from driving; they prevent cars from going off the edge. Guardrails don't replace model alignment (which makes the model less likely to produce harmful outputs in the first place) — they add a systematic enforcement layer on top.

## The problem it solves

Even well-aligned models produce undesirable outputs under some conditions. A model fine-tuned (further trained on your own data) for customer support may discuss competitors when asked cleverly. A medical information bot may provide specific dosage guidance it shouldn't. A general-purpose assistant may generate content that violates your terms of service when given the right context.

You cannot enumerate in advance every input that violates your policy. Guardrails provide a systematic, testable, updatable mechanism for enforcing policy without modifying the model itself.

## How it works under the hood

### Input guardrails

Applied to the user's message before it reaches the model. Common patterns:

**Blocklist / regex filter** — fast, deterministic (same input always produces the same output, with no randomness), zero latency (no delay waiting for a response). Catches known bad patterns (profanity, competitor names, SQL injection — a decades-old attack where user input gets run as a literal database command, specific phrases). Brittle against rephrasing.

The function below checks the input text against a list of regex — text-pattern — matches, and rejects it if any pattern hits:

```python
import re

BLOCKED_PATTERNS = [
    r"\b(competitor_name|rival_brand)\b",
    r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"\b(credit\s*card\s*number|cvv|ssn|social\s*security)\b",
]

def input_blocklist(text: str) -> tuple[bool, str]:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Input blocked: matches policy pattern"
    return True, ""
```

**Topic classifier** — a lightweight model that classifies the input against a set of allowed/blocked topics. More flexible than regex, more expensive.

```python
import anthropic

client = anthropic.Anthropic()

ALLOWED_TOPICS = ["product questions", "order status", "billing", "technical support"]

def topic_guard(user_input: str) -> tuple[bool, str]:
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",  # use fast/cheap model for guardrails
        max_tokens=64,
        messages=[{"role": "user", "content": f"""Is this message on one of these topics: {', '.join(ALLOWED_TOPICS)}?
Message: "{user_input}"
Answer with JSON: {{"on_topic": true/false, "topic": "topic or null"}}"""}]
    )
    import json
    return json.loads(result.content[0].text)
```

**Prompt injection detector** — a classifier specifically trained to detect injection attempts — adversarial text trying to override the system's real instructions (see [Prompt Injection](./prompt-injection) for the full taxonomy). LlamaGuard (Meta's purpose-built safety classifier, covered below) and similar models handle this.

### Output guardrails

Applied to the model's response before it reaches the user. Common patterns:

**PII detection** — scan the output for personal information that shouldn't be exposed. Use a PII detection library like `presidio` or a regex pass.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_pii(text: str) -> str:
    results = analyzer.analyze(text=text, language="en")
    if results:
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    return text
```

**Content classifier** — classify the output against a policy (harmful, toxic, off-topic, false claim). Flag or block outputs that fail.

```python
def output_policy_check(output: str, system_context: str) -> tuple[bool, str]:
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": f"""Check if this AI response violates our policy.
Policy: responses must be accurate, helpful, and not make specific medical/legal/financial recommendations.
Response: "{output}"
JSON: {{"passes": true/false, "violation": "description or null"}}"""}]
    )
    import json
    data = json.loads(result.content[0].text)
    return data["passes"], data.get("violation", "")
```

**Hallucination check** — verify that factual claims in the output — which could otherwise be fabricated and stated as if true — are grounded in the provided context (see [Output Validation](./output-validation) for the full approach).

### LlamaGuard

Meta's LlamaGuard is a purpose-built safety classifier fine-tuned on a taxonomy of harmful content categories. It operates as a separate model call that classifies (input, output) pairs:

```python
# Using LlamaGuard via Hugging Face or Together AI
# Input: conversation history as a formatted prompt
# Output: "safe" or "unsafe\n<violated category>"

LLAMAGUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex Crimes",
    "S4": "Child Exploitation",
    "S5": "Specialized Advice (medical/legal/financial)",
    "S6": "Privacy",
    "S7": "Intellectual Property",
    "S8": "Indiscriminate Weapons",
    "S9": "Hate",
    "S10": "Suicide & Self-Harm",
    "S11": "Sexual Content",
    "S12": "Elections",
    "S13": "Code Interpreter Abuse",
}

def llamaguard_check(user_message: str, assistant_response: str = "") -> dict:
    """Call LlamaGuard via Hugging Face Inference API (or swap for Together AI endpoint)."""
    # Format conversation for LlamaGuard's expected input template
    conversation = f"[INST] {user_message} [/INST]"
    if assistant_response:
        conversation += f" {assistant_response}"

    import httpx
    response = httpx.post(
        "https://api-inference.huggingface.co/models/meta-llama/LlamaGuard-7b",
        headers={"Authorization": "Bearer YOUR_HF_TOKEN"},
        json={"inputs": conversation},
        timeout=10,
    )
    result_text = response.json()[0]["generated_text"].strip()
    # Output is "safe" or "unsafe\nS<N>" where S<N> is the violated category
    safe = result_text.startswith("safe")
    category = result_text.split("\n")[1] if not safe and "\n" in result_text else None
    return {
        "safe": safe,
        "category": LLAMAGUARD_CATEGORIES.get(category) if category else None,
        "raw": result_text,
    }
```

LlamaGuard has important limitations to understand before deploying it. It covers the harms in its training taxonomy (S1–S13) reliably on typical inputs, but accuracy varies by category and degrades on adversarially crafted inputs. It also has no knowledge of domain-specific harms — "never reveal competitor pricing," "don't make specific medication dosage claims," or "don't confirm a user's account balance" are not in its taxonomy. If your policy includes harms outside S1–S13, you'll need a custom classifier alongside LlamaGuard. Before deploying it, validate it on 50+ examples from your domain where you know the ground-truth safety verdict — the actual correct answer, checked by a human, that the classifier's output should be compared against.

### Guardrail pipeline

The standard pattern is a chain of guards applied in order, with short-circuit on failure — stop at the first guard that fails, rather than running every remaining check anyway:

```python
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class GuardrailResult:
    passed: bool
    blocked_by: Optional[str] = None
    reason: Optional[str] = None

def run_input_guards(user_input: str) -> GuardrailResult:
    # Apply guards in order of increasing cost
    passed, reason = input_blocklist(user_input)
    if not passed:
        return GuardrailResult(passed=False, blocked_by="blocklist", reason=reason)

    result = topic_guard(user_input)
    if not result["on_topic"]:
        return GuardrailResult(passed=False, blocked_by="topic_classifier", reason="Off-topic input")

    return GuardrailResult(passed=True)

def run_output_guards(output: str, system_context: str) -> GuardrailResult:
    output = scrub_pii(output)  # scrubbing modifies but doesn't block

    passes, violation = output_policy_check(output, system_context)
    if not passes:
        return GuardrailResult(passed=False, blocked_by="policy_check", reason=violation)

    return GuardrailResult(passed=True)
```

## Concrete example

A complete guardrail pipeline for a customer support bot — check the input against an injection blocklist, then a topic classifier, then generate a response, then check its length and run it past a policy check before returning it:

```python
import anthropic
import json
import re

client = anthropic.Anthropic()

SYSTEM = """You are a helpful customer support agent for Acme Corp.
Only answer questions about our products, orders, and billing.
Do not discuss competitors or provide legal/medical/financial advice."""

def customer_support(user_message: str, session_id: str) -> dict:
    # --- Input guards ---
    # 1. Blocklist
    injection_patterns = [
        r"(ignore|forget|disregard).{0,20}(instructions|prompt|system)",
        r"you are now",
        r"act as (a|an|the)",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {"response": "I can only help with Acme Corp support questions.", "blocked": True, "reason": "injection_attempt"}

    # 2. Topic check (fast classifier)
    topic_check = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": f'Is this a customer support question for a software company (yes/no only): "{user_message[:200]}"'}]
    )
    if "no" in topic_check.content[0].text.lower():
        return {"response": "I can only help with Acme Corp support questions.", "blocked": True, "reason": "off_topic"}

    # --- Generate response ---
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_message}]
    )
    output = response.content[0].text

    # --- Output guards ---
    # 3. Length check (simple)
    if len(output) < 10:
        return {"response": "I'm sorry, I couldn't generate a response. Please try again.", "blocked": True, "reason": "empty_response"}

    # 4. Policy check
    policy_result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": f'Does this response make specific medical, legal, or financial recommendations? (yes/no only): "{output[:300]}"'}]
    )
    if "yes" in policy_result.content[0].text.lower():
        return {"response": "I'm not able to provide that kind of advice. Please consult a professional.", "blocked": True, "reason": "policy_violation"}

    return {"response": output, "blocked": False}
```

## When to use it / when not to

#### Guardrails are required when

- Your system handles user-generated input at scale — you cannot review every message
- The domain has specific prohibited outputs (medical advice, competitor names, PII)
- You are deploying in a regulated context (financial services, healthcare, children's products)
- Your system can take consequential actions (send emails, execute code, make purchases)

#### Guardrails alone are insufficient when

- The threat model — the specific set of attackers and attack methods you're actually defending against — includes sophisticated adversarial users; guardrails raise the bar but don't eliminate the risk
- Your policy changes frequently — classifier-based guardrails need retraining; rule-based ones need rewriting
- You're over-guardrailing and blocking legitimate requests — measure your false positive rate

#### The practical question

What is the worst output this system could produce, and how bad would it be? If the answer is "mildly embarrassing," lighter guardrails suffice. If the answer is "medical harm" or "financial fraud," you need defense in depth: aligned model + input classifiers + output classifiers + human review for flagged cases.

:::tip[My take]

Order your guards by cost: put the cheapest (regex blocklist, length check) first and the most expensive (LLM-as-judge policy check — a second model call that grades the output against a rubric) last. A regex check that takes 0ms should run before a model call that takes 300ms. You'll block most bad inputs before they ever reach the expensive check.

The false positive problem is real. Overly aggressive guardrails make your system useless — users get blocked on legitimate questions and lose trust. Track your guardrail trigger rate by category. If a category triggers on > 5% of legitimate-looking inputs, your policy for that category is too broad.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| LlamaGuard (Meta) | Safety classifier for harmful content categories; open weights |
| Anthropic's Constitutional AI / built-in safety | Model-level alignment; handles common harms without extra calls |
| NVIDIA NeMo Guardrails | Framework for programming LLM behavior with dialog flows and guardrails |
| Guardrails AI | Output validation framework; schema + semantic validators |
| `presidio` | PII detection and anonymization |
| Azure Content Safety | Managed content moderation API; text and image |
| OpenAI Moderation API | Fast, cheap moderation endpoint; useful as a first-pass filter |

## Common failure modes and gotchas

**1. Guardrails as security theater.** A regex blocklist that catches "ignore previous instructions" but misses "disregard the above" or "override your directives" provides false confidence without real protection. Adversaries know the common blocklists. Layer with a model-based classifier.

**2. High false positive rate.** Blocking 10% of legitimate requests to prevent 0.1% of bad requests is usually the wrong trade-off. Measure your false positive rate on production traffic. If it's above 1–2%, your guardrail is too aggressive.

**3. Latency addition.** Each guard adds latency. A chain of: input classifier (200ms) + model call (600ms) + output classifier (200ms) = 1,000ms total, where the actual generation was only 600ms. Use fast models (Haiku) for guards and run input and output guards asynchronously — running them at the same time rather than one after another and waiting for each — where possible.

**4. Guardrails on the wrong layer.** Input guardrails that only check the last user message miss injection payloads embedded in retrieved documents (indirect injection). Check retrieved content before injecting it into the prompt.

**5. Not logging guardrail triggers.** If you don't log every guardrail trigger with the input that caused it, you can't: audit who tried to circumvent the system, measure false positive rate, or improve your guards over time.

**6. Treating guardrails as a substitute for alignment.** Guardrails are a safety net, not a primary defense. A highly capable adversary will eventually find a way around any checklist. The model's alignment is the first line of defense; guardrails catch what slips through.

## Project ideas

**1. Guardrail false positive audit** — Deploy a simple topic classifier guardrail on a customer support bot. Generate 100 realistic customer questions (including edge cases: short questions, unusual phrasings, multi-language). Measure the false positive rate (legitimate questions blocked). Tune the classifier threshold until you achieve < 2% false positives.

**2. Injection resistance test** — Build the prompt injection detection guard above. Then craft 20 injection attempts ranging from obvious ("ignore all previous instructions") to subtle (role-playing scenarios, indirect injection via retrieved documents). Measure detection rate. Document which attempts slip through.

**3. Guard latency profiler** — Add timing to each guard in your chain. Run 100 requests. Measure the latency contribution of each guard stage. Is the expensive model-based guard worth the latency on every request? Experiment with running it only on inputs that pass initial heuristics — quick, rule-of-thumb checks that are cheap but imperfect.

**4. Defense-in-depth comparison** — Compare three configurations on the same test set: (a) no guards, (b) input guards only, (c) input + output guards. Measure harmful output rate for each. Quantify how much risk each layer eliminates and at what latency cost. This makes the trade-off concrete.

## Going deeper

#### Foundational reading

- Perez & Ribeiro, "Ignore Previous Prompt: Attack Techniques For Language Models" (2022) — the prompt injection taxonomy that guardrails are designed to address.
- Inan et al., "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations" (Meta, 2023) — the LlamaGuard model card and methodology.
- OWASP Top 10 for LLM Applications (owasp.org) — the most comprehensive taxonomy of LLM security risks and mitigations; the standard reference for AI security engineers.

#### Tools

- NVIDIA NeMo Guardrails documentation (docs.nvidia.com/nemo-guardrails) — the most feature-rich open-source guardrails framework; supports dialog flows, topic filtering, and fact-checking.
- Guardrails AI documentation (docs.guardrailsai.com) — output-focused; strong on schema validation and semantic constraints.
