---
sidebar_position: 6
title: Output Validation
description: Schema validation, semantic checks, and hallucination detection — the layer that catches bad outputs before they reach users.
---

# Output Validation

## What it is

Output validation is the practice of programmatically verifying that a model's response meets quality, correctness, and safety requirements before it reaches the user or downstream system. It sits between the model's raw generation and whatever acts on that output next — a user, an API consumer, a database write, an agent's next step.

Unlike guardrails, which primarily filter for policy violations (harmful content, off-topic responses), output validation is concerned with correctness: is the output well-formed? does it answer the question? are its factual claims grounded in the provided context? does it match the expected schema? Guardrails enforce *what should not happen*; output validation enforces *what should happen*. A response can pass guardrails but fail validation (accurate, on-topic, but missing a required field), and vice versa.

## The problem it solves

Models produce probabilistic outputs. On most inputs, those outputs are correct and useful. On some inputs — unusual phrasing, missing context, edge cases the model generalizes poorly — the output is wrong, malformed, or contradicts the source material. Without validation, those failures reach users silently.

The failure modes vary by task:
- A structured extraction pipeline produces a JSON object with a missing required field, crashing the downstream parser.
- A RAG-based Q&A system confidently answers a question with a fact not in the retrieved documents — hallucinating an answer.
- A summarization system generates a summary that contradicts the source document on a key detail.
- A code generation system produces syntactically valid code that has a logic error the tests don't catch.

Each of these can be caught by a validation layer if you design it thoughtfully.

## How it works under the hood

### Layer 1: Schema validation

The simplest and cheapest validation: does the output conform to the expected structure? For structured outputs (JSON, Pydantic models), this is straightforward. For prose, it means checking length, format markers, or required sections.

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional
import json

class ExtractionResult(BaseModel):
    entity_name: str
    category: Literal["person", "organization", "location", "product"]
    confidence: float = Field(ge=0.0, le=1.0)
    context_quote: Optional[str] = None

def validate_schema(raw_output: str) -> tuple[ExtractionResult | None, str]:
    try:
        data = json.loads(raw_output)
        result = ExtractionResult(**data)
        return result, ""
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except ValidationError as e:
        return None, f"Schema violation: {e}"
```

With `instructor`, this validation + retry loop is handled automatically:

```python
import anthropic
import instructor
from pydantic import BaseModel

client = instructor.from_anthropic(anthropic.Anthropic())

result = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    response_model=ExtractionResult,
    messages=[{"role": "user", "content": "Extract entities from: 'Apple Inc. released the iPhone.'"}]
)
# result is a validated ExtractionResult — schema enforcement + retry included
```

### Layer 2: Semantic validation (LLM-as-judge)

Schema validation catches structural problems. Semantic validation asks whether the content is correct — a question that often requires another model call.

```python
import anthropic
import json

client = anthropic.Anthropic()

def check_faithfulness(question: str, context: str, answer: str) -> dict:
    prompt = f"""Check whether this AI answer is faithful to the context — i.e., whether all factual claims in the answer are supported by the context.

Context: {context}

Question: {question}

Answer: {answer}

JSON response: {{"faithful": true/false, "unsupported_claims": ["list of claims not in context"], "reasoning": "one sentence"}}"""

    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(result.content[0].text)

def check_relevance(question: str, answer: str) -> dict:
    prompt = f"""Does this answer address the question asked? Score 1-5 (5 = directly answers, 1 = completely off-topic).

Question: {question}
Answer: {answer}

JSON: {{"relevance_score": 1-5, "reasoning": "one sentence"}}"""

    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(result.content[0].text)
```

### Layer 3: Hallucination detection

Hallucination detection is a specialization of faithfulness checking: does the answer make factual claims that aren't in the provided context? This is the critical validation for RAG systems.

```python
def detect_hallucination(context: str, answer: str) -> dict:
    prompt = f"""You are checking whether an AI answer introduces facts not present in the provided context.

Context (the only source of truth):
{context}

AI Answer:
{answer}

For each factual claim in the answer:
1. Is it directly stated or clearly implied by the context?
2. Does it logically follow from what the context states? (not just something that could be true)
3. Is it not in the context (potential hallucination)?

Examples of the distinction:
- Faithful inference: context says "ships in 5 business days", answer says "delivery takes about a week" — reasonable rounding.
- Hallucination: context says "ships in 5 business days", answer says "ships overnight" — contradicts context.

JSON: {{
  "hallucination_detected": true/false,
  "confidence": "high/medium/low",
  "unsupported_claims": ["list"],
  "verdict": "one sentence summary"
}}"""

    result = client.messages.create(
        model="claude-sonnet-4-6",  # use stronger model for hallucination detection
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    return json.loads(result.content[0].text)
```

### Layer 4: Consistency checks

For systems that generate multiple related outputs (multi-step reasoning, multi-document synthesis), check that the outputs are internally consistent:

```python
def check_numerical_consistency(output: str) -> dict:
    """Extract numbers and verify they're internally consistent (e.g., line items sum to total)."""
    import re
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", output)
    # Domain-specific: check that subtotals + tax = total, quantities × price = line total, etc.
    # This is application-specific logic that knows the expected relationships
    return {"numbers_found": numbers, "consistency_check": "manual verification required"}
```

### Validation pipeline

Chain validators in order of increasing cost, short-circuit on failure:

```python
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class ValidationResult:
    passed: bool
    stage: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[dict] = None

def validate_rag_output(
    question: str,
    context: str,
    answer: str,
    min_length: int = 20,
    min_relevance_score: int = 3,
) -> ValidationResult:
    # Stage 1: length check (free)
    if len(answer.strip()) < min_length:
        return ValidationResult(passed=False, stage="length", reason="Answer too short")

    # Stage 2: relevance check (cheap model call)
    relevance = check_relevance(question, answer)
    if relevance["relevance_score"] < min_relevance_score:
        return ValidationResult(
            passed=False,
            stage="relevance",
            reason=f"Low relevance: {relevance['reasoning']}",
            details=relevance
        )

    # Stage 3: faithfulness check (cheap model call)
    faithfulness = check_faithfulness(question, context, answer)
    if not faithfulness["faithful"]:
        return ValidationResult(
            passed=False,
            stage="faithfulness",
            reason=f"Answer not grounded in context",
            details=faithfulness
        )

    return ValidationResult(passed=True)
```

### When to retry vs. block

When validation fails, you have three options:

**Retry** — re-run the generation with the validation failure in context. Works for schema violations and some semantic failures. Use with a retry limit (2–3 attempts maximum).

```python
import anthropic

client = anthropic.Anthropic()

def generate_with_validation(prompt: str, context: str, max_retries: int = 2) -> str:
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=messages
        )
        answer = response.content[0].text
        validation = check_faithfulness(prompt, context, answer)

        if validation["faithful"]:
            return answer

        if attempt < max_retries:
            # Add the failure as context for retry
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": f"Your answer contained unsupported claims: {validation['unsupported_claims']}. Please answer again using only the provided context."})

    # All retries failed — return a safe fallback
    return "I don't have enough information in the provided context to answer this question accurately."
```

**Fallback** — return a safe default response when validation fails after retries. Appropriate when the model consistently fails on a class of inputs.

**Block + log** — reject the output and log the failure for review. Used when neither retry nor fallback is acceptable (e.g., the output is potentially harmful or the downstream system can't handle a fallback).

## Concrete example

A complete validation pipeline for a RAG-based document Q&A system:

```python
import anthropic
import json
import re

client = anthropic.Anthropic()

def answer_question(question: str, context_docs: list[str]) -> dict:
    context = "\n\n".join(context_docs)

    # Generate answer
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context. If the answer isn't there, say so."
        }]
    )
    answer = response.content[0].text

    # Validate length
    if len(answer.strip()) < 20:
        return {"answer": None, "blocked": True, "reason": "empty_response"}

    # Validate faithfulness
    try:
        faithfulness_check = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": f"""Is every factual claim in this answer supported by the context? (yes/no only)

Context: {context[:1000]}
Answer: {answer}"""}]
        )
        faithfulness_response = faithfulness_check.content[0].text.lower()
    except Exception:
        # If validation call fails, treat as inconclusive — log in production
        faithfulness_response = "yes"

    if "no" in faithfulness_response:
        # Retry once with explicit instruction
        retry_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context. If the answer isn't there, say so."},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Your answer may have gone beyond the context. Please answer again, sticking strictly to what the context states."}
            ]
        )
        answer = retry_response.content[0].text

        # Check retry
        try:
            retry_check = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=32,
                messages=[{"role": "user", "content": f"Is this answer supported by the context? (yes/no) Context: {context[:500]} Answer: {answer}"}]
            )
            retry_response_text = retry_check.content[0].text.lower()
        except Exception:
            retry_response_text = "yes"

        if "no" in retry_response_text:
            return {"answer": "I cannot find a reliable answer to this question in the available documents.", "blocked": False, "fallback": True}

    return {"answer": answer, "blocked": False, "fallback": False}
```

## When to use it / when not to

#### Validate when

- Your system makes factual claims that could be wrong (RAG, summarization, information extraction)
- Downstream code parses the output (structured extraction, JSON responses)
- The output feeds into another model or system step — errors compound
- The cost of a wrong answer is high (medical information, financial data, legal documents)
- You operate in a regulated domain where outputs must be auditable

#### Skip or simplify when

- The output is creative and correctness is subjective (drafting, brainstorming)
- The user will review and act on the output directly — they are the validator
- Latency is extremely constrained and validation adds unacceptable overhead
- The model already produces high-accuracy outputs on your input distribution (measure first)

#### The practical question

What happens when the model produces a wrong output? If the answer is "a user reads something incorrect," validation is helpful. If the answer is "a database gets corrupted, a transaction executes, or downstream code crashes," validation is mandatory.

:::tip[My take]

Faithfulness checking against retrieved context is the single most impactful validation you can add to a RAG system. The most common real-world failure isn't harmful content — it's a model confidently answering from its parametric knowledge when the retrieved context doesn't cover the question. A quick LLM-as-judge faithfulness check catches this class of error at low cost.

The retry loop is your first line of response to validation failure, but have a ceiling. If a model fails faithfulness validation three times on the same question, the answer is probably "the context doesn't contain what's needed" — fall back to saying so explicitly rather than infinitely retrying. Infinite retries mask systematic gaps in your retrieval pipeline.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| `instructor` | Pydantic schema validation + automatic retry; works with Anthropic, OpenAI, Gemini |
| RAGAS | RAG-specific metrics: faithfulness, answer relevance, context precision/recall |
| DeepEval | Unit-test-style semantic validators; faithfulness, hallucination, bias metrics |
| Guardrails AI | Output validation framework; schema + semantic validators with retry orchestration |
| `pydantic` v2 | Schema definition and validation — the foundation for structured output validation |
| `pytest` | Custom validators run as unit tests in CI pipelines |

## Common failure modes and gotchas

**1. Validating schema but not semantics.** A response that passes Pydantic validation can still be semantically wrong — the right structure filled with wrong values. Schema and semantic validation are complementary, not substitutes.

**2. Circular validation.** Using the same model to generate and validate its own output. The model that hallucinates an answer is unlikely to correctly identify its own hallucination — and using a *weaker* model (e.g., Haiku validating a Sonnet-generated answer) makes this worse, not better: a smaller model is more likely to miss subtle hallucinations than the model that generated them. Use a model at least as capable as the generator for validation, with an explicitly skeptical prompt ("Find any claim not directly supported by the context"). If you use the same model, at minimum change the prompt configuration to make it adversarial toward the output it's checking.

**3. Validator overconfidence.** LLM-as-judge validators have their own failure modes: they miss subtle hallucinations, have biases toward confident-sounding answers, and can be fooled by plausible-but-wrong outputs. Calibrate your validator against human judgments before trusting it at scale.

**4. Retry amplification of systematic errors.** If a class of inputs consistently fails validation, retrying doesn't help — it just costs more. Log your validation failures by input type and treat persistent failures as a signal that retrieval or generation needs fixing.

**5. Not validating tool outputs before acting on them.** In agentic systems, the model's output often becomes an action (function call, database write, API call). Validate the intended action before executing it — not just after the model generates it.

**6. Latency from synchronous validation.** A validation model call adds 200–400ms per generation. For latency-sensitive applications, run validation asynchronously and surface failures after-the-fact (for logging and retraining) rather than blocking the response. Reserve synchronous validation for high-stakes outputs.

**7. Using validation output as training signal without care.** Outputs that "passed" validation are not necessarily correct — they passed a probabilistic classifier. Treat validation passes as "probably okay" rather than ground truth for fine-tuning.

## Project ideas

**1. Faithfulness audit on a real RAG corpus** — Build a RAG pipeline over a document corpus. Run 50 questions through it. For each answer, run the faithfulness check and manually verify whether the validator is correct. Measure: (a) what fraction of answers are unfaithful? (b) what is the validator's precision and recall against your manual judgments? This gives you a calibrated sense of how much to trust automated faithfulness checking.

**2. Retry effectiveness study** — Run the generate-validate-retry loop on 100 inputs that initially fail faithfulness validation. Measure: what fraction pass after one retry? After two? After three? Plot the diminishing returns curve. Use this to set your retry ceiling.

**3. Schema validation stress test** — Take a structured extraction task. Generate outputs on 100 inputs, 20 of which are edge cases (empty inputs, inputs with missing fields, inputs in different languages). Measure schema validation failure rates. Then test with instructor's retry loop — how many of the failures does it recover?

**4. Validator comparison** — Build a test set of 30 (answer, context) pairs where you know ground-truth faithfulness (15 faithful, 15 unfaithful). Compare three validators: (a) regex/heuristic check, (b) Haiku-based LLM judge, (c) Sonnet-based LLM judge. Measure F1 score for each. The cost-accuracy trade-off is usually more nuanced than expected.

## Going deeper

#### Foundational reading

- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (EACL 2024) — the RAGAS framework; defines faithfulness, answer relevance, context precision, and context recall as measurable metrics.
- Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models" (2023) — sampling-based hallucination detection; useful when you have no ground-truth context.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) — the foundational study of LLM-as-judge reliability and known biases.

#### Libraries

- `instructor` (GitHub: `jxnl/instructor`) — the most practical starting point for schema validation + retry.
- RAGAS documentation (docs.ragas.io) — RAG-specific metrics with good worked examples for faithfulness and relevance.
- DeepEval documentation (docs.confident-ai.com) — strongest for unit-test-style semantic validators in CI.
