---
sidebar_position: 2
title: Evals
description: How to measure whether your AI system actually works — LLM-as-judge, unit evals, human eval pipelines, and the traps.
---

# Evals

## What it is

An eval (evaluation) is a test that measures whether an AI system produces outputs you'd consider correct, useful, or safe — on a defined set of inputs. It is the AI equivalent of a unit test suite, with two important differences: the outputs are often non-deterministic, and "correct" often requires judgment rather than exact matching.

Without evals, you are flying blind. You don't know if a prompt change helped or hurt, whether a model upgrade is actually better for your task, whether a fine-tuned model degraded on cases you didn't train on, or whether your system's accuracy has drifted over the past month.

## The problem it solves

The evaluation problem in AI is harder than in traditional software for three reasons:

**Non-determinism.** The same input can produce different outputs on different runs (due to sampling temperature). You can't assert `output == expected_string`.

**Judgment-dependent correctness.** Whether "The cancellation fee is $50" is a correct answer depends on your documents, your domain, and what "correct" means to you. There is often no ground truth string to compare against.

**Distribution shift.** Your system may work well on the examples you tested it on and fail on the 5% of real inputs that look slightly different. Small test sets miss this.

Evals provide a principled way to measure system quality, track it over time, and make changes confidently.

## How it works under the hood

### Eval types

**Exact match / regex match** — the output must match a specific string or pattern. Use for tasks with deterministic answers: structured extraction, factual lookup, code generation with known expected output. Fast, cheap, no LLM required.

```python
def eval_exact(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()

def eval_contains(output: str, required_substring: str) -> bool:
    return required_substring.lower() in output.lower()
```

**Model-based scoring (LLM-as-judge)** — a second LLM evaluates the first LLM's output against a rubric. The judge model sees the input, the output, and (optionally) a reference answer, and returns a score or verdict.

```python
import anthropic

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response on a scale of 1-5.

Criteria:
- Accuracy: Does it correctly answer the question based on the provided context?
- Conciseness: Is it appropriately brief without omitting important information?
- Grounding: Does it rely only on the provided context, not outside knowledge?

Input: {input}
Context: {context}
AI Response: {response}
Reference answer (if available): {reference}

Respond with JSON: {{"score": 1-5, "reasoning": "one sentence"}}"""

def llm_judge(input_text: str, context: str, response: str, reference: str = "") -> dict:
    prompt = JUDGE_PROMPT.format(
        input=input_text, context=context,
        response=response, reference=reference or "Not provided"
    )
    # temperature=0 makes scoring deterministic: the same input always produces the
    # same score. At temperature > 0, identical outputs can receive different scores
    # across runs, making eval comparisons noisy.
    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    return json.loads(result.content[0].text)
```

**Human eval** — humans score outputs on a rubric or do pairwise comparisons (A vs B). The gold standard for quality but slow and expensive. Use for calibrating your automated evals — if your LLM judge agrees with humans 80%+ of the time, it's likely trustworthy for automation.

**Pairwise comparison** — instead of scoring outputs on an absolute scale, compare two outputs (e.g., old system vs new system) and ask which is better. Pairwise is more reliable than absolute scoring because it's easier for judges to say "A is better than B" than to assign both a number consistently.

### Eval dimensions

For most AI systems, four dimensions cover most of what matters:

| Dimension | Question | Eval method |
|---|---|---|
| **Correctness** | Is the answer right? | Exact match, LLM-as-judge with reference |
| **Faithfulness** | Does the answer stay within the provided context? | LLM-as-judge, RAGAS |
| **Relevance** | Does the answer address the question asked? | LLM-as-judge |
| **Safety** | Does the answer avoid harmful content? | Classifier, LlamaGuard, human review |

### Building an eval set

An eval set is a collection of (input, expected output or rubric) pairs. Quality matters more than quantity — 50 well-chosen examples with clear expected behavior is more useful than 500 edge cases with unclear ground truth.

**Sources:**
- Sample real user inputs from production logs (anonymized)
- Manually author examples for known-hard cases (edge cases, adversarial inputs)
- Have domain experts create gold-standard (input, reference answer) pairs
- Use existing benchmarks if your task overlaps a public dataset (e.g., TriviaQA for QA, HumanEval for code)

**Coverage:** ensure your eval set covers the range of input types you expect in production. A customer support eval set should include billing questions, technical questions, account questions, and edge cases — not just the easy, well-formed examples.

### Running evals at scale

```python
import json
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    id: str
    input: str
    context: str
    reference: str

@dataclass  
class EvalResult:
    case_id: str
    output: str
    score: int
    reasoning: str

def run_eval(
    cases: list[EvalCase],
    system_under_test: Callable[[str, str], str],
    judge: Callable[..., dict]
) -> list[EvalResult]:
    results = []
    for case in cases:
        output = system_under_test(case.input, case.context)
        judgment = judge(case.input, case.context, output, case.reference)
        results.append(EvalResult(
            case_id=case.id,
            output=output,
            score=judgment["score"],
            reasoning=judgment["reasoning"]
        ))
    return results

def summarize(results: list[EvalResult]) -> dict:
    scores = [r.score for r in results]
    return {
        "mean_score": sum(scores) / len(scores),
        "pass_rate": sum(1 for s in scores if s >= 4) / len(scores),
        "failures": [r for r in results if r.score <= 2]
    }
```

## Concrete example

A complete eval pipeline for a RAG-based customer support system:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Sample eval set
EVAL_SET = [
    {
        "id": "cancellation-fee",
        "input": "What is the cancellation fee?",
        "context": "Our cancellation policy: fees apply after 24 hours. Standard: $25. Priority: $50.",
        "reference": "The cancellation fee is $25 for standard and $50 for priority orders.",
        "expected_topics": ["$25", "$50"]
    },
    {
        "id": "no-answer",
        "input": "What is your return policy for digital goods?",
        "context": "Our cancellation policy: fees apply after 24 hours. Standard: $25. Priority: $50.",
        "reference": "The provided context does not contain information about digital goods returns.",
        "expected_topics": ["not", "context"]  # should say info not in context
    },
]

def rag_system(question: str, context: str) -> str:
    """Stub: replace with your actual RAG pipeline."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer using only the context. If not found, say so."}]
    )
    return response.content[0].text

def judge_response(question: str, context: str, answer: str, reference: str) -> dict:
    prompt = f"""Score this AI response 1-5 on accuracy and faithfulness.

Question: {question}
Context: {context}
Answer: {answer}
Reference: {reference}

JSON response: {{"score": 1-5, "faithful": true/false, "reasoning": "one sentence"}}"""
    
    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(result.content[0].text)

# Run evals
scores = []
for case in EVAL_SET:
    answer = rag_system(case["input"], case["context"])
    judgment = judge_response(case["input"], case["context"], answer, case["reference"])
    scores.append(judgment["score"])
    print(f"[{case['id']}] Score: {judgment['score']} — {judgment['reasoning']}")

print(f"\nMean score: {sum(scores)/len(scores):.1f}/5")
```

## When to use it / when not to

#### Always run evals

Every AI system that goes to production needs evals. There is no exception. The question is not whether to eval but what to eval and how rigorously.

**Minimum bar:** 20–50 handcrafted examples, scored by LLM-as-judge, run before every prompt change or model upgrade. This takes a few hours to set up and catches most regressions.

**Production bar:** 200+ examples, human-validated judge calibration, automated runs in CI, dashboards tracking scores over time.

#### When LLM-as-judge is sufficient vs. when you need humans

LLM-as-judge works when: the task has a reasonably clear right/wrong distinction, the judge model is at least as capable as the system under test, and you've validated judge-human agreement on a sample.

Require human eval when: the domain requires expert knowledge (legal, medical), the stakes are high (safety-critical), or the task is inherently subjective (creative quality, tone matching a specific brand voice).

:::tip[My take]

The most common eval mistake is building the eval set from the same distribution as your training or prompting data — you end up measuring how well the system handles cases it was designed for, not how well it handles real production traffic. The second most common mistake is using your eval set as a feedback signal for prompt iteration without holding out a test set. Once you've used examples to tune your system, they can't tell you how the system performs on unseen inputs.

Run your evals before you ship a change, not after. The purpose of evals is to catch regressions before they hit users. If you run them after, they're a post-mortem, not a safety net.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| RAGAS | RAG-specific evals: faithfulness, answer relevance, context precision/recall |
| DeepEval | Unit-test-style eval framework; LLM-as-judge metrics; CI integration |
| LangSmith | Eval datasets, LLM-as-judge, prompt versioning, production traces |
| Weights & Biases Weave | Experiment tracking, eval runs, model comparison |
| Braintrust | Eval platform; datasets, scoring, prompt comparison |
| Anthropic's `eval` framework | Internal; for Claude-specific evaluations |
| `pytest` + custom scorers | Lightweight: run evals as unit tests; no external dependencies |

## Common failure modes and gotchas

**1. Eval set memorization.** You use the eval set to iterate on your prompt, then report the eval score as validation accuracy. This is overfitting: the eval set is now training data. Always hold out a test set that you never look at during development.

**2. LLM judge bias.** The judge model has its own biases — it favors verbosity, its own style, and confident-sounding answers. This can cause it to score incorrect-but-confident outputs higher than correct-but-hedged ones. Calibrate your judge against human scores on 50+ examples before trusting it.

**3. Narrow coverage.** An eval set of 30 clean examples misses the 5% of production inputs that break your system. Add adversarial cases, long inputs, multilingual inputs, and examples with missing context before claiming your system is reliable.

**4. Correct average, broken tail.** A mean score of 4.2/5 looks great until you notice the 8% of examples scoring 1/5. AI systems often degrade catastrophically on specific input patterns. Always review your lowest-scoring examples — they tell you more than the mean.

**5. Eval-metric gaming.** After enough iterations, you optimize for your eval metric rather than actual quality. The metric and quality diverge. Rotate your judge model, add new eval cases regularly, and occasionally do human spot-checks even when automated scores look good.

**6. Ignoring latency and cost.** An eval that only scores quality misses the operational picture. Include latency (P50, P95) and token cost per call in your eval dashboard — quality improvements that triple cost or double latency may not be acceptable.

## Project ideas

**1. Eval set construction** — Take a production AI system (or a simple RAG pipeline). Collect 50 real user inputs (or simulate them). Have a domain expert write reference answers for 40 of them. Use 30 for development (tuning prompts, testing changes) and hold out 20 as a test set you never touch during development. This forces the discipline of a real eval workflow.

**2. Judge calibration study** — Create 30 AI response pairs where you have a human judgment of which is better. Run LLM-as-judge on all 30. Measure agreement rate between the judge and human. Vary: the judge model, the scoring prompt, absolute scoring (1-5) vs. pairwise comparison. Document which configuration produces the highest human agreement.

**3. Regression detection** — Build a simple eval runner that computes a score on your eval set. Make a deliberate prompt change that you expect to improve things (and one you expect to degrade things). Run the eval before and after. Report whether the eval correctly detects both the improvement and the regression.

**4. RAGAS pipeline** — Build a RAG system. Run the RAGAS evaluation suite on it (faithfulness, answer relevance, context precision, context recall). Fix the lowest-scoring metric. Re-run RAGAS and verify improvement. This gives hands-on experience with the most common RAG eval framework.

## Going deeper

#### Foundational reading

- Perez et al., "Red-Teaming Language Models with Language Models" (2022) — using models to generate adversarial eval cases.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) — the foundational paper on LLM-as-judge; covers calibration and known failure modes.
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (EACL 2024) — the RAGAS framework paper; specific metrics for RAG evaluation.

#### Tools and platforms

- DeepEval documentation (docs.confident-ai.com) — the most practical guide to setting up LLM evals in CI.
- LangSmith documentation — strongest for teams already using LangChain; tight integration for prompt versioning + eval.
- Braintrust (braintrustdata.com) — newer platform; clean UI for comparing prompt versions against eval datasets.
