---
sidebar_position: 14
title: Model Distillation
description: Knowledge distillation and teacher-student training — how to get a small model to behave like a big one.
---

# Model Distillation

## What it is

Model distillation is a training technique where a smaller "student" model is trained to mimic the behavior of a larger "teacher" model. The student learns not just from labeled examples (as in standard supervised training) but from the teacher's full output distribution — the probabilities the teacher assigns to each possible answer, not just which answer it chose.

The insight is that the teacher's soft probability distribution contains more information than a hard label. When a teacher assigns 70% probability to "Paris," 15% to "London," and 5% to "Berlin," it's implicitly encoding that "London" is the second-most-plausible city for this question type — information a binary correct/wrong label doesn't capture. The student learns from this richer signal.

## The problem it solves

Large, capable models are slow and expensive to run at scale. A 70B parameter model produces better outputs than a 7B model on most tasks, but costs ~10× more to run and has ~10× higher latency. For high-volume, latency-sensitive applications, this is often a non-starter.

Distillation offers a path to smaller, faster models that retain most of the teacher's quality on a specific task — without the compute requirements of the teacher. The tradeoff: distilled models perform well on the task distribution they were trained on and degrade faster on out-of-distribution inputs than general-purpose large models.

## How it works under the hood

### Core distillation objective

Standard supervised training minimizes cross-entropy loss against hard labels (the correct answer). Distillation adds a second term: the Kullback-Leibler divergence between the teacher's output distribution and the student's:

```
Loss = α × CrossEntropy(student_output, hard_label)
     + (1 - α) × KL_divergence(teacher_soft_logits, student_soft_logits)
```

The temperature parameter T controls how "soft" the teacher's distribution appears. Higher temperature makes the distribution flatter (more uncertainty), which exaggerates the relative probabilities of non-top candidates and provides more signal to the student:

```python
import torch
import torch.nn.functional as F

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    # Soft targets: teacher distribution at temperature T
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss (scaled by T^2 to compensate for temperature scaling)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    # Hard label cross-entropy loss
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * hard_loss + (1 - alpha) * distill_loss
```

### Data for distillation

You need inputs for which the teacher generates outputs. Three approaches:

:::caution[ToS check required]

Anthropic's Terms of Service prohibit using Claude API outputs to train or fine-tune competing models. The black-box distillation approach below generates Claude outputs for fine-tuning an open model (Llama) — this may violate your provider agreement. Before implementing any API-based distillation pipeline, read your provider's terms carefully. If you need to distill from Claude for commercial use, contact Anthropic. As an alternative, distill from an open teacher model (Llama 3 70B, Mixtral) to avoid these restrictions.

:::

**Labeled dataset**: use an existing labeled dataset. Teacher generates logits, student trains on both hard labels and teacher soft targets.

**Unlabeled data (self-labeling)**: the teacher generates responses to unlabeled inputs. The student trains on (input, teacher_response) pairs using the teacher's generated text as the "gold" label. Simpler to implement; doesn't require true labels.

**Synthetic data**: generate inputs using another LLM, get teacher responses, train student on (synthetic_input, teacher_response). Bootstraps a dataset for specialized tasks where labeled data is scarce.

### Distillation via API (black-box distillation)

When you don't have access to teacher logits (e.g., distilling from a closed API model like Claude), you can only use the generated text — not the full probability distribution. This is "black-box" or "response distillation":

```python
import anthropic
from datasets import Dataset
import json

client = anthropic.Anthropic()

def generate_teacher_responses(inputs: list[str], system: str = "") -> list[dict]:
    """Generate teacher outputs for a list of inputs."""
    dataset = []
    for user_input in inputs:
        response = client.messages.create(
            model="claude-sonnet-4-6",  # teacher: large capable model
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_input}]
        )
        dataset.append({
            "input": user_input,
            "output": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })
    return dataset

# Generate teacher outputs for a customer support task
system_prompt = "You are a helpful customer support agent for Acme Corp. Answer only product questions."
training_inputs = [
    "How do I reset my password?",
    "What is your return policy?",
    "How long does shipping take?",
    "Can I change my order after placing it?",
]

teacher_data = generate_teacher_responses(training_inputs, system=system_prompt)

# Save for fine-tuning
with open("distillation_data.jsonl", "w") as f:
    for item in teacher_data:
        f.write(json.dumps({"messages": [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]}) + "\n")

print(f"Generated {len(teacher_data)} training examples")
print(f"Total tokens: {sum(d['input_tokens'] + d['output_tokens'] for d in teacher_data)}")
```

This dataset can then be used to fine-tune a smaller open model (Llama 3 8B, Mistral 7B) to behave like the larger teacher on this task distribution.

### Fine-tuning the student

With a dataset of (input, teacher_response) pairs, fine-tune the student using standard supervised fine-tuning:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

# Load small model (student)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# LoRA config for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Load distillation dataset
dataset = load_dataset("json", data_files="distillation_data.jsonl", split="train")

training_args = TrainingArguments(
    output_dir="./distilled-support-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./distilled-support-model")
```

## Concrete example

A complete black-box distillation pipeline for a product classification task:

```python
import anthropic
import json
from pathlib import Path

client = anthropic.Anthropic()

TEACHER_SYSTEM = """Classify this customer support message into exactly one category.

Categories:
- billing: payment, invoice, charge, subscription, refund
- technical: bug, error, not working, how to, installation
- account: login, password, profile, settings, access
- shipping: delivery, tracking, address, order status
- other: anything that doesn't fit above

Respond with JSON only: {"category": "...", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""

def get_teacher_classifications(messages: list[str]) -> list[dict]:
    results = []
    for msg in messages:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            system=TEACHER_SYSTEM,
            messages=[{"role": "user", "content": msg}]
        )
        try:
            result = json.loads(response.content[0].text)
            results.append({"input": msg, **result})
        except json.JSONDecodeError:
            results.append({"input": msg, "category": "other", "confidence": 0.5, "reasoning": "parse error"})
    return results

# Sample messages for distillation dataset
support_messages = [
    "My payment failed but I was still charged",
    "I can't log in to my account",
    "The app keeps crashing on startup",
    "Where is my order? It's been 2 weeks",
    "How do I cancel my subscription?",
    "I need to update my shipping address",
]

teacher_outputs = get_teacher_classifications(support_messages)

# Write training data for fine-tuning a smaller classifier
for item in teacher_outputs:
    print(f"Category: {item['category']} (conf: {item['confidence']:.2f}) | {item['input'][:50]}")

# The teacher's outputs + high-confidence threshold becomes your labeled dataset
high_confidence = [t for t in teacher_outputs if t["confidence"] >= 0.8]
print(f"\n{len(high_confidence)}/{len(teacher_outputs)} examples with confidence >= 0.8")
```

## When to use it / when not to

#### Distillation is the right approach when

- You have a high-volume, latency-sensitive task where a large model is too slow or too expensive
- The task distribution is well-defined and bounded — the model will mostly see inputs similar to the training examples
- You can generate enough teacher responses (500–10K examples) to capture the task's input distribution
- A smaller model's reduced capability on edge cases is an acceptable tradeoff

#### Prefer the raw large model when

- The task is diverse and open-ended — the student's performance on OOD inputs will degrade
- You need the teacher's full reasoning capability (long-horizon reasoning, creative writing, complex analysis)
- Your volume is low and cost/latency isn't a binding constraint
- Data collection for distillation is expensive relative to just running the large model

#### The practical question

What fraction of your production queries fall within the task distribution the student was trained on? If > 90%, distillation is viable. If 30% of queries are OOD, you'll see significant quality degradation in that fraction.

:::tip[My take]

Black-box distillation (generate teacher outputs, fine-tune student on them) is underused. If you have a well-defined task — classification, extraction, a specific format of response — you can often get a 7–13B student model within 10–15% of the teacher's quality at 10× lower inference cost, by generating 2,000–5,000 teacher examples.

The quality filter matters. Don't use all teacher outputs blindly. Filter out examples where the teacher's confidence is low or where the output doesn't parse correctly. A smaller, high-quality dataset trains a better student than a large noisy one.

The OOD degradation problem is real and is the main reason distillation fails in practice. Teams distill for a specific task, deploy, and then users ask something slightly different. The student model fails badly on the 10% of queries that are outside its training distribution. Add a routing layer: if a query looks OOD (by classifier or embedding similarity to training data), send it to the teacher.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| `trl` (Hugging Face) | SFT and distillation training; SFTTrainer handles dataset formatting |
| `peft` | LoRA adapters for efficient fine-tuning of the student |
| `transformers` | Model loading, tokenization, and training infrastructure |
| Anthropic API | Generating teacher responses for black-box distillation datasets |
| Axolotl | Fine-tuning framework with good defaults for instruction-tuning |
| LLM Foundry (MosaicML) | Large-scale distillation training with efficient data pipelines |

## Common failure modes and gotchas

**1. OOD degradation.** The student performs well on the training distribution and poorly on edge cases. Add OOD detection and route unusual queries to the teacher.

**2. Low-quality teacher outputs in training data.** If the teacher makes mistakes, the student learns those mistakes. Filter teacher outputs by confidence score or run a separate quality check before including examples in training data.

**3. Task distribution mismatch.** Generating training data from a clean, well-formed input set while production traffic includes typos, mixed languages, and incomplete sentences trains a student that fails on real traffic. Include realistic noise in your training inputs.

**4. Teacher data confidentiality.** Check your model provider's terms of service. Some providers restrict using API outputs to train or fine-tune competing models. The exact terms vary — read them before building a distillation pipeline.

**5. Catastrophic forgetting.** Fine-tuning the student on distillation data can cause it to forget general capabilities. Use LoRA adapters (fine-tune only a small fraction of parameters) to minimize forgetting, and evaluate on held-out general-capability benchmarks alongside task-specific evals.

**6. Insufficient training data.** 100 examples is rarely enough. Most tasks need 1,000–5,000 examples to show meaningful improvement, and tail behaviors (rare edge cases) require even more. Generate broadly from your expected input distribution, not just the easy cases.

## Project ideas

**1. End-to-end distillation on a classification task** — Pick a classification task (topic detection, sentiment analysis, intent classification). Use Claude Sonnet as the teacher to label 1,000 inputs. Fine-tune Llama 3.2 3B on the labeled data using LoRA. Measure accuracy on a held-out test set. Compare: teacher vs. student on test set accuracy, inference latency, and cost per 1,000 calls.

**2. Quality filter study** — Generate 500 teacher responses for a task. Group them by confidence score (low/medium/high). Train three student models: on all 500, on medium+high confidence only, on high confidence only. Measure whether filtering improves student quality or just reduces training data harmfully.

**3. OOD detection** — Build the OOD detector described in the My take: embed all training inputs, embed production queries, compute cosine similarity to nearest training example. Threshold at 0.7: below threshold, route to teacher; above, use student. Measure how many production queries are OOD and what the quality difference is for in-distribution vs. OOD queries.

**4. Data volume study** — Generate teacher responses at 100, 500, 1,000, 2,000, and 5,000 examples. Train a student on each dataset size. Plot student quality vs. data volume. Find the knee of the curve — the point where adding more data gives diminishing returns. This tells you the minimum viable training set size for your task.

## Going deeper

#### Foundational reading

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) — the original knowledge distillation paper; introduces soft targets and temperature scaling.
- Ho et al., "Large Language Models Are Reasoning Teachers" (2023) — chain-of-thought distillation; shows that including teacher reasoning steps in student training data significantly improves student performance.

#### Libraries and tools

- `trl` documentation (huggingface.co/docs/trl) — the reference for SFTTrainer and other fine-tuning utilities; includes distillation examples.
- `peft` documentation (huggingface.co/docs/peft) — LoRA and other parameter-efficient fine-tuning methods; essential for distilling into smaller models without catastrophic forgetting.
