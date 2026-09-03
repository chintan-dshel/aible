---
sidebar_position: 3
title: Structured Outputs
description: Getting reliable JSON, schemas, and typed data from language models — JSON mode, constrained decoding, and Pydantic integration.
---

# Structured Outputs

## What it is

Structured outputs are the practice of constraining a model's generation to a specific format — typically JSON or YAML (two standard, machine-readable text formats for structured data) or a typed schema (a template describing exactly what fields the output must have and what type each one is) — so that downstream code can parse and use the result without fragile string manipulation.

A model that returns:
```json
{"sentiment": "negative", "confidence": 0.91, "topics": ["battery", "screen"]}
```
instead of:
```
The sentiment is negative. I'm fairly confident (around 91%). The review mentions the battery life and the screen.
```
...is a model you can build reliable software on.

## The problem it solves

Language models generate text. Applications consume data. The gap between these two facts is where most production LLM (large language model) systems break.

The naive approach — prompt the model to "output JSON" and parse the result — fails in practice for several reasons: the model sometimes adds prose before or after the JSON block, field names drift, numeric strings appear instead of numbers, nested structures are inconsistently formatted, and required fields go missing on unusual inputs. Every one of these requires a special case in your parsing code, and special cases accumulate. After enough of them, you have a fragile parser that breaks on novel inputs.

## How it works under the hood

### JSON mode

Most major APIs expose a JSON mode that guarantees the model's output is valid JSON, even if it can't guarantee the specific schema. The mechanism varies: some use constrained decoding at the token level (the sampler is prevented from selecting tokens that would make the output invalid JSON at any point); others use output-layer enforcement with retries.

JSON mode ensures valid JSON but not valid schema — the model can still omit fields, add unexpected fields, or use wrong types.

### Schema-constrained generation (tool use)

The more robust approach: define the output as a tool schema. The model is instructed to "call" the tool with arguments matching your schema, and the API enforces the types. (If you're not yet familiar with tool use, see [Function Calling](./function-calling) — in brief: tools let you define a typed schema that the API guarantees the model will populate.)

The code below defines that schema, forces the model to "call" it, and gets back a JSON object guaranteed to match:

```python
import anthropic
import json

client = anthropic.Anthropic()

EXTRACTION_TOOL = {
    "name": "extract_product",
    "description": "Extract structured product information from text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "price_usd": {"type": "number"},
            "in_stock": {"type": "boolean"},
        },
        "required": ["product_name", "price_usd", "in_stock"]
    }
}

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    tools=[EXTRACTION_TOOL],
    tool_choice={"type": "tool", "name": "extract_product"},
    messages=[{
        "role": "user",
        "content": "The Acme Blender Pro is $89.99 and currently in stock."
    }]
)

result = json.loads(response.content[0].input)
# {'product_name': 'Acme Blender Pro', 'price_usd': 89.99, 'in_stock': True}
```

`tool_choice={"type": "tool", "name": "..."}` forces the model to call that specific tool — you're guaranteed a JSON object matching the schema.

### Pydantic + Instructor

The `instructor` library wraps the Anthropic and OpenAI clients to add Pydantic (a Python library for defining the shape data must have, and having it checked automatically) model validation with automatic retries. You define your output as a Pydantic model; instructor handles the schema, the tool call, parsing, validation, and retry logic:

```python
import anthropic
import instructor
from pydantic import BaseModel, Field
from typing import Literal

client = instructor.from_anthropic(anthropic.Anthropic())

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)  # ge/le: reject any value outside 0.0-1.0
    key_phrases: list[str]
    reasoning: str

result = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    response_model=SentimentResult,
    messages=[{
        "role": "user",
        "content": "Classify: 'The battery lasts forever but the screen is terrible.'"
    }]
)

print(result.sentiment)    # "negative"
print(result.confidence)   # 0.72
```

If the model returns a confidence of 1.5, Pydantic rejects it; instructor retries with the validation error in context. This handles a large class of schema violations automatically.

### Constrained decoding (Outlines)

For local models, `outlines` provides true token-level constrained decoding: the sampling mask is modified at each step so that only tokens leading to valid outputs can be selected. This makes invalid outputs structurally impossible rather than correcting them after the fact:

```python
import outlines
from pydantic import BaseModel

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

generator = outlines.generate.json(model, Product)
result = generator("Extract: 'Widget Pro at $29.99, available now.'")
# result is a validated Product instance — invalid JSON is impossible
```

## Concrete example

An invoice extraction pipeline:

```python
import anthropic
import instructor
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

client = instructor.from_anthropic(anthropic.Anthropic())

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    invoice_number: str
    vendor_name: str
    invoice_date: date
    due_date: Optional[date] = None
    line_items: list[LineItem]
    subtotal: float
    tax_amount: float = Field(default=0.0)
    total_amount: float

def extract_invoice(text: str) -> Invoice:
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        response_model=Invoice,
        messages=[{"role": "user", "content": f"Extract invoice data:\n\n{text}"}]
    )

invoice = extract_invoice("""
    Invoice #INV-2024-0392
    From: Acme Supplies Ltd
    Date: January 15, 2024  Due: February 15, 2024

    3x Widget Pro @ $29.99 = $89.97
    1x Shipping = $9.99

    Subtotal: $99.96  Tax (8%): $8.00  Total: $107.96
""")

print(invoice.total_amount)   # 107.96
print(invoice.line_items[0])  # LineItem(description='Widget Pro', quantity=3, ...)
```

## When to use it / when not to

#### Use structured outputs when

- Downstream code needs to parse model output — always prefer schema enforcement over string parsing
- You're building pipelines where one model's output feeds another
- You need type guarantees (a number, not a string that looks like a number)
- Consistency across thousands of calls matters more than flexibility

#### Consider unstructured output when

- You're generating prose (summaries, drafts, explanations) — forcing JSON adds overhead with no benefit
- The output schema is unknown or highly variable — structured output requires knowing the shape in advance
- The user reads the output directly in a chat interface

#### The practical question

Can you write a Pydantic model that describes what a good output looks like? If yes, use instructor. The act of defining the class forces clarity about what you actually need — vague schemas produce vague outputs.

:::tip[My take]

Use Pydantic models as your schema definition language even if you're not using instructor. Writing the class forces you to be explicit about types, optionality, and constraints — which clarifies your own thinking about the task, not just the model's. "A confidence score" is vague; `confidence: float = Field(ge=0.0, le=1.0)` is precise.

The retry loop in instructor is underrated. Models fail schema validation more often on edge-case inputs than on clean examples. Rather than building your own retry logic, let instructor handle it and log the failures — patterns in what causes validation errors often reveal ambiguity in your schema or your prompt.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| `instructor` | Pydantic + LLM client integration with retry logic; works with Anthropic, OpenAI, Gemini |
| `outlines` | Token-level constrained decoding for local models; strongest guarantees |
| `marvin` | Higher-level abstraction for extraction, classification, generation tasks |
| Anthropic tool use | Native schema enforcement via `tool_choice`; no extra library needed |
| OpenAI Structured Outputs | Native JSON Schema enforcement in the API |
| Pydantic v2 | Schema definition, validation, and serialization — the foundation |

## Common failure modes and gotchas

**1. Schema hallucination.** The model produces values that pass schema validation but are factually wrong. A model can return `{"price": 0.01}` for a $99 item and Pydantic won't complain. Schema validation and factual accuracy are orthogonal problems.

**2. Over-constrained enums.** An enum — a field restricted to one of a fixed, named list of values, `category: Literal["billing", "technical", "account"]` here — that's missing "shipping" will cause the model to misclassify shipping tickets rather than flag them as uncovered. Add an `other` category with an explanation field for any classifier.

**3. Retry amplification.** Instructor's retry loop is useful but masks systematic errors. If 30% of calls fail validation and retry succeeds, your schema or prompt is wrong. Log validation failures and audit the patterns.

**4. Nested schema brittleness.** Complex deeply-nested schemas with many required fields fail more often on ambiguous inputs. Keep schemas as flat as possible; mark fields `Optional` when the source document might not contain them.

**5. Date and number parsing drift.** Models handle dates inconsistently across formats. Pydantic's `date` type usually handles common variants, but validate your extraction on inputs with unusual formatting before deploying.

**6. JSON in markdown.** Models sometimes wrap JSON in a markdown code block. Tool use and JSON mode prevent this. If you're doing string-parsed JSON, strip code fences before parsing.

## Project ideas

**1. Invoice extractor with format variety** — Build the invoice extraction pipeline above with 10 sample invoices in varying formats (different date formats, optional fields, varying line item structures). Measure per-field extraction accuracy. Then make `due_date` required and observe how the model handles invoices with no due date. This concretely demonstrates the over-constrained schema failure mode.

**2. Schema vs. string parsing comparison** — Take an extraction task with ground-truth data. Implement two versions: free-form JSON prompt with string parsing, and instructor with Pydantic. Measure accuracy and parse failure rate on a 50-item test set, including 10 adversarial inputs designed to break the schema. The gap on adversarial inputs is usually large.

**3. Outlines local extraction** — Run the same extraction task on a 7B local model (Mistral, Llama 3 8B) with and without constrained decoding via outlines. Measure how often the unconstrained model produces invalid JSON or schema violations vs. the constrained version. Constrained decoding often recovers significant accuracy on structured tasks from smaller models.

## Going deeper

#### Foundational resources

- Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023) — the paper behind outlines; explains token-level constrained decoding using finite state machines.
- Pydantic documentation (docs.pydantic.dev) — especially `Field()` validators; the foundation for instructor-based schemas.
- Anthropic tool use documentation — the native way to enforce schemas without extra libraries.

#### Libraries

- `instructor` (GitHub: `jxnl/instructor`) — best starting point; excellent documentation with worked examples for Anthropic and OpenAI.
- `outlines` (GitHub: `outlines-dev/outlines`) — for local models or cases where retry-based approaches aren't sufficient.
- `marvin` (GitHub: `prefecthq/marvin`) — higher-level; good for extraction and classification with minimal boilerplate.
