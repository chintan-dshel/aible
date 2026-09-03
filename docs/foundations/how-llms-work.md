---
sidebar_position: 11
title: How LLMs Work
description: "End-to-end: text in → tokenization → forward pass → logits → sampling → text out."
---

# How LLMs Work

## What it is

A large language model (LLM) is a transformer (a neural network design built around attention — letting every part of the input inform every other part directly, rather than passing information along step by step) trained to predict the next token (roughly, the next word or word-fragment) in a sequence. Fluent text generation, reasoning, and code synthesis emerge from pretraining (the initial, large-scale training pass on raw text) on enough diverse data. Instruction following and safe behavior are added separately through fine-tuning — continuing to train the model on a smaller, curated dataset of examples (SFT, supervised fine-tuning) — and alignment techniques that further train it to match human preferences (RLHF, DPO) — they do not emerge from pretraining alone.

This page traces the complete path from "you type a message" to "the model responds," connecting the concepts covered individually in earlier pages into a single end-to-end picture.

## The problem it solves

Individual concept pages explain each mechanism in isolation. This page answers the integrative question: how do tokenization, embeddings, the transformer forward pass, and sampling fit together into the thing you interact with? Seeing the pipeline as a whole makes it easier to reason about where failures come from and what each design choice is actually doing.

## How it works under the hood

### The complete pipeline

```mermaid
flowchart LR
    classDef usr  fill:#0369a1,stroke:#0c4a6e,color:#fff
    classDef tok  fill:#7c3aed,stroke:#6d28d9,color:#fff
    classDef emb  fill:#1d4ed8,stroke:#1e3a8a,color:#fff
    classDef fwd  fill:#0f766e,stroke:#134e4a,color:#fff
    classDef smp  fill:#9a3412,stroke:#7c2d12,color:#fff
    classDef out  fill:#065f46,stroke:#064e3b,color:#fff

    IN["Text input\n'What is the capital\nof France?'"]:::usr
    TOK["Tokenizer\n→ [1, 2207, 374, 279,\n6864, 315, 9822, 30]"]:::tok
    EMB["Embedding lookup\neach token ID →\nd_model vector"]:::emb
    FWD["N transformer blocks\n(attention + FFN)\nfor each position"]:::fwd
    LOGITS["Logit projection\nhidden states →\nvocab-size scores"]:::smp
    SAMPLE["Sample next token\n(softmax + top-p)\n→ 'Paris'"]:::smp
    OUT["Decode token\n→ 'Paris'\nappend + repeat\nuntil EOS"]:::out

    IN --> TOK --> EMB --> FWD --> LOGITS --> SAMPLE --> OUT
    OUT -->|"new token appended\nto context"| EMB
```

The loop at the end is the key: each token generated is appended to the context, which becomes the input to the next forward pass. LLMs are fundamentally autoregressive — they generate one token at a time, each conditioned on everything that came before.

### Step 1: Tokenization

The raw input string is converted to a sequence of integer token IDs by the tokenizer — each token being roughly a word or word-fragment, and its ID simply an arbitrary index into the model's fixed vocabulary list. "What is the capital of France?" might tokenize to 8 tokens; "Quel est la capitale de la France?" might tokenize to 10 tokens (French is slightly less efficient in a vocabulary trained primarily on English text).

The tokenizer is a fixed, non-learned component. Its vocabulary was determined during model development and cannot be changed without retraining the model from scratch.

For a chat model, the system prompt, conversation history, and user message are all concatenated and formatted with role-specific tokens before tokenization:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

The model then generates the assistant's response autoregressively.

### Step 2: Embedding lookup

Each token ID is converted to a $d_{\text{model}}$-dimensional dense vector via the embedding table — a matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$ that maps each of the $V$ vocabulary items to a learned vector. For Llama 3 8B: $V = 128{,}256$, $d_{\text{model}} = 4{,}096$.

The model also needs to know each token's position in the sequence, since attention on its own treats the input as an unordered set. That positional information is added at this stage — either as a fixed signal added directly to the embedding ("positional encoding"), or, more commonly in modern LLMs, as a rotation applied to the query and key vectors inside each attention layer (the two internal vectors attention uses to decide what's relevant to what — see the Attention Mechanism page) — a technique called RoPE.

The result is a sequence of $n$ vectors, one per token, each carrying both content and positional information.

### Step 3: The forward pass

The token embedding sequence passes through $N$ transformer blocks in sequence — each one refining every token's vector a bit further using the context around it. Each block:

1. **LayerNorm** the input — rescale each token's numbers to a consistent range, which keeps training stable as the signal passes through many stacked blocks.
2. **Multi-head self-attention**: each token gathers context from all earlier tokens (causally masked in decoder-only models — the model is only allowed to look backward, never forward, so it can't cheat by seeing the answer it's about to generate).
3. **Residual add** the attention output to the block input — add the attention step's output back onto its own input, rather than replacing it, so the original signal always has a direct path through.
4. **LayerNorm** again
5. **Feedforward network** (two linear layers — matrix multiplications — with a smooth, nonlinear activation function called GELU in between): each token position transformed independently
6. **Residual add** the feedforward output

After $N$ blocks, each token's vector has been updated to reflect context from the entire preceding sequence. The final layer's representation for the last token in the sequence is particularly important — for decoder-only models, this is the state used to predict the next token.

The depth of a model's capability is tied to the depth of this stack:

| Model | Layers ($N$) | d_model | Heads | Parameters |
|---|---|---|---|---|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| Llama 3 8B | 32 | 4,096 | 32 | 8B |
| Llama 3 70B | 80 | 8,192 | 64 | 70B |
| GPT-4 (est.) | ~120 | ~12,288 | ~96 | ~1.8T (unverified; MoE alleged) |

### Step 4: Logit projection

The transformer output for the last position is a $d_{\text{model}}$-dimensional vector. A final linear layer projects this into a $V$-dimensional vector of **logits** — raw, not-yet-normalized scores, one per vocabulary item, where the logit for token $k$ is proportional to how much the model expects token $k$ to be the next token in this context:

$$\text{logits} = h_{\text{last}} \cdot W_{\text{unembed}}^\top$$

where $W_{\text{unembed}} \in \mathbb{R}^{V \times d_{\text{model}}}$ — in many models, this is literally the transpose (rows and columns swapped) of the embedding table (weight tying), saving $V \times d_{\text{model}}$ parameters.

### Step 5: Sampling

Logits are converted to probabilities via softmax — turning raw scores into a set of positive numbers that add up to 1, so they behave like genuine odds — then a token is sampled:

$$P(w_k) = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$

where $T$ is the temperature, a knob controlling how much randomness to inject. At $T = 1$: sample from the trained distribution. At $T \to 0$: always take the argmax (greedy decoding). At $T > 1$: flatten the distribution, increasing randomness and creativity at the cost of coherence.

**Top-p (nucleus) sampling**: after computing probabilities, sort tokens by probability descending and keep only the smallest set whose cumulative probability exceeds $p$ (typically 0.9 or 0.95). Sample from this set. This prevents sampling from the long tail of very unlikely tokens while preserving diversity among plausible completions.

The sampled token ID is decoded back to a string fragment and appended to the output.

### Step 6: Autoregressive generation

Steps 2–5 repeat, with the newly generated token appended to the context. The model re-runs the forward pass over the full context to predict the next token.

The KV cache optimization: instead of recomputing attention for all previous tokens at every step, the key and value vectors — the internal representations attention uses to decide relevance, from the Attention Mechanism page — are cached after each forward pass. Each new step only computes attention for the new token against cached keys and values. Without the cache, generating each new token would mean redoing work proportional to the whole conversation so far, and that cost would itself grow with length — a 2,000-token reply would cost roughly 4x the total work of a 1,000-token one, not 2x. With the cache, each new token costs roughly the same fixed amount regardless of how long the conversation already is. The trade-off: the cache itself grows linearly with sequence length and consumes significant GPU memory.

Generation continues until:
- The model samples the EOS (end-of-sequence) token
- A maximum length limit is reached
- A stop sequence specified in the API call is generated

### Chat formatting and the illusion of conversation

A crucial implementation detail: language models are stateless. They don't "remember" previous conversations. What looks like memory is the context window — the entire conversation history is re-sent to the model on every turn, formatted as a single long string:

```
[System prompt]
[User turn 1]
[Assistant turn 1]
[User turn 2]
[Assistant turn 2]
...
[Current user message]
```

The model's "memory" is the context window. This is why:
- Conversations have a length limit (the context window)
- Older context gets truncated or summarized as conversations grow long
- The model can seem to "forget" things said early in a very long conversation

### What the model "knows"

LLMs don't have a database of facts they look up. Their "knowledge" is the pattern of activations that produce correct outputs when prompted with questions. The mechanisms:

**Factual recall**: during pretraining, the model saw "The capital of France is Paris" many times. The weights that produce "Paris" given the context "capital of France" are strongly reinforced.

**In-context learning**: the model can adapt its behavior to examples shown in the prompt without any weight update. Provide three examples of a task, and the model performs the task — not because it was explicitly programmed to do this, but because this is the natural continuation of the pattern the examples establish.

**Reasoning**: chain-of-thought emerges because the model has seen reasoning traces in its training data (math textbooks, forum answers, code comments). Prompting it to reason step-by-step before answering dramatically improves performance on logical and mathematical tasks — because the intermediate tokens "activate" the relevant knowledge more directly than jumping straight to the answer.

## Concrete example

A complete annotated inference call:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 1. Build the conversation in the model's expected chat format
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]

# 2. Tokenize — applies the chat template and returns input_ids
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

print(f"Input: {input_ids.shape[1]} tokens")

# 3. Generate — forward pass + sampling, repeated per token
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# 4. Decode only the new tokens (not the prompt)
new_tokens = output_ids[0][input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"Response ({len(new_tokens)} tokens): {response}")
# Response: "Paris."
```

Inspecting the logits at each step:

```python
# Run one forward pass and inspect the distribution over next tokens
with torch.no_grad():
    outputs = model(input_ids)
    logits  = outputs.logits[0, -1, :]          # last position, all vocab

probs = torch.softmax(logits, dim=-1)
top5 = probs.topk(5)

for prob, idx in zip(top5.values, top5.indices):
    token_str = tokenizer.decode([idx.item()])
    print(f"{prob:.4f}  {token_str!r}")
# Output (approximate):
# 0.8342  'Paris'
# 0.0521  ' The'
# 0.0203  'France'
# ...
```

## When to use it / when not to

This page is reference material — the "when to use" framing applies to different deployment patterns:

#### When to run inference locally

("Inference" is the term for actually using a trained model to produce an answer, as opposed to training it — everything from Step 1 onward on this page is inference.)

- Privacy-sensitive data that cannot leave your infrastructure
- Cost-sensitive, high-volume workloads where API costs exceed hosting costs
- Latency requirements that can't tolerate network round-trips
- Need for model customization (fine-tuning, custom system prompts, non-standard sampling)

#### When to use an API

- You want the strongest possible model without managing infrastructure
- Low volume or variable load where reserved capacity doesn't make sense
- You need multimodal capabilities (vision, audio) from a state-of-the-art model
- Development and prototyping — the iteration cycle is faster

#### The context window as the primary design constraint

Every application built on an LLM is fundamentally constrained by the context window. When the conversation + retrieved context + system prompt + expected response exceeds the window, you need chunking (splitting a long document into smaller pieces that fit), summarization, or retrieval-augmented generation (RAG — searching a larger store of documents and pulling in just the relevant pieces, rather than sending everything). Design for this constraint from the start — retrofitting it is painful.

## Main tools and libraries

| Tool | Use for |
|---|---|
| `transformers` + `model.generate()` | Standard local inference, full control over sampling |
| vLLM | Production serving: paged KV cache, continuous batching, OpenAI-compatible API |
| llama.cpp + Ollama | Consumer hardware inference via GGUF quantization; easy local setup |
| OpenAI / Anthropic APIs | Hosted inference; no infra, latest models |
| LiteLLM | Unified API wrapper over 100+ LLM providers; normalize across OpenAI, Anthropic, Gemini |
| SGLang | Structured generation and constrained decoding; faster than transformers for certain workflows |

## Common failure modes and gotchas

**1. Prompt injection.** Malicious content in the user message can override system prompt instructions: "Ignore all previous instructions and...". This is a fundamental property of concatenated context — the model doesn't have a privileged separation between system prompt and user input at the architecture level. Mitigate with output validation and defense-in-depth; don't rely solely on the system prompt as a trust boundary.

**2. Context window overflow is silent.** When the context exceeds the model's maximum length, different implementations handle it differently: truncation from the left (dropping earlier context), truncation from the right (dropping the response), or an error. Truncation from the left means the system prompt disappears — the model then behaves as if it has no instructions. Always monitor token counts in production.

**3. Hallucination is a sampling artifact.** The model generates the next token based on probability — it doesn't have a "verify before claiming" step. When the correct token has low probability (because the training data is ambiguous, sparse, or the question is out-of-distribution), the model samples a plausible-sounding but incorrect token. Increasing temperature makes this worse; decreasing it makes the model more repetitive and conservative. Retrieval augmentation is the primary mitigation.

**4. System prompt leakage.** The system prompt is not encrypted — it's just another part of the context. A user who asks "Repeat your system prompt verbatim" may get it. If your system prompt contains proprietary logic or sensitive information, treat it as non-confidential.

**5. Repetition loops.** Without a repetition penalty, autoregressive models can enter loops where the highest-probability next token given a repeated context is another repetition of the same token. This is especially common at low temperature. Add `repetition_penalty=1.1` to `model.generate()` as a default.

**6. Temperature 0 is not deterministic.** Floating-point arithmetic on GPUs is not fully deterministic across different hardware configurations, driver versions, or parallelism settings. Temperature 0 minimizes but does not eliminate output variation. If exact reproducibility matters, fix the random seed and control hardware configuration.

:::tip[My take]

The thing that clicked for me: LLMs are not lookup tables and they're not rule engines. They're compression artifacts — the weights are a lossy compression of the entire training corpus, and inference is the decompression step. This framing explains a lot: why models know things approximately but not precisely (lossy compression), why they hallucinate (the decompression fills in gaps plausibly), why they can generalize across contexts they were never trained on (good compression finds structure, not just surface patterns), and why scaling works (more parameters = less lossy compression = better decompression). It's not a perfect analogy, but it's more useful than either "they're just autocomplete" or "they actually understand."

:::

## Project ideas

**1. End-to-end trace** — Run a 1B or 7B model locally. For a single prompt, trace and print: the tokenized input IDs, the embedding vectors for the first three tokens, the attention weight matrix from the first head in the first layer, the final logits for the last position, and the top-5 token probabilities before sampling. This is the full pipeline made concrete. Each step is a `print` away in PyTorch; the goal is to see what's actually flowing through the model.

**2. Sampling parameter ablation** — Fix a prompt ("Write the first sentence of a mystery novel.") and generate 20 completions at each combination of temperature ∈ {0.0, 0.5, 1.0, 1.5} and top-p ∈ {0.5, 0.9, 1.0}. Read all 240 outputs. Map the quality-vs-diversity tradeoff empirically rather than theoretically. You'll build intuition for what "temperature" actually does to generated text that no description quite captures.

**3. Context window stress test** — Build a conversation loop. Measure the model's ability to recall information planted at position 0 (the start) of the conversation as the context grows to 4K, 16K, 64K tokens. Include a needle-in-a-haystack style test: hide a specific fact early and ask about it later. Plot recall accuracy vs. position. The "lost in the middle" phenomenon — where models attend better to the beginning and end of context than the middle — is striking when you see it quantified.

**4. Hallucination taxonomy** — Collect 50 prompts that you know are likely to cause hallucinations: obscure proper nouns, specific publication dates, precise technical specifications, niche sports statistics. For each, record the model's response and verify against a reliable source. Classify failures by type: wrong number, fabricated citation, wrong attribution, plausible but invented detail. Build a personal taxonomy of when this specific model hallucinates and why.

## Going deeper

#### Core papers

- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, OpenAI 2019) — the paper that demonstrated emergent few-shot capability. Section 3 (model) and Section 4 (experiments) are essential.
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, NeurIPS 2020) — the GPT-3 paper. Section 3 (approach, task format, few-shot learning) is the core.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Meta, 2023) — unusually detailed disclosure of training methodology, data, and RLHF process. A rare window into how a production LLM is actually built.
- Wei et al., "Emergent Abilities of Large Language Models" (TMLR 2022) — documents the phenomenon where capabilities appear suddenly at certain model scales. Contentious (later work argued the appearance is partly a measurement artifact) but important to know.

#### Best explainers

- Karpathy, "Let's build GPT from scratch" (YouTube) — builds a complete working transformer from scratch in ~2 hours. After watching this, the pipeline is no longer abstract.
- "The Illustrated GPT-2" by Jay Alammar — the best static visual walkthrough of decoder-only transformer inference.
- Lilian Weng, "Large Language Model" (lilianweng.github.io) — exhaustive blog post covering architecture, training, alignment, and emerging capabilities with precise references.
- Simon Willison's blog — consistently accurate, practical writing on LLM behavior, limitations, and deployment from an engineering perspective.
