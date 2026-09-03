---
sidebar_position: 1
title: AI Foundations
sidebar_label: AI Foundations
description: What AI and ML actually are under the hood — the concepts everything else builds on.
---

# AI Foundations

Before frameworks, APIs, or products: what is actually happening when a model generates text, classifies an image, or finds a nearest neighbor? This section builds the mental model from first principles.

Only one page here is assumed later in the book: [How LLMs Work](./how-llms-work). Read that one, then read as much or as little of the rest as you want — the patterns chapters do not require the rest. The two history pages are context, not mechanism; they are ordered last on purpose.

## Reading order

**The one page this book assumes:**
[How LLMs Work](./how-llms-work) — the end-to-end pipeline, text in to tokens out. If you've never called an LLM API, start here before the patterns chapters.

**If you want the full mechanism, in order:**
[What is AI?](./what-is-ai) → [How LLMs Work](./how-llms-work) → [Neural Networks](./neural-networks) → [Transformers](./transformers) → [Attention Mechanism](./attention) → [Embeddings](./embeddings) → [Tokenization](./tokenization) → [Training vs Inference](./training-vs-inference)

**Optional, for context rather than mechanism:**
[History — Timeline](./history-timeline) → [History — Narrative](./history-narrative)

## Pages in this section

| Page | What it covers |
|---|---|
| [What is AI?](./what-is-ai) | The AI/ML/DL distinction; optimization over parameters; when to use each level |
| [How LLMs Work](./how-llms-work) | End-to-end pipeline: text in → tokens → forward pass → logits → generation |
| [Neural Networks](./neural-networks) | Perceptrons, activation functions, backpropagation, residual connections |
| [Transformers](./transformers) | Encoder/decoder/encoder-decoder; positional encoding; the full block architecture |
| [Attention Mechanism](./attention) | QKV attention from first principles; multi-head attention; FlashAttention |
| [Embeddings](./embeddings) | word2vec to contextual embeddings; geometry; retrieval and ANN search |
| [Tokenization](./tokenization) | BPE, WordPiece, SentencePiece; vocabulary design; where tokenization shapes behavior |
| [Training vs Inference](./training-vs-inference) | Pretraining, SFT, RLHF/DPO, LoRA/QLoRA, inference optimization |
| [History — Timeline](./history-timeline) *(optional)* | Key milestones 1950–2025, dense and scannable |
| [History — Narrative](./history-narrative) *(optional)* | The story behind the milestones — winters, eruptions, pattern recognition |
