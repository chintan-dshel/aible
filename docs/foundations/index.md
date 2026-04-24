---
sidebar_position: 1
title: Foundations
description: What AI and ML actually are under the hood — the concepts everything else builds on.
---

# Foundations

Before frameworks, APIs, or products: what is actually happening when a model generates text, classifies an image, or finds a nearest neighbor? This section builds the mental model from first principles — starting with definitions and working up to a complete picture of how a modern language model produces its output.

The pages build on each other in order, but each stands alone as a reference.

## Reading order

**Start here if you're new to ML:**
[What is AI?](./what-is-ai) → [History — Narrative](./history-narrative) → [Neural Networks](./neural-networks) → [Transformers](./transformers) → [Attention Mechanism](./attention) → [How LLMs Work](./how-llms-work)

**If you're already comfortable with neural networks:**
[Transformers](./transformers) → [Attention Mechanism](./attention) → [Embeddings](./embeddings) → [Tokenization](./tokenization) → [Training vs Inference](./training-vs-inference) → [How LLMs Work](./how-llms-work)

## Pages in this section

| Page | What it covers |
|---|---|
| [What is AI?](./what-is-ai) | The AI/ML/DL distinction; optimization over parameters; when to use each level |
| [History — Timeline](./history-timeline) | Key milestones 1950–2025, dense and scannable |
| [History — Narrative](./history-narrative) | The story behind the milestones — winters, eruptions, pattern recognition |
| [Neural Networks](./neural-networks) | Perceptrons, activation functions, backpropagation, residual connections |
| [Transformers](./transformers) | Encoder/decoder/encoder-decoder; positional encoding; the full block architecture |
| [Attention Mechanism](./attention) | QKV attention from first principles; multi-head attention; FlashAttention |
| [Embeddings](./embeddings) | word2vec to contextual embeddings; geometry; retrieval and ANN search |
| [Tokenization](./tokenization) | BPE, WordPiece, SentencePiece; vocabulary design; where tokenization shapes behavior |
| [Training vs Inference](./training-vs-inference) | Pretraining, SFT, RLHF/DPO, LoRA/QLoRA, inference optimization |
| [How LLMs Work](./how-llms-work) | End-to-end pipeline: text in → tokens → forward pass → logits → generation |
