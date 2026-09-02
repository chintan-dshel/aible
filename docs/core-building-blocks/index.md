---
sidebar_position: 1
title: Building Blocks
sidebar_label: Building Blocks
description: The primitives you combine to build AI applications — prompting, RAG, tool use, agents, memory, and the two pages that lead directly into Part 1.
---

# Building Blocks

Foundations explained what language models are and how they work. This section covers what you build with them: the primitives that appear in almost every production AI system, ending with the two pages that lead directly into Part 1 — Patterns.

Each page follows the same pattern — what it is, when to use it, how it works, and where it breaks. The pages cross-reference each other because the building blocks compose: RAG feeds context into prompts; tool use produces structured outputs; multi-agent systems orchestrate all of the above.

Two pages here are cross-listed from elsewhere in the reference library because they are prerequisite tooling for Part 1, not reference material to look up later: [Orchestration Frameworks](../meta-infrastructure/orchestration-frameworks) (the tooling for the multi-agent topologies on this page) and [Long-Horizon Agents](../aspirational/long-horizon-agents) (checkpointing and human-in-the-loop gates — the most directly applicable page in the whole book, despite its folder).

## Reading order

**If you're new to building with LLMs:**
[Prompting](./prompting) → [Structured Outputs](./structured-outputs) → [Function Calling](./function-calling) → [RAG](./rag)

**If you're headed into Part 1 — Patterns:**
[Function Calling](./function-calling) → [Multi-Agent Systems](./multi-agent-systems) → [Orchestration Frameworks](../meta-infrastructure/orchestration-frameworks) → [Memory Architectures](./memory-architectures) → [Long-Horizon Agents](../aspirational/long-horizon-agents)

**If you're building with AI APIs right now and skipping the book:**
[Prompting](./prompting) → [Structured Outputs](./structured-outputs) → [Function Calling](./function-calling) → [RAG](./rag) → [Evals](../meta-infrastructure/evals) → [Guardrails](../meta-infrastructure/guardrails) → [Prompt Injection](../meta-infrastructure/prompt-injection). Then, by what you're building: caching-heavy → [Caching](../meta-infrastructure/caching) + [Latency Optimization](../meta-infrastructure/latency-optimization); cost-sensitive → [Model Routing](../meta-infrastructure/model-routing) + [Cost Tracking](../meta-infrastructure/cost-tracking); document-heavy → [Document Processing](../meta-infrastructure/document-processing) + [Vector Databases](../meta-infrastructure/vector-databases).

## Pages in this section

| Page | What it covers |
|---|---|
| [Prompting](./prompting) | Zero/few-shot, chain-of-thought, system prompts, injection risks, patterns that hold up |
| [Structured Outputs](./structured-outputs) | JSON mode, tool-use schemas, Pydantic + Instructor, constrained decoding |
| [Function Calling](./function-calling) | Tool schemas, the execution loop, parallel calls, error handling |
| [RAG](./rag) | Chunking, dense retrieval, reranking, hybrid search, full pipeline |
| [Multi-Agent Systems](./multi-agent-systems) | Sequential, parallel, hierarchical topologies; coordination patterns; reliability |
| [Orchestration Frameworks](../meta-infrastructure/orchestration-frameworks) | LangChain, LlamaIndex, DSPy, raw SDK — the tooling choice behind a multi-agent build |
| [Memory Architectures](./memory-architectures) | In-context, external semantic, episodic, and parametric memory; when to use each |
| [Long-Horizon Agents](../aspirational/long-horizon-agents) | Planning, checkpointing, human-in-the-loop gates, the dead-man's switch |
