---
sidebar_position: 1
title: Core Building Blocks
description: The primitives you combine to build AI applications — prompting, RAG, tool use, agents, memory.
---

# Core Building Blocks

The foundations section explained what language models are and how they work. This section covers what you build with them: the primitives that appear in almost every production AI system.

Each page follows the same pattern — what it is, when to use it, how it works, and where it breaks. The pages cross-reference each other because the building blocks compose: RAG feeds context into prompts; tool use produces structured outputs; multi-agent systems orchestrate all of the above.

## Reading order

**If you're new to building with LLMs:**
[Prompting](./prompting) → [Structured Outputs](./structured-outputs) → [Function Calling](./function-calling) → [RAG](./rag)

**If you're building agents:**
[Function Calling](./function-calling) → [Multi-Agent Systems](./multi-agent-systems) → [Memory Architectures](./memory-architectures)

## Pages in this section

| Page | What it covers |
|---|---|
| [Prompting](./prompting) | Zero/few-shot, chain-of-thought, system prompts, injection risks, patterns that hold up |
| [Structured Outputs](./structured-outputs) | JSON mode, tool-use schemas, Pydantic + Instructor, constrained decoding |
| [Function Calling](./function-calling) | Tool schemas, the execution loop, parallel calls, error handling |
| [RAG](./rag) | Chunking, dense retrieval, reranking, hybrid search, full pipeline |
| [Multi-Agent Systems](./multi-agent-systems) | Sequential, parallel, hierarchical topologies; coordination patterns; reliability |
| [Memory Architectures](./memory-architectures) | In-context, external semantic, episodic, and parametric memory; when to use each |
