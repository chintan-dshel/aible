---
sidebar_position: 1
title: 'Reference: Infrastructure'
sidebar_label: 'Reference: Infrastructure'
description: The layer beneath AI apps — evals, observability, guardrails, vector databases, MCP, and everything else that makes production AI work.
---

# Reference: Infrastructure

This is the layer most tutorials skip. It's the difference between a demo that works once and a system that works reliably.

These are reference pages, not chapters — each one is a deep, single-topic page meant to be entered from a Part 1 chapter's "Reference material" section, not read cold from top to bottom. If you arrived here directly: [Part 1 — Patterns](../patterns/) is the book; this is what you look up while building it.

One page that lives here, [Orchestration Frameworks](./orchestration-frameworks), is *listed* under [Building Blocks](../core-building-blocks/) in the sidebar instead — it is prerequisite tooling for Part 1 rather than something to look up later. The file has not moved; only where it appears in navigation has.

## Topics

- [Evals](./evals) — measuring whether your AI system actually works
- [Observability](./observability) — traces, spans, prompt logging, latency tracking
- [Guardrails](./guardrails) — input/output filtering and policy enforcement
- [Prompt Injection](./prompt-injection) — attack taxonomy and defenses
- [Output Validation](./output-validation) — schema validation, semantic checks, hallucination detection
- [Vector Databases](./vector-databases) — HNSW, IVF, pgvector vs Pinecone vs Weaviate
- [Caching](./caching) — semantic cache, exact cache, KV cache, prompt caching
- [MCP](./mcp) — Model Context Protocol: what it is and why it matters
- [Cost Tracking](./cost-tracking) — token accounting, budget guardrails, per-request attribution
- [Latency Optimization](./latency-optimization) — streaming, batching, speculative decoding, quantization
- [Model Routing](./model-routing) — capability-based routing, fallback chains, cost routing
- [Model Distillation](./model-distillation) — knowledge distillation, teacher-student, fine-tuning from teacher outputs
- [Synthetic Data](./synthetic-data) — generation strategies, quality filtering, bootstrap loops
- [Document Processing](./document-processing) — parsing, chunking strategies, OCR, table extraction
- [Knowledge Graphs](./knowledge-graphs) — entities, triples, GraphRAG, hybrid retrieval
- [Red Teaming](./red-teaming) — adversarial prompting, jailbreaks, structured red team process
- [Audit Logs](./audit-logs) — what to log, immutability, PII scrubbing, retention
- [Access Control for Agents](./access-control) — least-privilege, OAuth for agents, permission scoping
