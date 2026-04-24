---
sidebar_position: 1
title: Meta-Infrastructure
description: The layer beneath AI apps — evals, observability, guardrails, vector databases, MCP, and everything else that makes production AI work.
---

# Meta-Infrastructure

This is the layer most tutorials skip. It's the difference between a demo that works once and a system that works reliably. Every serious AI application needs most of what's here.

## Topics

- [Evals](./evals) — measuring whether your AI system actually works
- [Observability](./observability) — traces, spans, prompt logging, latency tracking
- [Guardrails](./guardrails) — input/output filtering and policy enforcement
- [Prompt Injection](./prompt-injection) — attack taxonomy and defenses
- [Output Validation](./output-validation) — schema validation, semantic checks, hallucination detection
- [Vector Databases](./vector-databases) — HNSW, IVF, pgvector vs Pinecone vs Weaviate
- [Caching](./caching) — semantic cache, exact cache, KV cache, prompt caching
- [Orchestration Frameworks](./orchestration-frameworks) — LangChain, LlamaIndex, DSPy, raw SDK
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
