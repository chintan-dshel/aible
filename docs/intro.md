---
sidebar_position: 0
title: Start Here
description: How to navigate this reference — reading paths from first principles to frontier, mapped by what you're trying to do.
---

# Start Here

A personal AI reference — built to be the resource I wanted when learning this field: technically honest, practically grounded, and honest about what isn't settled. Each page follows the same structure: what it is, how it works, working code, when to use it, failure modes, and where to go deeper.

## What's here

| Section | What it covers | Pages |
|------|------|------|
| [Foundations](./foundations/) | How AI/ML works under the hood — attention, transformers, embeddings, training | 10 |
| [Core Building Blocks](./core-building-blocks/) | Prompting, RAG, tool use, agents, memory | 6 |
| [Meta-Infrastructure](./meta-infrastructure/) | The layer beneath AI apps: evals, observability, guardrails, vector DBs, MCP | 16 |
| [Production Concerns](./production-concerns/) | Reliability, monitoring, graceful degradation, deployment patterns | 6 |
| [Aspirational](./aspirational/) | Where the field is heading — agentic systems, on-device AI, safety research | 7 |

---

## Reading paths

### "I'm new to AI and want to understand how it actually works"

Start with Foundations, read in order:

1. [What is AI?](./foundations/what-is-ai) — situates ML in the broader landscape
2. [How LLMs Work](./foundations/how-llms-work) — the high-level picture before the details
3. [Neural Networks](./foundations/neural-networks) — the underlying mechanism
4. [Transformers](./foundations/transformers) — the architecture everything is built on
5. [Attention](./foundations/attention) — the key idea that makes transformers work
6. [Embeddings](./foundations/embeddings) — how meaning becomes geometry
7. [Tokenization](./foundations/tokenization) — how text becomes numbers
8. [Training vs. Inference](./foundations/training-vs-inference) — what happens before and after deployment

The history pages ([Timeline](./foundations/history-timeline), [Narrative](./foundations/history-narrative)) are optional — good context, not mechanistic prerequisites.

---

### "I'm a developer building with AI APIs right now"

You probably don't need the full Foundations section. Start with:

1. [Prompting](./core-building-blocks/prompting) — system prompts, few-shot, chain-of-thought
2. [Structured Outputs](./core-building-blocks/structured-outputs) — reliable JSON extraction
3. [Function Calling](./core-building-blocks/function-calling) — tools, parallel calls, execution loops
4. [RAG](./core-building-blocks/rag) — retrieval-augmented generation end to end
5. [Evals](./meta-infrastructure/evals) — how to know if your system is working
6. [Guardrails](./meta-infrastructure/guardrails) — input/output safety
7. [Prompt Injection](./meta-infrastructure/prompt-injection) — the attack surface you most need to understand

Then pick from Meta-Infrastructure based on what you're building:
- Caching heavy workloads → [Caching](./meta-infrastructure/caching) + [Latency Optimization](./meta-infrastructure/latency-optimization)
- Managing costs → [Model Routing](./meta-infrastructure/model-routing) + [Cost Tracking](./meta-infrastructure/cost-tracking)
- Complex pipelines → [Orchestration Frameworks](./meta-infrastructure/orchestration-frameworks) + [MCP](./meta-infrastructure/mcp)
- Document-heavy workloads → [Document Processing](./meta-infrastructure/document-processing) + [Vector Databases](./meta-infrastructure/vector-databases)

---

### "I'm taking a system to production"

Go straight to Production Concerns, then work backward to whatever foundations are needed:

1. [Reliability](./production-concerns/reliability) — circuit breakers, retries, backoff
2. [Graceful Degradation](./production-concerns/graceful-degradation) — fallback hierarchy
3. [Monitoring](./production-concerns/monitoring) — LLM-as-judge, structural metrics, alerts
4. [Deployment Patterns](./production-concerns/deployment-patterns) — shadow mode, canary, A/B
5. [Confidence Estimation](./production-concerns/confidence-estimation) — when to trust the model
6. [Fallbacks](./production-concerns/fallbacks) — multi-model fallback chains

Supplement with: [Observability](./meta-infrastructure/observability), [Evals](./meta-infrastructure/evals), [Output Validation](./meta-infrastructure/output-validation).

---

### "I'm building multi-agent systems"

1. [Multi-Agent Systems](./core-building-blocks/multi-agent-systems) — topologies, trust, coordination
2. [Memory Architectures](./core-building-blocks/memory-architectures) — in-context, external, parametric
3. [Function Calling](./core-building-blocks/function-calling) — the tool layer
4. [Access Control for Agents](./meta-infrastructure/access-control) — least-privilege, permission scoping
5. [Prompt Injection](./meta-infrastructure/prompt-injection) — the key attack surface in agentic systems
6. [Long-Horizon Agents](./aspirational/long-horizon-agents) — planning, checkpointing, human-in-the-loop gates
7. [Agentic Computer Use](./aspirational/agentic-computer-use) — GUI automation, screenshot-act loops

---

### "I want to understand where AI is heading"

The [Aspirational](./aspirational/) section covers what's emerging or not yet production-ready. Confidence levels are lower by design — the field hasn't settled on answers, and each page says so explicitly.

Start anywhere. Suggested order if you want a coherent narrative:

1. [Long-Horizon Agents](./aspirational/long-horizon-agents) — the hardest near-term problem
2. [Agentic Computer Use](./aspirational/agentic-computer-use) — GUI control as the general interface
3. [World Models](./aspirational/world-models) — the debate about what LLMs actually understand
4. [Multimodal Frontier](./aspirational/multimodal-frontier) — where vision, audio, and text meet
5. [On-Device AI](./aspirational/on-device-ai) — privacy and the quantization inflection point
6. [Novel Interaction Paradigms](./aspirational/novel-interaction) — voice, ambient, spatial
7. [AI Safety Research](./aspirational/ai-safety) — alignment, interpretability, the control problem

---

### "I only have 30 minutes"

Read these four pages:

- [How LLMs Work](./foundations/how-llms-work) — the mental model everything else builds on
- [Prompting](./core-building-blocks/prompting) — the highest-leverage practical skill
- [RAG](./core-building-blocks/rag) — how to ground models in your own data
- [Evals](./meta-infrastructure/evals) — the thing most teams skip and most wish they hadn't

---

## How sections connect

```
Foundations
  └── explains the mechanisms behind
        Core Building Blocks
          └── enabled by / measured by
                Meta-Infrastructure
                  └── hardened by
                        Production Concerns
                          └── extended by
                                Aspirational
```

Cross-references throughout the site use `[[WikiLink]]` syntax pointing to related pages. Every page has a "Going deeper" section with papers, docs, and internal cross-references.
