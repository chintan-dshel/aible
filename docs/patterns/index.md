---
sidebar_position: 0
title: Agentic Architecture Patterns
description: Eight design decisions you will face in any multi-agent system, each given as a problem, a diagram, and a test you can apply to your own system.
---

# Agentic Architecture Patterns

Eight design decisions you will face in any multi-agent system worth running in production. Each chapter has the same shape:

1. **The problem in one diagram** — what goes wrong without the pattern
2. **The pattern** — the mechanism, drawn before it is described, with a small worked instance
3. **Decision rules** — when to use it and when not to, closing with one bolded test you can run against your own system
4. **Failure modes** — what goes wrong when the pattern is applied badly
5. **Where it shows up in the teardowns** — proof, not preview: links into the teardowns showing the pattern holding or breaking in real code
6. **Reference material** — the deeper reference pages for this topic

## Before you start

These chapters assume you've read [How LLMs Work](../foundations/how-llms-work) and the [Agentic Building Blocks](../core-building-blocks/) section, in particular [Function Calling](../core-building-blocks/function-calling) (the agent loop that chapter 1's "autonomy" end of the spectrum *is*) and [Multi-Agent Systems](../core-building-blocks/multi-agent-systems) (the topologies chapter 1 refines).

Two numbers worth having before chapter 1: a Sonnet-tier call at a few thousand tokens of context costs a fraction of a cent, so an agent loop that runs unchecked for ten steps is not free — it's ten of those. And if five agent calls each hand off to the next at 95% reliability, the chain as a whole succeeds about 77% of the time (0.95⁵), not 95%. Chapter 2 is the pattern that exists because of that second number.

## Chapters

| # | Chapter | One line |
|---|---|---|
| 1 | [Orchestration vs. autonomy](./orchestration-vs-autonomy) | Can you write down, right now, the fixed sequence of calls this task requires? If yes, you don't need a loop. |
| 2 | [Stage gates](./stage-gates) | Ask: if this gate fails on a real run tomorrow, what happens next, and who finds out? |
| 3 | [State machines](./state-machines) | Grep the codebase for every place this state gets written — do they all call the same function? |
| 4 | [LLM-as-judge](./llm-as-judge) | If I ran this judge against twenty examples I've already scored myself, would it agree with me eight times out of ten? |
| 5 | [QA pipelines](./qa-pipelines) | *In progress* |
| 6 | [Cost-aware model routing](./cost-aware-model-routing) | *In progress* |
| 7 | [Memory layers](./memory-layers) | *In progress* |
| 8 | [Failure modes and attack surfaces](./failure-modes-and-attack-surfaces) | *In progress* |

## Reading the diagrams

Every diagram on this site uses the same seven node types:

- <span className="dg-swatch dg-swatch--orch"></span> **Orchestrator / control** — code that decides what runs next
- <span className="dg-swatch dg-swatch--agent"></span> **Agent** — a model call with a prompt and, usually, a job
- <span className="dg-swatch dg-swatch--gate"></span> **Gate / check** — a point where output is validated before anything else happens
- <span className="dg-swatch dg-swatch--store"></span> **Store** — a database, file, or memory the system reads and writes
- <span className="dg-swatch dg-swatch--human"></span> **Human** — a person in the loop
- <span className="dg-swatch dg-swatch--external"></span> **External / untrusted** — input or a service the system does not control
- <span className="dg-swatch dg-swatch--fail"></span> **Failure path** — where things go when a check fails

Solid arrows are control flow. Dashed arrows are side effects or things that happen without waiting.
