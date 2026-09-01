---
sidebar_position: 0
title: Part 1 — Patterns
description: The ground rules of agentic system design. Eight patterns, each explained diagram-first and traced into the real systems in Part 2.
---

# Part 1 — Patterns

Eight design patterns that show up in every multi-agent system worth running in production. Each chapter has the same shape:

1. **The problem in one diagram** — what goes wrong without the pattern
2. **The pattern** — the mechanism, drawn before it is described
3. **Decision rules** — when to use it and when not to, stated as tests you can apply
4. **Where it shows up in the teardowns** — forward links into Part 2
5. **Failure modes** — what goes wrong when the pattern is applied badly
6. **Reference material** — the deeper reference pages for this topic

## Chapters

| # | Chapter | One line |
|---|---|---|
| 1 | Orchestration vs. autonomy | *In progress* |
| 2 | [Stage gates](./stage-gates) | A gate is where you decide whether to spend the next dollar or take the next irreversible step. |
| 3 | State machines | *In progress* |
| 4 | QA pipelines | *In progress* |
| 5 | LLM-as-judge | *In progress* |
| 6 | Cost-aware model routing | *In progress* |
| 7 | Memory layers | *In progress* |
| 8 | Failure modes and attack surfaces | *In progress* |

## Reading the diagrams

Every diagram in Parts 1 and 2 uses the same seven node types:

- <span className="dg-swatch dg-swatch--orch"></span> **Orchestrator / control** — code that decides what runs next
- <span className="dg-swatch dg-swatch--agent"></span> **Agent** — a model call with a prompt and, usually, a job
- <span className="dg-swatch dg-swatch--gate"></span> **Gate / check** — a point where output is validated before anything else happens
- <span className="dg-swatch dg-swatch--store"></span> **Store** — a database, file, or memory the system reads and writes
- <span className="dg-swatch dg-swatch--human"></span> **Human** — a person in the loop
- <span className="dg-swatch dg-swatch--external"></span> **External / untrusted** — input or a service the system does not control
- <span className="dg-swatch dg-swatch--fail"></span> **Failure path** — where things go when a check fails

Solid arrows are control flow. Dashed arrows are side effects or things that happen without waiting.
