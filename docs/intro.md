---
sidebar_position: 0
title: Start Here
sidebar_label: Start Here
description: What this book is, who it assumes, the two kinds of page in it, and three ways to read it.
---

# Agentic Systems: Patterns and Teardowns

This site is two things stacked on top of each other: a short book — the one named above — and a reference library it's built on top of. The book is **Agentic Architecture Patterns** and **Teardowns: Systems I Built** in the sidebar. Everything else is the library: prerequisite material and deep single-topic pages you enter from a chapter, not a syllabus you work through first.

**Agentic Architecture Patterns** is eight design decisions you will face in any multi-agent system, each given as a problem, a diagram, and a test you can apply to your own system. **Teardowns: Systems I Built** takes systems I actually built and takes them apart in the same order every time, so you can watch the eight patterns hold and fail in real code.

## Who this assumes

You can read a flowchart. You've called an LLM API at least once — or you're willing to read one page, [How LLMs Work](./foundations/how-llms-work), that explains what that means. You've never designed a system where several model calls hand work to each other, and you want to before you build one badly.

## Two kinds of page

The book and the library use different templates on purpose — if you notice the shift, that's intentional, not inconsistent.

**Chapters** (the two book sections) are argument-shaped: a diagram first, then the prose that explains it. Each pattern chapter follows the same six sections — the problem in one diagram, the pattern, decision rules, failure modes, where it shows up in the teardowns, and reference material — and ends with one bolded test you can run against your own system. Each teardown follows a fixed eight-section shape, and every claim in it is read from the system's actual source code; anything inferred is flagged `[VERIFY]`.

**Reference pages** (AI Foundations, Agentic Building Blocks, and the three reference sections) are encyclopedic: one topic, nine fixed sections — what it is, how it works, working code, when to use it, failure modes, and where to go deeper. They're meant to be entered from a chapter's "Reference material" list when you need depth on one thing, not read cold from the top.

## The map

```mermaid
flowchart TB
  accTitle: How the site is organised
  accDescr: AI Foundations feeds Agentic Building Blocks, which feeds Agentic Architecture Patterns, the first half of the book. That is checked against Teardowns, Systems I Built, the second half. Both halves link out to three reference sections — Infrastructure and Tooling, Running in Production, and The Frontier — entered from a chapter, not read in sequence.
  F["AI Foundations<br/>how a model works"]:::store --> B["Agentic Building Blocks<br/>prompt, tool call,<br/>RAG, memory, multi-agent"]:::agent
  B --> P1["Agentic Architecture Patterns<br/>8 design decisions"]:::gate
  P1 --> P2["Teardowns: Systems I Built<br/>real systems, read from source"]:::orch
  P2 -. "confirms or breaks" .-> P1
  P1 -. "reference material" .-> R1["Infrastructure<br/>and Tooling"]:::external
  P1 -. "reference material" .-> R2["Running in<br/>Production"]:::external
  R3["The Frontier"]:::external
```

<p className="fig-caption"><strong>Figure 0.1</strong> — The book is the two middle boxes. Foundations and Building Blocks lead into it; the three reference sections hang off it, entered from a chapter's own links rather than read in sequence.</p>

## Three ways to read it

### The book, in order

[How LLMs Work](./foundations/how-llms-work) → the [Agentic Building Blocks](./core-building-blocks/) pages, in their listed order → the [Agentic Architecture Patterns](./patterns/) chapters in order → the [Teardowns](./teardowns/). Honestly: a weekend, not an afternoon.

### Show me a real system first

If you distrust pattern language until you've seen it survive contact with code: start at the [ProjectOS teardown](./teardowns/projectos), then read the chapters it anchors most directly — [Stage Gates](./patterns/stage-gates) and [Orchestration vs. Autonomy](./patterns/orchestration-vs-autonomy) — then work outward from there into the rest of the patterns.

### Something is already wrong with the system I have

The highest-value path if you're not here to learn in the abstract:

- Surprise bill → Cost-Aware Model Routing
- Bad output shipped to a user → [Stage Gates](./patterns/stage-gates) + QA Pipelines
- Can't tell what state a run is in, or can't resume one → State Machines
- Got prompt-injected → Failure Modes and Attack Surfaces
- Reviewers flag everything and nothing ships → LLM-as-Judge

### Thirty minutes

[Stage Gates](./patterns/stage-gates) end to end, then the first three figures of the [ProjectOS teardown](./teardowns/projectos). More chapters are landing; this path gets longer as they do.

---

## What's in the reference library

| Section | What it covers | Pages |
|------|------|------|
| [AI Foundations](./foundations/) | How a model works under the hood — one required page, nine optional | 10 |
| [Agentic Building Blocks](./core-building-blocks/) | Prompting, RAG, tool use, memory, multi-agent — the pieces the patterns chapters assume | 8 |
| [Infrastructure and Tooling](./meta-infrastructure/) | Evals, observability, guardrails, vector DBs, MCP, and more — entered from a chapter | 18 |
| [Running in Production](./production-concerns/) | Reliability, monitoring, graceful degradation, deployment patterns | 6 |
| [The Frontier](./aspirational/) | Where the field is heading, confidence levels stated explicitly | 6 |

Cross-references throughout the site use `[[WikiLink]]` syntax pointing to related pages. Every reference page has a "Going deeper" section with papers, docs, and internal cross-references.
