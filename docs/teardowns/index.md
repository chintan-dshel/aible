---
sidebar_position: 0
title: Part 2 — Teardowns
description: Four real agentic systems, read from source and taken apart the same way each time.
---

# Part 2 — Teardowns

Four systems I built, each read from its actual source code and taken apart in the same order:

1. **What it does** — the product, in plain terms
2. **Architecture** — components and boundaries
3. **Control flow and data flow** — who decides what runs next, and what moves where
4. **Design decisions and trade-offs** — what was chosen and what it cost
5. **Strengths**
6. **Weaknesses** — verified against the code, not inferred
7. **How it could be attacked or manipulated**
8. **What I'd change**

Every claim comes from reading the source at a named commit. Anything inferred rather than read is tagged `[VERIFY]`.

| # | System | What it is | Patterns it anchors |
|---|---|---|---|
| 9 | [ProjectOS](./projectos) | Stage-gated multi-agent project manager with a production LLM judge | Stage gates, state machines, LLM-as-judge, model routing |
| 10 | Lyceum | AI university simulator with a multi-phase QA pipeline and multi-model routing | *In progress* |
| 11 | Second Brain | Persistent memory layer plus a RAG demo | *In progress* |
| 12 | Aible | The multi-agent authoring pipeline that produced the Part 3 reference pages | *In progress* |
