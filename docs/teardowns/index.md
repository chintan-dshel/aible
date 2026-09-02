---
sidebar_position: 0
title: 'Teardowns: Systems I Built'
description: Real agentic systems, read from source and taken apart the same way each time.
---

# Teardowns: Systems I Built

ProjectOS and Lyceum are the two systems with enough real application code to take apart the same way each time:

1. **What it does** — the product, in plain terms
2. **Architecture** — components and boundaries
3. **Control flow and data flow** — who decides what runs next, and what moves where
4. **Design decisions and trade-offs** — what was chosen and what it cost
5. **Strengths**
6. **Weaknesses** — verified against the code, not inferred
7. **How it could be attacked or manipulated**
8. **What I'd change**

Second Brain gets a shorter case study instead of the full eight sections — most of what runs it is instructions a Claude session re-reads each time, not code, so there's less architecture to take apart and more to say about what that shape itself implies.

Every claim comes from reading the source at a named commit, or, where a system isn't version-controlled, from the source on disk on a named date. Anything inferred rather than read is tagged `[VERIFY]`.

| System | What it is | Patterns it anchors |
|---|---|---|
| [ProjectOS](./projectos) | Stage-gated multi-agent project manager with a production LLM judge | Stage gates, state machines, LLM-as-judge, model routing |
| [Lyceum](./lyceum) | AI university simulator with a nine-agent, spec-then-review-then-generate content pipeline | Stage gates, state machines, QA pipelines, model routing |
| [Second Brain](./second-brain) | A personal knowledge wiki maintained by an LLM re-reading instructions, with no persisted state machine of its own | Memory layers, state machines |
