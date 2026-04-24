---
sidebar_position: 3
title: Long-Horizon Agents
description: Planning, interruption, checkpointing, and trust — the hard problems in getting agents to complete multi-day tasks.
---

# Long-Horizon Agents

An agent that takes a task and runs for hours or days without continuous supervision — planning, adapting to failures, persisting state across sessions, and knowing when to stop and ask.

## The problem it solves

Most LLM applications are single-turn or short-session. A long-horizon agent must handle what those can't: a context window that fills up over a long task, errors that compound if not caught, irreversible actions that require confirmation, and the need to resume from where it left off after a restart.

The target use cases are research pipelines, code generation tasks that span files and repositories, automated data processing jobs, and anything that a skilled contractor would bill days of time for.

## How it works under the hood

**Hierarchical planning.** Break the task into milestones before executing any step. The agent maintains a high-level plan (5–10 milestones) and a local step queue (2–3 next actions). After each milestone completes, it re-evaluates the plan against the current state.

**Checkpointing.** Serialize the full agent state (plan, completed steps, discovered artifacts, accumulated context summary) after each milestone. A checkpoint enables: resuming after a crash, rolling back to a previous known-good state, and human review at defined intervals.

**Context management.** Context windows fill up. After every N steps, summarize what has been accomplished into a compact representation and drop the raw turn history. The agent operates on this rolling summary rather than the full transcript. Key artifacts (code written, files created, decisions made) are stored externally and retrieved as needed — see [[RAG]] for retrieval patterns and [[Memory Architectures]] for how to structure persistent agent state.

**Human-in-the-loop gates.** Before any irreversible action (deleting files, sending emails, committing to a branch, making an API call with side effects), pause and request confirmation. The agent presents: "I'm about to do X because Y. Confirm?" The human can approve, redirect, or abort.

**Dead-man's switch.** If the agent hasn't reported progress in N minutes, alert the operator. This catches infinite loops, stuck states, and runaway API spending before they become catastrophic.

## Concrete example

```python
import anthropic
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

client = anthropic.Anthropic()


@dataclass
class AgentState:
    task: str
    plan: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    context_summary: str = ""
    artifacts: dict = field(default_factory=dict)
    step_count: int = 0
    checkpoint_path: str = ""

    def save(self) -> None:
        if self.checkpoint_path:
            Path(self.checkpoint_path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str) -> "AgentState":
        # Security: validate `path` before use — a stored checkpoint path can enable
        # path traversal if it came from an untrusted source. Validate the deserialized
        # content before trusting it in prompts: a tampered checkpoint is an injection vector.
        data = json.loads(Path(path).read_text())
        return cls(**data)


# Actions that require human confirmation before execution.
# Important: this gate works because the agent emits action names as structured text
# that we parse before executing. If the agent calls tools directly (MCP, function
# calling), the gate must be enforced at the tool execution layer, not by text parsing.
GATED_ACTIONS = {"delete_file", "send_email", "git_commit", "api_call_with_side_effects"}


def requires_confirmation(action_name: str) -> bool:
    return action_name in GATED_ACTIONS


def request_confirmation(action_name: str, description: str) -> bool:
    print(f"\n[CONFIRMATION REQUIRED]\nAction: {action_name}\nDescription: {description}")
    response = input("Approve? [y/N]: ").strip().lower()
    return response == "y"


def summarize_progress(state: AgentState) -> str:
    """Compress completed steps into a running summary using the LLM."""
    if not state.completed:
        return ""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Summarize the following completed steps for an agent working on: {state.task}\n\n"
                f"Completed steps:\n" + "\n".join(f"- {s}" for s in state.completed[-20:]) +
                f"\n\nExisting summary: {state.context_summary}\n\n"
                f"Write a compact paragraph (under 200 words) capturing what is done and what matters."
            ),
        }],
    )
    return response.content[0].text


def generate_plan(state: AgentState) -> list[str]:
    """Generate or refresh a milestone plan given the current state."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {state.task}\n\n"
                f"Progress so far: {state.context_summary or 'Not started'}\n"
                f"Completed steps: {len(state.completed)}\n\n"
                f"Generate a numbered list of 5–8 milestones remaining. "
                f"Each milestone should be one line. Reply with only the numbered list."
            ),
        }],
    )
    lines = response.content[0].text.strip().splitlines()
    return [l.lstrip("0123456789. ").strip() for l in lines if l.strip()]


def execute_next_step(state: AgentState) -> tuple[str, bool]:
    """
    Ask the agent what to do next and return (step_description, is_done).
    In a real agent this would call tools; here we simulate with a text step.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "You are a long-horizon task agent. Execute tasks step by step. "
            "When a step requires a gated action (delete, send, commit), say: "
            "GATED_ACTION:<action_name>: <description>. "
            "When the task is complete, say: DONE: <summary>."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Task: {state.task}\n"
                f"Plan: {chr(10).join(state.plan)}\n"
                f"Progress: {state.context_summary or 'Not started'}\n\n"
                f"Execute the next step. Be specific about what you're doing."
            ),
        }],
    )
    text = response.content[0].text.strip()

    if text.startswith("DONE:"):
        return text, True

    if text.startswith("GATED_ACTION:"):
        _, _, rest = text.partition(":")
        action_name, _, description = rest.strip().partition(":")
        action_name = action_name.strip()
        if not requires_confirmation(action_name) or request_confirmation(action_name, description.strip()):
            return f"Executed gated action: {action_name}", False
        else:
            return f"Gated action '{action_name}' rejected by operator — agent will find alternative", False

    return text, False


def run_long_horizon_agent(
    task: str,
    checkpoint_path: str = "agent_checkpoint.json",
    max_steps: int = 50,
    summarize_every: int = 10,
) -> str:
    # Resume from checkpoint if available
    if Path(checkpoint_path).exists():
        state = AgentState.load(checkpoint_path)
        print(f"[RESUME] Loaded checkpoint at step {state.step_count}")
    else:
        state = AgentState(task=task, checkpoint_path=checkpoint_path)
        state.plan = generate_plan(state)
        state.save()
        print(f"[START] Plan: {len(state.plan)} milestones")

    # Track cumulative token usage and surface it in checkpoint logs.
    # A 50-step agent at 3 API calls/step can exceed budget before you notice.
    # See [[Cost Tracking]] for per-run budget enforcement patterns.
    last_progress_time = time.time()

    for _ in range(max_steps):
        step_result, is_done = execute_next_step(state)
        state.completed.append(step_result)
        state.step_count += 1
        last_progress_time = time.time()

        print(f"[STEP {state.step_count}] {step_result[:100]}")

        if is_done:
            state.save()
            return step_result

        # Compress context every N steps
        if state.step_count % summarize_every == 0:
            state.context_summary = summarize_progress(state)
            state.plan = generate_plan(state)  # refresh plan given progress
            state.completed = []  # old steps are captured in the summary
            state.save()
            print(f"[CHECKPOINT] Saved at step {state.step_count}")

        # Dead-man's switch — illustrative only. In a synchronous loop,
        # last_progress_time is reset on every iteration, so this condition
        # can never trigger mid-step. In production, run this check in a
        # separate monitoring thread that tracks wall-clock time independently.
        if time.time() - last_progress_time > 300:  # 5 minutes
            print("[ALERT] No progress in 5 minutes — manual review required")
            break

    return f"Reached step limit ({max_steps}) — checkpoint saved for resumption"
```

The agent checkpoints every 10 steps, summarizes completed work to manage context, gates irreversible actions through human confirmation, and can resume from a crash at the last checkpoint.

## When to use it / when not to

**Use when:**
- The task is too large for a single context window (multi-file code refactors, long research pipelines)
- The task involves irreversible actions that need human review at key points
- The task needs to survive process restarts (overnight jobs, multi-day pipelines)
- A skilled human would need hours to days to complete it manually

**Don't use when:**
- Real-time response is required — long-horizon agents are slow
- The task can be done in a single API call or short session
- You have no way to verify intermediate results — blind multi-day execution is high-risk

**Current state of readiness.** Long-horizon agent reliability degrades with task length. Error rates compound; a 50-step task with 95% per-step success completes cleanly only 8% of the time without recovery logic. Human checkpoints and strong error detection are non-negotiable, not optional.

## Main tools and libraries

| Tool | Role |
|------|------|
| LangGraph | Stateful agent graphs with built-in checkpointing (SQLite or Postgres) |
| Temporal | Production-grade workflow orchestration with durable execution, retries, and visibility |
| Prefect / Airflow | Workflow DAGs with monitoring — good for data pipeline agents |
| CrewAI | Multi-agent framework with role-based task decomposition |
| `sqlite3` / PostgreSQL | Durable state storage for checkpoints |

Temporal is the right choice when you need guarantees: durable execution that survives process crashes, automatic retries with backoff, and built-in observability. For simpler agents, LangGraph's checkpointing is easier to set up.

## Common failure modes and gotchas

**Error propagation.** A wrong step early in a long task corrupts everything downstream. The agent needs to validate its own output at each milestone — "does the current state match what I expected after this step?" — and fail loudly rather than proceeding on a wrong foundation.

**Oscillation.** Without a stopping condition, agents loop: try step A, fail, try alternative, fail, try A again. Add: "if you've tried the same approach twice and failed, stop and report rather than retrying." Track attempts per step.

**Runaway API spending.** A 50-step agent making 3 API calls per step at $15/Mtok can cost more than expected before you notice. Set per-run budget limits and surface token counts in checkpoint logs — see [[Cost Tracking]] for enforcement patterns.

**Context amnesia.** After summarization, the agent may forget specific decisions it made earlier that have implications for later steps. Store key decisions and constraints explicitly in a `artifacts["decisions"]` list that survives summarization.

**Irreversible mistakes without gates.** An agent that calls a payment API, sends an email blast, or deletes a database record before confirming will cause real-world damage. Every action with external side effects must be gated by default, not as an afterthought.

## Project ideas

- **Research assistant**: given a research question, plan a multi-step investigation — search, read, synthesize, write — with checkpoints between major phases and a human review before the final output is published.
- **Codebase refactor agent**: given a refactor spec, enumerate affected files, plan the changes, apply them file by file with test runs between batches, stopping for human review if tests fail.
- **Data pipeline constructor**: given a data source and desired output schema, plan and build an ETL pipeline — discover schema, write transformations, validate with sample data, generate tests — persisting state so the pipeline can be resumed if any step fails.

## Going deeper

- [LangGraph — checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/) — durable state for stateful agents
- [Temporal documentation](https://docs.temporal.io/) — production-grade durable execution
- [SWE-bench](https://www.swebench.com/) — benchmark for long-horizon coding agents; tracks the state of the art
- [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) — long-horizon embodied agent in Minecraft with skill accumulation
- [[Agentic Computer Use]] — the GUI action layer that long-horizon agents use to operate software
- [[Access Control for Agents]] — least-privilege and permission scoping for agents with real-world capabilities
