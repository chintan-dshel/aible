---
sidebar_position: 19
title: Audit Logs
description: What to log, immutability requirements, PII scrubbing, and retention policies for AI systems.
---

# Audit Logs

## What it is

Audit logs for AI systems are append-only records of every significant event: what was sent to the model, what it returned, what tools it called, what decisions it made, and any errors or policy violations. Unlike application logs optimized for debugging, audit logs are optimized for accountability — they must be tamper-evident, retained according to compliance requirements, and queryable after the fact.

The distinction matters: a structured log that gets rotated and overwritten is a debugging aid. An audit log that is immutable, retained for years, and tied to user identity is an accountability record. Most AI systems need both, and they have different storage, access control, and retention requirements.

Audit logs often have legal standing. HIPAA (a US law setting requirements for handling health information) requires 6-year retention for covered entities. SOX (Sarbanes-Oxley, a US law governing financial reporting) mandates 7 years for financial records. GDPR (the EU's data-protection law)'s right of access means you must be able to produce records about data subjects on request. A log that can be modified or deleted after the fact cannot serve as a compliance record — which is why immutability is a hard requirement, not a best practice. Immutability must be enforced at the storage layer: a mutable database table where the application can run DELETE or UPDATE is not an audit log, regardless of application-level conventions.

## The problem it solves

LLM behavior is probabilistic and opaque. When something goes wrong — a harmful output, a data leak, a tool call that deleted the wrong record — you need to answer:

- What was the model sent? (exact system prompt + user message)
- What did it return? (exact output, before any post-processing)
- What tools did it call, with what arguments, and what did they return?
- Which user triggered this, and when?
- What guardrail decisions were made, and why?

Without structured audit logs, incident response is guesswork. You may have application logs that record "model call made" but not the actual content — leaving you unable to reconstruct the failure.

There's also the compliance dimension. Healthcare systems (HIPAA), financial systems (SOX, MiFID II — an EU financial-markets regulation), and legal systems have mandatory record-keeping requirements. AI systems operating in these domains must demonstrate that every significant decision can be audited.

## How it works under the hood

### What to log

Every AI audit log entry should capture a minimum set of fields:

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
import hashlib
import json

@dataclass
class AuditEvent:
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # uuid: a
    # randomly generated, effectively-unique ID
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str = ""
    session_id: str = ""
    request_id: str = ""

    # Model call
    model: str = ""
    system_prompt_hash: str = ""   # hash (a scrambled, fixed-length fingerprint
    # of the data, not the data itself) only — don't log the raw system prompt in
    # hot storage (the fast, frequently-queried tier described below)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    # Content (scrubbed)
    user_message_scrubbed: str = ""   # PII (personally identifiable
    # information -- names, emails, SSNs, and the like) -scrubbed version
    response_scrubbed: str = ""

    # Tool calls
    tool_calls: list = field(default_factory=list)  # [{"tool": "...", "args_hash": "..."}]

    # Policy decisions
    guardrail_triggered: bool = False  # guardrail: a check run on input or
    # output to catch unsafe or policy-violating content
    guardrail_action: str = ""   # "block", "redact", "warn", ""
    validation_passed: bool = True

    # Cost
    estimated_cost_usd: float = 0.0

    # Error
    error: str = ""
    error_type: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))
```

### PII scrubbing

Never log raw user content in long-term audit storage. Scrub PII before writing.

**Important limitation:** regex (a text-pattern matcher)-based scrubbing catches known patterns (email format, US phone format, credit card format) but will miss names in running text, non-standard formatting, and non-English/non-US PII formats. These patterns are US-centric and illustrative — for production systems handling international users, use a dedicated library like Microsoft Presidio or Google DLP, and validate against your actual user population's data formats. Regex scrubbing is a complement to access controls, not a substitute.

**Scrubbing the log record does not scrub the value in memory.** The raw `user_message` string is still on the call stack while the API call is in flight. APM (application performance monitoring) agents (Sentry, Datadog, OpenTelemetry) — background tools that watch your running application and record what happened — may capture it from exception frames (the snapshot of local variables saved when an error is thrown) or distributed traces (the record of a single request's path across multiple services). Scrub at ingestion — before passing the value to any function — rather than only before writing the log record.

```python
import re

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),               # SSN
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD]"),  # Credit card
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
]

def scrub_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def hash_content(text: str) -> str:
    """One-way hash for deduplication and integrity checks without storing content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

### Audit logger

A structured logger that writes to append-only storage:

```python
import anthropic
import time
import json
from pathlib import Path

client = anthropic.Anthropic()

PRICING = {
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
}

class AuditLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: AuditEvent) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

    def call_with_audit(
        self,
        user_id: str,
        session_id: str,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-6",
    ) -> str:
        request_id = str(uuid.uuid4())
        start = time.monotonic()

        event = AuditEvent(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            model=model,
            system_prompt_hash=hash_content(system_prompt),
            user_message_scrubbed=scrub_pii(user_message),
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            response_text = response.content[0].text

            prices = PRICING.get(model, {"input": 0, "output": 0})
            event.input_tokens = response.usage.input_tokens
            event.output_tokens = response.usage.output_tokens
            event.estimated_cost_usd = (
                response.usage.input_tokens * prices["input"] +
                response.usage.output_tokens * prices["output"]
            )
            event.response_scrubbed = scrub_pii(response_text)
            event.latency_ms = (time.monotonic() - start) * 1000

            return response_text

        except Exception as e:
            event.error = str(e)
            event.error_type = type(e).__name__
            event.latency_ms = (time.monotonic() - start) * 1000
            raise

        finally:
            self.log(event)

logger = AuditLogger("logs/audit.jsonl")
response = logger.call_with_audit(
    user_id="user_123",
    session_id="session_abc",
    system_prompt="You are a helpful assistant.",
    user_message="What is the return policy?",
)
```

### Tool call audit trail

When agents call tools, log every call with its arguments and result hash:

```python
def audit_tool_call(
    event: AuditEvent,
    tool_name: str,
    args: dict,
    result: str,
) -> None:
    event.tool_calls.append({
        "tool": tool_name,
        "args_hash": hash_content(json.dumps(args, sort_keys=True)),
        "result_hash": hash_content(result),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
```

### Retention tiers

Different data has different retention and access requirements. Hot, warm, and cold are shorthand for storage tiers with different cost and access-latency tradeoffs:

- **Hot**: fast, expensive storage queried frequently — recent events in a live database or fast object store. Typically the last 30 days.
- **Warm**: cheaper, slower storage for data still occasionally accessed — compressed in object storage. Typically 30 days to 1–2 years.
- **Cold**: archival, very cheap but slow to retrieve — Glacier-class storage for regulatory retention minimums. Access measured in hours, not milliseconds.

Retention periods are jurisdiction- and regulation-specific — consult legal counsel for your context. Do not treat any duration listed here as a compliance target.

:::caution[Hot tier must not be local disk for compliance workloads]

Local disk is not append-only, not tamper-evident, and lost on instance termination. For any regulated workload, hot-tier audit logs must go to a durable, append-only sink: an object store with write-once policies, a managed log service, or a database with audited append-only access controls.

:::

```python
RETENTION_POLICY = {
    "hot": {
        "duration_days": 30,
        "storage": "write-once object storage or managed log service (NOT local disk for compliance workloads)",
        "content": "scrubbed structured events",
        "access": "on-call engineers for incident response",
    },
    "warm": {
        "duration_days": 365,
        "storage": "compressed object storage (S3, GCS -- Amazon and Google's file-storage cloud services) with retention policy",
        "content": "scrubbed structured events (same as hot, older)",
        "access": "compliance team, legal holds",
    },
    "cold": {
        "duration_days": 2557,  # example only — determine from legal/compliance requirements
        "storage": "archival (Glacier -- AWS's cheap, slow-to-retrieve storage tier, cold storage) with object lock",
        "content": "aggregates + metadata only; no raw content",
        "access": "legal requests, regulatory audits only",
    },
}
```

## Concrete example

A complete audit logging setup for a multi-tool AI customer support agent:

```python
import anthropic
import json
import time
import uuid
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

client = anthropic.Anthropic()

PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD]"),
]

def scrub(text: str) -> str:
    for pattern, repl in PII_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text

def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up order status by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }
]

def handle_tool(tool_name: str, args: dict) -> str:
    if tool_name == "lookup_order":
        return json.dumps({"order_id": args["order_id"], "status": "shipped", "eta": "2026-04-25"})
    return "Unknown tool"

def run_support_query(user_id: str, session_id: str, user_message: str) -> str:
    log_path = Path("logs/audit.jsonl")
    log_path.parent.mkdir(exist_ok=True)

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "user_message_scrubbed": scrub(user_message),
        "tool_calls": [],
        "model": "claude-sonnet-4-6",
        "guardrail_triggered": False,
        "error": None,
    }

    system = "You are a customer support agent. Use tools to look up order information."
    messages = [{"role": "user", "content": user_message}]
    start = time.monotonic()

    try:
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = handle_tool(block.name, block.input)
                        event["tool_calls"].append({
                            "tool": block.name,
                            "args_hash": sha(json.dumps(block.input, sort_keys=True)),
                            "result_hash": sha(result),
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                event["response_scrubbed"] = scrub(final_text)
                event["input_tokens"] = response.usage.input_tokens
                event["output_tokens"] = response.usage.output_tokens
                event["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
                break

    except Exception as e:
        event["error"] = str(e)
        event["error_type"] = type(e).__name__
        raise

    finally:
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    return event.get("response_scrubbed", "")

result = run_support_query("user_42", "sess_99", "Where is my order ORD-8821?")
print(result)
```

## When to use it / when not to

#### Invest in audit logging when

- Your system handles actions with real-world consequences: transactions, file writes, emails, database modifications
- Compliance requirements exist (HIPAA, SOX, GDPR right-of-access) that mandate record retention
- Multiple users share the system and you need to attribute actions to individuals
- You're deploying agents with tools — every tool call needs a traceable record
- Incident response capability matters — you need to reconstruct what happened after a failure

#### Lighter logging suffices when

- The system is single-user internal tooling with no sensitive data
- Every output is reviewed by a human before any action is taken — the human is the audit trail
- You're in early prototyping where the system prompt and behavior change daily

#### The practical question

If a regulator, legal counsel, or your CISO asked "show me every action this system took for user X over the past year," could you answer? If not, audit logging is under-invested.

:::tip[My take]

The most common audit logging mistake is logging content instead of hashes for sensitive fields. Raw user messages in long-term storage create GDPR deletion problems — when a user requests deletion, you have to find and redact every occurrence in your logs. Store scrubbed content in hot logs and one-way hashes in warm/cold storage. If you ever need the original (for a legal hold), encrypt and store it separately under strict access control.

The second most common mistake is treating audit logs as append-only in name only. A writable log file that an application process can overwrite isn't an audit log. For real immutability, write to an object store with object lock enabled (S3 Object Lock, GCS retention policy), or use a write-once database. This is the difference between a log and an audit log in any compliance context.

Tool call auditing is more important than LLM output auditing for agents. The LLM output is a plan; the tool calls are the actions. A log that captures "the model decided to delete the file" but not "the file was actually deleted" is insufficient. Log tool call arguments, results, and timestamps as first-class audit events.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| Python `logging` / `structlog` | Structured JSON logging to files or streams |
| AWS S3 Object Lock / GCS Retention | True immutability for audit log storage |
| OpenTelemetry | Distributed tracing; trace IDs link audit events across service boundaries |
| Langfuse | LLM observability platform; captures prompts, completions, latency, cost |
| `presidio` (Microsoft) | PII detection and anonymization before logging |
| Elasticsearch / OpenSearch | Queryable log storage for incident response and audit queries |
| PostgreSQL with audit triggers | Relational audit log for structured event data with user attribution |

## Common failure modes and gotchas

**1. Logging raw PII in long-term storage.** User emails, phone numbers, and health data in plain-text logs create GDPR, HIPAA, and CCPA (California's consumer-privacy law) obligations. Scrub or pseudonymize before writing to any log that persists beyond 24 hours.

**2. Writable audit logs.** If your application process can overwrite or delete log files, they're not audit logs. Use append-only log targets — object storage with object lock, or a database with immutable rows.

**3. Missing request correlation.** Without a request_id or session_id that flows through every event, you can't reconstruct multi-step agent sequences. Add correlation IDs at request entry and propagate them through every downstream call.

**4. System prompt in hot logs.** System prompts are often proprietary. Logging them verbatim leaks intellectual property to anyone with log access. Log the hash and store the versioned prompt separately in a secrets-controlled location.

**5. No retention policy enforcement.** Logs that grow indefinitely create storage costs and compliance risk (retaining data longer than policy requires is itself a liability). Automate deletion at retention boundaries — don't leave it to manual cleanup.

**6. Audit logs in the same failure domain as the application.** If the application goes down, audit log writes should still succeed. Write audit events to a separate, durable sink (separate disk, object store, or message queue) that the application cannot corrupt.

**7. Forgetting failed calls.** A failed API call still consumed tokens if it got a response before failing. Log every API call attempt, including failures, with whatever partial information is available.

## Project ideas

**1. Audit log schema design** — For one of your existing AI applications, design a complete audit log schema. List every event type, every field, the PII handling decision for each field, the retention tier, and who should have access at each tier. Write it as a data dictionary. This is the design artifact compliance teams ask for.

**2. PII scrubbing pipeline** — Build a PII scrubbing function that handles 5 PII types (email, phone, SSN, credit card, name). Test it on 50 real-looking examples. Measure: false positive rate (legitimate content scrubbed) and false negative rate (PII that passes through). Use `presidio` to compare against your regex approach.

**3. Immutable log sink** — Set up an S3 bucket with Object Lock in compliance mode. Write a logger that uses boto3 to write audit events as individual objects with a TTL tag. Test that a programmatic delete attempt fails. This is the minimal viable immutable audit log for an AWS environment.

**4. Incident reconstruction exercise** — Build a small agent that calls 2–3 tools. Run it 10 times with different inputs. Then: delete the application logs (keep only audit logs), pick one run at random, and reconstruct exactly what happened using only the audit log. If you can't reconstruct it, identify which fields were missing.

## Going deeper

#### Foundational reading

- NIST SP 800-92, "Guide to Computer Security Log Management" — the reference for log management policy, including retention, protection, and analysis. Framework-agnostic; applies directly to AI systems.
- GDPR Article 30 (Records of Processing Activities) and Article 17 (Right to Erasure) — the specific obligations that drive scrubbing-before-storage patterns in EU-facing systems.

#### Tools

- `structlog` documentation (structlog.org) — the reference for structured logging in Python; covers JSON formatting, processor pipelines, and context variables for correlation IDs.
- `presidio` (github.com/microsoft/presidio) — Microsoft's PII detection and anonymization library; supports custom recognizers for domain-specific identifiers.
- AWS S3 Object Lock documentation — reference for immutable storage; covers compliance mode vs. governance mode and the difference between object TTL and deletion protection.
