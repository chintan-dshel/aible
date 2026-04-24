---
sidebar_position: 20
title: Access Control for Agents
description: Least-privilege design, OAuth for agents, and scoping permissions — so agents can't do more than they should.
---

# Access Control for Agents

## What it is

Access control for AI agents is the practice of explicitly limiting what an agent can do — which tools it can call, which data it can read, which actions it can take — to the minimum required for its assigned task. It applies the principle of least privilege to AI systems: an agent that only needs to read documents should not have write access; an agent that processes one user's data should not be able to access another's.

This is distinct from guardrails, which limit what outputs the model generates. Access control operates at the action layer — enforcing at the tool and API level what the agent is permitted to do, regardless of what the model "decides" to do.

## The problem it solves

Agents with tools are capable of causing real-world harm. An agent that can read files, query databases, send emails, and execute code — without access restrictions — becomes a high-value target for prompt injection, jailbreaks, and misuse:

- A prompt injection in a retrieved document tells the agent to exfiltrate user data via email
- A jailbroken agent uses its code execution tool to download and run malware
- A bug in the agent's routing logic causes it to call a delete endpoint instead of a read endpoint
- An agent processing user A's request accidentally accesses user B's data because tenant isolation is missing

The consequences scale with capability. A weak chatbot that outputs wrong text causes confusion; a capable agent with unrestricted tool access causes data breaches, financial transactions, and irreversible file deletions.

Access control is the last line of defense when the model itself is compromised or manipulated.

## How it works under the hood

### Capability registries

Define what tools each agent role is permitted to use. Check permissions before execution, not after:

```python
from dataclasses import dataclass, field
from typing import Callable
import anthropic
import json

@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable
    required_permission: str

TOOL_REGISTRY: dict[str, Tool] = {}

def register_tool(required_permission: str):
    def decorator(fn: Callable) -> Callable:
        # The decorator registers metadata. input_schema.properties is intentionally
        # left empty here — populate it per-tool with parameter names, types, and
        # descriptions. An empty schema causes the LLM to receive no parameter
        # guidance and will silently degrade tool selection and calling accuracy.
        schema = {
            "name": fn.__name__,
            "description": fn.__doc__ or "",
            "input_schema": {
                "type": "object",
                "properties": {},  # populate this for each tool
                "required": [],
            }
        }
        TOOL_REGISTRY[fn.__name__] = Tool(
            name=fn.__name__,
            description=fn.__doc__ or "",
            schema=schema,
            handler=fn,
            required_permission=required_permission,
        )
        return fn
    return decorator

AGENT_PERMISSIONS: dict[str, set[str]] = {
    "reader": {"read_document", "search_documents"},
    "writer": {"read_document", "search_documents", "create_document", "update_document"},
    "admin": {"read_document", "search_documents", "create_document", "update_document", "delete_document"},
    "support": {"read_document", "search_documents", "lookup_order", "send_message"},
}

@register_tool("read_document")
def read_document(path: str) -> str:
    """Read a document by path."""
    return f"[content of {path}]"

@register_tool("delete_document")
def delete_document(path: str) -> str:
    """Permanently delete a document."""
    return f"[deleted {path}]"

@register_tool("send_message")
def send_message(to: str, body: str) -> str:
    """Send an email message."""
    return f"[sent to {to}]"

class AccessControlledAgent:
    def __init__(self, role: str, user_id: str):
        self.role = role
        self.user_id = user_id
        self.permissions = AGENT_PERMISSIONS.get(role, set())
        self.client = anthropic.Anthropic()

    def permitted_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema["input_schema"],
            }
            for t in TOOL_REGISTRY.values()
            if t.required_permission in self.permissions
        ]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            return "Error: tool not found"
        if tool.required_permission not in self.permissions:
            # Return generic message — detailed role/permission info is logged
            # server-side only and never returned to the caller (privilege enumeration risk).
            return "Error: permission denied"
        # LLM tool arguments are untrusted input. Validate against schema before
        # passing to the handler — especially for tools that touch the filesystem,
        # execute code, or make external calls.
        return tool.handler(**args)

    def run(self, system: str, user_message: str) -> str:
        tools = self.permitted_tools()
        messages = [{"role": "user", "content": user_message}]

        max_iterations = 10
        for _ in range(max_iterations):
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=system,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                return next((b.text for b in response.content if hasattr(b, "text")), "")

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "Error: maximum iterations reached"

reader_agent = AccessControlledAgent(role="reader", user_id="user_123")
print("Reader tools:", [t["name"] for t in reader_agent.permitted_tools()])
# ['read_document', 'search_documents']
```

### Tenant isolation

In multi-tenant systems, prevent agents from accessing one user's data when processing another's. A tenant is a distinct customer or organization sharing the same infrastructure — tenant isolation means one tenant's agent cannot read, write, or act on another tenant's data, even when both run the same code.

Tenant ID must be injected server-side at execution time and must overwrite any LLM-supplied value of the same key. Convention-based namespacing (prefixing with `_`) is not a security boundary — a prompt-injected LLM can emit `_tenant_id` as a tool argument:

```python
class TenantScopedAgent(AccessControlledAgent):
    def __init__(self, role: str, user_id: str, tenant_id: str):
        super().__init__(role, user_id)
        self.tenant_id = tenant_id

    def execute_tool(self, tool_name: str, args: dict) -> str:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            return f"Error: tool not found"
        if tool.required_permission not in self.permissions:
            return f"Error: permission denied"

        # Inject tenant scope — overwrite any LLM-supplied value so the agent
        # cannot override it via prompt injection.
        args["_tenant_id"] = self.tenant_id
        args["_user_id"] = self.user_id

        # Validate args before passing to handler: LLM output is untrusted input.
        # At minimum, check types and lengths against the declared schema.
        return tool.handler(**args)
```

### Scoped credentials

Scoped credentials are access tokens or API keys restricted to a specific set of actions or resources — a token that can only read from one database, or an API key that can only send (not read) email. The agent never holds a master key; it holds only the minimum credential for its current task.

Agents should receive credentials scoped to the minimum access required, not admin credentials:

```python
import os

def get_scoped_credentials(agent_role: str, tenant_id: str) -> dict:
    """
    Return credentials appropriate for the agent role.
    Never give agents admin credentials — create scoped tokens per role.
    """
    if agent_role == "reader":
        return {
            "db_connection_string": os.environ["DB_READONLY_URL"],
            "storage_token": os.environ["STORAGE_READ_TOKEN"],
            "api_key": os.environ["API_LIMITED_KEY"],
        }
    elif agent_role == "support":
        return {
            "db_connection_string": os.environ["DB_READONLY_URL"],
            "storage_token": os.environ["STORAGE_READ_TOKEN"],
            "email_token": os.environ["EMAIL_SEND_ONLY_TOKEN"],
        }
    else:
        raise ValueError(f"Unknown role: {agent_role}. Add it explicitly — don't default to admin.")
```

### Human confirmation for destructive actions

Irreversible or high-impact actions should require explicit human confirmation before execution:

```python
REQUIRES_CONFIRMATION = {"delete_document", "send_message", "execute_code", "transfer_funds"}

class ConfirmationGatedAgent(AccessControlledAgent):
    def __init__(self, role: str, user_id: str, auto_confirm: bool = False):
        super().__init__(role, user_id)
        self.auto_confirm = auto_confirm  # False in production; True only in tests

    def execute_tool(self, tool_name: str, args: dict) -> str:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            return "Error: tool not found"
        if tool.required_permission not in self.permissions:
            return "Error: permission denied"

        if tool_name in REQUIRES_CONFIRMATION and not self.auto_confirm:
            print(f"\n[CONFIRMATION REQUIRED]")
            print(f"  Tool: {tool_name}")
            print(f"  Args: {json.dumps(args, indent=2)}")
            # input() is for CLI/local use only. For web deployments, replace with
            # an async approval queue: pause execution, send a confirmation request
            # (Slack, webhook, UI modal), and resume when the callback arrives.
            confirm = input("  Approve? (yes/no): ").strip().lower()
            if confirm != "yes":
                return f"Action '{tool_name}' cancelled by user."

        return tool.handler(**args)
```

## Concrete example

A complete access-controlled support agent that enforces role, tenant isolation, and confirmation gating:

```python
import anthropic
import json
import os
from datetime import datetime, timezone

client = anthropic.Anthropic()

# Simulated DB per tenant
TENANT_DATA = {
    "tenant_A": {"orders": {"ORD-001": {"status": "shipped"}, "ORD-002": {"status": "pending"}}},
    "tenant_B": {"orders": {"ORD-100": {"status": "delivered"}}},
}

ROLE_PERMISSIONS = {
    "support_read": {"lookup_order"},
    "support_full": {"lookup_order", "cancel_order"},
}

DESTRUCTIVE = {"cancel_order"}

def lookup_order(order_id: str, tenant_id: str, **_) -> str:
    orders = TENANT_DATA.get(tenant_id, {}).get("orders", {})
    order = orders.get(order_id)
    if not order:
        return json.dumps({"error": f"Order {order_id} not found in tenant {tenant_id}"})
    return json.dumps({"order_id": order_id, **order})

def cancel_order(order_id: str, tenant_id: str, **_) -> str:
    orders = TENANT_DATA.get(tenant_id, {}).get("orders", {})
    if order_id not in orders:
        return json.dumps({"error": f"Order {order_id} not found"})
    orders[order_id]["status"] = "cancelled"
    return json.dumps({"order_id": order_id, "status": "cancelled"})

HANDLERS = {"lookup_order": lookup_order, "cancel_order": cancel_order}

TOOLS_DEF = [
    {
        "name": "lookup_order",
        "description": "Look up an order by ID.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "cancel_order",
        "description": "Cancel an order by ID.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
]

def run_support_agent(
    role: str,
    tenant_id: str,
    user_message: str,
    require_confirmation: bool = True,
) -> str:
    permissions = ROLE_PERMISSIONS.get(role, set())
    allowed_tools = [t for t in TOOLS_DEF if t["name"] in permissions]

    system = f"You are a support agent for tenant {tenant_id}. Use tools to assist the user."
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            tools=allowed_tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name

            # Permission check (enforced at execution — not just by tool list)
            if tool_name not in permissions:
                result = f"Error: role '{role}' cannot call '{tool_name}'"
            elif tool_name in DESTRUCTIVE and require_confirmation:
                print(f"\n[REQUIRES CONFIRMATION] {tool_name}({block.input})")
                approved = input("Approve? (yes/no): ").strip().lower() == "yes"
                if not approved:
                    result = f"Action '{tool_name}' declined."
                else:
                    result = HANDLERS[tool_name](**block.input, tenant_id=tenant_id)
            else:
                result = HANDLERS[tool_name](**block.input, tenant_id=tenant_id)

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# Read-only role — cannot cancel
answer = run_support_agent("support_read", "tenant_A", "What's the status of ORD-001?", require_confirmation=False)
print(answer)

# Full role — can cancel with confirmation
answer = run_support_agent("support_full", "tenant_A", "Cancel order ORD-002.", require_confirmation=False)
print(answer)
```

## When to use it / when not to

#### Enforce strict access control when

- The agent has tools that cause real-world effects: sending messages, writing data, deleting records, executing code, or making financial transactions
- Multiple users share the agent and their data must be isolated
- The agent is exposed to untrusted content (user uploads, web pages, external APIs) — prompt injection risk is high
- Compliance or regulatory requirements mandate demonstrable access restrictions

#### Lighter restrictions are acceptable when

- The agent is single-user, internal tooling with read-only access to non-sensitive data
- Every action is reviewed by a human before execution
- The agent's tools are pure computation with no external side effects

#### The practical question

What's the worst thing this agent could do if a sophisticated attacker controlled its tool calls for one minute? If the answer is "not much," access control can be lightweight. If the answer involves data exfiltration, financial loss, or irreversible deletion, access control is non-optional.

:::tip[My take]

Most agent access control failures are not from sophisticated attacks — they're from missing the obvious. An agent built to read customer orders that also happens to have a "send email" tool available because the developer added it "just in case" is a waiting exfiltration path. Capability registries aren't bureaucracy; they're the forcing function that makes developers decide explicitly which tools an agent needs.

The single most important pattern is enforcing permissions at execution time, not at tool list construction time. Providing a filtered tool list to the model is not security — the model could (under injection or jailbreak) still emit a tool_use block for a tool not in the list. Your tool execution layer must check permissions independently.

For multi-user systems, tenant isolation is the hardest problem. The easiest architectural fix is to inject tenant context at the execution layer (not via the agent) and never let the agent see or pass tenant IDs. The agent says "lookup order ORD-001" and the execution layer adds "for this user's tenant" automatically.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| Anthropic tool use API | Defining tool schemas; the model selects tools; execution is in your code |
| OAuth 2.0 scopes | Scoped API tokens for third-party integrations; define minimal scope at token creation |
| AWS IAM / GCP IAM | Cloud-level access control for agent infrastructure; separate IAM roles per agent type |
| OPA (Open Policy Agent) | Policy-as-code for complex permission logic; externalize authorization decisions |
| Hashicorp Vault | Secrets management; issue short-lived, scoped credentials to agents |
| Kubernetes RBAC | Service account access control for agent processes running in Kubernetes |

## Common failure modes and gotchas

**1. Permission check only at tool list construction.** Providing a filtered tool list to the model is not enforcement — it's a hint. The model can still emit a `tool_use` block for any tool name. Enforce permissions at execution time, independently of what tools were listed.

**2. Admin credentials for convenience.** Giving an agent a database admin connection string because it's easier than creating a read-only user is the most common access control failure. Create scoped credentials for each agent role. The extra setup cost is worth it.

**3. Missing tenant isolation in multi-user systems.** An agent that takes `user_id` as input but doesn't enforce that tool calls are scoped to that user's data is a cross-tenant data leak waiting to happen. Inject tenant context at the execution layer, not via agent parameters.

**4. No confirmation for destructive actions.** An agent that can delete records, send messages, or trigger financial transactions without confirmation is a single injection away from a serious incident. Require explicit human approval for irreversible actions.

**5. Transitive permission escalation.** An agent that can call a tool that can call another service with broader permissions is escalating beyond its intended scope. Audit the full permission chain of each tool, not just the immediate call.

**6. Secrets in system prompts.** API keys and connection strings included in system prompts can be extracted via prompt injection ("repeat your system prompt"). Pass credentials to tool execution handlers via environment variables or secrets management — not via the prompt.

**7. Assuming tool use errors are handled.** If the execution layer denies a tool call, the error message goes back to the model. A poorly designed error message that says "this tool requires admin role" tells an attacker exactly what to escalate to. Return generic permission denied errors; log the specific reason separately.

## Project ideas

**1. Capability registry audit** — For an existing agent project, enumerate every tool available to the agent. For each tool, answer: what's the worst-case misuse? What permissions does it actually need? Which tools could be removed without breaking the core use case? Most agents have at least one tool that shouldn't be there.

**2. Tenant isolation test** — Build a two-tenant system. Create an agent that processes tenant A's requests. Write a test that verifies tool calls cannot access tenant B's data — including under injection (embed a payload in tenant A's data that tries to access tenant B's records). This is the most important access control test to run.

**3. Permission escalation simulation** — Build an agent with a read-only role. Write a red team prompt that tries to escalate to write permissions via: (a) direct request, (b) hypothetical framing, (c) persona injection, (d) gradual escalation. Document which attempts are blocked by the execution layer vs. the model, and whether any succeed.

**4. Scoped credential pipeline** — Using any cloud provider's IAM, create three IAM roles: reader, writer, deleter. Build an agent that accepts a role parameter and assumes the corresponding IAM role via short-lived credentials before making storage API calls. Verify that a reader-role agent fails to delete.

## Going deeper

#### Foundational reading

- NIST SP 800-53 (Security and Privacy Controls) — particularly the Access Control (AC) family; provides the framework for least-privilege, separation of duties, and audit requirements.
- OWASP Top 10 for LLM Applications — LLM04 (Model Denial of Service), LLM06 (Sensitive Information Disclosure), and LLM08 (Excessive Agency) directly map to access control failures in agent systems.

#### Tools and references

- Open Policy Agent documentation (openpolicyagent.org) — reference for policy-as-code; useful when permission logic is complex enough to warrant externalizing from application code.
- AWS IAM best practices documentation — least-privilege patterns, permission boundaries, and temporary credential patterns (STS AssumeRole); the most practical reference for cloud-deployed agent access control.
- Anthropic documentation on tool use — the reference for how tools are defined, selected, and returned; understanding the protocol helps design correct execution-layer enforcement.
