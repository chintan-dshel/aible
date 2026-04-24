---
sidebar_position: 10
title: MCP — Model Context Protocol
description: What the Model Context Protocol is, why it exists, and what it changes about how agents connect to tools.
---

# MCP — Model Context Protocol

## What it is

The Model Context Protocol (MCP) is an open standard protocol, introduced by Anthropic in late 2024, that defines a uniform interface for connecting LLMs to external tools, data sources, and services. It specifies how a host application (a Claude client, an IDE, an agent runtime) communicates with servers that expose capabilities — file systems, databases, APIs, search engines — in a standardized way.

The analogy is USB-C for AI: before MCP, every tool integration was custom — a different interface, a different authentication pattern, a different capability contract. MCP provides one connector that works with any tool that implements the server spec.

## The problem it solves

The dominant approach before MCP was custom tool definitions per application: you write a tool schema for your agent, implement the function, wire it into your prompt, and parse the response. This works but creates M × N integration complexity — M agents each needing to integrate N tools produces M×N custom implementations, each with its own bugs, auth handling, and maintenance burden.

MCP collapses this to M + N: tools implement the MCP server spec once, and any MCP-compatible host can use them without custom integration code.

Concretely:
- A file system MCP server, once written, works in Claude Desktop, VS Code Copilot, and any other MCP-compatible client
- A database MCP server works across agents without per-agent schema wiring
- Users (not just developers) can extend what an AI assistant can do by adding MCP servers to their configuration

## How it works under the hood

### Architecture

MCP has three layers:

**Host** — the application or runtime that embeds the LLM (Claude Desktop, an IDE extension, an agent runtime). The host manages connections to one or more MCP servers.

**Client** — the protocol layer within the host that communicates with servers. Each client maintains a 1:1 connection with one server.

**Server** — an external process that exposes capabilities via the MCP protocol. Servers declare what they offer and respond to requests.

```
Host (Claude Desktop / agent runtime)
  └── Client ──► Server A (filesystem)
  └── Client ──► Server B (database)
  └── Client ──► Server C (web search)
```

### Transport

MCP servers communicate over standard I/O (for local servers) or HTTP with Server-Sent Events (for remote servers). Local servers are spawned as child processes by the host.

:::caution[Security: stdio servers run with your full permissions]

When Claude Desktop or any MCP host spawns an stdio server, that server process inherits the host's user account permissions — it can read your files, write to disk, make network calls, and execute code with no additional sandboxing. MCP provides no built-in isolation between server and host.

Before configuring any MCP server from an external or untrusted source: read its code, verify what it does, and treat it with the same scrutiny you would apply to any third-party binary running as your user. "Just add this MCP server" is equivalent to "just run this script."

:::



### Server capabilities

An MCP server can expose three types of capabilities:

**Tools** — functions the LLM can call. Defined with a name, description, and JSON schema for arguments. This is the most common capability type.

**Resources** — data the LLM can read. Files, database records, API responses — any content that should be retrieved and injected into context. Resources have URIs and mime types.

**Prompts** — pre-defined prompt templates that the host can present to the user or inject into the LLM's context.

### Implementing an MCP server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio

app = Server("example-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="read_file",
            description="Read the contents of a file by path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="search_db",
            description="Search the product database by keyword.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "read_file":
        path = arguments["path"]
        try:
            with open(path, "r") as f:
                content = f.read()
            return [types.TextContent(type="text", text=content)]
        except FileNotFoundError:
            return [types.TextContent(type="text", text=f"File not found: {path}")]

    elif name == "search_db":
        # Your actual database search logic here
        results = [{"id": 1, "name": "Widget Pro", "price": 29.99}]  # stub
        import json
        return [types.TextContent(type="text", text=json.dumps(results))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Connecting to an MCP server from a client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["my_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:", [t.name for t in tools_result.tools])

            # Call a tool
            result = await session.call_tool("read_file", {"path": "README.md"})
            print(result.content[0].text)

asyncio.run(use_mcp_server())
```

### MCP in Claude Desktop

In Claude Desktop, MCP servers are configured in a JSON settings file. Users point to a server process; Claude Desktop spawns it and makes its tools available to Claude:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/alice/Documents"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "postgresql://localhost/mydb"
      }
    }
  }
}
```

Once configured, Claude can read files from the specified directory and query the database in natural language — without any additional code.

## Concrete example

An MCP server that exposes a product catalog for a customer support agent:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio
import json

app = Server("product-catalog")

PRODUCTS = {
    "WP-001": {"name": "Widget Pro", "price": 29.99, "stock": 150, "category": "widgets"},
    "GP-002": {"name": "Gadget Plus", "price": 49.99, "stock": 0, "category": "gadgets"},
    "AC-003": {"name": "Accessory Kit", "price": 9.99, "stock": 500, "category": "accessories"},
}

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_product",
            description="Get details for a specific product by ID.",
            inputSchema={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"]
            }
        ),
        types.Tool(
            name="search_products",
            description="Search products by name or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "in_stock_only": {"type": "boolean", "default": False}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="check_stock",
            description="Check the current stock level for a product.",
            inputSchema={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_product":
        product = PRODUCTS.get(arguments["product_id"])
        if not product:
            return [types.TextContent(type="text", text="Product not found.")]
        return [types.TextContent(type="text", text=json.dumps(product))]

    elif name == "search_products":
        query = arguments["query"].lower()
        in_stock = arguments.get("in_stock_only", False)
        results = [
            {"id": pid, **p}
            for pid, p in PRODUCTS.items()
            if (query in p["name"].lower() or query in p["category"])
            and (not in_stock or p["stock"] > 0)
        ]
        return [types.TextContent(type="text", text=json.dumps(results))]

    elif name == "check_stock":
        product = PRODUCTS.get(arguments["product_id"])
        if not product:
            return [types.TextContent(type="text", text="Product not found.")]
        status = "in stock" if product["stock"] > 0 else "out of stock"
        return [types.TextContent(type="text", text=f"{product['name']}: {product['stock']} units ({status})")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

With this server running, a Claude Desktop user or an MCP-compatible agent can ask "Is the Widget Pro in stock?" or "Show me all in-stock gadgets" and Claude will call `check_stock` or `search_products` automatically.

## When to use it / when not to

#### MCP is the right choice when

- You're building a tool that multiple AI clients should be able to use (write once, use everywhere)
- You're adding AI capabilities to an existing application and want users to control which tools Claude can access
- You want users (not just developers) to be able to extend what an AI assistant can do via configuration
- You need tool access in Claude Desktop or another MCP-compatible host without building a custom agent

#### Direct tool use (Anthropic tool_choice API) is better when

- You're building a single agent for a specific task and don't need cross-client portability
- You need precise control over when and how tools are called
- Your tool logic is tightly coupled to your application state in ways that don't translate to a standalone server
- You're not targeting MCP-compatible hosts

#### The practical question

Is portability across clients the goal? MCP solves the multi-client integration problem. If you're building one agent for one task and never need to reuse the tools elsewhere, direct tool use via the Anthropic API is simpler.

:::tip[My take]

MCP matters most for tool builders, not application builders. If you're exposing a database, file system, or API as capabilities for AI assistants — and you want those capabilities to work in Claude Desktop, VS Code, and custom agents without per-client integration — MCP is the right abstraction.

If you're building a specific agent for a specific task, the raw Anthropic tool use API gives you more control with less overhead. MCP's value is in standardization and reuse across clients.

The ecosystem is still maturing — the protocol has stabilized but the tooling and community of pre-built servers continues to grow. Check the official MCP server registry before building a server for common integrations (filesystem, database, web search) — there may already be a server you can use directly.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| `mcp` Python SDK | Building MCP servers and clients in Python |
| `@modelcontextprotocol/sdk` | TypeScript SDK for MCP servers and clients |
| MCP Inspector | GUI for testing and debugging MCP servers during development |
| Claude Desktop | End-user host that supports MCP server configuration |
| Official MCP servers repo | Pre-built servers for filesystem, PostgreSQL, GitHub, Google Drive, etc. |

## Common failure modes and gotchas

**1. Server lifecycle management.** MCP servers are separate processes. If the server crashes, the host must restart it. Build health checks and automatic restart into your server deployment — don't assume it will stay running indefinitely.

**2. No authentication in the stdio transport.** Stdio-based MCP servers run as child processes of the host and inherit its trust context. They don't need authentication within that channel. But if you expose a server over HTTP (remote transport), you need proper authentication — don't assume the stdio security model transfers.

**3. Tool description quality.** Claude uses tool descriptions to decide when to call a tool. Vague descriptions ("does stuff with files") result in incorrect or missed tool calls. Write descriptions as if you're writing for a competent engineer who hasn't seen your code: "Read the contents of a local file by path. Returns the full file text as a string."

**4. Not handling tool errors gracefully.** If a tool call fails (file not found, database error, timeout), return a structured error message rather than raising an exception or returning empty content. Claude can handle "File not found: /path/to/file" and tell the user clearly; an unhandled exception returns an opaque error.

**5. Resource exposure without access control.** If your MCP server exposes file system or database access, scope it to the minimum necessary. The filesystem server example above uses the MCP path restrictions argument. Don't expose a root filesystem server when only one directory is needed.

**6. Stateful servers without cleanup.** If your server maintains state (open database connections, in-progress operations), implement proper cleanup on shutdown. The `stdio_server()` context manager handles connection lifecycle; your server's cleanup should hook into it.

## Project ideas

**1. Build and test an MCP server** — Implement an MCP server that exposes three tools (e.g., search a local CSV file, fetch a URL, calculate a value). Connect it to Claude Desktop and test with natural language queries. Observe which tools Claude calls and when. Improve the tool descriptions based on incorrect calls.

**2. MCP server for an existing API** — Take an API you already use (a weather API, a CRM, an internal service). Wrap it as an MCP server. Write integration tests that call each tool with valid and invalid inputs. Deploy it locally and connect it to Claude Desktop.

**3. Multi-server agent** — Configure two MCP servers in Claude Desktop (e.g., a filesystem server and a database server). Test queries that require both (e.g., "Read the CSV file and store its contents in the database"). Observe how Claude orchestrates multiple tool calls across servers.

**4. MCP Inspector debugging session** — Use the MCP Inspector to connect to your server. Send tool calls manually. Observe the raw JSON protocol messages. This gives you visibility into the protocol layer and is the fastest way to debug unexpected server behavior.

## Going deeper

#### Foundational reading

- MCP specification (modelcontextprotocol.io/docs) — the authoritative spec for the protocol; covers message formats, transport options, and capability types.
- Anthropic MCP announcement blog post — the original design rationale and goals; explains the USB-C analogy and the M×N integration problem.

#### Tools and resources

- MCP Python SDK documentation (github.com/modelcontextprotocol/python-sdk) — the reference for building Python MCP servers; includes worked examples for all capability types.
- Official MCP servers (github.com/modelcontextprotocol/servers) — pre-built servers for common integrations (filesystem, PostgreSQL, GitHub, Slack, Google Drive); check here before building your own.
- MCP Inspector (github.com/modelcontextprotocol/inspector) — essential development tool; connects to any MCP server and lets you call tools manually to debug during development.
