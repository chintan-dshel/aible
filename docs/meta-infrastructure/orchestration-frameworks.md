---
sidebar_position: 9
title: Orchestration Frameworks
description: LangChain, LlamaIndex, DSPy, and raw SDK — honest tradeoffs and when to reach for each.
---

# Orchestration Frameworks

## What it is

Orchestration frameworks are libraries that provide abstractions for building LLM applications — chains of model calls, retrieval pipelines, agent loops, memory management, and tool integrations — so you don't build these from scratch on top of the raw model API.

The tradeoff is explicit: frameworks reduce boilerplate and provide tested implementations of common patterns, at the cost of abstraction overhead, framework lock-in, and debugging opacity when something breaks.

## The problem it solves

A minimal LLM application requires gluing together: API calls, prompt construction, output parsing, context management, retrieval, tool execution, error handling, and logging. Doing this well from scratch is 500–2000 lines of engineering before you've shipped anything to users.

Orchestration frameworks provide pre-built versions of these components, along with composability primitives that let you chain them. The value is speed of initial development; the cost is that the framework's abstractions may not fit your requirements precisely, and debugging failures through framework internals is harder than debugging your own code.

## How it works under the hood

### LangChain

LangChain provides composable components: LLMs, prompts, output parsers, retrievers, chains, and agents. Components implement a standard `invoke()` interface and can be chained with the `|` operator (LCEL — LangChain Expression Language):

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatAnthropic(model="claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | model | StrOutputParser()

result = chain.invoke({"question": "What is the capital of France?"})
print(result)  # "Paris"
```

**LangGraph** extends LangChain with explicit graph-based agent orchestration — nodes are functions, edges define control flow, and state is a typed dictionary that flows through the graph:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    context: str
    response: str

def retrieve(state: AgentState) -> AgentState:
    # Fetch relevant documents
    state["context"] = f"Context for: {state['query']}"
    return state

def generate(state: AgentState) -> AgentState:
    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-sonnet-4-6")
    response = model.invoke(f"Context: {state['context']}\n\nQuestion: {state['query']}")
    state["response"] = response.content
    return state

builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"query": "What is the capital of France?", "context": "", "response": ""})
```

### LlamaIndex

LlamaIndex is focused on RAG and knowledge retrieval — indexing documents, querying them, and integrating the results into model calls. Its primary abstractions are `Document`, `Index`, and `QueryEngine`:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic

# Load documents and build an index
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine(
    llm=Anthropic(model="claude-sonnet-4-6"),
)
response = query_engine.query("What is our return policy?")
print(response)
```

LlamaIndex handles chunking, embedding, indexing, retrieval, and prompt assembly. It's the fastest path to a working RAG prototype; less suited for complex multi-agent orchestration.

### DSPy

DSPy takes a different approach: instead of manually crafting prompts, you define *signatures* (typed input/output contracts) and DSPy optimizes the prompt — or fine-tunes the model — to satisfy them. It's more like programming than prompting:

```python
import dspy

lm = dspy.LM("anthropic/claude-sonnet-4-6")
dspy.configure(lm=lm)

class QASignature(dspy.Signature):
    """Answer a question about a product given the provided context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.Predict(QASignature)
result = qa(
    context="Our return policy is 30 days for all products.",
    question="How long do I have to return something?"
)
print(result.answer)
```

DSPy's value proposition is that you can replace manual prompt engineering with an optimizer that empirically finds better prompts (or finetune data) on your eval set. The learning curve is steep; most teams don't need DSPy until manual prompt tuning has hit diminishing returns.

### Raw SDK

For anything that requires precise control, the raw Anthropic SDK is the right choice. No abstractions, no hidden behavior, full visibility:

```python
import anthropic

client = anthropic.Anthropic()

def build_rag_prompt(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))
    return f"""Answer the question using only the provided context.

<context>
{context}
</context>

Question: {query}
Answer:"""

def rag_call(query: str, chunks: list[str]) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": build_rag_prompt(query, chunks)}]
    )
    return response.content[0].text
```

This approach is more code upfront but easier to debug, test, and adapt. There are no framework internals to understand when something breaks.

## Concrete example

The same RAG pipeline implemented four ways, showing the tradeoff:

```python
# Option 1: LangChain (most abstraction)
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.from_texts(["Our return policy is 30 days."], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context.\n\nContext: {context}"),
    ("human", "{question}")
])
model = ChatAnthropic(model="claude-sonnet-4-6")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
answer_lc = chain.invoke("What is the return policy?")

# Option 2: Raw SDK (no abstraction)
import anthropic
client = anthropic.Anthropic()

def raw_rag(query: str, docs: list[str]) -> str:
    context = "\n".join(docs)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}]
    )
    return response.content[0].text

answer_raw = raw_rag("What is the return policy?", ["Our return policy is 30 days."])
```

Both produce similar results. The LangChain version is 3× more code but easier to swap components. The raw SDK version is easier to debug and has no hidden behavior.

## When to use it / when not to

#### Use a framework when

- You're building a standard pattern (RAG, chat, extraction) and want to ship quickly
- You need components the framework already has (document loaders, vector store integrations, agent loops with pre-built tool sets)
- Your team is already familiar with the framework and the ramp-up cost is low

#### Use the raw SDK when

- You need precise control over every aspect of the request — prompts, parsing, retries, context management
- You're building a non-standard pattern that frameworks don't have a good abstraction for
- Debugging production issues — frameworks hide the actual API calls and make stack traces harder to follow
- You're optimizing for latency — framework abstractions add overhead that matters at scale

#### Framework-specific guidance

| Framework | Best for | Avoid when |
|---|---|---|
| LangChain | Prototyping; teams needing breadth of integrations | You need precise prompt control; production systems where you need to understand every API call |
| LangGraph | Complex multi-step agent flows with explicit state management | Simple linear pipelines where a direct function call suffices |
| LlamaIndex | RAG-first applications; complex document indexing | Multi-agent orchestration; tasks beyond retrieval |
| DSPy | When manual prompt engineering has hit diminishing returns; systematic prompt optimization | New projects; teams without a strong eval set to optimize against |
| Raw SDK | Production systems; custom patterns; maximum debuggability | Rapid prototyping of standard patterns where framework boilerplate is acceptable |

:::tip[My take]

Start with the raw SDK for any non-trivial production system. The 20% of additional boilerplate code is worth it for 80% better debuggability. When something breaks at 2am, you want to see the exact API call and response, not a framework traceback.

Reach for LangChain or LlamaIndex when you need their integrations (document loaders, vector store connectors, pre-built tool sets) or when you're building a prototype and speed of development matters more than long-term maintainability.

DSPy is the most interesting of the three for the long run — treating prompts as optimizable parameters rather than hand-crafted artifacts is the right direction. But it requires a strong eval set (which you should have anyway) and a mindset shift from prompt engineering to program synthesis. Reserve it for after you've manually tuned prompts and hit a plateau.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| LangChain / LangGraph | Broad integrations; prototyping; agent state machines |
| LlamaIndex | RAG and document-centric applications |
| DSPy | Systematic prompt optimization; programmatic pipeline construction |
| Anthropic SDK (raw) | Production systems; custom patterns; maximum control |
| Instructor | Structured output extraction; works alongside any framework |
| Prefect / Airflow | Workflow orchestration for batch LLM pipelines with scheduling |

## Common failure modes and gotchas

**1. Framework version churn.** LangChain especially has broken APIs between minor versions. Pin your framework version in `requirements.txt` and test upgrades explicitly. Don't assume a patch version is safe.

**2. Hidden API calls.** Frameworks often make extra model calls you didn't know about — for routing, classification, or retry. Log API calls at the HTTP level (not the framework level) to see everything that's actually happening.

**3. Abstraction debugging overhead.** When a chain fails, the error message is often from deep inside the framework. Add logging at the function level before and after each chain step — don't rely on framework-level tracing alone.

**4. Over-engineering with frameworks.** Using LangGraph for a pipeline that's three sequential function calls. The framework adds cognitive overhead without value for simple linear pipelines.

**5. Vendor lock-in in production.** A LangChain-heavy codebase is harder to migrate away from than a raw SDK codebase. If you need to switch models, frameworks, or providers in 12 months, raw SDK code is easier to adapt.

**6. DSPy without an eval set.** DSPy's optimizer needs examples with known good outputs to optimize against. Using DSPy without a strong eval set results in optimizing for noise rather than real quality improvements.

## Project ideas

**1. Framework comparison for the same task** — Implement a RAG pipeline three ways: LangChain, LlamaIndex, and raw SDK. On the same query set, measure: lines of code, debug time when you deliberately introduce a bug (wrong prompt, bad retrieval), and latency. Decide which framework's tradeoffs fit your team.

**2. LangGraph state machine** — Build a multi-step agent with LangGraph: plan → research → draft → review → deliver. Use LangGraph's conditional edges to handle failure paths (research fails → retry; review fails → revise). This gives hands-on experience with explicit state management vs. implicit agent loops.

**3. DSPy optimization loop** — Take a prompt you've manually written for a classification task. Build a DSPy signature for the same task. Run DSPy's optimizer on a 50-example eval set. Compare: did the optimized prompt beat your manual one? How many evals did the optimizer consume?

**4. Framework internals dive** — Pick one framework (LangChain or LlamaIndex). Run a simple chain and instrument the HTTP-level calls using a proxy. Document every API call made, the exact prompts sent, and token counts. Most developers are surprised by what their framework is actually doing.

## Going deeper

#### Foundational reading

- LangChain documentation (python.langchain.com) — the reference for LCEL and LangGraph; the Getting Started section is the fastest path to understanding the abstraction model.
- LlamaIndex documentation (docs.llamaindex.ai) — strong on the indexing and retrieval abstractions; good worked examples for RAG patterns.
- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023) — the DSPy paper; explains the program synthesis approach and the optimizer.

#### Comparison resources

- "LangChain vs LlamaIndex vs DSPy" — search for recent comparisons; the landscape changes quickly enough that blog posts from the past 6 months are more reliable than anything older.
