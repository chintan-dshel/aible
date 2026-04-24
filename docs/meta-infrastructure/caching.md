---
sidebar_position: 8
title: Caching
description: Prompt caching, semantic cache, and exact-match cache — where each fits, what it saves, and how to measure hit rate.
---

# Caching

## What it is

Caching in LLM systems means storing and reusing previously computed results so that repeated or similar requests don't repeat the expensive work. There are four distinct caching layers in AI systems, each working at a different level:

- **Prompt caching** — cache the KV states of a long static prefix (system prompt, RAG context) so repeated requests don't reprocess it
- **Exact-match cache** — return a stored response for requests identical to a previous one
- **Semantic cache** — return a stored response for requests *similar* to a previous one (by embedding distance)
- **KV cache** — the internal attention cache the model maintains *during* a single generation (not under application control)

The first three are under your control as an application developer. The fourth is managed by the inference runtime.

## The problem it solves

LLM inference is expensive and slow. The primary cost axes are:

- **Latency**: a model call takes 200ms–3s depending on model size and output length
- **Token cost**: you pay per input token and per output token on every call
- **Repetition**: many systems repeatedly process the same large contexts — a system prompt, a policy document, a user's profile — on every request

Caching reduces cost and latency for the fraction of requests that can be served from cache without calling the model at all, or that can skip processing of a repeated prefix.

## How it works under the hood

### Prompt caching (Anthropic / provider-side)

Anthropic's prompt caching allows you to mark a portion of your prompt as a cacheable prefix. The API caches the internal KV attention states of that prefix after the first call. Subsequent requests that share the same prefix skip the attention computation for those tokens.

The benefit: input token computation is expensive. Caching a 10K-token system prompt + RAG context block means subsequent calls within the cache TTL pay a fraction of the normal input token cost.

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = "You are an expert product support agent..." + ("." * 5000)  # long system prompt

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # mark this prefix as cacheable
        }
    ],
    messages=[{"role": "user", "content": "How do I reset my password?"}]
)

# Subsequent calls with the same system prefix benefit from cached KV states
# Cache TTL: 5 minutes. Cached tokens cost ~10% of normal input token price.
print(response.usage.cache_creation_input_tokens)  # tokens written to cache (first call)
print(response.usage.cache_read_input_tokens)       # tokens read from cache (subsequent calls)
```

**When prompt caching pays off:** the static prefix must be large (≥ 1,024 tokens for Sonnet) and repeated frequently. A 20K-token RAG context block that's different for every query won't benefit. A fixed 10K-token system prompt used on every call will see ~90% cost reduction on those tokens.

### Exact-match cache

Cache the full (input, output) pair. Return the cached output immediately if the exact same request is seen again. Best for idempotent, deterministic-feeling requests.

```python
import hashlib
import json
import anthropic

client = anthropic.Anthropic()
_cache: dict[str, str] = {}  # in production: Redis or memcached

def cache_key(messages: list, system: str, model: str) -> str:
    payload = json.dumps({"messages": messages, "system": system, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def cached_call(messages: list, system: str = "", model: str = "claude-sonnet-4-6") -> str:
    key = cache_key(messages, system, model)

    if key in _cache:
        return _cache[key]  # cache hit: return immediately, no model call

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=messages,
    )
    output = response.content[0].text
    _cache[key] = output
    return output
```

**Limitations:** any change in the request (typo correction, rephrasing, different punctuation) produces a cache miss. Hit rate on natural language input is typically low unless the same queries repeat exactly (FAQ bots, fixed pipeline steps).

### Semantic cache

Use embedding similarity to match semantically equivalent queries to cached responses, even when the exact text differs.

```python
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

client = anthropic.Anthropic()
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.entries: list[dict] = []  # {embedding, query, response}

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> str | None:
        if not self.entries:
            return None
        query_emb = embed_model.encode(query)
        best_score, best_response = 0.0, None
        for entry in self.entries:
            score = self._similarity(query_emb, entry["embedding"])
            if score > best_score:
                best_score, best_response = score, entry["response"]
        if best_score >= self.threshold:
            return best_response
        return None

    def set(self, query: str, response: str) -> None:
        embedding = embed_model.encode(query)
        self.entries.append({"embedding": embedding, "query": query, "response": response})

cache = SemanticCache(similarity_threshold=0.92)

def semantic_cached_call(query: str, system: str = "") -> str:
    cached = cache.get(query)
    if cached:
        return cached

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": query}]
    )
    output = response.content[0].text
    cache.set(query, output)
    return output
```

**Threshold calibration**: For text embeddings, similarity of 0.90+ is often *not* conservative — sentences that share key terms but have different answers ("What is your return policy?" vs "What is your return policy for electronics?") can easily score 0.90+. The right threshold is domain-dependent. Test against your domain's hardest negative pairs — questions that look similar but require different answers — and find the lowest threshold that correctly rejects them. For most FAQ domains, this falls in the 0.88–0.93 range; run the calibration rather than assuming a number.

### KV cache (model-internal)

Not a developer-controlled mechanism — this is the attention key-value cache the model maintains during a single generation pass. Each generated token attends to all previous tokens; without caching, that attention recomputes all previous KV pairs on every step. The KV cache stores them.

Relevance for application developers: the KV cache grows linearly with context length, consuming GPU memory. For long-context requests (> 32K tokens), KV cache memory can exceed model parameter memory. Inference providers manage this automatically; self-hosting requires planning for this.

### Combining layers

In a production RAG pipeline, you might combine:
1. **Prompt caching**: cache the static system prompt prefix
2. **Semantic cache**: cache query → response pairs for repeated user questions
3. **Result cache**: cache retrieved chunks for a query (avoid re-embedding and re-retrieving)

```python
# Layer order: cheapest check first
def serve_request(query: str, system: str) -> str:
    # 1. Semantic cache (fast, local)
    cached = cache.get(query)
    if cached:
        return cached

    # 2. Retrieve context
    chunks = retrieve_chunks(query)  # your retrieval function

    # 3. Model call with prompt caching on system prefix
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Context:\n{chunks}\n\nQuestion: {query}"}]
    )
    output = response.content[0].text
    cache.set(query, output)
    return output
```

## Concrete example

Measuring cache hit rate and cost savings for a customer support FAQ bot:

```python
import anthropic
import hashlib
import json
import time

client = anthropic.Anthropic()

class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_latency_ms = 0.0
        self.cached_latency_ms = 0.0
        self.model_latency_ms = 0.0
        self.tokens_saved = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

metrics = CacheMetrics()
_exact_cache: dict[str, tuple[str, int]] = {}  # key → (response, output_tokens)

SYSTEM = "You are a helpful customer support agent."

def tracked_call(user_message: str) -> str:
    key = hashlib.sha256(user_message.encode()).hexdigest()
    start = time.time()

    if key in _exact_cache:
        response, saved_tokens = _exact_cache[key]
        elapsed = (time.time() - start) * 1000
        metrics.hits += 1
        metrics.cached_latency_ms += elapsed
        metrics.total_latency_ms += elapsed
        metrics.tokens_saved += saved_tokens
        return response

    api_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}]
    )
    output = api_response.content[0].text
    output_tokens = api_response.usage.output_tokens

    elapsed = (time.time() - start) * 1000
    metrics.misses += 1
    metrics.model_latency_ms += elapsed
    metrics.total_latency_ms += elapsed
    _exact_cache[key] = (output, output_tokens)
    return output

# Simulate traffic with repeated questions
queries = [
    "How do I reset my password?",
    "What is your return policy?",
    "How do I reset my password?",  # repeat
    "How do I contact support?",
    "What is your return policy?",  # repeat
    "How do I reset my password?",  # repeat again
]

for q in queries:
    result = tracked_call(q)

print(f"Hit rate: {metrics.hit_rate:.1%}")
print(f"Avg latency (cache hit): {metrics.cached_latency_ms / max(metrics.hits, 1):.1f}ms")
print(f"Avg latency (model call): {metrics.model_latency_ms / max(metrics.misses, 1):.1f}ms")
print(f"Output tokens saved: {metrics.tokens_saved}")
```

## When to use it / when not to

#### Use prompt caching when

- Your system prompt or RAG context block is large (≥ 1K tokens) and repeated on many requests
- You're building a document Q&A system where the full document is included in every call
- You're running multi-turn conversations with a long conversation history prefix

#### Use exact-match cache when

- You have a known set of high-frequency questions (FAQ bot, fixed pipeline stages)
- Determinism matters and you don't want responses to vary between identical inputs
- You're caching intermediate pipeline steps that always have the same input (e.g., "classify this category")

#### Use semantic cache when

- Users ask the same questions in different phrasings ("return policy", "how do I return something", "can I get a refund")
- Responses don't need to be personalized or real-time accurate
- You've measured a cache hit rate > 20% in testing — below that, the embedding overhead isn't worth it

#### Don't cache when

- Responses must be fresh (live data, personalized answers, user-specific context)
- Exact accuracy matters more than speed (medical, legal, financial)
- Your query distribution is highly diverse — you won't get meaningful hit rates

:::tip[My take]

Prompt caching is the easiest win in this list. If your system prompt or context block is large and repeated on every call, adding `cache_control` takes five minutes and cuts input token costs by up to 90% on those tokens. Measure your `cache_read_input_tokens` vs `cache_creation_input_tokens` ratio — if it's not > 3:1 after a few hundred calls, your prefix isn't being repeated frequently enough to benefit.

Semantic caching is appealing but harder to get right. The threshold calibration problem is real: too aggressive and you return slightly wrong answers; too conservative and you get near-zero hit rates. Measure on your actual query distribution before committing. And never semantic-cache personalized responses — if the answer depends on who's asking, caching breaks it.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| Anthropic API prompt caching | Native; reduces input token cost for repeated static prefixes |
| Redis | Production exact-match cache backend; TTL, LRU eviction, persistence |
| GPTCache | Semantic cache library; integrates with FAISS/Qdrant for embedding-based cache lookup |
| LangChain `CacheBackedEmbeddings` | Cache embedding results (reduce embedding API calls for repeated text) |
| FAISS / Qdrant | Vector store backend for semantic cache similarity lookup |

## Common failure modes and gotchas

**1. Stale cache responses.** If your underlying data changes (policy updates, product changes), cached responses become incorrect. Implement cache invalidation on data updates, or use short TTLs (minutes/hours, not days) for content-sensitive caches.

**2. Semantic cache false matches.** A threshold too low returns cached responses for semantically related but distinct questions. "What's your return policy for electronics?" might match "What's your return policy?" at 0.88 similarity — but the answer to the first may be different. Test extensively with domain-specific question pairs.

**3. Prompt cache TTL mismatch.** Anthropic's prompt cache TTL is 5 minutes. If your traffic is sparse (< 1 request per 5 minutes), the cache will never be warm. Prompt caching only pays off with sustained traffic.

**4. Caching non-idempotent calls.** Caching a response that includes the current time, user-specific data, or real-time pricing returns stale data. Mark these calls as non-cacheable.

**5. Not measuring hit rate.** A semantic cache that achieves 5% hit rate and adds 20ms of embedding overhead is a net negative. Always instrument hit rate, latency delta (cache hit vs miss), and token savings. If hit rate is below ~15%, the overhead likely isn't worth it.

**6. No eviction policy.** An in-memory exact-match cache without eviction grows unbounded. Use an LRU cache with a maximum size, or use Redis with a TTL.

## Project ideas

**1. Prompt cache efficiency analysis** — Build a pipeline with a 5K-token system prompt. Make 100 calls in sequence. Plot `cache_creation_input_tokens` vs `cache_read_input_tokens` per call. Compute the effective cost reduction. Then deliberately break the cache (change one token in the prefix) and observe the impact. This makes the cache TTL behavior concrete.

**2. Semantic cache threshold sweep** — Take 200 real queries from a FAQ domain. Create 50 semantically equivalent pairs (same meaning, different wording) and 50 semantically distinct pairs (similar topic, different answer). Run semantic cache matching at thresholds 0.80, 0.85, 0.90, 0.92, 0.95. Plot precision and recall for each threshold. Find the optimal operating point for your domain.

**3. Cache layer comparison** — On a FAQ bot dataset with 1,000 queries (heavy repetition), compare: (a) no cache, (b) exact-match cache, (c) semantic cache, (d) both. Measure: hit rate, average response latency, total token cost for the batch. This quantifies the value of each layer.

**4. Cache invalidation pipeline** — Build a simple document-based Q&A system with semantic caching. Then simulate a document update (change the source of truth). Build a cache invalidation mechanism triggered by the document update. Verify that post-update queries return fresh responses, not stale cached ones.

## Going deeper

#### Foundational reading

- Anthropic prompt caching documentation — the reference for how to configure `cache_control` and interpret `cache_read_input_tokens` in usage.
- Bang et al., "GPTCache: A Data or Model Cache for Large Language Model" (2023) — the paper behind the GPTCache library; explains semantic cache design and evaluation metrics.

#### Libraries

- GPTCache (GitHub: `zilliztech/GPTCache`) — the most complete open-source semantic cache implementation; supports multiple vector backends.
- Redis documentation — the standard for production exact-match cache backends; supports TTL, LRU eviction, and persistence.
