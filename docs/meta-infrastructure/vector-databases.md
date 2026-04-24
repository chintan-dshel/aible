---
sidebar_position: 7
title: Vector Databases
description: HNSW, IVF, and the tradeoffs between pgvector, Pinecone, Weaviate, Qdrant, and Chroma — choosing the right store for your scale.
---

# Vector Databases

## What it is

A vector database is a storage system designed to efficiently index and retrieve high-dimensional embedding vectors by similarity — finding the nearest neighbors to a query vector across millions or billions of stored vectors in milliseconds.

Unlike a relational database that retrieves rows by exact key match, a vector database retrieves vectors by *geometric proximity*. Given a query embedding, it returns the k most similar stored embeddings (and their associated metadata) using approximate nearest neighbor (ANN) search algorithms that trade a small amount of recall for large gains in speed.

## The problem it solves

Embeddings enable semantic search — but only if you can search them efficiently. A naive approach (compute cosine similarity between the query vector and every stored vector) scales linearly with corpus size. At 1 million vectors, this is ~3 seconds per query on CPU. At 100 million vectors, it's impractical entirely.

Vector databases solve this with specialized indexes — data structures that precompute geometric relationships between vectors so that at query time, you don't compare against every vector, only a subset of candidates. This makes similarity search fast enough for production use: sub-10ms queries at millions of vectors.

## How it works under the hood

### Approximate nearest neighbor (ANN) algorithms

**Flat search (exact)** — compute distance from the query to every stored vector. Exact recall, but O(n) per query. Acceptable up to ~100K vectors; impractical beyond.

**IVF (Inverted File Index)** — partition the vector space into clusters using k-means. At query time, search only the nearest N clusters rather than all vectors. Faster than flat search at the cost of recall (vectors near cluster boundaries may be missed).

```
Training phase: k-means clusters all vectors → k centroids
Query phase: find the nearest m centroids → search vectors in those m clusters only
Trade-off: m controls the speed/recall trade-off (nprobe parameter in FAISS)
```

**HNSW (Hierarchical Navigable Small World)** — builds a multi-layer graph where each vector connects to its nearest neighbors. Queries traverse from the top layer (coarse navigation) to the bottom (fine search). Best-in-class recall and speed in practice; higher memory usage.

```
Layers: top = few long-range connections, bottom = many short-range connections
Query: start at top, navigate down by greedy best-first search
Trade-off: ef_construction controls build recall vs. time; ef_search controls query recall vs. time
```

**Product Quantization (PQ)** — compresses vectors before storing by splitting them into subvectors and quantizing each independently. Reduces memory 4–16× at the cost of some recall. Often combined with IVF (IVF-PQ) for large-scale deployment.

### Index selection in practice

```
< 10K vectors  → Flat (FAISS IndexFlatL2); fine for prototypes and offline pipelines
10K–10M vectors → HNSW for best recall, IVF for lower memory
> 10M vectors  → IVF-PQ; memory is the constraint
```

For user-facing latency requirements (< 500ms), use HNSW even at small corpus sizes — flat search on 50K–100K vectors can take 1–3 seconds on CPU, which is at or above the user friction threshold. Reserve flat search for batch jobs, offline evaluation, and prototypes where latency doesn't matter.

### Metadata filtering

Pure ANN search returns the most similar vectors globally. In practice, you almost always need filtered search — "find the nearest neighbors among vectors that belong to user X" or "find the nearest neighbors from documents tagged `legal`."

Filtering strategies:
- **Pre-filter**: apply the metadata filter before ANN search (restrict the search space). Requires the index to support efficient pre-filtering on metadata columns.
- **Post-filter**: run ANN search and then filter results. Simple but may return < k results after filtering; requires over-fetching.
- **Hybrid**: use the metadata filter to select a subset of the index, then run ANN within that subset.

```python
# Qdrant: pre-filter by user_id, then similarity search
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value="user_123"))]
    ),
    limit=10,
)
```

### Distance metrics

| Metric | Use when |
|---|---|
| Cosine similarity | Most embedding models (text, images) — captures angular similarity |
| Euclidean (L2) | Embeddings not normalized; absolute magnitude matters |
| Dot product | Embeddings trained with dot product objective (some retrieval models) |

For text embeddings, cosine similarity is almost always correct. **Do not assume your vector database normalizes automatically.** Qdrant and Weaviate do not normalize vectors by default — if you configure dot product distance, you get raw dot product, not cosine similarity. To get cosine semantics with dot product configuration, L2-normalize your embeddings before indexing them. Pinecone normalizes internally; check your specific tool's documentation before assuming normalization behavior.

## Concrete example

A complete RAG vector store setup with Qdrant:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Initialize
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.recreate_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def index_documents(docs: list[dict]) -> None:
    """docs: list of {"id": str, "text": str, "metadata": dict}"""
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": doc["text"], **doc["metadata"]},
        )
        for doc, embedding in zip(docs, embeddings)
    ]
    client.upsert(collection_name="knowledge_base", points=points)

def search(query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    query_embedding = model.encode(query).tolist()

    search_filter = None
    if filters:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        search_filter = Filter(must=conditions)

    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload["text"], "score": r.score, **r.payload} for r in results]

# Index
index_documents([
    {"id": "1", "text": "Acme Corp was founded in 1987.", "metadata": {"source": "about_page", "user_id": "all"}},
    {"id": "2", "text": "Our return policy is 30 days for all products.", "metadata": {"source": "policy", "user_id": "all"}},
])

# Search
results = search("How long do I have to return something?", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | {r['text']}")
```

## When to use it / when not to

#### Use a vector database when

- Your corpus exceeds ~50K documents (flat search becomes too slow)
- You need filtered search (by user, date, category, source)
- You need real-time updates — documents are indexed as they arrive
- You need production monitoring (query latency, index size, filter hit rates)

#### Simpler alternatives when

- **Corpus < 50K documents**: FAISS in-memory with flat index. No infrastructure, no ops, sub-second queries. Use this for prototypes and small internal tools.
- **You're already on PostgreSQL**: `pgvector` adds vector search to Postgres. You get SQL filtering, transactions, and backups — at the cost of some ANN speed. Often the right choice for teams already operating Postgres who don't want another service.
- **You need full-text + semantic search**: Elasticsearch/OpenSearch with dense vector support gives you hybrid BM25+semantic search in one system.

:::tip[My take]

Start with the simplest thing that works at your scale. For a prototype, FAISS with a flat index takes 15 minutes to set up and has no moving parts. For a production system with 1–10M vectors, Qdrant or pgvector with HNSW covers 90% of use cases without the operational overhead of a managed cloud service.

The main split point is metadata filtering. If you need to filter by user_id, date, or any other attribute *before* similarity search (not after), you need a vector database that supports indexed metadata — not just FAISS. Qdrant, Weaviate, and Pinecone all support this well; FAISS does not.

Don't underestimate the operational cost of managed vector database services. Pinecone and Weaviate Cloud are convenient but expensive at scale. If you're running millions of queries per day, self-hosting Qdrant on a dedicated VM is often 5–10× cheaper and straightforward to operate.

:::

## Main tools and libraries

| Tool | Best for |
|---|---|
| Qdrant | Self-hosted production; strong filtering; active development; Rust-based for performance |
| pgvector | Teams already on PostgreSQL; SQL filtering + vector search in one system |
| Pinecone | Managed cloud; zero ops; expensive at scale |
| Weaviate | Hybrid search (BM25 + vector); schema-first; managed and self-hosted |
| Chroma | Local development and prototyping; embedded mode requires no server |
| FAISS | In-memory exact/approximate search; no persistence; great for prototyping |
| Milvus | High-scale (100M+ vectors); complex ops; good for dedicated retrieval teams |

## Common failure modes and gotchas

**1. Mixing embedding models.** If you switch embedding models mid-deployment and don't rebuild the index, you have vectors from two different geometric spaces in the same index. Cosine similarity across models is meaningless. Always delete and rebuild when switching models.

**2. Not testing filter recall.** With aggressive pre-filtering (e.g., filtering to a single user's documents), ANN search may have only 100 candidates to work with, degrading to near-linear search. Test your P95 query latency under your most selective filters.

**3. Index build time at scale.** HNSW indexes require significant build time as the corpus grows (O(n log n) approximately). For a 10M vector corpus, index build may take hours. Plan for incremental indexing strategies and batch rebuild windows.

**4. No monitoring on similarity scores.** Returning a result with cosine similarity 0.3 is often worse than returning "no results found." Set a minimum similarity threshold (typically 0.7–0.8 for most text embedding models) and filter below it.

**5. Forgetting to namespace for multi-tenancy.** Without user-level namespacing or filtering, every user can retrieve every other user's documents. Test namespace isolation explicitly before deploying to production: index docs for user A, query as user B, verify zero cross-user retrieval.

**6. Over-relying on top-1 retrieval.** The closest vector is often correct, but not always. Always retrieve top-k (k ≥ 3) and pass all k chunks to the LLM. The second or third result sometimes contains the answer when the first doesn't.

## Project ideas

**1. ANN benchmark on your corpus** — Index 100K vectors with both HNSW and IVF in FAISS. Measure: query latency (P50/P95), recall@10 against flat search ground truth, index build time, memory usage. Plot the speed-recall curve at different nprobe / ef_search values. This makes the algorithm trade-offs concrete.

**2. Metadata filter performance study** — Build a Qdrant collection with 1M vectors and a `user_id` field with 1,000 distinct users. Measure query latency for searches filtered to 1 user (1K candidates), 100 users (100K candidates), and all users (1M candidates). Observe how filter selectivity affects latency. Set a minimum filter selectivity threshold for your SLO.

**3. pgvector vs Qdrant comparison** — Implement the same RAG pipeline using both pgvector (with HNSW index) and Qdrant. Index 500K vectors. Measure: query latency, storage size, filter performance, and operational complexity. Decide which fits your infrastructure better and document the trade-offs.

**4. Threshold tuning** — Run 100 semantic search queries where you know ground-truth relevance. Plot precision and recall at different similarity thresholds (0.5, 0.6, 0.7, 0.8, 0.9). Find the threshold that maximizes F1 for your corpus. This is the minimum similarity cutoff for your RAG pipeline.

## Going deeper

#### Foundational reading

- Johnson et al., "Billion-scale similarity search with GPUs" (FAISS paper, 2017) — the paper behind FAISS; explains IVF, PQ, and GPU-accelerated search.
- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2020) — the HNSW paper; the algorithm most production vector databases use.

#### Tools

- FAISS documentation (github.com/facebookresearch/faiss/wiki) — the reference for ANN indexing algorithms; useful even if you're not using FAISS directly.
- Qdrant documentation (qdrant.tech/documentation) — the most practical modern reference for production vector search with filtering.
- pgvector (github.com/pgvector/pgvector) — the simplest path if you're already on PostgreSQL; supports IVF and HNSW indexes.
