---
sidebar_position: 17
title: Knowledge Graphs
description: Entities, triples, GraphRAG, and hybrid retrieval — when structured knowledge beats flat vector search.
---

# Knowledge Graphs

## What it is

A knowledge graph represents information as entities connected by typed relationships. Data is stored as triples — subject, predicate, object — forming a network: `(Paris, capital_of, France)`, `(France, member_of, EU)`. This is unlike embedding vectors — a piece of text turned into a list of numbers so that similar meanings end up with similar numbers, then searched by comparing those numbers ("similarity search") — which flatten text into a single point in space with no explicit connections between points. A knowledge graph instead encodes *relationships* explicitly, making multi-hop queries possible — a "hop" is one edge traversal, so a two-hop question follows two connections in sequence: "which cities are capitals of EU member states?" traverses edges that a similarity search can't follow.

In the context of AI systems, knowledge graphs serve two roles. First, as a retrieval substrate for GraphRAG — hybrid retrieval that combines vector similarity search with graph traversal to answer relational questions. Second, as structured memory for agents that need to accumulate and query facts with more precision than embeddings provide.

## The problem it solves

Vector RAG (retrieval-augmented generation: searching a document store and pasting the relevant results into the prompt) retrieves chunks (smaller pieces a document has been split into) semantically similar to a query. It works well for factual lookup ("what does the return policy say?") but struggles with relational questions:

- "Which employees reported to Alice during the Q4 acquisition?" — requires traversing org-chart edges
- "What drugs interact with both compound A and compound B?" — requires multi-hop graph intersection
- "Which papers cite this method AND were published after 2022?" — filter + relationship + date

These queries require structured traversal over explicit relationships. A knowledge graph stores those relationships directly; vector search on embeddings cannot recover them.

Knowledge graphs also handle **entity resolution** — recognizing that "the CEO," "John Smith," and "JS" in different documents refer to the same node — a task that flat retrieval doesn't attempt.

## How it works under the hood

### Entity and relation extraction

Before building a graph, extract entities and relations from source documents:

```python
import anthropic
import json

client = anthropic.Anthropic()

EXTRACTION_PROMPT = """Extract entities and relationships from this text.

Return JSON:
{
  "entities": [{"name": "...", "type": "person|org|place|concept"}],
  "relations": [{"subject": "...", "predicate": "...", "object": "..."}]
}

Text: {text}

Return only JSON."""

ALLOWED_PREDICATES = {
    "works_at", "reports_to", "founded", "acquired", "located_in", "role", "member_of"
}
MAX_ENTITY_LEN = 100

def extract_triples(text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}]
    )
    try:
        raw = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}

    # Validate before writing to graph — treat LLM output as untrusted input.
    # A prompt injection in source documents could cause the LLM to emit
    # malicious predicates or oversized strings that corrupt the graph schema.
    valid_relations = [
        r for r in raw.get("relations", [])
        if r.get("predicate") in ALLOWED_PREDICATES
        and len(r.get("subject", "")) <= MAX_ENTITY_LEN
        and len(r.get("object", "")) <= MAX_ENTITY_LEN
    ]
    return {"entities": raw.get("entities", []), "relations": valid_relations}

# Example
doc = "Alice Chen joined Acme Corp in 2019 as VP of Engineering. She reports to CEO Bob Kim."
result = extract_triples(doc)
print(result["relations"])
# [{"subject": "Alice Chen", "predicate": "works_at", "object": "Acme Corp"},
#  {"subject": "Alice Chen", "predicate": "reports_to", "object": "Bob Kim"}]
```

### Graph storage and traversal

Store triples in a graph structure. NetworkX works for small graphs; Neo4j for production scale:

```python
import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # a graph structure allowing more than
        # one directed edge between the same two nodes (e.g. Alice both
        # "works_at" and "reports_to" Bob)

    def add_triple(self, subject: str, predicate: str, obj: str, **attrs):
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(subject, obj, predicate=predicate, **attrs)

    def neighbors(self, entity: str, predicate: str | None = None) -> list[tuple]:
        """Return (neighbor, predicate) pairs for an entity."""
        results = []
        for _, neighbor, data in self.graph.out_edges(entity, data=True):
            if predicate is None or data.get("predicate") == predicate:
                results.append((neighbor, data["predicate"]))
        return results

    def multi_hop(self, start: str, hops: int = 2) -> set[str]:
        """Return all entities reachable within N hops (one hop = one edge traversal)."""
        return set(nx.single_source_shortest_path_length(
            self.graph, start, cutoff=hops
        ).keys())

    def find_path(self, source: str, target: str) -> list[str] | None:
        """Return shortest path between two entities."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

kg = KnowledgeGraph()
kg.add_triple("Alice Chen", "works_at", "Acme Corp")
kg.add_triple("Alice Chen", "reports_to", "Bob Kim")
kg.add_triple("Bob Kim", "works_at", "Acme Corp")
kg.add_triple("Bob Kim", "role", "CEO")

print(kg.neighbors("Alice Chen"))
# [('Acme Corp', 'works_at'), ('Bob Kim', 'reports_to')]
print(kg.multi_hop("Alice Chen", hops=2))
# {'Alice Chen', 'Acme Corp', 'Bob Kim', 'CEO'}
```

### GraphRAG — hybrid vector + graph retrieval

Vector search retrieves chunks similar to a query but loses the *relationships between* entities — a question like "who reported to the VP who approved the merger?" is unanswerable from embeddings alone. GraphRAG addresses this by using vector similarity to find entry-point entities, then following graph edges to gather their relational context.

Always set a hop limit — unbounded traversal on a large graph can return thousands of entities (a DoS risk — denial of service, meaning the system becomes too slow or overloaded to serve anyone) and cross tenant boundaries (a "tenant" is one customer or organization sharing infrastructure with others; "crossing" means one tenant's data leaking into another's results) in multi-user deployments.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

class GraphRAG:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.entity_embeddings: dict[str, np.ndarray] = {}

    def index_entities(self):
        entities = list(self.kg.graph.nodes())
        if not entities:
            return
        embeddings = embed_model.encode(entities, normalize_embeddings=True)
        for entity, emb in zip(entities, embeddings):
            self.entity_embeddings[entity] = emb

    def find_entry_entities(self, query: str, top_k: int = 3) -> list[str]:
        """Find the entities most relevant to the query."""
        if not self.entity_embeddings:
            return []
        query_emb = embed_model.encode([query], normalize_embeddings=True)[0]
        entities = list(self.entity_embeddings.keys())
        embeddings = np.array([self.entity_embeddings[e] for e in entities])
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [entities[i] for i in top_indices]

    def retrieve(self, query: str, hops: int = 2) -> str:
        """Return a text summary of the graph context relevant to the query."""
        entry_entities = self.find_entry_entities(query)
        context_entities = set()
        for entity in entry_entities:
            context_entities |= self.kg.multi_hop(entity, hops=hops)

        triples = []
        for u, v, data in self.kg.graph.edges(data=True):
            if u in context_entities or v in context_entities:
                triples.append(f"{u} --[{data['predicate']}]--> {v}")

        return "\n".join(triples) if triples else "No relevant graph context found."

rag = GraphRAG(kg)
rag.index_entities()
context = rag.retrieve("Who does Alice report to?")
print(context)
# Alice Chen --[works_at]--> Acme Corp
# Alice Chen --[reports_to]--> Bob Kim
# Bob Kim --[works_at]--> Acme Corp
# Bob Kim --[role]--> CEO
```

### Neo4j for production scale

For graphs with millions of nodes, use Neo4j with Cypher queries. Cypher uses `(node)` for nodes and `-[edge]->` for directed relationships — the ASCII pattern mirrors the graph structure.

:::caution[Security: credentials and tenant isolation]

Never hardcode Neo4j credentials. Load them from environment variables. In multi-tenant deployments, label every node with a `tenant_id` and include it in all MATCH/MERGE conditions — without this, multi-hop traversal can cross tenant boundaries and expose one user's data to another.

:::

```python
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)

def add_triple_neo4j(subject: str, predicate: str, obj: str):
    with driver.session() as session:
        session.run(
            "MERGE (s:Entity {name: $subject}) "
            "MERGE (o:Entity {name: $object}) "
            "MERGE (s)-[r:RELATION {type: $predicate}]->(o)",
            subject=subject, predicate=predicate, object=obj
        )

def query_neighbors_neo4j(entity: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(
            "MATCH (e:Entity {name: $name})-[r]->(neighbor) "
            "RETURN r.type AS predicate, neighbor.name AS neighbor",
            name=entity
        )
        return [{"predicate": r["predicate"], "neighbor": r["neighbor"]} for r in result]

# Multi-hop: find everything 2 hops from Alice Chen
def multi_hop_neo4j(entity: str, hops: int = 2) -> list[str]:
    with driver.session() as session:
        result = session.run(
            f"MATCH (e:Entity {{name: $name}})-[*1..{hops}]->(neighbor) "
            "RETURN DISTINCT neighbor.name AS name",
            name=entity
        )
        return [r["name"] for r in result]
```

## Concrete example

End-to-end pipeline that builds a knowledge graph from documents and answers relational queries:

```python
import anthropic
import networkx as nx
import json

client = anthropic.Anthropic()

DOCS = [
    "Alice Chen is VP of Engineering at Acme Corp. She reports to Bob Kim, the CEO.",
    "Bob Kim founded Acme Corp in 2015. The company is headquartered in San Francisco.",
    "Alice Chen previously worked at Beta Inc before joining Acme Corp.",
    "Beta Inc was acquired by Acme Corp in 2021.",
]

def build_graph_from_docs(docs: list[str]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for doc in docs:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""Extract all entities and relationships.
Return JSON: {{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
Text: {doc}
Only JSON."""}]
        )
        try:
            extracted = json.loads(response.content[0].text)
            for rel in extracted.get("relations", []):
                graph.add_edge(rel["subject"], rel["object"], predicate=rel["predicate"])
        except json.JSONDecodeError:
            continue
    return graph

def answer_with_graph(query: str, graph: nx.MultiDiGraph) -> str:
    # Serialize relevant graph as context
    edges = [(u, v, d["predicate"]) for u, v, d in graph.edges(data=True)]
    graph_context = "\n".join(f"{u} --[{p}]--> {v}" for u, v, p in edges)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": f"""Answer using only the knowledge graph below.

Knowledge graph:
{graph_context}

Question: {query}"""}]
    )
    return response.content[0].text

graph = build_graph_from_docs(DOCS)
print(f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

queries = [
    "Who does Alice Chen report to?",
    "What company did Alice work at before Acme Corp?",
    "What happened to Beta Inc?",
]
for q in queries:
    print(f"\nQ: {q}")
    print(f"A: {answer_with_graph(q, graph)}")
```

## When to use it / when not to

#### Knowledge graphs are the right choice when

- Queries require multi-hop relational reasoning ("find all employees who joined after the acquisition of Beta Inc")
- Entity identity matters — multiple documents refer to the same entity with different names
- Your domain has a natural ontology — a fixed, well-understood set of categories and relationships, the way an org chart or a drug database already has one: org charts, drug databases, code dependency graphs, legal entity hierarchies
- You need to explain *why* two entities are related, not just that their text is similar

#### Prefer vector RAG when

- Queries are factual lookup — "what does the policy say about returns?" — with no relational structure
- Documents don't have consistent entity mentions that would survive extraction
- Your corpus is unstructured narrative where relation extraction would be noisy
- Development speed matters more than relational precision

#### The practical question

Can you write the query in the form "find X where X is related to Y via Z"? If yes, a knowledge graph helps. If your queries are "find text that talks about topic T," vector RAG is sufficient and cheaper to build.

:::tip[My take]

Knowledge graphs are underused for structured enterprise data and overused for free-form text corpora. If your documents are org charts, dependency trees, medical ontologies, or legal hierarchies, a knowledge graph gives you the relational queries that embeddings can't. If your documents are customer support tickets, blog posts, or policy PDFs, the extraction noise will dominate and vector RAG will outperform a noisy graph.

The hardest part is entity resolution — recognizing that "Alice," "Alice Chen," and "Ms. Chen, VP Eng" refer to the same node. Without it, your graph fragments. Use an LLM to do fuzzy entity normalization before insertion, or use a dedicated library like spaCy's entity linker with a known entity list.

GraphRAG (vector entry-point + graph traversal) is a useful hybrid: it tolerates imperfect extraction because it falls back to semantic similarity. Use the full graph for structured queries; use GraphRAG for mixed natural-language questions.

:::

## Main tools and libraries

| Tool | Use for |
|---|---|
| NetworkX | In-memory knowledge graphs for prototyping and small corpora |
| Neo4j | Production graph database; Cypher query language; handles millions of nodes |
| `neo4j` Python driver | Connecting to Neo4j from Python |
| spaCy + `spacy-transformers` | Entity extraction and entity linking from text |
| Microsoft GraphRAG | Reference implementation of graph-augmented RAG from Microsoft Research |
| LlamaIndex graph connectors | LlamaIndex's graph index abstractions for GraphRAG pipelines |
| Wikidata / SPARQL | Open knowledge base for bootstrapping entity ontologies |

## Common failure modes and gotchas

**1. Noisy entity extraction.** LLM-based extraction is imperfect, especially for co-references and implicit relations. A sentence like "the company was sold" only resolves if you know which company the pronoun refers to. Always review a sample of extracted triples before trusting the graph.

**2. Entity fragmentation from poor normalization.** "Alice Chen," "Alice," and "A. Chen" create three separate nodes unless you normalize. Build a normalization pass that clusters near-identical entity names before insertion.

**3. Graph sparsity.** If documents don't explicitly state relationships, extraction won't find them. A graph built from summaries and abstracts will be sparser than one built from full documents. Measure edge-to-node ratio — below 2:1 is often too sparse for meaningful traversal.

**4. Query scope explosion.** Multi-hop traversal without hop limits retrieves everything. A 2-hop query from a highly connected node may return thousands of triples — more noise than signal. Always set a hop limit and filter by relevance before passing graph context to the LLM.

**5. Temporal drift.** Knowledge graphs go stale as entities change state. "Alice reports to Bob" may be true now but wrong in 6 months. Add timestamps to edges and filter on recency for time-sensitive domains.

**6. Confusing graph retrieval with reasoning.** Retrieving graph context gives the LLM structured facts; it doesn't guarantee the LLM will reason correctly over them. Test relational queries on a held-out set — LLMs still make multi-hop reasoning errors even with correct graph context.

## Project ideas

**1. Build a document knowledge graph** — Take 20 Wikipedia articles from a single domain (e.g., tech companies or historical events). Extract entities and relations using Claude. Build a NetworkX graph. Write 10 multi-hop queries by hand and measure how many the graph can answer correctly.

**2. GraphRAG vs vector RAG comparison** — On the same corpus, build a vector index and a knowledge graph. Write 20 queries: 10 factual lookup and 10 relational. Measure retrieval accuracy for each approach on each query type. Graph the results.

**3. Entity resolution pipeline** — Take a corpus where the same entities are mentioned differently across documents (company names, person names). Build a normalization pipeline: extract all entity mentions, cluster near-duplicates by embedding similarity, assign canonical names. Measure how many disconnected nodes you eliminate.

**4. Neo4j GraphRAG with Cypher** — Deploy Neo4j locally. Build a graph from structured data (CSV of org chart or dependency data). Write Cypher queries for 5 relational questions. Compare Cypher precision vs. LLM graph-context answer precision on the same questions.

## Going deeper

#### Foundational reading

- Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research, 2024) — the reference GraphRAG paper; community detection + local search + global search over graphs.
- Hogan et al., "Knowledge Graphs" (ACM Computing Surveys, 2021) — comprehensive survey covering representation, construction, reasoning, and embedding of knowledge graphs.

#### Tools and resources

- Neo4j documentation (neo4j.com/docs) — reference for Cypher query language and graph data modeling; the graph data modeling guide is especially useful for schema design.
- Microsoft GraphRAG (github.com/microsoft/graphrag) — production-grade implementation with community detection, entity summarization, and global/local query modes.
- spaCy entity linker documentation (spacy.io/api/entitylinker) — reference for linking extracted mentions to known entities in a knowledge base.
