"""
Builds and queries the RAG vector store using:
- sentence-transformers (all-mpnet-base-v2) for LOCAL, free embeddings
  -- no API key, no network dependency at query time, fine on a CPU-only
  laptop at this corpus's small scale.
- Chroma as the vector DB -- chosen specifically because it has
  built-in metadata filtering (audience="patient_safe" vs
  "clinician_only"), which is the core mechanism enforcing that
  dosage-level content never reaches chat-facing retrieval.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = os.path.join(os.path.dirname(__file__), "vector_store")
COLLECTION_NAME = "burn_guidance"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )
    return _collection


def index_ready() -> bool:
    """True when the vector store already holds passages (so it does not need building)."""
    try:
        return _get_collection().count() > 0
    except Exception:
        return False


def rebuild_index(chunks: list):
    """
    Wipes and rebuilds the vector store from scratch with the given
    chunks. Call this whenever source documents change.

    Args:
        chunks: list of {"text": str, "audience": str, "source": str,
                         "section": str, "topic": str}
    """
    global _client, _collection
    _client = chromadb.PersistentClient(path=DB_PATH)

    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- fine

    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )

    # The stored document is the chunk text (what is shown and cited); the EMBEDDING
    # also sees the section title, which gives every chunk its topical context.
    embed_inputs = [f"{c['section']}. {c['text']}" if c.get("section") else c["text"] for c in chunks]
    embeddings = [list(map(float, e)) for e in _embedding_fn(embed_inputs)]
    _collection.add(
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"audience": c["audience"], "source": c["source"],
                    "section": c.get("section", ""), "topic": c.get("topic", "general")} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"  [vector_store] Indexed {len(chunks)} chunks.")


def query(query_text: str, top_k: int = 5, audience: str = "patient_safe") -> list:
    """
    Args:
        query_text: natural-language query.
        top_k: how many passages to retrieve.
        audience: "patient_safe" (chat-facing, default), "clinician_only"
                  (escalation-report-only, unrestricted single-category
                  lookup), or "all" (no filter, mixed results -- debug
                  use only, never used by real tool code).

    Returns:
        list of {"text", "source", "section", "topic", "distance"} (Chroma L2 distance;
        smaller = closer)
    """
    collection = _get_collection()

    where_filter = None
    if audience == "patient_safe":
        where_filter = {"audience": "patient_safe"}
    elif audience == "clinician_only":
        where_filter = {"audience": "clinician_only"}
    # audience == "all" -> no filter

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        passages.append({"text": doc, "source": meta.get("source", "Unknown source"),
                         "section": meta.get("section", ""), "topic": meta.get("topic", "general"),
                         "distance": round(float(dist), 3)})

    return passages