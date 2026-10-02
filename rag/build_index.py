"""
Builds the RAG index from the source documents in rag/source_docs/.

Run it after any change to a source document or to rag/chunker.py:
    python -m rag.build_index
"""

import os
from rag.chunker import parse_docx
from rag.vector_store import index_ready, rebuild_index

SOURCE_DOCS = [
    "Dressing Selection & Application.docx",
    "Immediate First-Aid & Safety.docx",
    "Rehabilitation__Scar___Deformity_Prevention.docx",
    "Triage, Severity Assessment.docx",
]

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "source_docs")


def build():
    all_chunks = []
    for fname in SOURCE_DOCS:
        path = os.path.join(SOURCE_DIR, fname)
        if not os.path.exists(path):
            print(f"  [build_index] WARNING: {fname} not found, skipping.")
            continue
        chunks = parse_docx(path)
        clinician_count = sum(1 for c in chunks if c["audience"] == "clinician_only")
        print(f"  [build_index] {fname}: {len(chunks)} chunks "
              f"({clinician_count} clinician_only)")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  [build_index] No guidance documents found in rag/source_docs/ (they are not part of the public "
              "repository - see rag/source_docs/README.md). The index is left empty and the app will answer "
              "'no guidance available' to treatment questions.")
        return []

    print(f"  [build_index] Total: {len(all_chunks)} chunks across {len(SOURCE_DOCS)} documents.")
    rebuild_index(all_chunks)
    return all_chunks


def ensure_index():
    """Build the index only if it is missing or empty. The app calls this at start-up, so a fresh checkout works
    without a manual build step (the generated index is not kept in git)."""
    if not index_ready():
        from rag.corpus import fetch_corpus_if_configured
        fetch_corpus_if_configured()              # a deployment may keep the documents in a private repo
        print("  [build_index] No search index found - building it from rag/source_docs/ ...")
        build()


if __name__ == "__main__":
    build()