"""
Step 2 – Embedding & Vector Database
--------------------------------------
Generates sentence embeddings for all provisions and stores them in a
persistent ChromaDB collection for semantic retrieval.

Usage:
    python -m src.embedding
"""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL, RAW_DIR

# ── Module-level singletons (loaded once, reused across all calls) ─────────────
_model = None          # type: SentenceTransformer | None
_chroma_client = None  # type: chromadb.PersistentClient | None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def load_provisions(path: Path | None = None) -> list[dict]:
    """Load the combined provisions JSON."""
    p = path or RAW_DIR / "all_provisions.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Provisions file not found: {p}\n"
            "Run src/extraction.py first."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(provisions: list[dict]) -> chromadb.Collection:
    """
    Embed all provisions and upsert into ChromaDB.
    Returns the ChromaDB collection object.
    """
    print(f"\nLoading embedding model: {EMBED_MODEL} …")
    model = _get_model()

    print("Initialising ChromaDB …")
    client = _get_client()

    # Reset collection if it already exists (re-run safe)
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"  Existing collection '{CHROMA_COLLECTION}' deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 256
    total  = len(provisions)
    print(f"Embedding {total:,} provisions in batches of {BATCH} …")

    for start in tqdm(range(0, total, BATCH), unit="batch"):
        batch = provisions[start : start + BATCH]
        texts = [p["text"] for p in batch]
        ids   = [p["id"]   for p in batch]

        # Metadata must be scalar types only
        metas = [
            {
                "agreement":     p["agreement"],
                "doc_type":      p["doc_type"],
                "chapter":       p.get("chapter", "")[:200],
                "article":       p.get("article", "")[:100],
                "paragraph_idx": int(p.get("paragraph_idx", 0)),
                "char_count":    int(p.get("char_count", 0)),
            }
            for p in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )

    print(f"\n✅ Vector store built: {collection.count():,} provisions indexed")
    print(f"   Collection: '{CHROMA_COLLECTION}'  →  {CHROMA_DIR}")
    return collection


def get_collection() -> chromadb.Collection:
    """Return an existing ChromaDB collection (read-only access)."""
    return _get_client().get_collection(CHROMA_COLLECTION)


def retrieve_similar(
    query: str,
    agreement_filter: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Retrieve the top-n most similar provisions for a query string.

    Args:
        query:            Natural-language query or a provision text.
        agreement_filter: If set, restrict results to one agreement (RCEP / AHKFTA / AANZFTA).
        n_results:        Number of results to return.

    Returns:
        List of dicts with keys: id, text, agreement, article, distance.
    """
    model      = _get_model()   # cached — no reload
    collection = get_collection()

    embedding = model.encode([query]).tolist()
    where     = {"agreement": agreement_filter} if agreement_filter else None

    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        output.append(
            {
                "id":        doc_id,
                "text":      results["documents"][0][i],
                "agreement": results["metadatas"][0][i]["agreement"],
                "doc_type":  results["metadatas"][0][i]["doc_type"],
                "chapter":   results["metadatas"][0][i]["chapter"],
                "article":   results["metadatas"][0][i]["article"],
                "distance":  round(results["distances"][0][i], 4),
            }
        )
    return output


if __name__ == "__main__":
    provisions = load_provisions()
    build_vector_store(provisions)
