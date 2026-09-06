"""
Retrieval - takes a question, converts it to a vector using the same model
used for the chunks, and finds the most similar chunks in the EXTERNAL
vector DB (Qdrant) instead of a local NumPy array.

=== EDIT LOG (see CHANGELOG.md, item #7) ===
- [FIX / redesign] `load_data()` no longer loads embeddings.npy into RAM.
  It now returns a QdrantClient connected to the external DB.
- [FIX / redesign] `search()` now calls client.search(...) instead of
  doing `embeddings @ query_vec` locally.
- Kept the same function names/signatures used by rag_pipeline.py where
  possible, so the rest of the pipeline barely has to change.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 3

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "medical_chunks")


def load_data():
    """Returns (qdrant_client, collection_name). Kept as a 2-tuple (instead
    of the old 3-tuple of chunk_lookup/meta/embeddings) since the vector DB
    now holds everything - see upload_to_vector_db.py."""
    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set. Copy .env.example to .env and fill in "
            "your Qdrant Cloud credentials (see upload_to_vector_db.py)."
        )
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
    return client, COLLECTION_NAME


def search(query: str, model, client, collection_name: str, top_k: int = TOP_K):
    # e5 models require a "query: " prefix on questions (chunks used "passage: ")
    query_vec = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # [FIX] newer qdrant-client versions removed client.search() in favor
    # of client.query_points(), which returns a QueryResponse whose actual
    # hits live in .points instead of being the return value directly.
    response = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
    )
    hits = response.points

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append({
            "score": float(hit.score),
            "chunk_id": hit.id,
            "source_file": payload.get("source_file"),
            "topic": payload.get("topic"),
            "text": payload.get("text", ""),
        })
    return results


def main():
    print("Loading model and connecting to Qdrant (takes a few seconds)...")
    client, collection_name = load_data()
    model = SentenceTransformer(MODEL_NAME)
    count = client.count(collection_name=collection_name).count
    print(f"Ready. Searching over {count} chunks in '{collection_name}'.\n")
    print("Type a medical question (English or Arabic). Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        results = search(query, model, client, collection_name)
        print()
        for i, r in enumerate(results, start=1):
            print(f"[{i}] score={r['score']:.3f}  source={r['source_file']}")
            print(f"    {r['text'][:250]}...")
            print()


if __name__ == "__main__":
    main()
