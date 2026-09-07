"""
Upload processed chunks and their embeddings to Qdrant.

Inputs:
    <repo>/data/processed/chunks.json
    <repo>/data/processed/embeddings.npy
    <repo>/data/processed/embeddings_meta.json

Pipeline position:
    generate_embeddings.py -> THIS FILE -> Qdrant -> search.py / RAG pipeline

Secrets are read from environment variables / .env:
    QDRANT_URL
    QDRANT_API_KEY
    QDRANT_COLLECTION (optional)
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
META_FILE = PROCESSED_DIR / "embeddings_meta.json"

load_dotenv(REPO_ROOT / ".env")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "medical_chunks")

UPLOAD_BATCH_SIZE = 128


def require_inputs():
    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set. Add it to .env or the environment."
        )

    for path in (CHUNKS_FILE, EMBEDDINGS_FILE, META_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run process_documents.py and "
                "generate_embeddings.py first."
            )


def main():
    require_inputs()

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with META_FILE.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embeddings array, got shape {embeddings.shape}"
        )

    if embeddings.shape[0] != len(meta):
        raise ValueError(
            "embeddings.npy and embeddings_meta.json are out of sync: "
            f"{embeddings.shape[0]} vectors vs {len(meta)} metadata entries."
        )

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    missing_ids = [
        item["chunk_id"]
        for item in meta
        if item["chunk_id"] not in chunk_lookup
    ]
    if missing_ids:
        raise ValueError(
            f"Metadata contains chunk IDs not present in chunks.json: "
            f"{missing_ids[:10]}"
        )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )

    vector_size = embeddings.shape[1]

    print(
        f"Preparing Qdrant collection '{COLLECTION_NAME}' "
        f"with vector size {vector_size}..."
    )

    # Rebuild the collection so the uploaded dataset always corresponds
    # exactly to the current chunks/embeddings files.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for i, item in enumerate(meta):
        chunk = chunk_lookup[item["chunk_id"]]

        payload = {
            "text": chunk["text"],
            "topic": chunk.get("topic"),
            "source_file": chunk.get("source_file"),
            "source_type": chunk.get("source_type"),
            "source": chunk.get("source"),
            "original_source": chunk.get("original_source"),
            "license": chunk.get("license"),
            "attribution": chunk.get("attribution"),
            "chunk_index_in_doc": chunk.get("chunk_index_in_doc"),
            "word_count": chunk.get("word_count"),
        }

        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=embeddings[i].tolist(),
                payload=payload,
            )
        )

    print(
        f"Uploading {len(points)} points "
        f"in batches of {UPLOAD_BATCH_SIZE}..."
    )

    for start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[start:start + UPLOAD_BATCH_SIZE]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        uploaded = min(
            start + UPLOAD_BATCH_SIZE,
            len(points),
        )
        print(f"  ...uploaded {uploaded}/{len(points)}")

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(
        f"\nDone. Collection '{COLLECTION_NAME}' "
        f"now has {count} points."
    )
    print(f"Qdrant: {QDRANT_URL}")


if __name__ == "__main__":
    main()
