"""
Generate embeddings for the processed chunks.

Input:
    <repo>/data/processed/chunks.json

Outputs:
    <repo>/data/processed/embeddings.npy
    <repo>/data/processed/embeddings_meta.json

Pipeline position:
    process_documents.py -> chunks.json -> THIS FILE -> upload_to_vector_db.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
META_FILE = PROCESSED_DIR / "embeddings_meta.json"

MODEL_NAME = "intfloat/multilingual-e5-base"


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CHUNKS_FILE}. Run process_documents.py first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError("chunks.json is empty.")

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    print(
        f"Loading model '{MODEL_NAME}' "
        "(first run downloads the model)..."
    )
    model = SentenceTransformer(MODEL_NAME)

    # E5 models expect "passage:" for stored documents.
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    print("Encoding chunks into vectors...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_FILE, embeddings)

    # Keep enough metadata here to trace every vector back to its chunk
    # and original source.
    meta = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk.get("source_file"),
            "source_type": chunk.get("source_type"),
            "source": chunk.get("source"),
            "topic": chunk.get("topic"),
            "chunk_index_in_doc": chunk.get("chunk_index_in_doc"),
        }
        for chunk in chunks
    ]

    with META_FILE.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(
        f"\nSaved {embeddings.shape[0]} vectors "
        f"of dimension {embeddings.shape[1]}"
    )
    print(f"Embeddings: {EMBEDDINGS_FILE}")
    print(f"Metadata:   {META_FILE}")
    print("\nNext: run upload_to_vector_db.py")


if __name__ == "__main__":
    main()
