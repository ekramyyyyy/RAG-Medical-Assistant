"""
Process ALL raw documents produced by the dataset-building scripts.

Inputs:
    <repo>/data/raw/*.txt

Output:
    <repo>/data/processed/chunks.json

Pipeline position:
    build_dataset_from_medquad.py
                 \
    build_dataset_from_wikipedia_category.py -> data/raw -> THIS FILE
                 /
    extract_book_pdf.py

Then:
    chunks.json -> generate_embeddings.py -> embeddings
"""

import json
import re
from pathlib import Path

import pysbd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.json"

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 60

_segmenter = pysbd.Segmenter(language="en", clean=False)


def parse_raw_document(raw_text: str):
    """Read standardized KEY: value metadata before the first blank line."""
    lines = raw_text.splitlines()
    metadata = {}
    body_start = 0

    for i, line in enumerate(lines[:20]):
        match = re.match(r"^([A-Z][A-Z_ ]*):\s*(.*)$", line)

        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
            body_start = i + 1
        elif not line.strip():
            body_start = i + 1
            break
        else:
            break

    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_by_sentence(text: str, chunk_size: int, overlap: int):
    sentences = _segmenter.segment(text)
    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > chunk_size:
            chunks.append(" ".join(current))

            carried = []
            carried_words = 0

            for previous_sentence in reversed(current):
                words = len(previous_sentence.split())

                if carried_words + words > overlap:
                    break

                carried.insert(0, previous_sentence)
                carried_words += words

            current = carried
            current_words = carried_words

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}\n"
            "Run the three dataset-building scripts first."
        )

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise RuntimeError(
            f"No .txt files found in {RAW_DIR}. "
            "Build the raw dataset first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_id = 1

    for file_path in raw_files:
        if "list_of" in file_path.name.lower():
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        metadata, body = parse_raw_document(raw_text)

        if not body.strip():
            print(f"WARNING: empty document skipped: {file_path.name}")
            continue

        source_type = metadata.get("SOURCE_TYPE", "unknown")
        source_name = metadata.get("SOURCE", file_path.stem)
        topic = metadata.get(
            "TOPIC",
            file_path.stem.replace("_", " ").title(),
        )
        original_source = metadata.get("ORIGINAL_SOURCE")
        attribution = metadata.get("ATTRIBUTION")
        license_name = metadata.get("LICENSE")

        cleaned_body = clean_text(body)
        chunks = chunk_by_sentence(
            cleaned_body,
            CHUNK_SIZE_WORDS,
            CHUNK_OVERLAP_WORDS,
        )

        for chunk_index, chunk in enumerate(chunks):
            tagged_text = f"[Topic: {topic}] {chunk}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": file_path.name,
                    "source_type": source_type,
                    "source": source_name,
                    "topic": topic,
                    "original_source": original_source,
                    "license": license_name,
                    "attribution": attribution,
                    "chunk_index_in_doc": chunk_index,
                    "word_count": len(tagged_text.split()),
                    "text": tagged_text,
                }
            )

            chunk_id += 1

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    document_count = len({c["source_file"] for c in all_chunks})

    print(f"Processed {document_count} document(s)")
    print(f"Created {len(all_chunks)} chunks")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
