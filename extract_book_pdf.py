"""
Extract the medical reference book into the same raw-document format
used by MedQuAD and Wikipedia.

Input:
    <repo>/data/raw_source/ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf

Output:
    <repo>/data/raw/book_gale_encyclopedia_of_medicine.txt

The PDF path below is intentionally an example so the repository does
not depend on one developer's local machine.
"""

import re
from pathlib import Path

import pdfplumber

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---- EDIT THIS PATH IF YOUR PDF IS STORED ELSEWHERE ----
PDF_PATH = REPO_ROOT / "data" / "raw_source" / "ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
# ---------------------------------------------------------

BOOK_TITLE = "The Gale Encyclopedia of Medicine (Second Edition)"

# Keep the license/terms explicit instead of claiming a license that
# may not apply to the specific copy of the book.
LICENSE = "Verify actual license/terms of use before deployment"

ATTRIBUTION = (
    "Longe, J. L. (Ed.). The Gale Encyclopedia of Medicine "
    "(2nd ed., Vol. 1-5). Gale Group. "
    "Available at https://archive.org/details/galeencyclopedia0000unse_h3o0"
)

RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_SLUG = "book_gale_encyclopedia_of_medicine"


def extract_text(pdf_path: Path) -> str:
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            pages_text.append(page.extract_text() or "")

            if page_number % 100 == 0:
                print(f"  ...processed {page_number}/{total} pages")

    return "\n".join(pages_text)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Put the PDF at the example path above or edit PDF_PATH."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting text from: {PDF_PATH.name}")
    cleaned = clean_text(extract_text(PDF_PATH))

    header = (
        "SOURCE_TYPE: book\n"
        f"SOURCE: {BOOK_TITLE}\n"
        f"LICENSE: {LICENSE}\n"
        f"TOPIC: {BOOK_TITLE}\n"
        f"ATTRIBUTION: {ATTRIBUTION}\n\n"
    )

    output_path = RAW_DIR / f"{OUTPUT_SLUG}.txt"
    output_path.write_text(header + cleaned, encoding="utf-8")

    print(f"\nSaved: {output_path}")
    print(f"Word count: {len(cleaned.split()):,}")


if __name__ == "__main__":
    main()
