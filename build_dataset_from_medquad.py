"""
Build the MedQuAD part of the medical-assistant dataset.

Input:
    <repo>/MedQuAD/**/*.xml

Output:
    <repo>/data/raw/*.txt

This is step 1 of the data pipeline:
MedQuAD -> data/raw -> process_documents.py
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MEDQUAD_DIR = REPO_ROOT / "MedQuAD"
RAW_DIR = REPO_ROOT / "data" / "raw"

SKIP_SOURCE_DIRS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}

TARGET_DISEASES = {
    "diabetes": "diabetes",
    "hypertension": "high blood pressure",
    "asthma": "asthma",
    "copd": "copd",
    "coronary_heart_disease": "coronary heart disease",
    "stroke": "stroke",
    "obesity": "overweight and obesity",
    "depression": "depression",
    "anxiety_disorders": "anxiety disorders",
    "rheumatoid_arthritis": "rheumatoid arthritis",
    "osteoarthritis": "osteoarthritis",
    "migraine": "migraine",
    "epilepsy": "epilepsy",
    "alzheimers_disease": "alzheimer's disease",
    "parkinsons_disease": "parkinson's disease",
    "kidney_disease": "kidney disease",
    "anemia": "anemia",
    "hepatitis_b": "what i need to know about hepatitis b",
    "hepatitis_c": "what i need to know about hepatitis c",
    "pneumonia": "pneumonia",
    "tuberculosis": "tuberculosis (tb)",
    "hiv_aids": "hiv/aids",
    "psoriasis": "psoriasis",
    "eczema": "eczema",
    "osteoporosis": "osteoporosis",
    "hypothyroidism": "hypothyroidism",
    "hyperthyroidism": "hyperthyroidism",
    "gerd": "gerd",
    "peptic_ulcer": "peptic ulcer",
    "irritable_bowel_syndrome": "irritable bowel syndrome",
    "celiac_disease": "celiac disease",
    "crohns_disease": "crohn's disease",
    "ulcerative_colitis": "ulcerative colitis",
    "gallstones": "gallstones",
    "kidney_stones": "kidney stones in adults",
    "urinary_tract_infection": "urinary tract infections",
    "prostate_cancer": "prostate cancer",
    "breast_cancer": "breast cancer",
    "lung_cancer": "lung cancer",
    "colon_cancer": "colon cancer",
    "melanoma": "melanoma",
    "multiple_sclerosis": "multiple sclerosis",
    "lupus": "lupus",
    "fibromyalgia": "fibromyalgia",
    "sleep_apnea": "sleep apnea",
    "insomnia": "insomnia",
    "sinusitis": "sinusitis",
    "gout": "gout",
    "common_cold": "common cold",
    "flu": "flu",
    "fever": "fever",
    "gastroenteritis": "gastroenteritis",
    "hay_fever": "hay fever",
    "food_allergy": "food allergy",
    "allergy": "allergy",
    "chickenpox": "chickenpox",
    "measles": "measles",
    "mumps": "mumps",
    "rubella": "rubella",
    "sore_throat": "sore throat",
    "ear_infections": "ear infections",
    "diarrhea": "diarrhea",
    "constipation": "constipation",
    "headache": "headache",
    "cough": "cough",
    "acute_bronchitis": "acute bronchitis",
}


def build_focus_index() -> dict[str, list[Path]]:
    index = {}
    for xml_file in MEDQUAD_DIR.glob("*/*.xml"):
        if xml_file.parent.name in SKIP_SOURCE_DIRS:
            continue
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue

        focus = root.findtext("Focus")
        if focus:
            index.setdefault(focus.strip().lower(), []).append(xml_file)
    return index


def extract_qa_pairs(xml_file: Path):
    root = ET.parse(xml_file).getroot()
    url = root.attrib.get("url", "")
    source = root.attrib.get("source", "")
    pairs = []

    for qa in root.iter("QAPair"):
        question = qa.findtext("Question")
        answer = qa.findtext("Answer")
        if question and answer and answer.strip():
            pairs.append((question.strip(), answer.strip()))

    return pairs, url, source


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def build_document(disease_name: str, xml_files: list[Path]) -> str:
    seen_answers = set()
    sections = []
    sources_seen = []

    for xml_file in xml_files:
        pairs, url, source = extract_qa_pairs(xml_file)

        if url and (url, source) not in sources_seen:
            sources_seen.append((url, source))

        for question, answer in pairs:
            key = answer[:200]
            if key in seen_answers:
                continue

            seen_answers.add(key)
            sections.append(f"Q: {question}\n{answer}")

    header_lines = [
        "SOURCE_TYPE: medquad",
        "SOURCE: MedQuAD dataset (NIH)",
        "LICENSE: Creative Commons Attribution 4.0 (CC BY 4.0)",
        f"TOPIC: {disease_name}",
    ]

    for url, source in sources_seen:
        header_lines.append(f"ORIGINAL_SOURCE: {source} - {url}")

    return clean_whitespace(
        "\n".join(header_lines) + "\n\n" + "\n\n".join(sections)
    )


def main():
    if not MEDQUAD_DIR.exists():
        raise FileNotFoundError(
            f"MedQuAD directory not found: {MEDQUAD_DIR}\n"
            "Example: <repo>/MedQuAD/"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    focus_index = build_focus_index()

    found, missing = [], []

    for slug, focus_key in TARGET_DISEASES.items():
        xml_files = focus_index.get(focus_key)
        if not xml_files:
            missing.append(focus_key)
            continue

        document = build_document(focus_key.title(), xml_files)
        output_path = RAW_DIR / f"{slug}.txt"
        output_path.write_text(document, encoding="utf-8")
        found.append((slug, len(document.split())))

    print(f"Built {len(found)} MedQuAD disease documents in {RAW_DIR}")
    if missing:
        print(f"Not found ({len(missing)}): {missing}")
    print(f"Total words: {sum(words for _, words in found):,}")


if __name__ == "__main__":
    main()
