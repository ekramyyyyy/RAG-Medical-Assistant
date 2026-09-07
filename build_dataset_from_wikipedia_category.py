"""
Build the Wikipedia part of the medical-assistant dataset.

Input:
    Wikipedia API

Output:
    <repo>/data/raw/wiki_*.txt

This is step 2 of the data pipeline:
Wikipedia -> data/raw -> process_documents.py
"""

import re
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MedicalAssistantProject/1.0 (student project; contact: student@example.com)"

CATEGORIES = [
    "Category:Infectious diseases",
    "Category:Cardiovascular diseases",
    "Category:Neurological disorders",
    "Category:Cutaneous conditions",
    "Category:Endocrine diseases",
    "Category:Genetic diseases and disorders",
    "Category:Musculoskeletal disorders",
    "Category:Gastrointestinal tract disorders",
    "Category:Cancer",
    "Category:Mental disorders",
]

# Set to an integer if you want a smaller test run.
# None means fetch the full category.
MAX_ARTICLES_PER_CATEGORY = None
REQUEST_DELAY_SECONDS = 1.0


def get_category_members(category: str, limit=None) -> list[str]:
    titles = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page",
            "cmlimit": 500,
            "format": "json",
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(
            WIKI_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()
        for member in data.get("query", {}).get("categorymembers", []):
            titles.append(member["title"])
            if limit is not None and len(titles) >= limit:
                return titles

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return titles


def fetch_wikipedia_extract(title: str, retries: int = 2):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                WIKI_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if response.status_code == 429 and attempt < retries:
                print(f"  Rate-limited on '{title}', waiting 10s...")
                time.sleep(10)
                continue
            print(f"  Skipping '{title}': {exc}")
            return None
        except requests.exceptions.RequestException as exc:
            print(f"  Skipping '{title}': {exc}")
            return None

        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or not page.get("extract"):
                return None
            return {
                "title": page.get("title", title),
                "text": page["extract"],
            }

    return None


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_titles = set()

    for category in CATEGORIES:
        members = get_category_members(
            category,
            limit=MAX_ARTICLES_PER_CATEGORY,
        )

        print(f"{category}: found {len(members)} articles")
        all_titles.update(members)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nTotal unique articles to fetch: {len(all_titles)}\n")

    saved = skipped = 0

    for i, title in enumerate(sorted(all_titles), start=1):
        result = fetch_wikipedia_extract(title)

        if not result or len(result["text"].split()) < 50:
            skipped += 1
            continue

        url = (
            "https://en.wikipedia.org/wiki/"
            + result["title"].replace(" ", "_")
        )

        header = (
            "SOURCE_TYPE: wikipedia\n"
            "SOURCE: Wikipedia\n"
            "LICENSE: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)\n"
            f"TOPIC: {result['title']}\n"
            f"ORIGINAL_SOURCE: {url}\n\n"
        )

        output_path = RAW_DIR / f"wiki_{slugify(result['title'])}.txt"
        output_path.write_text(
            header + clean_text(result["text"]),
            encoding="utf-8",
        )

        saved += 1

        if i % 25 == 0:
            print(f"  ...progress: {i}/{len(all_titles)} processed")

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Saved {saved} articles, skipped {skipped}.")


if __name__ == "__main__":
    main()
