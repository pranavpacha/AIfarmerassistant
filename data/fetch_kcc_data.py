"""
Fetch Kisan Call Centre (KCC) farmer query/answer transcripts from the official
data.gov.in API and merge them with the curated seed corpus into one file the
RAG index is built from.

Requires a free API key from https://api.data.gov.in/signup/ (see SETUP.md).
Without a key, build_index.py falls back to seed_corpus.json alone.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

# The resource UUID is visible in the URL of the dataset's page on data.gov.in
# (https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers)
# after you open it logged in — copy it from there, it is not guessable in advance.
RESOURCE_ID = os.environ.get("KCC_RESOURCE_ID", "")
API_BASE = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
PAGE_SIZE = 500
MAX_RECORDS = int(os.environ.get("KCC_MAX_RECORDS", 5000))

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "kcc_raw.json"
SEED_PATH = DATA_DIR / "seed_corpus.json"
CORPUS_PATH = DATA_DIR / "corpus.json"


def fetch_kcc_records(api_key: str) -> list[dict]:
    records = []
    offset = 0
    while offset < MAX_RECORDS:
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        resp = requests.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("records", [])
        if not batch:
            break
        records.extend(batch)
        print(f"  fetched {len(records)} records so far...")
        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite to the public API
    return records


def kcc_record_to_entry(rec: dict, idx: int) -> dict | None:
    query = (rec.get("QueryText") or rec.get("query_text") or "").strip()
    answer = (rec.get("KccAns") or rec.get("kcc_ans") or rec.get("answer") or "").strip()
    if not query or not answer:
        return None
    crop = rec.get("Crop") or rec.get("crop") or "General"
    state = rec.get("StateName") or rec.get("state_name") or ""
    category = rec.get("Category") or rec.get("category") or "Farmer Advisory"
    return {
        "id": f"kcc-{idx}",
        "crop": crop,
        "category": category,
        "state": state,
        "query": query,
        "answer": answer,
        "source": "Kisan Call Centre (data.gov.in)",
    }


def main():
    api_key = os.environ.get("DATA_GOV_IN_API_KEY")
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))

    if not api_key or not RESOURCE_ID:
        print("DATA_GOV_IN_API_KEY or KCC_RESOURCE_ID not set — using seed corpus only "
              f"({len(seed)} entries). See SETUP.md to enable the full KCC dataset.")
        CORPUS_PATH.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    print("Fetching KCC transcripts from data.gov.in...")
    try:
        raw = fetch_kcc_records(api_key)
    except requests.RequestException as e:
        print(f"KCC API request failed ({e}); falling back to seed corpus only.", file=sys.stderr)
        CORPUS_PATH.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    RAW_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    kcc_entries = []
    for i, rec in enumerate(raw):
        entry = kcc_record_to_entry(rec, i)
        if entry:
            kcc_entries.append(entry)

    corpus = seed + kcc_entries
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(corpus)} total entries ({len(seed)} seed + {len(kcc_entries)} KCC) to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
