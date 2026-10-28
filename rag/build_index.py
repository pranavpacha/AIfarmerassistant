"""
Build a FAISS similarity index over the advisory corpus (data/corpus.json, or
data/seed_corpus.json if the full fetch hasn't been run) using multilingual
sentence embeddings, so queries in any supported Indian language retrieve
relevant English-language advisory entries without needing translation first.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_corpus() -> list[dict]:
    corpus_path = DATA_DIR / "corpus.json"
    if not corpus_path.exists():
        corpus_path = DATA_DIR / "seed_corpus.json"
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def entry_to_text(entry: dict) -> str:
    # Embed query+answer together so the index captures both what farmers ask
    # and what the grounded answer actually contains.
    return f"{entry['crop']} | {entry['category']} | Q: {entry['query']} A: {entry['answer']}"


def build():
    INDEX_DIR.mkdir(exist_ok=True)
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} corpus entries")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [entry_to_text(e) for e in corpus]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "corpus.faiss"))
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote FAISS index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {INDEX_DIR}")


if __name__ == "__main__":
    build()
