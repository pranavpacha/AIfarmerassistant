"""Load the FAISS index built by build_index.py and retrieve top-k relevant
advisory entries for a given (English or Indic-language) query."""
import json
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Retriever:
    def __init__(self):
        if not (INDEX_DIR / "corpus.faiss").exists():
            raise FileNotFoundError(
                "No index found. Run `python rag/build_index.py` first."
            )
        self.index = faiss.read_index(str(INDEX_DIR / "corpus.faiss"))
        self.metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
        # Forced to CPU: this model is small and retrieval isn't latency-critical,
        # so there's no good reason to spend GPU memory on it when AgriParam and
        # two translation models are already competing for a T4's limited VRAM.
        self.model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
