"""
[NEW FILE - item #12/#19-20 of the plan]

Cross-encoder reranker: takes the top-20 candidates from the vector DB
(cheap, approximate) and re-scores them with a cross-encoder (slower,
but reads query+chunk together so it's a much better relevance judge),
keeping only the top-5 for the final context.

Also de-duplicates near-identical chunks (item #20/#21 - MedQuAD and
Wikipedia can both describe the same fact) before truncating to top_k.
"""

import hashlib
import re

from sentence_transformers import CrossEncoder

# Multilingual cross-encoder (works for Arabic + English query/chunk pairs).
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self._model_name = model_name
        self._model = None  # lazy-loaded so importing this module stays cheap

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    @staticmethod
    def _dedup_key(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return hashlib.sha256(normalized[:500].encode("utf-8")).hexdigest()

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """`candidates` is the raw top-20 list from search.search().
        Returns up to `top_k` items, deduplicated, sorted by rerank_score
        (each item keeps its original fields plus a new 'rerank_score')."""
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        deduped, seen = [], set()
        for c in candidates:
            key = self._dedup_key(c["text"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
            if len(deduped) >= top_k:
                break

        return deduped
