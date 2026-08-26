"""Pure TF-IDF cosine baseline -- the other half of the hybrid, isolated.

Having both halves available as first-class adapters means the A/B endpoint can attribute a
win to fusion itself rather than to one of its two components.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.errors import EmptyCorpusError, IndexNotBuiltError
from ...domain.model import Passage, RetrievalHit
from ._lexical import TfidfCosine


class TfidfIndex:
    """Cosine similarity over TF-IDF 1-2 grams."""

    name = "tfidf"

    def __init__(self) -> None:
        self._passages: list[Passage] = []
        self._tfidf: TfidfCosine | None = None

    def size(self) -> int:
        return len(self._passages)

    def build(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise EmptyCorpusError("cannot index an empty passage list")
        self._passages = list(passages)
        self._tfidf = TfidfCosine([p.text for p in self._passages])

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._tfidf is None:
            raise IndexNotBuiltError("TfidfIndex.search called before build")
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self._tfidf.scores(query)
        ranked = sorted(((i, float(s)) for i, s in enumerate(scores) if s > 0),
                        key=lambda kv: (-kv[1], kv[0]))[:k]
        return [RetrievalHit(passage=self._passages[i], score=round(s, 6), rank=r)
                for r, (i, s) in enumerate(ranked, start=1)]
