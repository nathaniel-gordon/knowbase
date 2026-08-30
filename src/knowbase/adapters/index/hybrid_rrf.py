"""The production retrieval adapter: BM25 + TF-IDF cosine fused by reciprocal rank.

This is the strategy that scored hit@1 = 1.0, hit@5 = 1.0, MRR = 1.0 on the bundled suite
before the refactor; the scoring path is preserved verbatim so the numbers are preserved too.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.errors import EmptyCorpusError, IndexNotBuiltError
from ...domain.model import Passage, RetrievalHit
from ...domain.text import tokenize
from ._lexical import BM25, TfidfCosine, reciprocal_rank_fusion


class HybridRrfIndex:
    """Lexical hybrid index. ``rrf_k`` damps the influence of deep ranks (60 is the standard)."""

    name = "hybrid_rrf"

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k
        self._passages: list[Passage] = []
        self._bm25: BM25 | None = None
        self._tfidf: TfidfCosine | None = None

    def size(self) -> int:
        return len(self._passages)

    def build(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise EmptyCorpusError("cannot index an empty passage list")
        self._passages = list(passages)
        self._bm25 = BM25([tokenize(p.text) for p in self._passages])
        self._tfidf = TfidfCosine([p.text for p in self._passages])

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if self._bm25 is None or self._tfidf is None:
            raise IndexNotBuiltError("HybridRrfIndex.search called before build")
        if k <= 0:
            raise ValueError("k must be positive")
        fused = reciprocal_rank_fusion(
            [self._bm25.scores(tokenize(query)), self._tfidf.scores(query)], self._rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [RetrievalHit(passage=self._passages[i], score=round(float(s), 6), rank=r)
                for r, (i, s) in enumerate(ranked, start=1)]
