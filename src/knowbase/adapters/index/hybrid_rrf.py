"""The production retrieval adapter: BM25 + TF-IDF cosine fused by reciprocal rank.

Fusing on rank rather than on score sidesteps the fact that BM25 scores and cosine
similarities live on incomparable scales. ``python run.py demo`` prints what it currently
scores on the two bundled suites.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.errors import EmptyCorpusError, IndexNotBuiltError
from ...domain.model import Passage, RetrievalHit
from ...domain.text import tokenize
from ._lexical import BM25, LexicalIndexState, TfidfCosine, reciprocal_rank_fusion


class HybridRrfIndex:
    """Lexical hybrid index. ``rrf_k`` damps the influence of deep ranks (60 is the standard)."""

    name = "hybrid_rrf"

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k
        self._state: LexicalIndexState | None = None

    def size(self) -> int:
        state = self._state
        return 0 if state is None else len(state.passages)

    def build(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise EmptyCorpusError("cannot index an empty passage list")
        held = tuple(passages)
        # Built first, published second: a reader never observes a half-swapped index.
        self._state = LexicalIndexState(passages=held,
                                        bm25=BM25([tokenize(p.text) for p in held]),
                                        tfidf=TfidfCosine([p.text for p in held]))

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        state = self._state
        if state is None or state.bm25 is None or state.tfidf is None:
            raise IndexNotBuiltError("HybridRrfIndex.search called before build")
        if k <= 0:
            raise ValueError("k must be positive")
        fused = reciprocal_rank_fusion(
            [state.bm25.scores(tokenize(query)), state.tfidf.scores(query)], self._rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [RetrievalHit(passage=state.passages[i], score=round(float(s), 6), rank=r)
                for r, (i, s) in enumerate(ranked, start=1)]
