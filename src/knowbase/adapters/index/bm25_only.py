"""Pure-BM25 baseline adapter.

Kept in production rather than in the test folder because a baseline is only useful if the A/B
endpoint can point live traffic at it: ``POST /compare`` runs the same eval battery through this
adapter and the hybrid one to justify (or refute) the extra cost of fusion.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.errors import EmptyCorpusError, IndexNotBuiltError
from ...domain.model import Passage, RetrievalHit
from ...domain.text import tokenize
from ._lexical import BM25, LexicalIndexState


class Bm25Index:
    """Okapi BM25 only -- no vector view, no fusion."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._state: LexicalIndexState | None = None

    def size(self) -> int:
        state = self._state
        return 0 if state is None else len(state.passages)

    def build(self, passages: Sequence[Passage]) -> None:
        if not passages:
            raise EmptyCorpusError("cannot index an empty passage list")
        held = tuple(passages)
        self._state = LexicalIndexState(
            passages=held,
            bm25=BM25([tokenize(p.text) for p in held], k1=self._k1, b=self._b))

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        state = self._state
        if state is None or state.bm25 is None:
            raise IndexNotBuiltError("Bm25Index.search called before build")
        if k <= 0:
            raise ValueError("k must be positive")
        scores = state.bm25.scores(tokenize(query))
        # Index as tie-break keeps the ordering total and therefore reproducible.
        ranked = sorted(((i, float(s)) for i, s in enumerate(scores) if s > 0),
                        key=lambda kv: (-kv[1], kv[0]))[:k]
        return [RetrievalHit(passage=state.passages[i], score=round(s, 6), rank=r)
                for r, (i, s) in enumerate(ranked, start=1)]
