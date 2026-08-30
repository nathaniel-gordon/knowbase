"""Shared lexical scoring machinery for the index adapters.

Deliberately *not* in the domain: it depends on numpy and scikit-learn, and the whole point of
the IndexPort seam is that those libraries stay on the outside of the hexagon.

The BM25 implementation and the TF-IDF configuration are carried over unchanged from the
pre-refactor engine that measured hit@1 = 1.0 / MRR = 1.0 on the bundled suite.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...domain.text import tokenize


class BM25:
    """Okapi BM25 over pre-tokenised documents.

    Vectorised per query term rather than per document: the corpus is scored as numpy arrays,
    which keeps a 12-question eval sweep in the low milliseconds.
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [list(d) for d in corpus_tokens]
        self.doc_len = np.array([len(d) for d in self.doc_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.doc_tokens else 0.0
        self.doc_freqs = [Counter(d) for d in self.doc_tokens]
        df: Counter[str] = Counter()
        for d in self.doc_tokens:
            df.update(set(d))
        n = len(self.doc_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_tokens))
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            tf = np.array([d.get(token, 0) for d in self.doc_freqs], dtype=float)
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
            out += idf * tf * (self.k1 + 1) / np.where(denom == 0, 1.0, denom)
        return out


class TfidfCosine:
    """TF-IDF over 1-2 grams with sublinear term frequency, scored by cosine similarity."""

    def __init__(self, texts: Sequence[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          tokenizer=tokenize, token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(list(texts))

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()


def reciprocal_rank_fusion(score_lists: Sequence[np.ndarray], rrf_k: int = 60) -> dict[int, float]:
    """Fuse several score vectors by rank rather than by magnitude.

    RRF is used because BM25 scores and cosine similarities live on incomparable scales; fusing
    on rank sidesteps the normalisation problem entirely and is what lifted the bundled suite
    to a perfect MRR.
    """
    fused: dict[int, float] = {}
    for scores in score_lists:
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order):
            if scores[idx] <= 0:
                continue
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused
