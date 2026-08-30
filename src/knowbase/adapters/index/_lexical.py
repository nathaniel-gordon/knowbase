"""Shared lexical scoring machinery for the index adapters.

Deliberately *not* in the domain: it depends on numpy and scikit-learn, and the whole point of
the IndexPort seam is that those libraries stay on the outside of the hexagon.

BM25 and the TF-IDF configuration are both here because the hybrid adapter fuses them and
the two single-strategy adapters expose each half on its own for the A/B endpoint.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...domain.errors import IndexBuildFailed
from ...domain.text import tokenize


class BM25:
    """Okapi BM25 over pre-tokenised documents.

    Vectorised per query term rather than per document: the corpus is scored as numpy arrays,
    which keeps a 12-question eval sweep in the low milliseconds.
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [list(d) for d in corpus_tokens]
        if not any(self.doc_tokens):
            # Matches what the TF-IDF half does on the same input, so every index adapter
            # rejects a degenerate corpus with the same domain error rather than one of them
            # silently building an index that can never return a hit.
            raise IndexBuildFailed(
                "the documents carry no indexable terms once stopwords are removed",
                reason="empty_vocabulary")
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
    """TF-IDF over 1-2 grams with sublinear term frequency, scored by cosine similarity.

    Fitting is where scikit-learn raises on degenerate input (a corpus of pure stopwords gives
    "empty vocabulary"). That is translated here so the driver's wording never reaches a
    client: the boundary rule in ``domain.errors`` applies to this adapter like any other.
    """

    def __init__(self, texts: Sequence[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          tokenizer=tokenize, token_pattern=None)
        try:
            self.matrix = self.vectorizer.fit_transform(list(texts))
        except ValueError as exc:
            raise IndexBuildFailed(
                "the documents carry no indexable terms once stopwords are removed",
                reason="empty_vocabulary") from exc

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()


def reciprocal_rank_fusion(score_lists: Sequence[np.ndarray], rrf_k: int = 60) -> dict[int, float]:
    """Fuse several score vectors by rank rather than by magnitude.

    RRF is used because BM25 scores and cosine similarities live on incomparable scales; fusing
    on rank sidesteps the normalisation problem entirely.
    """
    fused: dict[int, float] = {}
    for scores in score_lists:
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order):
            if scores[idx] <= 0:
                continue
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
    return fused
