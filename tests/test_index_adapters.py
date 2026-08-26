"""Run the IndexPort battery against every retrieval strategy.

This is where substitutability earns its keep: the hybrid index, the BM25 baseline and the
TF-IDF baseline are interchangeable exactly because all three satisfy the identical battery,
including its quality floor.
"""
from __future__ import annotations

import pytest

from contracts.index_contract import IndexContract
from kb.adapters.index.bm25_only import Bm25Index
from kb.adapters.index.hybrid_rrf import HybridRrfIndex
from kb.adapters.index.tfidf_only import TfidfIndex


class TestHybridRrfIndex(IndexContract):
    @pytest.fixture
    def make_index(self):
        return HybridRrfIndex

    def test_fusion_scores_are_rrf_shaped(self, index):
        """RRF scores live in (0, 2/(rrf_k+1)]: two rank lists, each worth 1/(60+rank).

        The tolerance absorbs the 6-decimal rounding the adapter applies to its scores.
        """
        hits = index.search("vacation days per year", k=3)
        assert hits and all(0 < h.score <= 2.0 / 61.0 + 1e-6 for h in hits)

    def test_fusion_beats_a_single_view_on_a_split_query(self, index, fixture_passages):
        """A query whose terms are split across views must still surface both documents."""
        hits = index.search("password rotation vacation accrual", k=4)
        assert {h.passage.doc_id for h in hits} >= {"passwords", "leave"}


class TestBm25Index(IndexContract):
    @pytest.fixture
    def make_index(self):
        return Bm25Index

    def test_scores_are_raw_bm25_magnitudes(self, index):
        """Unlike RRF, BM25 emits unbounded term-weighted scores."""
        hits = index.search("password rotation", k=2)
        assert hits and hits[0].score > 0.5


class TestTfidfIndex(IndexContract):
    @pytest.fixture
    def make_index(self):
        return TfidfIndex

    def test_scores_are_cosine_similarities(self, index):
        hits = index.search("vacation days", k=3)
        assert hits and all(0.0 < h.score <= 1.0 + 1e-9 for h in hits)


def test_every_registered_index_adapter_has_a_contract_subclass():
    from kb.adapters import INDEX_ADAPTERS
    covered = {cls.__name__ for cls in (HybridRrfIndex, Bm25Index, TfidfIndex)}
    assert {cls.__name__ for cls in INDEX_ADAPTERS.values()} == covered


def test_adapters_disagree_on_ranking_but_agree_on_the_floor(fixture_passages):
    """Substitutable does not mean identical -- otherwise the A/B endpoint would be pointless."""
    scores_by_adapter = {}
    for cls in (HybridRrfIndex, Bm25Index, TfidfIndex):
        index = cls()
        index.build(fixture_passages)
        hits = index.search("password rotation policy", k=3)
        scores_by_adapter[index.name] = [round(h.score, 6) for h in hits]
        assert hits[0].passage.doc_id == "passwords"
    assert scores_by_adapter["hybrid_rrf"] != scores_by_adapter["bm25"]
    assert scores_by_adapter["bm25"] != scores_by_adapter["tfidf"]
