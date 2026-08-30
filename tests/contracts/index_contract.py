"""Contract battery for IndexPort.

The quality floor is the important clause: an adapter is only substitutable if it actually
retrieves. Without it, a "return nothing" index would pass every structural assertion.
"""
from __future__ import annotations

import pytest

from knowbase.domain.errors import EmptyCorpusError, IndexBuildFailed, IndexNotBuiltError
from knowbase.domain.model import Passage
from knowbase.ports import IndexPort

from .corpus_fixture import FIXTURE_PROBES


class IndexContract:
    """Every IndexPort adapter must pass all of these.

    Subclasses provide ``make_index`` returning a fresh, unbuilt adapter.
    """

    @pytest.fixture
    def make_index(self):
        raise NotImplementedError("contract subclass must provide a 'make_index' fixture")

    @pytest.fixture
    def index(self, make_index, fixture_passages) -> IndexPort:
        built = make_index()
        built.build(fixture_passages)
        return built

    # ------------------------------------------------------------- structure

    def test_satisfies_the_protocol(self, index):
        assert isinstance(index, IndexPort)
        assert isinstance(index.name, str) and index.name == index.name.lower()

    def test_search_before_build_is_refused(self, make_index):
        with pytest.raises(IndexNotBuiltError):
            make_index().search("anything", k=3)

    def test_building_an_empty_corpus_is_refused(self, make_index):
        with pytest.raises(EmptyCorpusError):
            make_index().build([])

    def test_building_a_corpus_of_pure_stopwords_raises_a_domain_error(self, make_index):
        """No driver exception may escape: sklearn's "empty vocabulary" is not an API error."""
        stopwords_only = [Passage(doc_id="noise", ordinal=0, title="Noise",
                                  text="the and or but if then it is at of to")]
        with pytest.raises(IndexBuildFailed):
            make_index().build(stopwords_only)

    def test_size_reflects_the_corpus(self, make_index, index, fixture_passages):
        assert make_index().size() == 0
        assert index.size() == len(fixture_passages)

    def test_rebuild_replaces_previous_content(self, index, fixture_passages):
        index.build(fixture_passages[:2])
        assert index.size() == 2
        assert {h.passage.passage_id for h in index.search("vacation days", k=5)} <= {
            p.passage_id for p in fixture_passages[:2]}

    def test_invalid_k_is_refused(self, index):
        with pytest.raises(ValueError):
            index.search("vacation", k=0)

    # ----------------------------------------------------------- result shape

    def test_results_are_ranked_and_capped(self, index):
        hits = index.search("password rotation policy", k=2)
        assert 0 < len(hits) <= 2
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(h.score > 0 for h in hits)

    def test_hits_reference_indexed_passages(self, index, fixture_passages):
        known = {p.passage_id: p for p in fixture_passages}
        for hit in index.search("expense report receipts", k=4):
            assert hit.passage.passage_id in known
            assert hit.passage.text == known[hit.passage.passage_id].text

    def test_unmatchable_query_returns_nothing(self, index):
        assert index.search("qwertyuiop zxcvbnmasdf", k=5) == []

    def test_search_is_deterministic(self, index):
        first = [(h.passage.passage_id, h.score) for h in index.search("vpn tunnel drops", k=4)]
        second = [(h.passage.passage_id, h.score) for h in index.search("vpn tunnel drops", k=4)]
        assert first == second

    # ------------------------------------------------------------ quality floor

    @pytest.mark.parametrize(("question", "expected_doc_id"), FIXTURE_PROBES,
                             ids=[doc for _, doc in FIXTURE_PROBES])
    def test_labelled_question_retrieves_its_document_first(self, index, question,
                                                            expected_doc_id):
        hits = index.search(question, k=3)
        assert hits, f"{index.name} retrieved nothing for {question!r}"
        assert hits[0].passage.doc_id == expected_doc_id, (
            f"{index.name} ranked {hits[0].passage.doc_id!r} above "
            f"{expected_doc_id!r} for {question!r}")
