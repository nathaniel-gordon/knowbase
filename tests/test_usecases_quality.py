"""Quality regression tests for the domain use-cases over the real bundled knowledge base.

These are the numbers the refactor had to preserve. The pre-refactor engine scored
hit@1 = 1.000, hit@5 = 1.000, MRR = 1.000 on the 12-question core suite; anything less here is
a regression, not a rounding difference.
"""
from __future__ import annotations

import pytest

from kb.adapters.cache.memory import InMemoryCache
from kb.adapters.clock.system import FrozenClock
from kb.adapters.generator.extractive import ExtractiveGenerator
from kb.adapters.index.bm25_only import Bm25Index
from kb.adapters.index.hybrid_rrf import HybridRrfIndex
from kb.adapters.index.tfidf_only import TfidfIndex
from kb.adapters.source.filesystem import FilesystemDocumentSource
from kb.app.corpus import CORE_SUITE, HARD_SUITE
from kb.domain.errors import IndexNotBuiltError, UnknownStrategyError
from kb.domain.usecases import AskQuestion, CompareStrategies, EvaluateRetrieval, IngestCorpus

#: Pre-refactor baseline, reproduced by the demo on every run.
BASELINE_CORE = {"hit@1": 1.0, "hit@5": 1.0, "mrr": 1.0}


@pytest.fixture(scope="module")
def wired(demo_corpus_dir):
    """One ingest pass feeding all three index adapters -- exactly what the service does."""
    indexes = {cls.name: cls() for cls in (HybridRrfIndex, Bm25Index, TfidfIndex)}
    ingest = IngestCorpus(source=FilesystemDocumentSource(demo_corpus_dir),
                          indexes=list(indexes.values()),
                          clock=FrozenClock())
    report = ingest.execute()
    return {"indexes": indexes, "report": report, "passages": ingest.passages}


# ------------------------------------------------------------------------ ingest

def test_ingest_reads_the_whole_corpus(wired):
    report = wired["report"]
    assert report.documents == 14
    assert report.passages == 14, "120-word windows leave each policy document in one passage"
    assert report.source == "filesystem"
    assert report.started_at.startswith("2023-11-14T")  # frozen clock, deterministic
    assert report.elapsed_ms > 0


def test_every_index_received_the_same_passages(wired):
    sizes = {name: index.size() for name, index in wired["indexes"].items()}
    assert set(sizes.values()) == {wired["report"].passages}


def test_passage_ids_are_stable_and_unique(wired):
    ids = [p.passage_id for p in wired["passages"]]
    assert len(ids) == len(set(ids))
    assert "expense-policy#0" in ids and "pto-policy#0" in ids


# -------------------------------------------------------------------- retrieval

@pytest.mark.parametrize("index_name", ["hybrid_rrf", "bm25", "tfidf"])
def test_core_suite_quality_is_preserved(wired, index_name):
    report = EvaluateRetrieval(wired["indexes"][index_name]).execute(CORE_SUITE, k=5,
                                                                     suite="core")
    assert report.n == 12
    assert report.hit_at_1 >= BASELINE_CORE["hit@1"], report.summary()
    assert report.hit_at_k >= BASELINE_CORE["hit@5"], report.summary()
    assert report.mrr >= BASELINE_CORE["mrr"], report.summary()
    assert all(row.rank == 1 for row in report.rows)


def test_hard_suite_is_actually_hard(wired):
    """If the paraphrase battery were saturated it could not discriminate strategies."""
    report = EvaluateRetrieval(wired["indexes"]["hybrid_rrf"]).execute(HARD_SUITE, k=5,
                                                                       suite="hard")
    assert report.n == 12
    assert 0.5 <= report.mrr < 1.0, report.summary()
    assert report.hit_at_k >= 0.7, report.summary()


def test_evaluation_rows_explain_every_miss(wired):
    report = EvaluateRetrieval(wired["indexes"]["hybrid_rrf"]).execute(HARD_SUITE, k=5,
                                                                       suite="hard")
    assert len(report.rows) == 12
    assert {r.question for r in report.rows} == {c.question for c in HARD_SUITE}
    for row in report.rows:
        assert 0 <= row.rank <= report.k
        assert (row.rank == 1) == (row.top_doc_id == row.expected_doc_id)
    # The documented ceiling of a lexical index: three paraphrases ("thumb drive" for
    # "removable USB storage") share almost no vocabulary with their document and are missed.
    assert sum(1 for r in report.rows if r.rank == 0) == 3
    recovered = sum(1.0 / r.rank for r in report.rows if r.rank) / len(report.rows)
    assert recovered == pytest.approx(report.mrr, abs=5e-4), "MRR must equal the row detail"


def test_evaluation_needs_cases(wired):
    with pytest.raises(ValueError):
        EvaluateRetrieval(wired["indexes"]["bm25"]).execute([])


# ------------------------------------------------------------------ comparison

def test_comparison_is_deterministic_and_ranked(wired):
    compare = CompareStrategies(wired["indexes"])
    first = compare.execute(HARD_SUITE, k=5, suite="hard")
    second = compare.execute(HARD_SUITE, k=5, suite="hard")
    assert first == second, "an A/B verdict that flaps is worthless"
    mrrs = [r.mrr for r in first.reports]
    assert mrrs == sorted(mrrs, reverse=True)
    assert first.winner in {"tie", *wired["indexes"]}


def test_comparison_finds_the_core_suite_saturated(wired):
    result = CompareStrategies(wired["indexes"]).execute(CORE_SUITE, k=5, suite="core")
    assert result.winner == "tie" and result.margin == 0.0
    assert {r.mrr for r in result.reports} == {1.0}


def test_comparison_separates_strategies_on_the_hard_suite(wired):
    """Golden A/B verdict: on paraphrases, fusing TF-IDF into BM25 costs one rank position.

    ``bm25`` places "What happens if we send too many API calls?" first; both TF-IDF-flavoured
    strategies push it to rank 2, worth 1/2 - 1 = -0.5 of a reciprocal rank over 12 questions
    (-0.0417 MRR). This is the finding the A/B endpoint exists to surface.
    """
    result = CompareStrategies(wired["indexes"]).execute(HARD_SUITE, k=5, suite="hard")
    by_name = {r.index: r for r in result.reports}
    assert set(by_name) == {"hybrid_rrf", "bm25", "tfidf"}
    assert result.winner == "bm25"
    assert result.margin == pytest.approx(0.5 / 12, abs=1e-3)
    assert by_name["bm25"].mrr == pytest.approx(0.75, abs=1e-3)
    assert by_name["hybrid_rrf"].mrr == pytest.approx(0.7083, abs=1e-3)
    assert by_name["tfidf"].mrr == pytest.approx(0.7083, abs=1e-3)
    assert by_name[result.winner].mrr == max(r.mrr for r in result.reports)


def test_comparison_rejects_an_unknown_adapter(wired):
    with pytest.raises(UnknownStrategyError, match="faiss"):
        CompareStrategies(wired["indexes"]).execute(CORE_SUITE, names=["hybrid_rrf", "faiss"])


def test_comparison_needs_two_strategies(wired):
    with pytest.raises(ValueError):
        CompareStrategies({"hybrid_rrf": wired["indexes"]["hybrid_rrf"]})


# --------------------------------------------------------------------- asking

@pytest.fixture
def ask(wired):
    clock = FrozenClock()
    return AskQuestion(index=wired["indexes"]["hybrid_rrf"], generator=ExtractiveGenerator(),
                       cache=InMemoryCache(clock), clock=clock, default_k=4)


def test_answer_carries_the_fact_and_earns_its_citations(ask):
    answer = ask.execute("How many PTO days do employees get per year?")
    assert "21" in answer.text or "1.75" in answer.text, answer.text
    assert answer.citations, "an answered question must be attributable"
    retrieved_ids = {h.passage.passage_id for h in answer.hits}
    assert {c.passage_id for c in answer.citations} <= retrieved_ids
    assert any(c.passage_id.startswith("pto-policy") for c in answer.citations)
    assert answer.generator == "extractive" and answer.index == "hybrid_rrf"


def test_citation_markers_index_the_retrieved_list(ask):
    answer = ask.execute("What should I do if the VPN keeps disconnecting?")
    ordered = [h.passage.passage_id for h in answer.hits]
    for citation in answer.citations:
        assert ordered[citation.marker - 1] == citation.passage_id


def test_repeated_question_is_served_from_cache(ask):
    first = ask.execute("What is the API rate limit per workspace?")
    second = ask.execute("what is the API RATE LIMIT per workspace?  ")
    assert first.cached is False and second.cached is True
    assert first.text == second.text, "cache must not change the answer"


def test_cache_key_separates_adapters(wired):
    clock = FrozenClock()
    cache = InMemoryCache(clock)
    hybrid = AskQuestion(wired["indexes"]["hybrid_rrf"], ExtractiveGenerator(), cache, clock)
    bm25 = AskQuestion(wired["indexes"]["bm25"], ExtractiveGenerator(), cache, clock)
    question = "How fast must the on-call primary acknowledge a page?"
    assert hybrid.cache_key(question, 4) != bm25.cache_key(question, 4)
    hybrid.execute(question)
    assert bm25.execute(question).cached is False, "a swapped index must not reuse cached prose"


def test_blank_question_is_refused(ask):
    with pytest.raises(ValueError):
        ask.execute("   ")


def test_asking_an_unbuilt_index_is_refused():
    clock = FrozenClock()
    with pytest.raises(IndexNotBuiltError):
        AskQuestion(HybridRrfIndex(), ExtractiveGenerator(), InMemoryCache(clock),
                    clock).execute("anything")
