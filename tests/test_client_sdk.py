"""Client SDK tests: typed views, error translation, and configuration plumbing."""
from __future__ import annotations

import dataclasses

import pytest

from kb.app.client import AnswerView, ComparisonView, EvalView, IngestView, running_service
from kb.app.composition import build_container, build_generator, build_source
from kb.app.config import AppConfig
from kb.domain.errors import UnknownStrategyError


@pytest.fixture(scope="module")
def client(demo_corpus_dir):
    with running_service(AppConfig(clock="frozen", generator="extractive")) as live:
        live.ingest_path(str(demo_corpus_dir))
        yield live


def test_views_are_frozen_dataclasses():
    """The SDK hands out immutable values, not dictionaries the caller can corrupt."""
    for view in (AnswerView, EvalView, IngestView, ComparisonView):
        assert dataclasses.is_dataclass(view)
        assert view.__dataclass_params__.frozen


def test_ask_returns_a_typed_answer_view(client):
    answer = client.ask("How quickly must a data subject deletion request be honored?")
    assert isinstance(answer, AnswerView)
    assert "30 days" in answer.answer
    rendered = answer.render()
    assert rendered.startswith("Q: How quickly")
    assert f"({answer.citations[0].passage_id})" in rendered
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.answer = "tampered"  # type: ignore[misc]


def test_eval_view_summary_is_human_readable(client):
    report = client.evaluate(suite="core", k=5, rows=False)
    summary = report.summary()
    assert "hybrid_rrf" in summary and "hit@1=1.000" in summary and "mrr=1.000" in summary


def test_comparison_view_table_names_the_winner(client):
    result = client.compare(["hybrid_rrf", "bm25"], suite="hard", k=5)
    table = result.table()
    assert table.count("\n") == 2
    assert "wins by" in table and result.winner in table


def test_ingest_view_round_trips_the_report(client, demo_corpus_dir):
    report = client.ingest_path(str(demo_corpus_dir))
    assert isinstance(report, IngestView)
    assert report.elapsed_ms > 0 and report.started_at.endswith("+00:00")


# ------------------------------------------------------- composition-root units

def test_config_is_immutable_and_copies_forward():
    config = AppConfig()
    updated = config.with_(index="bm25")
    assert config.index == "hybrid_rrf" and updated.index == "bm25"
    assert updated.compare_indexes == config.compare_indexes


def test_config_round_trips_through_json(tmp_path):
    original = AppConfig(index="tfidf", generator="extractive", cache="sqlite", top_k=7)
    path = tmp_path / "kb.json"
    path.write_text(__import__("json").dumps(original.to_dict()), encoding="utf-8")
    restored = AppConfig.from_json_file(path)
    assert restored.index == "tfidf" and restored.cache == "sqlite" and restored.top_k == 7


def test_index_names_puts_the_active_adapter_first():
    assert AppConfig(index="bm25").index_names() == ("bm25", "hybrid_rrf", "tfidf")


def test_generator_falls_back_when_claude_is_unavailable(monkeypatch):
    """Offline operation is a guarantee, not an accident: no key means extractive, not failure."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    generator, note = build_generator("claude")
    assert generator.name == "extractive"
    assert "claude unavailable" in note


def test_unknown_adapters_are_rejected_by_the_composition_root():
    with pytest.raises(UnknownStrategyError, match="index"):
        build_container(AppConfig(index="faiss"))
    with pytest.raises(UnknownStrategyError, match="generator"):
        build_generator("gpt")
    with pytest.raises(UnknownStrategyError, match="source"):
        build_source({"source": "s3", "path": "x"})


def test_memory_source_requires_documents():
    with pytest.raises(ValueError, match="non-empty 'documents'"):
        build_source({"source": "memory"})


def test_container_closes_its_sqlite_cache(tmp_path):
    container = build_container(AppConfig(cache="sqlite",
                                          sqlite_path=str(tmp_path / "c.sqlite3"),
                                          clock="frozen"))
    container.close()
    with pytest.raises(Exception):
        container.cache.stats()
