"""Grounding and durability of the corpus, at the level a client can observe.

Two behaviours are pinned here because both were broken and both are invisible from a single
request: an answer must never come from a corpus the service is not holding, and a restart
with the sqlite store bound must not lose the corpus.
"""
from __future__ import annotations

import pytest

from knowbase.app.client import ServiceError, running_service
from knowbase.app.composition import build_container
from knowbase.app.config import AppConfig
from knowbase.app.validation import IngestRequest
from knowbase.domain.model import Generated
from knowbase.domain.text import corpus_fingerprint
from knowbase.domain.usecases import AskQuestion

CORPUS_A = {"retention": "# Data Retention\nCustomer records are deleted after 90 days."}
CORPUS_B = {"badges": "# Badges\nVisitor badges are returned at reception each evening."}
QUESTION = "How long are customer records kept?"


def _ingest(container, documents):
    return container.ingest(IngestRequest(source="memory",
                                          documents=tuple(documents.items()),
                                          label="test"))


# --------------------------------------------------- cache grounding (the bug)

def test_a_cache_written_against_another_corpus_is_never_served(tmp_path):
    """The reported defect, replayed.

    Corpus A is answered with a sqlite cache at a path. A second service starts, ingests a
    completely different corpus B, and is pointed at the same cache file. Asking the same
    question used to return A's answer text with ``cached=true``, no citations and nothing
    retrieved. It must now be recomputed against B.
    """
    db = str(tmp_path / "shared.sqlite3")

    first = build_container(AppConfig(clock="frozen", generator="extractive",
                                      cache="sqlite", sqlite_path=db))
    try:
        _ingest(first, CORPUS_A)
        original = first.ask(QUESTION)
        assert "90 days" in original.text
        assert first.cache.stats().entries == 1
    finally:
        first.close()

    second = build_container(AppConfig(clock="frozen", generator="extractive",
                                       cache="memory", sqlite_path=db))
    try:
        _ingest(second, CORPUS_B)
        second.rebind(cache="sqlite")
        assert second.cache.stats().entries == 1, "the old entry is still in the file"

        answer = second.ask(QUESTION)
        assert answer.cached is False
        assert "90 days" not in answer.text
        assert original.text not in answer.text
    finally:
        second.close()


def test_the_cache_key_changes_when_the_corpus_does(tmp_path):
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        _ingest(container, CORPUS_A)
        key_a = AskQuestion(container.index, container.generator, container.cache,
                            container.clock,
                            corpus_fingerprint=container.fingerprint).cache_key(QUESTION, 4)
        _ingest(container, CORPUS_B)
        key_b = AskQuestion(container.index, container.generator, container.cache,
                            container.clock,
                            corpus_fingerprint=container.fingerprint).cache_key(QUESTION, 4)
    finally:
        container.close()
    assert key_a != key_b


def test_editing_a_document_changes_the_fingerprint(tmp_path):
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        first = _ingest(container, CORPUS_A).fingerprint
        edited = {"retention": CORPUS_A["retention"].replace("90 days", "180 days")}
        second = _ingest(container, edited).fingerprint
    finally:
        container.close()
    assert first != second, "same document ids, different text -- the cache must not be shared"


def test_re_ingesting_the_identical_corpus_keeps_the_cache_warm():
    """A durable cache that is emptied on every ingest is durable across nothing."""
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        _ingest(container, CORPUS_A)
        container.ask(QUESTION)
        assert container.cache.stats().entries == 1
        _ingest(container, CORPUS_A)
        assert container.cache.stats().entries == 1
        assert container.ask(QUESTION).cached is True
    finally:
        container.close()


def test_a_cached_answer_whose_markers_no_longer_resolve_is_recomputed():
    """Serving cached prose with its citations silently stripped is worse than regenerating."""
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        _ingest(container, CORPUS_A)
        use_case = AskQuestion(container.index, container.generator, container.cache,
                               container.clock, default_k=1,
                               corpus_fingerprint=container.fingerprint)
        key = use_case.cache_key(QUESTION, 1)
        container.cache.set(key, Generated(text="stale prose [7]", cited=(7,)))

        answer = use_case.execute(QUESTION, k=1)
        assert answer.cached is False
        assert answer.text != "stale prose [7]"
        assert answer.citations, "the recomputed answer is grounded again"
    finally:
        container.close()


def test_every_citation_points_at_a_retrieved_passage():
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        _ingest(container, {**CORPUS_A, **CORPUS_B})
        answer = container.ask(QUESTION)
        retrieved = {h.passage.passage_id for h in answer.hits}
        assert {c.passage_id for c in answer.citations} <= retrieved
        assert answer.citations, "an answer with no source is not an answer"
    finally:
        container.close()


# ------------------------------------------------------------------ durability

def test_the_sqlite_store_survives_a_restart(tmp_path):
    db = str(tmp_path / "corpus.sqlite3")
    config = AppConfig(clock="frozen", generator="extractive", store="sqlite", sqlite_path=db)

    first = build_container(config)
    try:
        report = _ingest(first, CORPUS_A)
        assert first.restored is False
    finally:
        first.close()

    second = build_container(config)
    try:
        assert second.restored is True
        assert second.fingerprint == report.fingerprint
        assert second.index.size() == report.passages
        answer = second.ask(QUESTION)   # no re-ingest needed
        assert "90 days" in answer.text
    finally:
        second.close()


def test_restored_passage_ids_are_identical_so_old_citations_still_resolve(tmp_path):
    db = str(tmp_path / "ids.sqlite3")
    config = AppConfig(clock="frozen", generator="extractive", store="sqlite", sqlite_path=db)

    first = build_container(config)
    try:
        _ingest(first, {**CORPUS_A, **CORPUS_B})
        before = [p.passage_id for p in first.passages]
    finally:
        first.close()

    second = build_container(config)
    try:
        assert [p.passage_id for p in second.passages] == before
    finally:
        second.close()


def test_the_memory_store_loses_the_corpus_on_restart(tmp_path):
    """The default is not durable, and that is a choice the operator can see on /health."""
    config = AppConfig(clock="frozen", generator="extractive", store="memory")
    first = build_container(config)
    try:
        _ingest(first, CORPUS_A)
    finally:
        first.close()

    second = build_container(config)
    try:
        assert second.restored is False
        assert second.index.size() == 0
    finally:
        second.close()


def test_health_reports_the_restored_corpus_over_http(tmp_path, corpus_roots):
    db = str(tmp_path / "http.sqlite3")
    config = AppConfig(clock="frozen", generator="extractive", store="sqlite",
                       sqlite_path=db, corpus_roots=corpus_roots)
    with running_service(config) as client:
        client.ingest_documents(CORPUS_A)

    with running_service(config) as client:
        health = client.health()
        assert health["corpus"]["restored"] is True
        assert health["indexed_passages"] == 1
        assert client.ask(QUESTION).answer


def test_asking_before_any_ingest_still_conflicts(corpus_roots):
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config) as client:
        with pytest.raises(ServiceError) as excinfo:
            client.ask(QUESTION)
    assert excinfo.value.code == "corpus_not_ingested"


def test_fingerprint_is_stable_for_the_same_passages():
    container = build_container(AppConfig(clock="frozen", generator="extractive"))
    try:
        report = _ingest(container, CORPUS_A)
        assert report.fingerprint == corpus_fingerprint(container.passages)
    finally:
        container.close()
