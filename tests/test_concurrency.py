"""The read path under a concurrent re-ingest.

The hazard was visible in the code rather than at runtime: an index adapter assigned its
passage list and each scorer in sequence, so a thread inside ``search`` could read new passages
against an old matrix. The fix is to publish an immutable state object with one assignment.
These tests are what turn that from an argument into a fact, so they hammer the seam directly
rather than trusting the shape of the code.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from knowbase.adapters.index.bm25_only import Bm25Index
from knowbase.adapters.index.hybrid_rrf import HybridRrfIndex
from knowbase.adapters.index.tfidf_only import TfidfIndex
from knowbase.adapters.cache.memory import InMemoryCache
from knowbase.adapters.cache.sqlite import SqliteCache
from knowbase.app.client import ServiceError, running_service
from knowbase.app.config import AppConfig
from knowbase.domain.model import Generated, Passage

GENERATIONS = 6
READERS = 6
READS_PER_THREAD = 40


def _passages(generation: int, count: int = 12) -> list[Passage]:
    """A corpus whose every passage carries its generation, so a mix is detectable."""
    return [Passage(doc_id=f"g{generation}-doc{i}", ordinal=0,
                    title=f"Generation {generation}",
                    text=(f"generation{generation} document{i} covers vacation policy, "
                          f"password rotation and remote access rules in detail."))
            for i in range(count)]


@pytest.mark.parametrize("factory", [HybridRrfIndex, Bm25Index, TfidfIndex],
                         ids=lambda f: f.name)
def test_search_never_sees_a_half_built_index(factory):
    index = factory()
    index.build(_passages(0))
    stop = threading.Event()
    seen: list[set[str]] = []
    failures: list[BaseException] = []

    def read() -> None:
        try:
            while not stop.is_set():
                hits = index.search("vacation policy password remote access", k=5)
                if hits:
                    # Every hit of one search must come from one generation. A mixed result is
                    # the corruption this test exists to catch.
                    seen.append({h.passage.title for h in hits})
        except BaseException as exc:  # noqa: BLE001 - the whole point is to collect it
            failures.append(exc)

    readers = [threading.Thread(target=read, name=f"reader-{i}") for i in range(READERS)]
    for thread in readers:
        thread.start()
    try:
        for generation in range(1, GENERATIONS + 1):
            index.build(_passages(generation))
    finally:
        stop.set()
        for thread in readers:
            thread.join(timeout=30)

    assert not failures, f"{factory.name} raised under concurrent rebuild: {failures[:3]}"
    assert seen, "the readers never observed a result"
    mixed = [titles for titles in seen if len(titles) > 1]
    assert not mixed, f"{factory.name} returned passages from several corpora at once: {mixed[:3]}"


def test_ask_and_ingest_can_run_at_the_same_time(corpus_roots):
    """Six askers against six re-ingests: no errors, and no answer from a corpus not loaded."""
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config) as client:
        client.ingest_documents({"g0": "# Corpus 0\nGeneration 0 holds the vacation policy."})
        stop = threading.Event()
        errors: list[str] = []
        answers: list[str] = []

        def ask() -> None:
            while not stop.is_set():
                try:
                    answers.append(client.ask("vacation policy generation").answer)
                except ServiceError as exc:
                    if exc.code != "corpus_not_ingested":
                        errors.append(f"{exc.code}: {exc.message}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

        with ThreadPoolExecutor(max_workers=READERS) as pool:
            futures = [pool.submit(ask) for _ in range(READERS)]
            try:
                for generation in range(1, GENERATIONS + 1):
                    client.ingest_documents({
                        f"g{generation}": f"# Corpus {generation}\n"
                                          f"Generation {generation} holds the vacation policy."})
            finally:
                stop.set()
                for future in futures:
                    future.result(timeout=30)

        assert not errors, errors[:3]
        assert answers
        for answer in answers:
            generations = {word for word in answer.split() if word.isdigit()}
            assert len(generations) <= 1, f"an answer mixed corpora: {answer}"


@pytest.mark.parametrize("cache_factory", ["memory", "sqlite"])
def test_cache_counters_are_not_lost_under_concurrent_use(cache_factory, frozen_clock, tmp_path):
    """``self._hits += 1`` is not atomic, and the counters are what /metrics reports."""
    cache = (InMemoryCache(frozen_clock) if cache_factory == "memory"
             else SqliteCache(frozen_clock, tmp_path / "counters.sqlite3"))
    try:
        cache.set("present", Generated(text="an answer", cited=(1,)))
        rounds = 200

        def work() -> None:
            for _ in range(rounds):
                cache.get("present")
                cache.get("absent")

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(work) for _ in range(8)]:
                future.result(timeout=60)

        stats = cache.stats()
        assert stats.hits == 8 * rounds
        assert stats.misses == 8 * rounds
    finally:
        closer = getattr(cache, "close", None)
        if closer:
            closer()


def test_the_sqlite_cache_tolerates_concurrent_writers(frozen_clock, tmp_path):
    """One connection shared across threads, which the adapter has always claimed is safe."""
    cache = SqliteCache(frozen_clock, tmp_path / "writers.sqlite3")
    try:
        def work(worker: int) -> None:
            for i in range(50):
                key = f"w{worker}-{i}"
                cache.set(key, Generated(text=f"answer {key}", cited=(1,)))
                assert cache.get(key) is not None

        with ThreadPoolExecutor(max_workers=6) as pool:
            for future in [pool.submit(work, w) for w in range(6)]:
                future.result(timeout=120)

        assert cache.stats().entries == 300
    finally:
        cache.close()
