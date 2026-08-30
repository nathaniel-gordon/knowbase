"""Contract battery for CorpusStorePort.

The store decides whether a restart costs a re-ingest, so the battery is about identity and
replacement, not about speed: a snapshot must come back byte-for-byte and in order, and
saving a new one must leave nothing of the old one behind.
"""
from __future__ import annotations

import pytest

from knowbase.domain.model import CorpusSnapshot, Passage
from knowbase.ports import CorpusStorePort

SNAPSHOT = CorpusSnapshot(
    fingerprint="fp-a",
    source="filesystem",
    documents=2,
    started_at="2024-01-01T00:00:00+00:00",
    passages=(
        Passage(doc_id="pto", ordinal=0, text="Employees receive 21 paid vacation days.",
                title="Time Off Policy", origin="file://pto.md"),
        Passage(doc_id="pto", ordinal=1, text="Unused days roll over once.",
                title="Time Off Policy", origin="file://pto.md"),
        Passage(doc_id="passwords", ordinal=0, text="Passwords must be 14 characters.",
                title="Password Rules", origin="file://passwords.md"),
    ))

OTHER = CorpusSnapshot(
    fingerprint="fp-b",
    source="memory",
    documents=1,
    started_at="2024-02-02T00:00:00+00:00",
    passages=(Passage(doc_id="only", ordinal=0, text="This is the only document now."),))

UNICODE = CorpusSnapshot(
    fingerprint="fp-u",
    source="memory",
    documents=1,
    started_at="2024-03-03T00:00:00+00:00",
    passages=(Passage(doc_id="fr", ordinal=0,
                      text="Réunion trimestrielle — 5 jours ouvrés · 90 % couverture",
                      title="Réunion", origin="memory://fr"),))


class CorpusStoreContract:
    """Every CorpusStorePort adapter must pass all of these.

    Subclasses provide one ``store`` fixture and nothing else.
    """

    @pytest.fixture
    def store(self) -> CorpusStorePort:
        raise NotImplementedError("contract subclass must provide a 'store' fixture")

    def test_satisfies_the_protocol(self, store):
        assert isinstance(store, CorpusStorePort)
        assert isinstance(store.name, str) and store.name == store.name.lower()

    def test_an_empty_store_holds_nothing(self, store):
        assert store.load() is None
        assert store.size() == 0

    def test_round_trip_preserves_the_snapshot(self, store):
        store.save(SNAPSHOT)
        assert store.load() == SNAPSHOT

    def test_passage_order_is_preserved(self, store):
        store.save(SNAPSHOT)
        loaded = store.load()
        assert loaded is not None
        assert [p.passage_id for p in loaded.passages] == [p.passage_id
                                                           for p in SNAPSHOT.passages]

    def test_size_counts_passages(self, store):
        store.save(SNAPSHOT)
        assert store.size() == len(SNAPSHOT.passages)

    def test_saving_replaces_the_previous_snapshot_entirely(self, store):
        """A stale passage surviving a re-ingest is a way to answer from a corpus nobody loaded."""
        store.save(SNAPSHOT)
        store.save(OTHER)
        loaded = store.load()
        assert loaded == OTHER
        assert store.size() == len(OTHER.passages)

    def test_non_ascii_text_survives(self, store):
        store.save(UNICODE)
        loaded = store.load()
        assert loaded is not None
        assert loaded.passages[0].text == UNICODE.passages[0].text

    def test_clear_empties_the_store(self, store):
        store.save(SNAPSHOT)
        store.clear()
        assert store.load() is None
        assert store.size() == 0

    def test_empty_fields_round_trip_as_empty_strings(self, store):
        bare = CorpusSnapshot(fingerprint="fp-bare", source="memory", documents=1,
                              started_at="2024-01-01T00:00:00+00:00",
                              passages=(Passage(doc_id="d", ordinal=0, text="body"),))
        store.save(bare)
        loaded = store.load()
        assert loaded is not None
        assert (loaded.passages[0].title, loaded.passages[0].origin) == ("", "")
