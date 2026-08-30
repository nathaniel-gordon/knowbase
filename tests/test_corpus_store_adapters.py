"""Run the CorpusStorePort battery against both adapters, plus the migration mechanism.

The migration tests are here rather than in a module of their own because the only reason the
runner exists is that these two adapters share a database file with a schema that will change.
"""
from __future__ import annotations

import sqlite3

import pytest

from contracts.corpus_store_contract import SNAPSHOT, CorpusStoreContract
from knowbase.adapters.db import LATEST_VERSION, MIGRATIONS, connect, migrate, schema_version
from knowbase.adapters.store.memory import InMemoryCorpusStore
from knowbase.adapters.store.sqlite import SqliteCorpusStore
from knowbase.domain.errors import CorpusStoreUnavailable


class TestInMemoryCorpusStore(CorpusStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryCorpusStore()


class TestSqliteCorpusStore(CorpusStoreContract):
    @pytest.fixture
    def store(self, frozen_clock, tmp_path):
        adapter = SqliteCorpusStore(frozen_clock, tmp_path / "db" / "knowbase.sqlite3")
        yield adapter
        adapter.close()

    def test_a_restart_finds_the_corpus_still_there(self, frozen_clock, tmp_path):
        """The whole point of the adapter: no re-ingest after a restart."""
        path = tmp_path / "restart.sqlite3"
        first = SqliteCorpusStore(frozen_clock, path)
        first.save(SNAPSHOT)
        first.close()

        second = SqliteCorpusStore(frozen_clock, path)
        try:
            assert second.load() == SNAPSHOT
        finally:
            second.close()

    def test_passages_are_stored_as_plain_readable_rows(self, frozen_clock, tmp_path):
        path = tmp_path / "plain.sqlite3"
        adapter = SqliteCorpusStore(frozen_clock, path)
        adapter.save(SNAPSHOT)
        adapter.close()

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute("SELECT doc_id, ordinal FROM passage ORDER BY position").fetchall()
        finally:
            conn.close()
        assert rows == [("pto", 0), ("pto", 1), ("passwords", 0)]

    def test_replacing_a_snapshot_cascades_its_passages_away(self, frozen_clock, tmp_path):
        path = tmp_path / "cascade.sqlite3"
        adapter = SqliteCorpusStore(frozen_clock, path)
        adapter.save(SNAPSHOT)
        adapter.clear()
        try:
            assert adapter.size() == 0, "the foreign key cascade must take the passages too"
        finally:
            adapter.close()

    def test_a_broken_connection_surfaces_as_a_domain_error(self, frozen_clock, tmp_path):
        adapter = SqliteCorpusStore(frozen_clock, tmp_path / "closed.sqlite3")
        adapter.close()
        with pytest.raises(CorpusStoreUnavailable):
            adapter.size()


def test_every_registered_store_adapter_has_a_contract_subclass():
    from knowbase.adapters import STORE_ADAPTERS
    assert {cls.__name__ for cls in STORE_ADAPTERS.values()} == {"InMemoryCorpusStore",
                                                                 "SqliteCorpusStore"}


# ------------------------------------------------------------------ migrations

def test_a_fresh_database_lands_on_the_latest_version(tmp_path):
    conn = connect(tmp_path / "fresh.sqlite3")
    try:
        assert schema_version(conn) == 0
        assert migrate(conn) == LATEST_VERSION
    finally:
        conn.close()


def test_migrations_are_ordered_and_numbered_without_gaps():
    versions = [m.version for m in MIGRATIONS]
    assert versions == list(range(1, len(MIGRATIONS) + 1))


def test_migrating_twice_is_a_no_op(tmp_path):
    path = tmp_path / "twice.sqlite3"
    conn = connect(path)
    try:
        migrate(conn)
        migrate(conn)
        (applied,) = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
    finally:
        conn.close()
    assert applied == LATEST_VERSION


def test_an_older_schema_is_upgraded_without_losing_rows(tmp_path):
    """Stop at migration 1, write a cache row the old way, then run the rest over it."""
    path = tmp_path / "legacy.sqlite3"
    conn = connect(path)
    try:
        migrate(conn, MIGRATIONS[:1])
        assert schema_version(conn) == 1
        conn.execute("INSERT INTO answers (key, text, cited, expires_at) VALUES (?, ?, ?, ?)",
                     ("legacy", "an answer written before the upgrade", "[1]", None))
        conn.commit()

        assert migrate(conn) == LATEST_VERSION
        row = conn.execute(
            "SELECT text, cited, created_at FROM answers WHERE key = 'legacy'").fetchone()
    finally:
        conn.close()
    assert row[0] == "an answer written before the upgrade"
    assert row[1] == "[1]"
    assert row[2] == 0, "the backfilled column takes its default, not a made-up timestamp"


def test_a_database_from_a_newer_build_is_refused(tmp_path):
    path = tmp_path / "future.sqlite3"
    conn = connect(path)
    try:
        migrate(conn)
        conn.execute("INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                     (LATEST_VERSION + 5, "from the future", 0.0))
        conn.commit()
        with pytest.raises(CorpusStoreUnavailable, match="newer build"):
            migrate(conn)
    finally:
        conn.close()


def test_a_failing_migration_leaves_the_last_good_version_in_place(tmp_path):
    from knowbase.adapters.db import Migration

    broken = (*MIGRATIONS, Migration(LATEST_VERSION + 1, "broken", ("NOT VALID SQL",)))
    path = tmp_path / "broken.sqlite3"
    conn = connect(path)
    try:
        with pytest.raises(CorpusStoreUnavailable, match="migration"):
            migrate(conn, broken)
        assert schema_version(conn) == LATEST_VERSION
    finally:
        conn.close()
