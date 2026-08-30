"""Run the CachePort battery against the in-memory and sqlite adapters."""
from __future__ import annotations

import sqlite3

import pytest

from contracts.cache_contract import VALUE, CacheContract
from knowbase.adapters.cache.memory import InMemoryCache
from knowbase.adapters.cache.sqlite import SqliteCache


class TestInMemoryCache(CacheContract):
    @pytest.fixture
    def cache(self, frozen_clock):
        return InMemoryCache(frozen_clock)


class TestSqliteCache(CacheContract):
    @pytest.fixture
    def cache(self, frozen_clock, tmp_path):
        adapter = SqliteCache(frozen_clock, tmp_path / "cache" / "answers.sqlite3")
        yield adapter
        adapter.close()

    def test_entries_are_readable_by_any_sqlite_client(self, frozen_clock, tmp_path):
        """Durability is the reason this adapter exists; prove the rows are plain and inspectable."""
        path = tmp_path / "durable.sqlite3"
        adapter = SqliteCache(frozen_clock, path)
        adapter.set("k", VALUE)
        adapter.close()

        conn = sqlite3.connect(path)
        try:
            text, cited = conn.execute("SELECT text, cited FROM answers WHERE key='k'").fetchone()
        finally:
            conn.close()
        assert text == VALUE.text and cited == "[1]"

    def test_a_restart_keeps_the_entries(self, frozen_clock, tmp_path):
        path = tmp_path / "restart.sqlite3"
        first = SqliteCache(frozen_clock, path)
        first.set("k", VALUE)
        first.close()

        second = SqliteCache(frozen_clock, path)
        try:
            assert second.get("k") == VALUE
            assert second.stats().entries == 1
        finally:
            second.close()


def test_every_registered_cache_adapter_has_a_contract_subclass():
    from knowbase.adapters import CACHE_ADAPTERS
    assert {cls.__name__ for cls in CACHE_ADAPTERS.values()} == {"InMemoryCache", "SqliteCache"}
