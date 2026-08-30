"""SQLite-backed answer cache -- durable, shared between worker threads, inspectable.

Serialisation is the adapter's own business: the domain hands over a :class:`Generated` and
gets an equal one back. JSON is used rather than pickle so the table stays readable with any
sqlite client, and so a cache file can never execute anything when it is read back.

The schema is owned by ``adapters/db.py`` and reached through its migration runner, not by a
``CREATE TABLE IF NOT EXISTS`` here: a cache that silently mis-reads a changed schema returns
wrong answers rather than failing.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ...domain.errors import CorpusStoreUnavailable
from ...domain.model import CacheStats, Generated
from ...ports import ClockPort
from ..db import open_database


class SqliteCache:
    """Durable cache. ``path=':memory:'`` gives an ephemeral store with identical semantics."""

    name = "sqlite"

    def __init__(self, clock: ClockPort, path: str | Path = ":memory:") -> None:
        self._clock = clock
        self._path = str(path)
        self._conn = open_database(self._path)
        # SQLite serialises the statements; the counters are plain Python ints and do not.
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Generated | None:
        row = self._query(
            "SELECT text, cited, expires_at FROM answers WHERE key = ?", (key,)).fetchone()
        if row is None:
            self._count(hit=False)
            return None
        text, cited_json, expires_at = row
        if expires_at is not None and self._clock.now() >= expires_at:
            self.delete(key)
            self._count(hit=False)
            return None
        self._count(hit=True)
        return Generated(text=text, cited=tuple(json.loads(cited_json)))

    def set(self, key: str, value: Generated, ttl_s: float | None = None) -> None:
        now = self._clock.now()
        expires_at = None if ttl_s is None else now + ttl_s
        self._write(
            "INSERT INTO answers (key, text, cited, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET text=excluded.text, cited=excluded.cited, "
            "created_at=excluded.created_at, expires_at=excluded.expires_at",
            (key, value.text, json.dumps(list(value.cited)), now, expires_at))

    def delete(self, key: str) -> bool:
        return self._write("DELETE FROM answers WHERE key = ?", (key,)).rowcount > 0

    def clear(self) -> None:
        self._write("DELETE FROM answers")

    def stats(self) -> CacheStats:
        (entries,) = self._query("SELECT COUNT(*) FROM answers").fetchone()
        with self._lock:
            return CacheStats(name=self.name, entries=int(entries),
                              hits=self._hits, misses=self._misses)

    def close(self) -> None:
        """Release the connection; the service calls this on shutdown (Windows file locks)."""
        self._conn.close()

    # --------------------------------------------------------------- internals

    def _count(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._hits += 1
            else:
                self._misses += 1

    def _query(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("the answer cache cannot be read",
                                         reason=type(exc).__name__) from exc

    def _write(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        try:
            with self._conn:
                return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("the answer cache cannot be written",
                                         reason=type(exc).__name__) from exc
