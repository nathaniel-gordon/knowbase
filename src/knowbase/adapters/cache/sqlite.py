"""SQLite-backed answer cache -- survives a service restart, shareable between workers.

Serialisation is the adapter's own business: the domain hands over a :class:`Generated` and
gets an equal one back. JSON is used rather than pickle so the table stays inspectable with any
sqlite client, which matters when an operator wants to audit what the service answered.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ...domain.model import CacheStats, Generated
from ...ports import ClockPort

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key        TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    cited      TEXT NOT NULL,
    expires_at REAL
)
"""


class SqliteCache:
    """Durable cache. ``path=':memory:'`` gives an ephemeral store with identical semantics."""

    name = "sqlite"

    def __init__(self, clock: ClockPort, path: str | Path = ":memory:") -> None:
        self._clock = clock
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the HTTP server is threaded and the connection is guarded by
        # sqlite's own locking plus short, single-statement transactions.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Generated | None:
        row = self._conn.execute(
            "SELECT text, cited, expires_at FROM answers WHERE key = ?", (key,)).fetchone()
        if row is None:
            self._misses += 1
            return None
        text, cited_json, expires_at = row
        if expires_at is not None and self._clock.now() >= expires_at:
            self.delete(key)
            self._misses += 1
            return None
        self._hits += 1
        return Generated(text=text, cited=tuple(json.loads(cited_json)))

    def set(self, key: str, value: Generated, ttl_s: float | None = None) -> None:
        expires_at = None if ttl_s is None else self._clock.now() + ttl_s
        self._conn.execute(
            "INSERT INTO answers (key, text, cited, expires_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET text=excluded.text, cited=excluded.cited, "
            "expires_at=excluded.expires_at",
            (key, value.text, json.dumps(list(value.cited)), expires_at))
        self._conn.commit()

    def delete(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM answers WHERE key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM answers")
        self._conn.commit()

    def stats(self) -> CacheStats:
        (entries,) = self._conn.execute("SELECT COUNT(*) FROM answers").fetchone()
        return CacheStats(name=self.name, entries=int(entries),
                          hits=self._hits, misses=self._misses)

    def close(self) -> None:
        """Release the connection; the service calls this on shutdown (Windows file locks)."""
        self._conn.close()
