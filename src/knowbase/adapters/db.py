"""The service database: one SQLite file, opened the same way everywhere, with migrations.

Two adapters persist state -- the answer cache and the corpus store -- and both used to be a
bare ``CREATE TABLE IF NOT EXISTS``. That is fine exactly once. The moment a column changes,
an existing file is read with the wrong assumptions and nobody finds out until the answers are
wrong, so schema changes go through the ordered list below instead.

The rules the runner follows:

* ``schema_version`` records every migration that has been applied, with a timestamp.
* Migrations are applied in order, each inside a transaction, so a failure leaves the database
  at the last good version rather than half-migrated.
* Migrations only ever move forward. There is no down-migration: reverting a deployment means
  restoring a backup, which is the honest answer for a service that holds a derived cache.
* A database newer than this build is refused rather than read with the wrong schema.

Connections use WAL so a reader is not blocked by the writer (the HTTP server is threaded),
and ``foreign_keys`` is switched on explicitly because SQLite leaves it off by default.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..domain.errors import CorpusStoreUnavailable


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward step. ``statements`` run in order inside a single transaction."""

    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "answer cache", (
        """
        CREATE TABLE IF NOT EXISTS answers (
            key        TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            cited      TEXT NOT NULL,
            expires_at REAL
        )
        """,
    )),
    Migration(2, "corpus store", (
        """
        CREATE TABLE corpus_snapshot (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            fingerprint TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            documents   INTEGER NOT NULL CHECK (documents >= 0),
            started_at  TEXT    NOT NULL,
            saved_at    REAL    NOT NULL
        )
        """,
        """
        CREATE TABLE passage (
            position    INTEGER PRIMARY KEY,
            snapshot_id INTEGER NOT NULL REFERENCES corpus_snapshot(id) ON DELETE CASCADE,
            doc_id      TEXT    NOT NULL,
            ordinal     INTEGER NOT NULL CHECK (ordinal >= 0),
            title       TEXT    NOT NULL DEFAULT '',
            origin      TEXT    NOT NULL DEFAULT '',
            text        TEXT    NOT NULL,
            UNIQUE (doc_id, ordinal)
        )
        """,
        "CREATE INDEX idx_passage_doc ON passage(doc_id)",
    )),
    Migration(3, "cache bookkeeping", (
        # Added after v1 shipped, which is why it is an ALTER rather than a wider CREATE:
        # an existing cache file keeps its rows and gains the column.
        "ALTER TABLE answers ADD COLUMN created_at REAL NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_answers_expires ON answers(expires_at)",
    )),
)

LATEST_VERSION = MIGRATIONS[-1].version

_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at REAL NOT NULL
)
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this service depends on.

    ``check_same_thread=False`` is deliberate -- the HTTP server is threaded -- but it is not
    on its own sufficient. Cross-thread use goes through :class:`Database`, which serialises
    access; see the note there for what a shared connection actually guarantees.
    """
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(target, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON")
        if target != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.Error as exc:
        raise CorpusStoreUnavailable("cannot open the service database",
                                     reason=type(exc).__name__) from exc
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """Highest applied migration, or 0 for a database this service has never touched."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    (version,) = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
    return int(version)


def migrate(conn: sqlite3.Connection,
            migrations: Sequence[Migration] = MIGRATIONS) -> int:
    """Bring ``conn`` up to the latest schema and return the version it now sits at."""
    conn.execute(_VERSION_TABLE)
    conn.commit()
    current = schema_version(conn)
    target = migrations[-1].version if migrations else 0
    if current > target:
        raise CorpusStoreUnavailable(
            "the database was written by a newer build of knowbase",
            found=current, supported=target)
    for migration in migrations:
        if migration.version <= current:
            continue
        try:
            with conn:  # one transaction per migration
                for statement in migration.statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, time.time()))
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable(
                f"migration {migration.version} ({migration.name}) failed",
                version=migration.version, reason=type(exc).__name__) from exc
    return schema_version(conn)


def open_database(path: str | Path) -> sqlite3.Connection:
    """Connect and migrate in one step -- the only way adapters should open the database."""
    conn = connect(path)
    try:
        migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


class Database:
    """A migrated connection plus the lock that makes it usable from several threads.

    The cache adapter used to carry a comment claiming sqlite's own locking was enough for a
    shared connection. It is not, and ``tests/test_concurrency.py`` shows why: sqlite
    serialises *statements*, but a transaction belongs to the connection, so two threads
    entering ``with conn:`` at once interleave one another's commits and a row that was just
    written can be missing on the next read.

    Serialising here is the cheap fix and the right one for this workload: every statement is
    short and the alternative -- a connection per thread -- buys nothing when the writes are
    single-row upserts.
    """

    def __init__(self, path: str | Path) -> None:
        self._conn = open_database(path)
        self._lock = threading.RLock()

    def query(self, sql: str, params: tuple[object, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple[object, ...] = ()) -> tuple[Any, ...] | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def write(self, sql: str, params: tuple[object, ...] = ()) -> int:
        """Run one statement in its own transaction; returns the affected row count."""
        with self._lock, self._conn:
            return int(self._conn.execute(sql, params).rowcount)

    @contextmanager
    def transaction(self) -> "Iterator[sqlite3.Connection]":
        """Hold the lock and one transaction for a multi-statement write."""
        with self._lock, self._conn:
            yield self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()
