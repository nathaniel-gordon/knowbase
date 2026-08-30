"""SQLite corpus store -- the adapter that makes a restart survivable.

The passages of the current ingest are written in one transaction against the schema in
``adapters/db.py``. Replacing a snapshot deletes the singleton row and lets the foreign key
cascade take its passages with it, so there is never a moment where half of corpus A and half
of corpus B are both readable.

Every statement binds its parameters. There is no string interpolation into SQL anywhere in
this file, and the values stored are plain text -- no pickle, so a corpus file cannot execute
anything when it is read back.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ...domain.errors import CorpusStoreUnavailable
from ...domain.model import CorpusSnapshot, Passage
from ...ports import ClockPort
from ..db import open_database


class SqliteCorpusStore:
    """Durable single-snapshot corpus store. ``path=':memory:'`` keeps the same semantics."""

    name = "sqlite"

    def __init__(self, clock: ClockPort, path: str | Path = ":memory:") -> None:
        self._clock = clock
        self._path = str(path)
        self._conn = open_database(self._path)

    def save(self, snapshot: CorpusSnapshot) -> None:
        rows = [(position, 1, p.doc_id, p.ordinal, p.title, p.origin, p.text)
                for position, p in enumerate(snapshot.passages)]
        try:
            with self._conn:
                self._conn.execute("DELETE FROM corpus_snapshot")
                self._conn.execute(
                    "INSERT INTO corpus_snapshot "
                    "(id, fingerprint, source, documents, started_at, saved_at) "
                    "VALUES (1, ?, ?, ?, ?, ?)",
                    (snapshot.fingerprint, snapshot.source, snapshot.documents,
                     snapshot.started_at, self._clock.now()))
                self._conn.executemany(
                    "INSERT INTO passage "
                    "(position, snapshot_id, doc_id, ordinal, title, origin, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot write the corpus snapshot",
                                         reason=type(exc).__name__) from exc

    def load(self) -> CorpusSnapshot | None:
        try:
            header = self._conn.execute(
                "SELECT fingerprint, source, documents, started_at "
                "FROM corpus_snapshot WHERE id = 1").fetchone()
            if header is None:
                return None
            rows = self._conn.execute(
                "SELECT doc_id, ordinal, text, title, origin FROM passage "
                "ORDER BY position").fetchall()
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot read the corpus snapshot",
                                         reason=type(exc).__name__) from exc
        if not rows:
            # A header with no passages is not a corpus anyone may answer from.
            return None
        fingerprint, source, documents, started_at = header
        passages = tuple(Passage(doc_id=r[0], ordinal=int(r[1]), text=r[2],
                                 title=r[3], origin=r[4]) for r in rows)
        return CorpusSnapshot(fingerprint=fingerprint, source=source,
                              documents=int(documents), started_at=started_at,
                              passages=passages)

    def clear(self) -> None:
        try:
            with self._conn:
                self._conn.execute("DELETE FROM corpus_snapshot")
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot clear the corpus snapshot",
                                         reason=type(exc).__name__) from exc

    def size(self) -> int:
        try:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM passage").fetchone()
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot read the corpus snapshot",
                                         reason=type(exc).__name__) from exc
        return int(count)

    def close(self) -> None:
        """Release the connection; the service calls this on shutdown (Windows file locks)."""
        self._conn.close()
