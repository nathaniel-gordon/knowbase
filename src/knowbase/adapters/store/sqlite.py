"""SQLite corpus store -- the adapter that makes a restart survivable.

The passages of the current ingest are written in one transaction against the schema in
``adapters/db.py``. Replacing a snapshot deletes the singleton row and lets the foreign key
cascade take its passages with it, so there is never a moment where half of corpus A and half
of corpus B are both readable.

Every statement binds its parameters. There is no string interpolation into SQL anywhere in
this file, and the values stored are plain text -- no pickle, so a corpus file cannot execute
anything when it is read back. Cross-thread access is serialised by :class:`Database`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ...domain.errors import CorpusStoreUnavailable
from ...domain.model import CorpusSnapshot, Passage
from ...ports import ClockPort
from ..db import Database


class SqliteCorpusStore:
    """Durable single-snapshot corpus store. ``path=':memory:'`` keeps the same semantics."""

    name = "sqlite"

    def __init__(self, clock: ClockPort, path: str | Path = ":memory:") -> None:
        self._clock = clock
        self._path = str(path)
        self._db = Database(self._path)

    def save(self, snapshot: CorpusSnapshot) -> None:
        rows = [(position, 1, p.doc_id, p.ordinal, p.title, p.origin, p.text)
                for position, p in enumerate(snapshot.passages)]
        try:
            with self._db.transaction() as conn:
                conn.execute("DELETE FROM corpus_snapshot")
                conn.execute(
                    "INSERT INTO corpus_snapshot "
                    "(id, fingerprint, source, documents, started_at, saved_at) "
                    "VALUES (1, ?, ?, ?, ?, ?)",
                    (snapshot.fingerprint, snapshot.source, snapshot.documents,
                     snapshot.started_at, self._clock.now()))
                conn.executemany(
                    "INSERT INTO passage "
                    "(position, snapshot_id, doc_id, ordinal, title, origin, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot write the corpus snapshot",
                                         reason=type(exc).__name__) from exc

    def load(self) -> CorpusSnapshot | None:
        try:
            header = self._db.query_one(
                "SELECT fingerprint, source, documents, started_at "
                "FROM corpus_snapshot WHERE id = 1")
            if header is None:
                return None
            rows = self._db.query(
                "SELECT doc_id, ordinal, text, title, origin FROM passage ORDER BY position")
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
            self._db.write("DELETE FROM corpus_snapshot")
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot clear the corpus snapshot",
                                         reason=type(exc).__name__) from exc

    def size(self) -> int:
        try:
            row = self._db.query_one("SELECT COUNT(*) FROM passage")
        except sqlite3.Error as exc:
            raise CorpusStoreUnavailable("cannot read the corpus snapshot",
                                         reason=type(exc).__name__) from exc
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Release the connection; the service calls this on shutdown (Windows file locks)."""
        self._db.close()
