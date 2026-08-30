"""Process-local corpus store: the default, and the one that loses everything on restart.

It exists so the store is a real port with two adapters rather than a sqlite dependency
bolted onto the container, and so tests that do not care about durability pay nothing for it.
Choosing it is choosing not to persist, which ``GET /health`` says out loud.
"""
from __future__ import annotations

import threading

from ...domain.model import CorpusSnapshot


class InMemoryCorpusStore:
    """Holds one snapshot in memory. Swapping it is a single guarded assignment."""

    name = "memory"

    def __init__(self) -> None:
        self._snapshot: CorpusSnapshot | None = None
        self._lock = threading.Lock()

    def save(self, snapshot: CorpusSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def load(self) -> CorpusSnapshot | None:
        return self._snapshot

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None

    def size(self) -> int:
        snapshot = self._snapshot
        return 0 if snapshot is None else len(snapshot.passages)
