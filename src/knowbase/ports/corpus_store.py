"""CorpusStorePort -- durable custody of the ingested corpus."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.model import CorpusSnapshot


@runtime_checkable
class CorpusStorePort(Protocol):
    """Holds the passages of the most recent ingest so a restart does not lose the corpus.

    Only one snapshot is kept. A knowledge base has a current state, not a history: the
    previous corpus is not something ``/ask`` may ever answer from, so keeping it would only
    create a way to serve ungrounded text.

    Contract (enforced by ``tests/contracts/corpus_store_contract.py``):

    * ``load`` on an empty store returns ``None``.
    * ``save`` then ``load`` returns an equal :class:`CorpusSnapshot`, passages in the same
      order, with every field round-tripped including empty titles and non-ASCII text.
    * ``save`` replaces the previous snapshot outright; nothing from it survives.
    * ``size`` is the number of stored passages and is 0 for an empty store.
    * ``clear`` empties the store, after which ``load`` returns ``None`` again.
    * A failure of the backing store surfaces as ``CorpusStoreUnavailable``, never as a driver
      exception.
    """

    name: str

    def save(self, snapshot: CorpusSnapshot) -> None:
        """Replace the stored corpus with ``snapshot``, atomically."""
        ...

    def load(self) -> CorpusSnapshot | None:
        """Return the stored corpus, or ``None`` if nothing has been ingested."""
        ...

    def clear(self) -> None:
        """Drop the stored corpus."""
        ...

    def size(self) -> int:
        """Number of passages currently held."""
        ...
