"""IndexPort -- the retrieval strategy seam.

This is the port the A/B endpoint exploits: because ranking lives behind a protocol, the
service can build two indexes over the identical passage list and run the same eval battery
through both without the domain knowing that BM25 or TF-IDF exist.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.model import Passage, RetrievalHit


@runtime_checkable
class IndexPort(Protocol):
    """A searchable index over passages.

    Contract (enforced by ``tests/contracts/index_contract.py``):

    * ``build(passages)`` is idempotent and replaces any previous content.
    * ``search`` before ``build`` raises ``IndexNotBuiltError``.
    * ``search(q, k)`` returns at most ``k`` hits, ranks numbered ``1..n`` ascending, scores
      non-increasing, and every returned passage is one that was indexed.
    * Search is deterministic: the same query twice yields identical passage ids and scores.
    * Quality floor: on the shared fixture corpus the labelled passage is retrieved at rank 1.
    """

    name: str

    def build(self, passages: Sequence[Passage]) -> None:
        """(Re)build the index over ``passages``. Raises ``EmptyCorpusError`` if empty."""
        ...

    def search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """Return the top-``k`` passages for ``query``, best first."""
        ...

    def size(self) -> int:
        """Number of indexed passages (0 before ``build``)."""
        ...
