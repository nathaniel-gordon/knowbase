"""DocumentSourcePort -- where raw knowledge-base documents come from."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..domain.model import Document


@runtime_checkable
class DocumentSourcePort(Protocol):
    """A repository of whole documents.

    Contract (enforced by ``tests/contracts/document_source_contract.py``):

    * ``name`` is a stable, lowercase adapter identifier.
    * ``documents()`` yields :class:`Document` values with unique, non-empty ``doc_id``.
    * Iteration order is deterministic and repeatable -- two passes yield the same ids in the
      same order, because ingestion order decides passage ordinals and therefore citations.
    * An unreadable backing store raises ``DocumentSourceUnavailable``, never a driver-specific
      exception, so use-cases stay technology-agnostic.
    """

    name: str

    def documents(self) -> Iterator[Document]:
        """Yield every document in the source, in a stable order."""
        ...

    def describe(self) -> str:
        """Human-readable location of the backing store (shown by ``/health``)."""
        ...
