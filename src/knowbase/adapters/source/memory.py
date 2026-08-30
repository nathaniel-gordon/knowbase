"""In-memory document source: documents handed over as a mapping.

This is the adapter the HTTP ``/ingest`` endpoint uses when a client posts documents inline,
and the one tests use to build fixture corpora without touching disk.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping

from ...domain.errors import DocumentSourceUnavailable
from ...domain.model import Document
from ...domain.text import title_of


class InMemoryDocumentSource:
    """Serves ``{doc_id: text}`` in sorted key order."""

    name = "memory"

    def __init__(self, documents: Mapping[str, str], label: str = "inline") -> None:
        self._documents = dict(documents)
        self._label = label

    def describe(self) -> str:
        return f"memory:{self._label}({len(self._documents)} docs)"

    def documents(self) -> Iterator[Document]:
        if not self._documents:
            raise DocumentSourceUnavailable("in-memory source holds no documents")
        for doc_id in sorted(self._documents):
            text = self._documents[doc_id]
            if not text.strip():
                continue
            yield Document(doc_id=doc_id, title=title_of(text, doc_id), text=text,
                           origin=f"memory://{self._label}/{doc_id}")
