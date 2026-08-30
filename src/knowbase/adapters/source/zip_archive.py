"""Zip-archive document source -- knowledge bases often arrive as an export bundle.

A third adapter behind the same port is the cheapest possible proof that the contract battery
is doing real work: this file contains no retrieval logic and no domain rules, only translation
from ``zipfile`` to :class:`Document`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence

from ...domain.errors import DocumentSourceUnavailable
from ...domain.model import Document
from ...domain.text import title_of

DEFAULT_EXTS: tuple[str, ...] = (".md", ".txt", ".markdown")


class ZipDocumentSource:
    """Reads text members of a .zip archive in sorted member order."""

    name = "zip"

    def __init__(self, archive: str | Path, exts: Sequence[str] = DEFAULT_EXTS) -> None:
        self._archive = Path(archive)
        self._exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)

    def describe(self) -> str:
        return f"zip:{self._archive}"

    def documents(self) -> Iterator[Document]:
        if not self._archive.is_file():
            raise DocumentSourceUnavailable(f"archive not found: {self._archive}")
        try:
            zf = zipfile.ZipFile(self._archive)
        except zipfile.BadZipFile as exc:
            raise DocumentSourceUnavailable(f"corrupt archive {self._archive}: {exc}") from exc
        with zf:
            members = sorted(n for n in zf.namelist()
                             if not n.endswith("/")
                             and PurePosixPath(n).suffix.lower() in self._exts)
            if not members:
                raise DocumentSourceUnavailable(
                    f"no {'/'.join(self._exts)} members in {self._archive}")
            for member in members:
                text = zf.read(member).decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                doc_id = PurePosixPath(member).stem
                yield Document(doc_id=doc_id, title=title_of(text, doc_id), text=text,
                               origin=f"{self._archive}!{member}")
