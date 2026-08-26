"""Filesystem document source: a folder of .md/.txt files, walked in sorted order."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from ...domain.errors import DocumentSourceUnavailable
from ...domain.model import Document
from ...domain.text import title_of

DEFAULT_EXTS: tuple[str, ...] = (".md", ".txt", ".markdown")


class FilesystemDocumentSource:
    """Reads every matching file under ``root`` recursively.

    Sorted traversal is part of the port contract, not a convenience: ingestion order fixes
    passage ordinals, and passage ids appear in citations that users may bookmark.
    """

    name = "filesystem"

    def __init__(self, root: str | Path, exts: Sequence[str] = DEFAULT_EXTS) -> None:
        self._root = Path(root)
        self._exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)

    def describe(self) -> str:
        return f"filesystem:{self._root}"

    def documents(self) -> Iterator[Document]:
        if not self._root.is_dir():
            raise DocumentSourceUnavailable(f"not a directory: {self._root}")
        files = sorted(p for p in self._root.rglob("*")
                       if p.is_file() and p.suffix.lower() in self._exts)
        if not files:
            raise DocumentSourceUnavailable(
                f"no {'/'.join(self._exts)} files under {self._root}")
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            yield Document(doc_id=path.stem,
                           title=title_of(text, path.stem),
                           text=text,
                           origin=str(path))
