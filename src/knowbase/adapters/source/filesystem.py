"""Filesystem document source: a folder of .md/.txt files, walked in sorted order.

The walk is bounded in three directions -- how many directory entries it may visit, how many
matching files it may take, and how large any one file may be -- because the root is chosen by
whoever calls ``/ingest``. An unbounded ``rglob`` over a client-supplied path is a denial of
service against a threaded server, not a convenience.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

from ...domain.errors import DocumentSourceUnavailable, ResourceLimitExceeded
from ...domain.model import Document
from ...domain.text import title_of

DEFAULT_EXTS: tuple[str, ...] = (".md", ".txt", ".markdown")
DEFAULT_MAX_FILES = 2_000
DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_SCANNED_ENTRIES = 50_000


class FilesystemDocumentSource:
    """Reads every matching file under ``root`` recursively.

    Sorted traversal is part of the port contract, not a convenience: ingestion order fixes
    passage ordinals, and passage ids appear in citations that users may bookmark.
    """

    name = "filesystem"

    def __init__(self, root: str | Path, exts: Sequence[str] = DEFAULT_EXTS,
                 max_files: int = DEFAULT_MAX_FILES,
                 max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
                 max_scanned_entries: int = DEFAULT_MAX_SCANNED_ENTRIES) -> None:
        self._root = Path(root)
        self._exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)
        self._max_files = int(max_files)
        self._max_file_bytes = int(max_file_bytes)
        self._max_scanned_entries = int(max_scanned_entries)

    def describe(self) -> str:
        return f"filesystem:{self._root.name or self._root}"

    def documents(self) -> Iterator[Document]:
        root = self._root.resolve()
        for path in self._matching_files():
            size = path.stat().st_size
            if size > self._max_file_bytes:
                raise ResourceLimitExceeded(
                    f"document exceeds the {self._max_file_bytes} byte per-file limit",
                    document=path.name, size=size, limit=self._max_file_bytes)
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            # Origin is relative to the ingest root: enough for a reader to find the file
            # again, without printing the server's directory layout into every citation.
            relative = path.resolve().relative_to(root).as_posix()
            yield Document(doc_id=path.stem,
                           title=title_of(text, path.stem),
                           text=text,
                           origin=f"file://{relative}")

    # ------------------------------------------------------------------ walk

    def _matching_files(self) -> list[Path]:
        """Bounded, sorted, symlink-safe listing of the candidate files."""
        if not self._root.is_dir():
            raise DocumentSourceUnavailable(f"not a directory: {self._root.name or self._root}")
        root = self._root.resolve()
        found: list[Path] = []
        scanned = 0
        for entry in self._root.rglob("*"):
            scanned += 1
            if scanned > self._max_scanned_entries:
                raise ResourceLimitExceeded(
                    f"corpus root holds more than {self._max_scanned_entries} entries",
                    limit=self._max_scanned_entries)
            if entry.suffix.lower() not in self._exts:
                continue
            try:
                if not entry.is_file():
                    continue
                real = entry.resolve()
            except OSError:
                continue
            # A symlink inside an allowed root can point anywhere; containment is therefore
            # re-checked per file and not only for the root itself.
            if real != root and root not in real.parents:
                continue
            found.append(entry)
            if len(found) > self._max_files:
                raise ResourceLimitExceeded(
                    f"corpus root holds more than {self._max_files} matching documents",
                    limit=self._max_files)
        if not found:
            raise DocumentSourceUnavailable(
                f"no {'/'.join(self._exts)} files under {self._root.name or self._root}")
        return sorted(found)
