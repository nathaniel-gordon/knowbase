"""Zip-archive document source -- knowledge bases often arrive as an export bundle.

A third adapter behind the same port is the cheapest possible proof that the contract battery
is doing real work: this file contains no retrieval logic and no domain rules, only translation
from ``zipfile`` to :class:`Document`.

It does own one piece of real defence, though. ``ZipFile.read`` decompresses a whole member
into memory, and an archive is client-supplied, so members are streamed under a hard byte cap
instead. The cap is applied to bytes actually read, not to the size the archive *claims* --
a bomb lies in its header.
"""
from __future__ import annotations

import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from ...domain.errors import DocumentSourceUnavailable, ResourceLimitExceeded
from ...domain.model import Document
from ...domain.text import title_of

DEFAULT_EXTS: tuple[str, ...] = (".md", ".txt", ".markdown")
DEFAULT_MAX_MEMBERS = 2_000
DEFAULT_MAX_MEMBER_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_CHUNK = 64 * 1024


class ZipDocumentSource:
    """Reads text members of a .zip archive in sorted member order."""

    name = "zip"

    def __init__(self, archive: str | Path, exts: Sequence[str] = DEFAULT_EXTS,
                 max_members: int = DEFAULT_MAX_MEMBERS,
                 max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
                 max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> None:
        self._archive = Path(archive)
        self._exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)
        self._max_members = int(max_members)
        self._max_member_bytes = int(max_member_bytes)
        self._max_total_bytes = int(max_total_bytes)

    def describe(self) -> str:
        return f"zip:{self._archive.name}"

    def documents(self) -> Iterator[Document]:
        if not self._archive.is_file():
            raise DocumentSourceUnavailable(f"archive not found: {self._archive.name}")
        try:
            zf = zipfile.ZipFile(self._archive)
        except zipfile.BadZipFile as exc:
            raise DocumentSourceUnavailable(
                f"corrupt archive: {self._archive.name}", reason=str(exc)) from exc
        with zf:
            members = sorted(n for n in zf.namelist()
                             if not n.endswith("/")
                             and PurePosixPath(n).suffix.lower() in self._exts)
            if not members:
                raise DocumentSourceUnavailable(
                    f"no {'/'.join(self._exts)} members in {self._archive.name}")
            if len(members) > self._max_members:
                raise ResourceLimitExceeded(
                    f"archive holds more than {self._max_members} matching members",
                    limit=self._max_members, members=len(members))
            budget = self._max_total_bytes
            for member in members:
                raw = self._read_capped(zf, member, budget)
                budget -= len(raw)
                text = raw.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                doc_id = PurePosixPath(member).stem
                yield Document(doc_id=doc_id, title=title_of(text, doc_id), text=text,
                               origin=f"zip://{self._archive.name}!{member}")

    def _read_capped(self, zf: zipfile.ZipFile, member: str, budget: int) -> bytes:
        """Stream one member, refusing to hold more than the per-member or total cap."""
        limit = min(self._max_member_bytes, budget)
        out = bytearray()
        try:
            with zf.open(member) as handle:
                while len(out) <= limit:
                    block = handle.read(_CHUNK)
                    if not block:
                        return bytes(out)
                    out.extend(block)
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            raise DocumentSourceUnavailable(
                f"unreadable member in {self._archive.name}", member=member) from exc
        raise ResourceLimitExceeded(
            "archive member expands past the decompression limit",
            member=member, limit=self._max_member_bytes)
