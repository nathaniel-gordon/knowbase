"""The corpus-root allow-list: the only place a client-supplied path becomes a real path.

``POST /ingest`` takes a path from whoever called it. Without containment that endpoint is an
arbitrary-file-read primitive, because whatever it ingests can then be quoted back by
``/ask``. So every path crosses this module first: it is expanded, resolved (which collapses
``..``, follows symlinks, and normalises drive letters and UNC prefixes on Windows), and then
required to sit inside one of the configured roots.

Resolution happens *before* the containment check, never after, and the check compares
resolved paths only -- a string prefix test would accept ``/corpus-evil`` for root
``/corpus``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..domain.errors import PathNotAllowed, ValidationError


class CorpusRoots:
    """An allow-list of directories that ``/ingest`` may read from.

    An empty configuration means "the working directory", which is the conservative reading
    of "no roots were configured": the service can still ingest the corpus a developer put
    next to it, and nothing else.
    """

    def __init__(self, roots: Iterable[str | Path] | None = None) -> None:
        raw = [Path(r) for r in (roots or ())] or [Path.cwd()]
        resolved: list[Path] = []
        for root in raw:
            try:
                resolved.append(root.expanduser().resolve())
            except (OSError, RuntimeError) as exc:  # unresolvable root is a config error
                raise ValueError(f"corpus root cannot be resolved: {root}") from exc
        self._roots: tuple[Path, ...] = tuple(sorted(set(resolved)))

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def describe(self) -> list[str]:
        """Roots as strings -- shown to operators on ``/health``, not to anonymous callers."""
        return [str(r) for r in self._roots]

    def contains(self, candidate: Path) -> bool:
        """Whether an already-resolved path sits inside a root."""
        return any(candidate == root or root in candidate.parents for root in self._roots)

    def resolve(self, candidate: str | Path) -> Path:
        """Resolve ``candidate`` and return it, or raise :class:`PathNotAllowed`.

        The rejection message echoes what the caller sent and never the resolved path or the
        configured roots, so a probe cannot map the host filesystem one 400 at a time.
        """
        text = str(candidate)
        if not text.strip():
            raise ValidationError("path must be a non-empty string", field="path")
        if "\x00" in text:
            raise ValidationError("path contains a null byte", field="path")
        try:
            resolved = Path(text).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise PathNotAllowed("path cannot be resolved", path=text) from exc
        if not self.contains(resolved):
            raise PathNotAllowed("path is outside the configured corpus roots", path=text)
        return resolved
