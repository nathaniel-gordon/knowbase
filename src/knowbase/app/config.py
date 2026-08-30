"""Declarative wiring configuration.

Every adapter choice the service makes is a string in this object. Swapping the retrieval
strategy or the answer generator is therefore a configuration change, never a code change --
which is the concrete payoff of the ports-and-adapters split.

Unknown keys are rejected rather than collected: a config file with a typo in it should fail
at start-up, not silently run with a default nobody chose.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .limits import Limits


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Names of the adapters to bind to each port, plus retrieval knobs and resource caps."""

    index: str = "hybrid_rrf"
    generator: str = "claude"          # degrades to 'extractive' when no credentials exist
    cache: str = "memory"
    store: str = "memory"              # CorpusStorePort; 'sqlite' survives a restart
    clock: str = "system"
    sqlite_path: str = ":memory:"      # the service database: answer cache and corpus store
    corpus_roots: tuple[str, ...] = ()  # empty means "the working directory"
    compare_indexes: tuple[str, ...] = ("hybrid_rrf", "bm25", "tfidf")
    top_k: int = 4
    eval_k: int = 5
    cache_ttl_s: float | None = None
    target_words: int = 120
    overlap: int = 25
    limits: Limits = field(default_factory=Limits)

    def with_(self, **changes: Any) -> AppConfig:
        """Return a copy with the given fields replaced (config objects stay immutable)."""
        for name in ("compare_indexes", "corpus_roots"):
            if changes.get(name) is not None and name in changes:
                changes[name] = tuple(changes[name])
        return replace(self, **changes)

    def index_names(self) -> tuple[str, ...]:
        """Active index first, then every strategy the A/B endpoint should be able to reach."""
        names = [self.index] + [n for n in self.compare_indexes if n != self.index]
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "generator": self.generator, "cache": self.cache,
            "store": self.store, "clock": self.clock, "sqlite_path": self.sqlite_path,
            "corpus_roots": list(self.corpus_roots),
            "compare_indexes": list(self.compare_indexes),
            "top_k": self.top_k, "eval_k": self.eval_k, "cache_ttl_s": self.cache_ttl_s,
            "target_words": self.target_words, "overlap": self.overlap,
            "limits": self.limits.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AppConfig:
        known = set(cls.__slots__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown configuration key(s): {unknown}; known: {sorted(known)}")
        kwargs = dict(data)
        for name in ("compare_indexes", "corpus_roots"):
            if kwargs.get(name) is not None and name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        if isinstance(kwargs.get("limits"), Mapping):
            kwargs["limits"] = Limits.from_mapping(kwargs["limits"])
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> AppConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
