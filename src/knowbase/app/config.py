"""Declarative wiring configuration.

Every adapter choice the service makes is a string in this object. Swapping the retrieval
strategy or the answer generator is therefore a configuration change, never a code change --
which is the concrete payoff of the ports-and-adapters split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Names of the adapters to bind to each port, plus retrieval knobs."""

    index: str = "hybrid_rrf"
    generator: str = "claude"          # degrades to 'extractive' when no credentials exist
    cache: str = "memory"
    clock: str = "system"
    source: str = "filesystem"
    source_path: str = ""
    sqlite_path: str = ":memory:"
    compare_indexes: tuple[str, ...] = ("hybrid_rrf", "bm25", "tfidf")
    top_k: int = 4
    eval_k: int = 5
    cache_ttl_s: float | None = None
    target_words: int = 120
    overlap: int = 25
    seed: int = 42
    extra: Mapping[str, Any] = field(default_factory=dict)

    def with_(self, **changes: Any) -> "AppConfig":
        """Return a copy with the given fields replaced (config objects stay immutable)."""
        if "compare_indexes" in changes and changes["compare_indexes"] is not None:
            changes["compare_indexes"] = tuple(changes["compare_indexes"])
        return replace(self, **changes)

    def index_names(self) -> tuple[str, ...]:
        """Active index first, then every strategy the A/B endpoint should be able to reach."""
        names = [self.index] + [n for n in self.compare_indexes if n != self.index]
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "generator": self.generator, "cache": self.cache,
            "clock": self.clock, "source": self.source, "source_path": self.source_path,
            "sqlite_path": self.sqlite_path, "compare_indexes": list(self.compare_indexes),
            "top_k": self.top_k, "eval_k": self.eval_k, "cache_ttl_s": self.cache_ttl_s,
            "target_words": self.target_words, "overlap": self.overlap, "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        known = {f for f in cls.__slots__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "compare_indexes" in kwargs and kwargs["compare_indexes"] is not None:
            kwargs["compare_indexes"] = tuple(kwargs["compare_indexes"])
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AppConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
