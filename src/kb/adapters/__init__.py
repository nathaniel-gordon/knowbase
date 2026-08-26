"""Driven adapters -- the outside of the hexagon.

Every module here implements exactly one port and may import ``kb.domain`` and ``kb.ports``.
Nothing in ``kb.domain`` may import anything from this package; the arrow points inward only.

The dictionaries below are the registries the composition root consults, so adding an adapter
is a one-line change in one file and requires no edit to any use-case.
"""
from __future__ import annotations

from typing import Callable

from .cache.memory import InMemoryCache
from .cache.sqlite import SqliteCache
from .clock.system import FrozenClock, SystemClock
from .generator.claude import ClaudeGenerator, anthropic_completion
from .generator.extractive import ExtractiveGenerator
from .index.bm25_only import Bm25Index
from .index.hybrid_rrf import HybridRrfIndex
from .index.tfidf_only import TfidfIndex
from .source.filesystem import FilesystemDocumentSource
from .source.memory import InMemoryDocumentSource
from .source.zip_archive import ZipDocumentSource

INDEX_ADAPTERS: dict[str, Callable[[], object]] = {
    HybridRrfIndex.name: HybridRrfIndex,
    Bm25Index.name: Bm25Index,
    TfidfIndex.name: TfidfIndex,
}

SOURCE_ADAPTERS: dict[str, type] = {
    FilesystemDocumentSource.name: FilesystemDocumentSource,
    InMemoryDocumentSource.name: InMemoryDocumentSource,
    ZipDocumentSource.name: ZipDocumentSource,
}

CACHE_ADAPTERS: dict[str, type] = {
    InMemoryCache.name: InMemoryCache,
    SqliteCache.name: SqliteCache,
}

CLOCK_ADAPTERS: dict[str, type] = {
    SystemClock.name: SystemClock,
    FrozenClock.name: FrozenClock,
}

GENERATOR_ADAPTERS: tuple[str, ...] = (ExtractiveGenerator.name, ClaudeGenerator.name)

__all__ = [
    "Bm25Index", "CACHE_ADAPTERS", "CLOCK_ADAPTERS", "ClaudeGenerator", "ExtractiveGenerator",
    "FilesystemDocumentSource", "FrozenClock", "GENERATOR_ADAPTERS", "HybridRrfIndex",
    "INDEX_ADAPTERS", "InMemoryCache", "InMemoryDocumentSource", "SOURCE_ADAPTERS",
    "SqliteCache", "SystemClock", "TfidfIndex", "ZipDocumentSource", "anthropic_completion",
]
