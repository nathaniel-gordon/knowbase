"""The composition root: the single place where concrete adapters meet the domain.

Nothing else in the application constructs an adapter. That is what keeps the use-cases
substitutable -- and what lets ``POST /config`` rebind a port at runtime without any component
inside the hexagon noticing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters import (
    CACHE_ADAPTERS,
    CLOCK_ADAPTERS,
    GENERATOR_ADAPTERS,
    INDEX_ADAPTERS,
    SOURCE_ADAPTERS,
    ClaudeGenerator,
    ExtractiveGenerator,
    FilesystemDocumentSource,
    InMemoryDocumentSource,
    SqliteCache,
    ZipDocumentSource,
    anthropic_completion,
)
from ..adapters.source.filesystem import DEFAULT_EXTS as FS_DEFAULT_EXTS
from ..adapters.source.zip_archive import DEFAULT_EXTS as ZIP_DEFAULT_EXTS
from ..domain.errors import AdapterUnavailable, IndexNotBuiltError, UnknownStrategyError
from ..domain.model import Answer, ComparisonReport, EvalCase, EvalReport, IngestReport
from ..domain.usecases import AskQuestion, CompareStrategies, EvaluateRetrieval, IngestCorpus
from ..ports import CachePort, ClockPort, DocumentSourcePort, GeneratorPort, IndexPort
from .config import AppConfig
from .corpus import SUITES
from .limits import Limits
from .paths import CorpusRoots
from .validation import CompareRequest, EvalRequest, IngestRequest


def build_clock(name: str) -> ClockPort:
    try:
        return CLOCK_ADAPTERS[name]()
    except KeyError:
        raise UnknownStrategyError(
            f"unknown clock adapter {name!r}; available: {sorted(CLOCK_ADAPTERS)}",
            requested=name, available=sorted(CLOCK_ADAPTERS)) from None


def build_cache(name: str, clock: ClockPort, sqlite_path: str = ":memory:") -> CachePort:
    if name not in CACHE_ADAPTERS:
        raise UnknownStrategyError(
            f"unknown cache adapter {name!r}; available: {sorted(CACHE_ADAPTERS)}",
            requested=name, available=sorted(CACHE_ADAPTERS))
    cls = CACHE_ADAPTERS[name]
    return cls(clock, sqlite_path) if cls is SqliteCache else cls(clock)


def build_index(name: str) -> IndexPort:
    try:
        return INDEX_ADAPTERS[name]()  # type: ignore[return-value]
    except KeyError:
        raise UnknownStrategyError(
            f"unknown index adapter {name!r}; available: {sorted(INDEX_ADAPTERS)}",
            requested=name, available=sorted(INDEX_ADAPTERS)) from None


def build_generator(name: str) -> tuple[GeneratorPort, str]:
    """Return the generator and a one-line note explaining what was actually bound.

    ``claude`` is a *request*, not a guarantee: with no API key the root binds the extractive
    adapter instead and records why. The service is therefore always able to answer.
    """
    if name == ExtractiveGenerator.name:
        return ExtractiveGenerator(), "extractive (deterministic, offline)"
    if name == ClaudeGenerator.name:
        try:
            generator = ClaudeGenerator(anthropic_completion())
        except AdapterUnavailable as exc:
            return ExtractiveGenerator(), f"extractive (claude unavailable: {exc})"
        return generator, f"claude ({generator.model})"
    raise UnknownStrategyError(
        f"unknown generator adapter {name!r}; available: {sorted(GENERATOR_ADAPTERS)}",
            requested=name, available=sorted(GENERATOR_ADAPTERS))


def build_source(request: IngestRequest, roots: CorpusRoots, limits: Limits) -> DocumentSourcePort:
    """Build a document source from an already-validated ingest request.

    Accepting the source per request (rather than once at start-up) is what lets a single
    running service ingest a folder, then a zip export, then inline documents. It is also the
    reason ``roots`` exists: the path comes from the caller, so it is resolved and required to
    sit inside the allow-list before any adapter is handed it.
    """
    if request.source not in SOURCE_ADAPTERS:
        raise UnknownStrategyError(
            f"unknown source adapter {request.source!r}; available: {sorted(SOURCE_ADAPTERS)}",
            requested=request.source, available=sorted(SOURCE_ADAPTERS))

    if request.source == InMemoryDocumentSource.name:
        documents = dict(request.documents or ())
        return InMemoryDocumentSource(documents, label=request.label)

    path = roots.resolve(request.path or "")
    exts = request.exts
    if request.source == ZipDocumentSource.name:
        return ZipDocumentSource(path, exts or ZIP_DEFAULT_EXTS,
                                 max_members=limits.max_zip_members,
                                 max_member_bytes=limits.max_zip_member_bytes,
                                 max_total_bytes=limits.max_zip_total_bytes)
    return FilesystemDocumentSource(path, exts or FS_DEFAULT_EXTS,
                                    max_files=limits.max_files,
                                    max_file_bytes=limits.max_file_bytes,
                                    max_scanned_entries=limits.max_scanned_entries)


class Container:
    """Holds the bound adapters and hands out freshly wired use-cases.

    One passage list feeds *every* registered index adapter, so an A/B comparison differs only
    in ranking strategy -- chunking is held constant by construction.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.limits = config.limits
        self.roots = CorpusRoots(config.corpus_roots)
        self.clock = build_clock(config.clock)
        self.cache = build_cache(config.cache, self.clock, config.sqlite_path)
        self.generator, self.generator_note = build_generator(config.generator)
        self.indexes: dict[str, IndexPort] = {n: build_index(n) for n in config.index_names()}
        self.source: DocumentSourcePort | None = None
        self.last_ingest: IngestReport | None = None
        self.passages: list[Any] = []
        self.started_at = self.clock.timestamp()

    # ---------------------------------------------------------------- wiring

    @property
    def index(self) -> IndexPort:
        return self.indexes[self.config.index]

    def wiring(self) -> dict[str, Any]:
        """What is bound to each port right now -- surfaced by ``GET /health``."""
        return {
            "index": self.config.index,
            "index_pool": sorted(self.indexes),
            "generator": self.generator.name,
            "generator_detail": self.generator_note,
            "cache": self.cache.name,
            "clock": self.clock.name,
            "source": self.source.describe() if self.source else None,
            "corpus_roots": len(self.roots.roots),
        }

    def rebind(self, index: str | None = None, generator: str | None = None,
               cache: str | None = None) -> dict[str, Any]:
        """Rebind ports at runtime. Indexes are pre-built, so an index swap is instantaneous."""
        changes: dict[str, Any] = {}
        if index is not None and index != self.config.index:
            if index not in INDEX_ADAPTERS:
                raise UnknownStrategyError(
                    f"unknown index adapter {index!r}; available: {sorted(INDEX_ADAPTERS)}",
                    requested=index, available=sorted(INDEX_ADAPTERS))
            if index not in self.indexes:
                fresh = build_index(index)
                if self.last_ingest is not None and self.passages:
                    fresh.build(self.passages)
                self.indexes[index] = fresh
            changes["index"] = {"from": self.config.index, "to": index}
            self.config = self.config.with_(index=index)
        if generator is not None and generator != self.generator.name:
            self.generator, self.generator_note = build_generator(generator)
            changes["generator"] = {"to": self.generator.name, "detail": self.generator_note}
            self.config = self.config.with_(generator=generator)
        if cache is not None and cache != self.cache.name:
            self._close_cache()
            self.cache = build_cache(cache, self.clock, self.config.sqlite_path)
            changes["cache"] = {"to": self.cache.name}
            self.config = self.config.with_(cache=cache)
        return changes

    # ------------------------------------------------------------- use-cases

    def ingest(self, request: IngestRequest) -> IngestReport:
        source = build_source(request, self.roots, self.limits)
        use_case = IngestCorpus(source=source,
                                indexes=[self.indexes[n] for n in sorted(self.indexes)],
                                clock=self.clock,
                                target_words=self.config.target_words,
                                overlap=self.config.overlap)
        report = use_case.execute()
        self.source = source
        self.passages = list(use_case.passages)
        self.last_ingest = report
        self.cache.clear()  # answers cached against a previous corpus are no longer grounded
        return report

    def ask(self, question: str, k: int | None = None) -> Answer:
        self._require_index()
        return AskQuestion(index=self.index, generator=self.generator, cache=self.cache,
                           clock=self.clock, default_k=self.config.top_k,
                           ttl_s=self.config.cache_ttl_s).execute(question, k=k)

    def evaluate(self, request: EvalRequest) -> EvalReport:
        self._require_index()
        cases = request.cases if request.cases is not None else self._suite(request.suite)
        suite = request.suite if request.cases is None else "custom"
        return EvaluateRetrieval(self.index, default_k=self.config.eval_k).execute(
            cases, k=request.k, suite=suite)

    def compare(self, request: CompareRequest) -> ComparisonReport:
        self._require_index()
        cases = request.cases if request.cases is not None else self._suite(request.suite)
        suite = request.suite if request.cases is None else "custom"
        return CompareStrategies(self.indexes).execute(
            cases, names=request.indexes, k=request.k or self.config.eval_k, suite=suite)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _suite(suite: str) -> Sequence[EvalCase]:
        try:
            return SUITES[suite]
        except KeyError:
            raise UnknownStrategyError(
                f"unknown eval suite {suite!r}; available: {sorted(SUITES)}",
                requested=suite, available=sorted(SUITES)) from None

    def _require_index(self) -> None:
        if self.index.size() == 0:
            raise IndexNotBuiltError("no corpus ingested yet; POST /ingest first")

    def _close_cache(self) -> None:
        closer = getattr(self.cache, "close", None)
        if callable(closer):
            closer()

    def close(self) -> None:
        """Release adapter resources (sqlite handles) -- important on Windows."""
        self._close_cache()


def build_container(config: AppConfig | Mapping[str, Any] | None = None) -> Container:
    """Public entry point of the composition root."""
    if config is None:
        config = AppConfig()
    elif isinstance(config, Mapping):
        config = AppConfig.from_mapping(config)
    return Container(config)
