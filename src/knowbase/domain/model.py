"""Domain entities and value objects for the knowledge-base bounded context.

Everything here is a frozen dataclass: the domain core is a value-oriented model so that
use-cases can be reasoned about (and replayed in tests) without worrying about aliasing.
No module in this package may import an adapter, a driver, or an IO library -- that rule is
enforced by ``tests/test_domain_purity.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """A whole source document as handed over by a DocumentSourcePort adapter."""

    doc_id: str
    title: str
    text: str
    origin: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("Document.doc_id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"Document {self.doc_id!r} has no text")


@dataclass(frozen=True, slots=True)
class Passage:
    """A retrievable slice of a document.

    Retrieval and citation both operate on passages rather than documents so that an answer
    can point at the exact window of text that supports it.
    """

    doc_id: str
    ordinal: int
    text: str
    title: str = ""
    origin: str = ""

    @property
    def passage_id(self) -> str:
        return f"{self.doc_id}#{self.ordinal}"


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One scored passage returned by an IndexPort, with its 1-based rank."""

    passage: Passage
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class Citation:
    """A grounded reference: marker ``n`` in the answer text points at this passage."""

    marker: int
    passage_id: str
    title: str
    origin: str = ""

    def render(self) -> str:
        return f"[{self.marker}] {self.title} ({self.passage_id})"


@dataclass(frozen=True, slots=True)
class Generated:
    """Raw output of a GeneratorPort: answer prose plus the 1-based markers it relied on."""

    text: str
    cited: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Answer:
    """The result of the AskQuestion use-case."""

    question: str
    text: str
    citations: tuple[Citation, ...]
    hits: tuple[RetrievalHit, ...]
    generator: str
    index: str
    cached: bool
    latency_ms: float

    def render(self) -> str:
        body = "\n".join(f"  {c.render()}" for c in self.citations) or "  (no citations)"
        tag = f"{self.generator}{', cached' if self.cached else ''}"
        return f"Q: {self.question}\nA ({tag}): {self.text}\n\nSources:\n{body}"


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Outcome of IngestCorpus -- what went in, and into which adapters."""

    source: str
    index: str
    documents: int
    passages: int
    started_at: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A labelled retrieval question: which document is expected to answer it."""

    question: str
    expected_doc_id: str


@dataclass(frozen=True, slots=True)
class EvalRow:
    """Per-question evaluation detail; ``rank`` is 0 when the expected doc was not retrieved."""

    question: str
    expected_doc_id: str
    rank: int
    top_doc_id: str


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Retrieval quality of one IndexPort over one labelled suite."""

    index: str
    suite: str
    k: int
    n: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    rows: tuple[EvalRow, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (f"{self.index:<12} suite={self.suite:<5} n={self.n:<3} "
                f"hit@1={self.hit_at_1:.3f} hit@{self.k}={self.hit_at_k:.3f} mrr={self.mrr:.3f}")


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """A/B(/C) result: the same eval battery run through several IndexPort adapters."""

    suite: str
    k: int
    reports: tuple[EvalReport, ...]
    winner: str
    margin: float

    def table(self) -> str:
        lines = [r.summary() for r in self.reports]
        lines.append(f"winner={self.winner} margin(mrr)={self.margin:+.3f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Hit/miss bookkeeping every CachePort adapter must maintain."""

    name: str
    entries: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total
