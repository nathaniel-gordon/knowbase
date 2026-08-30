"""Application use-cases: the inside of the hexagon.

Each class receives its collaborators as *ports* through the constructor and knows nothing
about filesystems, sqlite, sklearn, HTTP or the Anthropic API. That is what makes the A/B
endpoint possible: ``CompareStrategies`` runs the identical evaluation logic through any
number of index adapters because it only ever sees :class:`IndexPort`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

from ..ports import (
    CachePort,
    ClockPort,
    CorpusStorePort,
    DocumentSourcePort,
    GeneratorPort,
    IndexPort,
)
from .errors import EmptyCorpusError, UnknownStrategyError
from .model import (
    Answer,
    Citation,
    ComparisonReport,
    CorpusSnapshot,
    Document,
    EvalCase,
    EvalReport,
    EvalRow,
    Generated,
    IngestReport,
    Passage,
)
from .text import DEFAULT_OVERLAP, DEFAULT_TARGET_WORDS, chunk_document, corpus_fingerprint

ABSTENTION = "No relevant information found in the knowledge base."


class IngestCorpus:
    """Read documents from a source port, chunk them, and build one or more indexes.

    Several indexes may be fed from a single pass so that an A/B comparison always evaluates
    strategies over *identical* passages -- otherwise the comparison would confound chunking
    differences with ranking differences.
    """

    def __init__(self,
                 source: DocumentSourcePort,
                 indexes: Sequence[IndexPort],
                 clock: ClockPort,
                 store: CorpusStorePort | None = None,
                 target_words: int = DEFAULT_TARGET_WORDS,
                 overlap: int = DEFAULT_OVERLAP) -> None:
        if not indexes:
            raise ValueError("IngestCorpus needs at least one index port")
        self._source = source
        self._indexes = list(indexes)
        self._clock = clock
        self._store = store
        self._target_words = target_words
        self._overlap = overlap
        self.passages: list[Passage] = []
        self.snapshot: CorpusSnapshot | None = None

    def execute(self) -> IngestReport:
        started_at = self._clock.timestamp()
        t0 = self._clock.now()

        documents: list[Document] = list(self._source.documents())
        if not documents:
            raise EmptyCorpusError(f"source {self._source.name!r} yielded no documents")

        passages: list[Passage] = []
        for document in documents:
            passages.extend(chunk_document(document, self._target_words, self._overlap))
        if not passages:
            raise EmptyCorpusError("chunking produced no passages")

        for index in self._indexes:
            index.build(passages)
        self.passages = passages

        snapshot = CorpusSnapshot(fingerprint=corpus_fingerprint(passages),
                                  source=self._source.name,
                                  documents=len(documents),
                                  started_at=started_at,
                                  passages=tuple(passages))
        # Persisted only after every index accepted the passages: a store holding a corpus
        # that cannot be indexed would make the next start-up fail instead of the ingest.
        if self._store is not None:
            self._store.save(snapshot)
        self.snapshot = snapshot

        return IngestReport(source=self._source.name,
                            index="+".join(i.name for i in self._indexes),
                            documents=len(documents),
                            passages=len(passages),
                            started_at=started_at,
                            elapsed_ms=round((self._clock.now() - t0) * 1000.0, 3),
                            fingerprint=snapshot.fingerprint)


class AskQuestion:
    """Retrieve passages, generate a grounded answer, and attach only earned citations.

    A citation is emitted **only** for a passage the generator declared it used. Reporting the
    whole retrieved set as "sources" would overstate grounding; the port contract makes the
    honest version cheap because every generator must return the markers it relied on.
    """

    def __init__(self,
                 index: IndexPort,
                 generator: GeneratorPort,
                 cache: CachePort,
                 clock: ClockPort,
                 default_k: int = 4,
                 ttl_s: float | None = None,
                 corpus_fingerprint: str = "") -> None:
        self._index = index
        self._generator = generator
        self._cache = cache
        self._clock = clock
        self._default_k = default_k
        self._ttl_s = ttl_s
        self._fingerprint = corpus_fingerprint

    def cache_key(self, question: str, k: int) -> str:
        """Identity of an answer: corpus, index, generator, k, and the normalised question.

        The corpus fingerprint is the load-bearing part. Without it, a durable cache written
        against one corpus is readable while a completely different corpus is loaded, and the
        reply carries the old prose with no citations at all -- the answer is not grounded in
        anything the service currently holds.
        """
        material = (f"{self._fingerprint}|{self._index.name}|{self._generator.name}|{k}|"
                    f"{' '.join(question.lower().split())}")
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def execute(self, question: str, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")
        k = self._default_k if k is None else k
        if k <= 0:
            raise ValueError("k must be positive")

        t0 = self._clock.now()
        hits = self._index.search(question, k=k)
        passages = [h.passage for h in hits]

        key = self.cache_key(question, k)
        cached = self._cache.get(key)
        if cached is not None and self._still_grounded(cached, passages):
            generated, was_cached = cached, True
        else:
            generated = self._generator.generate(question, passages)
            self._cache.set(key, generated, ttl_s=self._ttl_s)
            was_cached = False

        return Answer(question=question,
                      text=generated.text,
                      citations=self._citations(generated, passages),
                      hits=tuple(hits),
                      generator=self._generator.name,
                      index=self._index.name,
                      cached=was_cached,
                      latency_ms=round((self._clock.now() - t0) * 1000.0, 3))

    @staticmethod
    def _still_grounded(cached: Generated, passages: Sequence[Passage]) -> bool:
        """Whether a cached answer's markers all resolve against today's retrieval.

        They used to be resolved silently: any marker past the end of the current hit list was
        dropped, and the cached prose was served with its citations quietly removed. Serving
        an answer with fewer sources than it was written with is worse than recomputing it,
        so a cached entry that no longer resolves is treated as a miss.
        """
        return all(1 <= marker <= len(passages) for marker in cached.cited)

    @staticmethod
    def _citations(generated: Generated, passages: Sequence[Passage]) -> tuple[Citation, ...]:
        """Map the generator's markers onto passages, dropping anything out of range."""
        out: list[Citation] = []
        for marker in generated.cited:
            if 1 <= marker <= len(passages):
                p = passages[marker - 1]
                out.append(Citation(marker=marker, passage_id=p.passage_id,
                                    title=p.title or p.doc_id, origin=p.origin))
        return tuple(out)


class EvaluateRetrieval:
    """Score an index against labelled (question -> expected document) pairs.

    Metrics are document-level: passages of the same document are collapsed in rank order, so
    an index that returns three windows of the right document still scores one hit at rank 1.
    """

    def __init__(self, index: IndexPort, default_k: int = 5) -> None:
        self._index = index
        self._default_k = default_k

    def execute(self, cases: Sequence[EvalCase], k: int | None = None,
                suite: str = "custom") -> EvalReport:
        if not cases:
            raise ValueError("evaluation needs at least one case")
        k = self._default_k if k is None else k

        hit1 = hitk = 0
        mrr = 0.0
        rows: list[EvalRow] = []
        for case in cases:
            doc_order = self._document_order(case.question, k)
            found = case.expected_doc_id in doc_order
            rank = doc_order.index(case.expected_doc_id) + 1 if found else 0
            if rank:
                hitk += 1
                mrr += 1.0 / rank
                hit1 += int(rank == 1)
            rows.append(EvalRow(question=case.question,
                                expected_doc_id=case.expected_doc_id,
                                rank=rank,
                                top_doc_id=doc_order[0] if doc_order else ""))

        n = len(cases)
        return EvalReport(index=self._index.name, suite=suite, k=k, n=n,
                          hit_at_1=round(hit1 / n, 4),
                          hit_at_k=round(hitk / n, 4),
                          mrr=round(mrr / n, 4),
                          rows=tuple(rows))

    def _document_order(self, question: str, k: int) -> list[str]:
        order: list[str] = []
        for hit in self._index.search(question, k=k):
            if hit.passage.doc_id not in order:
                order.append(hit.passage.doc_id)
        return order


class CompareStrategies:
    """Run one eval battery through several index adapters and declare a winner.

    This use-case is only expressible because retrieval is a port: it takes a mapping of
    adapter name -> :class:`IndexPort` and is completely blind to how any of them rank.
    """

    def __init__(self, indexes: Mapping[str, IndexPort]) -> None:
        if len(indexes) < 2:
            raise ValueError("comparison needs at least two index adapters")
        self._indexes = dict(indexes)

    def execute(self, cases: Sequence[EvalCase], names: Iterable[str] | None = None,
                k: int = 5, suite: str = "custom") -> ComparisonReport:
        selected = list(names) if names is not None else list(self._indexes)
        unknown = [n for n in selected if n not in self._indexes]
        if unknown:
            raise UnknownStrategyError(f"unknown index adapter(s): {sorted(unknown)}; "
                                       f"available: {sorted(self._indexes)}")
        if len(selected) < 2:
            raise ValueError("comparison needs at least two named adapters")

        reports = [EvaluateRetrieval(self._indexes[name]).execute(cases, k=k, suite=suite)
                   for name in selected]
        # Deterministic ordering: MRR first, hit@1 as tie-break, then name so ties never flap.
        ranked = sorted(reports, key=lambda r: (-r.mrr, -r.hit_at_1, r.index))
        margin = round(ranked[0].mrr - ranked[1].mrr, 4)
        winner = ranked[0].index if margin > 0 else "tie"
        return ComparisonReport(suite=suite, k=k, reports=tuple(ranked),
                                winner=winner, margin=margin)
