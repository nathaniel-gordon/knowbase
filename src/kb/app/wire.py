"""Transport mapping: domain objects <-> JSON payloads.

Serialisation lives here, not on the entities, so the domain never grows a dependency on the
wire format. The client SDK re-hydrates the same shapes into typed views, and both directions
are exercised by the HTTP integration tests.
"""
from __future__ import annotations

from typing import Any

from ..domain.model import Answer, ComparisonReport, EvalReport, IngestReport


def ingest_to_dict(report: IngestReport) -> dict[str, Any]:
    return {"source": report.source, "index": report.index, "documents": report.documents,
            "passages": report.passages, "started_at": report.started_at,
            "elapsed_ms": report.elapsed_ms}


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "answer": answer.text,
        "generator": answer.generator,
        "index": answer.index,
        "cached": answer.cached,
        "latency_ms": answer.latency_ms,
        "citations": [{"marker": c.marker, "passage_id": c.passage_id, "title": c.title,
                       "origin": c.origin} for c in answer.citations],
        "retrieved": [{"rank": h.rank, "passage_id": h.passage.passage_id,
                       "doc_id": h.passage.doc_id, "score": h.score,
                       "title": h.passage.title} for h in answer.hits],
    }


def eval_to_dict(report: EvalReport, with_rows: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": report.index, "suite": report.suite, "k": report.k, "n": report.n,
        "hit@1": report.hit_at_1, f"hit@{report.k}": report.hit_at_k, "mrr": report.mrr,
    }
    if with_rows:
        payload["rows"] = [{"question": r.question, "expected_doc_id": r.expected_doc_id,
                            "rank": r.rank, "top_doc_id": r.top_doc_id} for r in report.rows]
    return payload


def comparison_to_dict(report: ComparisonReport) -> dict[str, Any]:
    return {"suite": report.suite, "k": report.k, "winner": report.winner,
            "margin": report.margin,
            "reports": [eval_to_dict(r, with_rows=False) for r in report.reports]}
