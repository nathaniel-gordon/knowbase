"""The domain core: entities, pure text services, and use-cases.

Import rule for this package (asserted by ``tests/test_domain_purity.py``): only the standard
library's pure-computation modules and ``knowbase.ports`` may appear here. No numpy, no sklearn, no
sqlite3, no pathlib, no http, no clock, no randomness.
"""
from __future__ import annotations

from .errors import (
    AdapterUnavailable,
    DocumentSourceUnavailable,
    EmptyCorpusError,
    IndexNotBuiltError,
    KbError,
    UnknownStrategyError,
)
from .model import (
    Answer,
    CacheStats,
    Citation,
    ComparisonReport,
    Document,
    EvalCase,
    EvalReport,
    EvalRow,
    Generated,
    IngestReport,
    Passage,
    RetrievalHit,
)
from .usecases import ABSTENTION, AskQuestion, CompareStrategies, EvaluateRetrieval, IngestCorpus

__all__ = [
    "ABSTENTION", "AdapterUnavailable", "Answer", "AskQuestion", "CacheStats", "Citation",
    "CompareStrategies", "ComparisonReport", "Document", "DocumentSourceUnavailable",
    "EmptyCorpusError", "EvalCase", "EvalReport", "EvalRow", "EvaluateRetrieval", "Generated",
    "IndexNotBuiltError", "IngestCorpus", "IngestReport", "KbError", "Passage", "RetrievalHit",
    "UnknownStrategyError",
]
