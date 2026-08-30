"""Domain-level errors.

Adapters translate their own failures (missing folder, unreachable API, locked database)
into these so that use-cases never need to know which technology failed.
"""
from __future__ import annotations


class KbError(Exception):
    """Base class for every error the knowledge-base domain raises."""


class EmptyCorpusError(KbError):
    """A DocumentSourcePort yielded no usable documents."""


class IndexNotBuiltError(KbError):
    """An IndexPort was queried before ``build`` was called."""


class DocumentSourceUnavailable(KbError):
    """The backing store of a DocumentSourcePort cannot be read."""


class AdapterUnavailable(KbError):
    """An adapter cannot be constructed (missing credentials, missing driver, ...).

    The composition root catches this and substitutes the configured fallback adapter,
    which is why the service stays fully operational offline.
    """


class UnknownStrategyError(KbError):
    """A comparison or configuration request named an adapter that is not registered."""
