"""Domain-level errors.

Adapters translate their own failures (missing folder, unreachable API, locked database,
an exploding vectoriser) into these so that use-cases never need to know which technology
failed. Nothing outside this hierarchy is allowed to reach a caller as a *client* error.

Every error carries a stable ``code``. The code is part of the API: clients switch on it,
whereas the message is prose and may be reworded. ``details`` holds machine-readable context
and must never contain a credential, an absolute host path, or a driver traceback.
"""
from __future__ import annotations

from typing import Any


class KbError(Exception):
    """Base class for every error the knowledge-base domain raises."""

    code = "kb_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details)

    def as_dict(self) -> dict[str, Any]:
        """The wire shape of an error, minus the request id the transport adds."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(KbError):
    """A request was structurally wrong: missing field, wrong type, out of range."""

    code = "invalid_request"


class ResourceLimitExceeded(KbError):
    """A request would consume more of some bounded resource than the service allows."""

    code = "resource_limit_exceeded"


class PathNotAllowed(KbError):
    """A client-supplied path resolves outside every configured corpus root."""

    code = "path_not_allowed"


class EmptyCorpusError(KbError):
    """A DocumentSourcePort yielded no usable documents."""

    code = "empty_corpus"


class IndexNotBuiltError(KbError):
    """An IndexPort was queried before ``build`` was called."""

    code = "corpus_not_ingested"


class IndexBuildFailed(KbError):
    """An IndexPort could not build over the passages it was given.

    This is the error adapters raise instead of letting a driver exception escape: a corpus
    of nothing but stopwords is a client problem, but ``empty vocabulary; perhaps the
    documents only contain stop words`` is scikit-learn's problem and must not reach the wire.
    """

    code = "index_build_failed"


class DocumentSourceUnavailable(KbError):
    """The backing store of a DocumentSourcePort cannot be read."""

    code = "source_unavailable"


class CorpusStoreUnavailable(KbError):
    """The corpus store cannot be opened, migrated, or read."""

    code = "corpus_store_unavailable"


class AdapterUnavailable(KbError):
    """An adapter cannot be constructed (missing credentials, missing driver, ...).

    The composition root catches this and substitutes the configured fallback adapter,
    which is why the service stays fully operational offline.
    """

    code = "adapter_unavailable"


class UnknownStrategyError(KbError):
    """A comparison or configuration request named an adapter that is not registered."""

    code = "unknown_adapter"
