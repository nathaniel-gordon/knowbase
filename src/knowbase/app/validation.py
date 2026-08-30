"""Request validation: raw JSON in, typed request objects out.

The HTTP handler used to poke at dicts with ``.get()`` and coerce with bare ``int()``, so
``{"k": []}`` became a ``TypeError`` and the client got a 500 carrying a Python exception
message. Everything a client can send is checked here instead, once, against an explicit type
and range, and every rejection is a :class:`ValidationError` with a stable code and the name
of the offending field.

This module knows nothing about HTTP. It maps JSON-shaped data onto domain values, which is
also why the ``/eval`` endpoint can now accept user-supplied labelled cases: parsing them is
the same job as parsing ``k``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..domain.errors import ValidationError
from ..domain.model import EvalCase
from .limits import Limits

_MISSING = object()


# --------------------------------------------------------------------- scalars

def _field(body: Mapping[str, Any], name: str, default: Any = _MISSING) -> Any:
    if name in body and body[name] is not None:
        return body[name]
    if default is _MISSING:
        raise ValidationError(f"'{name}' is required", field=name)
    return default


def as_str(value: Any, field: str, *, max_chars: int, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"'{field}' must be a string", field=field,
                              got=type(value).__name__)
    if not allow_blank and not value.strip():
        raise ValidationError(f"'{field}' must not be blank", field=field)
    if len(value) > max_chars:
        raise ValidationError(f"'{field}' exceeds {max_chars} characters",
                              field=field, limit=max_chars, got=len(value))
    return value


def as_int(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    # bool is an int subclass and True would silently become k=1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"'{field}' must be an integer", field=field,
                              got=type(value).__name__)
    if not minimum <= value <= maximum:
        raise ValidationError(f"'{field}' must be between {minimum} and {maximum}",
                              field=field, minimum=minimum, maximum=maximum, got=value)
    return value


def as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"'{field}' must be a boolean", field=field,
                              got=type(value).__name__)
    return value


def as_str_list(value: Any, field: str, *, max_items: int, max_chars: int) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError(f"'{field}' must be a list of strings", field=field,
                              got=type(value).__name__)
    if len(value) > max_items:
        raise ValidationError(f"'{field}' holds more than {max_items} items",
                              field=field, limit=max_items, got=len(value))
    out: list[str] = []
    for position, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(f"'{field}[{position}]' must be a non-empty string",
                                  field=field, position=position,
                                  got=type(item).__name__)
        if len(item) > max_chars:
            raise ValidationError(f"'{field}[{position}]' exceeds {max_chars} characters",
                                  field=field, position=position, limit=max_chars)
        out.append(item)
    return tuple(out)


# -------------------------------------------------------------- request objects

@dataclass(frozen=True, slots=True)
class Page:
    """Slice of a list response. ``limit`` is always bounded by ``Limits.max_page_size``."""

    limit: int
    offset: int

    def apply(self, rows: Sequence[Any]) -> tuple[list[Any], dict[str, int]]:
        window = list(rows[self.offset:self.offset + self.limit])
        return window, {"limit": self.limit, "offset": self.offset,
                        "total": len(rows), "returned": len(window)}


@dataclass(frozen=True, slots=True)
class IngestRequest:
    source: str
    path: str | None = None
    documents: tuple[tuple[str, str], ...] | None = None
    label: str = "inline"
    exts: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class AskRequest:
    question: str
    k: int | None = None


@dataclass(frozen=True, slots=True)
class EvalRequest:
    suite: str
    cases: tuple[EvalCase, ...] | None = None
    k: int | None = None
    rows: bool = True
    page: Page = Page(limit=50, offset=0)


@dataclass(frozen=True, slots=True)
class CompareRequest:
    suite: str
    indexes: tuple[str, ...] | None = None
    cases: tuple[EvalCase, ...] | None = None
    k: int | None = None


@dataclass(frozen=True, slots=True)
class ConfigRequest:
    index: str | None = None
    generator: str | None = None
    cache: str | None = None


# ------------------------------------------------------------------- validators

class RequestValidator:
    """Turns request bodies into the typed objects above, under one set of limits."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits

    def _reject_unknown(self, body: Mapping[str, Any], allowed: Sequence[str]) -> None:
        unknown = sorted(set(body) - set(allowed))
        if unknown:
            raise ValidationError(f"unknown field(s): {unknown}", fields=unknown,
                                  allowed=sorted(allowed))

    def page(self, body: Mapping[str, Any]) -> Page:
        limit = as_int(_field(body, "limit", self.limits.default_page_size), "limit",
                       minimum=1, maximum=self.limits.max_page_size)
        offset = as_int(_field(body, "offset", 0), "offset", minimum=0, maximum=1_000_000)
        return Page(limit=limit, offset=offset)

    def ingest(self, body: Mapping[str, Any]) -> IngestRequest:
        self._reject_unknown(body, ("source", "path", "documents", "label", "exts"))
        source = as_str(_field(body, "source", "filesystem"), "source", max_chars=64)
        exts = None
        if body.get("exts") is not None:
            exts = as_str_list(body["exts"], "exts", max_items=32, max_chars=16)
        if source == "memory":
            return IngestRequest(source=source, documents=self._documents(body),
                                 label=as_str(_field(body, "label", "inline"), "label",
                                              max_chars=120),
                                 exts=exts)
        path = as_str(_field(body, "path"), "path", max_chars=4096)
        return IngestRequest(source=source, path=path, exts=exts)

    def _documents(self, body: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        documents = _field(body, "documents")
        if not isinstance(documents, Mapping):
            raise ValidationError("'documents' must be an object of {doc_id: text}",
                                  field="documents", got=type(documents).__name__)
        if not documents:
            raise ValidationError("'documents' must not be empty", field="documents")
        if len(documents) > self.limits.max_documents:
            raise ValidationError(
                f"'documents' holds more than {self.limits.max_documents} entries",
                field="documents", limit=self.limits.max_documents, got=len(documents))
        total = 0
        out: list[tuple[str, str]] = []
        for doc_id, text in documents.items():
            key = as_str(doc_id, "documents key", max_chars=self.limits.max_doc_id_chars)
            value = as_str(text, f"documents[{key}]", max_chars=self.limits.max_document_bytes,
                           allow_blank=True)
            total += len(value.encode("utf-8"))
            if total > self.limits.max_inline_total_bytes:
                raise ValidationError(
                    f"inline documents exceed {self.limits.max_inline_total_bytes} bytes",
                    field="documents", limit=self.limits.max_inline_total_bytes)
            out.append((key, value))
        return tuple(out)

    def ask(self, body: Mapping[str, Any]) -> AskRequest:
        self._reject_unknown(body, ("question", "k"))
        question = as_str(_field(body, "question"), "question",
                          max_chars=self.limits.max_question_chars)
        k = None
        if body.get("k") is not None:
            k = as_int(body["k"], "k", minimum=1, maximum=self.limits.max_k)
        return AskRequest(question=question, k=k)

    def evaluate(self, body: Mapping[str, Any]) -> EvalRequest:
        self._reject_unknown(body, ("suite", "cases", "k", "rows", "limit", "offset"))
        return EvalRequest(suite=as_str(_field(body, "suite", "core"), "suite", max_chars=64),
                           cases=self.cases(body),
                           k=self._eval_k(body),
                           rows=as_bool(_field(body, "rows", True), "rows"),
                           page=self.page(body))

    def compare(self, body: Mapping[str, Any]) -> CompareRequest:
        self._reject_unknown(body, ("suite", "cases", "k", "indexes"))
        indexes = None
        if body.get("indexes") is not None:
            indexes = as_str_list(body["indexes"], "indexes", max_items=16, max_chars=64)
            if len(indexes) < 2:
                raise ValidationError("'indexes' must name at least two adapters",
                                      field="indexes", got=len(indexes))
        return CompareRequest(suite=as_str(_field(body, "suite", "hard"), "suite", max_chars=64),
                              indexes=indexes, cases=self.cases(body), k=self._eval_k(body))

    def configure(self, body: Mapping[str, Any]) -> ConfigRequest:
        self._reject_unknown(body, ("index", "generator", "cache"))
        if not any(body.get(f) is not None for f in ("index", "generator", "cache")):
            raise ValidationError("name at least one port to rebind",
                                  allowed=["index", "generator", "cache"])
        return ConfigRequest(
            index=self._opt_name(body, "index"),
            generator=self._opt_name(body, "generator"),
            cache=self._opt_name(body, "cache"))

    # ------------------------------------------------------------------ shared

    def _opt_name(self, body: Mapping[str, Any], field: str) -> str | None:
        value = body.get(field)
        return None if value is None else as_str(value, field, max_chars=64)

    def _eval_k(self, body: Mapping[str, Any]) -> int | None:
        if body.get("k") is None:
            return None
        return as_int(body["k"], "k", minimum=1, maximum=self.limits.max_k)

    def cases(self, body: Mapping[str, Any]) -> tuple[EvalCase, ...] | None:
        """Parse a caller-supplied labelled suite: ``[{question, expected_doc_id}, ...]``."""
        raw = body.get("cases")
        if raw is None:
            return None
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise ValidationError("'cases' must be a list of {question, expected_doc_id}",
                                  field="cases", got=type(raw).__name__)
        if not raw:
            raise ValidationError("'cases' must not be empty", field="cases")
        if len(raw) > self.limits.max_eval_cases:
            raise ValidationError(f"'cases' holds more than {self.limits.max_eval_cases} items",
                                  field="cases", limit=self.limits.max_eval_cases,
                                  got=len(raw))
        out: list[EvalCase] = []
        for position, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValidationError(f"'cases[{position}]' must be an object",
                                      field="cases", position=position)
            unknown = sorted(set(item) - {"question", "expected_doc_id"})
            if unknown:
                raise ValidationError(f"'cases[{position}]' has unknown field(s): {unknown}",
                                      field="cases", position=position)
            try:
                question = as_str(_field(item, "question"), f"cases[{position}].question",
                                  max_chars=self.limits.max_question_chars)
                expected = as_str(_field(item, "expected_doc_id"),
                                  f"cases[{position}].expected_doc_id",
                                  max_chars=self.limits.max_doc_id_chars)
            except ValidationError as exc:
                # Which row is wrong is the first thing a caller wants to know when they post
                # a suite of 500, so the position travels with the rejection.
                exc.details["position"] = position
                raise
            out.append(EvalCase(question=question, expected_doc_id=expected))
        return tuple(out)
