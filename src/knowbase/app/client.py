"""Typed Python client SDK for the knowledge-base service.

The SDK exists so callers get dataclasses and real exceptions instead of raw dictionaries, and
so the service can be driven identically from the demo, the tests, and a user's own script.

``running_service()`` is the piece that makes the whole thing self-contained: it starts a server
on an ephemeral 127.0.0.1 port in a background thread, yields a client bound to it, and always
shuts it down -- no ports to pick, nothing left running.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .service import API_PREFIX, create_server


class ServiceError(RuntimeError):
    """The service answered with a non-2xx status.

    ``code`` is the stable machine-readable identifier from the error taxonomy; switch on it
    rather than on the message, which is prose and may be reworded.
    """

    def __init__(self, status: int, payload: Mapping[str, Any]) -> None:
        error = payload.get("error")
        error = error if isinstance(error, Mapping) else {"code": "unknown", "message": str(error)}
        super().__init__(f"HTTP {status} [{error.get('code')}]: {error.get('message')}")
        self.status = status
        self.code = str(error.get("code", "unknown"))
        self.message = str(error.get("message", ""))
        self.details = dict(error.get("details") or {})
        self.request_id = str(error.get("request_id", ""))
        self.payload = dict(payload)


@dataclass(frozen=True, slots=True)
class CitationView:
    marker: int
    passage_id: str
    title: str
    origin: str = ""


@dataclass(frozen=True, slots=True)
class HitView:
    rank: int
    passage_id: str
    doc_id: str
    score: float
    title: str = ""


@dataclass(frozen=True, slots=True)
class AnswerView:
    question: str
    answer: str
    generator: str
    index: str
    cached: bool
    latency_ms: float
    citations: tuple[CitationView, ...] = ()
    retrieved: tuple[HitView, ...] = ()

    def render(self) -> str:
        cites = "\n".join(f"  [{c.marker}] {c.title} ({c.passage_id})" for c in self.citations)
        tag = f"{self.generator}{', cached' if self.cached else ''}"
        return f"Q: {self.question}\nA ({tag}): {self.answer}\n\nSources:\n{cites or '  (none)'}"


@dataclass(frozen=True, slots=True)
class IngestView:
    source: str
    index: str
    documents: int
    passages: int
    started_at: str
    elapsed_ms: float
    fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class EvalView:
    index: str
    suite: str
    k: int
    n: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    rows: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    page: Mapping[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.index:<11} suite={self.suite:<5} n={self.n:<3} "
                f"hit@1={self.hit_at_1:.3f} hit@{self.k}={self.hit_at_k:.3f} mrr={self.mrr:.3f}")


@dataclass(frozen=True, slots=True)
class ComparisonView:
    suite: str
    k: int
    winner: str
    margin: float
    reports: tuple[EvalView, ...] = ()

    def table(self) -> str:
        lines = [r.summary() for r in self.reports]
        verdict = ("tie" if self.winner == "tie"
                   else f"{self.winner} wins by {self.margin:+.3f} MRR")
        lines.append(f"-> {verdict}")
        return "\n".join(lines)


def _eval_view(payload: Mapping[str, Any]) -> EvalView:
    k = int(payload["k"])
    return EvalView(index=payload["index"], suite=payload["suite"], k=k, n=int(payload["n"]),
                    hit_at_1=float(payload["hit@1"]), hit_at_k=float(payload[f"hit@{k}"]),
                    mrr=float(payload["mrr"]),
                    rows=tuple(payload.get("rows", ())),
                    page=dict(payload.get("page", {})))


class KbClient:
    """Typed HTTP client. One method per endpoint, one dataclass per response."""

    def __init__(self, base_url: str, timeout_s: float = 30.0,
                 token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._token = token or ""

    # ------------------------------------------------------------ transport

    def _request(self, method: str, path: str,
                 payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method,
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                decoded: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return decoded
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": {"code": "unparseable_response", "message": raw[:500]}}
            raise ServiceError(exc.code, body) from None

    def _api(self, method: str, name: str,
             payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._request(method, f"{API_PREFIX}/{name}", payload)

    # ------------------------------------------------------------- endpoints

    def index(self) -> dict[str, Any]:
        """The unversioned service index: where the API lives and whether auth is on."""
        return self._request("GET", "/")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ready(self) -> bool:
        """True once a corpus is queryable. A 503 here is an answer, not a failure."""
        try:
            return bool(self._request("GET", "/ready")["ready"])
        except ServiceError as exc:
            if exc.status == 503:
                return False
            raise

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def adapters(self) -> dict[str, Any]:
        return self._api("GET", "adapters")

    def ingest_path(self, path: str, source: str = "filesystem",
                    exts: Sequence[str] | None = None) -> IngestView:
        body: dict[str, Any] = {"source": source, "path": str(path)}
        if exts:
            body["exts"] = list(exts)
        return IngestView(**self._api("POST", "ingest", body))

    def ingest_documents(self, documents: Mapping[str, str], label: str = "inline") -> IngestView:
        return IngestView(**self._api("POST", "ingest",
                                      {"source": "memory", "documents": dict(documents),
                                       "label": label}))

    def ask(self, question: str, k: int | None = None) -> AnswerView:
        body: dict[str, Any] = {"question": question}
        if k is not None:
            body["k"] = k
        payload = self._api("POST", "ask", body)
        return AnswerView(
            question=payload["question"], answer=payload["answer"],
            generator=payload["generator"], index=payload["index"],
            cached=bool(payload["cached"]), latency_ms=float(payload["latency_ms"]),
            citations=tuple(CitationView(**c) for c in payload["citations"]),
            retrieved=tuple(HitView(**h) for h in payload["retrieved"]))

    def evaluate(self, suite: str = "core", k: int | None = None, rows: bool = True,
                 cases: Sequence[Mapping[str, str]] | None = None,
                 limit: int | None = None, offset: int | None = None) -> EvalView:
        """Evaluate the active index. Pass ``cases`` to score your own labelled questions."""
        body: dict[str, Any] = {"suite": suite, "rows": rows}
        if k is not None:
            body["k"] = k
        if cases is not None:
            body["cases"] = [dict(c) for c in cases]
        if limit is not None:
            body["limit"] = limit
        if offset is not None:
            body["offset"] = offset
        return _eval_view(self._api("POST", "eval", body))

    def compare(self, indexes: Sequence[str] | None = None, suite: str = "hard",
                k: int | None = None,
                cases: Sequence[Mapping[str, str]] | None = None) -> ComparisonView:
        body: dict[str, Any] = {"suite": suite}
        if indexes:
            body["indexes"] = list(indexes)
        if k is not None:
            body["k"] = k
        if cases is not None:
            body["cases"] = [dict(c) for c in cases]
        payload = self._api("POST", "compare", body)
        return ComparisonView(suite=payload["suite"], k=int(payload["k"]),
                              winner=payload["winner"], margin=float(payload["margin"]),
                              reports=tuple(_eval_view(r) for r in payload["reports"]))

    def configure(self, index: str | None = None, generator: str | None = None,
                  cache: str | None = None) -> dict[str, Any]:
        body = {k: v for k, v in
                (("index", index), ("generator", generator), ("cache", cache)) if v is not None}
        return self._api("POST", "config", body)


@contextmanager
def running_service(config: AppConfig | Mapping[str, Any] | None = None,
                    host: str = "127.0.0.1", port: int = 0,
                    timeout_s: float = 30.0,
                    token: str | None = None) -> Iterator[KbClient]:
    """Start the service on an ephemeral port, yield a client, and always shut it down."""
    server = create_server(config, host, port, token=token)
    thread = threading.Thread(target=server.serve_forever, name="knowbase-http", daemon=True)
    thread.start()
    try:
        yield KbClient(server.url, timeout_s=timeout_s, token=token or server.auth.token)
    finally:
        server.shutdown()
        thread.join(timeout=10.0)
        server.server_close()
        server.container.close()
