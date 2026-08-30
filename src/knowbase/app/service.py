"""The primary (driving) adapter: a stdlib JSON microservice over the use-cases.

HTTP is deliberately a thin shell -- authenticate, parse, validate, delegate to a use-case, map
the result to JSON. All knowledge-base behaviour lives inside the hexagon, which is why the
same logic is reachable from the tests without a socket.

Nothing is coerced in a handler: a request body becomes a typed object in ``validation.py`` or
it becomes a 400. Errors on the wire are ``{"error": {"code", "message", "details",
"request_id"}}`` with a stable code; the Python exception type is never part of the response.

Routes
    GET  /                  what this service is and where its endpoints live
    GET  /health            liveness plus the bound adapters and the corpus on hand
    GET  /ready             readiness: 200 once a corpus is queryable, 503 before that
    GET  /metrics           per-route counters, latency buckets, cache and corpus gauges
    GET  /api/v1/adapters   every adapter registered for every port
    POST /api/v1/ingest     build the passage set and every index from a document source
    POST /api/v1/ask        cited answer for a question
    POST /api/v1/eval       retrieval metrics over a labelled suite, bundled or your own
    POST /api/v1/compare    A/B: the same suite through several index adapters, with a winner
    POST /api/v1/config     rebind index / generator / cache at runtime

The operational endpoints sit outside the version prefix on purpose: a probe or a scrape
should not have to be updated when the API is versioned.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

from ..domain.errors import (
    AdapterUnavailable,
    CorpusStoreUnavailable,
    DocumentSourceUnavailable,
    EmptyCorpusError,
    IndexBuildFailed,
    IndexNotBuiltError,
    KbError,
    PathNotAllowed,
    ResourceLimitExceeded,
    ValidationError,
)
from .auth import Authenticator, Unauthenticated
from .composition import Container, build_container
from .config import AppConfig
from .observability import (
    Metrics,
    configure_logging,
    get_logger,
    new_request_id,
    redact,
    request_id_var,
)
from .validation import RequestValidator
from .wire import (
    answer_to_dict,
    comparison_to_dict,
    corpus_to_dict,
    eval_to_dict,
    ingest_to_dict,
)

API_PREFIX = "/api/v1"

#: Which HTTP status each domain error maps onto. Everything here is the client's fault
#: except the 5xx entries, which mean the service could not do its job.
ERROR_STATUS: dict[type[KbError], HTTPStatus] = {
    ValidationError: HTTPStatus.BAD_REQUEST,
    IndexNotBuiltError: HTTPStatus.CONFLICT,
    EmptyCorpusError: HTTPStatus.BAD_REQUEST,
    IndexBuildFailed: HTTPStatus.BAD_REQUEST,
    DocumentSourceUnavailable: HTTPStatus.BAD_REQUEST,
    PathNotAllowed: HTTPStatus.FORBIDDEN,
    Unauthenticated: HTTPStatus.UNAUTHORIZED,
    ResourceLimitExceeded: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    AdapterUnavailable: HTTPStatus.SERVICE_UNAVAILABLE,
    CorpusStoreUnavailable: HTTPStatus.SERVICE_UNAVAILABLE,
}


def status_for(error: KbError) -> HTTPStatus:
    for kind in type(error).__mro__:
        if kind in ERROR_STATUS:
            return ERROR_STATUS[kind]
    return HTTPStatus.BAD_REQUEST


class KbHTTPServer(ThreadingHTTPServer):
    """Threading server carrying the container, the validator, the auth check and the metrics."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], container: Container,
                 auth: Authenticator | None = None) -> None:
        self.container = container
        self.validator = RequestValidator(container.limits)
        self.auth = auth or Authenticator()
        self.metrics = Metrics()
        self._lock = threading.Lock()
        super().__init__(address, KbRequestHandler)

    @property
    def lock(self) -> threading.Lock:
        """Serialises mutating operations (ingest / rebind) across worker threads."""
        return self._lock

    @property
    def url(self) -> str:
        """Loopback URL of the bound socket.

        ``server_address`` is a str/int pair only for AF_INET; decoding defensively keeps a
        UNIX-socket family from rendering as ``b'/tmp/sock'`` in the URL the SDK connects to.
        """
        host, port = self.server_address[0], self.server_address[1]
        if isinstance(host, bytes):
            host = host.decode("utf-8", errors="replace")
        return f"http://{host}:{port}"


class KbRequestHandler(BaseHTTPRequestHandler):
    """Routes requests to container use-cases and maps domain errors onto status codes."""

    server_version = "knowbase/0.1"
    protocol_version = "HTTP/1.1"

    #: Endpoints a probe must be able to reach without a credential.
    OPEN_ROUTES = frozenset({"/", "/health", "/ready"})

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        """Suppress the stdlib access line; the structured line in ``_dispatch`` replaces it."""

    @property
    def kb(self) -> KbHTTPServer:
        """The server that owns this handler, typed. ``BaseServer`` knows none of our fields."""
        return cast(KbHTTPServer, self.server)

    @property
    def container(self) -> Container:
        return self.kb.container

    @property
    def validator(self) -> RequestValidator:
        return self.kb.validator

    def setup(self) -> None:
        # Without a socket timeout a client that opens a connection and never finishes a
        # request holds a worker thread forever -- the classic slowloris.
        self.request.settimeout(self.container.limits.request_timeout_s)
        super().setup()

    def _send(self, status: HTTPStatus | int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", request_id_var.get())
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus | int, code: str, message: str,
                    details: Mapping[str, Any] | None = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message,
                                 "request_id": request_id_var.get()}
        if details:
            error["details"] = dict(details)
        self._send(status, {"error": error})

    def _read_json(self) -> dict[str, Any]:
        """Read a JSON body, or raise ValidationError. Chunked bodies are refused outright.

        The old handler trusted Content-Length and ignored Transfer-Encoding, so a chunked
        POST was silently read as ``{}`` and processed as if the client had sent nothing.
        """
        if self.headers.get("Transfer-Encoding"):
            raise ValidationError("chunked request bodies are not supported; send "
                                  "Content-Length", header="Transfer-Encoding")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except ValueError:
            raise ValidationError("Content-Length is not an integer",
                                  header="Content-Length") from None
        limit = self.container.limits.max_body_bytes
        if length < 0:
            raise ValidationError("Content-Length is negative", header="Content-Length")
        if length > limit:
            raise ResourceLimitExceeded(f"request body exceeds {limit} bytes", limit=limit,
                                        got=length)
        if length == 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("body is not valid UTF-8 JSON", reason=str(exc)) from exc
        if not isinstance(body, dict):
            raise ValidationError("body must be a JSON object", got=type(body).__name__)
        return body

    # --------------------------------------------------------------- routing

    def _dispatch(self, routes: Mapping[str, Callable[..., Mapping[str, Any]]],
                  method: str) -> None:
        """One place where a request becomes a status, a log line and a metric."""
        request_id_var.set(new_request_id())
        started = time.perf_counter()
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        status = HTTPStatus.OK
        facts: dict[str, Any] = {}
        try:
            handler = routes.get(path)
            if handler is None:
                raise _NoSuchEndpoint(path, sorted(routes))
            if path not in self.OPEN_ROUTES:
                self.kb.auth.authorise(self.headers.get("Authorization"))
            body = self._read_json() if method == "POST" else {}
            payload, facts = _split(handler(body))
            status = HTTPStatus(facts.pop("status", HTTPStatus.OK))
            self._send(status, payload)
        except _NoSuchEndpoint as exc:
            status = HTTPStatus.NOT_FOUND
            self._send_error(status, "no_such_endpoint", f"no such endpoint: {exc.path}",
                             {"routes": exc.routes})
        except KbError as exc:
            status = status_for(exc)
            self._send_error(status, exc.code, exc.message, exc.details)
            if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
                get_logger().error("request_failed", extra={"code": exc.code, "route": path})
        except Exception:  # noqa: BLE001 - the last line of defence for the whole handler
            # Nothing internal reaches the wire. The traceback goes to the log, under the same
            # request id the client was handed, which is the only way to connect the two.
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            get_logger().exception("unhandled_error", extra={"route": path, "method": method})
            self._send_error(status, "internal_error",
                             "the service failed to handle this request")
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.kb.metrics.observe(path, int(status), duration_ms)
            get_logger().info("request", extra=redact([
                ("method", method), ("route", path), ("status", int(status)),
                ("duration_ms", duration_ms), *facts.items(),
            ]))

    def do_GET(self) -> None:
        self._dispatch({"/": self._index,
                        "/health": self._health,
                        "/ready": self._ready,
                        "/metrics": self._metrics,
                        f"{API_PREFIX}/adapters": self._adapters}, "GET")

    def do_POST(self) -> None:
        self._dispatch({f"{API_PREFIX}/ingest": self._ingest,
                        f"{API_PREFIX}/ask": self._ask,
                        f"{API_PREFIX}/eval": self._eval,
                        f"{API_PREFIX}/compare": self._compare,
                        f"{API_PREFIX}/config": self._config}, "POST")

    # ------------------------------------------------------------- handlers

    def _index(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"service": "knowbase", "api": API_PREFIX,
                "operational": ["/health", "/ready", "/metrics"],
                "endpoints": [f"{API_PREFIX}/{name}" for name in
                              ("adapters", "ingest", "ask", "eval", "compare", "config")],
                "authentication": "bearer" if self.kb.auth.enabled else "none"}

    def _health(self, _: dict[str, Any]) -> dict[str, Any]:
        c = self.container
        stats = c.cache.stats()
        return {"status": "ok", "wiring": c.wiring(),
                "indexed_passages": c.index.size(),
                "corpus": corpus_to_dict(c.corpus, c.last_ingest, c.restored),
                "cache": {"entries": stats.entries, "hits": stats.hits,
                          "misses": stats.misses},
                "started_at": c.started_at}

    def _ready(self, _: dict[str, Any]) -> dict[str, Any]:
        """Readiness is corpus-shaped: the process is up long before it can answer anything."""
        passages = self.container.index.size()
        ready = passages > 0
        return {"ready": ready,
                "reason": None if ready else "no corpus ingested yet",
                "indexed_passages": passages,
                "_facts": {"status": HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE}}

    def _metrics(self, _: dict[str, Any]) -> dict[str, Any]:
        c = self.container
        stats = c.cache.stats()
        return self.kb.metrics.snapshot({
            "cache": {"name": stats.name, "entries": stats.entries, "hits": stats.hits,
                      "misses": stats.misses, "hit_rate": round(stats.hit_rate, 4)},
            "corpus": {"passages": c.index.size(),
                       "documents": c.corpus.documents if c.corpus else 0,
                       "fingerprint": c.fingerprint},
            "wiring": c.wiring()})

    def _adapters(self, _: dict[str, Any]) -> dict[str, Any]:
        from ..adapters import (
            CACHE_ADAPTERS,
            CLOCK_ADAPTERS,
            GENERATOR_ADAPTERS,
            INDEX_ADAPTERS,
            SOURCE_ADAPTERS,
            STORE_ADAPTERS,
        )
        return {"ports": {"DocumentSourcePort": sorted(SOURCE_ADAPTERS),
                          "IndexPort": sorted(INDEX_ADAPTERS),
                          "GeneratorPort": sorted(GENERATOR_ADAPTERS),
                          "CachePort": sorted(CACHE_ADAPTERS),
                          "CorpusStorePort": sorted(STORE_ADAPTERS),
                          "ClockPort": sorted(CLOCK_ADAPTERS)},
                "bound": self.container.wiring()}

    def _ingest(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.ingest(body)
        with self.kb.lock:
            report = self.container.ingest(request)
        payload = ingest_to_dict(report)
        return {**payload, "_facts": {"source": report.source, "documents": report.documents,
                                      "passages": report.passages}}

    def _ask(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.ask(body)
        answer = self.container.ask(request.question, k=request.k)
        # The question itself is user content and stays out of the log; its length does not.
        return {**answer_to_dict(answer),
                "_facts": {"question_chars": len(request.question),
                           "index": answer.index, "generator": answer.generator,
                           "hits": len(answer.hits), "citations": len(answer.citations),
                           "cache": "hit" if answer.cached else "miss"}}

    def _eval(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.evaluate(body)
        report = self.container.evaluate(request)
        payload = eval_to_dict(report, rows=request.rows, page=request.page)
        return {**payload, "_facts": {"suite": report.suite, "cases": report.n,
                                      "mrr": report.mrr}}

    def _compare(self, body: dict[str, Any]) -> dict[str, Any]:
        report = self.container.compare(self.validator.compare(body))
        return {**comparison_to_dict(report),
                "_facts": {"suite": report.suite, "winner": report.winner,
                           "indexes": len(report.reports)}}

    def _config(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.configure(body)
        with self.kb.lock:
            changes = self.container.rebind(index=request.index, generator=request.generator,
                                            cache=request.cache)
        return {"changed": changes, "wiring": self.container.wiring(),
                "_facts": {"rebound": ",".join(sorted(changes)) or "nothing"}}


class _NoSuchEndpoint(Exception):
    """Internal control flow for an unrouted path; never reaches a client as a type."""

    def __init__(self, path: str, routes: list[str]) -> None:
        super().__init__(path)
        self.path = path
        self.routes = routes


def _split(result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the response body from the ``_facts`` a handler wants logged."""
    payload = dict(result)
    facts = dict(payload.pop("_facts", {}) or {})
    return payload, facts


def create_server(config: AppConfig | Mapping[str, Any] | None = None,
                  host: str = "127.0.0.1", port: int = 0,
                  token: str | None = None) -> KbHTTPServer:
    """Bind a server. ``port=0`` asks the OS for an ephemeral port (the default everywhere).

    ``token`` exists for tests; in operation the bearer token comes from ``KB_API_TOKEN`` and
    binding a non-loopback address without one is refused here, before the socket opens.
    """
    auth = Authenticator.from_environment(token)
    auth.check_bind(host)
    return KbHTTPServer((host, port), build_container(config), auth)


def serve_forever(config: AppConfig | Mapping[str, Any] | None = None,
                  host: str = "127.0.0.1", port: int = 0,
                  log_level: str = "INFO") -> None:  # pragma: no cover - manual
    """Run in the foreground until interrupted (``python -m knowbase serve``)."""
    configure_logging(log_level)
    server = create_server(config, host, port)
    log = get_logger()
    log.info("listening", extra={"url": server.url, "wiring": server.container.wiring(),
                                 "auth": "bearer" if server.auth.enabled else "none"})
    print(f"knowbase listening on {server.url} (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        server.shutdown()
        server.server_close()
        server.container.close()
        log.info("stopped")
