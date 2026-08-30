"""The primary (driving) adapter: a stdlib JSON microservice over the use-cases.

HTTP is deliberately a thin shell -- parse, validate, delegate to a use-case, map the result to
JSON. All knowledge-base behaviour lives inside the hexagon, which is why the same logic is
reachable from the tests without a socket.

Nothing is coerced in a handler any more: a request body becomes a typed object in
``validation.py`` or it becomes a 400. Errors on the wire are ``{"error": {"code", "message",
"details"}}`` with a stable code; the Python exception type is never part of the response.

Endpoints
    GET  /health    bound adapters, corpus size, uptime
    GET  /adapters  every adapter registered for every port
    POST /ingest    build the passage set and every index from a document source
    POST /ask       cited answer for a question
    POST /eval      retrieval metrics for the active index over a labelled suite
    POST /compare   A/B: the same suite through several index adapters, with a winner
    POST /config    rebind index / generator / cache at runtime
"""
from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

from ..domain.errors import (
    AdapterUnavailable,
    DocumentSourceUnavailable,
    EmptyCorpusError,
    IndexBuildFailed,
    IndexNotBuiltError,
    KbError,
    PathNotAllowed,
    ResourceLimitExceeded,
    ValidationError,
)
from .composition import Container, build_container
from .config import AppConfig
from .validation import RequestValidator
from .wire import answer_to_dict, comparison_to_dict, eval_to_dict, ingest_to_dict

#: Which HTTP status each domain error maps onto. Everything here is the client's fault
#: except the two 5xx entries, which mean the service could not do its job.
ERROR_STATUS: dict[type[KbError], HTTPStatus] = {
    ValidationError: HTTPStatus.BAD_REQUEST,
    IndexNotBuiltError: HTTPStatus.CONFLICT,
    EmptyCorpusError: HTTPStatus.BAD_REQUEST,
    IndexBuildFailed: HTTPStatus.BAD_REQUEST,
    DocumentSourceUnavailable: HTTPStatus.BAD_REQUEST,
    PathNotAllowed: HTTPStatus.FORBIDDEN,
    ResourceLimitExceeded: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    AdapterUnavailable: HTTPStatus.SERVICE_UNAVAILABLE,
}


def status_for(error: KbError) -> HTTPStatus:
    for kind in type(error).__mro__:
        if kind in ERROR_STATUS:
            return ERROR_STATUS[kind]
    return HTTPStatus.BAD_REQUEST


class KbHTTPServer(ThreadingHTTPServer):
    """Threading server that carries the container the handlers delegate to."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], container: Container) -> None:
        self.container = container
        self.validator = RequestValidator(container.limits)
        self._lock = threading.Lock()
        super().__init__(address, KbRequestHandler)

    @property
    def lock(self) -> threading.Lock:
        """Serialises mutating operations (ingest / rebind) across worker threads."""
        return self._lock

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"


class KbRequestHandler(BaseHTTPRequestHandler):
    """Routes requests to container use-cases and maps domain errors onto status codes."""

    server_version = "knowbase/0.1"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the stdlib access line; structured logging replaces it."""

    @property
    def container(self) -> Container:
        return self.server.container  # type: ignore[attr-defined]

    @property
    def validator(self) -> RequestValidator:
        return self.server.validator  # type: ignore[attr-defined]

    def setup(self) -> None:
        # Without a socket timeout a client that opens a connection and never finishes a
        # request holds a worker thread forever -- the classic slowloris.
        self.timeout = self.container.limits.request_timeout_s
        super().setup()

    def _send(self, status: HTTPStatus | int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus | int, code: str, message: str,
                    details: Mapping[str, Any] | None = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
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

    def _dispatch(self, routes: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]],
                  body: dict[str, Any] | None = None) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        handler = routes.get(path)
        if handler is None:
            self._send_error(HTTPStatus.NOT_FOUND, "no_such_endpoint",
                             f"no such endpoint: {path}", {"routes": sorted(routes)})
            return
        try:
            self._send(HTTPStatus.OK, handler(body if body is not None else {}))
        except KbError as exc:
            self._send_error(status_for(exc), exc.code, exc.message, exc.details)
        except Exception:  # noqa: BLE001 - nothing internal is allowed onto the wire
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error",
                             "the service failed to handle this request")

    # --------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch({"/health": self._health, "/adapters": self._adapters})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        try:
            body = self._read_json()
        except KbError as exc:
            self._send_error(status_for(exc), exc.code, exc.message, exc.details)
            return
        self._dispatch({"/ingest": self._ingest, "/ask": self._ask, "/eval": self._eval,
                        "/compare": self._compare, "/config": self._config}, body)

    # ------------------------------------------------------------- handlers

    def _health(self, _: dict[str, Any]) -> dict[str, Any]:
        c = self.container
        stats = c.cache.stats()
        return {"status": "ok", "wiring": c.wiring(),
                "indexed_passages": c.index.size(),
                "corpus": ingest_to_dict(c.last_ingest) if c.last_ingest else None,
                "cache": {"entries": stats.entries, "hits": stats.hits,
                          "misses": stats.misses},
                "started_at": c.started_at}

    def _adapters(self, _: dict[str, Any]) -> dict[str, Any]:
        from ..adapters import (CACHE_ADAPTERS, CLOCK_ADAPTERS, GENERATOR_ADAPTERS,
                                INDEX_ADAPTERS, SOURCE_ADAPTERS)
        return {"ports": {"DocumentSourcePort": sorted(SOURCE_ADAPTERS),
                          "IndexPort": sorted(INDEX_ADAPTERS),
                          "GeneratorPort": sorted(GENERATOR_ADAPTERS),
                          "CachePort": sorted(CACHE_ADAPTERS),
                          "ClockPort": sorted(CLOCK_ADAPTERS)},
                "bound": self.container.wiring()}

    def _ingest(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.ingest(body)
        with self.server.lock:  # type: ignore[attr-defined]
            return ingest_to_dict(self.container.ingest(request))

    def _ask(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.ask(body)
        return answer_to_dict(self.container.ask(request.question, k=request.k))

    def _eval(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.evaluate(body)
        return eval_to_dict(self.container.evaluate(request),
                            rows=request.rows, page=request.page)

    def _compare(self, body: dict[str, Any]) -> dict[str, Any]:
        return comparison_to_dict(self.container.compare(self.validator.compare(body)))

    def _config(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self.validator.configure(body)
        with self.server.lock:  # type: ignore[attr-defined]
            changes = self.container.rebind(index=request.index, generator=request.generator,
                                            cache=request.cache)
        return {"changed": changes, "wiring": self.container.wiring()}


def create_server(config: AppConfig | Mapping[str, Any] | None = None,
                  host: str = "127.0.0.1", port: int = 0) -> KbHTTPServer:
    """Bind a server. ``port=0`` asks the OS for an ephemeral port (the default everywhere)."""
    return KbHTTPServer((host, port), build_container(config))


def serve_forever(config: AppConfig | Mapping[str, Any] | None = None,
                  host: str = "127.0.0.1", port: int = 0) -> None:  # pragma: no cover - manual
    """Run in the foreground until interrupted (``python -m knowbase serve``)."""
    server = create_server(config, host, port)
    print(f"knowbase listening on {server.url} (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        server.container.close()
