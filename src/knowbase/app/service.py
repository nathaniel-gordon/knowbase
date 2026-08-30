"""The primary (driving) adapter: a stdlib JSON microservice over the use-cases.

HTTP is deliberately a thin shell -- parse, delegate to a use-case, map the result to JSON. All
knowledge-base behaviour lives inside the hexagon, which is why the same logic is reachable from
the tests without a socket.

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

from ..domain.errors import KbError, UnknownStrategyError
from .composition import Container, build_container
from .config import AppConfig
from .wire import answer_to_dict, comparison_to_dict, eval_to_dict, ingest_to_dict

MAX_BODY_BYTES = 4 * 1024 * 1024


class KbHTTPServer(ThreadingHTTPServer):
    """Threading server that carries the container the handlers delegate to."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], container: Container) -> None:
        super().__init__(address, KbRequestHandler)
        self.container = container
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        """Serialises mutating operations (ingest / rebind) across worker threads."""
        return self._lock

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class KbRequestHandler(BaseHTTPRequestHandler):
    """Routes requests to container use-cases and maps domain errors onto status codes."""

    server_version = "knowbase/0.1"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence per-request logging; the demo prints its own transcript."""

    @property
    def container(self) -> Container:
        return self.server.container  # type: ignore[attr-defined]

    def _send(self, status: HTTPStatus | int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed JSON body: {exc}") from exc

    def _dispatch(self, routes: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]],
                  body: dict[str, Any] | None = None) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        handler = routes.get(path)
        if handler is None:
            self._send(HTTPStatus.NOT_FOUND,
                       {"error": f"no such endpoint: {path}", "routes": sorted(routes)})
            return
        try:
            self._send(HTTPStatus.OK, handler(body if body is not None else {}))
        except UnknownStrategyError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc), "kind": "unknown_adapter"})
        except (ValueError, KbError) as exc:
            self._send(HTTPStatus.BAD_REQUEST,
                       {"error": str(exc), "kind": type(exc).__name__})
        except Exception as exc:  # pragma: no cover - last-resort guard
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": str(exc), "kind": type(exc).__name__})

    # --------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch({"/health": self._health, "/adapters": self._adapters})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc), "kind": "bad_request"})
            return
        self._dispatch({"/ingest": self._ingest, "/ask": self._ask, "/eval": self._eval,
                        "/compare": self._compare, "/config": self._config}, body)

    # ------------------------------------------------------------- handlers

    def _health(self, _: dict[str, Any]) -> dict[str, Any]:
        c = self.container
        return {"status": "ok", "wiring": c.wiring(),
                "indexed_passages": c.index.size(),
                "corpus": ingest_to_dict(c.last_ingest) if c.last_ingest else None,
                "cache": {"entries": c.cache.stats().entries,
                          "hits": c.cache.stats().hits,
                          "misses": c.cache.stats().misses},
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
        with self.server.lock:  # type: ignore[attr-defined]
            return ingest_to_dict(self.container.ingest(body))

    def _ask(self, body: dict[str, Any]) -> dict[str, Any]:
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("'question' must be a non-empty string")
        k = body.get("k")
        return answer_to_dict(self.container.ask(question, k=int(k) if k is not None else None))

    def _eval(self, body: dict[str, Any]) -> dict[str, Any]:
        k = body.get("k")
        return eval_to_dict(self.container.evaluate(suite=str(body.get("suite", "core")),
                                                    k=int(k) if k is not None else None),
                            with_rows=bool(body.get("rows", True)))

    def _compare(self, body: dict[str, Any]) -> dict[str, Any]:
        names = body.get("indexes")
        k = body.get("k")
        return comparison_to_dict(self.container.compare(
            names=list(names) if names else None,
            suite=str(body.get("suite", "hard")),
            k=int(k) if k is not None else None))

    def _config(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.server.lock:  # type: ignore[attr-defined]
            changes = self.container.rebind(index=body.get("index"),
                                            generator=body.get("generator"),
                                            cache=body.get("cache"))
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
