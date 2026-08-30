"""Structured logging and request metrics.

The service used to silence its own access log and import ``logging`` nowhere, so a running
instance offered nothing to look at when something went wrong. Every request now emits exactly
one JSON line on stderr, carrying the same request id the client was given in the
``X-Request-Id`` header and in any error body, so an operator can go from a user's screenshot
to the line that produced it.

What is deliberately *not* logged: the question text, document bodies, and the API token.
Questions are user content and may carry personal data, so the log records how long a question
was and what it retrieved, never what it said.
"""
from __future__ import annotations

import json
import logging
import secrets
import sys
import threading
import time
from bisect import bisect_left
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

LOGGER_NAME = "knowbase"

#: Per-request correlation id, set by the handler and read by anything that logs during it.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

#: Latency buckets in milliseconds. Coarse on purpose: this is a service whose slow path is
#: an ingest, not a hot loop, and a handful of buckets keeps /metrics readable by a human.
BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"}


def new_request_id() -> str:
    """Short, unguessable, and unique enough to grep for across a day of logs."""
    return secrets.token_hex(8)


class JsonFormatter(logging.Formatter):
    """Renders a record as one JSON object per line.

    Extra keyword fields passed as ``extra={...}`` become top-level keys, which is what makes
    the log queryable: ``status``, ``duration_ms`` and ``route`` are fields, not prose.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        rid = getattr(record, "request_id", "") or request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", stream: Any = None) -> logging.Logger:
    """Attach a single JSON handler to the ``knowbase`` logger and stop propagating.

    Called by the service entry points. Importing this module has no side effect, so a library
    user who wants their own logging configuration simply does not call this.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class RouteMetrics:
    """Counters and a latency histogram for one route."""

    requests: int = 0
    errors: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    buckets: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS_MS) + 1))
    total_ms: float = 0.0
    max_ms: float = 0.0

    def observe(self, status: int, duration_ms: float) -> None:
        self.requests += 1
        if status >= 400:
            self.errors += 1
        key = f"{status // 100}xx"
        self.by_status[key] = self.by_status.get(key, 0) + 1
        self.buckets[bisect_left(BUCKETS_MS, duration_ms)] += 1
        self.total_ms += duration_ms
        self.max_ms = max(self.max_ms, duration_ms)

    def quantile(self, q: float) -> float:
        """Upper bound of the bucket holding the ``q`` quantile, or -1 for the overflow bucket.

        A histogram cannot give an exact percentile and this does not pretend otherwise: the
        answer is the bucket edge, and ``-1`` means "slower than the largest bucket".
        """
        if self.requests == 0:
            return 0.0
        target = q * self.requests
        seen = 0
        for edge, count in zip(BUCKETS_MS, self.buckets):
            seen += count
            if seen >= target:
                return float(edge)
        return -1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "by_status": dict(sorted(self.by_status.items())),
            "avg_ms": round(self.total_ms / self.requests, 3) if self.requests else 0.0,
            "max_ms": round(self.max_ms, 3),
            "p50_ms_upper_bound": self.quantile(0.50),
            "p95_ms_upper_bound": self.quantile(0.95),
            "buckets_ms": dict(zip([str(int(b)) for b in BUCKETS_MS], self.buckets)),
            "over_bucket_ms": self.buckets[-1],
        }


class Metrics:
    """Process-wide request counters. Every mutation is under one lock (threaded server)."""

    def __init__(self, clock_now: Any = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._routes: dict[str, RouteMetrics] = {}
        self._now = clock_now
        self._started = self._now()

    def observe(self, route: str, status: int, duration_ms: float) -> None:
        with self._lock:
            self._routes.setdefault(route, RouteMetrics()).observe(status, duration_ms)

    def snapshot(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            routes = {name: m.to_dict() for name, m in sorted(self._routes.items())}
            totals = {"requests": sum(m.requests for m in self._routes.values()),
                      "errors": sum(m.errors for m in self._routes.values())}
        payload: dict[str, Any] = {
            "uptime_s": round(self._now() - self._started, 3),
            "totals": totals,
            "routes": routes,
            "buckets_ms": list(BUCKETS_MS),
        }
        if extra:
            payload.update(extra)
        return payload


def redact(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Drop empty values so a log line carries only the fields that were actually set."""
    return {k: v for k, v in pairs if v is not None and v != ""}
