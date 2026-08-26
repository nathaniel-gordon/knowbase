"""Application layer: composition root, driving HTTP adapter, client SDK, demo.

This is the *outside* of the hexagon on the driving side. It knows both the domain and the
adapters; nothing here contains knowledge-base rules.
"""
from __future__ import annotations

from .client import (
    AnswerView,
    ComparisonView,
    EvalView,
    IngestView,
    KbClient,
    ServiceError,
    running_service,
)
from .composition import Container, build_container
from .config import AppConfig
from .service import KbHTTPServer, create_server, serve_forever

__all__ = ["AnswerView", "AppConfig", "ComparisonView", "Container", "EvalView", "IngestView",
           "KbClient", "KbHTTPServer", "ServiceError", "build_container", "create_server",
           "running_service", "serve_forever"]
