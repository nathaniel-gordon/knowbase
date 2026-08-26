"""Enterprise knowledge-base RAG, built as a hexagon.

    kb.domain    entities + pure text services + use-cases   (imports nothing outward)
    kb.ports     the five Protocols the domain talks through
    kb.adapters  concrete implementations, >= 2 per port
    kb.app       composition root, HTTP service, client SDK, demo

Start here::

    from kb.app import AppConfig, running_service

    with running_service(AppConfig(index="hybrid_rrf")) as client:
        client.ingest_path("docs/")
        print(client.ask("What is our refund policy?").render())
"""
from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
