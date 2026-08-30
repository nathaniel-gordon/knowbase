"""Enterprise knowledge-base RAG, built as a hexagon.

    knowbase.domain    entities + pure text services + use-cases   (imports nothing outward)
    knowbase.ports     the five Protocols the domain talks through
    knowbase.adapters  concrete implementations, >= 2 per port
    knowbase.app       composition root, HTTP service, client SDK, demo

Start here::

    from knowbase.app import AppConfig, running_service

    with running_service(AppConfig(index="hybrid_rrf")) as client:
        client.ingest_path("docs/")
        print(client.ask("What is our refund policy?").render())
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
