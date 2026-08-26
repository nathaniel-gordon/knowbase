"""The five ports of the hexagon.

A port is a ``typing.Protocol``: the domain states the shape of collaboration it needs, and
adapters conform structurally without inheriting anything. Dependencies therefore point
*inward* -- ``kb.adapters`` imports ``kb.ports`` and ``kb.domain``, never the other way round.

Every port here has at least two production adapters and a shared contract test battery in
``tests/contracts`` that is parametrised over all of them.
"""
from __future__ import annotations

from .cache import CachePort
from .clock import ClockPort
from .document_source import DocumentSourcePort
from .generator import GeneratorPort
from .index import IndexPort

__all__ = ["CachePort", "ClockPort", "DocumentSourcePort", "GeneratorPort", "IndexPort"]
