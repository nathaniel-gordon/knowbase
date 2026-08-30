"""ClockPort -- the only source of time the domain is allowed to consult.

Injecting time is what lets the ingest/ask/eval use-cases report latency while remaining
deterministic under test: the fixtures wire a ``FrozenClock`` and every timestamp becomes
reproducible.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClockPort(Protocol):
    """Wall-clock reader.

    Contract (enforced by ``tests/contracts/clock_contract.py``):

    * ``now()`` returns epoch seconds as a float and is non-decreasing across calls.
    * ``timestamp()`` returns an ISO-8601 string that ``datetime.fromisoformat`` can parse and
      that corresponds to the value ``now()`` would return.
    """

    name: str

    def now(self) -> float:
        """Epoch seconds."""
        ...

    def timestamp(self) -> str:
        """ISO-8601 rendering of the current instant."""
        ...
