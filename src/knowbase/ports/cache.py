"""CachePort -- memoisation of generated answers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.model import CacheStats, Generated


@runtime_checkable
class CachePort(Protocol):
    """A keyed store of generated answers with optional time-to-live.

    Contract (enforced by ``tests/contracts/cache_contract.py``):

    * ``get`` on an absent key returns ``None`` and counts a miss; on a present key it returns
      an equal :class:`Generated` and counts a hit.
    * ``set`` overwrites an existing key.
    * Expiry is decided against an injected ``ClockPort``, never ``time.time()`` -- which is why
      the TTL behaviour can be tested deterministically with a frozen clock.
    * ``clear`` empties the store but preserves the cumulative hit/miss counters.
    * Values survive round-tripping unchanged, including non-ASCII text and empty citations.
    """

    name: str

    def get(self, key: str) -> Generated | None:
        """Return the cached value, or ``None`` if absent or expired."""
        ...

    def set(self, key: str, value: Generated, ttl_s: float | None = None) -> None:
        """Store ``value`` under ``key``, expiring after ``ttl_s`` seconds if given."""
        ...

    def delete(self, key: str) -> bool:
        """Remove ``key``; return whether it was present."""
        ...

    def clear(self) -> None:
        """Drop all entries (counters are retained)."""
        ...

    def stats(self) -> CacheStats:
        """Current entry count and cumulative hit/miss counters."""
        ...
