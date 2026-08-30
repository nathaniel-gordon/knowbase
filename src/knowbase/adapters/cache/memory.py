"""Process-local answer cache."""
from __future__ import annotations

from ...domain.model import CacheStats, Generated
from ...ports import ClockPort


class InMemoryCache:
    """Dictionary-backed cache. Expiry is evaluated against the injected clock."""

    name = "memory"

    def __init__(self, clock: ClockPort) -> None:
        self._clock = clock
        self._store: dict[str, tuple[Generated, float | None]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Generated | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expires_at = entry
        if expires_at is not None and self._clock.now() >= expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, key: str, value: Generated, ttl_s: float | None = None) -> None:
        expires_at = None if ttl_s is None else self._clock.now() + ttl_s
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> CacheStats:
        return CacheStats(name=self.name, entries=len(self._store),
                          hits=self._hits, misses=self._misses)
