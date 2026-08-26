"""Contract battery for CachePort.

Expiry is asserted against an injected frozen clock, which is only possible because the port
forbids adapters from reading the wall clock themselves.
"""
from __future__ import annotations

import pytest

from kb.domain.model import Generated
from kb.ports import CachePort

VALUE = Generated(text="Employees receive 21 paid vacation days each year. [1]", cited=(1,))
OTHER = Generated(text="Passwords must be at least 14 characters. [2]", cited=(2,))
UNICODE = Generated(text="Réunion trimestrielle — 5 jours ouvrés · 90 % couverture", cited=())


class CacheContract:
    """Every CachePort adapter must pass all of these.

    Subclasses provide a ``cache`` fixture built on the battery's ``clock`` fixture.
    """

    @pytest.fixture
    def cache(self, frozen_clock) -> CachePort:
        raise NotImplementedError("contract subclass must provide a 'cache' fixture")

    def test_satisfies_the_protocol(self, cache):
        assert isinstance(cache, CachePort)
        assert isinstance(cache.name, str) and cache.name == cache.name.lower()

    def test_missing_key_returns_none_and_counts_a_miss(self, cache):
        assert cache.get("absent") is None
        stats = cache.stats()
        assert (stats.hits, stats.misses, stats.entries) == (0, 1, 0)

    def test_round_trip_preserves_the_value_and_counts_a_hit(self, cache):
        cache.set("k", VALUE)
        assert cache.get("k") == VALUE
        stats = cache.stats()
        assert (stats.hits, stats.misses, stats.entries) == (1, 0, 1)
        assert stats.hit_rate == 1.0

    def test_set_overwrites(self, cache):
        cache.set("k", VALUE)
        cache.set("k", OTHER)
        assert cache.get("k") == OTHER
        assert cache.stats().entries == 1

    def test_delete_reports_whether_the_key_existed(self, cache):
        cache.set("k", VALUE)
        assert cache.delete("k") is True
        assert cache.delete("k") is False
        assert cache.get("k") is None

    def test_clear_empties_the_store_but_keeps_the_counters(self, cache):
        cache.set("a", VALUE)
        cache.set("b", OTHER)
        assert cache.get("a") == VALUE
        cache.clear()
        assert cache.stats().entries == 0
        assert cache.get("a") is None
        assert cache.stats().hits == 1, "cumulative counters survive a flush"

    def test_values_survive_unicode_and_empty_citations(self, cache):
        cache.set("u", UNICODE)
        restored = cache.get("u")
        assert restored == UNICODE
        assert restored.cited == ()

    def test_entries_expire_against_the_injected_clock(self, cache):
        # frozen_clock advances 1.0s per read; a 2.5s TTL therefore survives exactly two reads.
        cache.set("k", VALUE, ttl_s=2.5)
        assert cache.get("k") == VALUE      # t+1
        assert cache.get("k") == VALUE      # t+2
        assert cache.get("k") is None       # t+3 -> expired
        assert cache.stats().entries == 0, "an expired entry is evicted, not merely hidden"

    def test_entries_without_ttl_do_not_expire(self, cache):
        cache.set("k", VALUE)
        for _ in range(5):
            assert cache.get("k") == VALUE
