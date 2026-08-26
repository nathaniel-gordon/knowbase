"""Contract battery for ClockPort."""
from __future__ import annotations

from datetime import datetime

import pytest

from kb.ports import ClockPort


class ClockContract:
    """Every ClockPort adapter must pass all of these.

    Subclasses provide a ``clock`` fixture.
    """

    @pytest.fixture
    def clock(self) -> ClockPort:
        raise NotImplementedError("contract subclass must provide a 'clock' fixture")

    def test_satisfies_the_protocol(self, clock):
        assert isinstance(clock, ClockPort)
        assert isinstance(clock.name, str) and clock.name == clock.name.lower()

    def test_now_returns_epoch_seconds(self, clock):
        value = clock.now()
        assert isinstance(value, float)
        assert value > 1_600_000_000.0, "epoch seconds, not a monotonic counter"

    def test_now_is_non_decreasing(self, clock):
        readings = [clock.now() for _ in range(5)]
        assert readings == sorted(readings)

    def test_timestamp_is_parseable_and_timezone_aware(self, clock):
        parsed = datetime.fromisoformat(clock.timestamp())
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() is not None

    def test_timestamp_agrees_with_now(self, clock):
        parsed = datetime.fromisoformat(clock.timestamp()).timestamp()
        assert parsed == pytest.approx(clock.now(), abs=1.0)
