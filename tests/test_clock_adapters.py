"""Run the ClockPort battery against the system clock and the frozen clock."""
from __future__ import annotations

import pytest

from contracts.clock_contract import ClockContract
from knowbase.adapters.clock.system import FrozenClock, SystemClock


class TestSystemClock(ClockContract):
    @pytest.fixture
    def clock(self):
        return SystemClock()


class TestFrozenClock(ClockContract):
    @pytest.fixture
    def clock(self):
        return FrozenClock(start=1_700_000_000.0, step_s=1.0)

    def test_advances_by_exactly_one_step(self, clock):
        assert [clock.now() for _ in range(3)] == [1_700_000_000.0, 1_700_000_001.0,
                                                   1_700_000_002.0]

    def test_two_instances_produce_identical_sequences(self):
        a, b = FrozenClock(step_s=0.25), FrozenClock(step_s=0.25)
        assert [a.now() for _ in range(4)] == [b.now() for _ in range(4)]


def test_every_registered_clock_adapter_has_a_contract_subclass():
    from knowbase.adapters import CLOCK_ADAPTERS
    assert {cls.__name__ for cls in CLOCK_ADAPTERS.values()} == {"SystemClock", "FrozenClock"}
