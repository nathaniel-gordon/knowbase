"""Clock adapters: real wall clock, and a frozen clock for reproducible runs."""
from __future__ import annotations

import time
from datetime import UTC, datetime


class SystemClock:
    """Driven adapter over the operating-system clock."""

    name = "system"

    def now(self) -> float:
        return time.time()

    def timestamp(self) -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")


class FrozenClock:
    """Deterministic clock that advances by a fixed step on every read.

    Used by the test fixtures and by ``--deterministic`` demo runs so that latency figures and
    ingest timestamps are byte-identical between runs while still being *computed*, not faked
    into constants.
    """

    name = "frozen"

    def __init__(self, start: float = 1_700_000_000.0, step_s: float = 0.001) -> None:
        self._t = float(start)
        self._step = float(step_s)

    def now(self) -> float:
        current = self._t
        self._t += self._step
        return current

    def timestamp(self) -> str:
        return datetime.fromtimestamp(self._t, tz=UTC).isoformat(timespec="milliseconds")
