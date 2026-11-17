"""A clock that moves only when the code under test waits or the test moves it: what a stop line (core/net/clock.py
StopLine) is read on in the tests of the shared session (tests/net/) and of the Drive acquire (tests/acquire/,
tests/drive_import/).

Not a test module. It imports nothing of either runtime.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

START = datetime(2026, 10, 17, 8, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    """now() and monotonic() move only when the transfer sleeps or the test advances them; sleeps are recorded."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start
        self._monotonic = 1_000.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds
