"""The import job's time: the clock it is handed, and its stop line.

The stop line is the writer deadline less a margin (`writes_end_at` − 120 s; the Hub's import design, G-7): no
request starts and no wait begins once it has passed, every request's time-outs are cut to what is left of it, a
request body stops giving bytes when it passes (BoundedBody, core/drive_import/transport.py), and the session's
watchdog, when it comes, shuts every connection of the job's requests (a TLS handshake under way included) and
latches the session's expiry, so that a read or a write blocked on one ends then, and a connection whose name
resolution or TCP connect straddled the stop line goes no further once they return (transport.RequestsSession;
what nothing cuts short is the name resolution itself, which no time-out bounds and which has no socket to shut).
A failure the job meets once the stop line has passed is the stop line's: the job answers `deadline`. The stop line
is held on the monotonic clock, so that a step of the wall clock during the job moves it neither way.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Protocol

from core.drive_import.failure import Code, ImportFailed, Stage

# How long before the writer deadline the job stops starting requests and ends with `deadline`.
STOP_MARGIN = timedelta(seconds=120)


class Clock(Protocol):
    """What the job reads the time from and waits with."""

    def now(self) -> datetime:
        """The wall-clock time, aware and in UTC."""

    def monotonic(self) -> float:
        """Seconds on a clock that never steps back."""

    def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""


class SystemClock:
    """The process's own clocks."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class DeadlineReached(Exception):
    """The stop line passed during a request, raised where the request stops: by a request body that had bytes left
    to send (transport.BoundedBody), by a connection of the real session whose setup the stop line overtook (after
    its TCP connect, after its TLS handshake, or before an address was tried), and by the real session itself for a
    request asked once it expired (transport.RequestsSession.get() and put())."""


class StopLine:
    """The instant, on the monotonic clock, after which the job starts nothing more."""

    def __init__(self, clock: Clock, at_monotonic: float) -> None:
        self._clock = clock
        self._at = at_monotonic

    @classmethod
    def before(cls, writes_end_at: datetime, clock: Clock) -> StopLine:
        """The stop line of a job whose writer deadline is `writes_end_at` (aware)."""
        seconds_left = (writes_end_at - STOP_MARGIN - clock.now()).total_seconds()
        return cls(clock, clock.monotonic() + seconds_left)

    def left_s(self) -> float:
        """Seconds until the stop line; zero or less once it has passed."""
        return self._at - self._clock.monotonic()

    def passed(self) -> bool:
        return self.left_s() <= 0

    def require(self, stage: Stage) -> None:
        """Raise ImportFailed(deadline) once the stop line has passed: what was about to start does not."""
        if self.passed():
            raise ImportFailed(Code.DEADLINE, stage)

    def timeout(self, stage: Stage, connect_s: float, read_s: float) -> tuple[float, float]:
        """The (connect, read) time-outs of a request about to start, each cut to what is left of the stop line, so
        always above zero; ImportFailed(deadline) instead once the stop line has passed (the request does not start).

        A time-out bounds one wait of the socket (a connection, a silence), not a request's whole time: that is the
        session's watchdog's, which shuts the request's connection at the stop line.
        """
        left = self.left_s()
        if left <= 0:
            raise ImportFailed(Code.DEADLINE, stage)
        return (min(connect_s, left), min(read_s, left))

    def failed(self, code: Code, stage: Stage) -> ImportFailed:
        """The failure of a try that failed with `code`: `deadline` instead once the stop line has passed, since the
        watchdog cuts every request then and a try that ended after it ended because of it."""
        return ImportFailed(Code.DEADLINE if self.passed() else code, stage)

    def wait(self, seconds: float, stage: Stage) -> None:
        """Wait `seconds` before a request is made again; ImportFailed(deadline) when the wait would not end
        before the stop line (no wait is made that the job could not use)."""
        if seconds >= self.left_s():
            raise ImportFailed(Code.DEADLINE, stage)
        self._clock.sleep(seconds)
