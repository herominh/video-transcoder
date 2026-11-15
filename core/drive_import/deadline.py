"""What the Drive transfer answers at its stop line (core/net/clock.py StopLine): `deadline`, at the stage it is in.

A failure the transfer meets once the stop line has passed is the stop line's: it answers `deadline`.
"""

from __future__ import annotations

from core.drive_import.failure import Code, ImportFailed, Stage
from core.net.clock import StopLine


def require(stop_line: StopLine, stage: Stage) -> None:
    """Raise ImportFailed(deadline) once the stop line has passed: what was about to start does not."""
    if stop_line.passed():
        raise ImportFailed(Code.DEADLINE, stage)


def timeout(stop_line: StopLine, stage: Stage, connect_s: float, read_s: float) -> tuple[float, float]:
    """The (connect, read) time-outs of a request about to start, each cut to what is left of the stop line, so
    always above zero; ImportFailed(deadline) instead once the stop line has passed (the request does not start)."""
    cut = stop_line.cut(connect_s, read_s)
    if cut is None:
        raise ImportFailed(Code.DEADLINE, stage)
    return cut


def failed(stop_line: StopLine, code: Code, stage: Stage) -> ImportFailed:
    """The failure of a try that failed with `code`: `deadline` instead once the stop line has passed, since the
    watchdog cuts every request then and a try that ended after it ended because of it."""
    return ImportFailed(Code.DEADLINE if stop_line.passed() else code, stage)


def wait(stop_line: StopLine, seconds: float, stage: Stage) -> None:
    """Wait `seconds` before a request is made again; ImportFailed(deadline) when the wait would not end before the
    stop line (no wait is made that the job could not use)."""
    if not stop_line.wait(seconds):
        raise ImportFailed(Code.DEADLINE, stage)
