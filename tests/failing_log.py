"""A log whose handler fails, for the tests of the stages around the job: the preflight (core.probe), the
encode (core.encode), the key (core.hls_key), the package (core.package) and the thumbnail (core.thumbnail).

Not a test module. A log handler that fails on a line does not raise: Python's logging prints a report
of its own to standard error, with every error that failure is chained to. A line written while an error
is still being handled chains its handler's failure to that error, so the report then holds the error's
own text: a path, an endpoint, a library's message. The `failing_log` fixture makes the handler of every
line those five modules log fail, and reported_log_failures() reads what logging then wrote.

tests/preflight/conftest.py and tests/encode/conftest.py import the fixture, so only the tests of those
two directories see it. (tests/job has one of its own, bound to the job's logger.)
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from core import encode, hls_key, package, probe, thumbnail

# The loggers of the five modules, each the module's own: none is made here, and none of their parents is
# touched, so a line of any other module (the job's among them) goes where it went.
STAGE_LOGGERS = (probe.logger, encode.logger, hls_key.logger, package.logger, thumbnail.logger)
# The text of the error a log line fails on when it cannot be written.
LOG_FAILURE_CANARY = "CANARY-quoted-by-a-failing-stage-log"
LOG_FAILURE_TEXT = f"no memory left for {LOG_FAILURE_CANARY}"


class _FailingStream(io.TextIOBase):
    """A standard error that fails every write."""

    def write(self, text: str) -> int:
        raise OSError(LOG_FAILURE_TEXT)


@pytest.fixture
def failing_log() -> Iterator[None]:
    """While the test runs, every line the five modules log also goes to a handler of the kind the
    process's log has (a stream handler), whose stream fails every write.

    Such a handler does not raise: Python's logging prints a report of its own to standard error
    (raiseExceptions, set here to its default), with every error the failure is chained to. A line
    written while an error is being handled chains its handler's failure to that error, so the report
    then holds the error's own text.

    Whatever the test does, the loggers are left with the handlers they had and raiseExceptions with
    the value it had: nothing else of the process's logging is touched (no level, no other logger)."""
    handler = logging.StreamHandler(_FailingStream())
    raise_exceptions_before = logging.raiseExceptions
    added_to: list[logging.Logger] = []
    try:
        logging.raiseExceptions = True
        for stage_logger in STAGE_LOGGERS:
            stage_logger.addHandler(handler)
            added_to.append(stage_logger)
        yield
    finally:
        for stage_logger in added_to:
            stage_logger.removeHandler(handler)
        handler.close()
        logging.raiseExceptions = raise_exceptions_before


def reported_log_failures(capsys: pytest.CaptureFixture[str], line_words: str) -> str:
    """What Python's logging wrote to standard error about its handler's failures, once that is shown to
    hold the report of a failure on the line that says `line_words` (a report quotes the line's message)."""
    written = capsys.readouterr().err
    assert LOG_FAILURE_CANARY in written, "logging reported no failure of its handler"
    assert line_words in written, "the handler did not fail on the line the test is about"
    return written
