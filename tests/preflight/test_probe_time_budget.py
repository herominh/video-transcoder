"""A preflight that ran out of its time: where the time went decides what it fails with.

No ffprobe runs here and no real time decides a test. The prober is a stand-in for run_bounded: it
makes the reads a test scripts of the loopback URL the preflight serves, then reports the ending the
test names. The reads take their time on a clock only the test moves, the one the preflight's budgeted
reader measures on. The execution's deadline and the preflight's own budget are deadlines the test
decides. Everything between is the real code: the loopback server, the budgeted reader, the typing.
(tests/preflight/test_probe.py has the same with the real ffprobe and a real clock.)
"""

from __future__ import annotations

import dataclasses
import functools
import http.client
import json
import logging
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from urllib.parse import urlsplit

import pytest

from core import probe as probe_module
from core.admission import RequestLimits
from core.bounded_process import BoundedResult, Deadline
from core.failure import Failure, WorkerFailure
from core.profile import PILOT_PROFILE
from core.source_reader import BudgetedSource

CHUNK_BYTES = 1024
SOURCE_BYTES = 16 * CHUNK_BYTES
PROFILE = dataclasses.replace(PILOT_PROFILE, probe=dataclasses.replace(PILOT_PROFILE.probe, chunk_bytes=CHUNK_BYTES))
OPEN_LIMITS = RequestLimits(max_source_bytes=SOURCE_BYTES, max_source_duration_ms=21_600_000)
LOOPBACK_URL_PREFIX = "http://127.0.0.1:"
# Where the test's clock starts: a monotonic clock does not read zero when a preflight begins.
CLOCK_STARTS_AT_S = 1_000.0
# What a decided deadline keeps where a Deadline keeps its instant, and what it says is left until it passes.
NEVER_S = 1.0e12
TIME_LEFT_S = 3_600.0
# How long a test waits for another thread to get somewhere, and a held read lets go by itself: far
# longer than a test takes, so that a regression fails a test instead of hanging the suite.
PATIENCE_S = 20.0
# The loopback server looks for its shutdown this often here, far more often than the worker's does.
SHUTDOWN_POLL_S = 0.005
# ffprobe's option for its own I/O time-out, in microseconds, and how much longer than what the
# preflight's budget leaves that time-out is: the worker's kill at the deadline comes first.
IO_TIMEOUT_OPTION = "-rw_timeout"
MICROSECONDS_PER_SECOND = 1_000_000
PROBER_IO_MARGIN_S = 5.0
# The two endings of the prober a test names: killed at the preflight's deadline, and the error exit
# ffprobe makes when the connection it reads from is cut.
TIMED_OUT = BoundedResult(
    returncode=None, stdout=b"", stderr_tail=b"", timed_out=True, stdout_overflow=False, stderr_truncated=False
)
EXITED_WITH_AN_ERROR = BoundedResult(
    returncode=1, stdout=b"", stderr_tail=b"", timed_out=False, stdout_overflow=False, stderr_truncated=False
)
# A third ending: ffprobe ended well, having printed what it prints of an ordinary MP4 of two seconds
# with one H.264 picture of 320x240 at 25 frames a second and one AAC sound.
PROBED_DOCUMENT = json.dumps(
    {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "2.000000"},
        "streams": [
            {
                "index": 0, "codec_name": "h264", "codec_type": "video", "width": 320, "height": 240,
                "sample_aspect_ratio": "1:1", "avg_frame_rate": "25/1", "r_frame_rate": "25/1",
                "field_order": "progressive", "color_transfer": "bt709", "pix_fmt": "yuv420p",
                "disposition": {"attached_pic": 0},
            },
            {
                "index": 1, "codec_name": "aac", "codec_type": "audio", "channels": 2,
                "disposition": {"attached_pic": 0},
            },
        ],
    }
).encode("ascii")
EXITED_WITH_THE_FACTS = BoundedResult(
    returncode=0, stdout=PROBED_DOCUMENT, stderr_tail=b"", timed_out=False, stdout_overflow=False,
    stderr_truncated=False,
)
READ_STALLED = Failure(
    error_class="source_unavailable", code="source_read_stalled", retryable=True,
    detail="a read of the source did not end within the time a read of its size may take",
)
SOURCE_CHANGED = Failure(
    error_class="source_integrity_mismatch", code="source_changed", retryable=False,
    detail="the object's ETag differs from the recorded one",
)
ACCESS_DENIED = Failure(
    error_class="source_unavailable", code="source_access_denied", retryable=False,
    detail="the storage refused the read",
)


class ReaderClock:
    """The monotonic clock the preflight's budgeted reader measures on, moved only by the test."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now_s = CLOCK_STARTS_AT_S

    def __call__(self) -> float:
        with self._lock:
            return self._now_s

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now_s += seconds


class DecidedDeadline(Deadline):
    """A Deadline that has passed once the test says so, whatever any clock reads. It is a Deadline to
    every isinstance check."""

    def __init__(self) -> None:
        super().__init__(NEVER_S)
        object.__setattr__(self, "_passed", False)

    def pass_now(self) -> None:
        object.__setattr__(self, "_passed", True)

    def expired(self) -> bool:
        return self._passed

    def remaining_s(self) -> float:
        return 0.0 if self._passed else TIME_LEFT_S


class ExecutionDeadline(DecidedDeadline):
    """The execution's own deadline, as the job runner hands it to the preflight. What the preflight
    makes of it for its own time budget is `budget`, which the test decides as well."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "budget", DecidedDeadline())

    def within_ms(self, ms: int) -> Deadline:
        return self.budget


class TimedSource:
    """A range source of zeros whose reads take time on the test's clock and nowhere else.

    Read number n (the first is 1) takes `read_seconds[n - 1]`, a read after the last none. The read
    numbered `held_read` takes no time of its own: it sets `held_read_entered` and waits until the
    test sets `release`, so the time it is in flight is what the test moves the clock by meanwhile.
    The read numbered `fails_at` raises `failure` once its time has passed, or once it is let go.
    """

    def __init__(
        self,
        clock: ReaderClock,
        read_seconds: Sequence[float] = (),
        *,
        held_read: int | None = None,
        fails_at: int | None = None,
        failure: Failure | None = None,
    ) -> None:
        self._clock = clock
        self._read_seconds = tuple(read_seconds)
        self._held_read = held_read
        self._fails_at = fails_at
        self._failure = failure
        self._count_lock = threading.Lock()
        self.reads = 0
        self.held_read_entered = threading.Event()
        self.release = threading.Event()

    @property
    def size_bytes(self) -> int:
        return SOURCE_BYTES

    def read(self, start: int, length: int) -> bytes:
        with self._count_lock:
            self.reads += 1
            number = self.reads
        if number == self._held_read:
            self.held_read_entered.set()
            self.release.wait(PATIENCE_S)
        elif number <= len(self._read_seconds):
            self._clock.advance(self._read_seconds[number - 1])
        if number == self._fails_at and self._failure is not None:
            raise WorkerFailure(self._failure)
        return bytes(length)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> ReaderClock:
    """The clock of the preflight's budgeted reader, in the test's hands."""
    reader_clock = ReaderClock()
    on_the_tests_clock = functools.partial(BudgetedSource, monotonic=reader_clock)
    monkeypatch.setattr(probe_module, "BudgetedSource", on_the_tests_clock)
    return reader_clock


@pytest.fixture(autouse=True)
def quick_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preflight's loopback server notices its shutdown within milliseconds: a test waits for nothing."""
    serve_forever = probe_module._LoopbackHTTPServer.serve_forever
    monkeypatch.setattr(
        probe_module._LoopbackHTTPServer, "serve_forever",
        lambda server: serve_forever(server, poll_interval=SHUTDOWN_POLL_S),
    )


@pytest.fixture
def held_reads() -> Iterator[list[TimedSource]]:
    """The sources of a test that hold a read: each is released when the test ends, and the threads the
    test started are then waited for, so that none outlives its test."""
    threads_before = set(threading.enumerate())
    sources: list[TimedSource] = []
    yield sources
    for source in sources:
        source.release.set()
    for thread in set(threading.enumerate()) - threads_before:
        thread.join(PATIENCE_S)


def _read_chunk(url: str, index: int) -> None:
    """One GET of one chunk of the loopback URL, as the prober makes them. A connection the server cuts
    is how a prober learns of a read that failed, so it is no error here."""
    address = urlsplit(url)
    first = index * CHUNK_BYTES
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=PATIENCE_S)
    try:
        connection.request("GET", address.path, headers={"Range": f"bytes={first}-{first + CHUNK_BYTES - 1}"})
        connection.getresponse().read()
    except (OSError, http.client.HTTPException):
        return
    finally:
        connection.close()


def _begin_reading_chunk(url: str, index: int) -> threading.Thread:
    """The same GET on a thread of its own, for a read the test holds in flight; the thread ends once
    the server has answered the GET or cut its connection."""
    thread = threading.Thread(target=_read_chunk, args=(url, index), daemon=True)
    thread.start()
    return thread


def _end_the_held_read_while_the_server_shuts_down(
    monkeypatch: pytest.MonkeyPatch, source: TimedSource, in_flight: list[threading.Thread]
) -> None:
    """The held read of `source` ends while the preflight's loopback server shuts down: it is let go
    as the shutdown begins, and the shutdown goes on only once every GET of `in_flight` is over, by
    when the server has recorded whatever the read raised."""
    shutdown = probe_module._LoopbackHTTPServer.shutdown

    def shutdown_while_the_read_ends(server: Any) -> None:
        source.release.set()
        for thread in in_flight:
            thread.join(PATIENCE_S)
        shutdown(server)

    monkeypatch.setattr(probe_module._LoopbackHTTPServer, "shutdown", shutdown_while_the_read_ends)


def _prober_that(run: Callable[[str], BoundedResult]) -> Callable[..., BoundedResult]:
    """A stand-in for run_bounded: `run` is handed the URL ffprobe was to read and returns how it ended."""

    def run_bounded(argv: Sequence[str], **limits: Any) -> BoundedResult:
        return run(next(argument for argument in argv if argument.startswith(LOOPBACK_URL_PREFIX)))

    return run_bounded


def _preflight(source: TimedSource, deadline: Deadline) -> probe_module.PreflightResult:
    # The prober's path only has to name a program that exists: the stand-in starts none.
    return probe_module.run_preflight(
        source, profile=PROFILE, limits=OPEN_LIMITS, deadline=deadline, ffprobe_path=sys.executable
    )


def _failure_of(source: TimedSource, deadline: Deadline) -> Failure:
    with pytest.raises(WorkerFailure) as raised:
        _preflight(source, deadline)
    return raised.value.failure


def _typed(failure: Failure) -> tuple[str, str, bool]:
    return failure.error_class, failure.code, failure.retryable


def _execution_whose_budget_has_passed() -> ExecutionDeadline:
    execution = ExecutionDeadline()
    execution.budget.pass_now()
    return execution


def _io_timeout_us(argv: Sequence[str]) -> int:
    """ffprobe's own I/O time-out in microseconds, as its command line names it."""
    return int(argv[argv.index(IO_TIMEOUT_OPTION) + 1])


def _probe_log(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    return [(record.levelno, record.getMessage()) for record in caplog.records if record.name == "core.probe"]


# --- the prober timed out ------------------------------------------------------------------------------


def test_preflight_when_its_budget_runs_out_with_a_slow_read_in_flight_should_fail_retryable_as_a_slow_source(
    clock, monkeypatch, held_reads
):
    # Arrange: two reads of 10 s each, 20 s of ffprobe's own work, then a third read that is 20 s in
    # flight when ffprobe is killed: 40 s in reads of 60 s, and only 20 s without the read in flight
    source = TimedSource(clock, [10.0, 10.0], held_read=3)
    held_reads.append(source)

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(20.0)
        _begin_reading_chunk(url, 2)
        assert source.held_read_entered.wait(PATIENCE_S)
        clock.advance(20.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert
    assert _typed(failure) == ("source_unavailable", "probe_source_slow", True)
    assert failure.detail == "the preflight ran out of time while the storage was slow"
    assert source.reads == 3


def test_preflight_when_its_budget_runs_out_after_slow_reads_should_fail_retryable_as_a_slow_source(
    clock, monkeypatch
):
    # Arrange: three reads of 15 s, 15 s and 10 s, none of them in flight when ffprobe is killed after
    # 20 s of work of its own: 40 s in reads of 60 s
    source = TimedSource(clock, [15.0, 15.0, 10.0])

    def ffprobe_run(url: str) -> BoundedResult:
        for index in range(3):
            _read_chunk(url, index)
        clock.advance(20.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert
    assert _typed(failure) == ("source_unavailable", "probe_source_slow", True)
    assert failure.detail == "the preflight ran out of time while the storage was slow"


@pytest.mark.parametrize(
    ("read_seconds", "ffprobe_seconds", "logged_seconds"),
    [
        pytest.param([10.0, 10.0], 40.0, "20.000 s in reads of 60.000 s elapsed", id="reads-took-a-third"),
        pytest.param([], 60.0, "0.000 s in reads of 60.000 s elapsed", id="no-read-at-all"),
        pytest.param([], 0.0, "0.000 s in reads of 0.000 s elapsed", id="no-read-and-no-time-seen-to-pass"),
    ],
)
def test_preflight_when_its_budget_runs_out_on_ffprobes_own_work_should_fail_for_good(
    clock, monkeypatch, caplog, read_seconds, ffprobe_seconds, logged_seconds
):
    # Arrange: ffprobe is killed after work of its own that took more than the reads did
    caplog.set_level(logging.DEBUG, logger="core.probe")
    source = TimedSource(clock, read_seconds)

    def ffprobe_run(url: str) -> BoundedResult:
        for index in range(len(read_seconds)):
            _read_chunk(url, index)
        clock.advance(ffprobe_seconds)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert: final, and one warning says where the time went and that the storage was not the slow
    # party, in our own words and the two numbers, with nothing of the URL the prober read
    assert _typed(failure) == ("deadline_exceeded", "probe_deadline_exceeded", False)
    assert failure.detail == "the preflight ran out of time"
    assert _probe_log(caplog) == [
        (
            logging.WARNING,
            f"the preflight ran out of time for good, the storage was not the slow party: {logged_seconds}",
        )
    ]


@pytest.mark.parametrize(
    ("read_seconds", "ffprobe_seconds", "expected"),
    [
        pytest.param(30.0, 30.0, ("source_unavailable", "probe_source_slow", True), id="exactly-half"),
        pytest.param(29.999, 30.001, ("deadline_exceeded", "probe_deadline_exceeded", False), id="just-under-half"),
        pytest.param(30.001, 29.999, ("source_unavailable", "probe_source_slow", True), id="just-over-half"),
    ],
)
def test_preflight_when_its_budget_runs_out_should_take_half_of_the_time_in_reads_for_a_slow_source(
    clock, monkeypatch, read_seconds, ffprobe_seconds, expected
):
    # Arrange: one read, then ffprobe's own work until it is killed, 60 s after the budget began
    source = TimedSource(clock, [read_seconds])

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        clock.advance(ffprobe_seconds)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert
    assert _typed(failure) == expected


@pytest.mark.parametrize(
    ("read_seconds", "ffprobe_seconds", "logged_seconds"),
    [
        pytest.param([25.0, 25.0], 10.0, "50.000 s in reads of 60.000 s elapsed", id="reads-took-five-sixths"),
        pytest.param([5.0, 5.0], 50.0, "10.000 s in reads of 60.000 s elapsed", id="reads-took-a-sixth"),
    ],
)
def test_preflight_when_the_executions_deadline_has_passed_too_should_fail_for_good_whatever_the_reads_took(
    clock, monkeypatch, caplog, read_seconds, ffprobe_seconds, logged_seconds
):
    # Arrange: reads that took most of the 60 s, or little of them, and the job's own wall time gone by
    # the time ffprobe is killed
    caplog.set_level(logging.DEBUG, logger="core.probe")
    source = TimedSource(clock, read_seconds)
    execution = ExecutionDeadline()

    def ffprobe_run(url: str) -> BoundedResult:
        for index in range(len(read_seconds)):
            _read_chunk(url, index)
        clock.advance(ffprobe_seconds)
        execution.pass_now()
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, execution)

    # Assert: final, and one warning names the execution's own deadline as what made it final, with
    # the two numbers: where they alone would have made a slow storage of it, and where the storage
    # was not the slow party either
    assert _typed(failure) == ("deadline_exceeded", "probe_deadline_exceeded", False)
    assert failure.detail == "the preflight ran out of time"
    assert _probe_log(caplog) == [
        (
            logging.WARNING,
            f"the preflight ran out of time for good, the execution's own deadline had passed: {logged_seconds}",
        )
    ]


# --- the budgeted reader refused a read past the time budget --------------------------------------------


def test_preflight_when_the_reader_refuses_a_read_past_the_budget_after_slow_reads_should_fail_retryable(
    clock, monkeypatch
):
    # Arrange: two reads of 20 s each and 20 s of ffprobe's own work use the budget up; the reader then
    # refuses the third read, and ffprobe ends on its cut connection, not on a time-out
    source = TimedSource(clock, [20.0, 20.0])
    execution = ExecutionDeadline()

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(20.0)
        execution.budget.pass_now()
        _read_chunk(url, 2)
        return EXITED_WITH_AN_ERROR

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, execution)

    # Assert: the storage was never asked for the third read
    assert _typed(failure) == ("source_unavailable", "probe_source_slow", True)
    assert failure.detail == "the preflight ran out of time while the storage was slow"
    assert source.reads == 2


def test_preflight_when_the_reader_refuses_a_read_past_the_budget_after_quick_reads_should_fail_for_good(
    clock, monkeypatch
):
    # Arrange: two reads of 5 s each, 50 s of ffprobe's own work, and then the refused third read
    source = TimedSource(clock, [5.0, 5.0])
    execution = ExecutionDeadline()

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(50.0)
        execution.budget.pass_now()
        _read_chunk(url, 2)
        return EXITED_WITH_AN_ERROR

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, execution)

    # Assert
    assert _typed(failure) == ("deadline_exceeded", "probe_deadline_exceeded", False)
    assert failure.detail == "the preflight ran out of time"
    assert source.reads == 2


def test_preflight_when_the_reader_refuses_a_read_after_the_executions_deadline_passed_should_fail_for_good(
    clock, monkeypatch
):
    # Arrange: slow reads that took 40 s of 60 s, but it is the job's wall time that ran out: the
    # execution's deadline has passed, and with it the preflight's budget, which is never the later one
    source = TimedSource(clock, [20.0, 20.0])
    execution = ExecutionDeadline()

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(20.0)
        execution.pass_now()
        execution.budget.pass_now()
        _read_chunk(url, 2)
        return EXITED_WITH_AN_ERROR

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, execution)

    # Assert
    assert _typed(failure) == ("deadline_exceeded", "probe_deadline_exceeded", False)


def test_preflight_when_the_reader_refused_a_read_past_the_budget_should_ask_the_storage_for_nothing_more(
    clock, monkeypatch
):
    # Arrange: two slow reads and the budget used up, as above. The storage would answer its next read
    # with a changed source, and ffprobe asks for two more chunks before it gives up
    source = TimedSource(clock, [20.0, 20.0], fails_at=3, failure=SOURCE_CHANGED)
    execution = ExecutionDeadline()

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(20.0)
        execution.budget.pass_now()
        _read_chunk(url, 2)
        _read_chunk(url, 3)
        return EXITED_WITH_AN_ERROR

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, execution)

    # Assert: neither chunk reached the storage, so no failure of a read can follow the reader's
    # refusal, and the time alone decides
    assert _typed(failure) == ("source_unavailable", "probe_source_slow", True)
    assert source.reads == 2


# --- a failure a read raised still wins -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("read_failure", "expected"),
    [
        pytest.param(READ_STALLED, ("source_unavailable", "source_read_stalled", True), id="a-stalled-read"),
        pytest.param(SOURCE_CHANGED, ("source_integrity_mismatch", "source_changed", False), id="a-changed-source"),
        pytest.param(ACCESS_DENIED, ("source_unavailable", "source_access_denied", False), id="a-refused-read"),
    ],
)
def test_preflight_when_a_slow_read_failed_before_its_budget_ran_out_should_fail_with_the_reads_own_failure(
    clock, monkeypatch, caplog, read_failure, expected
):
    # Arrange: a read of 10 s, then one that fails after 35 s; ffprobe goes on until it is killed, so
    # the reads took 45 s of 60 s, which alone would make a slow source of it
    caplog.set_level(logging.DEBUG, logger="core.probe")
    source = TimedSource(clock, [10.0, 35.0], fails_at=2, failure=read_failure)

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        _read_chunk(url, 1)
        clock.advance(15.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert: the read's failure as it was raised, retryable or not, and no word of a slow storage
    assert _typed(failure) == expected
    assert failure.detail == read_failure.detail
    assert _probe_log(caplog) == []


def test_preflight_when_the_request_budget_was_passed_before_its_time_ran_out_should_fail_on_the_request_budget(
    clock, monkeypatch
):
    # Arrange: two requests allowed, both slow; the third is refused for the budget of requests, and
    # ffprobe goes on until it is killed, with 50 s in reads of 60 s
    profile = dataclasses.replace(PROFILE, probe=dataclasses.replace(PROFILE.probe, max_requests=2))
    source = TimedSource(clock, [25.0, 25.0])

    def ffprobe_run(url: str) -> BoundedResult:
        for index in range(3):
            _read_chunk(url, index)
        clock.advance(10.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        probe_module.run_preflight(
            source, profile=profile, limits=OPEN_LIMITS, deadline=ExecutionDeadline(), ffprobe_path=sys.executable
        )

    # Assert
    assert _typed(raised.value.failure) == ("input_limits_exceeded", "probe_budget_exceeded", False)
    assert source.reads == 2


def test_preflight_when_a_read_in_flight_fails_for_good_while_the_server_shuts_down_should_fail_with_that_failure(
    clock, monkeypatch, caplog, held_reads
):
    # Arrange: a read of 10 s, 10 s of ffprobe's own work, then a read that is 40 s in flight when
    # ffprobe is killed: 50 s in reads of 60 s. The storage answers that read only while the loopback
    # server shuts down, and its answer is that the source has changed
    caplog.set_level(logging.DEBUG, logger="core.probe")
    source = TimedSource(clock, [10.0], held_read=2, fails_at=2, failure=SOURCE_CHANGED)
    held_reads.append(source)
    in_flight: list[threading.Thread] = []
    _end_the_held_read_while_the_server_shuts_down(monkeypatch, source, in_flight)

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        clock.advance(10.0)
        in_flight.append(_begin_reading_chunk(url, 1))
        assert source.held_read_entered.wait(PATIENCE_S)
        clock.advance(40.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert: the read's own failure, final, although the reads took most of the time
    assert _typed(failure) == ("source_integrity_mismatch", "source_changed", False)
    assert failure.detail == "the object's ETag differs from the recorded one"
    assert _probe_log(caplog) == []


def test_preflight_when_a_read_in_flight_ends_well_while_the_server_shuts_down_should_fail_retryable_as_a_slow_source(
    clock, monkeypatch, held_reads
):
    # Arrange: the same sequence, but the read that was 40 s in flight returns its bytes while the
    # loopback server shuts down: nothing but the time speaks against the storage
    source = TimedSource(clock, [10.0], held_read=2)
    held_reads.append(source)
    in_flight: list[threading.Thread] = []
    _end_the_held_read_while_the_server_shuts_down(monkeypatch, source, in_flight)

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        clock.advance(10.0)
        in_flight.append(_begin_reading_chunk(url, 1))
        assert source.held_read_entered.wait(PATIENCE_S)
        clock.advance(40.0)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert
    assert _typed(failure) == ("source_unavailable", "probe_source_slow", True)
    assert failure.detail == "the preflight ran out of time while the storage was slow"
    assert source.reads == 2


def test_preflight_when_a_read_fails_for_good_only_after_the_prober_succeeded_should_return_its_facts_and_decision(
    clock, monkeypatch, held_reads
):
    # Arrange: ffprobe has all it needs after one read and ends well, while a read it no longer needs
    # is still in flight. The storage answers that read only while the loopback server shuts down, and
    # its answer is that the source has changed
    source = TimedSource(clock, [10.0], held_read=2, fails_at=2, failure=SOURCE_CHANGED)
    held_reads.append(source)
    in_flight: list[threading.Thread] = []
    _end_the_held_read_while_the_server_shuts_down(monkeypatch, source, in_flight)

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        in_flight.append(_begin_reading_chunk(url, 1))
        assert source.held_read_entered.wait(PATIENCE_S)
        clock.advance(40.0)
        return EXITED_WITH_THE_FACTS

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    result = _preflight(source, ExecutionDeadline())

    # Assert: the probe that succeeded stands, although the second read reached the storage and had
    # failed before the preflight returned
    video = result.facts.video_streams[0]
    assert (video.display_width, video.display_height) == (320, 240)
    assert result.facts.duration_ms == 2000
    assert result.decision.admissible
    assert source.reads == 2
    assert in_flight and not in_flight[0].is_alive()


# --- the loopback server after a read recorded a failure ------------------------------------------------


def test_loopback_server_when_a_read_recorded_a_failure_should_read_nothing_more_of_the_source():
    # Arrange: the second read of the source fails with a changed source. The storage would answer a
    # third read with bytes, and the budgeted reader would let it through: its deadline never passes
    # here, and its budgets cover the whole source
    source = TimedSource(ReaderClock(), fails_at=2, failure=SOURCE_CHANGED)
    reader = BudgetedSource(
        source, max_bytes=SOURCE_BYTES, max_requests=SOURCE_BYTES // CHUNK_BYTES, deadline=DecidedDeadline()
    )
    server = probe_module.LoopbackRangeServer(reader, chunk_bytes=CHUNK_BYTES)

    # Act: a chunk that is read, the chunk whose read fails, then a request for a chunk not read yet
    with server:
        _read_chunk(server.url, 0)
        _read_chunk(server.url, 1)
        recorded = server.failure
        _read_chunk(server.url, 2)

    # Assert: the request that followed the recorded failure reached the source with no read
    assert recorded == SOURCE_CHANGED
    assert source.reads == 2


# --- the facts are asked of a prober that timed out -----------------------------------------------------


@pytest.mark.parametrize(
    "timed_out_result",
    [
        pytest.param(TIMED_OUT, id="nothing-else-wrong"),
        pytest.param(dataclasses.replace(TIMED_OUT, stdout_overflow=True), id="its-output-over-the-cap-too"),
    ],
)
def test_facts_from_when_the_prober_timed_out_should_fail_for_good_as_a_preflight_out_of_time(timed_out_result):
    # Act: the facts are asked of a prober the worker killed at the deadline, which run_preflight never does
    with pytest.raises(WorkerFailure) as raised:
        probe_module._facts_from(timed_out_result)

    # Assert: the final ending of a preflight out of time, before anything else the result holds, and
    # never the retryable one of a prober that crashed
    assert _typed(raised.value.failure) == ("deadline_exceeded", "probe_deadline_exceeded", False)
    assert raised.value.failure.detail == "the preflight ran out of time"


# --- ffprobe's own time-out stays behind the worker's kill ----------------------------------------------


@pytest.mark.parametrize(
    ("make_execution", "seconds_left"),
    [
        pytest.param(ExecutionDeadline, TIME_LEFT_S, id="an-hour-left"),
        pytest.param(_execution_whose_budget_has_passed, 0.0, id="none-left"),
    ],
)
def test_preflight_when_it_starts_ffprobe_should_set_its_io_time_out_seconds_past_what_the_budget_leaves(
    clock, monkeypatch, make_execution, seconds_left
):
    # Arrange: a budget that leaves a known time, and a prober that only keeps its command line
    command_lines: list[Sequence[str]] = []

    def run_bounded(argv: Sequence[str], **limits: Any) -> BoundedResult:
        command_lines.append(argv)
        return EXITED_WITH_THE_FACTS

    monkeypatch.setattr(probe_module, "run_bounded", run_bounded)

    # Act
    _preflight(TimedSource(clock), make_execution())

    # Assert: ffprobe would give up a read by itself only seconds after the worker has killed it
    assert [_io_timeout_us(argv) for argv in command_lines] == [
        (seconds_left + PROBER_IO_MARGIN_S) * MICROSECONDS_PER_SECOND
    ]


# --- what is logged ------------------------------------------------------------------------------------


def test_preflight_when_it_fails_as_a_slow_source_should_log_one_warning_with_the_seconds_in_reads_and_elapsed(
    clock, monkeypatch, caplog
):
    # Arrange: 40.5 s in reads of 60.75 s
    caplog.set_level(logging.DEBUG, logger="core.probe")
    source = TimedSource(clock, [40.5])

    def ffprobe_run(url: str) -> BoundedResult:
        _read_chunk(url, 0)
        clock.advance(20.25)
        return TIMED_OUT

    monkeypatch.setattr(probe_module, "run_bounded", _prober_that(ffprobe_run))

    # Act
    failure = _failure_of(source, ExecutionDeadline())

    # Assert: our own words and the two numbers, nothing of the URL the prober read
    assert failure.code == "probe_source_slow"
    assert _probe_log(caplog) == [
        (
            logging.WARNING,
            "the preflight ran out of time while the storage was slow: 40.500 s in reads of 60.750 s elapsed",
        )
    ]
