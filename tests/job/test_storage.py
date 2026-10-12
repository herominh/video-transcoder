"""The two ways the job touches a bucket.

download_source() copies the whole source through the real S3RangeSource (so through its conditional,
checked range reads) against the fake S3 client of conftest. upload_generation() stores exactly the
files it is told, each under the granted prefix, the manifest last, and refuses anything that would put
an object elsewhere before it sends a single request.

A request that "never returns" is one a test double keeps until the test lets it go (_Hold). Two tests
keep a real, short deadline (100 ms and 500 ms); no other needs a thread to get anywhere within a time
limit, so no other depends on how loaded the host is: where a deadline is to pass under a kept request,
it starts to run out only once that request is kept (_PassedSoonAfter), and the job starts to wait for
a request only once the request has ended or is kept (the `requests_settle_first` fixture). A test
asserts how the job ended and only a generous bound on how long that took.

No test waits between two attempts for real: every transfer is handed a stand-in for `sleep` that
records the waits it is asked for (_Waits) and takes no time. Where a test states its waits exactly,
the jitter's draw is replaced by one fixed factor (the `fixed_factor` fixture). No test waits out the
half minute and more a request may take either: where a request must outlast its ceiling, the ceiling
is made tens of milliseconds long, and only for requests the test keeps (the `short_ceiling` fixture,
_StalledFirst), so that no other request can stall on a loaded host.

The last section sends an upload through the real boto3 client that upload_client() makes: a hook on
the client answers every prepared request before it leaves the process, so no connection is opened.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import stat
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    HTTPClientError,
    ParamValidationError,
    ReadTimeoutError,
)

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import Failure, WorkerFailure
from core.job import storage
from core.job.storage import DeadlineSource, download_source, storage_client, upload_client, upload_generation
from core.package import ManifestOutput
from core.protocol.settings import StorageProfile
from core.source_reader import S3RangeSource
from tests.contract.v2_support import FakeClock
from tests.job.conftest import (
    MANIFEST_NAME,
    ClockDeadline,
    FakeS3,
    PutCall,
    ScriptedDeadline,
    client_error,
    reported_log_failures,
    settle_requests_first,
)

BUCKET = "test-source-bucket"
KEY = "test/sources/source-0001.mp4"
ETAG = "test-etag-0001"
OTHER_ETAG = "test-etag-0002"
CHUNK_BYTES = 1000
# Three whole chunks and a shorter fourth.
OBJECT = bytes(range(256)) * 13 + b"tail"
GENEROUS_MS = 60_000
# Long enough for the thread that waits for a request to be waiting before the request ends.
A_MOMENT_S = 0.05
# How long a deadline still runs once the request it is to pass under is where the test keeps it
# (_PassedSoonAfter). The job waits it out for real, and nothing has to happen meanwhile.
PASSES_AFTER_S = 0.05
# Farther off than any wait can last (threading.TIMEOUT_MAX is about 292 years).
NEVER_S = 1.0e12
# How long a kept request stays at most when no test lets it go: a test that would hang fails instead.
KEPT_AT_MOST_S = 20.0
# The generous bound: on how long a job may take to give up at a short deadline, and on how long a
# request that was let go may take to end.
PROMPT_S = 10.0
# The factor every wait is multiplied by where a test states its waits exactly, in place of the jitter's
# draw. The nominal waits of 0.5, 2 and 8 s times it are 0.375, 1.5 and 6 s, each exact in binary.
FIXED_FACTOR = 0.75
# The factor every wait is multiplied by where a test measures its waits against the time a deadline has
# left. That time is also how long the job waits, for real, for a request's thread to end (_within), so
# such a test gives it tens of seconds: more than a loaded host delays a thread. The nominal waits of
# 0.5, 2 and 8 s times this factor are 40, 160 and 640 s, each exact in binary, and none is waited out.
LONG_WAITS_FACTOR = 80.0
# A request ceiling a test waits out, where the worker's own is half a minute and more; and one no
# request of a test reaches.
SHORT_CEILING_S = 0.05
GENEROUS_CEILING_S = 60.0
HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300
HTTP_MOVED_PERMANENTLY = 301
HTTP_FOUND = 302
HTTP_TEMPORARY_REDIRECT = 307
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_REQUEST_TIMEOUT = 408
HTTP_CONFLICT = 409
HTTP_TOO_MANY_REQUESTS = 429
HTTP_LAST_CLIENT_ERROR = 499
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
OUTPUT_BUCKET = "test-output-bucket"
PREFIX = "test/media/22222222-2222-7222-8222-222222222222/55555555-5555-7555-8555-555555555555"
ENDPOINT_CANARY = "https://account-canary.r2.example.invalid"
SECRET_CANARY = "SECRETcanary0123456789"
ERROR_TEXT_CANARY = f"{SECRET_CANARY} at {ENDPOINT_CANARY}"
# A local path as the operating system's own error text could name it: what no failure and no log line may hold.
PATH_CANARY = "/CANARY/scratch/output/of/another/job"
UNREACHABLE = EndpointConnectionError(endpoint_url=f"{ENDPOINT_CANARY}/{OUTPUT_BUCKET}/{PREFIX}/master.m3u8")
ACCESS_DENIED = client_error("AccessDenied", HTTP_FORBIDDEN, "PutObject", ERROR_TEXT_CANARY)
OWNER_ONLY = 0o600
# Every byte value mapped to another one: a file given through it keeps its length and none of its bytes.
OTHER_BYTES = bytes(value ^ 0xFF for value in range(256))
# A segment a client cannot hash, or send, in one read, and no file buffer can hold: read a second time,
# it comes from the disk again.
LARGE_SEGMENT_PATH = "360p/segment_0002.ts"
LARGE_SEGMENT = b"\x47" + bytes(range(256)) * (3 * 4096) + b"the end of a large segment"
# How much a client that reads a body a piece at a time asks for: of the small files, and of the large one.
SMALL_PIECE_BYTES = 7
LARGE_PIECE_BYTES = 64 * 1024
# Where macOS and Linux list the descriptors a process holds open.
OPEN_DESCRIPTORS = "/dev/fd"
# What tells botocore to measure a body by its end before it sends it (it then rewinds it).
BOTOCORE_MEASURES_BODIES = "BOTO_EXPERIMENTAL__NO_EMPTY_CONTINUE"
ARTIFACT_KIND_BY_SUFFIX = {
    ".m3u8": "hls_media_playlist",
    ".ts": "hls_segment",
    ".jpg": "thumbnail",
}


def _failure_of(call: Callable[[], Any]) -> Failure:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value.failure


def _deadline() -> Deadline:
    return Deadline.after_ms(GENEROUS_MS)


def _passed_deadline() -> Deadline:
    return Deadline(time.monotonic() - 1)


class _Arrival(threading.Event):
    """An event a request sets once it is where a test wants it, which notes when that was: `at_s` is
    the monotonic clock's reading at the first set(), and None until then."""

    def __init__(self) -> None:
        super().__init__()
        self.at_s: float | None = None

    def set(self) -> None:
        if self.at_s is None:
            self.at_s = time.monotonic()
        super().set()


class _Hold:
    """Where a test double keeps a request that never returns: until the test lets it go.

    release() lets every kept request go and waits for every thread started since this object was made
    to end. So no thread outlives its test, and whatever an abandoned request did once it was let go has
    happened by the time release() returns. The fixture calls it when the test ends.

    `entered` is set once a request is kept. A test asserts it: where the job gave up without its
    request ever having been kept, the test would pass without its subject.
    """

    def __init__(self) -> None:
        self._let_go = threading.Event()
        self._threads_before = set(threading.enumerate())
        self.entered = _Arrival()

    def keep(self) -> None:
        """Called by a double inside a request, on the request's own thread: return once the test lets go."""
        self.entered.set()
        self._let_go.wait(KEPT_AT_MOST_S)

    def release(self) -> None:
        self._let_go.set()
        started = [thread for thread in threading.enumerate() if thread not in self._threads_before]
        for thread in started:
            thread.join(PROMPT_S)
        assert not any(thread.is_alive() for thread in started), "a request's thread outlived its test"


@pytest.fixture
def hold() -> Iterator[_Hold]:
    kept = _Hold()
    yield kept
    kept.release()


class _HandPassedDeadline(Deadline):
    """A Deadline a test passes by hand: it leaves plenty of time until pass_now(), and none from then on."""

    def __init__(self) -> None:
        super().__init__(time.monotonic() + GENEROUS_MS / 1000)
        object.__setattr__(self, "_passed", threading.Event())

    def pass_now(self) -> None:
        self._passed.set()

    def remaining_s(self) -> float:
        return 0.0 if self._passed.is_set() else super().remaining_s()

    def expired(self) -> bool:
        return self._passed.is_set() or super().expired()


class _PassedSoonAfter(Deadline):
    """A Deadline that passes under a request a test keeps, however long the host took to get there: it
    has PASSES_AFTER_S left whenever it is asked until `arrival` is set (the request is where the test
    wants it), and from that moment it runs out for real, PASSES_AFTER_S later."""

    def __init__(self, arrival: _Arrival) -> None:
        super().__init__(time.monotonic() + GENEROUS_MS / 1000)
        object.__setattr__(self, "_arrival", arrival)

    def remaining_s(self) -> float:
        arrived_at_s = self._arrival.at_s
        if arrived_at_s is None:
            return PASSES_AFTER_S
        return max(0.0, arrived_at_s + PASSES_AFTER_S - time.monotonic())

    def expired(self) -> bool:
        return self.remaining_s() <= 0


class _TimeLeft(Deadline):
    """A Deadline that has `seconds` left whenever it is asked and never passes: what a test measures
    the next wait against, without a clock."""

    def __init__(self, seconds: float) -> None:
        super().__init__(time.monotonic() + GENEROUS_MS / 1000)
        object.__setattr__(self, "_seconds", seconds)

    def remaining_s(self) -> float:
        return self._seconds

    def expired(self) -> bool:
        return False


class _PassedOnceAsked(Deadline):
    """A Deadline that has plenty of time left the first time it is asked how much, and none from then
    on: it passes while the first request it was asked for is in flight, whichever thread is quicker."""

    def __init__(self) -> None:
        super().__init__(time.monotonic() + GENEROUS_MS / 1000)
        object.__setattr__(self, "_asked", threading.Event())

    def remaining_s(self) -> float:
        # The first ask comes from the thread that is about to start a request, before that request's own.
        if self._asked.is_set():
            return 0.0
        self._asked.set()
        return GENEROUS_MS / 1000

    def expired(self) -> bool:
        return self._asked.is_set()


class _PassedJustAfterARequestEnded(Deadline):
    """A Deadline that has plenty of time left until the thread of a storage request has asked it how
    much, and none from then on. A request's own thread asks once, to note as the request ends whether
    time was left. So this deadline passes just after a request ended in time, before the thread that
    waited for it comes to look."""

    def __init__(self) -> None:
        super().__init__(time.monotonic() + GENEROUS_MS / 1000)
        object.__setattr__(self, "_passed", threading.Event())

    def remaining_s(self) -> float:
        if self._passed.is_set():
            return 0.0
        if threading.current_thread().name == storage.REQUEST_THREAD_NAME:
            self._passed.set()
        return GENEROUS_MS / 1000

    def expired(self) -> bool:
        return self._passed.is_set()


class _Waits:
    """Stands where a transfer is handed `sleep`: it records every wait it is asked for, in order, and
    takes no time."""

    def __init__(self) -> None:
        self.asked: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.asked.append(seconds)


class _KeptRequests:
    """A pause for a request double: it keeps every request that reaches it until the test lets it go
    (a _Hold's keep()). `arrived` counts them, whatever thread each came on."""

    def __init__(self, hold: _Hold) -> None:
        self._hold = hold
        self._arrivals: list[None] = []

    @property
    def arrived(self) -> int:
        return len(self._arrivals)

    def __call__(self) -> None:
        # One append per request: a list takes them whole from any number of threads.
        self._arrivals.append(None)
        self._hold.keep()


class _StalledFirst:
    """Makes the first attempt of a request stall and every later request run free, however loaded the
    host is. One object stands in three places:

    request() is called by a request double on the request's own thread and keeps the first request
    that arrives (a _Hold's keep()). ceiling_s() stands in for storage._request_ceiling_s: a ceiling of
    SHORT_CEILING_S until the first wait was made, a generous one from then on, so that no request but
    the kept one can outlast its ceiling. sleep() is the stand-in the transfer waits through: it records
    the wait in `asked`, and returns only once the kept request has arrived, so that the next attempt
    cannot overtake it.
    """

    def __init__(self, hold: _Hold) -> None:
        self._hold = hold
        self._arrived = 0
        self.asked: list[float] = []

    def request(self) -> None:
        # Never two threads at once: the second request starts only after sleep() saw the first arrive.
        self._arrived += 1
        if self._arrived == 1:
            self._hold.keep()

    def ceiling_s(self, size_bytes: int) -> float:
        return GENEROUS_CEILING_S if self.asked else SHORT_CEILING_S

    def sleep(self, seconds: float) -> None:
        self.asked.append(seconds)
        assert self._hold.entered.wait(PROMPT_S), "the request that was to stall never started"


@pytest.fixture
def fixed_factor(monkeypatch: pytest.MonkeyPatch) -> float:
    """Every wait is its nominal seconds times FIXED_FACTOR: the jitter's draw is replaced."""
    monkeypatch.setattr(storage, "_jitter_factor", lambda: FIXED_FACTOR)
    return FIXED_FACTOR


@pytest.fixture
def long_waits(monkeypatch: pytest.MonkeyPatch) -> float:
    """Every wait is its nominal seconds times LONG_WAITS_FACTOR, in place of the jitter's draw: for a
    test whose deadline has a fixed time left, which the waits are measured against."""
    monkeypatch.setattr(storage, "_jitter_factor", lambda: LONG_WAITS_FACTOR)
    return LONG_WAITS_FACTOR


@pytest.fixture
def requests_settle_first(hold: _Hold, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job begins to wait for a request only once the request has ended or is kept (a _Hold's keep()):
    a deadline or a ceiling of tens of milliseconds cannot pass before, however late a loaded host runs
    the request's thread."""
    settle_requests_first(monkeypatch, hold.entered)


@pytest.fixture
def short_ceiling(monkeypatch: pytest.MonkeyPatch) -> float:
    """Every request's ceiling starts from SHORT_CEILING_S where the worker's starts from half a minute.
    Only for a test all of whose requests are kept: any other request could stall on a loaded host."""
    monkeypatch.setattr(storage, "REQUEST_BASE_S", SHORT_CEILING_S)
    return SHORT_CEILING_S


def _change_in_place(path: Path) -> None:
    """Give a file other bytes of the same length: a change its recorded size cannot tell."""
    path.write_bytes(path.read_bytes().translate(OTHER_BYTES))


def _holds_open(path: Path) -> bool:
    """Whether this process holds an open descriptor of the file at `path`, told by device and inode."""
    wanted = os.stat(path)
    for name in os.listdir(OPEN_DESCRIPTORS):
        try:
            held = os.fstat(int(name))
        except OSError:
            # The descriptor the listing itself used is gone by now.
            continue
        if (held.st_dev, held.st_ino) == (wanted.st_dev, wanted.st_ino):
            return True
    return False


# --- the tests' own instruments ----------------------------------------------------------------------
# What many tests below rely on to tell them something: each is shown here to be able to tell it.


def test_holds_open_when_this_process_has_the_file_open_should_see_its_descriptor_until_it_is_closed(tmp_path):
    # Arrange
    path = tmp_path / "held"
    path.write_bytes(b"a file a descriptor is open on")

    # Act
    with path.open("rb"):
        seen_while_open = _holds_open(path)
    seen_once_closed = _holds_open(path)

    # Assert: a test that finds no file held open could have found one, on this platform
    assert seen_while_open
    assert not seen_once_closed


def test_settle_requests_first_when_a_request_outlasts_what_the_deadline_leaves_should_end_it_before_the_job_waits(
    monkeypatch,
):
    # Arrange: a request that takes a moment, under a deadline that leaves a thousandth of that moment
    # whenever it is asked. Left to itself the job would stop waiting long before the request ends
    settle_requests_first(monkeypatch)
    expired = Failure("deadline_exceeded", "request_deadline_exceeded", False, "the test's request ran out of time")

    def after_a_moment() -> str:
        threading.Event().wait(A_MOMENT_S)
        return "answered"

    # Act
    returned = storage._within(_TimeLeft(A_MOMENT_S / 1000), after_a_moment, expired=expired)

    # Assert: the job came to wait only once its request had ended, so the shortest wait was enough. A
    # test that relies on this would otherwise depend on which thread the host runs first
    assert returned == "answered"


def test_settle_requests_first_when_a_request_is_kept_should_let_its_starter_get_on_once_it_arrived(
    hold, monkeypatch
):
    # Arrange: a request, on a thread named as a storage request's, that is kept only after a moment
    settle_requests_first(monkeypatch, hold.entered)

    def kept_after_a_moment() -> None:
        threading.Event().wait(A_MOMENT_S)
        hold.keep()

    request = threading.Thread(target=kept_after_a_moment, name=storage.REQUEST_THREAD_NAME, daemon=True)

    # Act
    request.start()
    arrived_when_its_starter_got_on = hold.entered.is_set()

    # Assert: not before the request was where the test keeps it, and without waiting for it to end
    assert arrived_when_its_starter_got_on
    assert request.is_alive()


def test_scripted_deadline_when_its_script_says_the_deadline_passed_should_have_no_time_left_from_then_on():
    # Arrange: a deadline that has not passed the first time it is asked, and has from the second on
    deadline = ScriptedDeadline(False, True)

    # Act: what is left before the first ask, after it and after the second
    left_s = [deadline.remaining_s()]
    for _ in range(2):
        deadline.expired()
        left_s.append(deadline.remaining_s())

    # Assert: more than any wait lasts until the script says it passed, nothing from then on, and
    # asking how much is left took no answer of the script
    assert left_s[0] > NEVER_S / 2 and left_s[1] > NEVER_S / 2
    assert left_s[2] == 0.0
    assert deadline.asked == 2


def test_clock_deadline_when_the_tests_clock_moves_should_have_left_what_that_clock_leaves_of_its_seconds():
    # Arrange: ten seconds on a clock only the test moves
    clock = FakeClock()
    deadline = ClockDeadline(clock, 10.0)

    # Act: what is left at once, 4 s on, at the tenth second and past it
    left_s = [deadline.remaining_s()]
    for seconds in (4.0, 6.0, 1.0):
        clock.advance(seconds)
        left_s.append(deadline.remaining_s())

    # Assert
    assert left_s == [10.0, 6.0, 0.0, 0.0]
    assert deadline.expired()


# --- the source -----------------------------------------------------------------------------------


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(storage, "DOWNLOAD_CHUNK_BYTES", CHUNK_BYTES)
    return CHUNK_BYTES


@pytest.fixture
def fake_s3() -> FakeS3:
    fake = FakeS3()
    fake.hold(BUCKET, KEY, OBJECT, ETAG)
    return fake


def _source(fake: FakeS3, *, size_bytes: int = len(OBJECT)) -> S3RangeSource:
    return S3RangeSource(fake, bucket=BUCKET, key=KEY, size_bytes=size_bytes, etag=ETAG)


def _never_answered(hold: _Hold) -> Callable[[int, int], bytes]:
    """An answer of a _StubSource that never comes: the read is kept until the test lets it go."""

    def never(start: int, length: int) -> bytes:
        hold.keep()
        return bytes(length)

    return never


class _StubSource:
    """A RangeSource that answers every read with what its `answer` function gives."""

    def __init__(self, size_bytes: int, answer: Callable[[int, int], bytes]) -> None:
        self.size_bytes = size_bytes
        self.reads: list[tuple[int, int]] = []
        self._answer = answer

    def read(self, start: int, length: int) -> bytes:
        self.reads.append((start, length))
        return self._answer(start, length)


def test_download_when_the_source_spans_several_chunks_should_write_exactly_its_bytes_in_order(
    tmp_path, fake_s3, small_chunks
):
    # Arrange
    destination = tmp_path / "source"

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)
    assert [call["Range"] for call in fake_s3.get_calls] == [
        "bytes=0-999", "bytes=1000-1999", "bytes=2000-2999", f"bytes=3000-{len(OBJECT) - 1}"
    ]


def test_download_when_it_reads_the_source_should_make_every_read_conditional_on_the_recorded_etag(
    tmp_path, fake_s3, small_chunks
):
    # Act
    download_source(_source(fake_s3), destination=tmp_path / "source", deadline=_deadline(), sleep=_Waits())

    # Assert
    assert len(fake_s3.get_calls) > 1
    assert {call["IfMatch"] for call in fake_s3.get_calls} == {f'"{ETAG}"'}


def test_download_when_it_creates_the_file_should_make_it_readable_by_its_owner_only(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"

    # Act
    download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    assert stat.S_IMODE(destination.stat().st_mode) & ~OWNER_ONLY == 0


def test_download_when_a_read_fails_retryably_should_read_that_chunk_again_and_finish(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: the second read never reaches the storage
    fake_s3.fail_get(2, UNREACHABLE)
    destination = tmp_path / "source"

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)
    assert [call["Range"] for call in fake_s3.get_calls][1:3] == ["bytes=1000-1999", "bytes=1000-1999"]


def test_download_when_a_chunk_keeps_failing_retryably_should_give_up_after_two_more_reads(
    tmp_path, fake_s3, small_chunks, fixed_factor, caplog
):
    # Arrange: every read from the second on fails
    caplog.set_level(logging.DEBUG)
    fake_s3.fail_get(2, UNREACHABLE, and_every_later_one=True)
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(
        lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=waits)
    )

    # Assert: the source's own failure after three reads of the chunk, a wait before the second and the
    # third (the nominal 0.5 s and 2 s, each times the factor 0.75), and one warning for each that says so
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_unreachable", True)
    assert [call["Range"] for call in fake_s3.get_calls] == ["bytes=0-999"] + ["bytes=1000-1999"] * 3
    assert waits.asked == [0.375, 1.5]
    warnings = [record.getMessage() for record in caplog.records]
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.WARNING]
    assert "0.375 s" in warnings[0] and "1.500 s" in warnings[1]
    assert all("source_unavailable/storage_unreachable" in warning for warning in warnings)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("failed_reads", "expected_waits"),
    [
        pytest.param(1, [0.375], id="once"),
        pytest.param(2, [0.375, 1.5], id="twice"),
    ],
)
def test_download_when_a_chunk_fails_retryably_and_then_arrives_should_wait_before_each_further_read_and_write_it_once(
    tmp_path, fake_s3, small_chunks, fixed_factor, failed_reads, expected_waits
):
    # Arrange: the read of the second chunk fails `failed_reads` times before it arrives
    for call_number in range(2, 2 + failed_reads):
        fake_s3.fail_get(call_number, UNREACHABLE)
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=waits)

    # Assert: the nominal 0.5 s, then 2 s, each times the factor 0.75; the chunk is in the file once
    assert waits.asked == expected_waits
    assert [call["Range"] for call in fake_s3.get_calls].count("bytes=1000-1999") == failed_reads + 1
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)


@pytest.mark.parametrize(
    ("seconds_left", "expected_waits", "expected_reads"),
    [
        pytest.param(30.0, [], 1, id="less-than-the-first-wait"),
        pytest.param(40.0, [], 1, id="exactly-the-first-wait"),
        pytest.param(100.0, [40.0], 2, id="less-than-the-second-wait"),
    ],
)
def test_download_when_the_next_wait_would_not_end_before_the_deadline_should_neither_wait_nor_read_again(
    tmp_path, fake_s3, small_chunks, long_waits, seconds_left, expected_waits, expected_reads
):
    # Arrange: every read fails retryably, and the deadline has `seconds_left` whenever it is asked; the
    # waits due are 40 s, then 160 s (the nominal 0.5 s and 2 s, each times the factor 80)
    fake_s3.fail_get(1, UNREACHABLE, and_every_later_one=True)
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(
        lambda: download_source(
            _source(fake_s3), destination=destination, deadline=_TimeLeft(seconds_left), sleep=waits
        )
    )

    # Assert: the last read's own retryable failure, not a deadline failure
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_unreachable", True)
    assert waits.asked == expected_waits
    assert len(fake_s3.get_calls) == expected_reads
    assert not destination.exists()


def test_download_when_a_wait_that_fitted_ends_after_the_deadline_passed_should_fail_as_a_deadline_and_not_read_again(
    tmp_path, fake_s3, small_chunks, fixed_factor
):
    # Arrange: every read fails retryably with a minute left on the deadline, and the deadline passes
    # while the download waits to read the chunk again
    fake_s3.fail_get(1, UNREACHABLE, and_every_later_one=True)
    deadline = _HandPassedDeadline()
    asked: list[float] = []

    def wait_past_the_deadline(seconds: float) -> None:
        asked.append(seconds)
        deadline.pass_now()

    destination = tmp_path / "source"

    # Act
    failure = _failure_of(
        lambda: download_source(
            _source(fake_s3), destination=destination, deadline=deadline, sleep=wait_past_the_deadline
        )
    )

    # Assert: the wait fitted when it was decided, so it was made; what it ran into is the deadline,
    # which is final, and the chunk is not read a second time
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert asked == [0.375]
    assert len(fake_s3.get_calls) == 1
    assert not destination.exists()


def test_download_when_a_read_fails_retryably_and_its_repeat_fails_for_good_should_stop_there_with_that_failure(
    tmp_path, fake_s3, small_chunks, fixed_factor
):
    # Arrange: the read of the second chunk never reaches the storage, and its repeat is refused
    fake_s3.fail_get(2, UNREACHABLE)
    fake_s3.fail_get(3, client_error("AccessDenied", HTTP_FORBIDDEN, "GetObject"))
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(
        lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=waits)
    )

    # Assert: one wait, before the repeat; the refusal ends the download at once, as itself
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_unavailable", "source_access_denied", False
    )
    assert waits.asked == [0.375]
    assert [call["Range"] for call in fake_s3.get_calls] == ["bytes=0-999", "bytes=1000-1999", "bytes=1000-1999"]
    assert not destination.exists()


def test_download_when_a_read_fails_for_good_should_not_read_again(tmp_path, small_chunks):
    # Arrange: the storage holds no such object, which no retry changes
    empty = FakeS3()
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(
        lambda: download_source(_source(empty), destination=destination, deadline=_deadline(), sleep=waits)
    )

    # Assert: read once, and waited for nothing
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "source_missing", False)
    assert len(empty.get_calls) == 1
    assert waits.asked == []
    assert not destination.exists()


def test_download_when_the_source_changes_between_two_chunks_should_fail_as_an_integrity_mismatch(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: after the first chunk the object is replaced by one of the same size and another ETag
    destination = tmp_path / "source"

    def replace_the_object(total_bytes: int) -> None:
        fake_s3.hold(BUCKET, KEY, bytes(len(OBJECT)), OTHER_ETAG)

    # Act
    failure = _failure_of(
        lambda: download_source(
            _source(fake_s3),
            destination=destination,
            deadline=_deadline(),
            sleep=_Waits(),
            on_bytes=replace_the_object,
        )
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_integrity_mismatch", "source_changed", False
    )
    assert len(fake_s3.get_calls) == 2
    assert not destination.exists()


def test_download_when_the_source_is_shorter_than_recorded_should_fail_as_an_integrity_mismatch(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: the dispatch records one chunk more than the object holds
    destination = tmp_path / "source"
    source = _source(fake_s3, size_bytes=len(OBJECT) + CHUNK_BYTES)

    # Act
    failure = _failure_of(
        lambda: download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())
    )

    # Assert
    assert (failure.error_class, failure.retryable) == ("source_integrity_mismatch", False)
    assert failure.code == "source_size_mismatch"
    assert not destination.exists()


@pytest.mark.parametrize("answered_bytes", [CHUNK_BYTES - 1, CHUNK_BYTES + 1], ids=["one-short", "one-long"])
def test_download_when_a_read_gives_another_length_than_asked_should_fail_as_a_size_mismatch(
    tmp_path, small_chunks, answered_bytes
):
    # Arrange: a source that does not keep its promise of exactly `length` bytes
    source = _StubSource(3 * CHUNK_BYTES, lambda start, length: bytes(answered_bytes))
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(
        lambda: download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_integrity_mismatch", "source_size_mismatch", False
    )
    assert source.reads == [(0, CHUNK_BYTES)]
    assert not destination.exists()


def test_download_when_the_deadline_has_passed_should_fail_before_it_reads(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(
        lambda: download_source(_source(fake_s3), destination=destination, deadline=_passed_deadline(), sleep=_Waits())
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert fake_s3.get_calls == []
    assert not destination.exists()


def test_download_when_the_deadline_passes_between_two_chunks_should_stop_reading(tmp_path, small_chunks):
    # Arrange: the deadline has not passed before the first chunk and has before the second
    deadline = ScriptedDeadline(False, True)
    source = _StubSource(3 * CHUNK_BYTES, lambda start, length: bytes(length))
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=deadline, sleep=_Waits()))

    # Assert
    assert failure.code == "download_deadline_exceeded"
    assert source.reads == [(0, CHUNK_BYTES)]
    assert deadline.asked == 2
    assert not destination.exists()


def test_download_when_a_read_never_returns_should_fail_at_the_deadline_and_remove_the_partial_file(
    tmp_path, small_chunks, hold, requests_settle_first
):
    # Arrange: the first chunk arrives, the read of the second never returns, and the deadline passes
    # a moment after that read began
    def first_chunk_only(start: int, length: int) -> bytes:
        if start > 0:
            hold.keep()
        return bytes(length)

    source = _StubSource(3 * CHUNK_BYTES, first_chunk_only)
    destination = tmp_path / "source"
    started_at = time.monotonic()

    # Act
    failure = _failure_of(
        lambda: download_source(
            source, destination=destination, deadline=_PassedSoonAfter(hold.entered), sleep=_Waits()
        )
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert hold.entered.is_set()
    assert time.monotonic() - started_at < PROMPT_S
    assert not destination.exists()


def test_download_when_a_read_ends_after_the_deadline_passed_should_fail_and_keep_none_of_its_bytes(tmp_path):
    # Arrange: the deadline passes while the only chunk is read, and the read then returns its bytes
    deadline = _HandPassedDeadline()

    def overdue(start: int, length: int) -> bytes:
        deadline.pass_now()
        return bytes(length)

    source = _StubSource(CHUNK_BYTES, overdue)
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=deadline, sleep=_Waits()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert source.reads == [(0, CHUNK_BYTES)]
    assert not destination.exists()


def test_download_when_a_read_stalls_and_the_next_one_arrives_should_wait_read_the_chunk_again_and_finish(
    tmp_path, small_chunks, hold, fixed_factor, monkeypatch
):
    # Arrange: the first read of the first chunk never returns, with a minute left on the deadline and
    # a ceiling of 50 ms; every later read answers at once
    stalled = _StalledFirst(hold)

    def first_read_never_returns(start: int, length: int) -> bytes:
        stalled.request()
        return OBJECT[start : start + length]

    source = _StubSource(len(OBJECT), first_read_never_returns)
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)
    destination = tmp_path / "source"

    # Act: the kept read is let go only once the download has finished without it
    written = download_source(source, destination=destination, deadline=_deadline(), sleep=stalled.sleep)
    hold.release()

    # Assert: the chunk was read again after the first wait (0.5 s times the factor 0.75), and whatever
    # the abandoned read returned in the end is in no file
    assert hold.entered.is_set()
    assert stalled.asked == [0.375]
    assert source.reads[:2] == [(0, CHUNK_BYTES), (0, CHUNK_BYTES)]
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)


def test_download_when_a_read_stalls_on_every_attempt_should_fail_retryable_as_stalled_after_three_reads(
    tmp_path, small_chunks, hold, fixed_factor, short_ceiling
):
    # Arrange: no read of the first chunk ever returns, with a minute left on the deadline and a ceiling
    # of some 50 ms
    kept = _KeptRequests(hold)

    def never(start: int, length: int) -> bytes:
        kept()
        return bytes(length)

    source = _StubSource(3 * CHUNK_BYTES, never)
    destination = tmp_path / "source"
    waits = _Waits()
    started_at = time.monotonic()

    # Act
    failure = _failure_of(
        lambda: download_source(source, destination=destination, deadline=_deadline(), sleep=waits)
    )
    gave_up_after_s = time.monotonic() - started_at
    assert hold.entered.wait(PROMPT_S)
    hold.release()

    # Assert: a failed attempt that passes, not a deadline; read three times, with both waits between
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "source_read_stalled", True)
    assert waits.asked == [0.375, 1.5]
    assert source.reads == [(0, CHUNK_BYTES)] * 3
    assert kept.arrived == 3
    assert gave_up_after_s < PROMPT_S
    assert not destination.exists()


def test_download_when_the_deadline_passes_under_a_read_that_outlasts_its_ceiling_should_fail_as_a_deadline(
    tmp_path, small_chunks, hold, short_ceiling
):
    # Arrange: a read that never returns under a ceiling of some 50 ms, and a deadline that has a minute
    # left when the read starts and none by the time the ceiling is reached
    source = _StubSource(3 * CHUNK_BYTES, _never_answered(hold))
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(
        lambda: download_source(source, destination=destination, deadline=_PassedOnceAsked(), sleep=waits)
    )
    assert hold.entered.wait(PROMPT_S)
    hold.release()

    # Assert: the deadline failure, final as ever, whatever the ceiling: no wait and no second read
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert waits.asked == []
    assert source.reads == [(0, CHUNK_BYTES)]
    assert not destination.exists()


def test_download_when_it_reads_a_chunk_should_hold_each_read_to_the_ceiling_of_its_own_length(
    tmp_path, fake_s3, small_chunks, monkeypatch
):
    # Arrange: note the size each ceiling is asked for; three whole chunks and a fourth of 332 bytes
    sizes: list[int] = []

    def noted_ceiling(size_bytes: int) -> float:
        sizes.append(size_bytes)
        return GENEROUS_CEILING_S

    monkeypatch.setattr(storage, "_request_ceiling_s", noted_ceiling)

    # Act
    download_source(_source(fake_s3), destination=tmp_path / "source", deadline=_deadline(), sleep=_Waits())

    # Assert
    assert sizes == [1000, 1000, 1000, 332]


def test_download_when_no_thread_can_be_started_for_a_read_should_fail_retryably_and_remove_the_file(
    tmp_path, fake_s3, monkeypatch
):
    # Arrange: the operating system gives this process no more threads
    destination = tmp_path / "source"

    def no_thread(thread: threading.Thread) -> None:
        raise RuntimeError(f"can't start new thread for {destination}")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", no_thread)
        with pytest.raises(WorkerFailure) as raised:
            download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True
    )
    assert str(tmp_path) not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert fake_s3.get_calls == []
    assert not destination.exists()


# --- a source read held to a deadline ---------------------------------------------------------------


def test_deadline_source_when_a_read_returns_in_time_should_give_the_inner_sources_bytes():
    # Arrange
    inner = _StubSource(len(OBJECT), lambda start, length: OBJECT[start : start + length])
    source = DeadlineSource(inner, _deadline())

    # Act
    data = source.read(1000, 24)

    # Assert
    assert data == OBJECT[1000:1024]
    assert inner.reads == [(1000, 24)]
    assert source.size_bytes == len(OBJECT)


def test_deadline_source_when_a_read_never_returns_should_fail_at_the_deadline(hold, requests_settle_first):
    # Arrange: a read that never returns, and a deadline that passes a moment after it began
    def never(start: int, length: int) -> bytes:
        hold.keep()
        return bytes(length)

    source = DeadlineSource(_StubSource(len(OBJECT), never), _PassedSoonAfter(hold.entered))
    started_at = time.monotonic()

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "source_read_deadline_exceeded", False
    )
    assert hold.entered.is_set()
    assert time.monotonic() - started_at < PROMPT_S


@pytest.mark.parametrize(
    ("deadline_ms", "ceiling_s", "expected"),
    [
        pytest.param(
            60_000, 0.05, ("source_unavailable", "source_read_stalled", True), id="the-ceiling-comes-first"
        ),
        pytest.param(
            500, 60.0, ("deadline_exceeded", "source_read_deadline_exceeded", False), id="the-deadline-comes-first"
        ),
    ],
)
def test_deadline_source_when_a_read_never_returns_should_fail_by_whichever_of_its_ceiling_and_the_deadline_is_nearer(
    hold, monkeypatch, deadline_ms, ceiling_s, expected
):
    # Arrange: a read that never returns, held to a deadline and to a ceiling of its own
    inner = _StubSource(len(OBJECT), _never_answered(hold))
    monkeypatch.setattr(storage, "REQUEST_BASE_S", ceiling_s)
    source = DeadlineSource(inner, Deadline.after_ms(deadline_ms))
    started_at = time.monotonic()

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))
    gave_up_after_s = time.monotonic() - started_at
    assert hold.entered.wait(PROMPT_S)
    hold.release()

    # Assert: stalled passes and is worth another attempt, a deadline is final; neither is read again here
    assert (failure.error_class, failure.code, failure.retryable) == expected
    assert inner.reads == [(0, CHUNK_BYTES)]
    assert gave_up_after_s < PROMPT_S


def test_deadline_source_when_the_deadline_passes_under_a_read_that_outlasts_its_ceiling_should_fail_as_a_deadline(
    hold, short_ceiling
):
    # Arrange: a read that never returns under a ceiling of some 50 ms, and a deadline that has a minute
    # left when the read starts and none by the time the ceiling is reached
    inner = _StubSource(len(OBJECT), _never_answered(hold))
    source = DeadlineSource(inner, _PassedOnceAsked())

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))
    assert hold.entered.wait(PROMPT_S)

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "source_read_deadline_exceeded", False
    )


def test_deadline_source_when_the_wait_was_for_the_deadline_should_fail_as_a_deadline_though_it_still_shows_time_left(
    hold,
):
    # Arrange: a read that never returns. The deadline, 50 ms away, is nearer than the read's ceiling of
    # 30 s, and its clock never moves: it shows the same time left after the wait as before it. So only
    # which of the two bounds was the nearer one when the wait began can tell what the read ran into
    inner = _StubSource(len(OBJECT), _never_answered(hold))
    source = DeadlineSource(inner, _TimeLeft(0.05))

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))
    assert hold.entered.wait(PROMPT_S)

    # Assert: the deadline, final, not a stalled attempt that would be worth making again
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "source_read_deadline_exceeded", False
    )
    assert inner.reads == [(0, CHUNK_BYTES)]


def test_within_when_a_call_that_is_given_no_ceiling_never_returns_should_fail_at_the_deadline_as_before(hold):
    # Arrange: a request held to the deadline alone, as a caller that names no ceiling holds it
    expired = Failure("deadline_exceeded", "request_deadline_exceeded", False, "the test's request ran out of time")
    started_at = time.monotonic()

    # Act
    failure = _failure_of(lambda: storage._within(Deadline.after_ms(100), hold.keep, expired=expired))
    gave_up_after_s = time.monotonic() - started_at
    assert hold.entered.wait(PROMPT_S)

    # Assert: the caller's own deadline failure, once the deadline's tenth of a second has passed
    assert failure is expired
    assert gave_up_after_s < PROMPT_S


def test_deadline_source_when_it_reads_should_hold_the_read_to_the_ceiling_of_its_length(monkeypatch):
    # Arrange: note the size the ceiling is asked for
    sizes: list[int] = []

    def noted_ceiling(size_bytes: int) -> float:
        sizes.append(size_bytes)
        return GENEROUS_CEILING_S

    monkeypatch.setattr(storage, "_request_ceiling_s", noted_ceiling)
    source = DeadlineSource(_StubSource(len(OBJECT), lambda start, length: OBJECT[start : start + length]), _deadline())

    # Act
    source.read(1000, 24)

    # Assert
    assert sizes == [24]


@pytest.mark.parametrize(
    ("size_bytes", "expected_s"),
    [
        pytest.param(0, 30.0, id="no-bytes"),
        pytest.param(256 * 1024, 31.0, id="256-kib"),
        pytest.param(8 * 1024 * 1024, 62.0, id="an-8-mib-chunk"),
        pytest.param(1024 * 1024 * 1024, 4126.0, id="a-1-gib-object"),
    ],
)
def test_request_ceiling_when_a_request_has_more_bytes_should_be_30_s_and_a_second_more_for_each_256_kib(
    size_bytes, expected_s
):
    # Act
    ceiling_s = storage._request_ceiling_s(size_bytes)

    # Assert
    assert ceiling_s == expected_s


@pytest.mark.parametrize(
    "overdue_ending",
    [
        pytest.param(None, id="it-returns-its-bytes"),
        pytest.param(
            WorkerFailure(Failure("source_unavailable", "short_read", True, "the storage ended the body early")),
            id="it-fails",
        ),
        pytest.param(RuntimeError("a bug in the source"), id="it-raises-another-error"),
    ],
)
def test_deadline_source_when_a_read_ends_after_the_deadline_passed_should_drop_how_it_ended(overdue_ending):
    # Arrange: the deadline passes while the inner source is read; only then does the read end
    deadline = _HandPassedDeadline()

    def overdue(start: int, length: int) -> bytes:
        deadline.pass_now()
        if overdue_ending is not None:
            raise overdue_ending
        return bytes(length)

    source = DeadlineSource(_StubSource(len(OBJECT), overdue), deadline)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        source.read(0, CHUNK_BYTES)

    # Assert: the deadline failure, with nothing of what the read returned or raised
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "source_read_deadline_exceeded", False
    )
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_deadline_source_when_the_deadline_passes_just_after_a_read_ended_in_time_should_still_give_its_bytes():
    # Arrange: a read that ends with time left, and a deadline that has passed by the time the thread
    # that waited for the read comes to look at how it ended
    deadline = _PassedJustAfterARequestEnded()
    source = DeadlineSource(_StubSource(len(OBJECT), lambda start, length: OBJECT[start : start + length]), deadline)

    # Act
    data = source.read(0, CHUNK_BYTES)

    # Assert: a request is judged by what was noted as it ended, not by when its waiter came to look: an
    # object stored in time is stored, however late the job's own thread is run again
    assert data == OBJECT[:CHUNK_BYTES]
    assert deadline.expired()


def test_deadline_source_when_the_deadline_is_farther_off_than_a_wait_can_last_should_still_read():
    # Arrange: a deadline no wait reaches (as the deadlines of conftest are), and a read that takes a moment
    def after_a_moment(start: int, length: int) -> bytes:
        threading.Event().wait(A_MOMENT_S)
        return OBJECT[start : start + length]

    source = DeadlineSource(_StubSource(len(OBJECT), after_a_moment), Deadline(time.monotonic() + NEVER_S))

    # Act
    data = source.read(0, CHUNK_BYTES)

    # Assert
    assert data == OBJECT[:CHUNK_BYTES]


def test_deadline_source_when_the_deadline_has_passed_should_fail_without_reading():
    # Arrange
    inner = _StubSource(len(OBJECT), lambda start, length: bytes(length))
    source = DeadlineSource(inner, _passed_deadline())

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "source_read_deadline_exceeded", False
    )
    assert inner.reads == []


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            WorkerFailure(Failure("source_unavailable", "short_read", True, "the storage ended the body early")),
            id="a-failure-of-the-source",
        ),
        pytest.param(RuntimeError("a bug in the source"), id="another-error"),
    ],
)
def test_deadline_source_when_the_inner_read_raises_should_raise_the_very_same_error(error):
    # Arrange
    def failing(start: int, length: int) -> bytes:
        raise error

    source = DeadlineSource(_StubSource(len(OBJECT), failing), _deadline())

    # Act
    with pytest.raises(type(error)) as raised:
        source.read(0, CHUNK_BYTES)

    # Assert
    assert raised.value is error


def test_deadline_source_when_no_thread_can_be_started_for_a_read_should_fail_retryably(monkeypatch):
    # Arrange: the operating system gives this process no more threads
    inner = _StubSource(len(OBJECT), lambda start, length: bytes(length))
    source = DeadlineSource(inner, _deadline())

    def no_thread(thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", no_thread)
        failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True
    )
    assert inner.reads == []


class _SizedOnly:
    """A source that has a size and cannot be read."""

    size_bytes = len(OBJECT)


@pytest.mark.parametrize(
    ("inner", "deadline"),
    [
        pytest.param(_SizedOnly(), _deadline(), id="an-inner-source-without-read"),
        pytest.param(_StubSource("10", lambda start, length: b""), _deadline(), id="a-size-as-text"),
        pytest.param(_StubSource(True, lambda start, length: b""), _deadline(), id="a-size-as-a-flag"),
        pytest.param(_StubSource(len(OBJECT), lambda start, length: b""), GENEROUS_MS, id="a-deadline-as-a-number"),
    ],
)
def test_deadline_source_when_an_argument_is_invalid_should_raise_type_error(inner, deadline):
    # Act and Assert
    with pytest.raises(TypeError):
        DeadlineSource(inner, deadline)


def test_download_when_the_destination_exists_should_refuse_it_and_leave_it_alone(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"
    destination.write_bytes(b"an earlier file")

    # Act
    with pytest.raises(ValueError):
        download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    assert destination.read_bytes() == b"an earlier file"
    assert fake_s3.get_calls == []


def test_download_when_the_destination_is_a_link_should_refuse_it_and_write_nothing_through_it(tmp_path, fake_s3):
    # Arrange: a link to a file that does not exist yet
    target = tmp_path / "elsewhere"
    destination = tmp_path / "source"
    destination.symlink_to(target)

    # Act
    with pytest.raises(ValueError):
        download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    assert not target.exists()
    assert fake_s3.get_calls == []


@pytest.mark.parametrize(
    ("error_number", "expected"),
    [
        pytest.param(errno.ENOSPC, ("resource_exhausted", "scratch_full", True), id="disk-full"),
        pytest.param(errno.EDQUOT, ("resource_exhausted", "scratch_full", True), id="quota-exceeded"),
        pytest.param(errno.EIO, ("output_write_failed", "source_write_failed", True), id="another-write-error"),
    ],
)
def test_download_when_a_write_fails_should_fail_typed_and_remove_the_partial_file(
    tmp_path, fake_s3, small_chunks, monkeypatch, error_number, expected
):
    # Arrange: the first chunk is written, the second write fails
    destination = tmp_path / "source"
    real_write = os.write
    writes: list[int] = []

    def failing_second_write(descriptor: int, data: bytes) -> int:
        writes.append(len(data))
        if len(writes) == 1:
            return real_write(descriptor, data)
        raise OSError(error_number, f"cannot write {destination}")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(os, "write", failing_second_write)
        failure = _failure_of(
            lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())
        )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == expected
    assert str(tmp_path) not in failure.detail
    assert len(writes) == 2
    assert not destination.exists()


def _is_a_descriptor_of(descriptor: int, path: Path) -> bool:
    """Whether `descriptor` is open on the file at `path`, told by device and inode."""
    try:
        held, wanted = os.fstat(descriptor), os.stat(path)
    except OSError:
        return False
    return (held.st_dev, held.st_ino) == (wanted.st_dev, wanted.st_ino)


def _fail_the_close_of(patched: pytest.MonkeyPatch, path: Path, error_number: int) -> list[int]:
    """Make os.close raise for a descriptor of the file at `path`, once it has really closed it (a close
    that fails gives the descriptor back all the same); every other close is the real one. Returns the
    list each failed close is noted in."""
    real_close = os.close
    failed: list[int] = []

    def close(descriptor: int) -> None:
        is_of_the_file = _is_a_descriptor_of(descriptor, path)
        real_close(descriptor)
        if is_of_the_file:
            failed.append(descriptor)
            raise OSError(error_number, f"cannot close {path}")

    patched.setattr(os, "close", close)
    return failed


@pytest.mark.parametrize(
    ("error_number", "expected"),
    [
        pytest.param(errno.ENOSPC, ("resource_exhausted", "scratch_full", True), id="disk-full"),
        pytest.param(errno.EDQUOT, ("resource_exhausted", "scratch_full", True), id="quota-exceeded"),
        pytest.param(errno.EIO, ("output_write_failed", "source_write_failed", True), id="another-write-error"),
    ],
)
def test_download_when_closing_the_file_fails_after_the_whole_copy_should_fail_typed_and_remove_the_file(
    tmp_path, fake_s3, small_chunks, monkeypatch, caplog, error_number, expected
):
    # Arrange: every chunk is read and written; then the close of the file fails, as where a disk
    # reports only then that it could not keep what was written
    caplog.set_level(logging.DEBUG)
    destination = tmp_path / "source"

    # Act
    with monkeypatch.context() as patched:
        failed_closes = _fail_the_close_of(patched, destination, error_number)
        failure = _failure_of(
            lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())
        )

    # Assert: a failed write, typed by its errno as any other, and no file is left for the encode
    assert (failure.error_class, failure.code, failure.retryable) == expected
    assert len(failed_closes) == 1
    assert len(fake_s3.get_calls) == 4
    assert str(tmp_path) not in failure.detail
    assert str(tmp_path) not in caplog.text
    assert not destination.exists()


@pytest.mark.parametrize(
    "copy_error",
    [
        pytest.param(
            WorkerFailure(
                Failure("source_integrity_mismatch", "source_changed", False, "the object's ETag differs")
            ),
            id="a-failure-of-the-source",
        ),
        pytest.param(RuntimeError("a bug in the source"), id="another-error"),
    ],
)
def test_download_when_the_copy_failed_and_closing_the_file_fails_too_should_raise_what_the_copy_failed_with(
    tmp_path, small_chunks, monkeypatch, caplog, copy_error
):
    # Arrange: the first chunk is written, the read of the second fails for good, and then the close of
    # the partial file fails as well
    caplog.set_level(logging.DEBUG)

    def second_chunk_fails(start: int, length: int) -> bytes:
        if start > 0:
            raise copy_error
        return bytes(length)

    source = _StubSource(3 * CHUNK_BYTES, second_chunk_fails)
    destination = tmp_path / "source"

    # Act
    with monkeypatch.context() as patched:
        failed_closes = _fail_the_close_of(patched, destination, errno.EIO)
        with pytest.raises(type(copy_error)) as raised:
            download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert: the copy's own failure, the very same error; the failed close replaced nothing
    assert raised.value is copy_error
    assert len(failed_closes) == 1
    assert str(tmp_path) not in caplog.text
    assert not destination.exists()


# The three places where a write of the source file can fail.
THE_FILE_CANNOT_BE_CREATED = "the-file-cannot-be-created"
A_WRITE_FAILS = "a-write-fails"
THE_CLOSE_FAILS_AFTER_THE_WHOLE_COPY = "the-close-fails-after-the-whole-copy"
WRITE_FAILURE_SITES = [THE_FILE_CANNOT_BE_CREATED, A_WRITE_FAILS, THE_CLOSE_FAILS_AFTER_THE_WHOLE_COPY]


def _download_whose_write_fails(site: str, tmp_path: Path, fake_s3: FakeS3) -> WorkerFailure:
    """Run a download whose source file cannot be written at `site`, each time on an error of the
    operating system whose own text names the destination, and return what the download raised."""
    destination = tmp_path / "source"
    if site == THE_FILE_CANNOT_BE_CREATED:
        # No stand-in fails here: the directory the file is to be made in does not exist.
        destination = tmp_path / "a-directory-that-is-gone" / "source"

    def failing_write(descriptor: int, data: bytes) -> int:
        raise OSError(errno.EIO, f"cannot write {destination}")

    with pytest.MonkeyPatch.context() as patched:
        if site == A_WRITE_FAILS:
            patched.setattr(os, "write", failing_write)
        if site == THE_CLOSE_FAILS_AFTER_THE_WHOLE_COPY:
            _fail_the_close_of(patched, destination, errno.EIO)
        with pytest.raises(WorkerFailure) as raised:
            download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())
    return raised.value


@pytest.mark.parametrize("site", WRITE_FAILURE_SITES)
def test_download_when_the_source_file_cannot_be_written_should_raise_a_failure_carrying_nothing_of_the_systems_error(
    tmp_path, fake_s3, small_chunks, site
):
    # Act
    raised = _download_whose_write_fails(site, tmp_path, fake_s3)

    # Assert: the typed failure alone. The operating system's error, whose text names a local path, is
    # neither its cause nor its context
    failure = raised.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "output_write_failed", "source_write_failed", True
    )
    assert str(tmp_path) not in str(raised)
    assert raised.__cause__ is None and raised.__context__ is None


# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while an error is still
# being handled would put that error's own text into the report. `failing_log` makes every line's
# handler fail; how the transfer ends is how it ends without it.


@pytest.mark.parametrize("site", WRITE_FAILURE_SITES)
def test_download_when_a_log_handler_fails_on_the_line_of_a_failed_write_should_keep_the_path_out_of_standard_error(
    tmp_path, fake_s3, small_chunks, capsys, failing_log, site
):
    # Act
    raised = _download_whose_write_fails(site, tmp_path, fake_s3)

    # Assert: logging did report its failure on that line, with nothing of the path the error named
    written = reported_log_failures(capsys, "the source file could not be written")
    assert str(tmp_path) not in written
    assert raised.failure.code == "source_write_failed"


def _a_bug_of_the_source_at_the_second_chunk(start: int, length: int) -> bytes:
    """The answers of a source whose first chunk arrives and whose second fails on a bug, in words that
    quote what it choked on."""
    if start > 0:
        raise RuntimeError(ERROR_TEXT_CANARY)
    return bytes(length)


def test_download_when_a_log_handler_fails_on_the_line_of_a_partial_file_not_closed_should_keep_the_copys_error_out(
    tmp_path, small_chunks, capsys, failing_log
):
    # Arrange: the copy fails on a bug of the source, and then the close of the partial file fails too
    source = _StubSource(3 * CHUNK_BYTES, _a_bug_of_the_source_at_the_second_chunk)
    destination = tmp_path / "source"

    # Act
    with pytest.MonkeyPatch.context() as patched:
        failed_closes = _fail_the_close_of(patched, destination, errno.EIO)
        with pytest.raises(RuntimeError) as raised:
            download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert: nothing of the copy's error, which is the one that is raised, nor of the path
    written = reported_log_failures(capsys, "a partial source file could not be closed")
    assert [told for told in (SECRET_CANARY, ENDPOINT_CANARY, str(tmp_path)) if told in written] == []
    assert str(raised.value) == ERROR_TEXT_CANARY
    assert len(failed_closes) == 1
    assert not destination.exists()


def _fail_the_removal_of(patched: pytest.MonkeyPatch, path: Path) -> None:
    """Make os.unlink fail for `path`, on an error whose own text names it; every other removal is the real one."""
    real_unlink = os.unlink

    def unlink(removed: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(removed) == path:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        real_unlink(removed, *arguments, **named_arguments)

    patched.setattr(os, "unlink", unlink)


def test_download_when_a_log_handler_fails_on_the_line_of_a_partial_file_not_removed_should_keep_every_errors_text_out(
    tmp_path, small_chunks, capsys, failing_log
):
    # Arrange: the copy fails on a bug of the source, and then the partial file cannot be removed
    source = _StubSource(3 * CHUNK_BYTES, _a_bug_of_the_source_at_the_second_chunk)
    destination = tmp_path / "source"

    # Act
    with pytest.MonkeyPatch.context() as patched:
        _fail_the_removal_of(patched, destination)
        with pytest.raises(RuntimeError) as raised:
            download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert: nothing of the copy's error, which is the one that is raised, nor of the path the removal's named
    written = reported_log_failures(capsys, "a partial source file could not be removed")
    assert [told for told in (SECRET_CANARY, ENDPOINT_CANARY, str(tmp_path)) if told in written] == []
    assert str(raised.value) == ERROR_TEXT_CANARY


def test_download_when_a_log_handler_fails_on_the_line_of_a_file_left_after_a_failed_close_should_keep_its_path_out(
    tmp_path, fake_s3, small_chunks, capsys, failing_log
):
    # Arrange: the close fails after the whole copy, and then the file cannot be removed
    destination = tmp_path / "source"

    # Act
    with pytest.MonkeyPatch.context() as patched:
        _fail_the_close_of(patched, destination, errno.EIO)
        _fail_the_removal_of(patched, destination)
        with pytest.raises(WorkerFailure) as raised:
            download_source(_source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    written = reported_log_failures(capsys, "a partial source file could not be removed")
    assert str(tmp_path) not in written
    assert raised.value.failure.code == "source_write_failed"


def test_download_when_a_log_handler_fails_on_the_line_of_a_read_that_is_tried_again_should_keep_the_errors_text_out(
    tmp_path, small_chunks, capsys, failing_log
):
    # Arrange: the first read fails in a way that can pass, with the error behind it as its cause, whose
    # text quotes the endpoint; every later read arrives
    def failing_once(start: int, length: int) -> bytes:
        if len(source.reads) == 1:
            passing = Failure("source_unavailable", "short_read", True, "the storage ended the body early")
            raise WorkerFailure(passing) from ConnectionError(ERROR_TEXT_CANARY)
        return OBJECT[start : start + length]

    source = _StubSource(len(OBJECT), failing_once)
    destination = tmp_path / "source"

    # Act
    written_bytes = download_source(source, destination=destination, deadline=_deadline(), sleep=_Waits())

    # Assert
    written = reported_log_failures(capsys, "a read of the source failed and is tried again")
    assert [told for told in (SECRET_CANARY, ENDPOINT_CANARY) if told in written] == []
    assert written_bytes == len(OBJECT)
    assert destination.read_bytes() == OBJECT


def test_download_when_a_log_handler_fails_on_the_line_of_a_progress_callback_that_failed_should_keep_its_text_out(
    tmp_path, fake_s3, small_chunks, capsys, failing_log
):
    # Arrange: a callback that fails on a bug, in words that quote what it choked on
    destination = tmp_path / "source"

    def broken_callback(total_bytes: int) -> None:
        raise RuntimeError(ERROR_TEXT_CANARY)

    # Act
    written_bytes = download_source(
        _source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits(), on_bytes=broken_callback
    )

    # Assert
    written = reported_log_failures(capsys, "progress callback failed")
    assert [told for told in (SECRET_CANARY, ENDPOINT_CANARY) if told in written] == []
    assert written_bytes == len(OBJECT)
    assert destination.read_bytes() == OBJECT


def test_download_when_progress_is_asked_for_should_report_the_running_total_after_each_chunk(
    tmp_path, fake_s3, small_chunks
):
    # Arrange
    totals: list[int] = []

    # Act
    download_source(
        _source(fake_s3), destination=tmp_path / "source", deadline=_deadline(), sleep=_Waits(), on_bytes=totals.append
    )

    # Assert
    assert totals == [1000, 2000, 3000, len(OBJECT)]


def test_download_when_the_progress_callback_raises_should_still_finish(tmp_path, fake_s3, small_chunks):
    # Arrange
    destination = tmp_path / "source"

    def broken_callback(total_bytes: int) -> None:
        raise RuntimeError("the progress display broke")

    # Act
    written = download_source(
        _source(fake_s3), destination=destination, deadline=_deadline(), sleep=_Waits(), on_bytes=broken_callback
    )

    # Assert
    assert written == len(OBJECT)
    assert destination.read_bytes() == OBJECT


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        pytest.param("destination", Path("relative/source"), ValueError, id="relative-destination"),
        pytest.param("destination", "/tmp/source-as-text", ValueError, id="destination-as-text"),
        pytest.param("deadline", 60_000, TypeError, id="deadline-as-a-number"),
        pytest.param("sleep", 0.5, TypeError, id="sleep-not-callable"),
        pytest.param("sleep", None, TypeError, id="sleep-as-nothing"),
        pytest.param("on_bytes", "not-callable", TypeError, id="progress-not-callable"),
    ],
)
def test_download_when_an_argument_is_invalid_should_refuse_it_before_it_reads(
    tmp_path, fake_s3, argument, value, error
):
    # Arrange
    arguments: dict[str, Any] = {
        "destination": tmp_path / "source", "deadline": _deadline(), "sleep": _Waits(), "on_bytes": None
    }
    arguments[argument] = value

    # Act
    with pytest.raises(error):
        download_source(_source(fake_s3), **arguments)

    # Assert
    assert fake_s3.get_calls == []


def test_download_when_no_sleep_is_given_should_refuse_it_before_it_reads(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"

    # Act
    with pytest.raises(TypeError):
        download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert fake_s3.get_calls == []
    assert not destination.exists()


# --- the generation --------------------------------------------------------------------------------


class _Generation:
    """A generation on disk: a few small files under an output root, and what a manifest records of them."""

    FILES = {
        "master.m3u8": b"#EXTM3U\n#EXT-X-VERSION:3\n",
        "thumbnail.jpg": b"\xff\xd8 a small picture \xff\xd9",
        "360p/playlist.m3u8": b"#EXTM3U\n#EXT-X-TARGETDURATION:6\n",
        "360p/segment_0000.ts": b"\x47" + bytes(187) + b"first segment",
        "360p/segment_0001.ts": b"\x47" + bytes(187) + b"second segment, a little longer",
    }
    MANIFEST = b'{"manifest_version":"1.0.0-draft","document_kind":"generation.manifest"}'

    def __init__(self, root: Path) -> None:
        self.output_root = root / "output"
        (self.output_root / "360p").mkdir(parents=True)
        for path, data in self.FILES.items():
            (self.output_root / path).write_bytes(data)
        (self.output_root / MANIFEST_NAME).write_bytes(self.MANIFEST)
        self.artifacts = [artifact_of(path, data) for path, data in self.FILES.items()]
        # What every upload of this generation waits through, unless a test hands it another `sleep`.
        self.waits = _Waits()
        self.manifest = manifest_recording(self.artifacts)

    def upload(self, client: Any, **changes: Any) -> int:
        arguments: dict[str, Any] = {
            "bucket": OUTPUT_BUCKET,
            "prefix": PREFIX,
            "output_root": self.output_root,
            "artifacts": self.artifacts,
            "manifest": self.manifest,
            "deadline": _deadline(),
            "sleep": self.waits,
        }
        arguments.update(changes)
        return upload_generation(client, **arguments)


def artifact_of(path: str, data: bytes) -> ArtifactFile:
    return ArtifactFile(
        path=path,
        kind=ARTIFACT_KIND_BY_SUFFIX[os.path.splitext(path)[1]],
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def manifest_recording(artifacts: Sequence[ArtifactFile]) -> ManifestOutput:
    """What the package stage returns of a manifest that lists exactly `artifacts`: how many they are and
    the bytes they hold together. The manifest's own file is the generation's, _Generation.MANIFEST."""
    return ManifestOutput(
        path=MANIFEST_NAME,
        sha256=hashlib.sha256(_Generation.MANIFEST).hexdigest(),
        size_bytes=len(_Generation.MANIFEST),
        artifact_count=len(artifacts),
        total_bytes=sum(artifact.size_bytes for artifact in artifacts),
    )


def unchecked_artifact(path: object, size_bytes: int = 1) -> ArtifactFile:
    """An ArtifactFile holding a path its own constructor refuses: what a bug elsewhere could hand the upload."""
    artifact = object.__new__(ArtifactFile)
    for name, value in (("path", path), ("kind", "hls_segment"), ("size_bytes", size_bytes), ("sha256", "0" * 64)):
        object.__setattr__(artifact, name, value)
    return artifact


def _with_a_large_segment(generation: _Generation) -> list[ArtifactFile]:
    """The generation's artifacts and one more on disk: LARGE_SEGMENT, recorded as it was written."""
    (generation.output_root / LARGE_SEGMENT_PATH).write_bytes(LARGE_SEGMENT)
    return [*generation.artifacts, artifact_of(LARGE_SEGMENT_PATH, LARGE_SEGMENT)]


@pytest.fixture
def generation(tmp_path: Path) -> _Generation:
    return _Generation(tmp_path)


def test_upload_when_the_generation_is_whole_should_put_every_file_under_the_prefix_in_the_bucket(generation):
    # Arrange
    client = FakeS3()

    # Act
    uploaded = generation.upload(client)

    # Assert
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert {call.key: call.body for call in client.stored} == expected
    assert {call.bucket for call in client.stored} == {OUTPUT_BUCKET}
    assert all(call.content_length == len(call.body) for call in client.stored)
    assert uploaded == sum(len(data) for data in expected.values())


def test_upload_when_the_generation_is_whole_should_put_the_artifacts_in_order_and_the_manifest_last(generation):
    # Arrange
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    assert [call.key for call in client.put_calls] == [
        f"{PREFIX}/master.m3u8",
        f"{PREFIX}/thumbnail.jpg",
        f"{PREFIX}/360p/playlist.m3u8",
        f"{PREFIX}/360p/segment_0000.ts",
        f"{PREFIX}/360p/segment_0001.ts",
        f"{PREFIX}/{MANIFEST_NAME}",
    ]


def test_upload_when_it_puts_a_file_should_name_the_content_type_of_its_suffix(generation):
    # Arrange
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    assert {call.key.removeprefix(f"{PREFIX}/"): call.content_type for call in client.stored} == {
        "master.m3u8": "application/vnd.apple.mpegurl",
        "thumbnail.jpg": "image/jpeg",
        "360p/playlist.m3u8": "application/vnd.apple.mpegurl",
        "360p/segment_0000.ts": "video/mp2t",
        "360p/segment_0001.ts": "video/mp2t",
        MANIFEST_NAME: "application/json",
    }


def test_upload_when_other_files_lie_in_the_output_root_should_put_only_what_it_was_told(generation):
    # Arrange: a key file, a stray segment and a leftover directory beside the generation's files
    root = generation.output_root
    (root / "enc.key").write_bytes(b"0123456789abcdef")
    (root / "360p" / "segment_0002.ts").write_bytes(b"\x47 a segment no manifest lists")
    (root / "720p").mkdir()
    (root / "720p" / "playlist.m3u8").write_bytes(b"#EXTM3U\n")
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    listed = {*_Generation.FILES, MANIFEST_NAME}
    assert {call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls} == listed


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("../outside.ts", id="parent-segment-first"),
        pytest.param("360p/../../outside.ts", id="parent-segment-inside"),
        pytest.param("/etc/outside.ts", id="absolute"),
        pytest.param("360p/Segment_0000.ts", id="upper-case"),
        pytest.param("a/b/c/d/segment.ts", id="five-segments"),
        pytest.param("360p/.segment.ts", id="hidden-file"),
        pytest.param("./360p/segment_0000.ts", id="current-directory-segment"),
        pytest.param("360p//segment_0000.ts", id="empty-segment"),
        pytest.param("360p/segment_0000.ts\n", id="trailing-line-break"),
        pytest.param("360p/segment 0000.ts", id="a-space"),
        pytest.param("", id="empty"),
        pytest.param(None, id="not-text"),
    ],
)
def test_upload_when_an_artifacts_path_is_no_contract_rel_path_should_refuse_it_before_any_request(
    tmp_path, generation, path
):
    # Arrange: where such a path points, a regular file exists, so only the path's form refuses it
    (tmp_path / "outside.ts").write_bytes(b"x")
    (generation.output_root / "360p" / "Segment_0000.ts").write_bytes(b"x")
    (generation.output_root / "360p" / ".segment.ts").write_bytes(b"x")
    (generation.output_root / "360p" / "segment 0000.ts").write_bytes(b"x")
    (generation.output_root / "a" / "b" / "c" / "d").mkdir(parents=True)
    (generation.output_root / "a" / "b" / "c" / "d" / "segment.ts").write_bytes(b"x")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, unchecked_artifact(path)])

    # Assert
    assert client.put_calls == []
    assert repr(path) in str(raised.value)
    assert str(generation.output_root) not in str(raised.value)
    assert PREFIX not in str(raised.value)


def test_upload_when_an_artifact_is_a_link_to_a_file_should_refuse_it_before_any_request(tmp_path, generation):
    # Arrange: the listed segment is replaced by a link to a file outside the output root
    outside = tmp_path / "outside.ts"
    outside.write_bytes(_Generation.FILES["360p/segment_0001.ts"])
    listed = generation.output_root / "360p" / "segment_0001.ts"
    listed.unlink()
    listed.symlink_to(outside)
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'360p/segment_0001.ts'" in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_upload_when_an_artifacts_directory_is_a_link_should_refuse_it_before_any_request(tmp_path, generation):
    # Arrange: the rendition directory is replaced by a link to a directory holding the same regular files
    real = generation.output_root / "360p"
    moved = tmp_path / "moved-360p"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'360p/playlist.m3u8'" in str(raised.value)


@pytest.mark.parametrize(
    "make_it",
    [
        pytest.param(lambda path: path.unlink(), id="missing"),
        pytest.param(lambda path: (path.unlink(), path.mkdir()), id="a-directory"),
    ],
)
def test_upload_when_an_artifact_is_not_a_regular_file_should_refuse_it_before_any_request(generation, make_it):
    # Arrange
    make_it(generation.output_root / "thumbnail.jpg")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'thumbnail.jpg'" in str(raised.value)


def test_upload_when_two_artifacts_share_a_path_should_refuse_them_before_any_request(generation):
    # Arrange
    twice = [*generation.artifacts, generation.artifacts[-1]]
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=twice)

    # Assert
    assert client.put_calls == []
    assert "'360p/segment_0001.ts'" in str(raised.value)


def test_upload_when_an_artifact_has_the_manifests_path_should_refuse_it_before_any_request(generation):
    # Arrange: an artifact that would be stored where the completion marker goes
    impostor = unchecked_artifact(MANIFEST_NAME, size_bytes=len(_Generation.MANIFEST))
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, impostor])

    # Assert
    assert client.put_calls == []
    assert repr(MANIFEST_NAME) in str(raised.value)


def _one_artifact_fewer(generation: _Generation) -> list[ArtifactFile]:
    return generation.artifacts[:-1]


def _one_artifact_more(generation: _Generation) -> list[ArtifactFile]:
    return _with_a_large_segment(generation)


def _fewer_artifacts_of_the_same_total_size(generation: _Generation) -> list[ArtifactFile]:
    """The two segments give their place to one on disk that is as long as both, recorded as it was written."""
    joined_path = "360p/segment_0009.ts"
    joined = _Generation.FILES["360p/segment_0000.ts"] + _Generation.FILES["360p/segment_0001.ts"]
    (generation.output_root / joined_path).write_bytes(joined)
    return [*generation.artifacts[:-2], artifact_of(joined_path, joined)]


def _more_artifacts_of_the_same_total_size(generation: _Generation) -> list[ArtifactFile]:
    """The last segment gives its place to two on disk that are together as long as it, each recorded as
    it was written."""
    whole = _Generation.FILES["360p/segment_0001.ts"]
    halves = {"360p/segment_0008.ts": whole[: len(whole) // 2], "360p/segment_0009.ts": whole[len(whole) // 2 :]}
    for path, half in halves.items():
        (generation.output_root / path).write_bytes(half)
    return [*generation.artifacts[:-1], *(artifact_of(path, half) for path, half in halves.items())]


def _as_many_artifacts_of_another_total_size(generation: _Generation) -> list[ArtifactFile]:
    """The last segment gives its place to another one on disk, a byte longer and recorded as it was written."""
    other_path, other_segment = "360p/segment_0009.ts", _Generation.FILES["360p/segment_0001.ts"] + b"\x00"
    (generation.output_root / other_path).write_bytes(other_segment)
    return [*generation.artifacts[:-1], artifact_of(other_path, other_segment)]


def _as_many_artifacts_of_a_smaller_total_size(generation: _Generation) -> list[ArtifactFile]:
    """The last segment gives its place to another one on disk, a byte shorter and recorded as it was written."""
    other_path, other_segment = "360p/segment_0009.ts", _Generation.FILES["360p/segment_0001.ts"][:-1]
    (generation.output_root / other_path).write_bytes(other_segment)
    return [*generation.artifacts[:-1], artifact_of(other_path, other_segment)]


@pytest.mark.parametrize(
    "given",
    [
        pytest.param(_one_artifact_fewer, id="one-artifact-fewer"),
        pytest.param(_one_artifact_more, id="one-artifact-more"),
        pytest.param(_fewer_artifacts_of_the_same_total_size, id="fewer-of-the-same-total-size"),
        pytest.param(_more_artifacts_of_the_same_total_size, id="more-of-the-same-total-size"),
        pytest.param(_as_many_artifacts_of_another_total_size, id="as-many-of-another-total-size"),
        pytest.param(_as_many_artifacts_of_a_smaller_total_size, id="as-many-of-a-smaller-total-size"),
    ],
)
def test_upload_when_the_artifacts_are_not_what_the_manifest_records_should_refuse_them_before_any_request(
    generation, given
):
    # Arrange: artifacts that are each a recorded regular file of the output root, where the manifest
    # counts five of them and the bytes of those five
    artifacts = given(generation)
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=artifacts)

    # Assert
    assert client.put_calls == []
    assert str(generation.output_root) not in str(raised.value)


def test_upload_when_an_artifact_has_a_suffix_without_a_content_type_should_refuse_it_before_any_request(generation):
    # Arrange: a well-formed path of a regular file, of a kind the worker never writes
    (generation.output_root / "360p" / "enc.key").write_bytes(b"0123456789abcdef")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, unchecked_artifact("360p/enc.key", 16)])

    # Assert
    assert client.put_calls == []
    assert "'360p/enc.key'" in str(raised.value)


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param(f"/{PREFIX}", id="leading-slash"),
        pytest.param(f"{PREFIX}/", id="trailing-slash"),
        pytest.param("test/media/../55555555-5555-7555-8555-555555555555", id="parent-segment"),
        pytest.param("../55555555-5555-7555-8555-555555555555", id="parent-segment-first"),
        pytest.param("test/./55555555-5555-7555-8555-555555555555", id="current-directory-segment"),
        pytest.param("test//55555555-5555-7555-8555-555555555555", id="empty-segment"),
        pytest.param("", id="empty"),
        pytest.param(None, id="not-text"),
    ],
)
def test_upload_when_the_prefix_is_no_contract_object_key_should_refuse_it_before_any_request(generation, prefix):
    # Arrange
    client = FakeS3()

    # Act
    with pytest.raises(ValueError):
        generation.upload(client, prefix=prefix)

    # Assert
    assert client.put_calls == []


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        pytest.param("bucket", "", ValueError, id="empty-bucket"),
        pytest.param("output_root", Path("relative/output"), ValueError, id="relative-output-root"),
        pytest.param("output_root", Path("/nonexistent/output/root"), ValueError, id="missing-output-root"),
        pytest.param("artifacts", "master.m3u8", TypeError, id="artifacts-as-text"),
        pytest.param("artifacts", [{"path": "master.m3u8"}], TypeError, id="artifacts-as-mappings"),
        pytest.param("manifest", MANIFEST_NAME, TypeError, id="manifest-as-text"),
        pytest.param("deadline", 60_000, TypeError, id="deadline-as-a-number"),
        pytest.param("sleep", 0.5, TypeError, id="sleep-not-callable"),
        pytest.param("sleep", None, TypeError, id="sleep-as-nothing"),
    ],
)
def test_upload_when_an_argument_is_invalid_should_refuse_it_before_any_request(generation, argument, value, error):
    # Arrange
    client = FakeS3()

    # Act
    with pytest.raises(error):
        generation.upload(client, **{argument: value})

    # Assert
    assert client.put_calls == []


@pytest.mark.parametrize(
    "bucket",
    [
        pytest.param("zq", id="two-characters"),
        pytest.param("q" * 64, id="sixty-four-characters"),
        pytest.param("..", id="two-dots"),
        pytest.param("test.output.bucket", id="dots"),
        pytest.param("Test-Output-Bucket", id="upper-case"),
        pytest.param("test_output_bucket", id="underscores"),
        pytest.param("-test-output-bucket", id="a-hyphen-first"),
        pytest.param("test-output-bucket-", id="a-hyphen-last"),
        pytest.param("test-output-bucket/other", id="a-slash"),
        pytest.param("test-output-bucket\n", id="a-trailing-line-break"),
        pytest.param("arn:aws:s3:us-east-1:123456789012:accesspoint/test-output", id="an-access-point-arn"),
    ],
)
def test_upload_when_the_bucket_is_no_contract_bucket_should_refuse_it_before_any_request_in_other_words_than_its_own(
    generation, bucket
):
    # Arrange: a name the storage client itself would take, or send and have refused
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, bucket=bucket)

    # Assert
    assert client.put_calls == []
    assert bucket not in str(raised.value)


@pytest.mark.parametrize(
    "bucket",
    [
        pytest.param("a1b", id="three-characters"),
        pytest.param("q" * 63, id="sixty-three-characters"),
        pytest.param("0videos-2026-10-9", id="digits-first-inside-and-last"),
    ],
)
def test_upload_when_the_bucket_is_a_contract_bucket_at_a_bound_of_its_shape_should_put_every_object_in_it(
    generation, bucket
):
    # Arrange: a name the contract allows, as short or as long as it may be, or with digits where they may stand
    client = FakeS3()

    # Act
    generation.upload(client, bucket=bucket)

    # Assert
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert {call.bucket for call in client.stored} == {bucket}


def test_upload_when_the_client_is_no_s3_client_should_raise_type_error(generation):
    # Act and Assert
    with pytest.raises(TypeError):
        generation.upload(object())


def test_upload_when_an_artifact_changed_size_on_disk_should_fail_without_putting_it_or_the_manifest(generation):
    # Arrange: the second segment grew after the manifest recorded it
    with (generation.output_root / "360p" / "segment_0001.ts").open("ab") as segment:
        segment.write(b"more")
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    uploaded = [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls]
    assert uploaded == ["master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts"]


def test_upload_when_a_fifo_stands_where_a_file_was_should_refuse_it_at_once_and_not_wait_for_a_writer(generation):
    # Arrange: once the first object is stored, a FIFO no process writes to takes the second file's place
    thumbnail = generation.output_root / "thumbnail.jpg"

    def put_a_fifo_in_its_place(total_bytes: int) -> None:
        if stat.S_ISREG(os.lstat(thumbnail).st_mode):
            thumbnail.unlink()
            os.mkfifo(thumbnail)

    client = FakeS3()
    endings: list[BaseException] = []

    def upload() -> None:
        try:
            generation.upload(client, on_bytes=put_a_fifo_in_its_place)
        except BaseException as error:  # noqa: BLE001 - kept for the test's own thread to judge
            endings.append(error)

    thread = threading.Thread(target=upload, daemon=True)

    # Act: on a thread of its own, because an open that waited for a writer would never return. One
    # that did wait is let go, by an open of the FIFO's other end, before anything is asserted
    thread.start()
    thread.join(PROMPT_S)
    waited_for_a_writer = thread.is_alive()
    if waited_for_a_writer:
        other_end = os.open(thumbnail, os.O_WRONLY | os.O_NONBLOCK)
        thread.join(PROMPT_S)
        os.close(other_end)

    # Assert: a FIFO did stand where the file was, and it was refused as a file that changed, as any
    # other file that is no longer a regular one, with nothing put after the first object; the refused
    # FIFO is not left open
    assert stat.S_ISFIFO(os.lstat(thumbnail).st_mode)
    assert not waited_for_a_writer
    assert [type(ending) for ending in endings] == [WorkerFailure]
    failure = endings[0].failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert "thumbnail.jpg" in failure.detail
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]
    assert not _holds_open(thumbnail)


def test_upload_when_a_link_stands_where_a_file_was_should_refuse_it_and_send_nothing_of_what_it_points_to(
    tmp_path, generation
):
    # Arrange: once the first object is stored, the second file gives its place to a link to a file of
    # the same bytes outside the output root
    thumbnail = generation.output_root / "thumbnail.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_Generation.FILES["thumbnail.jpg"])

    def put_a_link_in_its_place(total_bytes: int) -> None:
        if not thumbnail.is_symlink():
            thumbnail.unlink()
            thumbnail.symlink_to(outside)

    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client, on_bytes=put_a_link_in_its_place))

    # Assert: the link was not followed, though what it points to has the recorded size and bytes: a
    # file is opened through no link, also when the link came after the upload was planned
    assert thumbnail.is_symlink()
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert failure.detail.startswith("thumbnail.jpg changed before its upload: it could not be opened: ")
    assert str(tmp_path) not in failure.detail
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]


def test_upload_when_a_file_is_gone_when_it_is_opened_should_raise_a_failure_carrying_nothing_of_the_systems_error(
    generation,
):
    # Arrange: once the first object is stored, the second file is removed
    thumbnail = generation.output_root / "thumbnail.jpg"
    client = FakeS3()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client, on_bytes=lambda total_bytes: thumbnail.unlink(missing_ok=True))

    # Assert: the typed failure alone. The operating system's error, whose text names the file's whole
    # path, is neither its cause nor its context
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert failure.detail == "thumbnail.jpg changed before its upload: it could not be opened: ENOENT"
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_upload_when_an_opened_file_cannot_be_measured_should_raise_a_failure_carrying_nothing_of_the_systems_error(
    generation, monkeypatch
):
    # Arrange: the first file opens, and asking the operating system for its size fails, in words that
    # name a path
    first_file = generation.output_root / "master.m3u8"
    wanted = os.stat(first_file)
    measure = os.fstat

    def fail_to_measure_the_first_file(descriptor: int) -> os.stat_result:
        status = measure(descriptor)
        if (status.st_dev, status.st_ino) == (wanted.st_dev, wanted.st_ino):
            raise OSError(errno.EIO, f"cannot measure {PATH_CANARY}")
        return status

    client = FakeS3()

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(os, "fstat", fail_to_measure_the_first_file)
        with pytest.raises(WorkerFailure) as raised:
            generation.upload(client)

    # Assert
    failure = raised.value.failure
    assert failure.detail == "master.m3u8 changed before its upload: it could not be measured: EIO"
    assert client.put_calls == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert not _holds_open(first_file)


def test_upload_when_the_manifest_changed_size_on_disk_should_fail_without_putting_it(generation):
    # Arrange
    (generation.output_root / MANIFEST_NAME).write_bytes(_Generation.MANIFEST[:-1])
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert f"{PREFIX}/{MANIFEST_NAME}" not in [call.key for call in client.put_calls]
    assert len(client.stored) == len(_Generation.FILES)


def test_upload_when_an_artifacts_bytes_changed_in_place_should_fail_and_never_put_the_manifest(generation):
    # Arrange: the second segment holds other bytes, of the same length, than the manifest recorded
    _change_in_place(generation.output_root / "360p" / "segment_0001.ts")
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the changed segment was sent once, and nothing after it
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert "360p/segment_0001.ts" in failure.detail
    assert str(generation.output_root) not in failure.detail
    uploaded = [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls]
    assert uploaded == [
        "master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts", "360p/segment_0001.ts"
    ]


def test_upload_when_the_manifests_bytes_changed_in_place_should_fail_and_not_send_it_again(generation):
    # Arrange
    _change_in_place(generation.output_root / MANIFEST_NAME)
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert MANIFEST_NAME in failure.detail
    assert [call.key for call in client.put_calls].count(f"{PREFIX}/{MANIFEST_NAME}") == 1


class _PreparingS3(FakeS3):
    """A FakeS3 that first does to the body what `prepare` does, as a client that looks at a body before
    it sends it, and then reads and records it as FakeS3 does."""

    def __init__(self, prepare: Callable[[Any], Any]) -> None:
        super().__init__()
        self._prepare = prepare

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        self._prepare(Body)
        return super().put_object(
            Bucket=Bucket, Key=Key, Body=Body, ContentLength=ContentLength, ContentType=ContentType
        )


def _read_through_in_pieces(body: Any, piece_bytes: int = SMALL_PIECE_BYTES) -> bytes:
    """Read a body as an HTTP client sends one: a piece at a time, until a read gives nothing."""
    pieces: list[bytes] = []
    while True:
        piece = body.read(piece_bytes)
        if not piece:
            return b"".join(pieces)
        pieces.append(piece)


@pytest.mark.parametrize(
    "rewind",
    [
        pytest.param(lambda body: body.seek(0), id="seek-to-zero"),
        pytest.param(lambda body: body.seek(0, os.SEEK_SET), id="seek-to-zero-from-the-start"),
    ],
)
def test_upload_when_the_client_reads_a_body_through_and_rewinds_it_should_accept_what_it_then_sends(
    generation, rewind
):
    # Arrange: a client that reads the whole body once before it sends it, as one that signs its content does
    def read_through_and_rewind(body: Any) -> None:
        _read_through_in_pieces(body)
        rewind(body)

    client = _PreparingS3(read_through_and_rewind)

    # Act
    uploaded = generation.upload(client)

    # Assert
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert {call.key: call.body for call in client.stored} == expected
    assert uploaded == sum(len(data) for data in expected.values())


def _skip_the_start(body: Any) -> None:
    body.seek(SMALL_PIECE_BYTES)


def _look_at_the_end_and_come_back(body: Any) -> None:
    body.read(SMALL_PIECE_BYTES)
    body.seek(0, os.SEEK_END)
    body.seek(SMALL_PIECE_BYTES)


@pytest.mark.parametrize(
    "move",
    [
        pytest.param(_skip_the_start, id="it-skips-the-start"),
        pytest.param(_look_at_the_end_and_come_back, id="it-comes-back-to-where-it-was"),
    ],
)
def test_upload_when_the_client_moves_within_a_body_without_a_rewind_should_refuse_what_it_cannot_check(
    generation, move
):
    # Arrange: a client that moves within each body, never back to its start, and then reads the rest
    client = _PreparingS3(move)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the first object is not taken for stored, and nothing follows it
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]


def _measure_by_the_end(body: Any) -> None:
    """What a client does that measures a body before it reads it (botocore's determine_content_length)."""
    position = body.tell()
    body.seek(0, os.SEEK_END)
    body.tell()
    body.seek(position)


def test_upload_when_the_client_measures_a_body_by_its_end_and_rewinds_it_should_accept_what_it_then_sends(
    generation,
):
    # Arrange
    client = _PreparingS3(_measure_by_the_end)

    # Act
    uploaded = generation.upload(client)

    # Assert
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert {call.key: call.body for call in client.stored} == expected
    assert uploaded == sum(len(data) for data in expected.values())


class _HalfReadingS3(FakeS3):
    """A FakeS3 that answers 200 after it read only the first half of each body."""

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        self.put_calls.append(
            PutCall(Bucket, Key, Body.read(ContentLength // 2), ContentLength, ContentType, failed=False)
        )
        return {"ResponseMetadata": {"HTTPStatusCode": HTTP_OK}}


def test_upload_when_the_client_reads_only_a_part_of_a_body_should_refuse_it_and_never_put_the_manifest(generation):
    # Arrange
    client = _HalfReadingS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert "master.m3u8" in failure.detail
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]


def test_upload_when_the_client_asks_a_body_for_no_bytes_should_not_take_that_for_the_files_end(generation):
    # Arrange: a client that asks each body for zero bytes before it reads it
    client = _PreparingS3(lambda body: body.read(0))

    # Act
    uploaded = generation.upload(client)

    # Assert
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert uploaded == sum(len(call.body) for call in client.stored)


class _StreamingS3(FakeS3):
    """A FakeS3 that reads each body as an HTTP client sends one: a piece at a time, until a read gives
    nothing. With `signs_first` it reads the body through like that once before, as a client that signs
    the content does, then rewinds it and calls `after_signing` with the key.

    `signed` holds the keys whose first pass ended as a whole body, `put_calls` the requests whose
    sending pass did, and `refused` what a read of a body raised where the client expected bytes.
    """

    def __init__(
        self,
        *,
        signs_first: bool = False,
        after_signing: Callable[[str], None] | None = None,
        piece_bytes: int = SMALL_PIECE_BYTES,
    ) -> None:
        super().__init__()
        self._signs_first = signs_first
        self._after_signing = after_signing
        self._piece_bytes = piece_bytes
        self.signed: list[str] = []
        self.refused: list[tuple[str, Exception]] = []

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        try:
            if self._signs_first:
                self._sign(Key, Body)
            sent = _read_through_in_pieces(Body, self._piece_bytes)
        except Exception as error:
            self.refused.append((Key, error))
            raise
        self.put_calls.append(PutCall(Bucket, Key, sent, ContentLength, ContentType, failed=False))
        return {"ResponseMetadata": {"HTTPStatusCode": HTTP_OK}}

    def _sign(self, key: str, body: Any) -> None:
        _read_through_in_pieces(body, self._piece_bytes)
        body.seek(0)
        self.signed.append(key)
        if self._after_signing is not None:
            self._after_signing(key)


def test_upload_when_a_client_reads_a_changed_file_to_its_end_should_refuse_it_there_and_never_put_the_manifest(
    generation,
):
    # Arrange: a client that sends as it reads; the second segment holds other bytes than the recorded ones
    _change_in_place(generation.output_root / "360p" / "segment_0001.ts")
    client = _StreamingS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: at the file's end the client got an error, in our own words, where it expected the body's end
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert "360p/segment_0001.ts" in failure.detail
    assert [key for key, _ in client.refused] == [f"{PREFIX}/360p/segment_0001.ts"]
    assert str(generation.output_root) not in str(client.refused[0][1])
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == [
        "master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts"
    ]


@pytest.mark.parametrize(
    ("changed", "sent_before_it"),
    [
        pytest.param(
            "360p/segment_0001.ts",
            ["master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts"],
            id="an-artifact",
        ),
        pytest.param(MANIFEST_NAME, list(_Generation.FILES), id="the-manifest"),
    ],
)
def test_upload_when_a_client_that_signs_first_reads_a_changed_file_should_refuse_it_before_it_is_sent(
    generation, changed, sent_before_it
):
    # Arrange: a client that reads each body through to sign it before it sends it
    _change_in_place(generation.output_root / changed)
    client = _StreamingS3(signs_first=True)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the pass that would have signed the file was refused, so nothing of it was sent
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert changed in failure.detail
    assert [key.removeprefix(f"{PREFIX}/") for key in client.signed] == sent_before_it
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == sent_before_it


def test_upload_when_a_client_that_signs_first_reads_each_file_twice_to_its_end_should_store_the_generation(
    generation,
):
    # Arrange
    client = _StreamingS3(signs_first=True)

    # Act
    uploaded = generation.upload(client)

    # Assert
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert {call.key: call.body for call in client.stored} == expected
    assert client.refused == []
    assert uploaded == sum(len(data) for data in expected.values())


def test_upload_when_a_file_changes_between_two_passes_of_a_client_should_refuse_the_pass_that_differs(generation):
    # Arrange: a file no buffer holds gets other bytes once the client has read it through, as recorded
    artifacts = _with_a_large_segment(generation)
    large_key = f"{PREFIX}/{LARGE_SEGMENT_PATH}"

    def change_the_large_segment(key: str) -> None:
        if key == large_key:
            _change_in_place(generation.output_root / LARGE_SEGMENT_PATH)

    client = _StreamingS3(signs_first=True, after_signing=change_the_large_segment, piece_bytes=LARGE_PIECE_BYTES)

    # Act
    failure = _failure_of(
        lambda: generation.upload(client, artifacts=artifacts, manifest=manifest_recording(artifacts))
    )

    # Assert: the first pass ended as recorded; the second was refused where it ended, and nothing follows
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert LARGE_SEGMENT_PATH in failure.detail
    assert client.signed[-1] == large_key
    assert [key for key, _ in client.refused] == [large_key]
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == list(_Generation.FILES)


class _AskingForMoreS3(FakeS3):
    """A FakeS3 that reads each body whole in one read and then asks it for more, the way `ask` does."""

    def __init__(self, ask: Callable[[Any], bytes]) -> None:
        super().__init__()
        self._ask = ask

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        sent = Body.read()
        self._ask(Body)
        self.put_calls.append(PutCall(Bucket, Key, sent, ContentLength, ContentType, failed=False))
        return {"ResponseMetadata": {"HTTPStatusCode": HTTP_OK}}


@pytest.mark.parametrize(
    "ask",
    [
        pytest.param(lambda body: body.read(), id="for-the-rest"),
        pytest.param(lambda body: body.read(None), id="for-the-rest-as-none"),
        pytest.param(lambda body: body.read(-1), id="for-the-rest-as-a-negative-amount"),
        pytest.param(lambda body: body.read(SMALL_PIECE_BYTES), id="for-a-piece"),
    ],
)
def test_upload_when_a_client_asks_a_changed_file_for_more_at_its_end_should_refuse_it_whatever_it_asked_for(
    generation, ask
):
    # Arrange: the first file holds other bytes than the recorded ones
    _change_in_place(generation.output_root / "master.m3u8")
    client = _AskingForMoreS3(ask)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the request for more was refused, so the client never came to its answer
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert client.put_calls == []


class _StubbornS3(FakeS3):
    """A FakeS3 that takes no refusal from a body: when a read raises it rewinds the body, asks it for a
    piece once more and answers 200 whatever came of that. `after_a_refusal` holds what came of it."""

    def __init__(self) -> None:
        super().__init__()
        self.after_a_refusal: list[bytes | Exception] = []

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        try:
            sent = _read_through_in_pieces(Body)
        except Exception:
            sent = b""
            self.after_a_refusal.append(self._asked_again(Body))
        self.put_calls.append(PutCall(Bucket, Key, sent, ContentLength, ContentType, failed=False))
        return {"ResponseMetadata": {"HTTPStatusCode": HTTP_OK}}

    @staticmethod
    def _asked_again(body: Any) -> bytes | Exception:
        body.seek(0)
        try:
            return body.read(SMALL_PIECE_BYTES)
        except Exception as error:
            return error


def test_upload_when_a_client_answers_200_after_a_body_refused_it_should_fail_and_give_it_no_more_bytes(generation):
    # Arrange: the first file holds other bytes than the recorded ones
    _change_in_place(generation.output_root / "master.m3u8")
    client = _StubbornS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the rewound body refused again, and the client's 200 does not count
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert "master.m3u8" in failure.detail
    assert len(client.after_a_refusal) == 1 and isinstance(client.after_a_refusal[0], Exception)
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]


def test_upload_when_the_storage_refuses_a_put_it_read_a_part_of_should_report_the_refusal_not_a_changed_file(
    generation,
):
    # Arrange: the storage answers the first put with a 403 after it read half of the file
    client = FakeS3()
    client.fail_put(1, ACCESS_DENIED)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert 0 < len(client.put_calls[0].body) < len(_Generation.FILES["master.m3u8"])
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert len(client.put_calls) == 1


class _AskingS3(FakeS3):
    """A FakeS3 that asks each body what a client may ask a binary file, before and after it read it."""

    def __init__(self) -> None:
        super().__init__()
        self.told: list[tuple[bool, bool, int, int]] = []
        self.offered: list[tuple[bool, bool]] = []

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        readable, seekable, position_before = Body.readable(), Body.seekable(), Body.tell()
        answer = super().put_object(
            Bucket=Bucket, Key=Key, Body=Body, ContentLength=ContentLength, ContentType=ContentType
        )
        self.told.append((readable, seekable, position_before, Body.tell()))
        self.offered.append((hasattr(Body, "fileno"), hasattr(Body, "name")))
        return answer


def test_upload_when_a_client_asks_a_body_what_a_file_can_tell_should_answer_as_the_file_and_name_no_path(
    generation,
):
    # Arrange
    client = _AskingS3()
    sizes = [len(data) for data in (*_Generation.FILES.values(), _Generation.MANIFEST)]

    # Act
    generation.upload(client)

    # Assert: readable, seekable, at its start and then at its end; no descriptor and no name to go around it
    assert client.told == [(True, True, 0, size) for size in sizes]
    assert set(client.offered) == {(False, False)}


def _files_held_open(generation: _Generation) -> list[str]:
    """The files of the generation, the manifest included, this process holds an open descriptor of."""
    return [path for path in (*_Generation.FILES, MANIFEST_NAME) if _holds_open(generation.output_root / path)]


def test_upload_when_the_generation_was_stored_should_leave_none_of_its_files_open(generation):
    # Arrange
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert _files_held_open(generation) == []


def _refused_at_the_third_put(generation: _Generation) -> FakeS3:
    client = FakeS3()
    client.fail_put(3, ACCESS_DENIED)
    return client


def _unreachable_from_the_third_put(generation: _Generation) -> FakeS3:
    client = FakeS3()
    client.fail_put(3, UNREACHABLE, and_every_later_one=True)
    return client


def _holding_a_changed_file(generation: _Generation) -> FakeS3:
    _change_in_place(generation.output_root / "360p" / "playlist.m3u8")
    return _StreamingS3()


@pytest.mark.parametrize(
    "failing",
    [
        pytest.param(_refused_at_the_third_put, id="a-refusal"),
        pytest.param(_unreachable_from_the_third_put, id="a-put-that-keeps-failing"),
        pytest.param(_holding_a_changed_file, id="a-changed-file"),
    ],
)
def test_upload_when_it_failed_should_leave_none_of_the_generations_files_open(generation, failing):
    # Arrange: an upload that fails at its third object, each time it is tried
    client = failing(generation)

    # Act
    with pytest.raises(WorkerFailure):
        generation.upload(client)

    # Assert
    assert _files_held_open(generation) == []


def test_upload_when_a_put_fails_once_should_send_the_whole_file_again_and_finish(generation):
    # Arrange: the fourth request breaks after half of the segment was read
    client = FakeS3()
    client.fail_put(4, UNREACHABLE)
    segment = _Generation.FILES["360p/segment_0000.ts"]

    # Act
    uploaded = generation.upload(client)

    # Assert
    broken, again = client.put_calls[3], client.put_calls[4]
    assert (broken.key, broken.failed) == (f"{PREFIX}/360p/segment_0000.ts", True)
    assert 0 < len(broken.body) < len(segment)
    assert (again.key, again.failed, again.body) == (f"{PREFIX}/360p/segment_0000.ts", False, segment)
    assert [call.key for call in client.stored][-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert uploaded == sum(len(call.body) for call in client.stored)


def _refused(code: str, status: int) -> ClientError:
    """An error answer of the storage to a PutObject; its text holds what must never be told."""
    return client_error(code, status, "PutObject", ERROR_TEXT_CANARY)


def _access_denied_with_metadata(metadata: object) -> ClientError:
    """An `AccessDenied` that holds `metadata` where a storage's answer names its HTTP status."""
    error = _refused("AccessDenied", HTTP_FORBIDDEN)
    error.response["ResponseMetadata"] = metadata
    return error


# Failures that can pass: the same request may be accepted when it is sent again.
PASSING_FAILURES = [
    pytest.param(UNREACHABLE, "EndpointConnectionError", id="no-answer"),
    pytest.param(ReadTimeoutError(endpoint_url=ENDPOINT_CANARY), "ReadTimeoutError", id="an-answer-that-never-came"),
    pytest.param(
        _refused("InternalError", HTTP_INTERNAL_SERVER_ERROR), "ClientError InternalError", id="500-internal-error"
    ),
    pytest.param(_refused("SlowDown", HTTP_SERVICE_UNAVAILABLE), "ClientError SlowDown", id="503-slow-down"),
    pytest.param(_refused("InternalError", HTTP_OK), "ClientError InternalError", id="an-error-inside-a-200"),
    pytest.param(_refused("Found", HTTP_FOUND), "ClientError Found", id="302"),
    pytest.param(
        _refused("TemporaryRedirect", HTTP_TEMPORARY_REDIRECT),
        "ClientError TemporaryRedirect",
        id="307-temporary-redirect",
    ),
    pytest.param(_refused("Timeout", HTTP_REQUEST_TIMEOUT), "ClientError Timeout", id="408"),
    pytest.param(_refused("OperationAborted", HTTP_CONFLICT), "ClientError OperationAborted", id="409"),
    pytest.param(_refused("TooManyRequests", HTTP_TOO_MANY_REQUESTS), "ClientError TooManyRequests", id="429"),
    pytest.param(_refused("RequestTimeout", HTTP_BAD_REQUEST), "ClientError RequestTimeout", id="400-request-timeout"),
    pytest.param(
        _refused("RequestTimeTooSkewed", HTTP_FORBIDDEN), "ClientError RequestTimeTooSkewed", id="403-clock-skew"
    ),
    pytest.param(_refused("IncompleteBody", HTTP_BAD_REQUEST), "ClientError IncompleteBody", id="400-incomplete-body"),
    pytest.param(_refused("BadDigest", HTTP_BAD_REQUEST), "ClientError BadDigest", id="400-bad-digest"),
    pytest.param(
        _refused("XAmzContentSHA256Mismatch", HTTP_BAD_REQUEST),
        "ClientError XAmzContentSHA256Mismatch",
        id="400-content-sha256-mismatch",
    ),
    pytest.param(_refused("SlowDown", HTTP_BAD_REQUEST), "ClientError SlowDown", id="400-slow-down"),
    pytest.param(
        ClientError({"Error": {"Code": "AccessDenied", "Message": ERROR_TEXT_CANARY}}, "PutObject"),
        "ClientError AccessDenied",
        id="an-answer-without-metadata",
    ),
    pytest.param(_access_denied_with_metadata(None), "ClientError AccessDenied", id="metadata-of-no-shape"),
    pytest.param(_access_denied_with_metadata({}), "ClientError AccessDenied", id="an-answer-without-a-status"),
    pytest.param(
        _access_denied_with_metadata({"HTTPStatusCode": str(HTTP_FORBIDDEN)}),
        "ClientError AccessDenied",
        id="a-status-as-text",
    ),
    pytest.param(
        _access_denied_with_metadata({"HTTPStatusCode": True}), "ClientError AccessDenied", id="a-status-as-a-flag"
    ),
]
# Refusals no second request changes: the storage's redirects and client errors, and what the client
# itself refuses to send.
FINAL_REFUSALS = [
    pytest.param(ACCESS_DENIED, "ClientError AccessDenied", id="403-access-denied"),
    pytest.param(
        _refused("InvalidAccessKeyId", HTTP_FORBIDDEN), "ClientError InvalidAccessKeyId", id="403-unknown-key"
    ),
    pytest.param(_refused("NoSuchBucket", HTTP_NOT_FOUND), "ClientError NoSuchBucket", id="404-no-such-bucket"),
    pytest.param(
        _refused("PermanentRedirect", HTTP_MOVED_PERMANENTLY), "ClientError PermanentRedirect", id="301-redirect"
    ),
    pytest.param(
        _refused("MultipleChoices", HTTP_MULTIPLE_CHOICES), "ClientError MultipleChoices", id="300-the-lowest-status"
    ),
    pytest.param(
        _refused("ClientClosedRequest", HTTP_LAST_CLIENT_ERROR),
        "ClientError ClientClosedRequest",
        id="499-the-highest-status",
    ),
    pytest.param(
        ClientError(
            {"Error": {"Message": ERROR_TEXT_CANARY}, "ResponseMetadata": {"HTTPStatusCode": HTTP_BAD_REQUEST}},
            "PutObject",
        ),
        "ClientError",
        id="400-without-an-error-code",
    ),
    pytest.param(
        ParamValidationError(report=ERROR_TEXT_CANARY), "ParamValidationError", id="parameters-the-client-refuses"
    ),
]


@pytest.mark.parametrize(("error", "named"), FINAL_REFUSALS)
def test_upload_when_a_put_is_refused_for_good_should_not_send_it_again_and_fail_as_not_retryable(
    generation, caplog, error, named
):
    # Arrange: only the third request is refused, so a second try of it would be accepted
    caplog.set_level(logging.DEBUG)
    client = FakeS3()
    client.fail_put(3, error)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: sent once, waited for nothing, nothing after it, and named by its class and its code only
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == [
        "master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8"
    ]
    assert generation.waits.asked == []
    assert failure.detail == f"an object could not be stored: {named}"
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    for told in (failure.detail, str(raised.value), caplog.text):
        assert "example.invalid" not in told
        assert SECRET_CANARY not in told
        assert "://" not in told
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize(
    ("code", "named"),
    [
        pytest.param("", "ClientError", id="empty"),
        pytest.param("Access Denied", "ClientError", id="a-space"),
        pytest.param("AccessDenied\n", "ClientError", id="a-trailing-line-break"),
        pytest.param("Access_Denied", "ClientError", id="an-underscore"),
        pytest.param("Access-Denied", "ClientError", id="a-hyphen"),
        pytest.param("<AccessDenied/>", "ClientError", id="markup"),
        pytest.param("Acc\u00e8sRefus\u00e9", "ClientError", id="letters-outside-ascii"),
        pytest.param("\u0664\u0660\u0663", "ClientError", id="digits-outside-ascii"),
        pytest.param(f"AccessDenied:{SECRET_CANARY}", "ClientError", id="a-colon-and-more"),
        pytest.param("A" * 65, "ClientError", id="sixty-five-characters"),
        pytest.param("A" * 64, "ClientError " + "A" * 64, id="sixty-four-characters"),
        pytest.param("Invalid.Argument.2", "ClientError Invalid.Argument.2", id="letters-digits-and-dots"),
    ],
)
def test_upload_when_a_refusal_names_an_error_code_should_keep_it_only_where_it_has_the_shape_of_a_code(
    generation, caplog, code, named
):
    # Arrange: the storage refuses the first put for good, and names its error with `code`
    caplog.set_level(logging.DEBUG)
    client = FakeS3()
    client.fail_put(1, _refused(code, HTTP_FORBIDDEN))

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: a code of letters, digits and dots, 64 at most, is kept; any other value is no code at all
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert failure.detail == f"an object could not be stored: {named}"
    assert [record.getMessage().partition(": ")[2] for record in caplog.records] == [named]


def test_upload_when_a_passing_failures_error_code_has_not_the_shape_of_a_code_should_tell_it_in_no_attempt(
    generation, caplog
):
    # Arrange: every put of the first object is answered with a 500 whose error code is no code
    caplog.set_level(logging.DEBUG)
    client = FakeS3()
    client.fail_put(1, _refused("Internal Error, try later", HTTP_INTERNAL_SERVER_ERROR), and_every_later_one=True)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: sent again three times as any passing failure, and neither a warning nor the failure tells the value
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert failure.detail == "an object could not be stored: ClientError"
    assert len(client.put_calls) == 4
    assert [record.getMessage().rpartition(": ")[2] for record in caplog.records] == ["ClientError"] * 3


@pytest.mark.parametrize("error", [pytest.param(case.values[0], id=case.id) for case in PASSING_FAILURES])
def test_upload_when_a_put_fails_once_in_a_way_that_can_pass_should_send_it_again_and_finish(generation, error):
    # Arrange: only the third request fails
    client = FakeS3()
    client.fail_put(3, error)

    # Act
    uploaded = generation.upload(client)

    # Assert
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls][2:4] == ["360p/playlist.m3u8"] * 2
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert uploaded == sum(len(call.body) for call in client.stored)


@pytest.mark.parametrize(("error", "named"), PASSING_FAILURES)
def test_upload_when_a_put_keeps_failing_should_give_up_after_three_more_requests_and_never_put_the_manifest(
    generation, caplog, fixed_factor, error, named
):
    # Arrange: every request from the third on fails
    caplog.set_level(logging.DEBUG)
    client = FakeS3()
    client.fail_put(3, error, and_every_later_one=True)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: four requests of the object, a wait before the second, third and fourth (the nominal 0.5,
    # 2 and 8 s, each times the factor 0.75), and one warning for each that says how long
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == [
        "master.m3u8",
        "thumbnail.jpg",
        "360p/playlist.m3u8",
        "360p/playlist.m3u8",
        "360p/playlist.m3u8",
        "360p/playlist.m3u8",
    ]
    assert generation.waits.asked == [0.375, 1.5, 6.0]
    warnings = [record.getMessage() for record in caplog.records]
    assert [record.levelno for record in caplog.records] == [logging.WARNING] * 3
    assert "0.375 s" in warnings[0] and "1.500 s" in warnings[1] and "6.000 s" in warnings[2]
    assert named in failure.detail
    for told in (failure.detail, str(raised.value), caplog.text):
        assert "example.invalid" not in told
        assert SECRET_CANARY not in told
        assert "://" not in told
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize(
    ("failed_requests", "expected_waits"),
    [
        pytest.param(1, [0.375], id="once"),
        pytest.param(2, [0.375, 1.5], id="twice"),
        pytest.param(3, [0.375, 1.5, 6.0], id="three-times"),
    ],
)
def test_upload_when_a_put_fails_in_a_passing_way_and_is_then_taken_should_wait_before_each_request_and_finish(
    generation, fixed_factor, failed_requests, expected_waits
):
    # Arrange: the put of the third object fails `failed_requests` times before the storage takes it
    client = FakeS3()
    for call_number in range(3, 3 + failed_requests):
        client.fail_put(call_number, UNREACHABLE)

    # Act
    uploaded = generation.upload(client)

    # Assert: the nominal 0.5 s, then 2 s, then 8 s, each times the factor 0.75; the generation is stored
    assert generation.waits.asked == expected_waits
    requested = [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls]
    assert requested.count("360p/playlist.m3u8") == failed_requests + 1
    assert [call.key for call in client.stored][-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert uploaded == sum(len(call.body) for call in client.stored)


def test_upload_when_it_waits_should_draw_the_factor_of_each_wait_anew_between_a_half_and_one(generation):
    # Arrange: fifty uploads whose third put fails three times, each wait with the jitter's own draw
    runs = 50

    # Act
    for _ in range(runs):
        client = FakeS3()
        for call_number in (3, 4, 5):
            client.fail_put(call_number, UNREACHABLE)
        generation.upload(client)

    # Assert: every wait is its nominal 0.5, 2 or 8 s times a factor of its own from 0.5 to 1
    waits = generation.waits.asked
    factors = [wait / nominal for wait, nominal in zip(waits, [0.5, 2.0, 8.0] * runs)]
    assert len(waits) == 3 * runs
    assert all(0.5 <= factor <= 1.0 for factor in factors)
    assert min(factors) < 0.6 and max(factors) > 0.9
    assert len(set(factors)) > runs


def _slow_down(retry_after: object) -> ClientError:
    """A `SlowDown` answer (503) to a PutObject whose headers, kept as botocore keeps them (the names in
    lower case, in the answer's metadata), hold `retry_after` as their `Retry-After`."""
    error = _refused("SlowDown", HTTP_SERVICE_UNAVAILABLE)
    error.response["ResponseMetadata"]["HTTPHeaders"] = {"retry-after": retry_after, "content-length": "0"}
    return error


@pytest.mark.parametrize(
    ("retry_after", "expected_wait"),
    [
        pytest.param("5", 5.0, id="five-seconds"),
        pytest.param("0005", 5.0, id="five-seconds-with-leading-zeros"),
        pytest.param("30", 30.0, id="exactly-the-most"),
        pytest.param("31", 30.0, id="a-second-more-than-the-most"),
        pytest.param("3600", 30.0, id="an-hour"),
        pytest.param("9" * 5000, 30.0, id="more-digits-than-python-converts"),
        pytest.param("0", 0.375, id="no-seconds-at-all"),
    ],
)
def test_upload_when_a_failed_puts_answer_asks_to_retry_after_whole_seconds_should_wait_that_long_and_30_s_at_most(
    generation, fixed_factor, retry_after, expected_wait
):
    # Arrange: the storage answers the third put once with a `SlowDown` that asks for a wait; without it
    # the wait would be 0.375 s
    client = FakeS3()
    client.fail_put(3, _slow_down(retry_after))

    # Act
    generation.upload(client)

    # Assert
    assert generation.waits.asked == [expected_wait]
    assert len(client.stored) == len(_Generation.FILES) + 1


@pytest.mark.parametrize(
    "retry_after",
    [
        pytest.param("-5", id="a-negative-number"),
        pytest.param("+5", id="a-signed-number"),
        pytest.param("5.5", id="a-fraction"),
        pytest.param("5s", id="a-number-with-a-unit"),
        pytest.param(" 5", id="a-number-after-a-space"),
        pytest.param("Wed, 21 Oct 2026 07:28:00 GMT", id="an-http-date"),
        pytest.param("", id="empty"),
        pytest.param("٥", id="a-digit-outside-ascii"),
        pytest.param(5, id="a-number-that-is-no-text"),
        pytest.param(None, id="nothing"),
        pytest.param(["5"], id="a-list"),
    ],
)
def test_upload_when_a_failed_puts_retry_after_is_no_whole_number_of_seconds_should_ignore_it(
    generation, fixed_factor, retry_after
):
    # Arrange
    client = FakeS3()
    client.fail_put(3, _slow_down(retry_after))

    # Act
    generation.upload(client)

    # Assert: the wait that was due anyway, 0.5 s times the factor 0.75
    assert generation.waits.asked == [0.375]
    assert len(client.stored) == len(_Generation.FILES) + 1


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(None, id="no-headers"),
        pytest.param("retry-after: 5", id="headers-as-text"),
        pytest.param([("retry-after", "5")], id="headers-as-a-list"),
        pytest.param({}, id="headers-without-it"),
    ],
)
def test_upload_when_a_failed_puts_answer_holds_its_headers_in_no_known_shape_should_wait_as_without_them(
    generation, fixed_factor, headers
):
    # Arrange
    slow_down = _refused("SlowDown", HTTP_SERVICE_UNAVAILABLE)
    slow_down.response["ResponseMetadata"]["HTTPHeaders"] = headers
    client = FakeS3()
    client.fail_put(3, slow_down)

    # Act
    generation.upload(client)

    # Assert
    assert generation.waits.asked == [0.375]
    assert len(client.stored) == len(_Generation.FILES) + 1


def test_upload_when_the_storage_asks_for_less_than_the_wait_that_is_due_should_keep_the_longer_wait(
    generation, fixed_factor
):
    # Arrange: the third put fails three times, and each answer asks for one second
    client = FakeS3()
    for call_number in (3, 4, 5):
        client.fail_put(call_number, _slow_down("1"))

    # Act
    generation.upload(client)

    # Assert: one second where 0.375 s was due, then the 1.5 s and the 6 s that were due anyway
    assert generation.waits.asked == [1.0, 1.5, 6.0]


def test_upload_when_a_failed_puts_answer_carries_headers_should_tell_nothing_of_them(
    generation, fixed_factor, caplog
):
    # Arrange: every put of the third object is answered with headers that hold what must never be told,
    # one of them where the seconds to wait would stand
    caplog.set_level(logging.DEBUG)
    slow_down = _refused("SlowDown", HTTP_SERVICE_UNAVAILABLE)
    slow_down.response["ResponseMetadata"]["HTTPHeaders"] = {
        "retry-after": ERROR_TEXT_CANARY,
        "location": f"{ENDPOINT_CANARY}/{OUTPUT_BUCKET}/{PREFIX}/360p/playlist.m3u8",
    }
    client = FakeS3()
    client.fail_put(3, slow_down, and_every_later_one=True)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: the warnings of the three further attempts and the failure name the error and the waits only
    assert generation.waits.asked == [0.375, 1.5, 6.0]
    assert len(caplog.records) == 3
    for told in (raised.value.failure.detail, str(raised.value), caplog.text):
        assert "example.invalid" not in told
        assert SECRET_CANARY not in told
        assert "://" not in told
        assert PREFIX not in told


@pytest.mark.parametrize(
    ("seconds_left", "expected_waits", "expected_requests"),
    [
        pytest.param(30.0, [], 1, id="less-than-the-first-wait"),
        pytest.param(40.0, [], 1, id="exactly-the-first-wait"),
        pytest.param(100.0, [40.0], 2, id="less-than-the-second-wait"),
        pytest.param(400.0, [40.0, 160.0], 3, id="less-than-the-third-wait"),
    ],
)
def test_upload_when_the_next_wait_would_not_end_before_the_deadline_should_neither_wait_nor_send_again(
    generation, long_waits, seconds_left, expected_waits, expected_requests
):
    # Arrange: every put of the first object fails in a passing way, and the deadline has `seconds_left`
    # whenever it is asked; the waits due are 40 s, 160 s, then 640 s (the nominal 0.5, 2 and 8 s, each
    # times the factor 80)
    client = FakeS3()
    client.fail_put(1, UNREACHABLE, and_every_later_one=True)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client, deadline=_TimeLeft(seconds_left))

    # Assert: the last request's own retryable failure, not a deadline failure, and no manifest
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert generation.waits.asked == expected_waits
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"] * expected_requests
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_upload_when_the_storage_asks_for_a_wait_that_would_not_end_before_the_deadline_should_not_send_again(
    generation, fixed_factor
):
    # Arrange: the storage asks for thirty seconds, and the deadline has twenty left, which the 0.375 s
    # that were due anyway would fit
    client = FakeS3()
    client.fail_put(1, _slow_down("30"), and_every_later_one=True)

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=_TimeLeft(20.0)))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert generation.waits.asked == []
    assert len(client.put_calls) == 1


def test_upload_when_a_wait_that_fitted_ends_after_the_deadline_passed_should_fail_as_a_deadline_and_not_send_again(
    generation, fixed_factor
):
    # Arrange: every put of the first object fails in a passing way with a minute left on the deadline,
    # and the deadline passes while the upload waits to send it again
    client = FakeS3()
    client.fail_put(1, UNREACHABLE, and_every_later_one=True)
    deadline = _HandPassedDeadline()
    asked: list[float] = []

    def wait_past_the_deadline(seconds: float) -> None:
        asked.append(seconds)
        deadline.pass_now()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=deadline, sleep=wait_past_the_deadline))

    # Assert: the wait fitted when it was decided, so it was made; what it ran into is the deadline,
    # which is final, and the object is not sent a second time
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert asked == [0.375]
    assert len(client.put_calls) == 1


def test_upload_when_a_put_fails_in_a_passing_way_and_its_repeat_is_refused_for_good_should_stop_there_as_refused(
    generation, fixed_factor
):
    # Arrange: the first put of the first object never reaches the storage, and its repeat is refused
    client = FakeS3()
    client.fail_put(1, UNREACHABLE)
    client.fail_put(2, ACCESS_DENIED)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: one wait, before the repeat; the refusal ends the upload at once and for good
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert generation.waits.asked == [0.375]
    assert [(call.key, call.failed) for call in client.put_calls] == [(f"{PREFIX}/master.m3u8", True)] * 2


def test_upload_when_no_sleep_is_given_should_refuse_it_before_any_request(generation):
    # Arrange
    client = FakeS3()

    # Act
    with pytest.raises(TypeError):
        upload_generation(
            client,
            bucket=OUTPUT_BUCKET,
            prefix=PREFIX,
            output_root=generation.output_root,
            artifacts=generation.artifacts,
            manifest=generation.manifest,
            deadline=_deadline(),
        )

    # Assert
    assert client.put_calls == []


def test_upload_when_the_deadline_has_passed_should_fail_before_any_request(generation):
    # Arrange
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=_passed_deadline()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert client.put_calls == []


def test_upload_when_the_deadline_passes_between_two_objects_should_stop_before_the_manifest(generation):
    # Arrange: the deadline has not passed before the first object and has before the second
    deadline = ScriptedDeadline(False, True)
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=deadline))

    # Assert
    assert failure.code == "upload_deadline_exceeded"
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]
    assert deadline.asked == 2


class _PausingS3(FakeS3):
    """A FakeS3 whose PutObject of one key calls `pause` in the middle of the request.

    It pauses before it reads the body: a request that has got nowhere yet. With `reads_first` it pauses
    once the storage has taken the bytes and its answer is still to come; that answer then is
    `late_error`, or a 200 when that is None. With `pause` a _Hold's keep(), the request does not return
    until the test lets it go; with a hand-passed deadline's pass_now(), the deadline passes inside it.
    """

    def __init__(
        self,
        pause: Callable[[], None],
        key: str,
        *,
        reads_first: bool = False,
        late_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._pause = pause
        self._paused_key = key
        self._reads_first = reads_first
        self._late_error = late_error

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        if Key == self._paused_key and not self._reads_first:
            self._pause()
        answer = super().put_object(
            Bucket=Bucket, Key=Key, Body=Body, ContentLength=ContentLength, ContentType=ContentType
        )
        if Key == self._paused_key and self._reads_first:
            self._pause()
            if self._late_error is not None:
                raise self._late_error
        return answer


def test_upload_when_a_put_never_returns_should_fail_at_the_deadline_and_never_put_the_manifest(
    generation, hold, requests_settle_first
):
    # Arrange: the storage never answers the put of the thumbnail, and the deadline passes a moment
    # after that put began
    client = _PausingS3(hold.keep, f"{PREFIX}/thumbnail.jpg")
    started_at = time.monotonic()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=_PassedSoonAfter(hold.entered)))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert hold.entered.is_set()
    assert time.monotonic() - started_at < PROMPT_S
    hold.release()
    assert f"{PREFIX}/{MANIFEST_NAME}" not in [call.key for call in client.put_calls]


def test_upload_when_a_put_was_abandoned_at_the_deadline_should_close_its_file_and_give_the_request_no_more_of_it(
    generation, hold, requests_settle_first, caplog, capfd, monkeypatch
):
    # Arrange: the first put gets nowhere before the deadline, which passes a moment after the put
    # began, and reads its body only once it is let go
    caplog.set_level(logging.DEBUG)
    client = _PausingS3(hold.keep, f"{PREFIX}/master.m3u8")
    printed_by_a_thread: list[Any] = []

    # Act: the job gives up while the request still runs; only then is the request let go
    with monkeypatch.context() as patched:
        patched.setattr(threading, "excepthook", printed_by_a_thread.append)
        failure = _failure_of(lambda: generation.upload(client, deadline=_PassedSoonAfter(hold.entered)))
        held_open_under_the_running_request = _holds_open(generation.output_root / "master.m3u8")
        hold.release()

    # Assert: the file was closed at once; the storage got none of its bytes, and nothing surfaces of it
    assert failure.code == "upload_deadline_exceeded"
    assert hold.entered.is_set()
    assert not held_open_under_the_running_request
    assert client.put_calls == []
    assert printed_by_a_thread == []
    assert caplog.records == []
    shown = capfd.readouterr()
    assert (shown.out, shown.err) == ("", "")


def test_upload_when_a_put_stalls_and_the_next_one_is_taken_should_wait_send_the_object_again_and_finish(
    generation, hold, fixed_factor, monkeypatch
):
    # Arrange: the storage never answers the first put of the first object, with a minute left on the
    # deadline and a ceiling of 50 ms; it takes every later put at once
    stalled = _StalledFirst(hold)
    client = _PausingS3(stalled.request, f"{PREFIX}/master.m3u8")
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)

    # Act: the kept request is let go only once the upload has finished without it
    uploaded = generation.upload(client, sleep=stalled.sleep)
    hold.release()

    # Assert: sent again after the first wait (0.5 s times the factor 0.75); the abandoned request got
    # none of the file, so the object the storage holds is the one request that was waited for
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert hold.entered.is_set()
    assert stalled.asked == [0.375]
    assert {call.key: call.body for call in client.stored} == expected
    assert [call.key for call in client.put_calls].count(f"{PREFIX}/master.m3u8") == 1
    assert uploaded == sum(len(data) for data in expected.values())


def test_upload_when_a_put_stalls_after_the_storage_took_its_whole_body_should_send_it_again_and_store_the_same_object(
    generation, hold, fixed_factor, requests_settle_first, monkeypatch
):
    # Arrange: the storage takes every byte of the first put of the first object and then withholds its
    # answer, with a minute left on the deadline and a ceiling of 50 ms; it takes every later put at once.
    # The ceiling's 50 ms begin once the storage has the body: no slow thread can lose it the bytes
    stalled = _StalledFirst(hold)
    master_key = f"{PREFIX}/master.m3u8"
    client = _PausingS3(stalled.request, master_key, reads_first=True)
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)

    # Act: the kept request is let go only once the upload has finished without its answer
    uploaded = generation.upload(client, sleep=stalled.sleep)
    hold.release()

    # Assert: the object arrived twice, the abandoned put's and its repeat's, with the same bytes both
    # times, so whichever of them lands the object is the same; the upload went on to its end, the
    # manifest last, and counts the object once
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    puts_of_the_object = [call for call in client.put_calls if call.key == master_key]
    assert hold.entered.is_set()
    assert stalled.asked == [0.375]
    assert [(call.body, call.failed) for call in puts_of_the_object] == [(b"#EXTM3U\n#EXT-X-VERSION:3\n", False)] * 2
    assert {call.key: call.body for call in client.stored} == expected
    assert client.put_calls[-1].key == f"{PREFIX}/{MANIFEST_NAME}"
    assert len(client.put_calls) == len(expected) + 1
    assert uploaded == sum(len(data) for data in expected.values())


def test_upload_when_a_put_stalls_on_every_attempt_should_fail_retryable_as_stalled_and_close_its_file(
    generation, hold, fixed_factor, short_ceiling, caplog, capfd, monkeypatch
):
    # Arrange: the storage never answers a put of the first object, with a minute left on the deadline
    # and a ceiling of some 50 ms
    caplog.set_level(logging.DEBUG)
    kept = _KeptRequests(hold)
    client = _PausingS3(kept, f"{PREFIX}/master.m3u8")
    printed_by_a_thread: list[Any] = []
    started_at = time.monotonic()

    # Act: the upload gives up while its requests still run; only then are they let go
    with monkeypatch.context() as patched:
        patched.setattr(threading, "excepthook", printed_by_a_thread.append)
        with pytest.raises(WorkerFailure) as raised:
            generation.upload(client)
        gave_up_after_s = time.monotonic() - started_at
        held_open_under_the_running_requests = _holds_open(generation.output_root / "master.m3u8")
        assert hold.entered.wait(PROMPT_S)
        hold.release()

    # Assert: a plain typed failure that passes, not a deadline, after four requests of the object with
    # the three waits between them
    failure = raised.value.failure
    assert type(raised.value) is WorkerFailure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_stalled", True)
    assert generation.waits.asked == [0.375, 1.5, 6.0]
    assert kept.arrived == 4
    assert gave_up_after_s < PROMPT_S
    # The file was closed at once; no request got a byte of it, and nothing follows, the manifest included
    assert not held_open_under_the_running_requests
    assert client.put_calls == []
    # Nothing surfaces of the abandoned requests: the only lines are the three warnings of the attempts
    assert printed_by_a_thread == []
    assert [record.levelno for record in caplog.records] == [logging.WARNING] * 3
    assert all("upload_stalled" in record.getMessage() for record in caplog.records)
    shown = capfd.readouterr()
    assert (shown.out, shown.err) == ("", "")
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_upload_when_a_put_stalls_and_the_next_ones_fail_should_give_them_one_count_of_attempts_together(
    generation, hold, fixed_factor, monkeypatch
):
    # Arrange: the storage never answers the first put of the first object, and no later put reaches it
    stalled = _StalledFirst(hold)
    client = _PausingS3(stalled.request, f"{PREFIX}/master.m3u8")
    client.fail_put(1, UNREACHABLE, and_every_later_one=True)
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)

    # Act
    failure = _failure_of(lambda: generation.upload(client, sleep=stalled.sleep))

    # Assert: four requests in all, the stalled one and three that failed, and the last one's own failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert stalled.asked == [0.375, 1.5, 6.0]
    assert [(call.key, call.failed) for call in client.put_calls] == [(f"{PREFIX}/master.m3u8", True)] * 3


def test_upload_when_the_deadline_passes_under_a_put_that_outlasts_its_ceiling_should_fail_as_a_deadline(
    generation, hold, short_ceiling
):
    # Arrange: a put that is never answered under a ceiling of some 50 ms, and a deadline that has a
    # minute left when the put starts and none by the time the ceiling is reached
    client = _PausingS3(hold.keep, f"{PREFIX}/master.m3u8")

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=_PassedOnceAsked()))
    assert hold.entered.wait(PROMPT_S)
    hold.release()

    # Assert: the deadline failure, final as ever, whatever the ceiling: no wait and no second request
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert generation.waits.asked == []
    assert client.put_calls == []


def test_upload_when_it_puts_an_object_should_hold_each_request_to_the_ceiling_of_its_recorded_size(
    generation, monkeypatch
):
    # Arrange: note the size each ceiling is asked for
    sizes: list[int] = []

    def noted_ceiling(size_bytes: int) -> float:
        sizes.append(size_bytes)
        return GENEROUS_CEILING_S

    monkeypatch.setattr(storage, "_request_ceiling_s", noted_ceiling)
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert: the recorded size of each file, in upload order, the manifest's last
    assert sizes == [len(data) for data in (*_Generation.FILES.values(), _Generation.MANIFEST)]
    assert sizes[0] == 25 and len(sizes) == 6


class _FileClosedUnderARead:
    """Stands for an opened artifact that is asked for bytes just as it is closed: the question passed the
    body's own check, then meets the closed file, and raises what a real file raises there. (A real
    file's close waits for a read under way; this stand-in lets the close through to reach that gap.)
    `reading` is set once a read of it is under way."""

    def __init__(self, opened: Any, reading: _Arrival) -> None:
        self._opened = opened
        self._was_closed = threading.Event()
        self._reading = reading

    def read(self, amount: int = -1) -> bytes:
        self._reading.set()
        self._was_closed.wait(KEPT_AT_MOST_S)
        return self._opened.read(amount)

    def close(self) -> None:
        self._opened.close()
        self._was_closed.set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._opened, name)


def test_upload_when_a_file_is_closed_under_a_read_of_an_abandoned_put_should_show_nothing_of_that_read(
    generation, monkeypatch, caplog, capfd
):
    # Arrange: the read of the first file's body is still under way when the job gives up and closes
    # it, at a deadline that passes a moment after that read began
    caplog.set_level(logging.DEBUG)
    open_the_artifact = storage._opened_artifact
    reading = _Arrival()

    def open_a_file_that_is_closed_under_its_read(upload: Any) -> _FileClosedUnderARead:
        return _FileClosedUnderARead(open_the_artifact(upload), reading)

    client = _StreamingS3()
    threads_before = set(threading.enumerate())
    printed_by_a_thread: list[Any] = []

    # Act: once the job has given up, wait for the request's thread to end
    with monkeypatch.context() as patched:
        patched.setattr(storage, "_opened_artifact", open_a_file_that_is_closed_under_its_read)
        patched.setattr(threading, "excepthook", printed_by_a_thread.append)
        settle_requests_first(patched, reading)
        failure = _failure_of(lambda: generation.upload(client, deadline=_PassedSoonAfter(reading)))
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(PROMPT_S)

    # Assert: the client was refused as after the deadline, not with the closed file's own error
    assert failure.code == "upload_deadline_exceeded"
    assert reading.is_set()
    assert [(key, isinstance(error, ValueError)) for key, error in client.refused] == [
        (f"{PREFIX}/master.m3u8", False)
    ]
    assert client.put_calls == []
    assert printed_by_a_thread == []
    assert caplog.records == []
    shown = capfd.readouterr()
    assert (shown.out, shown.err) == ("", "")


class _FileThatFailsUnderARead:
    """Stands for an opened artifact on a disk that fails: its reads give the file's bytes until the
    read numbered `failing_read` (the first is 1), which raises `error`, as every read after it does.
    With `only_that_read`, every read after it gives the file's bytes again."""

    def __init__(self, opened: Any, error: OSError, failing_read: int, only_that_read: bool) -> None:
        self._opened = opened
        self._error = error
        self._failing_read = failing_read
        self._only_that_read = only_that_read
        self._reads = 0

    def read(self, amount: int = -1) -> bytes:
        self._reads += 1
        if self._reads == self._failing_read or (self._reads > self._failing_read and not self._only_that_read):
            raise self._error
        return self._opened.read(amount)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._opened, name)


def _fail_the_reads_of_every_file(
    monkeypatch: pytest.MonkeyPatch, error: OSError, failing_read: int = 1, *, only_that_read: bool = False
) -> None:
    """Make the upload open each file of the generation as one whose reads fail (_FileThatFailsUnderARead)."""
    open_the_artifact = storage._opened_artifact

    def open_a_file_that_fails_under_a_read(upload: Any) -> _FileThatFailsUnderARead:
        return _FileThatFailsUnderARead(open_the_artifact(upload), error, failing_read, only_that_read)

    monkeypatch.setattr(storage, "_opened_artifact", open_a_file_that_fails_under_a_read)


@pytest.mark.parametrize(
    ("error", "failing_read", "named"),
    [
        pytest.param(OSError(errno.EIO, f"cannot read {PATH_CANARY}"), 1, "EIO", id="its-first-read"),
        pytest.param(OSError(errno.EIO, f"cannot read {PATH_CANARY}"), 2, "EIO", id="a-read-after-its-bytes"),
        pytest.param(OSError(f"cannot read {PATH_CANARY}"), 1, "unknown", id="an-error-without-a-number"),
    ],
)
def test_upload_when_a_read_of_a_file_fails_while_the_upload_client_signs_it_should_fail_typed_by_path_and_errno_only(
    generation, monkeypatch, caplog, error, failing_read, named
):
    # Arrange: the first file cannot be read through while the real client hashes it to sign its request
    caplog.set_level(logging.DEBUG)
    _fail_the_reads_of_every_file(monkeypatch, error, failing_read)
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: a typed failure where the operating system's own error came out of the client; no request
    # of the file was ever to be sent, and nothing of the error's text or of the error itself is told
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert failure.detail == f"master.m3u8 changed before its upload: it could not be read: {named}"
    assert prepared == []
    assert generation.waits.asked == []
    assert PATH_CANARY not in caplog.text
    assert raised.value.__cause__ is None and raised.value.__context__ is None


class _WrappingS3(FakeS3):
    """A FakeS3 that reports whatever a body raised the way botocore reports what a body raised while it
    was being sent: wrapped in an HTTPClientError. `requests` counts its PutObject calls, and `wrapped`
    holds what it wrapped."""

    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.wrapped: list[Exception] = []

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        self.requests += 1
        try:
            return super().put_object(
                Bucket=Bucket, Key=Key, Body=Body, ContentLength=ContentLength, ContentType=ContentType
            )
        except Exception as error:
            self.wrapped.append(error)
            raise HTTPClientError(error=error) from None


def test_upload_when_a_read_of_a_file_fails_while_a_client_sends_it_should_fail_as_that_file_and_not_send_it_again(
    generation, monkeypatch, caplog
):
    # Arrange: the first file cannot be read, and the client reports that as a failed request of its own
    caplog.set_level(logging.DEBUG)
    _fail_the_reads_of_every_file(monkeypatch, OSError(errno.EIO, f"cannot read {PATH_CANARY}"))
    client = _WrappingS3()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: the file's own failure, not a failed request that is worth sending again, although the
    # client did wrap what the body raised in an error of its own
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert failure.detail == "master.m3u8 changed before its upload: it could not be read: EIO"
    assert client.requests == 1
    assert len(client.wrapped) == 1
    assert generation.waits.asked == []
    assert caplog.records == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    # What the body handed the client, which may log or keep what it is handed: an error in our own
    # words, and nothing of the operating system's own, whose text names a path
    (handed,) = client.wrapped
    assert not isinstance(handed, OSError)
    assert PATH_CANARY not in str(handed)
    assert handed.__cause__ is None and handed.__context__ is None


def test_upload_when_a_read_of_a_file_failed_should_refuse_every_later_read_of_it_though_the_file_would_read_again(
    generation, monkeypatch
):
    # Arrange: the first read of each file fails and every later one would give its bytes; the client
    # takes no refusal: it rewinds the body, asks it for a piece once more and answers 200
    _fail_the_reads_of_every_file(monkeypatch, OSError(errno.EIO, f"cannot read {PATH_CANARY}"), only_that_read=True)
    client = _StubbornS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: a body whose file failed once gives no more of that file, in our own words again: what a
    # client read of it afterwards could otherwise go out as the object
    assert failure.detail == "master.m3u8 changed before its upload: it could not be read: EIO"
    (asked_again,) = client.after_a_refusal
    assert isinstance(asked_again, Exception) and not isinstance(asked_again, OSError)
    assert PATH_CANARY not in str(asked_again)
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]


def test_upload_when_the_client_raises_an_os_error_of_its_own_should_not_take_it_for_an_error_of_the_file(generation):
    # Arrange: the first request fails with an OSError that no read of the file raised
    error = OSError(errno.EMFILE, "the client could not open a socket")
    client = FakeS3()
    client.fail_put(1, error)

    # Act
    with pytest.raises(OSError) as raised:
        generation.upload(client)

    # Assert: raised as it is, the very same error, and not as a file that changed
    assert raised.value is error
    assert len(client.put_calls) == 1


@pytest.mark.parametrize(
    "late_error",
    [
        pytest.param(None, id="a-late-200"),
        pytest.param(_refused("InternalError", HTTP_INTERNAL_SERVER_ERROR), id="a-late-error"),
    ],
)
def test_upload_when_the_manifests_answer_comes_after_the_deadline_should_fail_and_show_nothing_of_it(
    generation, hold, requests_settle_first, caplog, capfd, monkeypatch, late_error
):
    # Arrange: the storage takes the manifest's bytes and answers only once the job has given up, at a
    # deadline that passes a moment after the storage had them
    caplog.set_level(logging.DEBUG)
    manifest_key = f"{PREFIX}/{MANIFEST_NAME}"
    client = _PausingS3(hold.keep, manifest_key, reads_first=True, late_error=late_error)
    printed_by_a_thread: list[Any] = []

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(threading, "excepthook", printed_by_a_thread.append)
        failure = _failure_of(lambda: generation.upload(client, deadline=_PassedSoonAfter(hold.entered)))
        hold.release()

    # Assert: a late answer is no success, and nothing of the abandoned request surfaces
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert hold.entered.is_set()
    assert manifest_key in [call.key for call in client.put_calls]
    assert printed_by_a_thread == []
    assert caplog.records == []
    shown = capfd.readouterr()
    assert (shown.out, shown.err) == ("", "")


@pytest.mark.parametrize(
    "overdue_error",
    [
        pytest.param(None, id="an-overdue-200"),
        pytest.param(ACCESS_DENIED, id="an-overdue-refusal"),
        pytest.param(UNREACHABLE, id="an-overdue-failure-that-could-pass"),
    ],
)
def test_upload_when_the_manifests_put_ends_after_the_deadline_passed_should_fail_and_drop_how_it_ended(
    generation, caplog, overdue_error
):
    # Arrange: the deadline passes once the storage has the manifest's bytes; only then does the put end
    caplog.set_level(logging.DEBUG)
    deadline = _HandPassedDeadline()
    manifest_key = f"{PREFIX}/{MANIFEST_NAME}"
    client = _PausingS3(deadline.pass_now, manifest_key, reads_first=True, late_error=overdue_error)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client, deadline=deadline)

    # Assert: the manifest does not count as stored, and nothing of the overdue ending is told or acted on
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert [call.key for call in client.put_calls].count(manifest_key) == 1
    assert caplog.records == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None


class _DeadlinePassingS3(FakeS3):
    """A FakeS3 during whose every PutObject the deadline passes, before the body is read. `as_the_client`
    makes it report what the body then raised the way botocore does: wrapped in an HTTPClientError."""

    def __init__(self, deadline: _HandPassedDeadline, *, as_the_client: bool) -> None:
        super().__init__()
        self._deadline = deadline
        self._as_the_client = as_the_client
        self.requests = 0

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        self.requests += 1
        self._deadline.pass_now()
        try:
            return super().put_object(
                Bucket=Bucket, Key=Key, Body=Body, ContentLength=ContentLength, ContentType=ContentType
            )
        except Exception as error:
            if not self._as_the_client:
                raise
            raise HTTPClientError(error=error) from None


@pytest.mark.parametrize("as_the_client", [False, True], ids=["raised-as-it-is", "wrapped-by-the-client"])
def test_upload_when_the_deadline_passes_inside_a_put_should_end_as_a_deadline_and_not_as_a_failed_request(
    generation, caplog, as_the_client
):
    # Arrange: the deadline passes while the client is in the first request, before it read the body
    caplog.set_level(logging.DEBUG)
    deadline = _HandPassedDeadline()
    client = _DeadlinePassingS3(deadline, as_the_client=as_the_client)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client, deadline=deadline)

    # Assert: the body gave nothing, the request was not sent again and no client error travels with ours
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert (client.requests, client.put_calls) == (1, [])
    assert caplog.records == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_upload_when_no_thread_can_be_started_for_a_put_should_fail_retryably_before_any_request(
    generation, monkeypatch
):
    # Arrange: the operating system gives this process no more threads
    client = FakeS3()

    def no_thread(thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", no_thread)
        failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True
    )
    assert client.put_calls == []


def test_upload_when_progress_is_asked_for_should_report_the_running_total_after_each_object(generation):
    # Arrange
    client = FakeS3()
    totals: list[int] = []

    # Act
    uploaded = generation.upload(client, on_bytes=totals.append)

    # Assert
    sizes = [len(call.body) for call in client.stored]
    assert totals == [sum(sizes[: count + 1]) for count in range(len(sizes))]
    assert totals[-1] == uploaded


def test_upload_when_the_progress_callback_raises_should_still_finish(generation):
    # Arrange
    client = FakeS3()

    def broken_callback(total_bytes: int) -> None:
        raise RuntimeError("the progress display broke")

    # Act
    generation.upload(client, on_bytes=broken_callback)

    # Assert
    assert len(client.stored) == len(_Generation.FILES) + 1


# --- the count of abandoned requests ----------------------------------------------------------------
# The count belongs to the process and only grows, so every test compares two readings of it.

# What a deadline that never passes has left where a test wants one request and no repeat of it: more
# than a short ceiling, so the request outlasts its ceiling with time left, and less than the least wait
# before a request is made again (0.5 s times a factor of a half at least), so it is not made again.
NO_TIME_FOR_A_REPEAT_S = 0.2


def _read_of_the_preflight(hold: _Hold, tmp_path: Path, generation: _Generation, deadline: Deadline) -> None:
    source = DeadlineSource(_StubSource(len(OBJECT), _never_answered(hold)), deadline)
    source.read(0, CHUNK_BYTES)


def _read_of_the_download(hold: _Hold, tmp_path: Path, generation: _Generation, deadline: Deadline) -> None:
    source = _StubSource(CHUNK_BYTES, _never_answered(hold))
    download_source(source, destination=tmp_path / "source", deadline=deadline, sleep=_Waits())


def _put_of_the_upload(hold: _Hold, tmp_path: Path, generation: _Generation, deadline: Deadline) -> None:
    generation.upload(_PausingS3(hold.keep, f"{PREFIX}/master.m3u8"), deadline=deadline)


@pytest.mark.parametrize(
    "request_that_never_returns",
    [
        pytest.param(_read_of_the_preflight, id="a-read-of-the-preflight"),
        pytest.param(_read_of_the_download, id="a-read-of-the-download"),
        pytest.param(_put_of_the_upload, id="a-put-of-the-upload"),
    ],
)
def test_abandoned_requests_when_a_request_is_given_up_at_the_deadline_while_it_still_runs_should_grow_by_one(
    tmp_path, generation, hold, requests_settle_first, request_that_never_returns
):
    # Arrange
    counted_before = storage.abandoned_requests()

    # Act: the job stops waiting at a deadline that passes a moment after the request began; the
    # request is let go only when the test ends
    failure = _failure_of(
        lambda: request_that_never_returns(hold, tmp_path, generation, _PassedSoonAfter(hold.entered))
    )

    # Assert: the request was under way when the job gave it up
    assert failure.error_class == "deadline_exceeded"
    assert hold.entered.is_set()
    assert storage.abandoned_requests() - counted_before == 1


@pytest.mark.parametrize(
    ("request_that_never_returns", "stalled_as"),
    [
        pytest.param(
            _read_of_the_preflight, ("source_unavailable", "source_read_stalled", True), id="a-read-of-the-preflight"
        ),
        pytest.param(
            _read_of_the_download, ("source_unavailable", "source_read_stalled", True), id="a-read-of-the-download"
        ),
        pytest.param(_put_of_the_upload, ("output_write_failed", "upload_stalled", True), id="a-put-of-the-upload"),
    ],
)
def test_abandoned_requests_when_a_request_is_given_up_at_its_ceiling_with_time_left_on_the_deadline_should_grow_by_one(
    tmp_path, generation, hold, short_ceiling, request_that_never_returns, stalled_as
):
    # Arrange: a ceiling of some 50 ms and a deadline that always has 0.2 s left: the request outlasts
    # its ceiling with time left, and that time is too short for the wait before a repeat of it
    counted_before = storage.abandoned_requests()

    # Act: the job stops waiting at the ceiling; the request is let go only when the test ends
    failure = _failure_of(
        lambda: request_that_never_returns(hold, tmp_path, generation, _TimeLeft(NO_TIME_FOR_A_REPEAT_S))
    )
    assert hold.entered.wait(PROMPT_S)

    # Assert: it stalled, which is no deadline, and it left its thread behind just the same
    assert (failure.error_class, failure.code, failure.retryable) == stalled_as
    assert storage.abandoned_requests() - counted_before == 1


def test_abandoned_requests_when_two_requests_are_given_up_one_after_the_other_should_count_each_of_them(
    tmp_path, generation, hold
):
    # Arrange: each request is kept, under a deadline of its own that leaves it some 50 ms
    counted_before = storage.abandoned_requests()

    # Act
    first_failure = _failure_of(
        lambda: _read_of_the_preflight(hold, tmp_path, generation, _TimeLeft(PASSES_AFTER_S))
    )
    second_failure = _failure_of(
        lambda: _put_of_the_upload(hold, tmp_path, generation, _TimeLeft(PASSES_AFTER_S))
    )

    # Assert
    assert (first_failure.code, second_failure.code) == ("source_read_deadline_exceeded", "upload_deadline_exceeded")
    assert storage.abandoned_requests() - counted_before == 2


def test_abandoned_requests_when_a_read_stalls_and_its_repeat_arrives_should_grow_by_one_though_the_download_succeeds(
    tmp_path, small_chunks, hold, monkeypatch
):
    # Arrange: the first read of the first chunk never returns, with a minute left on the deadline and a
    # ceiling of 50 ms; every later read answers at once
    stalled = _StalledFirst(hold)

    def first_read_never_returns(start: int, length: int) -> bytes:
        stalled.request()
        return OBJECT[start : start + length]

    source = _StubSource(len(OBJECT), first_read_never_returns)
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)
    destination = tmp_path / "source"
    counted_before = storage.abandoned_requests()

    # Act: the kept read is let go only when the test ends
    written = download_source(source, destination=destination, deadline=_deadline(), sleep=stalled.sleep)

    # Assert: the source is whole on disk, and the read that was given up still runs and is counted
    assert hold.entered.is_set()
    assert written == len(OBJECT)
    assert destination.read_bytes() == OBJECT
    assert storage.abandoned_requests() - counted_before == 1


def test_abandoned_requests_when_a_put_stalls_and_its_repeat_is_taken_should_grow_by_one_though_the_upload_succeeds(
    generation, hold, monkeypatch
):
    # Arrange: the storage never answers the first put of the first object, with a minute left on the
    # deadline and a ceiling of 50 ms; it takes every later put at once
    stalled = _StalledFirst(hold)
    client = _PausingS3(stalled.request, f"{PREFIX}/master.m3u8")
    monkeypatch.setattr(storage, "_request_ceiling_s", stalled.ceiling_s)
    counted_before = storage.abandoned_requests()

    # Act: the kept request is let go only when the test ends
    uploaded = generation.upload(client, sleep=stalled.sleep)

    # Assert: the generation is stored, its manifest last, and the put that was given up is counted
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert hold.entered.is_set()
    assert {call.key: call.body for call in client.stored} == expected
    assert client.stored[-1].key == f"{PREFIX}/{MANIFEST_NAME}"
    assert uploaded == sum(len(data) for data in expected.values())
    assert storage.abandoned_requests() - counted_before == 1


def test_abandoned_requests_when_a_request_returns_in_time_should_not_grow():
    # Arrange
    source = DeadlineSource(_StubSource(len(OBJECT), lambda start, length: bytes(length)), _deadline())
    counted_before = storage.abandoned_requests()

    # Act
    data = source.read(0, CHUNK_BYTES)

    # Assert
    assert len(data) == CHUNK_BYTES
    assert storage.abandoned_requests() - counted_before == 0


def test_abandoned_requests_when_a_request_raises_in_time_should_not_grow():
    # Arrange
    def failing(start: int, length: int) -> bytes:
        raise RuntimeError("a bug in the source")

    source = DeadlineSource(_StubSource(len(OBJECT), failing), _deadline())
    counted_before = storage.abandoned_requests()

    # Act
    with pytest.raises(RuntimeError):
        source.read(0, CHUNK_BYTES)

    # Assert
    assert storage.abandoned_requests() - counted_before == 0


@pytest.mark.parametrize(
    "overdue_ending",
    [
        pytest.param(None, id="it-returns-its-bytes"),
        pytest.param(RuntimeError("a bug in the source"), id="it-raises"),
    ],
)
def test_abandoned_requests_when_a_request_ends_after_the_deadline_passed_should_not_grow(overdue_ending):
    # Arrange: the deadline passes while the request runs, and the request then ends: no thread is left behind
    deadline = _HandPassedDeadline()

    def overdue(start: int, length: int) -> bytes:
        deadline.pass_now()
        if overdue_ending is not None:
            raise overdue_ending
        return bytes(length)

    source = DeadlineSource(_StubSource(len(OBJECT), overdue), deadline)
    counted_before = storage.abandoned_requests()

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert: out of time, and not abandoned
    assert failure.code == "source_read_deadline_exceeded"
    assert storage.abandoned_requests() - counted_before == 0


def test_abandoned_requests_when_no_thread_can_be_started_for_a_request_should_not_grow(monkeypatch):
    # Arrange: the operating system gives this process no more threads, so no request is ever under way
    source = DeadlineSource(_StubSource(len(OBJECT), lambda start, length: bytes(length)), _deadline())
    counted_before = storage.abandoned_requests()

    def no_thread(thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", no_thread)
        failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert
    assert failure.code == "storage_request_start_failed"
    assert storage.abandoned_requests() - counted_before == 0


def test_abandoned_requests_when_the_deadline_had_passed_before_the_request_should_not_grow():
    # Arrange
    inner = _StubSource(len(OBJECT), lambda start, length: bytes(length))
    source = DeadlineSource(inner, _passed_deadline())
    counted_before = storage.abandoned_requests()

    # Act
    failure = _failure_of(lambda: source.read(0, CHUNK_BYTES))

    # Assert: no request was started, so none was abandoned
    assert failure.code == "source_read_deadline_exceeded"
    assert inner.reads == []
    assert storage.abandoned_requests() - counted_before == 0


# --- the clients ----------------------------------------------------------------------------------

ACCESS_PROFILE = StorageProfile(
    credential_ref="default",
    endpoint="https://storage.test.invalid",
    region="auto",
    access_key_id="test-access-key-id",
    secret_access_key="test-secret-access-key",
)
STREAMING_PAYLOAD_PREFIX = "STREAMING-"
CHUNKED_CONTENT_ENCODING = "aws-chunked"
CHECKSUM_WHEN_REQUIRED = "when_required"


@dataclass(frozen=True)
class _PreparedRequest:
    """One request as the real client was about to send it; header names in lower case."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes


class _EmptyRawResponse:
    """The body of an answer that has none, as botocore reads one."""

    def stream(self, *arguments: Any, **named_arguments: Any) -> Any:
        return iter(())


def _text(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _answered_before_any_network(client: Any) -> list[_PreparedRequest]:
    """Answer every PutObject of the real client with a 200 before it leaves the process, and return the
    list each prepared request is recorded in."""
    prepared: list[_PreparedRequest] = []

    def answer(request: Any, **named_arguments: Any) -> AWSResponse:
        body = request.body.read() if hasattr(request.body, "read") else request.body
        headers = {name.lower(): _text(value) for name, value in request.headers.items()}
        prepared.append(_PreparedRequest(request.method, request.url, headers, body))
        return AWSResponse(request.url, HTTP_OK, {}, _EmptyRawResponse())

    client.meta.events.register("before-send.s3.PutObject", answer)
    return prepared


def test_upload_when_sent_through_the_upload_client_should_address_each_object_by_bucket_prefix_and_path(generation):
    # Arrange
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    files = {**_Generation.FILES, MANIFEST_NAME: _Generation.MANIFEST}

    # Act
    uploaded = generation.upload(client)

    # Assert: path-style addresses under the granted prefix, in order, each carrying its file
    assert [(request.method, request.url) for request in prepared] == [
        ("PUT", f"https://storage.test.invalid/{OUTPUT_BUCKET}/{PREFIX}/{path}") for path in files
    ]
    assert [request.body for request in prepared] == list(files.values())
    assert prepared[-1].headers["content-type"] == "application/json"
    assert uploaded == sum(len(data) for data in files.values())


def test_upload_when_sent_through_the_upload_client_should_be_a_plain_put_with_its_content_length(generation):
    # Arrange
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    sizes = [len(data) for data in (*_Generation.FILES.values(), _Generation.MANIFEST)]

    # Act
    generation.upload(client)

    # Assert: no `aws-chunked` body with a checksum trailer, which not every S3-compatible store accepts
    assert [request.headers["content-length"] for request in prepared] == [str(size) for size in sizes]
    for request in prepared:
        assert CHUNKED_CONTENT_ENCODING not in request.headers.get("content-encoding", "")
        assert "x-amz-trailer" not in request.headers
        assert "transfer-encoding" not in request.headers
        assert not request.headers.get("x-amz-content-sha256", "").startswith(STREAMING_PAYLOAD_PREFIX)


def test_upload_when_the_upload_client_measures_each_body_by_its_end_first_should_still_store_the_generation(
    generation, monkeypatch
):
    # Arrange: told so, botocore seeks to the end of each body to measure it, and rewinds it, before it signs it
    monkeypatch.setenv(BOTOCORE_MEASURES_BODIES, "true")
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    files = {**_Generation.FILES, MANIFEST_NAME: _Generation.MANIFEST}

    # Act
    uploaded = generation.upload(client)

    # Assert
    assert [request.body for request in prepared] == list(files.values())
    assert uploaded == sum(len(data) for data in files.values())


def test_upload_when_sent_through_the_upload_client_should_accept_a_file_the_client_reads_twice(generation):
    # Arrange
    artifacts = _with_a_large_segment(generation)
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)

    # Act
    uploaded = generation.upload(client, artifacts=artifacts, manifest=manifest_recording(artifacts))

    # Assert: the client read the whole file to sign the request and rewound it; what it then sent was accepted
    sent = next(request for request in prepared if request.url.endswith(LARGE_SEGMENT_PATH))
    assert sent.headers["x-amz-content-sha256"] == hashlib.sha256(LARGE_SEGMENT).hexdigest()
    assert sent.body == LARGE_SEGMENT
    assert prepared[-1].url.endswith(MANIFEST_NAME)
    assert uploaded == sum(len(request.body) for request in prepared)


@pytest.mark.parametrize(
    ("changed", "sent_before_it"),
    [
        pytest.param(
            "360p/segment_0001.ts",
            ["master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts"],
            id="an-artifact",
        ),
        pytest.param(MANIFEST_NAME, list(_Generation.FILES), id="the-manifest"),
    ],
)
def test_upload_when_sent_through_the_upload_client_should_refuse_a_changed_file_before_a_byte_of_it_is_sent(
    generation, changed, sent_before_it
):
    # Arrange: one file holds other bytes, of the same length, than the manifest recorded
    _change_in_place(generation.output_root / changed)
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    generation_address = f"https://storage.test.invalid/{OUTPUT_BUCKET}/{PREFIX}/"

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: the client read the file to sign it and was refused, so no request of it was ever to be sent
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    assert changed in failure.detail
    assert [request.url.removeprefix(generation_address) for request in prepared] == sent_before_it


def test_upload_when_a_file_changes_after_the_upload_client_signed_it_should_refuse_what_was_sent(generation):
    # Arrange: once the client has signed the large segment's request, the file gets other bytes
    artifacts = _with_a_large_segment(generation)
    client = upload_client(ACCESS_PROFILE)

    def change_the_signed_file(request: Any, **named_arguments: Any) -> None:
        if request.url.endswith(LARGE_SEGMENT_PATH):
            _change_in_place(generation.output_root / LARGE_SEGMENT_PATH)

    client.meta.events.register("before-send.s3.PutObject", change_the_signed_file)
    prepared = _answered_before_any_network(client)

    # Act
    failure = _failure_of(
        lambda: generation.upload(client, artifacts=artifacts, manifest=manifest_recording(artifacts))
    )

    # Assert: the request was signed for the recorded bytes and carried others, so it does not count
    sent = prepared[-1]
    assert sent.url.endswith(LARGE_SEGMENT_PATH)
    assert sent.headers["x-amz-content-sha256"] == hashlib.sha256(LARGE_SEGMENT).hexdigest()
    assert sent.body == LARGE_SEGMENT.translate(OTHER_BYTES)
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")


def test_upload_when_the_upload_clients_answer_asks_to_retry_after_seconds_should_wait_that_long_and_send_again(
    generation, fixed_factor
):
    # Arrange: the real client's first request is answered with a 503 that asks for five seconds, as a
    # storage sends the header; every later request is answered with a 200
    client = upload_client(ACCESS_PROFILE)
    answered: list[str] = []

    def answer(request: Any, **named_arguments: Any) -> AWSResponse:
        answered.append(request.url)
        if len(answered) == 1:
            return AWSResponse(request.url, HTTP_SERVICE_UNAVAILABLE, {"Retry-After": "5"}, _EmptyRawResponse())
        request.body.read()
        return AWSResponse(request.url, HTTP_OK, {}, _EmptyRawResponse())

    client.meta.events.register("before-send.s3.PutObject", answer)
    first_address = f"https://storage.test.invalid/{OUTPUT_BUCKET}/{PREFIX}/master.m3u8"

    # Act
    uploaded = generation.upload(client)

    # Assert: five seconds where 0.375 s was due, then the same object again and the rest of the generation
    assert generation.waits.asked == [5.0]
    assert answered[:2] == [first_address, first_address]
    assert len(answered) == len(_Generation.FILES) + 2
    assert uploaded == sum(len(data) for data in (*_Generation.FILES.values(), _Generation.MANIFEST))


class _RawResponse:
    """The body of an answer that has one, as botocore reads one."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def stream(self, *arguments: Any, **named_arguments: Any) -> Any:
        return iter((self._data,))


def _answered_alike_before_any_network(
    client: Any, status: int, *, headers: dict[str, str] | None = None, body: bytes = b""
) -> list[_PreparedRequest]:
    """Answer every request of the real client, whatever it asks for, with the same `status`, `headers`
    and `body` before it leaves the process, and return the list each prepared request is recorded in
    (without its body, which is not read)."""
    prepared: list[_PreparedRequest] = []

    def answer(request: Any, **named_arguments: Any) -> AWSResponse:
        sent_headers = {name.lower(): _text(value) for name, value in request.headers.items()}
        prepared.append(_PreparedRequest(request.method, request.url, sent_headers, b""))
        return AWSResponse(request.url, status, headers or {}, _RawResponse(body))

    # Every operation, not PutObject alone: what the client would ask on its own account is answered too.
    client.meta.events.register("before-send.s3", answer)
    return prepared


# What a storage answers a request it takes to be signed for the wrong region: the region to sign for.
OTHER_REGION = "other-region-1"
WRONG_REGION_ANSWER = (
    '<?xml version="1.0" encoding="UTF-8"?><Error><Code>AuthorizationHeaderMalformed</Code>'
    f"<Message>the region is wrong</Message><Region>{OTHER_REGION}</Region></Error>"
).encode("ascii")
FIRST_OBJECT_ADDRESS = f"https://storage.test.invalid/{OUTPUT_BUCKET}/{PREFIX}/master.m3u8"
SOURCE_ADDRESS = f"https://storage.test.invalid/{BUCKET}/{KEY}"


@pytest.mark.parametrize("status", [HTTP_FOUND, HTTP_TEMPORARY_REDIRECT], ids=["302", "307"])
def test_upload_when_the_upload_client_is_answered_with_a_passing_redirect_should_send_one_put_for_each_attempt(
    generation, fixed_factor, status
):
    # Arrange: a host that answers every request of the real client with the same redirect
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_alike_before_any_network(client, status)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert: the answer came back as the storage's own, a passing refusal: the first object was sent
    # four times with the three waits between, and the client asked for nothing on its own account
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert [(request.method, request.url) for request in prepared] == [("PUT", FIRST_OBJECT_ADDRESS)] * 4
    assert generation.waits.asked == [0.375, 1.5, 6.0]


def test_upload_when_the_upload_client_is_answered_with_a_301_should_send_one_put_and_be_refused_for_good(generation):
    # Arrange: a host that answers every request of the real client with a permanent redirect
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_alike_before_any_network(client, HTTP_MOVED_PERMANENTLY)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: one request, no wait, and nothing asked on the client's own account
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert [(request.method, request.url) for request in prepared] == [("PUT", FIRST_OBJECT_ADDRESS)]
    assert generation.waits.asked == []


@pytest.mark.parametrize(
    ("status", "headers", "body"),
    [
        pytest.param(HTTP_BAD_REQUEST, {}, WRONG_REGION_ANSWER, id="in-its-error-body"),
        pytest.param(HTTP_MOVED_PERMANENTLY, {"x-amz-bucket-region": OTHER_REGION}, b"", id="in-a-header"),
    ],
)
def test_upload_when_the_upload_clients_answer_names_another_region_should_not_send_the_put_again_signed_for_it(
    generation, status, headers, body
):
    # Arrange: the storage refuses every request and names the region it wants requests signed for
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_alike_before_any_network(client, status, headers=headers, body=body)

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert: one request, signed for the access profile's region, and the refusal is final
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_refused", False)
    assert [(request.method, request.url) for request in prepared] == [("PUT", FIRST_OBJECT_ADDRESS)]
    assert f"/{ACCESS_PROFILE.region}/s3/aws4_request" in prepared[0].headers["authorization"]


@pytest.mark.parametrize(
    "status", [HTTP_MOVED_PERMANENTLY, HTTP_FOUND, HTTP_TEMPORARY_REDIRECT], ids=["301", "302", "307"]
)
def test_download_when_the_storage_client_is_answered_with_a_redirect_should_make_one_ranged_read_for_each_attempt(
    tmp_path, fixed_factor, status
):
    # Arrange: a host that answers every request of the real client with the same redirect
    client = storage_client(ACCESS_PROFILE)
    prepared = _answered_alike_before_any_network(client, status)
    source = S3RangeSource(client, bucket=BUCKET, key=KEY, size_bytes=len(OBJECT), etag=ETAG)
    destination = tmp_path / "source"
    waits = _Waits()

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=_deadline(), sleep=waits))

    # Assert: the source's own retryable failure after three reads of the chunk with the two waits
    # between, each read one ranged request, and nothing asked on the client's own account
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_error", True)
    assert [(request.method, request.url) for request in prepared] == [("GET", SOURCE_ADDRESS)] * 3
    assert {request.headers["range"] for request in prepared} == {f"bytes=0-{len(OBJECT) - 1}"}
    assert waits.asked == [0.375, 1.5]
    assert not destination.exists()


def test_download_when_the_storage_clients_answer_names_another_region_should_not_read_again_signed_for_it(tmp_path):
    # Arrange: the storage refuses every read and names the region it wants requests signed for
    client = storage_client(ACCESS_PROFILE)
    prepared = _answered_alike_before_any_network(client, HTTP_BAD_REQUEST, body=WRONG_REGION_ANSWER)
    source = S3RangeSource(client, bucket=BUCKET, key=KEY, size_bytes=len(OBJECT), etag=ETAG)
    waits = _Waits()

    # Act
    with pytest.raises(WorkerFailure):
        download_source(source, destination=tmp_path / "source", deadline=_deadline(), sleep=waits)

    # Assert: one request for each of the three attempts, every one signed for the access profile's region
    assert [(request.method, request.url) for request in prepared] == [("GET", SOURCE_ADDRESS)] * 3
    assert all(f"/{ACCESS_PROFILE.region}/s3/aws4_request" in request.headers["authorization"] for request in prepared)


def test_upload_client_when_given_an_access_profile_should_reach_its_endpoint_as_the_source_reader_does():
    # Act: making a client opens no connection
    writer, reader = upload_client(ACCESS_PROFILE), storage_client(ACCESS_PROFILE)

    # Assert: the reader's signing, addressing, time-outs and single attempt; a checksum only where required
    assert writer.meta.endpoint_url == "https://storage.test.invalid"
    assert writer.meta.region_name == "auto"
    assert writer.meta.service_model.service_name == "s3"
    for setting in ("signature_version", "s3", "connect_timeout", "read_timeout", "retries"):
        assert getattr(writer.meta.config, setting) == getattr(reader.meta.config, setting), setting
    assert writer.meta.config.request_checksum_calculation == CHECKSUM_WHEN_REQUIRED
    assert writer.meta.config.response_checksum_validation == CHECKSUM_WHEN_REQUIRED


def test_storage_client_when_given_an_access_profile_should_make_a_client_of_its_endpoint_and_region():
    # Act: making a client opens no connection
    client = storage_client(ACCESS_PROFILE)

    # Assert
    assert client.meta.endpoint_url == "https://storage.test.invalid"
    assert client.meta.region_name == "auto"
    assert client.meta.service_model.service_name == "s3"


@pytest.mark.parametrize("make_client", [storage_client, upload_client])
def test_client_when_given_something_else_than_a_profile_should_raise_type_error(make_client):
    # Act and Assert
    with pytest.raises(TypeError):
        make_client({"endpoint": "https://storage.test.invalid"})
