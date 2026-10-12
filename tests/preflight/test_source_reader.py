"""Conditional range reads against a fake S3 endpoint, through the real boto3 client."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, ReadTimeoutError

from core.bounded_process import Deadline
from core.failure import Failure, WorkerFailure
from core.source_reader import BudgetedSource, S3RangeSource, make_s3_client

BUCKET = "test-bucket"
KEY = "test/sources/source-0001.mp4"
ETAG = "0123456789abcdef0123456789abcdef"
OBJECT = bytes(range(256)) * 64  # 16 KiB of known bytes
# Where a test's own clock starts: a monotonic clock does not read zero when a reader is made.
CLOCK_STARTS_AT_S = 500.0
# How long a test waits for another thread to get somewhere, and a held read lets go by itself: far
# longer than a test takes, so that a regression fails a test instead of hanging the suite.
PATIENCE_S = 20.0
REGION = "us-east-1"
# A region the fake names where a test has its answer say that the bucket lies elsewhere, and the header
# a storage names a bucket's region in.
OTHER_REGION = "other-region-1"
BUCKET_REGION_HEADER = "x-amz-bucket-region"
HTTP_PARTIAL_CONTENT = 206
HTTP_FIRST_REDIRECT = 300
HTTP_MOVED_PERMANENTLY = 301
HTTP_FIRST_CLIENT_ERROR = 400
HTTP_INTERNAL_SERVER_ERROR = 500
# Where the fake's redirect answers say the object has moved: another path of the fake itself, so that a
# request that followed one would arrive, and be counted, there.
MOVED_TO = "/moved-elsewhere"
# An endpoint as the storage client's own errors name it: what no failure may carry.
ENDPOINT_CANARY = "https://account-canary.r2.example.invalid"


@dataclass
class FakeS3State:
    data: bytes = OBJECT
    etag: str = ETAG
    ignore_range: bool = False
    ignore_if_match: bool = False
    answer_etag: str | None = None
    answer_total: int | None = None
    shift_range_by: int = 0
    status: int | None = None
    error_code: str = ""
    # The region an error answer names in the bucket-region header; None for an answer without that header.
    answer_region: str | None = None
    cut_body_to: int | None = None
    requests: list[dict[str, str]] = field(default_factory=list)
    # The path of every HEAD request: the reader makes none, so each is one the client made on its own account.
    head_requests: list[str] = field(default_factory=list)


def _handler(state: FakeS3State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            state.requests.append({key.lower(): value for key, value in self.headers.items()})
            if self.path != f"/{BUCKET}/{KEY}":
                self._error(404, "NoSuchKey")
                return
            if state.status is not None:
                self._error(state.status, state.error_code)
                return
            if_match = self.headers.get("If-Match")
            if not state.ignore_if_match and if_match is not None and if_match != f'"{state.etag}"':
                self._error(412, "PreconditionFailed")
                return
            range_header = self.headers.get("Range")
            data = state.data
            total = len(data) if state.answer_total is None else state.answer_total
            if range_header is None or state.ignore_range:
                self._send(200, data, None)
                return
            first, last = range_header.removeprefix("bytes=").split("-")
            start, end = int(first) + state.shift_range_by, int(last) + state.shift_range_by
            self._send(206, data[start:end + 1], f"bytes {start}-{end}/{total}")

        def do_HEAD(self) -> None:
            # Answered as a GET is where a test sets a status, without a body: a HEAD's answer has none.
            state.head_requests.append(self.path)
            status = 200 if state.status is None else state.status
            self.send_response(status)
            self._send_where_to_go_instead(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, status: int, body: bytes, content_range: str | None) -> None:
            self.send_response(status)
            self.send_header("ETag", f'"{state.answer_etag or state.etag}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            if content_range is not None:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            sent = body if state.cut_body_to is None else body[:state.cut_body_to]
            self.wfile.write(sent)
            if state.cut_body_to is not None:
                self.close_connection = True

        def _send_where_to_go_instead(self, status: int) -> None:
            """What a real storage's error answer carries besides its body: a redirect names an address
            in `Location`, and an answer about a bucket of another region may name that region."""
            if HTTP_FIRST_REDIRECT <= status < HTTP_FIRST_CLIENT_ERROR:
                self.send_header("Location", f"http://{self.headers['Host']}{MOVED_TO}{self.path}")
            if state.answer_region is not None:
                self.send_header(BUCKET_REGION_HEADER, state.answer_region)

        def _error(self, status: int, code: str) -> None:
            body = f"<Error><Code>{code}</Code><Message>m</Message></Error>".encode()
            self.send_response(status)
            self._send_where_to_go_instead(status)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The client under test drops connections on purpose (an ignored range, a cut body).
        return


@pytest.fixture
def fake_s3() -> Iterator[tuple[FakeS3State, str]]:
    state = FakeS3State()
    server = QuietServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _source(endpoint: str, *, size_bytes: int = len(OBJECT), session_token: str | None = None) -> S3RangeSource:
    client = make_s3_client(
        endpoint_url=endpoint,
        region=REGION,
        access_key_id="test-access-key",
        secret_access_key="test-secret-key",
        session_token=session_token,
    )
    return S3RangeSource(client, bucket=BUCKET, key=KEY, size_bytes=size_bytes, etag=ETAG)


def _failure_of(call: Any) -> Any:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value.failure


def test_read_when_storage_honours_the_range_should_return_exactly_those_bytes(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    source = _source(endpoint)

    # Act
    data = source.read(100, 50)

    # Assert
    assert data == OBJECT[100:150]
    assert state.requests[0]["if-match"] == f'"{ETAG}"'


def test_read_when_the_object_changed_should_fail_as_source_changed(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.etag = "a-newer-etag"
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_integrity_mismatch", "source_changed", False)


def test_read_when_storage_ignores_if_match_should_fail_as_source_changed(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.ignore_if_match = True
    state.answer_etag = "a-newer-etag"
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.error_class, failure.code) == ("source_integrity_mismatch", "source_changed")


def test_read_when_storage_ignores_the_range_should_fail_closed(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.ignore_range = True
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_unavailable", "range_not_honoured", False)


def test_read_when_storage_answers_another_range_should_fail_closed(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.shift_range_by = 1
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert failure.code == "range_not_honoured"


def test_read_when_the_object_size_differs_from_the_record_should_fail_as_size_mismatch(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.answer_total = len(OBJECT) + 1
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.error_class, failure.code) == ("source_integrity_mismatch", "source_size_mismatch")


@pytest.mark.parametrize(
    ("status", "error_code", "expected_code", "expected_retryable"),
    [
        (404, "NoSuchKey", "source_missing", False),
        (403, "AccessDenied", "source_access_denied", False),
        (416, "InvalidRange", "source_size_mismatch", False),
        (500, "InternalError", "storage_error", True),
    ],
)
def test_read_when_storage_refuses_should_report_a_typed_failure_after_one_request(
    fake_s3, status, error_code, expected_code, expected_retryable,
):
    # Arrange
    state, endpoint = fake_s3
    state.status = status
    state.error_code = error_code
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.code, failure.retryable) == (expected_code, expected_retryable)
    assert len(state.requests) == 1  # no hidden retry spent a request the budget never saw


@pytest.mark.parametrize(
    ("status", "error_code"),
    [(301, "PermanentRedirect"), (302, "Found"), (307, "TemporaryRedirect")],
    ids=["301", "302", "307"],
)
def test_read_when_storage_answers_every_request_with_a_redirect_should_make_one_request_and_fail_typed(
    fake_s3, status, error_code,
):
    # Arrange: a host that answers whatever it is asked, a bucket's head included, with the same redirect
    state, endpoint = fake_s3
    state.status = status
    state.error_code = error_code
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert: the answer came back as the storage's own, and the client followed it nowhere: neither to
    # the address its `Location` names (a request there would be counted too) nor to the bucket's head
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_error", True)
    assert len(state.requests) == 1
    assert state.head_requests == []


def test_read_when_a_redirect_names_the_buckets_region_in_a_header_should_not_read_again_signed_for_that_region(
    fake_s3,
):
    # Arrange: a redirect whose answer says, in the header a storage names a bucket's region in, that
    # the bucket lies in another region than the client's
    state, endpoint = fake_s3
    state.status = HTTP_MOVED_PERMANENTLY
    state.error_code = "PermanentRedirect"
    state.answer_region = OTHER_REGION
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert: one request, signed for the client's own region, and none signed again for the one named
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_error", True)
    assert len(state.requests) == 1
    assert state.head_requests == []
    assert f"/{REGION}/" in state.requests[0]["authorization"]
    assert OTHER_REGION not in state.requests[0]["authorization"]


def test_read_when_the_body_is_cut_short_should_fail_retryable(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    state.cut_body_to = 5
    source = _source(endpoint)

    # Act
    failure = _failure_of(lambda: source.read(0, 10))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "short_read", True)


def test_read_when_a_session_token_is_configured_should_send_it_on_the_request(fake_s3):
    # Arrange
    state, endpoint = fake_s3
    source = _source(endpoint, session_token="test-session-token")

    # Act
    source.read(0, 10)

    # Assert
    assert state.requests[0]["x-amz-security-token"] == "test-session-token"


class _BodyThatFails:
    """A response body whose read the storage client fails with an error of its own."""

    def __init__(self, error: BotoCoreError) -> None:
        self._error = error
        self.closed = False

    def read(self, amount: int | None = None) -> bytes:
        raise self._error

    def close(self) -> None:
        self.closed = True


class _FailingClient:
    """Stands in for the S3 client where a test needs it to fail in a way of its own: get_object raises
    `request_error` when it is given one, and otherwise answers the range asked, as recorded, with a
    body whose read raises `body_error`."""

    def __init__(self, *, request_error: Exception | None = None, body_error: BotoCoreError | None = None) -> None:
        self._request_error = request_error
        self.body = None if body_error is None else _BodyThatFails(body_error)

    def get_object(self, *, Bucket: str, Key: str, Range: str, IfMatch: str) -> dict[str, Any]:  # noqa: N803
        if self._request_error is not None:
            raise self._request_error
        first, last = (int(bound) for bound in Range.removeprefix("bytes=").split("-"))
        return {
            "ResponseMetadata": {"HTTPStatusCode": HTTP_PARTIAL_CONTENT},
            "ETag": IfMatch,
            "ContentRange": f"bytes {first}-{last}/{len(OBJECT)}",
            "ContentLength": last - first + 1,
            "Body": self.body,
        }


# Each way the storage client's own error ends a read, and the code of the typed failure raised in its place.
CLIENT_ERRORS = [
    pytest.param(
        {
            "request_error": ClientError(
                {
                    "Error": {"Code": "InternalError", "Message": f"{ENDPOINT_CANARY}/{BUCKET}/{KEY}"},
                    "ResponseMetadata": {"HTTPStatusCode": HTTP_INTERNAL_SERVER_ERROR},
                },
                "GetObject",
            )
        },
        "storage_error",
        id="the-storage-answers-with-an-error",
    ),
    pytest.param(
        {"request_error": EndpointConnectionError(endpoint_url=f"{ENDPOINT_CANARY}/{BUCKET}/{KEY}")},
        "storage_unreachable",
        id="the-storage-cannot-be-reached",
    ),
    pytest.param(
        {"body_error": ReadTimeoutError(endpoint_url=f"{ENDPOINT_CANARY}/{BUCKET}/{KEY}")},
        "short_read",
        id="the-body-breaks-under-its-read",
    ),
]


@pytest.mark.parametrize(("client_fails_with", "expected_code"), CLIENT_ERRORS)
def test_read_when_the_storage_clients_own_error_ends_it_should_raise_a_failure_that_carries_nothing_of_that_error(
    client_fails_with, expected_code
):
    # Arrange: the client fails with an error of its own, whose text names the endpoint and the object
    client = _FailingClient(**client_fails_with)
    source = S3RangeSource(client, bucket=BUCKET, key=KEY, size_bytes=len(OBJECT), etag=ETAG)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        source.read(0, 10)

    # Assert: the typed failure alone, in our own words: the client's error is neither its cause nor its
    # context, so nothing that keeps or shows the failure keeps or shows the endpoint
    failure = raised.value.failure
    assert (failure.error_class, failure.code) == ("source_unavailable", expected_code)
    assert "example.invalid" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert client.body is None or client.body.closed


@pytest.mark.parametrize(("start", "length"), [(-1, 10), (0, 0), (len(OBJECT) - 5, 10)])
def test_read_when_the_range_leaves_the_object_should_raise_value_error(fake_s3, start, length):
    # Arrange
    _, endpoint = fake_s3
    source = _source(endpoint)

    # Act / Assert
    with pytest.raises(ValueError):
        source.read(start, length)


class MemorySource:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[tuple[int, int]] = []

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def read(self, start: int, length: int) -> bytes:
        self.reads.append((start, length))
        return self.data[start:start + length]


def test_budgeted_read_when_within_budget_should_count_bytes_and_requests():
    # Arrange
    inner = MemorySource(OBJECT)
    source = BudgetedSource(inner, max_bytes=100, max_requests=3, deadline=Deadline.after_ms(10_000))

    # Act
    source.read(0, 40)
    source.read(1000, 60)

    # Assert
    assert (source.stats.bytes_read, source.stats.requests) == (100, 2)


def test_budgeted_read_when_bytes_would_exceed_the_budget_should_refuse_before_reading():
    # Arrange
    inner = MemorySource(OBJECT)
    source = BudgetedSource(inner, max_bytes=100, max_requests=10, deadline=Deadline.after_ms(10_000))
    source.read(0, 60)

    # Act
    failure = _failure_of(lambda: source.read(60, 41))

    # Assert
    assert (failure.error_class, failure.code) == ("input_limits_exceeded", "probe_budget_exceeded")
    assert inner.reads == [(0, 60)]


def test_budgeted_read_when_requests_would_exceed_the_budget_should_refuse_before_reading():
    # Arrange
    inner = MemorySource(OBJECT)
    source = BudgetedSource(inner, max_bytes=10_000, max_requests=2, deadline=Deadline.after_ms(10_000))
    source.read(0, 1)
    source.read(1, 1)

    # Act
    failure = _failure_of(lambda: source.read(2, 1))

    # Assert
    assert failure.code == "probe_budget_exceeded"
    assert len(inner.reads) == 2


def test_budgeted_read_when_the_deadline_passed_should_refuse_before_reading():
    # Arrange
    inner = MemorySource(OBJECT)
    deadline = Deadline(expires_at=0.0)
    source = BudgetedSource(inner, max_bytes=10_000, max_requests=10, deadline=deadline)

    # Act
    failure = _failure_of(lambda: source.read(0, 1))

    # Assert
    assert (failure.error_class, failure.code) == ("deadline_exceeded", "probe_deadline_exceeded")
    assert inner.reads == []


# The time in reads, measured on a clock only the test moves: no real time decides a test below.


class HandClock:
    """A monotonic clock that only the test moves."""

    def __init__(self, now_s: float = CLOCK_STARTS_AT_S) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds


class TimedSource(MemorySource):
    """Each read takes the next of `read_seconds` on the test's clock; a read after the last takes none."""

    def __init__(self, clock: HandClock, read_seconds: list[float]) -> None:
        super().__init__(OBJECT)
        self._clock = clock
        self._read_seconds = list(read_seconds)

    def read(self, start: int, length: int) -> bytes:
        if self._read_seconds:
            self._clock.advance(self._read_seconds.pop(0))
        return super().read(start, length)


class HeldSource(MemorySource):
    """Every read waits until the test lets it go: read number n (the first is 1) sets `entered[n]` and
    waits on `release[n]`. The test's clock is the only thing that moves while a read is held."""

    def __init__(self, reads: int) -> None:
        super().__init__(OBJECT)
        self.entered = {number: threading.Event() for number in range(1, reads + 1)}
        self.release = {number: threading.Event() for number in range(1, reads + 1)}
        self._count_lock = threading.Lock()
        self._count = 0

    def read(self, start: int, length: int) -> bytes:
        with self._count_lock:
            self._count += 1
            number = self._count
        self.entered[number].set()
        self.release[number].wait(PATIENCE_S)
        return super().read(start, length)

    def release_all(self) -> None:
        for released in self.release.values():
            released.set()


def _timed(inner: MemorySource, clock: HandClock) -> BudgetedSource:
    return BudgetedSource(
        inner, max_bytes=10_000, max_requests=10, deadline=Deadline.after_ms(60_000), monotonic=clock
    )


def _reading_on_a_thread(source: BudgetedSource, start: int) -> threading.Thread:
    thread = threading.Thread(target=source.read, args=(start, 1), daemon=True)
    thread.start()
    return thread


def _measured(source: BudgetedSource) -> tuple[float, float]:
    read_time = source.read_time()
    return read_time.seconds_in_reads, read_time.seconds_elapsed


def test_read_time_when_no_read_was_made_should_show_no_time_in_reads_of_the_time_since_the_reader_was_made():
    # Arrange
    clock = HandClock()
    source = _timed(MemorySource(OBJECT), clock)
    clock.advance(7.0)

    # Act
    measured = _measured(source)

    # Assert
    assert measured == (0.0, 7.0)


def test_read_time_when_reads_took_time_should_sum_what_each_took_and_leave_out_the_time_between_them():
    # Arrange: 5 s pass, a read of 3 s, 2 s pass, a read of 4 s, 6 s pass
    clock = HandClock()
    source = _timed(TimedSource(clock, [3.0, 4.0]), clock)
    clock.advance(5.0)
    source.read(0, 1)
    clock.advance(2.0)
    source.read(1, 1)
    clock.advance(6.0)

    # Act
    measured = _measured(source)

    # Assert
    assert measured == (7.0, 20.0)


def test_read_time_when_a_read_raises_should_count_the_time_it_took():
    # Arrange: a read that fails after 8 s, as a stalled read is given up
    clock = HandClock()
    stalled = WorkerFailure(Failure(
        error_class="source_unavailable", code="source_read_stalled", retryable=True,
        detail="a read of the source did not end in time",
    ))

    class FailingSlowly(MemorySource):
        def read(self, start: int, length: int) -> bytes:
            clock.advance(8.0)
            raise stalled

    source = _timed(FailingSlowly(OBJECT), clock)
    failure = _failure_of(lambda: source.read(0, 1))
    clock.advance(2.0)

    # Act
    measured = _measured(source)

    # Assert
    assert failure.code == "source_read_stalled"
    assert measured == (8.0, 10.0)


def test_read_time_when_a_read_is_refused_by_the_budget_should_count_no_time_for_it():
    # Arrange: one request allowed; the second is refused before it reaches the storage
    clock = HandClock()
    source = BudgetedSource(
        TimedSource(clock, [3.0, 4.0]), max_bytes=10_000, max_requests=1,
        deadline=Deadline.after_ms(60_000), monotonic=clock,
    )
    source.read(0, 1)
    refusal = _failure_of(lambda: source.read(1, 1))
    clock.advance(1.0)

    # Act
    measured = _measured(source)

    # Assert
    assert refusal.code == "probe_budget_exceeded"
    assert measured == (3.0, 4.0)


def test_read_time_when_a_read_is_still_in_flight_should_count_it_up_to_the_moment_of_asking():
    # Arrange: 5 s pass, then a read that has not returned
    clock = HandClock()
    inner = HeldSource(reads=1)
    source = _timed(inner, clock)
    clock.advance(5.0)
    reading = _reading_on_a_thread(source, 0)

    # Act
    try:
        assert inner.entered[1].wait(PATIENCE_S)
        clock.advance(12.0)
        while_in_flight = _measured(source)
        clock.advance(4.0)
        later_in_flight = _measured(source)
        inner.release[1].set()
        reading.join(PATIENCE_S)
        clock.advance(6.0)
        after_it_returned = _measured(source)
    finally:
        inner.release_all()

    # Assert: counted while it runs, and no longer once it has returned
    assert while_in_flight == (12.0, 17.0)
    assert later_in_flight == (16.0, 21.0)
    assert after_it_returned == (16.0, 27.0)


def test_read_time_when_reads_overlap_on_two_threads_should_count_their_shared_time_once_and_never_exceed_the_elapsed():
    # Arrange: one read runs from 0 s to 20 s, another from 10 s to 30 s: 20 s each, 30 s between them
    clock = HandClock()
    inner = HeldSource(reads=2)
    source = _timed(inner, clock)
    first = _reading_on_a_thread(source, 0)

    # Act
    try:
        assert inner.entered[1].wait(PATIENCE_S)
        clock.advance(10.0)
        second = _reading_on_a_thread(source, 1)
        assert inner.entered[2].wait(PATIENCE_S)
        clock.advance(5.0)
        while_both_run = _measured(source)
        clock.advance(5.0)
        inner.release[1].set()
        first.join(PATIENCE_S)
        clock.advance(10.0)
        inner.release[2].set()
        second.join(PATIENCE_S)
        clock.advance(5.0)
        after_both = _measured(source)
    finally:
        inner.release_all()

    # Assert: the time in which a read was under way, not the 15 s + 5 s and the 20 s + 20 s of the two
    assert while_both_run == (15.0, 15.0)
    assert after_both == (30.0, 35.0)


def test_read_time_when_the_readings_do_not_add_up_exactly_should_still_not_exceed_the_elapsed_time():
    # Arrange: three reads back to back from the reader's first instant on, on readings whose three
    # differences sum to 1.4000000000000001 where the first and the last are 1.4 apart
    clock = HandClock(now_s=0.1)
    source = _timed(TimedSource(clock, [0.1, 0.2, 1.1]), clock)
    for start in range(3):
        source.read(start, 1)

    # Act
    in_reads, elapsed = _measured(source)

    # Assert
    assert in_reads <= elapsed
    assert elapsed == pytest.approx(1.4)
    assert in_reads == pytest.approx(1.4)
