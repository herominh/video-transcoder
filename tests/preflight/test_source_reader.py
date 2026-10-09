"""Conditional range reads against a fake S3 endpoint, through the real boto3 client."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from core.bounded_process import Deadline
from core.failure import WorkerFailure
from core.source_reader import BudgetedSource, S3RangeSource, make_s3_client

BUCKET = "test-bucket"
KEY = "test/sources/source-0001.mp4"
ETAG = "0123456789abcdef0123456789abcdef"
OBJECT = bytes(range(256)) * 64  # 16 KiB of known bytes


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
    cut_body_to: int | None = None
    requests: list[dict[str, str]] = field(default_factory=list)


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

        def _error(self, status: int, code: str) -> None:
            body = f"<Error><Code>{code}</Code><Message>m</Message></Error>".encode()
            self.send_response(status)
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
        region="us-east-1",
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
