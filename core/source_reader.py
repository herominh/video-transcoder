"""Conditional, budgeted range reads of one exact source object.

A preflight never downloads the source. It reads byte ranges of the object the dispatch names,
each read conditional on the recorded ETag, and it stops at a byte, request and time budget that
it enforces itself, whatever the storage or the prober would do. The budgeted reader also measures
how much of its time the reads themselves took, so that a preflight that ran out of time can tell
a slow storage from anything else that used the time (core/probe.py decides).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from core.bounded_process import Deadline
from core.failure import Failure, WorkerFailure

S3_CONNECT_TIMEOUT_S = 5
S3_READ_TIMEOUT_S = 10
# One attempt per read: botocore's own retries would be requests the budget never sees. (Its S3
# region redirector would be another source of them; without_region_redirects() stops it.)
S3_MAX_ATTEMPTS = 1
# The event every request of an S3 client passes on its way out, and where botocore's region redirector
# keeps, in the request's context, whether the request was redirected already.
BEFORE_S3_CALL_EVENT = "before-call.s3"
REDIRECT_CONTEXT_KEY = "s3_redirect"
REDIRECTED_KEY = "redirected"

HTTP_PARTIAL_CONTENT = 206
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_PRECONDITION_FAILED = 412
HTTP_RANGE_NOT_SATISFIABLE = 416

PRECONDITION_FAILED_CODES = frozenset({"PreconditionFailed", "412"})
MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "NotFound", "404"})
ACCESS_DENIED_CODES = frozenset({"AccessDenied", "403"})
INVALID_RANGE_CODES = frozenset({"InvalidRange", "416"})

# What a read past the time budget is refused with, and what a preflight that ran out of time fails
# with for good. The preflight types the reader's refusal again where it surfaces: when the storage
# was the slow party, it fails with a retryable failure instead (core/probe.py).
PROBE_DEADLINE_EXCEEDED = Failure(
    error_class="deadline_exceeded", code="probe_deadline_exceeded", retryable=False,
    detail="the preflight ran out of time",
)


class RangeSource(Protocol):
    """One immutable object, read by exact byte ranges."""

    @property
    def size_bytes(self) -> int: ...

    def read(self, start: int, length: int) -> bytes:
        """Return exactly `length` bytes from `start`, or raise WorkerFailure."""
        ...


def make_s3_client(
    *,
    endpoint_url: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str | None = None,
) -> Any:
    """An S3 client for range reads: path-style, short timeouts, no automatic retries and no region redirect."""
    import boto3

    return without_region_redirects(
        boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            config=Config(
                signature_version="s3v4",
                connect_timeout=S3_CONNECT_TIMEOUT_S,
                read_timeout=S3_READ_TIMEOUT_S,
                retries={"total_max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )
    )


def without_region_redirects(client: Any) -> Any:
    """The same S3 client, made to follow no region redirect: one call of it is one request.

    botocore registers a region redirector on every S3 client. After an answer that is a redirect (301,
    302, 307) or names another region it sends the request again, signed for the region the answer
    names; when the answer names none it first asks for the bucket's head, with the same client, so
    against a host that answers every request with a redirect it asks until Python's recursion limit.
    The redirector leaves alone a request whose context says it was redirected already, so every
    request is marked so before it is sent: the answer then reaches the caller as the ClientError it
    is. Only botocore's public event system is used.
    """
    client.meta.events.register(BEFORE_S3_CALL_EVENT, _mark_as_redirected)
    return client


def _mark_as_redirected(context: dict[str, Any], **named_arguments: Any) -> None:
    """Handles `before-call.s3`: the redirector made the request's entry while the parameters were built."""
    context.setdefault(REDIRECT_CONTEXT_KEY, {})[REDIRECTED_KEY] = True


def _failure(error_class: str, code: str, detail: str, *, retryable: bool = False) -> WorkerFailure:
    return WorkerFailure(Failure(error_class=error_class, code=code, retryable=retryable, detail=detail))


def _unquote_etag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


class S3RangeSource:
    """Range reads of one recorded object, each bound to its recorded size and ETag.

    Every response is checked before its body is read: a storage that ignores the Range header
    (200 instead of 206, or another range) or the If-Match condition (another ETag) fails closed
    without streaming the object.

    A typed failure raised in place of the storage client's own error carries nothing of that error,
    neither as its cause nor as its context: the client's error names the endpoint and the object, so
    the failure is raised once the handler that caught it has ended.
    """

    def __init__(self, client: Any, *, bucket: str, key: str, size_bytes: int, etag: str) -> None:
        if not bucket or not key:
            raise ValueError("bucket and key are required")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")
        if not etag or '"' in etag:
            raise ValueError("etag must be the unquoted recorded ETag")
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size_bytes = size_bytes
        self._etag = etag

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    def read(self, start: int, length: int) -> bytes:
        _check_range(start, length, self._size_bytes)
        end = start + length - 1
        response = self._answer(start, end)
        body = response.get("Body")
        try:
            self._check_response(response, start, end, length)
            if body is None:
                raise _failure("source_unavailable", "short_read", "the storage sent no body", retryable=True)
            data = _body_bytes(body, length)
            if len(data) != length:
                raise _failure(
                    "source_unavailable", "short_read",
                    f"the storage sent {len(data)} of {length} bytes", retryable=True,
                )
            return data
        finally:
            if body is not None:
                body.close()

    def _answer(self, start: int, end: int) -> dict[str, Any]:
        """The storage's answer to one read of the range, conditional on the recorded ETag; a typed
        failure where the client raised an error of its own instead."""
        try:
            return self._client.get_object(
                Bucket=self._bucket,
                Key=self._key,
                Range=f"bytes={start}-{end}",
                IfMatch=f'"{self._etag}"',
            )
        except ClientError as error:
            failure = _client_error_failure(error)
        except BotoCoreError:
            failure = _failure(
                "source_unavailable", "storage_unreachable",
                "the storage could not be reached", retryable=True,
            )
        # Raised here, after the handlers have ended: the client's error is neither cause nor context.
        raise failure

    def _check_response(self, response: dict[str, Any], start: int, end: int, length: int) -> None:
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != HTTP_PARTIAL_CONTENT:
            raise _failure(
                "source_unavailable", "range_not_honoured",
                f"the storage answered a range read with status {status}",
            )
        observed_etag = _unquote_etag(response.get("ETag"))
        if observed_etag != self._etag:
            raise _failure(
                "source_integrity_mismatch", "source_changed",
                "the object's ETag differs from the recorded one",
            )
        content_range = response.get("ContentRange")
        expected_range = f"bytes {start}-{end}/{self._size_bytes}"
        if content_range != expected_range:
            if isinstance(content_range, str) and not content_range.endswith(f"/{self._size_bytes}"):
                raise _failure(
                    "source_integrity_mismatch", "source_size_mismatch",
                    "the object's size differs from the recorded one",
                )
            raise _failure(
                "source_unavailable", "range_not_honoured",
                "the storage answered another range than the one asked",
            )
        if response.get("ContentLength") != length:
            raise _failure(
                "source_unavailable", "range_not_honoured",
                "the storage announced another length than the range asked",
            )


def _body_bytes(body: Any, length: int) -> bytes:
    """What the answer's body gives of the `length` bytes asked; a typed failure where the client
    failed the read with an error of its own (the connection broke, or went silent)."""
    try:
        return body.read(length)
    except BotoCoreError:
        failure = _failure(
            "source_unavailable", "short_read",
            "the storage ended the body early", retryable=True,
        )
    # Raised here, after the handler has ended: the client's error is neither cause nor context.
    raise failure


def _client_error_failure(error: ClientError) -> WorkerFailure:
    details = error.response.get("Error", {})
    code = str(details.get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == HTTP_PRECONDITION_FAILED or code in PRECONDITION_FAILED_CODES:
        return _failure(
            "source_integrity_mismatch", "source_changed",
            "the object's ETag differs from the recorded one",
        )
    if status == HTTP_RANGE_NOT_SATISFIABLE or code in INVALID_RANGE_CODES:
        return _failure(
            "source_integrity_mismatch", "source_size_mismatch",
            "the object is shorter than its recorded size",
        )
    if status == HTTP_NOT_FOUND or code in MISSING_OBJECT_CODES:
        return _failure("source_unavailable", "source_missing", "the source object does not exist")
    if status == HTTP_FORBIDDEN or code in ACCESS_DENIED_CODES:
        return _failure("source_unavailable", "source_access_denied", "the storage refused the read")
    return _failure(
        "source_unavailable", "storage_error",
        f"the storage answered status {status}", retryable=True,
    )


def _check_range(start: int, length: int, size_bytes: int) -> None:
    for value in (start, length):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("start and length must be integers")
    if start < 0 or length < 1 or start + length > size_bytes:
        raise ValueError(f"range {start}+{length} is outside the object of {size_bytes} bytes")


@dataclass(frozen=True, slots=True)
class ReadStats:
    bytes_read: int
    requests: int


@dataclass(frozen=True, slots=True)
class ReadTime:
    """Where a budgeted reader's time went so far, both numbers from one reading of its clock."""

    seconds_in_reads: float  # in which a read of the source was under way; never above seconds_elapsed
    seconds_elapsed: float  # since the reader was made, which is when its budget began


class BudgetedSource:
    """A RangeSource that refuses any read past its byte, request or time budget.

    A read is charged before it is issued: storage bills the request whether or not the bytes
    arrive. Thread-safe, because the prober may hold several connections at once.

    A read past the time budget is refused with PROBE_DEADLINE_EXCEEDED. Whether the storage was what
    used the time is not decided here: the reader measures, on `monotonic`, the time in which at
    least one read of the inner source was under way, so reads that overlap on several threads count
    once and the time in reads never exceeds the time since the reader was made. A read still in
    flight counts up to the moment of asking, and a read that raised counts like one that returned.
    `read_time()` gives the measure to whoever types a preflight that ran out of time.
    """

    def __init__(
        self,
        inner: RangeSource,
        *,
        max_bytes: int,
        max_requests: int,
        deadline: Deadline,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_bytes < 1 or max_requests < 1:
            raise ValueError("budgets must be positive")
        self._inner = inner
        self._max_bytes = max_bytes
        self._max_requests = max_requests
        self._deadline = deadline
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._bytes_read = 0
        self._requests = 0
        # The time in reads: when the budget began, how many reads are under way, since when at least
        # one has been, and the seconds of every such stretch that has ended.
        self._made_at = monotonic()
        self._reads_under_way = 0
        self._reading_since = self._made_at
        self._seconds_in_ended_reads = 0.0

    @property
    def size_bytes(self) -> int:
        return self._inner.size_bytes

    @property
    def stats(self) -> ReadStats:
        with self._lock:
            return ReadStats(bytes_read=self._bytes_read, requests=self._requests)

    def read_time(self) -> ReadTime:
        """The seconds the reads of the inner source have taken so far, a read in flight included up
        to now, and the seconds since this reader was made."""
        with self._lock:
            now = self._monotonic()
            in_reads = self._seconds_in_ended_reads
            if self._reads_under_way > 0:
                in_reads += now - self._reading_since
            elapsed = now - self._made_at
        # Each stretch is the difference of two readings, so their sum can round a hair above the whole.
        return ReadTime(seconds_in_reads=min(in_reads, elapsed), seconds_elapsed=elapsed)

    def read(self, start: int, length: int) -> bytes:
        _check_range(start, length, self._inner.size_bytes)
        with self._lock:
            if self._deadline.expired():
                raise WorkerFailure(PROBE_DEADLINE_EXCEEDED)
            if self._requests + 1 > self._max_requests:
                raise _failure(
                    "input_limits_exceeded", "probe_budget_exceeded",
                    f"the preflight needs more than {self._max_requests} storage requests",
                )
            if self._bytes_read + length > self._max_bytes:
                raise _failure(
                    "input_limits_exceeded", "probe_budget_exceeded",
                    f"the preflight needs more than {self._max_bytes} bytes of the source",
                )
            self._requests += 1
            self._bytes_read += length
            self._read_began()
        try:
            return self._inner.read(start, length)
        finally:
            with self._lock:
                self._read_ended()

    def _read_began(self) -> None:
        """Called with the lock held, as _read_ended is: the clock is read in the order the reads are counted."""
        if self._reads_under_way == 0:
            self._reading_since = self._monotonic()
        self._reads_under_way += 1

    def _read_ended(self) -> None:
        self._reads_under_way -= 1
        if self._reads_under_way == 0:
            self._seconds_in_ended_reads += self._monotonic() - self._reading_since
