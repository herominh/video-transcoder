"""How the job touches a bucket: it reads the source object and stores the generation, nothing else.

Each side has its own client of an access profile: storage_client() reads the source, upload_client()
stores the generation. Neither follows a region redirect. botocore's region redirector, registered on
every S3 client, would send a request again, signed for the region the storage's answer names, and
ask for the bucket's head where the answer names none; it is stopped where each client is made
(without_region_redirects() of core/source_reader.py). So one call of a client is one request, and a
redirect answer comes back as the storage's own error, a ClientError.

Every request is held to the execution's deadline while it is in flight, and to a time ceiling of its
own. The client's read time-out bounds a silence only, so a storage that trickles a body or withholds
its answer would hold a billed worker without limit. Each request therefore runs on a helper thread,
and the job waits for it no longer than the deadline leaves, nor longer than a request of its size may
take (_within, _request_ceiling_s). A request that has not ended by then is abandoned: the job goes on
to its typed failure, the helper thread ends when its request does, and whatever the request returns
or raises afterwards is dropped with it, unseen. A request that ends after the deadline is dropped the
same way, however soon after it ends. An abandoned upload is given no more of its file, which the
job's own thread closes as it stops waiting. A request abandoned at the deadline is the deadline
failure, which is final. One abandoned at its ceiling while the deadline still has time left has
stalled, which passes: it is a failed attempt (source_read_stalled, upload_stalled, both retryable).
DeadlineSource holds to the same rules the reads this module does not make itself (the preflight's).

A failed attempt that can pass is made again, but only after a wait: a storage that throttles or
stumbles is not helped by a request that comes back at once. The waits grow from attempt to attempt
(DOWNLOAD_RETRY_WAITS_S, UPLOAD_RETRY_WAITS_S), each is shortened by a factor drawn at random, so that
workers that failed together do not come back together, and an upload waits at least as long as the
storage's answer asks (`Retry-After`, RETRY_AFTER_MAX_S at most). No wait goes past the deadline: when
the next one would, the request is not made again and its last failure stands. Because a request that
stalled at its ceiling is made again, a further attempt can start while an abandoned request for the
same object still runs. That is safe. An abandoned
read's bytes are dropped, whenever they arrive. An abandoned PutObject and its repeat can only ever
store the recorded bytes of the same file under the same key (a body whose bytes are not the recorded
ones is refused), so whichever of them lands, and in whatever order, the object is the same.

abandoned_requests() counts the requests this process abandoned while they still ran, at the deadline
or at their ceiling. Each keeps its thread and its connection until the storage ends it, so once the
count is above zero the entry point asks for a fresh process with every job it returns.

download_source() copies the source into the scratch directory through RangeSource.read(), so every byte
arrives through the reader's own checks: in production an S3RangeSource, whose ranged reads are each
conditional on the recorded ETag. A source that changed while it was read fails the job. The close of
the file is its last write: one that fails after the whole copy is a failed write, and no file is left.

upload_generation() stores exactly the files the manifest lists and then the manifest, each under
`<prefix>/<relative path>` (contract README 15.1, rule 3: every object a runtime writes has a key that
starts with the granted prefix and `/`). The manifest is the completion marker, so it goes last, only
after every artifact was stored. It takes its list from the caller, never from a walk of the output
directory, and refuses, before any request, a list that is not as many artifacts, or not of the total
size, the manifest records, and a bucket that is no contract bucket. It only ever puts: it lists
nothing, reads nothing back and deletes nothing. The bytes a client reads of a file are counted and
hashed as it reads them, and an object counts as stored only when they have the size and the SHA-256
the manifest records.
A pass over a file that ends with other bytes is refused where it ends: before anything is sent, with a
client that reads a body through to sign it. A file is opened anew for every request, without waiting
(a FIFO put where it was is refused at once, as any file that is no longer a regular one of the
recorded size), and a file that cannot be opened, or fails under a read, ends the upload as a file
that changed, named by its relative path and the errno's name.
A refusal that sending the object again cannot change (the credentials may not write, the bucket does
not exist) ends the upload at once and for good; every other failed request is sent again, after its
wait.

No log line and no failure detail holds an endpoint, a URL, a credential or the storage client's own
error text: a storage error is named by its class and, for an answer of the storage, its error code
where that has the shape of one (ERROR_CODE_PATTERN; any other value counts as no code at all), and of
a `Retry-After` header nothing is kept but its number of seconds. An error of the operating system is
named by its errno's symbolic name, never by its own text, which may name a path. A typed failure this
module raises in place of another error (the storage client's, the operating system's, the
interpreter's for a thread that could not be started) carries nothing of that error, neither as its
cause nor as its context: it is raised once the handler that caught the error has ended. The line that
tells of an error is written after that `except` block too. A log handler that fails on a line does not
raise: Python's logging prints a report of its own, with every error that failure is chained to, which
inside the block would be the caught error with its own text.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import io
import json
import logging
import os
import random
import re
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import Failure, WorkerFailure, make_detail
from core.package import ManifestOutput
from core.profile import KIB, MIB
from core.protocol import schema
from core.protocol.outbox import SealedMessage, runpod_output
from core.protocol.registry import MessageKind
from core.protocol.settings import StorageProfile
from core.source_reader import (
    S3_CONNECT_TIMEOUT_S,
    S3_MAX_ATTEMPTS,
    S3_READ_TIMEOUT_S,
    RangeSource,
    make_s3_client,
    without_region_redirects,
)

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_BYTES = 8 * MIB
# How long the job waits before it makes a failed request again, in seconds: one entry for each further
# attempt. So one chunk's read is tried twice more at most after a failure the source calls retryable,
# and one object's upload three times more after a failed request (a failed upload throws away a
# finished encode). These are the nominal waits: each is multiplied by a factor drawn between
# RETRY_JITTER_MIN and RETRY_JITTER_MAX (_jitter_factor).
DOWNLOAD_RETRY_WAITS_S = (0.5, 2.0)
UPLOAD_RETRY_WAITS_S = (0.5, 2.0, 8.0)
RETRY_JITTER_MIN = 0.5
RETRY_JITTER_MAX = 1.0
# The most seconds a storage's `Retry-After` can ask one wait to last, and the name botocore keeps that
# header under: lower case, among the answer's other headers.
RETRY_AFTER_MAX_S = 30.0
RETRY_AFTER_HEADER = "retry-after"
# What one storage request may take, whatever the deadline leaves (_request_ceiling_s): a base, and the
# time its bytes take at the slowest rate that still counts as moving.
REQUEST_BASE_S = 30.0
REQUEST_MIN_BYTES_PER_S = 256 * KIB
SOURCE_FILE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
SOURCE_FILE_MODE = 0o600
# How a file to upload is opened, besides for reading: through no link at its last part, and without
# waiting. An open that waits never returns for a FIFO put where the file was, until a writer comes.
ARTIFACT_OPEN_FLAGS = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
UNKNOWN_ERRNO_NAME = "unknown"
KEY_SEPARATOR = "/"
RESULT_NAME = "result.json"
# How the upload client signs, addresses and retries: as the source reader's client does (core/source_reader.py).
SIGNATURE_VERSION = "s3v4"
ADDRESSING_STYLE = "path"
RETRY_MODE = "standard"
# botocore's setting for "a checksum only where the operation requires one".
CHECKSUM_WHEN_REQUIRED = "when_required"
# What the Hub's delivery and a player expect of each file the worker writes; the manifest is the one JSON file.
CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".jpg": "image/jpeg",
    ".json": "application/json",
}
_COMMON_DEFINITIONS = schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/"
_REL_PATH_DEFINITION = _COMMON_DEFINITIONS + "rel_path"
_OBJECT_KEY_DEFINITION = _COMMON_DEFINITIONS + "object_key"
_BUCKET_DEFINITION = _COMMON_DEFINITIONS + "bucket"
# The helper thread one storage request runs on (_within).
REQUEST_THREAD_NAME = "vh-storage-request"
# A PutObject the storage answered with a redirect or a client error is refused for good: the same
# request would get the same answer, so it is not sent again and the job is not worth dispatching again.
HTTP_FIRST_REDIRECT = 300
HTTP_LAST_CLIENT_ERROR = 499
HTTP_FOUND = 302
HTTP_TEMPORARY_REDIRECT = 307
HTTP_REQUEST_TIMEOUT = 408
HTTP_CONFLICT = 409
HTTP_TOO_MANY_REQUESTS = 429
# The answers in that range that pass all the same, by their status or by their error code.
PASSING_REFUSAL_STATUSES = frozenset(
    {
        HTTP_FOUND,  # the storage says the object's address is elsewhere for now
        HTTP_TEMPORARY_REDIRECT,  # as 302: elsewhere for now, not for good
        HTTP_REQUEST_TIMEOUT,  # the storage stopped waiting for the request to arrive
        HTTP_CONFLICT,  # another operation on the same key was under way
        HTTP_TOO_MANY_REQUESTS,  # the storage limits how fast it is asked
    }
)
PASSING_REFUSAL_CODES = frozenset(
    {
        "RequestTimeout",  # the body arrived too slowly for the storage's own time-out
        "RequestTimeTooSkewed",  # this host's clock is off: another start, on another host, can pass
        "IncompleteBody",  # fewer bytes arrived than were announced: the connection broke
        "BadDigest",  # the bytes that arrived are not the bytes that were signed: damaged on the way
        "XAmzContentSHA256Mismatch",  # as BadDigest, by the payload's signed hash: damaged or changed on the way
        "SlowDown",  # the storage limits how fast it is asked, whatever status it answers with
    }
)

# The shape an error code of the storage's answer is kept in. The answer is the storage's own text: a
# value of any other shape is no code at all, to a failure's detail, to a log line and to the decision
# whether a refusal passes.
ERROR_CODE_PATTERN = re.compile(r"[A-Za-z0-9.]{1,64}")

DOWNLOAD_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "download_deadline_exceeded", False,
    "the job ran out of time while it downloaded the source",
)
UPLOAD_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "upload_deadline_exceeded", False,
    "the job ran out of time while it uploaded the generation",
)
SOURCE_READ_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "source_read_deadline_exceeded", False,
    "the job ran out of time while it read the source",
)
# A request that outlasted its own ceiling while the deadline still had time left: a failed attempt
# that passes, unlike a deadline (a slow network or storage, not a limit that is too small).
SOURCE_READ_STALLED = Failure(
    "source_unavailable", "source_read_stalled", True,
    "a read of the source did not end within the time a read of its size may take",
)
UPLOAD_STALLED = Failure(
    "output_write_failed", "upload_stalled", True,
    "an upload did not end within the time an object of its size may take",
)
STORAGE_REQUEST_START_FAILED = Failure(
    "resource_exhausted", "storage_request_start_failed", True,
    "a storage request could not be started: no thread could be made for it",
)
SCRATCH_FULL = Failure("resource_exhausted", "scratch_full", True, "the scratch disk is full")

_Returned = TypeVar("_Returned")


def storage_client(profile: StorageProfile) -> Any:
    """The S3 client the source is read with: the endpoint and credentials a dispatch's credential_ref names."""
    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a StorageProfile")
    return make_s3_client(
        endpoint_url=profile.endpoint,
        region=profile.region,
        access_key_id=profile.access_key_id,
        secret_access_key=profile.secret_access_key,
    )


def upload_client(profile: StorageProfile) -> Any:
    """The S3 client the generation is stored with: the access profile's endpoint and credentials, and
    the source reader's signing, path-style addressing, time-outs and single attempt.

    It adds a checksum only where an operation requires one: botocore's default since 1.36 sends a
    PutObject as an `aws-chunked` body with a checksum trailer, which not every S3-compatible store
    accepts, while a plain signed PUT with a Content-Length is accepted by all of them.
    """
    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a StorageProfile")
    return without_region_redirects(
        boto3.client(
            "s3",
            endpoint_url=profile.endpoint,
            region_name=profile.region,
            aws_access_key_id=profile.access_key_id,
            aws_secret_access_key=profile.secret_access_key,
            config=Config(
                signature_version=SIGNATURE_VERSION,
                connect_timeout=S3_CONNECT_TIMEOUT_S,
                read_timeout=S3_READ_TIMEOUT_S,
                retries={"total_max_attempts": S3_MAX_ATTEMPTS, "mode": RETRY_MODE},
                s3={"addressing_style": ADDRESSING_STYLE},
                request_checksum_calculation=CHECKSUM_WHEN_REQUIRED,
                response_checksum_validation=CHECKSUM_WHEN_REQUIRED,
            ),
        )
    )


def _errno_name(error: OSError) -> str:
    # The errno's symbolic name only: an OSError's own message may carry a path.
    return errno.errorcode.get(error.errno or 0, UNKNOWN_ERRNO_NAME)


def _notify(on_bytes: Callable[[int], None] | None, total_bytes: int) -> None:
    """Tell the caller how far the transfer is; whatever its callback raises never fails the transfer."""
    if on_bytes is None:
        return
    error_class: str | None = None
    try:
        on_bytes(total_bytes)
    except Exception as error:  # noqa: BLE001 - progress is the caller's convenience, never the transfer's outcome
        # Only the class name is kept past this block: what a callback raises can say anything.
        error_class = type(error).__name__
    if error_class is not None:
        logger.warning("a transfer's progress callback failed: %s", error_class)


def _require_transfer_arguments(deadline: object, sleep: object, on_bytes: object) -> None:
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if not callable(sleep):
        raise TypeError("sleep must be callable")
    if on_bytes is not None and not callable(on_bytes):
        raise TypeError("on_bytes must be callable or None")


# --- the wait before a failed request is made again -------------------------------------------------


def _jitter_factor() -> float:
    """The factor one wait is multiplied by, drawn anew for every wait: workers whose requests failed
    together then do not all come back together."""
    # `random` will do: the draw spreads load, it protects nothing.
    return random.uniform(RETRY_JITTER_MIN, RETRY_JITTER_MAX)


def _next_wait_s(nominal_waits_s: Iterator[float], deadline: Deadline, *, at_least_s: float = 0.0) -> float | None:
    """How many seconds to wait before a failed request is made again; None when it is not made again:
    its attempts are used up, or the wait would not end before the deadline.

    `nominal_waits_s` gives the nominal wait before each further attempt of one request and ends with
    the last of them. The wait is the nominal one times the drawn factor, or `at_least_s` when that is
    longer (what the storage's answer asked for). Whoever gets None raises the request's last failure as
    it is: time too short for another attempt is not a deadline the request ran into.
    """
    nominal_s = next(nominal_waits_s, None)
    if nominal_s is None:
        return None
    wait_s = max(nominal_s * _jitter_factor(), at_least_s)
    return wait_s if wait_s < deadline.remaining_s() else None


def _retry_after_s(error: BotoCoreError | ClientError) -> float:
    """The seconds the storage's answer asks the next request to wait, RETRY_AFTER_MAX_S at most; 0 when
    the error is no answer of the storage, or its answer asks for none.

    Only a `Retry-After` of whole seconds counts, a string of ASCII digits and nothing else: an HTTP
    date, a fraction, a negative number or an answer of any other shape asks for nothing. Nothing of
    the header is kept but that number.
    """
    if not isinstance(error, ClientError):
        return 0.0
    metadata = error.response.get("ResponseMetadata")
    headers = metadata.get("HTTPHeaders") if isinstance(metadata, Mapping) else None
    asked = headers.get(RETRY_AFTER_HEADER) if isinstance(headers, Mapping) else None
    if not isinstance(asked, str) or not (asked.isascii() and asked.isdigit()):
        return 0.0
    significant = asked.lstrip("0")
    # More digits than the most seconds have is more seconds than that, whatever the digits. So no
    # number of any length is converted: Python's int() refuses a very long one (3.11, and 3.10 from 3.10.7).
    if len(significant) > len(str(int(RETRY_AFTER_MAX_S))):
        return RETRY_AFTER_MAX_S
    return min(float(significant or "0"), RETRY_AFTER_MAX_S)


# --- a request in flight -------------------------------------------------------------------------

# How many requests this process stopped waiting for while they still ran, at the deadline or at their
# ceiling (_within). It only grows. Under a lock, because two threads can stop waiting at the same
# instant: the preflight's reads wait on the threads of its loopback server (core/probe.py), which the
# preflight does not wait for when it ends, so one of them may still be waiting when the job's own
# thread waits for a download's read, and both wait to the execution's one deadline.
_abandoned_requests = 0
_abandoned_requests_lock = threading.Lock()


def abandoned_requests() -> int:
    """How many storage requests this process has abandoned while they still ran, at the deadline or at
    their ceiling.

    Each of them keeps its helper thread and its connection until the storage ends the request, and it
    belongs to no single job (a read of the preflight can be given up after its job returned). The count
    never falls: once it is above zero the process is marked, and the entry point asks for a fresh
    worker with every job it returns.
    """
    with _abandoned_requests_lock:
        return _abandoned_requests


def _count_abandoned_request() -> None:
    global _abandoned_requests
    with _abandoned_requests_lock:
        _abandoned_requests += 1


class _Request:
    """One storage request on its helper thread, and how it ended once `ended` is set."""

    def __init__(self, call: Callable[[], Any], deadline: Deadline) -> None:
        self._call = call
        self._deadline = deadline
        self.ended = threading.Event()
        self.ended_in_time = False
        self.returned: Any = None
        self.raised: BaseException | None = None

    def run(self) -> None:
        # Whatever the call raises is kept for the thread that waits, or dropped with an abandoned
        # request. Nothing leaves this thread: the interpreter would print an error that escaped it.
        try:
            self.returned = self._call()
        except BaseException as error:  # noqa: BLE001 - raised again by the thread that waits
            self.raised = error
        finally:
            # Stamped here, where the request ends: the thread that waits may come to look much later.
            self.ended_in_time = self._deadline.remaining_s() > 0
            self.ended.set()


def _request_ceiling_s(size_bytes: int) -> float:
    """The seconds one storage request of `size_bytes` may take, whatever the deadline leaves:
    REQUEST_BASE_S for the exchange itself, and the time its bytes take at REQUEST_MIN_BYTES_PER_S, the
    slowest rate that still counts as moving. A request that takes longer has stalled. (A size below
    zero is a caller's bug, which the request itself refuses: it gets the base.)"""
    return REQUEST_BASE_S + max(size_bytes, 0) / REQUEST_MIN_BYTES_PER_S


def _within(
    deadline: Deadline,
    call: Callable[[], _Returned],
    *,
    expired: Failure,
    stalled: Failure | None = None,
    ceiling_s: float | None = None,
) -> _Returned:
    """What `call()` returns, or raises, when it ends before `deadline`; WorkerFailure(expired) when it does not.

    Nothing can cut short a request the storage client is in the middle of, so the call runs on a helper
    thread and this one waits for it only as long as the deadline leaves. What the call raises in time is
    raised here, the same object. A call that has not ended in time is abandoned: its thread goes on until
    the request ends, and what it then returns or raises is dropped, with nothing printed or logged. So
    is the outcome of a call that ended after the deadline, however soon after: the helper thread notes
    at the call's ending whether time was left, and this thread goes by that note, not by when it came
    to look. When no time is left, no request is started. A helper thread that cannot be started is a
    retryable WorkerFailure.

    `ceiling_s` and `stalled`, given together, hold the call to a time of its own as well: this thread
    then waits no longer than the ceiling, and a call that has not ended at its ceiling while the
    deadline still has time left is abandoned in the same way and raises WorkerFailure(stalled), a
    failed attempt its caller may make again. A call that has not ended when the deadline passes is
    WorkerFailure(expired) whatever its ceiling: when the deadline leaves no more time than the ceiling,
    this thread waits for the deadline, and a ceiling the deadline passed under counts for nothing. A
    call that ended, in time or not, is judged as without a ceiling: the ceiling bounds this thread's
    wait, so a call that has ended by the time this thread looks counts as ended, even a moment past it.

    Only a call this thread stopped waiting for while it still ran is counted (abandoned_requests()),
    whether it was given up at its ceiling or at the deadline: a call that ended, in time or not, left
    no thread and no connection behind.
    """
    remaining_s = deadline.remaining_s()
    if remaining_s <= 0:
        raise WorkerFailure(expired)
    request = _Request(call, deadline)
    # A daemon thread: an abandoned request never keeps the process from ending.
    thread = threading.Thread(target=request.run, name=REQUEST_THREAD_NAME, daemon=True)
    try:
        thread.start()
        started = True
    except RuntimeError:
        # The interpreter's own text says nothing the typed failure does not.
        started = False
    if not started:
        # Raised once the handler has ended: the interpreter's error is neither cause nor context.
        raise WorkerFailure(STORAGE_REQUEST_START_FAILED)
    # The ceiling counts only where it is the nearer bound. A wait that was for the deadline ends as the
    # deadline failure, even should a clock's grain then show a sliver of time left.
    wait_s = remaining_s
    if stalled is not None and ceiling_s is not None and ceiling_s < remaining_s:
        wait_s = ceiling_s
    # A wait longer than TIMEOUT_MAX is refused by the lock it waits on (OverflowError).
    ended = request.ended.wait(min(wait_s, threading.TIMEOUT_MAX))
    if not ended:
        # Still running, at its ceiling or at the deadline: its thread and its connection stay until
        # the storage ends the request.
        _count_abandoned_request()
    if not ended and stalled is not None and wait_s < remaining_s and deadline.remaining_s() > 0:
        raise WorkerFailure(stalled)
    if not ended or not request.ended_in_time:
        raise WorkerFailure(expired)
    if request.raised is not None:
        raise request.raised
    return request.returned


class DeadlineSource:
    """A RangeSource whose every read returns or fails by a deadline, and within the time a read of its
    length may take: for the reads of a source that this module does not make itself (the preflight's).

    A read that has not ended by the deadline raises WorkerFailure (deadline_exceeded /
    source_read_deadline_exceeded, not retryable) and is abandoned, as _within describes. A read that
    has not ended at the ceiling of its length (_request_ceiling_s) while the deadline still has time
    left is abandoned in the same way and raises WorkerFailure (source_unavailable /
    source_read_stalled, retryable). It is not read again here: that is its caller's to decide, and
    whatever an abandoned read still fetches is dropped. A read that ends in time returns what the
    inner source returned or raises what it raised, unchanged.
    """

    def __init__(self, inner: RangeSource, deadline: Deadline) -> None:
        if not callable(getattr(inner, "read", None)):
            raise TypeError("inner must be a RangeSource")
        size_bytes = getattr(inner, "size_bytes", None)
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise TypeError("inner must be a RangeSource whose size_bytes is an integer")
        if not isinstance(deadline, Deadline):
            raise TypeError("deadline must be a Deadline")
        self._inner = inner
        self._deadline = deadline

    @property
    def size_bytes(self) -> int:
        return self._inner.size_bytes

    def read(self, start: int, length: int) -> bytes:
        return _within(
            self._deadline,
            functools.partial(self._inner.read, start, length),
            expired=SOURCE_READ_DEADLINE_EXCEEDED,
            stalled=SOURCE_READ_STALLED,
            ceiling_s=_request_ceiling_s(length),
        )


# --- the source ----------------------------------------------------------------------------------


def download_source(
    source: RangeSource,
    *,
    destination: Path,
    deadline: Deadline,
    sleep: Callable[[float], None],
    on_bytes: Callable[[int], None] | None = None,
) -> int:
    """Copy the whole source object to the new file `destination` and return the bytes written.

    The object is read in order, DOWNLOAD_CHUNK_BYTES at a time, only through `source.read()`. A chunk
    whose read fails with a retryable WorkerFailure is read again, once for each entry of
    DOWNLOAD_RETRY_WAITS_S at most and each time only after a wait: that entry's seconds times a factor
    drawn between RETRY_JITTER_MIN and RETRY_JITTER_MAX, waited through `sleep(seconds)`. A wait that
    would not end before `deadline` is not made: the chunk is not read again, and its read's last
    failure is raised as it is. Every other failure of a read ends the download at once, with no wait.

    No read starts once `deadline` has passed, and a read still in flight when it passes is abandoned
    (_within). A read that has not ended at the ceiling of its length (_request_ceiling_s) while the
    deadline still has time left is abandoned as well and is a failed attempt of the retryable kind
    (source_unavailable / source_read_stalled). The chunk may then be read again while the abandoned
    read still runs, which is safe: whatever an abandoned read returns is dropped. `on_bytes` is called
    with the bytes written so far after each chunk.

    Raises WorkerFailure for a failed read, a chunk of another length than asked (the object is not the
    recorded size), a passed deadline and a failed write; the partial file is removed first. The close
    of the file is its last write: one that fails after a whole copy is a failed write like any other
    (a file system may report only then that it could not keep the bytes), and the file is removed.
    Behind a copy that failed, the copy's own failure stands whatever the close does. Raises
    TypeError or ValueError for invalid arguments or an existing destination, before anything is read.
    """
    size_bytes = _validated_download(source, destination)
    _require_transfer_arguments(deadline, sleep, on_bytes)
    descriptor = _created_source_file(destination)
    copy_error: BaseException | None = None
    try:
        written = _copy_source(source, size_bytes, descriptor, deadline, sleep, on_bytes)
    except BaseException as error:  # noqa: BLE001 - kept, and raised again below as it is
        copy_error = error
    if copy_error is not None:
        # Behind the `except` block, not inside it: a line either step logs is then not written while
        # the copy's error is still being handled (see the module's last paragraph).
        _close_behind_a_failure(descriptor)
        _remove_partial(destination)
        try:
            raise copy_error
        finally:
            # Let go of, as Python lets go of a caught error where its block ends: the error's stack
            # holds this frame, and this frame would hold the error.
            del copy_error
    failed_close = _failed_close_of(descriptor)
    if failed_close is not None:
        _remove_partial(destination)
        raise _source_write_failure(failed_close)
    return written


def _validated_download(source: object, destination: object) -> int:
    """The source's size, once the source and the destination are what a download can use."""
    if not callable(getattr(source, "read", None)):
        raise TypeError("source must be a RangeSource")
    size_bytes = getattr(source, "size_bytes", None)
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
        raise ValueError("the source's size_bytes must be a positive integer")
    if not isinstance(destination, Path) or not destination.is_absolute():
        raise ValueError("destination must be an absolute Path")
    if os.path.lexists(destination):
        raise ValueError("destination already exists")
    return size_bytes


def _copy_source(
    source: RangeSource,
    size_bytes: int,
    descriptor: int,
    deadline: Deadline,
    sleep: Callable[[float], None],
    on_bytes: Callable[[int], None] | None,
) -> int:
    written = 0
    while written < size_bytes:
        length = min(DOWNLOAD_CHUNK_BYTES, size_bytes - written)
        data = _read_chunk(source, written, length, deadline, sleep)
        if len(data) != length:
            raise WorkerFailure(
                Failure(
                    "source_integrity_mismatch", "source_size_mismatch", False,
                    f"the source gave {len(data)} bytes where {length} were read",
                )
            )
        failed_write = _failed_write_of(descriptor, data)
        if failed_write is not None:
            raise _source_write_failure(failed_write)
        written += length
        _notify(on_bytes, written)
    return written


def _read_chunk(
    source: RangeSource, start: int, length: int, deadline: Deadline, sleep: Callable[[float], None]
) -> bytes:
    """One chunk through the source's own checks, each read held to the deadline and to the ceiling of
    its length, read again only after a failure it calls retryable and the wait that follows it.

    A read that stalled (it outlasted its ceiling while the deadline had time left) is such a failure.
    It may still be running when the chunk is read again: its bytes are dropped, so only the read
    that is waited for can give the chunk. When no attempt is left, or the next wait would not end
    before the deadline, the last failure is raised as it is, a stalled read's included."""
    nominal_waits_s = iter(DOWNLOAD_RETRY_WAITS_S)
    while True:
        if deadline.expired():
            raise WorkerFailure(DOWNLOAD_DEADLINE_EXCEEDED)
        try:
            return _within(
                deadline,
                functools.partial(source.read, start, length),
                expired=DOWNLOAD_DEADLINE_EXCEEDED,
                stalled=SOURCE_READ_STALLED,
                ceiling_s=_request_ceiling_s(length),
            )
        except WorkerFailure as failed:
            wait_s = _next_wait_s(nominal_waits_s, deadline) if failed.failure.retryable else None
            if wait_s is None:
                raise
            # Only the typed failure is kept past this block, not the error that carried it.
            failure = failed.failure
        logger.warning(
            "a read of the source failed and is tried again after %.3f s: %s/%s",
            wait_s, failure.error_class, failure.code,
        )
        sleep(wait_s)


def _write_all(descriptor: int, data: bytes) -> None:
    """Write every byte of `data`, however few of them each call takes."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "a write made no progress")
        remaining = remaining[written:]


@dataclass(frozen=True)
class _FailedWrite:
    """What is kept of an error of the operating system that failed a write of the source file, once
    the `except` block that caught it has ended: its class's name, its errno's symbolic name and whether
    it says that the disk is full. The error itself is not kept: its own text may name a path."""

    error_class: str
    errno_name: str
    disk_full: bool

    @classmethod
    def of(cls, error: OSError) -> _FailedWrite:
        return cls(type(error).__name__, _errno_name(error), error.errno in DISK_FULL_ERRNOS)


def _created_source_file(destination: Path) -> int:
    """The descriptor of the new, empty source file. A file that cannot be created is a failed write:
    nothing of ours is then to be removed (the file was not created, or it was not created by this call)."""
    try:
        return os.open(destination, SOURCE_FILE_FLAGS, SOURCE_FILE_MODE)
    except OSError as error:
        failed = _FailedWrite.of(error)
    raise _source_write_failure(failed)


def _failed_write_of(descriptor: int, data: bytes) -> _FailedWrite | None:
    """Write a chunk to the source file: None once every byte of it is written, and what is kept of
    the operating system's error when the write fails."""
    try:
        _write_all(descriptor, data)
    except OSError as error:
        return _FailedWrite.of(error)
    return None


def _failed_close_of(descriptor: int) -> _FailedWrite | None:
    """Close the source file a whole copy was written to: None when that worked, and what is kept of
    the operating system's error when it did not (the close is the file's last write)."""
    try:
        os.close(descriptor)
    except OSError as error:
        return _FailedWrite.of(error)
    return None


def _source_write_failure(failed: _FailedWrite) -> WorkerFailure:
    """The typed failure of a write of the source file that failed, and the one line that tells of it.

    Called once the `except` block that caught the operating system's error has ended, with what was
    kept of that error: the error is then neither what a failing log handler's report is chained to
    nor the failure's cause or context."""
    logger.warning("the source file could not be written: %s errno=%s", failed.error_class, failed.errno_name)
    if failed.disk_full:
        return WorkerFailure(SCRATCH_FULL)
    return WorkerFailure(
        Failure(
            "output_write_failed", "source_write_failed", True,
            f"the source file could not be written: {failed.errno_name}",
        )
    )


def _close_behind_a_failure(descriptor: int) -> None:
    """Close the source file a copy failed on. What the copy failed with stands, so a close that fails
    too is logged, by its errno's name, and raises nothing."""
    errno_name: str | None = None
    try:
        os.close(descriptor)
    except OSError as error:
        errno_name = _errno_name(error)
    if errno_name is not None:
        logger.warning("a partial source file could not be closed: errno=%s", errno_name)


def _remove_partial(path: Path) -> None:
    """Best effort: the job's scratch cleanup removes whatever is left. Paths are never logged."""
    errno_name: str | None = None
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as error:
        # Only the errno's name is kept past this block: the error's own text names the path.
        errno_name = _errno_name(error)
    if errno_name is not None:
        logger.warning("a partial source file could not be removed: errno=%s", errno_name)


# --- the generation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Object:
    """One object's key and the bytes its PUT must carry."""

    relative_path: str
    key: str
    size_bytes: int
    sha256: str
    content_type: str


@dataclass(frozen=True)
class _Upload(_Object):
    """An object read from a local file, with the size and hash recorded by the manifest."""

    local_path: Path


def upload_result(
    client: Any, *, bucket: str, prefix: str, result: SealedMessage, deadline: Deadline
) -> None:
    """Attempt one bounded PUT of the terminal result's RunPod carrier (README 15.1, rule 9).

    The carrier lives in memory, so even a failure before scratch setup can be stored. It is no
    artifact and contributes to no inventory or byte count. The caller keeps the sealed result
    whatever this attempt raises; there is no retry delaying its report.
    """
    if result.kind not in {MessageKind.TRANSCODE_RESULT_COMPLETED, MessageKind.TRANSCODE_RESULT_FAILED}:
        raise ValueError("only a terminal result may be stored")
    if not callable(getattr(client, "put_object", None)):
        raise TypeError("client must be an S3 client")
    if not isinstance(bucket, str) or schema.error_pointers_for_schema({"$ref": _BUCKET_DEFINITION}, bucket):
        raise ValueError("bucket must be a contract bucket")
    if not isinstance(prefix, str) or schema.error_pointers_for_schema({"$ref": _OBJECT_KEY_DEFINITION}, prefix):
        raise ValueError("prefix must be a contract object key")
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    carrier = json.dumps(runpod_output(result), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    upload = _Object(
        relative_path=RESULT_NAME,
        key=f"{prefix}{KEY_SEPARATOR}{RESULT_NAME}",
        size_bytes=len(carrier),
        sha256=hashlib.sha256(carrier).hexdigest(),
        content_type=CONTENT_TYPES[".json"],
    )
    _send_body(client, bucket, upload, io.BytesIO(carrier), deadline)


def upload_generation(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    output_root: Path,
    artifacts: Sequence[ArtifactFile],
    manifest: ManifestOutput,
    deadline: Deadline,
    sleep: Callable[[float], None],
    on_bytes: Callable[[int], None] | None = None,
) -> int:
    """Store the generation in `bucket` and return the bytes uploaded: `artifacts` in order, then the manifest.

    Every key is `prefix`, `/` and the file's path relative to `output_root`. Each object is one
    PutObject of the file, whose size on disk must still be the recorded one and whose bytes, as the
    client read them, must have the recorded size and SHA-256. A failed request is sent again, from the
    start of the file, once for each entry of UPLOAD_RETRY_WAITS_S at most and each time only after a
    wait, unless it was refused for good: an answer with an HTTP status from 300 to 499 that is none of
    PASSING_REFUSAL_STATUSES and carries none of PASSING_REFUSAL_CODES, or parameters the client itself
    refuses, is final and waits for nothing. The wait is that entry's seconds times a factor drawn
    between RETRY_JITTER_MIN and RETRY_JITTER_MAX, and at least the whole seconds a `Retry-After` header
    of the storage's answer names, RETRY_AFTER_MAX_S at most; it is waited through `sleep(seconds)`. A
    wait that would not end before `deadline` is not made: the object is not sent again, and its
    request's last failure is raised as it is.

    No request starts once `deadline` has passed; a request still in flight when it passes is abandoned
    (_within), its file is closed and it is given no more of it; a request that ends after it has passed
    does not count either. A request that has not ended at the ceiling of its object's recorded size
    (_request_ceiling_s) while the deadline still has time left is abandoned in the same way, its file
    closed, and is a failed attempt that passes (upload_stalled). The object may then be sent again
    while the abandoned request still runs. That is safe: both can only ever store the recorded bytes of
    the same file under the same key (a body whose bytes are not the recorded ones is refused), so
    whichever of them lands, the object is the same. `on_bytes` is called with the bytes uploaded so far
    after each object.

    Raises TypeError or ValueError, before any request, for anything that would put an object where the
    generation does not belong, or store another generation than the manifest's (a wiring bug): a
    bucket that is no contract bucket, a prefix that is no contract object key, a path that is no
    contract rel_path, two files with one path, a file that is not a regular file of the output root
    or has no content type here, artifacts that are not as many as the manifest's `artifact_count` or
    whose recorded sizes do not add up to its `total_bytes`; a ValueError about a file names its
    relative path only, and one about the bucket does not repeat it. Raises
    WorkerFailure for a file that changed, a passed deadline, a refusal (not retryable) and an upload
    that kept failing or stalling (retryable): no later object is then put, so the manifest is stored only after
    every artifact was. With a client that reads a body through before it sends it (botocore does, to
    sign the payload), a file whose bytes are not the recorded ones is refused before a byte of it is
    sent, the manifest included; a file that changes between that pass and the sending pass is refused
    at the end of the sending pass, when its bytes have gone out under a signature that no longer fits
    them, which a storage that checks the signed payload hash refuses. Only with a client that reads a
    body once, or a storage that does not check that hash, can a wrong object have been stored before
    the upload fails.
    """
    if not callable(getattr(client, "put_object", None)):
        raise TypeError("client must be an S3 client")
    if not isinstance(bucket, str) or schema.error_pointers_for_schema({"$ref": _BUCKET_DEFINITION}, bucket):
        # The refused name is not repeated: what a bug hands over here need not be a bucket's name at all.
        raise ValueError(
            "bucket must be a contract bucket: 3 to 63 lower-case letters, digits and hyphens, no hyphen first or last"
        )
    _require_transfer_arguments(deadline, sleep, on_bytes)
    uploads = _planned_uploads(prefix, output_root, artifacts, manifest)
    _require_what_the_manifest_records(artifacts, manifest)
    uploaded = 0
    for upload in uploads:
        _store(client, bucket, upload, deadline, sleep)
        uploaded += upload.size_bytes
        _notify(on_bytes, uploaded)
    return uploaded


def _planned_uploads(prefix: object, output_root: object, artifacts: object, manifest: object) -> list[_Upload]:
    """Every object of the generation in upload order, the manifest last; refused as a whole for a wiring bug."""
    if not isinstance(prefix, str) or schema.error_pointers_for_schema({"$ref": _OBJECT_KEY_DEFINITION}, prefix):
        raise ValueError("prefix must be a contract object key: no leading or trailing `/`, no `.` or `..` segment")
    if not isinstance(output_root, Path) or not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("output_root must be an absolute Path of an existing directory")
    # Resolved once, so that a link below the root is the only thing that can still move a file.
    root = output_root.resolve()
    return [
        _planned_upload(prefix, root, relative_path, size_bytes, sha256)
        for relative_path, size_bytes, sha256 in _listed_files(artifacts, manifest)
    ]


def _require_what_the_manifest_records(artifacts: Sequence[ArtifactFile], manifest: ManifestOutput) -> None:
    """The artifacts must be exactly what the manifest records: as many as its `artifact_count`, their
    recorded sizes adding up to its `total_bytes`. (The package stage counts both over the artifacts
    the manifest lists; the manifest's own file is in neither.) Asked once each file was found to be
    one the upload can store, so that a file's own fault is named first."""
    if len(artifacts) != manifest.artifact_count:
        raise ValueError(
            f"{len(artifacts)} artifacts were given where the manifest records {manifest.artifact_count}"
        )
    given_bytes = sum(artifact.size_bytes for artifact in artifacts)
    if given_bytes != manifest.total_bytes:
        raise ValueError(
            f"the artifacts given record {given_bytes} bytes where the manifest records {manifest.total_bytes}"
        )


def _listed_files(artifacts: object, manifest: object) -> list[tuple[str, int, str]]:
    """(relative path, recorded size, recorded SHA-256) of every artifact in order, then of the manifest;
    no path twice."""
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise TypeError("artifacts must be a sequence of ArtifactFile")
    if not all(isinstance(artifact, ArtifactFile) for artifact in artifacts):
        raise TypeError("artifacts must hold ArtifactFile entries")
    if not isinstance(manifest, ManifestOutput):
        raise TypeError("manifest must be a ManifestOutput")
    listed = [(artifact.path, artifact.size_bytes, artifact.sha256) for artifact in artifacts]
    listed.append((manifest.path, manifest.size_bytes, manifest.sha256))
    seen: set[str] = set()
    for relative_path, _, _ in listed:
        if not isinstance(relative_path, str) or schema.error_pointers_for_schema(
            {"$ref": _REL_PATH_DEFINITION}, relative_path
        ):
            raise ValueError(f"{relative_path!r} is not a contract rel_path")
        # The manifest comes last, so an artifact that takes its path is found here too.
        if relative_path in seen:
            raise ValueError(f"{relative_path!r} names two files of the generation")
        seen.add(relative_path)
    return listed


def _planned_upload(prefix: str, root: Path, relative_path: str, size_bytes: int, sha256: str) -> _Upload:
    content_type = CONTENT_TYPES.get(os.path.splitext(relative_path)[1])
    if content_type is None:
        raise ValueError(f"{relative_path!r} has no content type this worker uploads")
    local_path = root / relative_path
    if not _is_regular_file_below(root, local_path):
        raise ValueError(f"{relative_path!r} is not a regular file of the output root")
    return _Upload(
        relative_path=relative_path,
        key=f"{prefix}{KEY_SEPARATOR}{relative_path}",
        local_path=local_path,
        size_bytes=size_bytes,
        sha256=sha256,
        content_type=content_type,
    )


def _is_regular_file_below(root: Path, local_path: Path) -> bool:
    """Whether `local_path`, a path below the resolved `root`, is a regular file reached through no link."""
    try:
        is_regular = stat.S_ISREG(os.lstat(local_path).st_mode)
        # `root` is resolved already: resolving changes the path only when a part below it is a link.
        return is_regular and local_path.resolve() == local_path
    except OSError:
        return False


def _store(client: Any, bucket: str, upload: _Upload, deadline: Deadline, sleep: Callable[[float], None]) -> None:
    """Put one object, each request held to the deadline and to the ceiling of the object's size. A
    failed request is sent again, once for each entry of UPLOAD_RETRY_WAITS_S at most and each time
    only after its wait; one that was refused for good is not, and waits for nothing.

    A request that stalled (it outlasted its ceiling while the deadline had time left) is a failed
    request like one the storage or the connection failed: it is sent again after the wait, and when no
    attempt is left, or the next wait would not end before the deadline, the upload ends with the last
    failure as it is (upload_stalled or upload_failed, both retryable). The abandoned request may still
    be running when the object is sent again. Its file is closed, so it gets no more of it, and it and
    its repeat can only ever store the recorded bytes of this one file under this one key: whichever of
    them lands, the object is the same."""
    nominal_waits_s = iter(UPLOAD_RETRY_WAITS_S)
    while True:
        if deadline.expired():
            raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
        try:
            _send(client, bucket, upload, deadline)
            return
        except WorkerFailure as stalled:
            # Every other typed failure of a request ends the upload as it is: the deadline, a file that
            # changed, a request no thread could be made for.
            if stalled.failure is not UPLOAD_STALLED:
                raise
            failed, name, asked_s, for_good = stalled, UPLOAD_STALLED.code, 0.0, False
        except (BotoCoreError, ClientError) as error:
            # The client's own error names the endpoint. Only what is read of it here goes on: the typed
            # failure is raised below, outside this handler, so that the error is neither its cause nor
            # its context.
            name, asked_s, for_good = _storage_error_name(error), _retry_after_s(error), _refused_for_good(error)
            failed = _upload_refused(name) if for_good else _upload_failed(name)
        if for_good:
            logger.warning("an upload was refused for good and is not sent again: %s", name)
            raise failed
        wait_s = _next_wait_s(nominal_waits_s, deadline, at_least_s=asked_s)
        if wait_s is None:
            raise failed
        logger.warning("an upload failed and is tried again after %.3f s: %s", wait_s, name)
        sleep(wait_s)


def _send(client: Any, bucket: str, upload: _Upload, deadline: Deadline) -> None:
    """One request of the file as it is on disk now, opened anew so that every attempt sends it from
    its start, held to the deadline and to the ceiling of the object's recorded size.

    The file is opened and closed here, on the thread that waits, whatever becomes of the request. A
    request that was abandoned, at the deadline or at its ceiling, goes on for as long as the storage
    lets it: it must not keep the file open until then, nor the disk space of a scratch file that was
    removed meanwhile. Closing waits for a local read of the file that request has under way, never for
    the storage.
    """
    _send_body(client, bucket, upload, _opened_artifact(upload), deadline)


def _send_body(client: Any, bucket: str, upload: _Object, source: BinaryIO, deadline: Deadline) -> None:
    """Send a file or an in-memory carrier with the same deadline, body checks and request ceiling."""
    with _CheckedBody(source, deadline, upload.size_bytes, upload.sha256) as body:
        _within(
            deadline,
            functools.partial(_put, client, bucket, upload, body, deadline),
            expired=UPLOAD_DEADLINE_EXCEEDED,
            stalled=UPLOAD_STALLED,
            ceiling_s=_request_ceiling_s(upload.size_bytes),
        )


def _put(client: Any, bucket: str, upload: _Object, body: _CheckedBody, deadline: Deadline) -> None:
    """One PutObject of `body`. It counts only when the bytes the client read of the file are the
    recorded ones.

    What the body itself refused the client decides first, whatever the client returned or raised: a
    read after the deadline, then an error of the file itself, then a pass over the file that ended with
    other bytes than the recorded ones. An error of the client's own comes next, raised again as it is:
    a pass it had read only a part of says nothing about the file. An OSError among them is the
    client's too: the file's own is known by the body's note of it, never by its class. A request that
    returned is judged by its last pass. (The step
    for a read after the deadline is a belt: such a request also ended out of time or was abandoned at
    its ceiling, its outcome is dropped either way, and _within's own
    note of that already makes it the deadline failure.)
    """
    # The helper thread may have waited to run after _within checked the deadline.
    if deadline.remaining_s() <= 0:
        raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
    try:
        client.put_object(
            Bucket=bucket,
            Key=upload.key,
            Body=body,
            ContentLength=upload.size_bytes,
            ContentType=upload.content_type,
        )
    except Exception:
        if not body.cut_at_deadline and body.file_error is None and body.found_difference is None:
            raise
    if body.cut_at_deadline:
        raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
    if body.file_error is not None:
        raise _artifact_changed(upload, f"it could not be read: {body.file_error}")
    difference = body.found_difference or body.difference_of_the_last_pass()
    if difference is not None:
        raise _artifact_changed(upload, difference)


class _BodyCutAtDeadline(Exception):
    """What an upload's body raises to a storage client that still uses it once its request was cut: the
    deadline has passed, or the thread that waited for the request has closed the body."""


class _BodyNotAsRecorded(Exception):
    """What an upload's body raises to a storage client where a pass over the file ended with other
    bytes than the recorded ones, and at every read after that."""


class _BodyFileFailed(Exception):
    """What an upload's body raises to a storage client where the file itself failed under a question
    (a read error of the disk), and at every read after that: in our own words, never the
    operating system's, which may name a path."""


class _CheckedBody:
    """The body of one PutObject: the opened file, read through a count and a SHA-256 of its bytes.

    What a client reads here is what it sends, so a pass over the file tells what was, or would be,
    handed to the HTTP layer. (One exception: when the storage answers before the body was sent, the
    client drops what it reads, and the storage's answer alone decides.) A client may read a body more
    than once (botocore reads it through to sign the request, rewinds it with `seek(0)` and reads it
    again to send it), so a rewind to the start
    begins the count and the hash again: what follows it is a whole pass, whatever seek came before
    (botocore can also measure a body by its end first, and rewinds it then). A seek anywhere else
    leaves bytes that were read out of order, or never: until the next rewind the body no longer tells
    what was sent, and its last pass matches nothing. Only read() gives bytes (there is no fileno() and
    no iteration), and nothing is kept of them but the hash, however large the file is.

    A pass ends where a read that asked for bytes gets none from the file. A pass that ends with another
    size or SHA-256 than the recorded ones is refused there: that read raises, and so does every read
    after it. So with a client that reads the body through before it sends it (botocore does, to sign
    the payload), a file whose bytes are not the recorded ones is refused before a byte of it is sent. A
    file that changes between that pass and the sending pass is refused at the end of the sending pass:
    its bytes have gone out by then, under a signature that no longer fits them, which a storage that
    checks the signed payload hash refuses. Only with a client that reads the body once, or a storage
    that does not check that hash, can a wrong object have been stored by the time it is refused. A
    client that stops before the end of the file is judged by its last pass once its request returned.

    A question the file answers with an error of the operating system (a disk that fails under a read)
    is refused in our own words, and so is every read after it. The body notes that its own file
    failed, by the errno's name: whatever the client makes of the refusal (it raises it as it is while
    it signs, and wraps it in an error of its own while it sends), the upload ends as a file that could
    not be read, and no OSError of another origin is taken for it.

    Once the deadline has passed read() raises too: the request was abandoned by then (_within), and an
    abandoned upload must not go on sending. The thread that waited closes the body when it stops
    waiting, while the request may still run on its helper thread: whatever that thread then asks of the
    closed body is refused the same way, and nothing else comes of it.
    """

    def __init__(self, artifact: BinaryIO, deadline: Deadline, size_bytes: int, sha256: str) -> None:
        self._artifact = artifact
        self._deadline = deadline
        self._recorded_bytes = size_bytes
        self._recorded_sha256 = sha256
        self._pass_digest = hashlib.sha256()
        self._pass_bytes = 0
        self._moved_within_the_pass = False
        self._cut_at_deadline = False
        self._file_error: str | None = None
        self._found_difference: str | None = None
        self._closed = False

    @property
    def cut_at_deadline(self) -> bool:
        """Whether the client was refused because the deadline had passed, or the body was closed for it."""
        return self._cut_at_deadline

    @property
    def file_error(self) -> str | None:
        """The errno's name of the first error the body's own file answered a question with; None when
        it answered every question so far."""
        return self._file_error

    @property
    def found_difference(self) -> str | None:
        """What told the first pass that ended otherwise from the recorded bytes, in our own words; None
        when every pass that ended so far ended as recorded."""
        return self._found_difference

    def difference_of_the_last_pass(self) -> str | None:
        """What tells the bytes read since the last rewind from the recorded ones, in our own words; None
        when nothing does. It is the check of a request that returned: it also finds a client that
        stopped before the end of the file, or moved within it and did not rewind it."""
        if self._moved_within_the_pass:
            return "the storage client moved within it, so the bytes it sent are not known"
        return self._difference_of_the_pass()

    def _difference_of_the_pass(self) -> str | None:
        if self._pass_bytes != self._recorded_bytes:
            return f"{self._pass_bytes} bytes of it were read where the manifest records {self._recorded_bytes}"
        if self._pass_digest.hexdigest() != self._recorded_sha256:
            return "the bytes read of it do not have the SHA-256 the manifest records"
        return None

    def read(self, amount: int | None = -1) -> bytes:
        # The question _within asks of the deadline, so the body stops when its request is abandoned.
        if self._closed or self._deadline.remaining_s() <= 0:
            raise self._cut()
        if self._file_error is not None:
            raise self._file_failed()
        self._refuse_a_pass_that_ended_otherwise()
        wanted = -1 if amount is None else amount
        data = self._of_the_open_file(functools.partial(self._artifact.read, wanted))
        self._pass_digest.update(data)
        self._pass_bytes += len(data)
        if not data and wanted != 0:
            # The file gave no more bytes: the pass ends here, as recorded or refused.
            self._found_difference = self._difference_of_the_pass()
            self._refuse_a_pass_that_ended_otherwise()
        return data

    def _refuse_a_pass_that_ended_otherwise(self) -> None:
        if self._found_difference is not None:
            raise _BodyNotAsRecorded(self._found_difference)

    def _file_failed(self) -> _BodyFileFailed:
        """What is raised to the client that asks a body whose file failed, at the failure and after it."""
        return _BodyFileFailed(f"an upload's file could not be read: {self._file_error}")

    def _cut(self) -> _BodyCutAtDeadline:
        """Note that the request was cut, and give what is raised to the client that still uses the body."""
        self._cut_at_deadline = True
        return _BodyCutAtDeadline("an upload's body was asked for more after its request was cut at the deadline")

    def _of_the_open_file(self, ask: Callable[[], _Returned]) -> _Returned:
        """What `ask()` gets of the file. A body that is closed, before the question or under it, is the
        body of a request that was cut: the client is refused as it is after the deadline. An error of
        the operating system under the question is the file's own: it is noted, and the client is
        refused in our own words."""
        if self._closed:
            raise self._cut()
        try:
            return ask()
        except ValueError:
            # What a file raises that is closed under a read. Any other ValueError is the client's own.
            if not self._closed:
                raise
            raise self._cut() from None
        except OSError as error:
            if self._closed:
                raise self._cut() from None
            # Noted here, where the file raised it: nowhere else can it be told from another OSError.
            self._file_error = self._file_error or _errno_name(error)
        # Raised outside the handler: the operating system's error, whose text may name a path, does not
        # travel with what the client is given.
        raise self._file_failed()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        position = self._of_the_open_file(functools.partial(self._artifact.seek, offset, whence))
        if offset == 0 and whence == os.SEEK_SET:
            # A rewind: what follows is a whole pass again, whatever seek came before it.
            self._pass_digest = hashlib.sha256()
            self._pass_bytes = 0
            self._moved_within_the_pass = False
        else:
            self._moved_within_the_pass = True
        return position

    def tell(self) -> int:
        return self._of_the_open_file(self._artifact.tell)

    def readable(self) -> bool:
        return self._of_the_open_file(self._artifact.readable)

    def seekable(self) -> bool:
        return self._of_the_open_file(self._artifact.seekable)

    def close(self) -> None:
        # Marked before the file is closed, so that a question asked just as it is closed is known for what it is.
        self._closed = True
        self._artifact.close()

    def __enter__(self) -> _CheckedBody:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def _open_without_following_or_waiting(path: str, flags: int) -> int:
    return os.open(path, flags | ARTIFACT_OPEN_FLAGS)


def _opened_artifact(upload: _Upload) -> BinaryIO:
    """The file to upload, open for reading, once it is still a regular file of the recorded size.

    The open never waits: whatever stands at the path by now is opened at once and then judged, so a
    FIFO there is refused as any other file that is no longer a regular one. (Reading a regular file
    is the same with or without the flag.) A file that is refused is not left open, and its failure
    is raised once the handler that caught the operating system's error has ended: that error, whose
    text names the file's whole path, is neither the failure's cause nor its context."""
    changed: str | None
    try:
        body = open(upload.local_path, "rb", opener=_open_without_following_or_waiting)
    except OSError as error:
        changed = f"it could not be opened: {_errno_name(error)}"
    else:
        changed = _difference_from_the_record(body, upload)
        if changed is None:
            return body
        body.close()
    raise _artifact_changed(upload, changed)


def _difference_from_the_record(body: BinaryIO, upload: _Upload) -> str | None:
    """What tells the opened file from the regular file of the recorded size, in our own words; None
    when nothing does."""
    try:
        status = os.fstat(body.fileno())
    except OSError as error:
        return f"it could not be measured: {_errno_name(error)}"
    if not stat.S_ISREG(status.st_mode) or status.st_size != upload.size_bytes:
        return f"it is no longer the file of {upload.size_bytes} bytes the manifest records"
    return None


def _artifact_changed(upload: _Object, what: str) -> WorkerFailure:
    return WorkerFailure(
        Failure(
            "internal_error", "artifact_changed_before_upload", True,
            make_detail(f"{upload.relative_path} changed before its upload: {what}"),
        )
    )


def _upload_failed(storage_error_name: str) -> WorkerFailure:
    return WorkerFailure(
        Failure(
            "output_write_failed", "upload_failed", True,
            make_detail(f"an object could not be stored: {storage_error_name}"),
        )
    )


def _upload_refused(storage_error_name: str) -> WorkerFailure:
    return WorkerFailure(
        Failure(
            "output_write_failed", "upload_refused", False,
            make_detail(f"an object could not be stored: {storage_error_name}"),
        )
    )


def _refused_for_good(error: BotoCoreError | ClientError) -> bool:
    """Whether sending the same PutObject again cannot change how it ended: the client refused its
    parameters before it sent anything, or the storage answered with a redirect or a client error that
    is none of the passing ones. An answer of an unknown shape counts as passing: sending an object
    again for nothing costs less than giving up a job that could have ended well."""
    if isinstance(error, ParamValidationError):
        return True
    if not isinstance(error, ClientError):
        # No answer of the storage at all: a connection that failed or timed out, which passes.
        return False
    status = _answered_status(error)
    if status is None or not HTTP_FIRST_REDIRECT <= status <= HTTP_LAST_CLIENT_ERROR:
        return False
    return status not in PASSING_REFUSAL_STATUSES and _error_code(error) not in PASSING_REFUSAL_CODES


def _answered_status(error: ClientError) -> int | None:
    """The HTTP status of the storage's error answer; None when the answer names none that is a number."""
    metadata = error.response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def _error_code(error: ClientError) -> str | None:
    """The error code of the storage's error answer; None when the answer names none, or names one
    that is not wholly of ERROR_CODE_PATTERN."""
    code = error.response.get("Error", {}).get("Code")
    return code if isinstance(code, str) and ERROR_CODE_PATTERN.fullmatch(code) else None


def _storage_error_name(error: BotoCoreError | ClientError) -> str:
    """The error's class name and, for an answer of the storage, its error code: never the error's own
    text, which names the endpoint and the object."""
    name = type(error).__name__
    code = _error_code(error) if isinstance(error, ClientError) else None
    if code is not None:
        name = f"{name} {code}"
    return make_detail(name)
