"""Shared helpers of the job tests. Nothing here touches a network, a real bucket or a paid service.

ScriptedDeadline and ClockDeadline are deadlines a test decides, so that no sleep and no real clock does.
settle_requests_first() takes the host's speed out of a test that has the job wait for a storage request.
held_at_most_a_while() keeps a test that holds a scratch directory's lock from hanging the suite.
FakeS3 stands in for the S3 client: it serves ranged, conditional reads of objects it holds with the
response shape core.source_reader.S3RangeSource checks, and records every PutObject in order. A claimed
execution is built from the contract's own fixtures (the P01/P02 requests, the P24/P27 grants) through
the worker's real acceptance and claim code, so a test hands the runner exactly what handle() would.
The Hub's side of a test (the contexts a result, a progress message and a stored manifest are validated
under) is built here from the dispatch and the grant, in the shape of the catalog's ctx-hub-push.json
and ctx-hub-storage.json.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest
from botocore.exceptions import ClientError

from core.bounded_process import Deadline
from core.job import storage
from core.job.config import JobSettings
from core.protocol import files, pipeline, signing
from core.protocol.claim import Grant, claim_execution
from core.protocol.context import TrustedContext
from core.protocol.handler import ClaimedExecution, Runtime
from core.protocol.outbox import SealedMessage
from core.protocol.settings import WorkerSettings
from core.protocol.transport import HttpAnswer
from tests.contract.v2_support import (
    CELL_ID,
    GRANT_ENCRYPTED,
    GRANT_UNENCRYPTED,
    HUB_KEY,
    PROTOCOL_VERSION,
    REQUEST_ENCRYPTED,
    REQUEST_UNENCRYPTED,
    RUNTIME_ID,
    WORKER_KEY,
    WORKER_KEY_ID,
    FakeClock,
    FakeTransport,
    accepted_dispatch,
    modified_fixture,
    signed_answer,
    timestamp_text,
    uuid_source,
    worker_settings,
)
from tests.encode.conftest import require_media_tools
from tests.preflight.conftest import LANDSCAPE_SIZE, _h264_aac

SETTINGS = worker_settings()
IMAGE_DIGEST = "sha256:" + "5e" * 32
HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206
HTTP_NOT_FOUND = 404
HTTP_PRECONDITION_FAILED = 412
HTTP_RANGE_NOT_SATISFIABLE = 416
DELIVERED = HttpAnswer(HTTP_OK, b"", None)
CLOCK_SKEW_S = 300
# How long a storage stand-in that never answers holds a request before it lets it go by itself, so that
# a regression fails a test instead of hanging the suite; far longer than any deadline a test sets.
HANG_RELEASE_S = 20.0
PROGRESS_KIND = "transcode.progress"
COMPLETED_KIND = "transcode.result.completed"
FAILED_KIND = "transcode.result.failed"
MANIFEST_KIND = "generation.manifest"
MANIFEST_VERSION = "1.0.0-draft"
MANIFEST_NAME = "generation-manifest.json"
MILLISECONDS_PER_SECOND = 1000


# --- deadlines no real clock decides -----------------------------------------------------------------

# A scripted or clock-driven deadline answers expired() itself; this stands where a Deadline keeps its instant.
_NEVER_S = 1.0e12


class ScriptedDeadline(Deadline):
    """A Deadline whose expired() answers from a script, not from a clock: each call takes the next answer,
    and the last one repeats. `asked` counts the calls. It is a Deadline to every isinstance check.

    remaining_s() goes by the script as well, and takes no answer of it: no time is left once expired()
    has said that the deadline passed (before the first call: when the first answer says so), and until
    then more time than any wait lasts. So what decides whether a wait is made, or a request started,
    agrees with what expired() last said."""

    def __init__(self, *answers: bool) -> None:
        if not answers:
            raise ValueError("a scripted deadline needs at least one answer")
        super().__init__(_NEVER_S)
        object.__setattr__(self, "_answers", list(answers))
        object.__setattr__(self, "_passed", answers[0])
        object.__setattr__(self, "asked", 0)

    def expired(self) -> bool:
        object.__setattr__(self, "asked", self.asked + 1)
        object.__setattr__(self, "_passed", self._answers.pop(0) if len(self._answers) > 1 else self._answers[0])
        return self._passed

    def remaining_s(self) -> float:
        return 0.0 if self._passed else _NEVER_S


class ClockDeadline(Deadline):
    """A Deadline on the test's own clock: it has passed once `seconds` have gone by on `clock.monotonic()`,
    and remaining_s() is what is left of them on that clock.

    The job waits for a storage request's thread no longer than its deadline leaves, in real seconds
    (storage._within). A test that puts storage requests under this deadline has them settle first
    (settle_requests_first), so that what is left on its clock bounds no real wait."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        super().__init__(_NEVER_S)
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_passes_at_s", clock.monotonic() + seconds)

    def expired(self) -> bool:
        return self._clock.monotonic() >= self._passes_at_s

    def remaining_s(self) -> float:
        return max(0.0, self._passes_at_s - self._clock.monotonic())


# --- storage requests that settle before the job waits for them ------------------------------------

# How long a request may take to end, or to get to where a test keeps it: far longer than any takes, so
# that a regression fails a test instead of hanging it. And how often the thread that waits looks again.
SETTLES_WITHIN_S = 10.0
SETTLE_TICK_S = 0.005


def settle_requests_first(patched: pytest.MonkeyPatch, arrival: threading.Event | None = None) -> None:
    """Make whoever starts a storage request's thread get on only once that request has ended or, where
    a test names one, `arrival` is set (a request is where the test keeps it).

    The job waits for a request's thread no longer than its deadline or its ceiling leaves, in real
    seconds (storage._within). Under this it begins to wait when the request needs no more time to get
    somewhere, so nothing a test's deadline leaves is a time a loaded host could use up first: neither
    the tens of milliseconds of a deadline or a ceiling that is to pass under a kept request, nor what
    is left on a test's own clock (ClockDeadline)."""
    start = threading.Thread.start

    def start_and_see_the_request_settle(thread: threading.Thread) -> None:
        start(thread)
        if thread.name != storage.REQUEST_THREAD_NAME:
            return
        give_up_at_s = time.monotonic() + SETTLES_WITHIN_S
        while thread.is_alive() and not (arrival is not None and arrival.is_set()):
            assert time.monotonic() < give_up_at_s, "a storage request neither ended nor got to where the test keeps it"
            thread.join(SETTLE_TICK_S)

    patched.setattr(threading.Thread, "start", start_and_see_the_request_settle)


# --- a lock a test holds, which no code under test may wait for --------------------------------------

# How long a test that holds a scratch directory's lock, as a live execution of another process would,
# keeps it before it lets go by itself. The code under test never waits for a lock. Were it ever to, on
# the test's own thread, it would wait for ever, and nothing times a test out: with this it gets on
# after a few seconds, and the test fails on what it did with a lock it should never have got.
LOCK_RELEASE_S = 5.0


def held_at_most_a_while(descriptor: int) -> Callable[[], None]:
    """Close `descriptor`, on which a test holds a lock, by itself after LOCK_RELEASE_S, and return the
    function that closes it at once if that has not happened. Either way it is closed exactly once."""
    closed_by_itself: list[bool] = []

    def close_by_itself() -> None:
        os.close(descriptor)
        closed_by_itself.append(True)

    timer = threading.Timer(LOCK_RELEASE_S, close_by_itself)
    timer.daemon = True
    timer.start()

    def let_go() -> None:
        # Once the timer's thread has ended it has closed the descriptor, or it never will.
        timer.cancel()
        timer.join()
        if not closed_by_itself:
            os.close(descriptor)

    return let_go


# --- a fake S3 client ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PutCall:
    """One PutObject the code under test made; `body` is what the call read of the body it was given."""

    bucket: str
    key: str
    body: bytes
    content_length: int
    content_type: str
    failed: bool


class _Body:
    """A response body with the two methods S3RangeSource uses."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self, amount: int | None = None) -> bytes:
        return self._data if amount is None else self._data[:amount]

    def close(self) -> None:
        self.closed = True


def client_error(code: str, status: int, operation: str, message: str = "refused") -> ClientError:
    """The error botocore raises for an error answer of the storage."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}}, operation
    )


class FakeS3:
    """An S3 client that holds its objects in memory or reads them from local files.

    get_object() answers a ranged read as a real storage does: 206 with the ETag, the Content-Range and the
    Content-Length of the range; `PreconditionFailed` (412) when If-Match is not the object's ETag;
    `NoSuchKey` (404) for an object it does not hold. put_object() reads the body it is given and records
    the call. fail_get() and fail_put() make one numbered call (the first is 1), or every call from that
    number on, raise instead; a failing put reads half of the body first, as a broken connection would.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], tuple[bytes | Path, str]] = {}
        self._get_failures: dict[int, Exception] = {}
        self._put_failures: dict[int, Exception] = {}
        self._gets_fail_from: tuple[int, Exception] | None = None
        self._puts_fail_from: tuple[int, Exception] | None = None
        self.get_calls: list[dict[str, str]] = []
        self.put_calls: list[PutCall] = []

    def hold(self, bucket: str, key: str, data: bytes | Path, etag: str) -> None:
        """Hold an object: its bytes, or the local file to read them from."""
        self._objects[(bucket, key)] = (data, etag)

    def fail_get(self, call_number: int, error: Exception, *, and_every_later_one: bool = False) -> None:
        if and_every_later_one:
            self._gets_fail_from = (call_number, error)
        else:
            self._get_failures[call_number] = error

    def fail_put(self, call_number: int, error: Exception, *, and_every_later_one: bool = False) -> None:
        if and_every_later_one:
            self._puts_fail_from = (call_number, error)
        else:
            self._put_failures[call_number] = error

    @property
    def request_count(self) -> int:
        """Every request the code under test made of the storage, reads and writes."""
        return len(self.get_calls) + len(self.put_calls)

    @property
    def stored(self) -> list[PutCall]:
        """The PutObject calls that succeeded, in order."""
        return [call for call in self.put_calls if not call.failed]

    def get_object(self, *, Bucket: str, Key: str, Range: str, IfMatch: str) -> dict[str, Any]:  # noqa: N803
        self.get_calls.append({"Bucket": Bucket, "Key": Key, "Range": Range, "IfMatch": IfMatch})
        failure = self._failure_of(len(self.get_calls), self._get_failures, self._gets_fail_from)
        if failure is not None:
            raise failure
        held = self._objects.get((Bucket, Key))
        if held is None:
            raise client_error("NoSuchKey", HTTP_NOT_FOUND, "GetObject")
        data, etag = held
        if IfMatch != f'"{etag}"':
            raise client_error("PreconditionFailed", HTTP_PRECONDITION_FAILED, "GetObject")
        content = data.read_bytes() if isinstance(data, Path) else data
        first, last = Range.removeprefix("bytes=").split("-")
        start, end = int(first), min(int(last), len(content) - 1)
        if start >= len(content):
            raise client_error("InvalidRange", HTTP_RANGE_NOT_SATISFIABLE, "GetObject")
        chunk = content[start : end + 1]
        return {
            "ResponseMetadata": {"HTTPStatusCode": HTTP_PARTIAL_CONTENT},
            "ETag": f'"{etag}"',
            "ContentRange": f"bytes {start}-{end}/{len(content)}",
            "ContentLength": len(chunk),
            "Body": _Body(chunk),
        }

    def put_object(  # noqa: N803
        self, *, Bucket: str, Key: str, Body: Any, ContentLength: int, ContentType: str
    ) -> dict[str, Any]:
        failure = self._failure_of(len(self.put_calls) + 1, self._put_failures, self._puts_fail_from)
        sent = Body.read() if failure is None else Body.read(ContentLength // 2)
        self.put_calls.append(PutCall(Bucket, Key, sent, ContentLength, ContentType, failed=failure is not None))
        if failure is not None:
            raise failure
        return {"ResponseMetadata": {"HTTPStatusCode": HTTP_OK}}

    @staticmethod
    def _failure_of(
        call_number: int, failures: Mapping[int, Exception], fail_from: tuple[int, Exception] | None
    ) -> Exception | None:
        if call_number in failures:
            return failures[call_number]
        if fail_from is not None and call_number >= fail_from[0]:
            return fail_from[1]
        return None


# --- a claimed execution, as handle() hands it to the runner ---------------------------------------


def claimed_execution(
    *, encrypted: bool = True, request_changes: Mapping[str, Any] | None = None, settings: WorkerSettings = SETTINGS
) -> ClaimedExecution:
    """The fixtures' dispatch (P01, or P02 unencrypted) with `request_changes` applied (JSON pointer to
    value), accepted by the worker's own code, and the fixtures' grant for it (P24 or P27), claimed by it.

    Its start began where every FakeClock begins and its claim took no time: a runtime on a fresh
    FakeClock has the dispatch's whole wall time left."""
    request_fixture, grant_fixture = (
        (REQUEST_ENCRYPTED, GRANT_ENCRYPTED) if encrypted else (REQUEST_UNENCRYPTED, GRANT_UNENCRYPTED)
    )
    dispatch = accepted_dispatch(modified_fixture(request_fixture, request_changes or {}), settings=settings)
    clock = FakeClock()
    wall_time_ends_at = clock.monotonic() + dispatch.limits.max_wall_time_ms / MILLISECONDS_PER_SECOND
    grant = claim_execution(
        dispatch,
        settings=settings,
        runtime_id=RUNTIME_ID,
        transport=FakeTransport(signed_answer(files.read_bytes(grant_fixture), HUB_KEY)),
        clock=clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        new_message_id=uuid_source(),
        wall_time_ends_at=wall_time_ends_at,
    )
    if not isinstance(grant, Grant):
        raise AssertionError("the fixtures' grant was not accepted for the changed dispatch")
    return ClaimedExecution(
        dispatch=dispatch, grant=grant, runtime_id=RUNTIME_ID, wall_time_ends_at=wall_time_ends_at
    )


def runtime_of(transport: FakeTransport, clock: FakeClock, settings: WorkerSettings = SETTINGS) -> Runtime:
    return Runtime(settings=settings, transport=transport, clock=clock, monotonic=clock.monotonic, sleep=clock.sleep, new_uuid=uuid_source())


def job_settings(scratch_root: Path) -> JobSettings:
    return JobSettings(image_digest=IMAGE_DIGEST, scratch_root=scratch_root)


# --- the Hub's side: what it validates a worker's message and manifest under ------------------------


def _recorded_expectations(claimed: ClaimedExecution) -> dict[str, Any]:
    dispatch, grant = claimed.dispatch, claimed.grant
    return {
        **grant.identity.to_dict(),
        "source_id": dispatch.source.source_id,
        "source_location_id": dispatch.source.location_id,
        "output_location_id": dispatch.output.location_id,
        "profile": dispatch.profile.to_dict(),
        "encryption": dispatch.encryption.to_dict(),
        "renditions": list(dispatch.renditions),
        "max_artifact_count": dispatch.limits.max_artifact_count,
    }


def hub_push_context(claimed: ClaimedExecution, *, now: datetime, last_event_seq: int = 0) -> TrustedContext:
    """The hub_receiver context of a pushed progress message or result of this execution."""
    return TrustedContext.from_dict(
        {
            "context_version": 1,
            "role": "hub_receiver",
            "channel": "push",
            "now": timestamp_text(now),
            "clock_skew_s": CLOCK_SKEW_S,
            "accepted_protocol_versions": [PROTOCOL_VERSION],
            "accepted_message_kinds": [PROGRESS_KIND, COMPLETED_KIND, FAILED_KIND],
            "expected_audience": f"hub:{CELL_ID}",
            "known_key_ids": [WORKER_KEY_ID],
            "expect": _recorded_expectations(claimed),
            "last_event_seq": last_event_seq,
        }
    )


def hub_storage_context(claimed: ClaimedExecution) -> TrustedContext:
    """The hub_receiver context of the generation manifest this execution stores."""
    expect = _recorded_expectations(claimed)
    manifest_fields = (
        "org_uuid", "video_uuid", "attempt_id", "dispatch_id", "execution_id", "source_id",
        "profile", "encryption", "renditions", "max_artifact_count",
    )
    return TrustedContext.from_dict(
        {
            "context_version": 1,
            "role": "hub_receiver",
            "channel": "storage",
            "accepted_manifest_versions": [MANIFEST_VERSION],
            "accepted_message_kinds": [MANIFEST_KIND],
            "expect": {name: expect[name] for name in manifest_fields},
        }
    )


def accepted_by_the_hub(
    sealed: SealedMessage, claimed: ClaimedExecution, *, now: datetime, last_event_seq: int = 0
) -> dict[str, Any]:
    """The decoded message, once its signature verified under the worker-to-Hub key and the Hub's
    validator accepted its exact bytes; an AssertionError says which of the two refused it."""
    return accepted_bytes_by_the_hub(sealed.body, sealed.signature, claimed, now=now, last_event_seq=last_event_seq)


def accepted_bytes_by_the_hub(
    body: bytes, signature: str, claimed: ClaimedExecution, *, now: datetime, last_event_seq: int = 0
) -> dict[str, Any]:
    assert signing.verify(body, WORKER_KEY, signature), "the signature does not verify under the worker-to-Hub key"
    verdict = pipeline.validate(body, hub_push_context(claimed, now=now, last_event_seq=last_event_seq))
    assert verdict.accepted, verdict
    return json.loads(body)


# --- media for the end-to-end tests ---------------------------------------------------------------


@pytest.fixture(scope="session")
def source_media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 320x240 H.264 + AAC MP4 of 2 s, generated by the local ffmpeg; skipped without ffmpeg and ffprobe."""
    require_media_tools()
    return _h264_aac(tmp_path_factory.mktemp("job-media") / "landscape.mp4", LANDSCAPE_SIZE)
