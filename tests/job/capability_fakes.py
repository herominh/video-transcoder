"""Shared helpers of the protocol 2.2.0 tests: a fake Hub of the capability exchange and a fake store behind the
shared session. Nothing here opens a connection.

Not a test module. 2.2.0 is a draft, so no environment loads it: capability_settings() builds the worker's settings
with it listed, as a test-built context may (README section 5). FakeHub stands behind the client's transport: it
judges every capability request with the contract's own validator under a `hub_receiver` context and the signature
under the worker-to-Hub key, records it, and answers what the test scripts, by default a signed
`execution.capabilities.granted` with URLs of the fake store's host, built here; every other POST (a progress
report, a callback) is taken. FakeStore stands behind the shared session's get() and put(): it serves the source by
ranges, conditional on its ETag, through the URLs the Hub issued for it, takes a PUT only through a URL issued for
its key with the declared Content-MD5 and a body of that MD5, records each object it stores and the most PUTs it
saw at once, and answers what a test scripts per path.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit

from core.failure import Failure
from core.net.clock import StopLine
from core.protocol import messages, pipeline, signing
from core.protocol.context import TrustedContext
from core.protocol.handler import ClaimedExecution
from core.protocol.outbox import SealedMessage, seal_message
from core.protocol.registry import MessageKind
from core.protocol.settings import WorkerSettings
from core.protocol.transport import HttpAnswer
from tests.contract.v2_support import (
    CELL_ID,
    HUB_KEY,
    HUB_KEY_ID,
    WORKER_AUDIENCE,
    WORKER_KEY,
    WORKER_KEY_ID,
    FIXTURE_NOW,
    FakeClock,
    compact,
    timestamp_text,
    worker_settings,
)
from tests.drive_import.fakes import FakeResponse
from tests.job.conftest import DELIVERED, claimed_execution

CAPABILITY_VERSION = "2.2.0"
CAPABILITIES_URL = "https://hub.test.invalid/api/transcode/v2/capabilities"
STORE_HOST = "store.test.invalid"
SOURCE_BUCKET = "test-source-bucket"
SOURCE_KEY = "test/sources/source-0001.mp4"
OUTPUT_BUCKET = "test-output-bucket"
SOURCE_ETAG = "test-etag-0001"
REQUEST_KIND = "execution.capabilities.request"
GRANTED_KIND = "execution.capabilities.granted"
# What a URL's signature holds: a canary no log line, error or result may show.
SIGNATURE_CANARY = "CANARYsig"
LEASE_S = 900
STAGE_S = 600
SOURCE_URL_S = 3600
BODY_BLOCK_BYTES = 16_384
HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_PRECONDITION_FAILED = 412


def capability_settings(**changes: Any) -> WorkerSettings:
    """The worker's settings of the fixtures, accepting 2.0.0 and the 2.2.0 draft, with the capability address and the
    storage host, and other fields replaced where given."""
    values = {
        "accepted_protocol_versions": ("2.0.0", CAPABILITY_VERSION),
        "capabilities_url": CAPABILITIES_URL,
        "storage_host": STORE_HOST,
        **changes,
    }
    return dataclasses.replace(worker_settings(), **values)


def capability_claimed(*, request_changes: Mapping[str, Any] | None = None, encrypted: bool = False) -> ClaimedExecution:
    """The fixtures' dispatch at 2.2.0 (P02 unencrypted unless asked), accepted, and its grant (P27 or P24)."""
    changes = {"/protocol_version": CAPABILITY_VERSION, **(request_changes or {})}
    return claimed_execution(encrypted=encrypted, request_changes=changes, settings=capability_settings())


def failed_result_of(claimed: ClaimedExecution) -> SealedMessage:
    """A sealed `transcode.result.failed` of the execution, as the runner returns one."""
    settings = capability_settings()
    message = messages.result_failed(
        protocol_version=claimed.dispatch.protocol_version, message_id="99999999-9999-7999-8999-0000000000f1",
        sent_at=FIXTURE_NOW, key_id=settings.worker_to_hub_key_id, identity=claimed.grant.identity,
        failure=Failure("encoder_failed", "test_failure", True, "a failure of the test"), stage="probing",
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_RESULT_FAILED, claimed.grant.identity, now=FIXTURE_NOW,
        settings=dataclasses.replace(settings, accepted_protocol_versions=(claimed.dispatch.protocol_version,)),
        source_id=claimed.dispatch.source.source_id,
    )
    return seal_message(message, context, settings=settings)


def _whole_second(moment: datetime) -> datetime:
    return moment.replace(microsecond=0)


def _amz_date(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def md5_of(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


@dataclass(frozen=True)
class Issued:
    """One URL the fake Hub signed: the method it allows, its bucket and key, its MD5 (a PUT's) and its expiry."""

    method: str
    bucket: str
    key: str
    md5: str | None
    expires_at: datetime


class FakeHub:
    """The capability route, and a callback address that takes every POST. `script` holds what the next capability
    requests get, in order: "grant" (the default once it is empty), an HttpAnswer, an exception the transport raises,
    or a callable of (request document, the default answer's bytes builder) returning an HttpAnswer. Every request
    must pass the Hub's validator (an AssertionError otherwise, recorded in `refused`)."""

    def __init__(self, claimed: ClaimedExecution, clock: FakeClock, *, lease_s: int = LEASE_S) -> None:
        self.claimed = claimed
        self.clock = clock
        self.lease_s = lease_s
        self.script: deque[Any] = deque()
        self.requests: list[dict[str, Any]] = []
        self.sent_at_monotonic: list[float] = []
        self.issued: dict[str, Issued] = {}
        self.declared: dict[str, tuple[int, str]] = {}
        self.callbacks: list[bytes] = []
        self.refused: list[str] = []
        self.timeouts: list[float] = []
        self._last_seq = 0
        self._issued_count = 0
        self._lock = threading.Lock()

    # --- the transport -----------------------------------------------------------------------------------

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        if url != CAPABILITIES_URL:
            self.callbacks.append(body)
            return DELIVERED
        with self._lock:
            self.timeouts.append(timeout_s)
            self.sent_at_monotonic.append(self.clock.monotonic())
            request = self._judged(body, signature)
            scripted = self.script.popleft() if self.script else "grant"
        if isinstance(scripted, Exception):
            raise type(scripted)(*scripted.args)
        if isinstance(scripted, HttpAnswer):
            return scripted
        if callable(scripted):
            return scripted(request, self)
        return self.grant(request)

    def _judged(self, body: bytes, signature: str) -> dict[str, Any]:
        if not signing.verify(body, WORKER_KEY, signature):
            self.refused.append("signature")
            raise AssertionError("a capability request does not verify under the worker-to-Hub key")
        verdict = pipeline.validate(body, self.receiver_context())
        if not verdict.accepted:
            self.refused.append(str(verdict.reason))
            raise AssertionError(f"the Hub refuses a capability request: {verdict}")
        request = json.loads(body)
        self._last_seq = request["seq"]
        self.requests.append(request)
        for item in (request["put"] or {}).get("objects", ()):
            declared = (item["size_bytes"], item["md5"])
            assert self.declared.setdefault(item["path"], declared) == declared, "a path was declared again otherwise"
        return request

    def receiver_context(self) -> TrustedContext:
        claimed = self.claimed
        return TrustedContext.from_dict({
            "context_version": 1,
            "role": "hub_receiver",
            "channel": "push",
            "accepted_message_kinds": [REQUEST_KIND],
            "now": timestamp_text(self.clock()),
            "clock_skew_s": 300,
            "accepted_protocol_versions": [CAPABILITY_VERSION],
            "expected_audience": f"hub:{CELL_ID}",
            "known_key_ids": [WORKER_KEY_ID],
            "expect": {**claimed.grant.identity.to_dict(), "runtime_id": claimed.runtime_id},
            "last_capability_seq": self._last_seq,
        })

    # --- the answers ---------------------------------------------------------------------------------------

    def granted_document(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """The answer the Hub signs for `request`: the lease, the stage, and the URLs it asked for."""
        now = self.clock()
        put = request["put"]
        document: dict[str, Any] = {
            "protocol_version": CAPABILITY_VERSION,
            "message_kind": GRANTED_KIND,
            "message_id": "99999999-9999-7999-8999-%012d" % (len(self.requests)),
            "sent_at": timestamp_text(now),
            "audience": WORKER_AUDIENCE,
            "key_id": HUB_KEY_ID,
            "identity": request["identity"],
            "runtime_id": request["runtime_id"],
            "seq": request["seq"],
            "lease": {"expires_at": timestamp_text(now + timedelta(seconds=self.lease_s))},
            "stage": {"name": request["stage"], "ends_at": timestamp_text(now + timedelta(seconds=STAGE_S))},
            "source": None,
            "put": None,
        }
        if request["source"]:
            expires = _whole_second(now) + timedelta(seconds=SOURCE_URL_S)
            document["source"] = {
                "expires_at": timestamp_text(expires),
                "url": self.signed_url("GET", SOURCE_BUCKET, SOURCE_KEY, None, now, expires),
            }
        if put is not None:
            expires = _whole_second(now) + timedelta(seconds=put["lifetime_seconds"])
            prefix = self.claimed.grant.output_prefix
            document["put"] = {
                "expires_at": timestamp_text(expires),
                "objects": [
                    {"path": item["path"],
                     "url": self.signed_url("PUT", OUTPUT_BUCKET, f"{prefix}/{item['path']}", item["md5"], now, expires)}
                    for item in put["objects"]
                ],
            }
        return document

    def grant(self, request: Mapping[str, Any], *, document: Mapping[str, Any] | None = None, key: bytes = HUB_KEY) -> HttpAnswer:
        body = compact(self.granted_document(request) if document is None else document)
        return HttpAnswer(status=HTTP_OK, body=body, signature=signing.sign(body, key))

    def signed_url(self, method: str, bucket: str, key: str, md5: str | None, now: datetime, expires: datetime) -> str:
        self._issued_count += 1
        date = _whole_second(now)
        seconds = int((expires - date).total_seconds())
        headers = "host" if method == "GET" else "content-md5%3Bhost"
        url = (
            f"https://{STORE_HOST}/{bucket}/{key}?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            f"&X-Amz-Credential={quote('test/' + date.strftime('%Y%m%d') + '/auto/s3/aws4_request', safe='')}"
            f"&X-Amz-Date={_amz_date(date)}&X-Amz-Expires={seconds}&X-Amz-SignedHeaders={headers}"
            f"&X-Amz-Signature={SIGNATURE_CANARY}{self._issued_count:06d}"
        )
        self.issued[url] = Issued(method, bucket, key, md5, expires)
        return url

    def urls_given(self) -> list[str]:
        return list(self.issued)


def hub_error(status: int, code: str, *, retryable: bool, retry_after_s: int | None = None) -> HttpAnswer:
    error: dict[str, Any] = {"code": code, "message": "The Hub refuses this request.", "retryable": retryable}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    body = compact({
        "protocol_version": "2.0.0",
        "message_kind": "hub.error",
        "message_id": "99999999-9999-7999-8999-0000000000e1",
        "sent_at": "2030-01-01T00:00:00.000Z",
        "error": error,
    })
    return HttpAnswer(status=status, body=body, signature=None)


# --- the store ----------------------------------------------------------------------------------------------


@dataclass
class StoredPut:
    """One PUT the store received: the object's path under the prefix, its headers, and what came of it."""

    path: str
    headers: dict[str, str]
    length: int
    md5: str | None
    status: int
    timeout: tuple[float, float]
    stop_line: StopLine = field(repr=False)


class FakeStore:
    """The store behind the shared session (get and put). `source` is the source object's bytes. `put_answers` maps a
    path under the output prefix to what its next PUTs get (a status, an exception raised, or a callable run before
    the default answer); `get_answers` lists what the next GETs get. `hold_until_active`, when set, holds each PUT
    until that many are under way at once or `hold_s` passed, so that a test sees how many can run together."""

    def __init__(self, hub: FakeHub, source: bytes = b"", *, etag: str = SOURCE_ETAG) -> None:
        self.hub = hub
        self.source = source
        self.etag = etag
        self.objects: dict[str, bytes] = {}
        self.stored_order: list[str] = []
        self.puts: list[StoredPut] = []
        self.gets: list[dict[str, Any]] = []
        self.put_answers: dict[str, list[Any]] = {}
        self.get_answers: list[Any] = []
        self.hold_until_active: int | None = None
        self.hold_s = 0.5
        self.peak = 0
        self.closed = False
        self._active = 0
        self._lock = threading.Condition()

    @property
    def prefix(self) -> str:
        return self.hub.claimed.grant.output_prefix

    def _issued(self, url: str, method: str) -> Issued | None:
        issued = self.hub.issued.get(url)
        if issued is None or issued.method != method:
            return None
        if self.hub.clock() >= issued.expires_at:
            return None
        return issued

    def get(self, url: str, headers: Mapping[str, str], timeout: tuple[float, float], stream: bool, *,
            stop_line: StopLine) -> FakeResponse:
        self.gets.append({"url": url, "headers": dict(headers), "timeout": timeout})
        if self.get_answers:
            scripted = self.get_answers.pop(0)
            if isinstance(scripted, BaseException):
                raise scripted
            if isinstance(scripted, FakeResponse):
                return scripted
            if isinstance(scripted, int):
                return FakeResponse(scripted)
        issued = self._issued(url, "GET")
        if issued is None or (issued.bucket, issued.key) != (SOURCE_BUCKET, SOURCE_KEY):
            return FakeResponse(HTTP_FORBIDDEN)
        if headers.get("If-Match") != f'"{self.etag}"':
            return FakeResponse(HTTP_PRECONDITION_FAILED)
        first, last = headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(first), int(last)
        chunk = self.source[start:end + 1]
        return FakeResponse(HTTP_PARTIAL_CONTENT, {
            "ETag": f'"{self.etag}"',
            "Content-Range": f"bytes {start}-{end}/{len(self.source)}",
            "Content-Length": str(len(chunk)),
        }, chunk)

    def put(self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *,
            stop_line: StopLine) -> FakeResponse:
        path = urlsplit(url).path.removeprefix(f"/{OUTPUT_BUCKET}/{self.prefix}/")
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self._lock.notify_all()
            if self.hold_until_active is not None:
                self._lock.wait_for(lambda: self._active >= self.hold_until_active, timeout=self.hold_s)
        try:
            return self._put(url, path, data, headers, timeout, stop_line)
        finally:
            with self._lock:
                self._active -= 1

    def _put(self, url: str, path: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float],
             stop_line: StopLine) -> FakeResponse:
        scripted_list = self.put_answers.get(path, [])
        scripted = scripted_list.pop(0) if scripted_list else None
        if isinstance(scripted, BaseException):
            raise scripted
        if callable(scripted):
            scripted()
            scripted = None
        body = bytearray()
        while True:
            block = data.read(BODY_BLOCK_BYTES)
            if not block:
                break
            body += block
        md5 = md5_of(bytes(body))
        status = scripted if isinstance(scripted, int) else self._status_of(url, headers, md5, len(body))
        self.puts.append(StoredPut(path, dict(headers), len(body), md5, status, timeout, stop_line))
        if status == HTTP_OK:
            with self._lock:
                self.objects[path] = bytes(body)
                self.stored_order.append(path)
        return FakeResponse(status)

    def _status_of(self, url: str, headers: Mapping[str, str], md5: str, length: int) -> int:
        issued = self._issued(url, "PUT")
        if issued is None or headers.get("Content-MD5") != issued.md5:
            return HTTP_FORBIDDEN
        if md5 != headers.get("Content-MD5") or str(length) != headers.get("Content-Length"):
            return 400
        return HTTP_OK

    def close(self) -> None:
        self.closed = True


def wait_until(condition: Callable[[], bool], *, within_s: float = 5.0) -> bool:
    """Whether `condition()` became true within `within_s` real seconds, looked at every few milliseconds."""
    give_up_at = time.monotonic() + within_s
    while time.monotonic() < give_up_at:
        if condition():
            return True
        time.sleep(0.005)
    return condition()
