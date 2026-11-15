"""How an execution at protocol 2.2.0 stores its generation: through the put URLs of the capability exchange, several
objects at a time (README 15.4; the Hub's register row TD-190 for the upload one at a time).

upload_generation_by_urls() stores exactly what core/job/storage.py upload_generation() stores: the files the
manifest lists, each under the granted prefix, then the manifest last, as the completion marker; and it refuses,
before any request, what that refuses (a path that is no contract rel_path, a file reached through a link, artifacts
that are not as many, or not of the total size, the manifest records), and an object a single PUT of the exchange
cannot carry (above 64 MiB).

The artifacts go in batches of at most BATCH_MAX_OBJECTS, in the manifest's order. For a batch, each file is read
once, its size and SHA-256 checked against the manifest and its MD5 computed (a file that changed is
internal_error/artifact_changed_before_upload, and nothing of the batch is sent); then the batch's URLs are asked
for (lifetime BATCH_URL_LIFETIME_S), declaring each object's path, size and MD5, which a path keeps whenever it is
asked again (rule 4); then the objects are PUT, at most UPLOAD_PARALLELISM at once, never more bodies than that in
memory. One PUT reads the file into memory and checks it again against the declared MD5 (what is sent is what was
declared), and carries `Content-MD5`, `Content-Length` and the object's content type, through the shared session
(core/net/transport.py) with a BoundedBody, a 10 s connect, the request's ceiling of core/job/storage.py (its body
cut there, its answer waited for no longer) and the execution's deadline as its stop line. A 200 is success; the
answer's body is never read. A 5xx, a 408, a 429, a time-out, a connection error or a request that outlasted its
ceiling is tried again after storage.py's waits (0.5, 2 and 8 s, each times a factor drawn between 0.5 and 1, a
whole-seconds `Retry-After` honoured up to 30 s); a 403, or a URL whose usable time (less URL_MARGIN_S) has passed
before its PUT could start, puts the object back to be asked again in a later batch, at most MAX_REASKS further
times, then output_write_failed/upload_url_refused (retryable); any other answer is output_write_failed/
upload_refused, final.

The first failure stops the rest: no new PUT starts, the PUTs under way are waited for, and that failure is the
upload's. A lost authority (the capability client's lost()) is such a failure. No write starts after the execution's
deadline, and the session's watchdog shuts every socket at it, so a PUT abandoned there is closed, not left open
(TD-345 for this path); one that still has not ended DRAIN_AFTER_DEADLINE_S later is given up and counted with
storage.abandoned_requests(), so the entry point asks for a fresh worker. Once every artifact is stored, the manifest
alone is asked for (lifetime LAST_URL_LIFETIME_S) and PUT.

upload_result_by_url() stores the terminal result's carrier at `result.json` through a URL of its own (lifetime
LAST_URL_LIFETIME_S), PUT once, best effort: it never replaces or delays the result (15.1 rule 9), and a runtime
whose authority is lost asks for no new URL (rule 6), so a result built then is not stored.

No URL, MD5, request or answer reaches a log line or a failure; a typed failure is raised once the handler that
caught the session's error (which quotes the whole URL) has ended.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import Failure, WorkerFailure
from core.net.clock import DeadlineReached, stop_line_within
from core.net.transport import TRANSFER_ERRORS, BoundedBody, HttpResponse, HttpSession, close_quietly
from core.package import ManifestOutput
from core.protocol.capabilities_client import CapabilityClient, PutUrl
from core.protocol.messages import CapabilityObject, CapabilityPut
from core.protocol.outbox import SealedMessage, runpod_output
from core.protocol.registry import MessageKind

from .storage import (
    CONTENT_TYPES,
    RESULT_NAME,
    RETRY_AFTER_MAX_S,
    UPLOAD_DEADLINE_EXCEEDED,
    UPLOAD_RETRY_WAITS_S,
    UPLOAD_STALLED,
    _artifact_changed,
    _count_abandoned_request,
    _errno_name,
    _next_wait_s,
    _notify,
    _Object,
    _opened_artifact,
    _planned_uploads,
    _request_ceiling_s,
    _require_transfer_arguments,
    _require_what_the_manifest_records,
    _Upload,
    _upload_failed,
    _upload_refused,
)

logger = logging.getLogger(__name__)

STAGE_UPLOADING = "uploading"
UPLOAD_PARALLELISM = 12
BATCH_MAX_OBJECTS = 128
BATCH_URL_LIFETIME_S = 180
LAST_URL_LIFETIME_S = 60
# A URL is not used for a PUT that starts later than this before its usable time ends.
URL_MARGIN_S = 5.0
MAX_REASKS = 3
CONNECT_TIMEOUT_S = 10.0
# The largest object one PUT of the exchange may carry (capabilities-common.schema.json object_size).
MAX_OBJECT_BYTES = 67_108_864
# How long the upload still waits for its PUTs once the deadline has passed (the watchdog shuts them at it).
DRAIN_AFTER_DEADLINE_S = 10.0
# How often the thread that waits for a batch looks at the authority and the deadline.
BATCH_POLL_S = 0.05
HASH_CHUNK_BYTES = 1_048_576
HTTP_OK = 200
HTTP_FORBIDDEN = 403
HTTP_REQUEST_TIMEOUT = 408
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
RETRY_AFTER_HEADER = "Retry-After"
POOL_THREAD_PREFIX = "vh-url-upload"

UPLOAD_URL_REFUSED = Failure(
    "output_write_failed", "upload_url_refused", True,
    "the storage refused the put URLs the Hub granted for an object, asked again each time",
)
OBJECT_TOO_LARGE = Failure(
    "output_write_failed", "object_too_large", False,
    "an object of the generation is larger than one put URL may carry",
)


@dataclass
class _Item:
    """One object on its way: what it is, the base64 MD5 it is declared with, and how often its URL was refused."""

    upload: _Object
    md5: str
    data: bytes | None = None
    refusals: int = 0

    @property
    def declared(self) -> CapabilityObject:
        return CapabilityObject(path=self.upload.relative_path, size_bytes=self.upload.size_bytes, md5=self.md5)


class _Reask:
    """The object's URL could not be used: refused by the store, or past its usable time before the PUT started."""


class _Skipped:
    """The upload had stopped before the object's PUT started."""


_REASK = _Reask()
_SKIPPED = _Skipped()


def _md5_of(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode("ascii")


def _prepared(upload: _Upload) -> _Item:
    """The object read once: its size and SHA-256 must be the manifest's; its MD5 is computed in the same pass."""
    sha256, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    changed: str | None = None
    with _opened_artifact(upload) as body:
        try:
            for chunk in iter(lambda: body.read(HASH_CHUNK_BYTES), b""):
                sha256.update(chunk)
                md5.update(chunk)
                size += len(chunk)
        except OSError as error:
            changed = f"it could not be read: {_errno_name(error)}"
    if changed is None and (size != upload.size_bytes or sha256.hexdigest() != upload.sha256):
        changed = "its bytes are not the ones the manifest records"
    if changed is not None:
        raise _artifact_changed(upload, changed)
    return _Item(upload=upload, md5=base64.b64encode(md5.digest()).decode("ascii"))


def _bytes_of(item: _Item) -> bytes:
    """The bytes one PUT sends: the object's own (a carrier in memory), or its file read now, which must still be
    of the declared size and MD5."""
    upload = item.upload
    if item.data is not None or not isinstance(upload, _Upload):
        if item.data is None:
            raise TypeError("an object without a file must carry its bytes")
        return item.data
    changed: str | None = None
    data = b""
    with _opened_artifact(upload) as body:
        try:
            data = body.read(upload.size_bytes + 1)
        except OSError as error:
            changed = f"it could not be read: {_errno_name(error)}"
    if changed is None and (len(data) != upload.size_bytes or _md5_of(data) != item.md5):
        changed = "its bytes are not the ones its put URL was asked for"
    if changed is not None:
        raise _artifact_changed(upload, changed)
    return data


def _retry_after_s(headers: object) -> float:
    """The whole seconds an answer's `Retry-After` asks for, RETRY_AFTER_MAX_S at most; 0 for any other value."""
    asked = headers.get(RETRY_AFTER_HEADER) if hasattr(headers, "get") else None  # type: ignore[union-attr]
    if not isinstance(asked, str) or not (asked.isascii() and asked.isdigit()):
        return 0.0
    significant = asked.lstrip("0")
    if len(significant) > len(str(int(RETRY_AFTER_MAX_S))):
        return RETRY_AFTER_MAX_S
    return min(float(significant or "0"), RETRY_AFTER_MAX_S)


@dataclass(frozen=True)
class _Answer:
    """How one PUT ended: the store's status (None for no answer), or that it outlasted its ceiling."""

    status: int | None
    retry_after_s: float = 0.0
    stalled: bool = False


class _UrlUpload:
    """The PUTs of one upload: what they share (the session, the deadline, the running total and the stop)."""

    def __init__(
        self,
        capabilities: CapabilityClient,
        session: HttpSession,
        deadline: Deadline,
        sleep: Callable[[float], None],
        on_bytes: Callable[[int], None] | None,
    ) -> None:
        self._capabilities = capabilities
        self._session = session
        self._deadline = deadline
        self._stop_line = stop_line_within(deadline.remaining_s)
        self._sleep = sleep
        self._on_bytes = on_bytes
        self._stopped = threading.Event()
        self._total_lock = threading.Lock()
        self.uploaded = 0

    def stop(self) -> None:
        """No PUT of this upload starts from now on."""
        self._stopped.set()

    def ask(self, items: Sequence[_Item], lifetime_s: int) -> dict[str, PutUrl]:
        self._capabilities.raise_if_lost()
        if self._deadline.expired():
            raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
        answer = self._capabilities.ask(
            STAGE_UPLOADING, put=CapabilityPut(lifetime_seconds=lifetime_s, objects=tuple(i.declared for i in items))
        )
        return {url.path: url for url in answer.puts}

    def put(self, item: _Item, url: PutUrl) -> _Reask | _Skipped | None:
        """Store one object through `url`, trying again as the module says: None once stored, _REASK when its URL
        could not be used, _SKIPPED when the upload stopped first; WorkerFailure for a failure that ends it, which
        stops the upload at once, before the pool hands its thread another object."""
        try:
            return self._put(item, url)
        except BaseException:
            self._stopped.set()
            raise

    def _put(self, item: _Item, url: PutUrl) -> _Reask | _Skipped | None:
        nominal_waits_s = iter(UPLOAD_RETRY_WAITS_S)
        while True:
            if self._stopped.is_set():
                return _SKIPPED
            if self._deadline.expired():
                raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
            self._capabilities.raise_if_lost()
            if self._capabilities.monotonic() >= url.use_before - URL_MARGIN_S:
                return _REASK
            answer = self.put_once(item, url, _bytes_of(item))
            if answer.status == HTTP_OK:
                self._count(item.upload.size_bytes)
                return None
            if answer.status == HTTP_FORBIDDEN:
                return _REASK
            if answer.status is not None and not _retryable(answer.status):
                logger.warning("an upload through its URL was refused for good: status %d", answer.status)
                raise _upload_refused(f"status {answer.status}")
            failed = WorkerFailure(UPLOAD_STALLED) if answer.stalled else _upload_failed(
                "no answer" if answer.status is None else f"status {answer.status}"
            )
            wait_s = _next_wait_s(nominal_waits_s, self._deadline, at_least_s=answer.retry_after_s)
            if wait_s is None:
                raise failed
            logger.warning("an upload through its URL failed and is tried again after %.3f s", wait_s)
            self._sleep(wait_s)

    def put_once(self, item: _Item, url: PutUrl, data: bytes) -> _Answer:
        """One PUT of `data` through `url`: how it ended; WorkerFailure once the deadline has passed."""
        upload = item.upload
        ceiling_ends_at = time.monotonic() + _request_ceiling_s(upload.size_bytes)
        # The body stops at the nearer of the execution's deadline and the request's own ceiling.
        body_stop = stop_line_within(lambda: min(self._deadline.remaining_s(), ceiling_ends_at - time.monotonic()))
        timeout = self._stop_line.cut(CONNECT_TIMEOUT_S, _request_ceiling_s(upload.size_bytes))
        if timeout is None:
            raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
        headers = {
            "Content-Length": str(upload.size_bytes),
            "Content-MD5": item.md5,
            "Content-Type": upload.content_type,
        }
        response: HttpResponse | None = None
        try:
            response = self._session.put(url.url, BoundedBody(data, body_stop), headers, timeout, stop_line=self._stop_line)
        except (DeadlineReached, *TRANSFER_ERRORS):
            # No answer: the session's error quotes the URL, so nothing of it is kept.
            response = None
        if self._stop_line.passed():
            if response is not None:
                close_quietly(response)
            raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
        if response is None:
            return _Answer(None, stalled=body_stop.passed())
        answer = _Answer(response.status_code, _retry_after_s(response.headers))
        close_quietly(response)
        return answer

    def _count(self, size_bytes: int) -> None:
        with self._total_lock:
            self.uploaded += size_bytes
            _notify(self._on_bytes, self.uploaded)

    def run_batch(self, pool: ThreadPoolExecutor, items: Sequence[_Item], urls: dict[str, PutUrl]) -> list[_Item]:
        """PUT a batch on the pool; the objects whose URLs must be asked again. The first failure stops the rest."""
        futures: dict[Future[_Reask | _Skipped | None], _Item] = {
            pool.submit(self.put, item, urls[item.upload.relative_path]): item for item in items
        }
        failure: WorkerFailure | None = None
        reasks: list[_Item] = []
        pending = set(futures)
        expired_since: float | None = None
        while pending:
            done, pending = wait(pending, timeout=BATCH_POLL_S, return_when=FIRST_COMPLETED)
            for future in done:
                outcome, raised = _outcome_of(future)
                if raised is not None and failure is None:
                    failure = raised
                elif outcome is _REASK:
                    reasks.append(futures[future])
            if failure is None:
                failure = self._capabilities.lost()
            if failure is None and self._deadline.expired():
                failure = WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
            if failure is not None:
                self._stopped.set()
            if pending and self._deadline.expired():
                expired_since = time.monotonic() if expired_since is None else expired_since
                if time.monotonic() - expired_since >= DRAIN_AFTER_DEADLINE_S:
                    _give_up(pending)
                    break
        if failure is not None:
            raise failure
        return reasks


def _outcome_of(future: Future[_Reask | _Skipped | None]) -> tuple[_Reask | _Skipped | None, WorkerFailure | None]:
    raised = future.exception()
    if raised is None:
        return future.result(), None
    if isinstance(raised, WorkerFailure):
        return None, raised
    # A bug on a pool thread: raised again as it is, on the job's own thread.
    raise raised


def _give_up(pending: set[Future[_Reask | _Skipped | None]]) -> None:
    for _ in pending:
        _count_abandoned_request()
    logger.warning("%d uploads through their URLs were given up after the deadline", len(pending))


def _retryable(status: int) -> bool:
    return status in (HTTP_REQUEST_TIMEOUT, HTTP_TOO_MANY_REQUESTS) or status >= HTTP_SERVER_ERROR


def _require_single_put_sizes(uploads: Sequence[_Object]) -> None:
    if any(upload.size_bytes > MAX_OBJECT_BYTES for upload in uploads):
        raise WorkerFailure(OBJECT_TOO_LARGE)


def upload_generation_by_urls(
    capabilities: CapabilityClient,
    session: HttpSession,
    *,
    prefix: str,
    output_root: Path,
    artifacts: Sequence[ArtifactFile],
    manifest: ManifestOutput,
    deadline: Deadline,
    sleep: Callable[[float], None],
    on_bytes: Callable[[int], None] | None = None,
    parallelism: int = UPLOAD_PARALLELISM,
) -> int:
    """Store the generation through granted put URLs and return the bytes uploaded: `artifacts` in batches, then the
    manifest. Raises what upload_generation() raises before any request, WorkerFailure as the module says."""
    if not isinstance(capabilities, CapabilityClient):
        raise TypeError("capabilities must be a CapabilityClient")
    if isinstance(parallelism, bool) or not isinstance(parallelism, int) or not 1 <= parallelism <= UPLOAD_PARALLELISM:
        raise ValueError(f"parallelism must be an integer from 1 to {UPLOAD_PARALLELISM}")
    _require_transfer_arguments(deadline, sleep, on_bytes)
    uploads = _planned_uploads(prefix, output_root, artifacts, manifest)
    _require_what_the_manifest_records(artifacts, manifest)
    _require_single_put_sizes(uploads)
    *artifact_uploads, manifest_upload = uploads
    run = _UrlUpload(capabilities, session, deadline, sleep, on_bytes)
    pool = ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix=POOL_THREAD_PREFIX)
    try:
        queue = deque(artifact_uploads)
        waiting: deque[_Item] = deque()
        while queue or waiting:
            batch: list[_Item] = []
            while (queue or waiting) and len(batch) < BATCH_MAX_OBJECTS:
                batch.append(waiting.popleft() if not queue else _prepared(queue.popleft()))
            urls = run.ask(batch, BATCH_URL_LIFETIME_S)
            for item in run.run_batch(pool, batch, urls):
                waiting.append(_refused_again(item))
        _put_alone(run, pool, _prepared(manifest_upload))
    finally:
        run.stop()
        pool.shutdown(wait=False, cancel_futures=True)
    return run.uploaded


def _refused_again(item: _Item) -> _Item:
    item.refusals += 1
    if item.refusals > MAX_REASKS:
        logger.warning("the put URLs of an object were refused %d times", item.refusals)
        raise WorkerFailure(UPLOAD_URL_REFUSED)
    return item


def _put_alone(run: _UrlUpload, pool: ThreadPoolExecutor, item: _Item) -> None:
    """One object asked for alone with the last, short lifetime, and PUT, asked again as a batch's object is."""
    while True:
        urls = run.ask((item,), LAST_URL_LIFETIME_S)
        if not run.run_batch(pool, (item,), urls):
            return
        _refused_again(item)


def upload_result_by_url(
    capabilities: CapabilityClient,
    session: HttpSession,
    *,
    result: SealedMessage,
    deadline: Deadline,
) -> bool:
    """Attempt one PUT of the terminal result's RunPod carrier at `result.json` through a URL of its own; whether it
    was stored. False, asking nothing, once the authority is lost or the deadline has passed; WorkerFailure for a
    refusal of the Hub or a PUT that failed. The caller keeps the sealed result whatever this does."""
    if result.kind not in {MessageKind.TRANSCODE_RESULT_COMPLETED, MessageKind.TRANSCODE_RESULT_FAILED}:
        raise ValueError("only a terminal result may be stored")
    if not isinstance(capabilities, CapabilityClient):
        raise TypeError("capabilities must be a CapabilityClient")
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if capabilities.lost() is not None or deadline.expired() or capabilities.stage is None:
        return False
    carrier = json.dumps(runpod_output(result), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    upload = _Object(
        relative_path=RESULT_NAME,
        key=RESULT_NAME,
        size_bytes=len(carrier),
        sha256=hashlib.sha256(carrier).hexdigest(),
        content_type=CONTENT_TYPES[".json"],
    )
    item = _Item(upload=upload, md5=_md5_of(carrier), data=carrier)
    run = _UrlUpload(capabilities, session, deadline, lambda seconds: None, None)
    url = run.ask((item,), LAST_URL_LIFETIME_S)[RESULT_NAME]
    if capabilities.monotonic() >= url.use_before - URL_MARGIN_S:
        return False
    answer = run.put_once(item, url, carrier)
    if answer.status != HTTP_OK:
        raise _upload_failed("no answer" if answer.status is None else f"status {answer.status}")
    return True
