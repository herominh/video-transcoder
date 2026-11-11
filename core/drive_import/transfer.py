"""The acquire's Google metadata check and bounded multipart transfer (contract 2.1.0)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from core.drive_import import google, store
from core.drive_import.clock import DeadlineReached, StopLine
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.transport import TRANSFER_ERRORS, HttpResponse, HttpSession, close_quietly

logger = logging.getLogger(__name__)

PART_SIZE = 67_108_864
MAX_PARTS = 160
DOWNLOAD_CHUNK_BYTES = 1_048_576
METADATA_TRIES = 3
DOWNLOAD_TRIES = 3
DOWNLOAD_RETRY_WAITS_S = (0.5, 2.0)
CHECKSUM_SHA256 = "sha256"
CHECKSUM_MD5 = "md5"
CHECKSUM_NONE = "none"


@dataclass(frozen=True)
class Part:
    number: int
    url: str = field(repr=False)


@dataclass(frozen=True)
class TransferPlan:
    file_id: str = field(repr=False)
    resource_key: str | None = field(repr=False)
    expected_size: int
    expected_md5: str | None
    expected_sha1: str | None
    expected_sha256: str | None
    expected_head_revision_id: str | None = field(repr=False)
    part_size: int
    parts: tuple[Part, ...] = field(repr=False)
    stop_line: StopLine = field(repr=False)
    reject_url_etags: bool = True

    def require_valid(self) -> None:
        integers = (self.expected_size, self.part_size, *(part.number for part in self.parts))
        if (any(type(value) is not int for value in integers)
                or not 0 < self.expected_size <= PART_SIZE * MAX_PARTS
                or self.part_size != PART_SIZE
                or len(self.parts) != (self.expected_size + PART_SIZE - 1) // PART_SIZE
                or tuple(part.number for part in self.parts) != tuple(range(1, len(self.parts) + 1))):
            raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)


@dataclass(frozen=True)
class CompletedPart:
    n: int
    etag: str = field(repr=False)

    def to_dict(self) -> dict[str, int | str]:
        return {"n": self.n, "etag": self.etag}


@dataclass(frozen=True)
class TransferResult:
    bytes: int
    checksum_verified: str
    parts: tuple[CompletedPart, ...] = field(repr=False)


@dataclass
class Tally:
    job: str | None = None
    bytes_read: int = 0
    parts_written: int = 0
    # The active transfer operation, including buffering/checking downloaded bytes and recording a PUT.
    stage: Stage = Stage.DOWNLOAD


class SelfCheckFailed(RuntimeError):
    """The transfer reached an inconsistent internal state."""


def transfer(plan: TransferPlan, api_key: str, http: HttpSession, tally: Tally | None = None) -> TransferResult:
    plan.require_valid()
    plan.stop_line.require(Stage.INPUT)
    counts = tally if tally is not None else Tally()
    _require_unchanged(plan, api_key, http, plan.stop_line, counts)
    parts, checksum = _Transfer(plan, api_key, http, plan.stop_line, counts).run()
    plan.stop_line.require(Stage.VERIFY)
    return TransferResult(plan.expected_size, checksum, tuple(parts))


# --- the metadata again (G-10) -------------------------------------------------------------------------------------


def _require_unchanged(job: TransferPlan, api_key: str, http: HttpSession, stop_line: StopLine, tally: Tally) -> None:
    """Ask Google for the file's metadata again and refuse a file that is no longer the pasted one."""
    url = google.metadata_url(job.file_id)
    headers = google.request_headers(api_key, job)
    for attempt in range(1, METADATA_TRIES + 1):
        if attempt > 1:
            stop_line.wait(DOWNLOAD_RETRY_WAITS_S[attempt - 2], Stage.METADATA)
        timeout = stop_line.timeout(Stage.METADATA, google.CONNECT_TIMEOUT_S, google.METADATA_READ_TIMEOUT_S)
        answer = _metadata_once(http, url, headers, timeout, stop_line)
        stop_line.require(Stage.METADATA)
        if answer is Code.DOWNLOAD_FAILED:
            # An answer cut short by the watchdog at the stop line reads as one Google did not give whole.
            raise stop_line.failed(answer, Stage.METADATA)
        if isinstance(answer, Code):
            raise ImportFailed(answer, Stage.METADATA)
        if isinstance(answer, dict):
            change = google.change_at_provider(job, answer)
            if change is not None:
                raise ImportFailed(change, Stage.METADATA)
            return
        _log_failed_try(tally, Stage.METADATA, None, attempt)
    raise stop_line.failed(Code.DOWNLOAD_FAILED, Stage.METADATA)


def _metadata_once(
    http: HttpSession, url: str, headers: Mapping[str, str], timeout: tuple[float, float], stop_line: StopLine
) -> dict[str, Any] | Code | google.Retry:
    response = _get(http, url, headers, timeout, stop_line, Stage.METADATA)
    if response is None:
        return google.RETRY
    try:
        return google.metadata_answer(response)
    finally:
        close_quietly(response)


def _get(
    http: HttpSession,
    url: str,
    headers: Mapping[str, str],
    timeout: tuple[float, float],
    stop_line: StopLine,
    stage: Stage,
) -> HttpResponse | None:
    """One GET, streamed and held to the stop line; None when it got no answer (the error's text names the URL, so
    nothing of it is kept). ImportFailed(deadline) when the session refused it, or its setup, at the stop line."""
    reached = False
    try:
        response = http.get(url, headers, timeout, True, stop_line=stop_line)
    except DeadlineReached:
        reached = True
    except TRANSFER_ERRORS:
        return None
    if reached:
        raise ImportFailed(Code.DEADLINE, stage)
    if stop_line.passed():
        close_quietly(response)
        raise ImportFailed(Code.DEADLINE, stage)
    return _TimedResponse(response, stop_line, stage)


class _TimedResponse:
    def __init__(self, response: HttpResponse, stop_line: StopLine, stage: Stage) -> None:
        self._response = response
        self._stop_line = stop_line
        self._stage = stage
        self.status_code = response.status_code
        self.headers = response.headers

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        self._stop_line.require(self._stage)
        chunks = self._response.iter_content(chunk_size)
        while True:
            self._stop_line.require(self._stage)
            chunk = next(chunks, None)
            self._stop_line.require(self._stage)
            if chunk is None:
                return
            yield chunk

    def close(self) -> None:
        self._response.close()


# --- the download and the parts (G-17, G-18) -----------------------------------------------------------------------


class _End:
    """The stream ended where its sender ended it."""


class _Broken:
    """The stream failed before its end: a reset, a time-out, an answer cut short."""


_END = _End()
_BROKEN = _Broken()


class _Transfer:
    """One download, written part by part as it comes: the part in flight is buffered whole, the whole stream is
    hashed part by part, and a break starts the part in flight again from its first byte."""

    def __init__(self, job: TransferPlan, api_key: str, http: HttpSession, stop_line: StopLine, tally: Tally) -> None:
        self._job = job
        self._api_key = api_key
        self._http = http
        self._stop_line = stop_line
        self._tally = tally
        parts_total = len(job.parts)
        last_size = job.expected_size - (parts_total - 1) * job.part_size
        self._sizes = [job.part_size] * (parts_total - 1) + [last_size]
        self._urls = {part.number: part.url for part in job.parts}
        self._sha256 = hashlib.sha256()
        self._md5 = hashlib.md5(usedforsecurity=False)
        # The part in flight (numbered from 1), the offset of its first byte, its bytes so far, and how many
        # requests reached it (the request that was running when the part before it was whole counts as its first).
        self._part = 1
        self._part_start = 0
        self._buffer = bytearray()
        self._tries = 0
        self._written: list[CompletedPart] = []

    def run(self) -> tuple[list[CompletedPart], str]:
        """Download the whole file and write every part; return the result's parts and `checksum_verified`."""
        while True:
            response = self._open()
            try:
                ended = self._read(response)
            finally:
                close_quietly(response)
            if ended:
                return self._finish()
            self._buffer = bytearray()
            self._log_failed_try()

    def _open(self) -> HttpResponse:
        """The next request of the download, from the first byte of the part in flight; ImportFailed when Google
        refuses, or when the part's requests are used up (`download_failed`)."""
        while True:
            self._tries += 1
            if self._tries > DOWNLOAD_TRIES:
                raise self._stop_line.failed(Code.DOWNLOAD_FAILED, Stage.DOWNLOAD)
            if self._tries > 1:
                self._stop_line.wait(DOWNLOAD_RETRY_WAITS_S[self._tries - 2], Stage.DOWNLOAD)
            answer = self._request()
            if self._stop_line.passed():
                if not isinstance(answer, (Code, google.Retry)):
                    close_quietly(answer)
                raise ImportFailed(Code.DEADLINE, Stage.DOWNLOAD)
            if answer is Code.DOWNLOAD_FAILED:
                raise self._stop_line.failed(answer, Stage.DOWNLOAD)
            if isinstance(answer, Code):
                raise ImportFailed(answer, Stage.DOWNLOAD)
            if not isinstance(answer, google.Retry):
                return answer
            self._log_failed_try()

    def _request(self) -> HttpResponse | Code | google.Retry:
        timeout = self._stop_line.timeout(Stage.DOWNLOAD, google.CONNECT_TIMEOUT_S, google.MEDIA_READ_TIMEOUT_S)
        headers = google.media_headers(self._api_key, self._job, self._part_start)
        response = _get(
            self._http, google.media_url(self._job.file_id), headers, timeout, self._stop_line, Stage.DOWNLOAD
        )
        if response is None:
            return google.RETRY
        answer = google.media_answer(response, self._part_start)
        if answer is not response:
            close_quietly(response)
        return answer

    def _read(self, response: HttpResponse) -> bool:
        """Read the stream until it ends (True) or breaks (False). The stop line is checked before each read and
        after it: no read starts once it has passed, a chunk that comes after it is not taken, and an end that comes
        after it is the watchdog's cut (a body framed by the connection's close ends cleanly when it is shut)."""
        chunks = _chunks_of(response)
        while chunks is not None:
            self._stop_line.require(Stage.DOWNLOAD)
            chunk = _next_chunk(chunks)
            if isinstance(chunk, _Broken):
                return False
            if isinstance(chunk, _End):
                self._stop_line.require(Stage.DOWNLOAD)
                return True
            self._stop_line.require(Stage.DOWNLOAD)
            self._tally.bytes_read += len(chunk)
            self._take(chunk)
        return False

    def _take(self, chunk: bytes) -> None:
        """Add `chunk` to the part in flight, writing each part that becomes whole but the last."""
        if self._part_start + len(self._buffer) + len(chunk) > self._job.expected_size:
            raise ImportFailed(Code.SIZE_EXCEEDED, Stage.DOWNLOAD)
        rest = memoryview(chunk)
        # The size check above makes the rest fit: the parts' sizes add up to the expected size.
        while rest:
            room = self._sizes[self._part - 1] - len(self._buffer)
            self._buffer += rest[:room]
            rest = rest[room:]
            if len(self._buffer) == self._sizes[self._part - 1] and self._part < len(self._sizes):
                self._write_whole_part()

    def _write_whole_part(self) -> None:
        """Write the part in flight, which is whole and not the last, and make the next one the part in flight."""
        self._sha256.update(self._buffer)
        self._md5.update(self._buffer)
        self._stop_line.require(Stage.DOWNLOAD)
        self._put_part()
        self._tally.stage = Stage.DOWNLOAD
        self._part_start += len(self._buffer)
        self._part += 1
        self._buffer = bytearray()
        self._tries = 1

    def _finish(self) -> tuple[list[CompletedPart], str]:
        """The stream ended: refuse a short one, verify the whole, then write the last part."""
        if self._part_start + len(self._buffer) < self._job.expected_size:
            raise ImportFailed(Code.SIZE_SHORT, Stage.DOWNLOAD)
        if self._part != len(self._sizes) or len(self._buffer) != self._sizes[-1]:
            raise SelfCheckFailed("a stream of the expected size did not end in the last part, whole")
        self._stop_line.require(Stage.VERIFY)
        self._sha256.update(self._buffer)
        self._md5.update(self._buffer)
        self._stop_line.require(Stage.VERIFY)
        checksum_verified = self._verified()
        self._stop_line.require(Stage.VERIFY)
        self._put_part()
        return self._written, checksum_verified

    def _verified(self) -> str:
        """The strongest of Google's checksums the dispatch carries, which the whole stream must match."""
        if self._job.expected_sha256 is not None:
            if self._sha256.hexdigest() != self._job.expected_sha256:
                raise ImportFailed(Code.INTEGRITY_FAILED, Stage.VERIFY)
            return CHECKSUM_SHA256
        if self._job.expected_md5 is not None:
            if self._md5.hexdigest() != self._job.expected_md5:
                raise ImportFailed(Code.INTEGRITY_FAILED, Stage.VERIFY)
            return CHECKSUM_MD5
        return CHECKSUM_NONE

    def _put_part(self) -> None:
        self._tally.stage = Stage.UPLOAD
        part = self._part
        etag = store.put(
            self._http,
            self._urls[part],
            self._buffer,
            {},
            self._stop_line,
            Stage.UPLOAD,
            lambda attempt: _log_failed_try(self._tally, Stage.UPLOAD, part, attempt),
        )
        if etag is None or (self._job.reject_url_etags and "://" in etag):
            raise ImportFailed(Code.UPLOAD_FAILED, Stage.UPLOAD)
        self._written.append(CompletedPart(part, etag))
        self._tally.parts_written += 1

    def _log_failed_try(self) -> None:
        _log_failed_try(self._tally, Stage.DOWNLOAD, self._part, self._tries)


def _chunks_of(response: HttpResponse) -> Iterator[bytes] | None:
    """The stream's chunks, or None when the answer cannot give them (its error's text names the URL: not kept)."""
    try:
        return response.iter_content(DOWNLOAD_CHUNK_BYTES)
    except TRANSFER_ERRORS:
        return None


def _next_chunk(chunks: Iterator[bytes]) -> bytes | _End | _Broken:
    """The next non-empty chunk of the stream, its end, or its break (the error's text names the URL: not kept)."""
    while True:
        try:
            chunk = next(chunks, None)
        except TRANSFER_ERRORS:
            return _BROKEN
        if chunk is None:
            return _END
        if chunk:
            return chunk


def _log_failed_try(tally: Tally, stage: Stage, part: int | None, attempt: int) -> None:
    if part is None:
        _say(logging.WARNING, "import %s: %s try %d failed", tally.job or "-", stage.value, attempt)
        return
    _say(logging.WARNING, "import %s: %s of part %d, try %d failed", tally.job or "-", stage.value, part, attempt)


def _say(level: int, message: str, *arguments: object) -> None:
    """One log line. It never raises: a line that cannot be written must not cost the job its ending."""
    try:
        logger.log(level, message, *arguments)
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return
