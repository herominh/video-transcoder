"""One import job (contract import/v1; the Hub's import design, sections 2 and 6, G-10, G-17, G-18): the input
checked, the Drive file's metadata asked again, the file downloaded and written part by part into the Hub's
multipart upload, then one signed receipt beside the original.

run_import() takes everything outside it as a parameter: the input document as RunPod hands it, the settings
(ImportSettings), the session it sends its requests through (transport.HttpSession) and the clock (clock.Clock).
In order:

1. The input against input.schema.json, the settings (all three, or `not_configured` before any request), the
   input's MAC under the import key (contract.input_mac_verifies(), README section 3.1: the worker acts only on an
   input the Hub wrote), then the input's other rules (input_check.check()); any refusal of the input is
   `input_refused`.
2. Google's metadata of the file again (google.change_at_provider()): before anything is written, a file that is no
   longer the pasted one is refused.
3. The download, from the start and, after a reset or a time-out, again from the start of the part in flight
   (`Range`), at most DOWNLOAD_TRIES requests for each part; its bytes counted against the expected size (the first
   byte beyond it is `size_exceeded` at once, fewer at the end `size_short`) and hashed (SHA-256, MD5).
4. Each part of 64 MiB, as soon as it is whole, PUT to its URL (store.put()), the store's ETag kept; the LAST part
   only once the whole stream matched Google's SHA-256 when the input carries one, else its MD5, else nothing
   (`checksum_verified`; a mismatch is `integrity_failed` and the last part is never written).
5. The receipt {contract, job, bytes, checksum_verified, parts, mac}, its MAC under the receipt key
   (contract.receipt_mac()), checked against receipt.schema.json and PUT to the receipt URL.

The output is {contract, status: "imported", code: null} or {contract, status: "failed", code}, checked against
output.schema.json before it is returned. A failure the job knows is an output, never an exception: run_import()
raises only on a bug. Nothing of a URL, the file id, a key, Google's text, an ETag or an answer's body goes into a
log line, an exception's text or the output; a log line holds the job's uuid, the stage, the code, counts and
seconds, and is written only once no error is being handled (a log handler that fails prints the error being
handled, whose text would name a URL).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Iterator, Mapping

from core.drive_import import contract, google, store
from core.drive_import.clock import Clock, DeadlineReached, StopLine
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.input_check import ImportInput, InputRefused, check
from core.drive_import.settings import ImportSettings
from core.drive_import.transport import TRANSFER_ERRORS, HttpResponse, HttpSession, close_quietly

logger = logging.getLogger(__name__)

IMPORTED = "imported"
FAILED = "failed"
# How many bytes the download asks of each read of the stream.
DOWNLOAD_CHUNK_BYTES = 1_048_576
# Requests of the metadata, and requests of the download for each part (the first that reaches it included).
METADATA_TRIES = 3
DOWNLOAD_TRIES = 3
# The wait before each request made again: before the second, before the third.
DOWNLOAD_RETRY_WAITS_S = (0.5, 2.0)
RECEIPT_HEADERS = {"Content-Type": "application/json"}
CHECKSUM_SHA256 = "sha256"
CHECKSUM_MD5 = "md5"
CHECKSUM_NONE = "none"


class SelfCheckFailed(RuntimeError):
    """A document the job built is not one of the contract: a bug, never a job's ending."""


class _Tally:
    """What the outcome's log line counts. `job` is the input's uuid once the schema accepted the input."""

    def __init__(self) -> None:
        self.job: str | None = None
        self.bytes_read = 0
        self.parts_written = 0


def run_import(
    input_document: object, settings: ImportSettings, http: HttpSession, clock: Clock
) -> dict[str, Any]:
    """One import job, from its input document to its output document (the module says how)."""
    started = clock.monotonic()
    tally = _Tally()
    failure: ImportFailed | None = None
    try:
        _import(input_document, settings, http, clock, tally)
    except ImportFailed as failed:
        failure = failed
    output = _output(failure)
    _log_outcome(tally, failure, clock.monotonic() - started)
    return output


def _import(document: object, settings: ImportSettings, http: HttpSession, clock: Clock, tally: _Tally) -> None:
    # The schema's first rule is that the input is an object; the isinstance() only says so to the type checker.
    if not isinstance(document, Mapping) or contract.first_error_pointer(contract.INPUT_SCHEMA, document) is not None:
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    tally.job = document["job"]
    api_key, receipt_key, storage_host = settings.google_api_key, settings.receipt_key, settings.storage_host
    if api_key is None or receipt_key is None or storage_host is None:
        raise ImportFailed(Code.NOT_CONFIGURED, Stage.CONFIGURATION)
    # The receipt key is the import key, which signs the input too (README section 3.1).
    if not contract.input_mac_verifies(document, receipt_key):
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    job = _checked_input(document, storage_host, clock)
    stop_line = StopLine.before(job.writes_end_at, clock)
    stop_line.require(Stage.INPUT)

    _require_unchanged(job, api_key, http, stop_line, tally)
    parts, checksum_verified = _Transfer(job, api_key, http, stop_line, tally).run()
    _write_receipt(job, receipt_key, http, stop_line, tally, parts, checksum_verified)


def _checked_input(document: Mapping[str, Any], storage_host: str, clock: Clock) -> ImportInput:
    """The input's rules beyond its schema (input_check.check()); a refusal is `input_refused`."""
    try:
        checked: ImportInput | None = check(document, storage_host, clock.now())
    except InputRefused:
        # The refusal's text names the rule only; it is answered below, once it is no longer being handled, so
        # that nothing of it travels with the job's own failure.
        checked = None
    if checked is None:
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    return checked


# --- the metadata again (G-10) -------------------------------------------------------------------------------------


def _require_unchanged(job: ImportInput, api_key: str, http: HttpSession, stop_line: StopLine, tally: _Tally) -> None:
    """Ask Google for the file's metadata again and refuse a file that is no longer the pasted one."""
    url = google.metadata_url(job.file_id)
    headers = google.request_headers(api_key, job)
    for attempt in range(1, METADATA_TRIES + 1):
        if attempt > 1:
            stop_line.wait(DOWNLOAD_RETRY_WAITS_S[attempt - 2], Stage.METADATA)
        timeout = stop_line.timeout(Stage.METADATA, google.CONNECT_TIMEOUT_S, google.METADATA_READ_TIMEOUT_S)
        answer = _metadata_once(http, url, headers, timeout, stop_line)
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
        return http.get(url, headers, timeout, True, stop_line=stop_line)
    except DeadlineReached:
        reached = True
    except TRANSFER_ERRORS:
        return None
    if reached:
        raise ImportFailed(Code.DEADLINE, stage)
    return None


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

    def __init__(self, job: ImportInput, api_key: str, http: HttpSession, stop_line: StopLine, tally: _Tally) -> None:
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
        self._written: list[dict[str, Any]] = []

    def run(self) -> tuple[list[dict[str, Any]], str]:
        """Download the whole file and write every part; return the receipt's parts and `checksum_verified`."""
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
        self._put_part()
        self._part_start += len(self._buffer)
        self._part += 1
        self._buffer = bytearray()
        self._tries = 1

    def _finish(self) -> tuple[list[dict[str, Any]], str]:
        """The stream ended: refuse a short one, verify the whole, then write the last part."""
        if self._part_start + len(self._buffer) < self._job.expected_size:
            raise ImportFailed(Code.SIZE_SHORT, Stage.DOWNLOAD)
        if self._part != len(self._sizes) or len(self._buffer) != self._sizes[-1]:
            raise SelfCheckFailed("a stream of the expected size did not end in the last part, whole")
        self._sha256.update(self._buffer)
        self._md5.update(self._buffer)
        checksum_verified = self._verified()
        self._put_part()
        return self._written, checksum_verified

    def _verified(self) -> str:
        """The strongest of Google's checksums the input carries, which the whole stream must match."""
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
        if etag is None:
            raise ImportFailed(Code.UPLOAD_FAILED, Stage.UPLOAD)
        self._written.append({"n": part, "etag": etag})
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


# --- the receipt (README section 5) --------------------------------------------------------------------------------


def _write_receipt(
    job: ImportInput,
    receipt_key: str,
    http: HttpSession,
    stop_line: StopLine,
    tally: _Tally,
    parts: list[dict[str, Any]],
    checksum_verified: str,
) -> None:
    receipt: dict[str, Any] = {
        "contract": contract.CONTRACT,
        "job": job.job,
        "bytes": job.expected_size,
        "checksum_verified": checksum_verified,
        "parts": parts,
    }
    receipt[contract.MAC_MEMBER] = contract.receipt_mac(receipt, receipt_key)
    if contract.first_error_pointer(contract.RECEIPT_SCHEMA, receipt) is not None:
        raise SelfCheckFailed("the receipt the job built is not one of the contract")
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")
    store.put(
        http,
        job.receipt_url,
        body,
        RECEIPT_HEADERS,
        stop_line,
        Stage.RECEIPT,
        lambda attempt: _log_failed_try(tally, Stage.RECEIPT, None, attempt),
    )


# --- the output and the log ----------------------------------------------------------------------------------------


def _output(failure: ImportFailed | None) -> dict[str, Any]:
    if failure is None:
        output: dict[str, Any] = {"contract": contract.CONTRACT, "status": IMPORTED, "code": None}
    else:
        output = {"contract": contract.CONTRACT, "status": FAILED, "code": failure.code.value}
    if contract.first_error_pointer(contract.OUTPUT_SCHEMA, output) is not None:
        raise SelfCheckFailed("the output the job built is not one of the contract")
    return output


def _log_outcome(tally: _Tally, failure: ImportFailed | None, seconds: float) -> None:
    job = tally.job or "-"
    if failure is None:
        _say(
            logging.INFO,
            "import %s: imported, %d bytes read, %d parts written, %.1f s",
            job, tally.bytes_read, tally.parts_written, seconds,
        )
        return
    _say(
        logging.WARNING,
        "import %s: failed at %s with %s, %d bytes read, %d parts written, %.1f s",
        job, failure.stage.value, failure.code.value, tally.bytes_read, tally.parts_written, seconds,
    )


def _log_failed_try(tally: _Tally, stage: Stage, part: int | None, attempt: int) -> None:
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
