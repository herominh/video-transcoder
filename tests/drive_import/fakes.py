"""Shared helpers of the import job's tests: a clock, a Drive file, and one session that stands for Google and the
store at once. Nothing here opens a connection.

Not a test module. FakeClock moves only when the job waits or a test moves it. DriveFile is the file's bytes, made
lazily block by block (each block of 1 MiB starts with its own number, so a byte out of place changes every
checksum), with its checksums computed once. FakeSession answers the job's requests as the test scripts them:
Google's metadata and download by default from the DriveFile, the store's PUTs with a 200 and the body's MD5 as the
ETag (as S3 answers an UploadPart); every request is recorded, and a PUT's body is read as urllib3 reads one.
import_input() builds a valid input document for a file, in the shape of the contract's running example, signed with
the receipt key of the test's settings; signed() signs a document a test changed again, so that it reaches the rules
after the MAC.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any

import requests
from requests.structures import CaseInsensitiveDict

from core.drive_import import contract
from core.drive_import.clock import DeadlineReached, StopLine
from core.drive_import.input_check import PART_SIZE
from core.drive_import.settings import ImportSettings

START = datetime(2026, 10, 17, 8, 0, 0, tzinfo=timezone.utc)
# The writer deadline of a test's input, from the clock's start: the stop line is 120 s before it.
WRITE_WINDOW = timedelta(hours=1)
STOP_MARGIN_S = 120.0

STORAGE_HOST = "storage.example.invalid"
BUCKET = "videohub-school-a"
VIDEO_UUID = "0199f1a0-1111-7000-8000-000000000001"
JOB_UUID = "0199f1a0-2222-7000-8000-000000000002"
FILE_ID = "1CanaryFileIdOfTheDriveFile0123xyz"
RESOURCE_KEY = "0-CanaryResourceKey01"
HEAD_REVISION = "0BCanaryHeadRevision0123"
UPLOAD_ID = "upload-canary-1"
# Values no log line, error text or output may show.
API_KEY = "CANARY-google-api-key-0123456789abcdef"
RECEIPT_KEY = "CANARY-receipt-key-0123456789abcdefABCDEF"
SIGNATURE_CANARY = "CANARY-signature"

GOOGLE_FILES = "https://www.googleapis.com/drive/v3/files/"
METADATA_URL = (
    f"{GOOGLE_FILES}{FILE_ID}?supportsAllDrives=true"
    "&fields=size,md5Checksum,sha1Checksum,sha256Checksum,headRevisionId,trashed,capabilities/canDownload"
)
MEDIA_URL = f"{GOOGLE_FILES}{FILE_ID}?alt=media&supportsAllDrives=true"
ORIGINALS = f"https://{STORAGE_HOST}/{BUCKET}/original-videos/{VIDEO_UUID}/"
RECEIPT_URL = f"{ORIGINALS}import-receipt-{JOB_UUID}.json?X-Amz-Expires=1013&X-Amz-Signature={SIGNATURE_CANARY}-r"
# Where a redirect a test scripts points: the job must never ask it.
REDIRECT_TARGET = "https://elsewhere.example.invalid/somewhere"

BLOCK_BYTES = 1_048_576
# The blocks urllib3 reads a file-like request body in.
BODY_BLOCK_BYTES = 16_384
MIB = 1_048_576
# The running example of the tests: three parts, the last one MiB long.
THREE_PARTS_SIZE = 2 * PART_SIZE + MIB


def part_url(n: int) -> str:
    query = f"partNumber={n}&uploadId={UPLOAD_ID}&X-Amz-Expires=1013&X-Amz-Signature={SIGNATURE_CANARY}-{n:03}"
    return f"{ORIGINALS}original.mp4?{query}"


def settings(**changes: str | None) -> ImportSettings:
    """The settings of a whole configuration, with `changes` (a None removes the variable)."""
    env: dict[str, str] = {
        "GOOGLE_DRIVE_API_KEY": API_KEY,
        "IMPORT_RECEIPT_KEY": RECEIPT_KEY,
        "IMPORT_STORAGE_HOST": STORAGE_HOST,
    }
    for name, value in changes.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return ImportSettings.from_env(env)


class FakeClock:
    """now() and monotonic() move only when the job sleeps or the test advances them; sleeps are recorded."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start
        self._monotonic = 1_000.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds

    def stop_line_in(self) -> float:
        """Seconds until the stop line of an input built by import_input() with the default window."""
        return (START + WRITE_WINDOW - timedelta(seconds=STOP_MARGIN_S) - self._now).total_seconds()


class DriveFile:
    """`size` bytes made block by block: block k is a random block of the seed with k written in its first bytes."""

    def __init__(self, size: int, seed: int = 7) -> None:
        self.size = size
        self._base = random.Random(seed).randbytes(BLOCK_BYTES)
        self._part_md5s: dict[int, str] = {}

    def _block(self, k: int) -> bytes:
        return k.to_bytes(8, "big") + self._base[8:]

    def chunks(self, start: int, end: int, chunk_size: int) -> Iterator[bytes]:
        """The bytes [start, end) in chunks of at most `chunk_size` (never across a block)."""
        position = start
        while position < end:
            k, offset = divmod(position, BLOCK_BYTES)
            take = min(BLOCK_BYTES - offset, chunk_size, end - position)
            yield self._block(k)[offset:offset + take]
            position += take

    def bytes(self, start: int = 0, end: int | None = None) -> bytes:
        return b"".join(self.chunks(start, self.size if end is None else end, BLOCK_BYTES))

    @cached_property
    def _digests(self) -> tuple[str, str, str]:
        sha256, md5, sha1 = hashlib.sha256(), hashlib.md5(), hashlib.sha1()
        for chunk in self.chunks(0, self.size, BLOCK_BYTES):
            sha256.update(chunk)
            md5.update(chunk)
            sha1.update(chunk)
        return sha256.hexdigest(), md5.hexdigest(), sha1.hexdigest()

    @property
    def sha256(self) -> str:
        return self._digests[0]

    @property
    def md5(self) -> str:
        return self._digests[1]

    @property
    def sha1(self) -> str:
        return self._digests[2]

    def part_md5(self, n: int) -> str:
        if n not in self._part_md5s:
            start = (n - 1) * PART_SIZE
            digest = hashlib.md5()
            for chunk in self.chunks(start, min(self.size, start + PART_SIZE), BLOCK_BYTES):
                digest.update(chunk)
            self._part_md5s[n] = digest.hexdigest()
        return self._part_md5s[n]

    def metadata(self) -> dict[str, Any]:
        """Google's metadata of the file, as the fields the job asks for come."""
        return {
            "size": str(self.size),
            "md5Checksum": self.md5,
            "sha1Checksum": self.sha1,
            "sha256Checksum": self.sha256,
            "headRevisionId": HEAD_REVISION,
            "trashed": False,
            "capabilities": {"canDownload": True},
        }


def import_input(
    file: DriveFile,
    *,
    md5: bool = True,
    sha1: bool = False,
    sha256: bool = True,
    head_revision: bool = True,
    resource_key: str | None = None,
    writes_end_at: datetime = START + WRITE_WINDOW,
) -> dict[str, Any]:
    """A valid input for `file`, carrying the checksums and head revision asked for."""
    parts_total = -(-file.size // PART_SIZE)
    return signed({
        "contract": "import/v1",
        "job": JOB_UUID,
        "file_id": FILE_ID,
        "resource_key": resource_key,
        "expected": {
            "size": file.size,
            "md5": file.md5 if md5 else None,
            "sha1": file.sha1 if sha1 else None,
            "sha256": file.sha256 if sha256 else None,
            "head_revision_id": HEAD_REVISION if head_revision else None,
        },
        "plan": {"part_size": PART_SIZE, "parts_total": parts_total},
        "parts": [{"n": n, "url": part_url(n)} for n in range(1, parts_total + 1)],
        "receipt_url": RECEIPT_URL,
        "writes_end_at": _contract_time(writes_end_at),
    })


def signed(document: dict[str, Any], key: str = RECEIPT_KEY) -> dict[str, Any]:
    """`document` with the MAC of its other members under `key` (README section 3.1)."""
    unsigned = {name: value for name, value in document.items() if name != "mac"}
    return {**unsigned, "mac": contract.input_mac(unsigned, key)}


def _contract_time(instant: datetime) -> str:
    """An aware time as the contract writes the writer deadline: UTC, seconds, a fraction only when there is one."""
    utc = instant.astimezone(timezone.utc)
    fraction = f".{utc.microsecond:06d}" if utc.microsecond else ""
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + fraction + "Z"


# --- what the fake answers -----------------------------------------------------------------------------------------


class FakeResponse:
    """An answer: a status, headers (case-insensitive) and a body given as chunks, possibly lazy."""

    def __init__(
        self,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        chunks: Callable[[int], Iterator[bytes]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(dict(headers or {}))
        self._chunks = chunks if chunks is not None else (lambda size: _split(body, size))
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        return self._chunks(chunk_size)

    def close(self) -> None:
        self.closed = True


def _blocks_of(body: Any) -> Iterator[bytes]:
    """A request body as urllib3 sends a file-like one: read() in blocks until it gives nothing."""
    while True:
        block = body.read(BODY_BLOCK_BYTES)
        if not block:
            return
        yield block


def _split(body: bytes, size: int) -> Iterator[bytes]:
    for start in range(0, len(body), size):
        yield body[start:start + size]


def json_answer(status: int, document: Any) -> FakeResponse:
    return FakeResponse(status, {"Content-Type": "application/json"}, json.dumps(document).encode("utf-8"))


def google_error(
    status: int, reason: str | None = None, domain: str | None = None, *, detail: str | None = None
) -> FakeResponse:
    """Google's error document, with a first error of `reason` and `domain` and an ErrorInfo of `detail`."""
    error: dict[str, Any] = {"code": status, "message": f"Google's own text naming {FILE_ID}"}
    if reason is not None:
        entry = {"reason": reason, "message": "more of Google's text"}
        if domain is not None:
            entry["domain"] = domain
        error["errors"] = [entry]
    if detail is not None:
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": detail}]
    return json_answer(status, {"error": error})


def redirect(status: int) -> FakeResponse:
    return FakeResponse(status, {"Location": REDIRECT_TARGET})


@dataclass
class Serve:
    """Serve the file from the offset asked (`Range`): a 200 from the start, else a 206. `break_after` bytes of this
    answer, the stream breaks (a reset); `extra` bytes are sent after the file's own; `offset` names another start
    in the 206's Content-Range than the one asked; `status` replaces the status the request calls for."""

    break_after: int | None = None
    extra: int = 0
    offset: int | None = None
    status: int | None = None
    # Seconds the clock moves just before the stream breaks.
    seconds_before_break: float = 0.0


@dataclass
class Fail:
    """The request gets no answer: `get` or `put` raises `error`."""

    error: BaseException = field(default_factory=lambda: requests.exceptions.ConnectionError(_url_text()))


def _url_text() -> str:
    """What requests' errors say: the request's whole URL (a presigned one included)."""
    return f"HTTPSConnectionPool(host='{STORAGE_HOST}'): Max retries exceeded with url: {part_url(1)} ({API_KEY})"


@dataclass
class Late:
    """`answer`, given once the clock moved `seconds`: a request that took that long."""

    answer: Any
    seconds: float


@dataclass
class Put:
    """How the store answers one PUT: a status (a 200 with the body's MD5 as ETag by default), headers in place of
    the default ETag, and seconds the clock moves while the body is read (per block) or after the answer."""

    status: int = 200
    headers: Mapping[str, str] | None = None
    seconds_per_block: float = 0.0
    seconds_after: float = 0.0


@dataclass
class Request:
    method: str
    url: str
    headers: dict[str, str]
    timeout: tuple[float, float]
    length: int = 0
    md5: str | None = None
    body: bytes | None = None
    cut: bool = False
    # How many bytes the download streams had given the job when this request was made.
    served_before: int = 0


class FakeSession:
    """Google and the store, as scripted. `metadata`, `media` and `puts` are lists of what the next request of that
    kind gets (a FakeResponse, a Serve, a Fail, a Put); once a list is empty, the default answer."""

    def __init__(self, file: DriveFile, clock: FakeClock, metadata: Mapping[str, Any] | None = None) -> None:
        self.file = file
        self.clock = clock
        self.metadata_document = dict(file.metadata() if metadata is None else metadata)
        self.metadata: list[Any] = []
        self.media: list[Any] = []
        self.puts: dict[str, list[Put | Fail | FakeResponse]] = {}
        self.requests: list[Request] = []
        # The bytes the download streams gave the job, and the ETag of each PUT the store answered with a 200.
        self.bytes_served = 0
        self.etags_given: dict[str, str] = {}
        self.closed = False
        self.answers: list[FakeResponse] = []
        # The stop line each request was handed, for the session's watchdog.
        self.stop_lines: list[StopLine] = []

    # --- the requests --------------------------------------------------------------------------------------------

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        stream: bool,
        *,
        stop_line: StopLine,
    ) -> FakeResponse:
        self.stop_lines.append(stop_line)
        self.requests.append(Request("GET", url, dict(headers), timeout, served_before=self.bytes_served))
        if url == METADATA_URL:
            return self._answer(self.metadata.pop(0) if self.metadata else json_answer(200, self.metadata_document))
        if url == MEDIA_URL:
            return self._answer(self.media.pop(0) if self.media else Serve(), headers)
        return self._answer(FakeResponse(404))

    def put(
        self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *, stop_line: StopLine
    ) -> FakeResponse:
        self.stop_lines.append(stop_line)
        scripted = self.puts.get(url, [])
        answer = scripted.pop(0) if scripted else Put()
        request = Request("PUT", url, dict(headers), timeout, served_before=self.bytes_served)
        self.requests.append(request)
        if isinstance(answer, Fail):
            raise answer.error
        if isinstance(answer, FakeResponse):
            return self._answer(answer)
        digest = hashlib.md5()
        seen = bytearray() if url == RECEIPT_URL else None
        try:
            for block in _blocks_of(data):
                digest.update(block)
                request.length += len(block)
                if seen is not None:
                    seen += block
                self.clock.advance(answer.seconds_per_block)
        except DeadlineReached:
            # The body stopped giving bytes: the request ends there, as urllib3 lets the error through.
            request.cut = True
            raise
        request.md5 = digest.hexdigest()
        request.body = bytes(seen) if seen is not None else None
        self.clock.advance(answer.seconds_after)
        headers_out = answer.headers if answer.headers is not None else {"ETag": f'"{request.md5}"'}
        if answer.status == 200 and answer.headers is None:
            self.etags_given[url] = request.md5
        return self._answer(FakeResponse(answer.status, headers_out))

    def close(self) -> None:
        self.closed = True

    # --- what was asked ------------------------------------------------------------------------------------------

    def of(self, method: str, url: str | None = None) -> list[Request]:
        return [r for r in self.requests if r.method == method and (url is None or r.url == url)]

    @property
    def part_puts(self) -> list[Request]:
        return [r for r in self.requests if r.method == "PUT" and r.url != RECEIPT_URL]

    @property
    def receipt_puts(self) -> list[Request]:
        return self.of("PUT", RECEIPT_URL)

    @property
    def media_gets(self) -> list[Request]:
        return self.of("GET", MEDIA_URL)

    def receipt(self) -> dict[str, Any]:
        (put,) = [r for r in self.receipt_puts if r.body is not None]
        assert put.body is not None
        return json.loads(put.body.decode("ascii"))

    # --- answering -----------------------------------------------------------------------------------------------

    def _answer(self, scripted: Any, headers: Mapping[str, str] | None = None) -> FakeResponse:
        if isinstance(scripted, Late):
            self.clock.advance(scripted.seconds)
            scripted = scripted.answer
        if isinstance(scripted, Fail):
            raise scripted.error
        if isinstance(scripted, Serve):
            scripted = self._serve(scripted, headers or {})
        self.answers.append(scripted)
        return scripted

    def _serve(self, serve: Serve, headers: Mapping[str, str]) -> FakeResponse:
        asked = headers.get("Range")
        start = int(asked.removeprefix("bytes=").removesuffix("-")) if asked else 0
        named_start = start if serve.offset is None else serve.offset
        status = serve.status if serve.status is not None else (206 if asked else 200)
        answer_headers = {"Content-Length": str(self.file.size - start + serve.extra)}
        if status == 206:
            answer_headers["Content-Range"] = f"bytes {named_start}-{self.file.size - 1}/{self.file.size}"

        def chunks(chunk_size: int) -> Iterator[bytes]:
            sent = 0
            for chunk in self._with_extra(start, chunk_size, serve.extra):
                if serve.break_after is not None and sent + len(chunk) > serve.break_after:
                    rest = chunk[: serve.break_after - sent]
                    self.bytes_served += len(rest)
                    yield rest
                    self.clock.advance(serve.seconds_before_break)
                    raise requests.exceptions.ChunkedEncodingError(_url_text())
                sent += len(chunk)
                self.bytes_served += len(chunk)
                yield chunk

        return FakeResponse(status, answer_headers, chunks=chunks)

    def _with_extra(self, start: int, chunk_size: int, extra: int) -> Iterator[bytes]:
        yield from self.file.chunks(start, self.file.size, chunk_size)
        if extra:
            yield from DriveFile(extra, seed=11).chunks(0, extra, chunk_size)
