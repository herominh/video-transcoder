"""The store, as the import job writes to it: one presigned PUT for each part of the upload, and one for the receipt
(the Hub's import design, G-18 and section 6).

Every PUT carries its length and `Content-MD5` (the base64 of the body's MD5, which the store checks), is bounded by
the stop line (its time-outs cut to what is left, its body stopping when it passes, its socket shut by the session's
watchdog when it comes), and is made again after a wait (0.5, 2 and 8 seconds, each only when it ends before the stop
line) on a 5xx, a 408, a 429 or a transfer that failed; any other answer but a 200 ends the job at once as
`upload_failed`, a redirect included (none is followed). Its answer's body is never read. Tries used up after the stop
line has passed are the stop line's: `deadline`.
A part's 200 must carry the store's ETag in the receipt's form: one pair of surrounding quotes taken off, then 1
to 128 printable ASCII characters with no space and no `"`; a missing or other one is `upload_failed`.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

from core.drive_import.clock import DeadlineReached, StopLine
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.transport import TRANSFER_ERRORS, BoundedBody, HttpResponse, HttpSession, close_quietly

CONNECT_TIMEOUT_S = 10.0
# A PUT's read time-out (the wait for the store's answer) grows with its body: at least a minute, and one second
# per MiB at most beyond it (64 s for a whole part).
MIN_READ_TIMEOUT_S = 60.0
READ_TIMEOUT_BYTES_PER_SECOND = 1_048_576
RETRY_WAITS_S = (0.5, 2.0, 8.0)

HTTP_OK = 200
HTTP_REQUEST_TIMEOUT = 408
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
RETRYABLE_STATUSES = frozenset({HTTP_REQUEST_TIMEOUT, HTTP_TOO_MANY_REQUESTS})

ETAG_HEADER = "ETag"
# A receipt's ETag (receipt.schema.json): printable ASCII but the space and `"`, 1 to 128 characters.
ETAG_TEXT = re.compile(r"[\x21\x23-\x7e]{1,128}")
QUOTE = '"'


@dataclass(frozen=True)
class _Answer:
    """What one PUT came to: the store's status and ETag, or no answer (`status` None), or the stop line."""

    status: int | None
    etag: str | None
    deadline: bool = False


def content_md5(payload: bytes | bytearray) -> str:
    """The `Content-MD5` of a body: the base64 of its MD5."""
    return base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode("ascii")


def etag_of(headers: Mapping[str, str]) -> str | None:
    """The store's ETag in the receipt's form, or None when the answer carries none or another."""
    value = headers.get(ETAG_HEADER)
    if not isinstance(value, str):
        return None
    if len(value) >= 2 and value.startswith(QUOTE) and value.endswith(QUOTE):
        value = value[1:-1]
    return value if ETAG_TEXT.fullmatch(value) else None


def put(
    http: HttpSession,
    url: str,
    payload: bytes | bytearray,
    headers: Mapping[str, str],
    stop_line: StopLine,
    stage: Stage,
    on_failed_try: Callable[[int], None],
) -> str | None:
    """PUT `payload` to `url` with `headers` and its length and Content-MD5, trying again as the module says.

    Returns the store's ETag in the receipt's form, None when its 200 carried none or another. Raises ImportFailed:
    `deadline` when the stop line passed before or during a try (or before a wait would end), `upload_failed`
    for an answer that is not to be tried again, or when the tries are used up. `on_failed_try` is called with the
    number of each try that failed but may be made again, after that try ended.
    """
    request_headers = {
        "Content-Length": str(len(payload)),
        "Content-MD5": content_md5(payload),
        **headers,
    }
    read_timeout_s = max(MIN_READ_TIMEOUT_S, len(payload) / READ_TIMEOUT_BYTES_PER_SECOND)
    for attempt in range(1, len(RETRY_WAITS_S) + 2):
        if attempt > 1:
            stop_line.wait(RETRY_WAITS_S[attempt - 2], stage)
        answer = _put_once(http, url, payload, request_headers, stop_line, stage, read_timeout_s)
        if answer.deadline:
            raise ImportFailed(Code.DEADLINE, stage)
        if answer.status == HTTP_OK:
            return answer.etag
        if answer.status is not None and not _retryable(answer.status):
            raise ImportFailed(Code.UPLOAD_FAILED, stage)
        on_failed_try(attempt)
    raise stop_line.failed(Code.UPLOAD_FAILED, stage)


def _put_once(
    http: HttpSession,
    url: str,
    payload: bytes | bytearray,
    headers: Mapping[str, str],
    stop_line: StopLine,
    stage: Stage,
    read_timeout_s: float,
) -> _Answer:
    timeout = stop_line.timeout(stage, CONNECT_TIMEOUT_S, read_timeout_s)
    response: HttpResponse | None = None
    reached = False
    try:
        response = http.put(url, BoundedBody(payload, stop_line), headers, timeout, stop_line=stop_line)
    except DeadlineReached:
        reached = True
    except TRANSFER_ERRORS:
        # No answer: the error's text names the URL, so nothing of it is kept.
        response = None
    if reached:
        return _Answer(None, None, deadline=True)
    if response is None:
        return _Answer(None, None)
    answer = _Answer(response.status_code, etag_of(response.headers))
    close_quietly(response)
    return answer


def _retryable(status: int) -> bool:
    return status in RETRYABLE_STATUSES or status >= HTTP_SERVER_ERROR
