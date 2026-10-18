"""Google Drive, as the import job asks it: the file's metadata again before the first write (the Hub's import
design, G-10), and the file's bytes (G-17).

Both requests go to one host of this module's own, built from the input's file id alone (valid by the schema:
letters, digits, `-` and `_`), the key in the `X-Goog-Api-Key` header and never in a URL, the link's resource key in
`X-Goog-Drive-Resource-Keys` when there is one, and the bytes asked as they are stored (`Accept-Encoding: identity`;
an answer in a content coding is refused); no redirect is followed (transport.RequestsSession). Every answer maps to
the metadata, the opened stream, a try to make again, or one code of the contract, as the Hub's own client maps
Google's answers (laravel/app/Services/Intake/Drive/HttpGoogleDriveFiles.php) with one addition of the worker's own:
the reasons of a 400 or a 403 that say the file is not there for the link (NOT_SHARED_REASONS, `link_not_shared`),
which the Hub's client, asking only for the metadata, answers as `provider_refused`. Nothing of Google's text is kept,
logged or returned, only the reasons this module compares with its own lists.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping
from urllib.parse import quote

from core.drive_import.failure import Code
from core.drive_import.input_check import ImportInput
from core.drive_import.transport import HttpResponse, read_small_answer

HOST = "www.googleapis.com"
FILES_URL = f"https://{HOST}/drive/v3/files/"
METADATA_QUERY = (
    "?supportsAllDrives=true"
    "&fields=size,md5Checksum,sha1Checksum,sha256Checksum,headRevisionId,trashed,capabilities/canDownload"
)
MEDIA_QUERY = "?alt=media&supportsAllDrives=true"
API_KEY_HEADER = "X-Goog-Api-Key"
RESOURCE_KEYS_HEADER = "X-Goog-Drive-Resource-Keys"
ACCEPT_ENCODING_HEADER = "Accept-Encoding"
CONTENT_ENCODING_HEADER = "Content-Encoding"

CONNECT_TIMEOUT_S = 10.0
METADATA_READ_TIMEOUT_S = 30.0
MEDIA_READ_TIMEOUT_S = 60.0

HTTP_OK = 200
HTTP_PARTIAL_CONTENT = 206
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_REQUEST_TIMEOUT = 408
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500

# The reasons of a 403 that are a quota or a rate limit (the Hub's QUOTA_REASONS).
QUOTA_REASONS = frozenset(
    {"downloadQuotaExceeded", "rateLimitExceeded", "userRateLimitExceeded", "dailyLimitExceeded", "quotaExceeded"}
)
# The reasons of a 400 or a 403 that are the key's own fault (the Hub's KEY_FAULT_REASONS); with the domain
# `usageLimits`, the reason `forbidden` is the key's fault too (a key not allowed to call the API).
KEY_FAULT_REASONS = frozenset({"keyInvalid", "accessNotConfigured"})
KEY_FAULT_FORBIDDEN_REASON = "forbidden"
KEY_FAULT_FORBIDDEN_DOMAIN = "usageLimits"
# The reasons of Google's ErrorInfo (an entry of `error.details`) that, on a 400 or a 403, are the key's own fault
# (the Hub's KEY_FAULT_DETAIL_REASONS).
KEY_FAULT_DETAIL_REASONS = frozenset(
    {
        "API_KEY_INVALID",
        "API_KEY_SERVICE_BLOCKED",
        "API_KEY_IP_ADDRESS_BLOCKED",
        "API_KEY_HTTP_REFERRER_BLOCKED",
        "API_KEY_ANDROID_APP_BLOCKED",
        "API_KEY_IOS_APP_BLOCKED",
        "SERVICE_DISABLED",
        "CONSUMER_INVALID",
        "CONSUMER_SUSPENDED",
    }
)
# The reasons of a 400 or a 403 that say the file is not there for whoever has the link, or may not be downloaded
# by them: the worker's own list (the Hub's client has none, and answers these as `provider_refused`). Any reason not
# named here (`cannotDownloadAbusiveFile` among them) is `provider_refused`.
NOT_SHARED_REASONS = frozenset(
    {"notFound", "insufficientFilePermissions", "cannotDownloadFile", "teamDriveMembershipRequired"}
)

# The fields of the metadata the job compares, as Google names them.
SIZE_FIELD = "size"
MD5_FIELD = "md5Checksum"
SHA1_FIELD = "sha1Checksum"
SHA256_FIELD = "sha256Checksum"
HEAD_REVISION_FIELD = "headRevisionId"
TRASHED_FIELD = "trashed"
CAPABILITIES_FIELD = "capabilities"
CAN_DOWNLOAD_FIELD = "canDownload"

# A size as Google writes it: a whole number of decimal digits.
DECIMAL_DIGITS = re.compile(r"[0-9]{1,19}")
# The start of the range a 206 answers with: `bytes <first>-<last>/<length or *>`.
CONTENT_RANGE = re.compile(r"bytes ([0-9]{1,19})-([0-9]{1,19})/([0-9]{1,19}|\*)")
IDENTITY = "identity"


class Retry:
    """A try that failed in a way another try may not: a transfer that failed, a 408 or a 5xx."""


RETRY = Retry()


def metadata_url(file_id: str) -> str:
    return FILES_URL + quote(file_id, safe="") + METADATA_QUERY


def media_url(file_id: str) -> str:
    return FILES_URL + quote(file_id, safe="") + MEDIA_QUERY


def request_headers(api_key: str, job: ImportInput) -> dict[str, str]:
    """The headers of both requests: the key, the link's resource key when it has one, and no content coding."""
    headers = {API_KEY_HEADER: api_key, ACCEPT_ENCODING_HEADER: IDENTITY}
    if job.resource_key is not None:
        headers[RESOURCE_KEYS_HEADER] = f"{job.file_id}/{job.resource_key}"
    return headers


def _in_content_coding(headers: Mapping[str, str]) -> bool:
    """Whether an answer's body comes in a content coding (anything but none or `identity`)."""
    coding = headers.get(CONTENT_ENCODING_HEADER)
    return coding is not None and coding.strip().lower() != IDENTITY


def is_retryable(status: int) -> bool:
    """A 408 or a 5xx: Google may answer another try."""
    return status == HTTP_REQUEST_TIMEOUT or status >= HTTP_SERVER_ERROR


def refusal_of(status: int, body: bytes | None) -> Code:
    """The code of an answer that is neither the expected one nor a try to make again, as the Hub maps it: a 404 is
    `file_removed`, a 429 `provider_quota`, a 400 or a 403 by Google's reasons (`body`, its error document, None
    when unread), and any other status (a redirect among them) `download_failed`."""
    if status == HTTP_NOT_FOUND:
        return Code.FILE_REMOVED
    if status == HTTP_TOO_MANY_REQUESTS:
        return Code.PROVIDER_QUOTA
    if status in (HTTP_BAD_REQUEST, HTTP_FORBIDDEN):
        return _refusal_by_reason(status, _error_of(body))
    return Code.DOWNLOAD_FAILED


def _refusal_by_reason(status: int, error: Mapping[str, Any]) -> Code:
    """A 400 or a 403, by the reason and the domain of Google's first error, either of which may be absent, and by
    the reasons of its ErrorInfo details (a disabled or blocked key may come with either status)."""
    reason, domain = _first_error_of(error)
    key_fault = (
        reason in KEY_FAULT_REASONS
        or (reason == KEY_FAULT_FORBIDDEN_REASON and domain == KEY_FAULT_FORBIDDEN_DOMAIN)
        or _details_name_a_key_fault(error)
    )
    if key_fault:
        return Code.NOT_CONFIGURED
    if status == HTTP_FORBIDDEN and reason in QUOTA_REASONS:
        return Code.PROVIDER_QUOTA
    if reason in NOT_SHARED_REASONS:
        return Code.LINK_NOT_SHARED
    return Code.PROVIDER_REFUSED


def _error_of(body: bytes | None) -> Mapping[str, Any]:
    """Google's error document `{"error": {...}}` as its inner object; empty when the body holds none."""
    document = _json_object(body)
    error = document.get("error") if document is not None else None
    return error if isinstance(error, dict) else {}


def _first_error_of(error: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The `reason` and `domain` of the first entry of `error.errors`, each None when absent or not text."""
    errors = error.get("errors")
    first = errors[0] if isinstance(errors, list) and errors and isinstance(errors[0], dict) else {}
    reason, domain = first.get("reason"), first.get("domain")
    return (reason if isinstance(reason, str) else None, domain if isinstance(domain, str) else None)


def _details_name_a_key_fault(error: Mapping[str, Any]) -> bool:
    details = error.get("details")
    if not isinstance(details, list):
        return False
    reasons = (detail.get("reason") for detail in details if isinstance(detail, dict))
    return any(isinstance(reason, str) and reason in KEY_FAULT_DETAIL_REASONS for reason in reasons)


def _json_object(body: bytes | None) -> dict[str, Any] | None:
    """`body` decoded as one JSON object, or None."""
    if body is None:
        return None
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        # Not one JSON object (a body nested deeper than the parser goes included): nothing of it is read.
        document = None
    return document if isinstance(document, dict) else None


# --- the metadata (G-10) -------------------------------------------------------------------------------------------


def metadata_answer(response: HttpResponse) -> dict[str, Any] | Code | Retry:
    """What one answer of the metadata request says: the metadata (a 200 whose body is one JSON object), a try to
    make again (a 408, a 5xx, a body that could not be read whole), or a code (a 200 in a content coding is
    `download_failed`). The caller closes the answer."""
    status = response.status_code
    if status == HTTP_OK:
        if _in_content_coding(response.headers):
            return Code.DOWNLOAD_FAILED
        body = read_small_answer(response)
        if body is None:
            return RETRY
        metadata = _json_object(body)
        return metadata if metadata is not None else Code.DOWNLOAD_FAILED
    if is_retryable(status):
        return RETRY
    return refusal_of(status, _error_body(status, response))


def change_at_provider(job: ImportInput, metadata: Mapping[str, Any]) -> Code | None:
    """Why the file Google describes now is not the one the Hub saw at the paste, or None when nothing differs.

    In the trash: `file_removed`. Its owner lets viewers no longer download it (`capabilities.canDownload` present
    and not true): `link_not_shared`. Its size, always, and each of the MD5, the SHA-1, the SHA-256 and the head
    revision the input carries (a value the input does not carry is not compared, however Google answers it now):
    any difference, a value Google no longer gives included, is `source_changed_at_provider`.
    """
    if metadata.get(TRASHED_FIELD, False) is not False:
        return Code.FILE_REMOVED
    capabilities = metadata.get(CAPABILITIES_FIELD)
    if isinstance(capabilities, dict) and CAN_DOWNLOAD_FIELD in capabilities:
        if capabilities[CAN_DOWNLOAD_FIELD] is not True:
            return Code.LINK_NOT_SHARED
    if _size_of(metadata.get(SIZE_FIELD)) != job.expected_size:
        return Code.SOURCE_CHANGED_AT_PROVIDER
    recorded_checksums = (
        (job.expected_md5, MD5_FIELD),
        (job.expected_sha1, SHA1_FIELD),
        (job.expected_sha256, SHA256_FIELD),
    )
    for expected, field in recorded_checksums:
        if expected is not None and _lower_text(metadata.get(field)) != expected:
            return Code.SOURCE_CHANGED_AT_PROVIDER
    revision = job.expected_head_revision_id
    if revision is not None and metadata.get(HEAD_REVISION_FIELD) != revision:
        return Code.SOURCE_CHANGED_AT_PROVIDER
    return None


def _size_of(value: Any) -> int | None:
    """Google's size (a string of digits, or a number) as an int; None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and DECIMAL_DIGITS.fullmatch(value):
        return int(value)
    return None


def _lower_text(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) else None


# --- the bytes (G-17) ----------------------------------------------------------------------------------------------


def media_headers(api_key: str, job: ImportInput, offset: int) -> dict[str, str]:
    """The download's headers: the key, the resource key and no content coding (request_headers()), and from `offset`
    on when it is not the start."""
    headers = request_headers(api_key, job)
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return headers


def media_answer(response: HttpResponse, offset: int) -> HttpResponse | Code | Retry:
    """What one answer of the download says: the stream to read (a 200 from the start; a 206 whose range starts at
    `offset` for a resumed download), a try to make again, or a code. A 200 or a 206 of another shape, and a body
    in a content coding, are `download_failed`. The caller closes the answer unless it is returned as the stream."""
    status = response.status_code
    starts_where_asked = (offset == 0 and status == HTTP_OK) or (
        offset > 0 and status == HTTP_PARTIAL_CONTENT and _range_start(response.headers) == offset
    )
    if starts_where_asked:
        return Code.DOWNLOAD_FAILED if _in_content_coding(response.headers) else response
    if status in (HTTP_OK, HTTP_PARTIAL_CONTENT):
        return Code.DOWNLOAD_FAILED
    if is_retryable(status):
        return RETRY
    return refusal_of(status, _error_body(status, response))


def _range_start(headers: Mapping[str, str]) -> int | None:
    value = headers.get("Content-Range")
    match = CONTENT_RANGE.fullmatch(value.strip()) if isinstance(value, str) else None
    return int(match.group(1)) if match is not None else None


def _error_body(status: int, response: HttpResponse) -> bytes | None:
    """The error document of a 400 or a 403, read whole within its limit; nothing is read of any other answer."""
    return read_small_answer(response) if status in (HTTP_BAD_REQUEST, HTTP_FORBIDDEN) else None
