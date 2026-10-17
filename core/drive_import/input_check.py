"""The import job's input, checked beyond its schema (contract import/v1, README section 3; the Hub's import design,
decision G-14) before the worker makes any request.

The schema (checked first, elsewhere) holds every member's shape; this module holds what a schema cannot say: that the
plan is the size's own, that every URL points at the one upload and the one receipt of one video in one bucket of the
store this endpoint is configured for, and that the deadline has not passed. A refusal says which rule, never what the
input held: no URL, id or key appears in its text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import SplitResult, parse_qsl, urlsplit

PART_SIZE = 67_108_864
MAX_PARTS = 160
EXTENSIONS: frozenset[str] = frozenset({"mp4", "mov", "avi", "mkv", "webm"})

# An S3 bucket name as every S3 service accepts it (lowercase letters, digits, dots and hyphens, 3 to 63 characters).
BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# A host name (no port, no user information): labels of letters, digits and hyphens, separated by dots.
HOST_PATTERN = re.compile(r"(?=.{1,253}\Z)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*")
# The writer deadline as the contract writes it: UTC, seconds, up to six digits of fraction.
DEADLINE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,6})?Z")
PRINTABLE_URL = re.compile(r"[!-~]+")
# Characters a presigned URL never holds: a backslash, a fragment, and the brackets of an IPv6 host.
NEVER_IN_A_URL = ("\\", "#", "[", "]")

PART_NUMBER = "partNumber"
UPLOAD_ID = "uploadId"
ORIGINALS_PREFIX = "original-videos"
ORIGINAL_STEM = "original."
RECEIPT_PREFIX = "import-receipt-"
RECEIPT_SUFFIX = ".json"


class InputRefused(Exception):
    """The input breaks one of the contract's rules beyond the schema; the text names the rule only."""


@dataclass(frozen=True)
class Part:
    number: int
    url: str


@dataclass(frozen=True)
class ImportInput:
    """What the worker acts on, once every rule held."""

    job: str
    file_id: str
    resource_key: str | None
    expected_size: int
    expected_md5: str | None
    expected_sha1: str | None
    expected_sha256: str | None
    expected_head_revision_id: str | None
    part_size: int
    parts: tuple[Part, ...]
    receipt_url: str
    writes_end_at: datetime


def storage_host_from(value: str | None) -> str:
    """The configured store's host (IMPORT_STORAGE_HOST): a bare host name, lowercased; anything else is a ValueError
    (the settings then name the variable as not usable, and every job answers `not_configured`)."""
    if value is None or not HOST_PATTERN.fullmatch(value.strip().lower()):
        raise ValueError("IMPORT_STORAGE_HOST must be a bare host name (no scheme, port, path or user information)")
    return value.strip().lower()


def check(document: Mapping[str, Any], storage_host: str, now: datetime) -> ImportInput:
    """The input $document, already valid against input.schema.json, checked against every other rule.

    $storage_host is the configured store's host (storage_host_from()); $now is an aware UTC time.
    """
    if now.tzinfo is None:
        raise ValueError("now must be an aware time")

    expected = document["expected"]
    plan = document["plan"]
    size = _integer(expected["size"])
    if _integer(plan["part_size"]) != PART_SIZE:
        raise InputRefused("the plan's part size is not the contract's")
    parts_total = _integer(plan["parts_total"])
    if parts_total != -(-size // PART_SIZE) or parts_total > MAX_PARTS:
        raise InputRefused("the plan's parts total is not the size's")

    numbers = [_integer(part["n"]) for part in document["parts"]]
    if sorted(numbers) != list(range(1, parts_total + 1)):
        raise InputRefused("the part numbers are not exactly one to the parts total, each once")

    deadline = _deadline_of(document["writes_end_at"])
    if deadline <= now:
        raise InputRefused("the writer deadline has passed")

    bucket: str | None = None
    video_uuid: str | None = None
    upload_id: str | None = None
    original: str | None = None
    parts: list[Part] = []
    for part in sorted(document["parts"], key=lambda item: _integer(item["n"])):
        number = _integer(part["n"])
        url_bucket, url_video, name, query = _split(part["url"], storage_host)
        bucket, video_uuid = _same(bucket, url_bucket, "bucket"), _same(video_uuid, url_video, "video")
        if not name.startswith(ORIGINAL_STEM) or name[len(ORIGINAL_STEM):] not in EXTENSIONS:
            raise InputRefused("a part URL does not name the video's original")
        original = _same(original, name, "original")
        if _only(query, PART_NUMBER) != str(number):
            raise InputRefused("a part URL's part number is not its own")
        upload_id = _same(upload_id, _only(query, UPLOAD_ID), "upload")
        parts.append(Part(number, part["url"]))

    receipt_bucket, receipt_video, receipt_name, receipt_query = _split(document["receipt_url"], storage_host)
    _same(bucket, receipt_bucket, "bucket")
    _same(video_uuid, receipt_video, "video")
    if receipt_name != RECEIPT_PREFIX + document["job"] + RECEIPT_SUFFIX:
        raise InputRefused("the receipt URL does not name this job's receipt")
    if any(name in (UPLOAD_ID, PART_NUMBER) for name, _ in receipt_query):
        raise InputRefused("the receipt URL names an upload")

    return ImportInput(
        job=document["job"],
        file_id=document["file_id"],
        resource_key=document["resource_key"],
        expected_size=size,
        expected_md5=expected["md5"],
        expected_sha1=expected["sha1"],
        expected_sha256=expected["sha256"],
        expected_head_revision_id=expected["head_revision_id"],
        part_size=PART_SIZE,
        parts=tuple(parts),
        receipt_url=document["receipt_url"],
        writes_end_at=deadline,
    )


def _integer(value: Any) -> int:
    """An integer member as the contract allows it: a JSON number with an integer value (15 or 15.0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise InputRefused("an integer member is not an integer")
    return int(value)


def _split(url: str, storage_host: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """The bucket, the video's uuid, the object's name and the query of a presigned path-style URL of the store.

    A refusal is raised once the parser's own error is no longer being handled, so that it carries nothing of that
    error (whose text can quote the URL) as its cause or its context.
    """
    # Brackets name an IPv6 host, which no store URL is, and make the URL parser raise with the bracket's content.
    if not PRINTABLE_URL.fullmatch(url) or any(character in url for character in NEVER_IN_A_URL):
        raise InputRefused("a URL holds a character a presigned URL never holds")
    try:
        split: SplitResult | None = urlsplit(url)
    except ValueError:
        split = None
    if split is None:
        raise InputRefused("a URL is not well formed")
    if split.scheme != "https":
        raise InputRefused("a URL is not https")
    # The whole authority, compared as written: no user information, no port, no other host.
    if split.netloc.lower() != storage_host or split.username is not None or split.port is not None:
        raise InputRefused("a URL does not point at the configured store")
    segments = split.path.split("/")
    # "", bucket, "original-videos", uuid, name: exactly, so no "..", empty or encoded segment slips in.
    if len(segments) != 5 or segments[0] != "" or "%" in split.path:
        raise InputRefused("a URL's path is not a video's object")
    _, bucket, prefix, video_uuid, name = segments
    if not BUCKET_PATTERN.fullmatch(bucket) or prefix != ORIGINALS_PREFIX or not UUID_PATTERN.fullmatch(video_uuid):
        raise InputRefused("a URL's path is not a video's object")
    if name in ("", ".", ".."):
        raise InputRefused("a URL's path is not a video's object")
    try:
        query: list[tuple[str, str]] | None = parse_qsl(split.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        query = None
    if query is None:
        raise InputRefused("a URL's query is not well formed")
    return bucket, video_uuid, name, query


def _only(query: list[tuple[str, str]], name: str) -> str:
    """The value of the one parameter $name of $query; none or several refuse."""
    values = [value for key, value in query if key == name]
    if len(values) != 1 or values[0] == "":
        raise InputRefused(f"a URL does not name exactly one {name}")
    return values[0]


def _same(seen: str | None, value: str, what: str) -> str:
    if seen is not None and seen != value:
        raise InputRefused(f"the URLs do not all name the same {what}")
    return value


def _deadline_of(value: str) -> datetime:
    match = DEADLINE_PATTERN.fullmatch(value)
    if match is None:
        raise InputRefused("the writer deadline is not a UTC time")
    year, month, day, hour, minute, second = (int(group) for group in match.groups()[:6])
    fraction = match.group(7)
    microsecond = int((fraction[1:] + "000000")[:6]) if fraction else 0
    try:
        deadline: datetime | None = datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
    except ValueError:
        deadline = None
    if deadline is None:
        raise InputRefused("the writer deadline is not a UTC time")
    return deadline
