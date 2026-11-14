"""The one reader of a path-style SigV4 presigned URL, shared by the acquire's A5 and the capabilities' K5.

The complete authority and the raw path are compared, never decoded or normalized; each query name and value
is decoded exactly once. Nothing here puts a URL in an exception or a log line.
"""

from __future__ import annotations

import re
from typing import Mapping, TYPE_CHECKING
from urllib.parse import unquote_plus

if TYPE_CHECKING:
    from .context import TrustedContext

MS_PER_SECOND = 1_000
SIGV4_MAX_SECONDS = 604_800
BUCKET = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
BAD_PERCENT = re.compile(r"%(?![0-9a-fA-F]{2})")
SIGV4_DATE = re.compile(r"[0-9]{8}T[0-9]{6}Z")
DECIMAL_SECONDS = re.compile(r"[1-9][0-9]{0,5}")


def split(url: str, host: str) -> tuple[str, str, dict[str, str]] | None:
    """The bucket, the raw key and the decoded query of an https URL of exactly `host`, or None."""
    if not url.startswith("https://") or any(c in url for c in "\\#[]"):
        return None
    authority, slash, rest = url[len("https://"):].partition("/")
    if authority != host or not slash:
        return None
    path, question, query = rest.partition("?")
    bucket, slash, key = path.partition("/")
    if not slash or not BUCKET.fullmatch(bucket) or "%" in path or not question or not query or BAD_PERCENT.search(query):
        return None
    params: dict[str, str] = {}
    for pair in query.split("&"):
        name, equals, value = pair.partition("=")
        if not equals or not name:
            return None
        name, value = unquote_plus(name, encoding="latin-1"), unquote_plus(value, encoding="latin-1")
        if name in params:
            return None
        params[name] = value
    return bucket, key, params


def signed_until_ms(params: Mapping[str, str], context: TrustedContext) -> int | None:
    """X-Amz-Date plus X-Amz-Expires in epoch milliseconds, or None.

    None when the date is no real YYYYMMDDTHHMMSSZ instant at or before now plus the skew, or the expiry is not
    canonical decimal seconds from 1 to 604,800.
    """
    from .semantic import parse_timestamp_ms

    date, seconds = params.get("X-Amz-Date", ""), params.get("X-Amz-Expires", "")
    if not SIGV4_DATE.fullmatch(date) or not DECIMAL_SECONDS.fullmatch(seconds):
        return None
    stamp = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[9:11]}:{date[11:13]}:{date[13:15]}.000Z"
    date_ms = parse_timestamp_ms(stamp)
    duration = int(seconds)
    if date_ms is None or date_ms > context.now_ms + context.skew_ms or duration > SIGV4_MAX_SECONDS:
        return None
    return date_ms + duration * MS_PER_SECOND
