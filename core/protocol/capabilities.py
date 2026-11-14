"""The capability checks K1-K5 of protocol 2.2.0 (README section 8.5); no URL is ever included in a failure.

The document has passed its schema. An instant that is not real is S1's, so K4 compares only real instants.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING

from .presigned_urls import MS_PER_SECOND, signed_until_ms, split
from .reasons import ReasonCode
from .registry import ContractRegistry

if TYPE_CHECKING:
    from .context import TrustedContext

# The longest a source URL may be granted for, from the answer's sending.
SOURCE_MAX_MS = 86_400_000
SOURCE_SIGNED_HEADERS = "host"
# A put URL stores only the bytes its declared MD5 names.
PUT_SIGNED_HEADERS = "content-md5;host"
WHOLE_SECOND = "000"
# The query parameters the Hub's SigV4 query presign writes, the only ones a capability URL may hold.
SIGNED_QUERY_PARAMETERS = frozenset({
    "X-Amz-Algorithm", "X-Amz-Content-Sha256", "X-Amz-Credential", "X-Amz-Date",
    "X-Amz-Expires", "X-Amz-Security-Token", "X-Amz-SignedHeaders", "X-Amz-Signature",
})

Finding = Optional[tuple[ReasonCode, str]]


def _ms(value: Any) -> int | None:
    from .semantic import parse_timestamp_ms

    return parse_timestamp_ms(value)


def sequence(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if not document["seq"] > context.last_capability_seq:
        return ReasonCode.STALE_CAPABILITY_SEQUENCE, "/seq"
    return None


def unique_paths(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    seen: set[str] = set()
    for index, item in enumerate((document["put"] or {}).get("objects", ())):
        if item["path"] in seen:
            return ReasonCode.DUPLICATE_CAPABILITY_PATH, f"/put/objects/{index}/path"
        seen.add(item["path"])
    return None


def answers_the_request(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    request = context.value("expect.capability_request")
    put = document["put"]
    mismatch = ReasonCode.CAPABILITY_ANSWER_MISMATCH
    if document["seq"] != request["seq"]:
        return mismatch, "/seq"
    if document["stage"]["name"] != request["stage"]:
        return mismatch, "/stage/name"
    if (document["source"] is None) == request["source"]:
        return mismatch, "/source"
    if (put is None) != (request["put"] is None):
        return mismatch, "/put"
    if put is not None:
        paths = request["put"]["paths"]
        if len(put["objects"]) != len(paths):
            return mismatch, "/put/objects"
        for index, item in enumerate(put["objects"]):
            if item["path"] != paths[index]:
                return mismatch, f"/put/objects/{index}/path"
    return None


def _not_after(value: str, sent_ms: int) -> bool:
    at = _ms(value)
    return at is not None and at <= sent_ms


def _url_expiry_valid(value: str, sent_ms: int, longest_ms: int) -> bool:
    at = _ms(value)
    return at is None or (value[20:23] == WHOLE_SECOND and sent_ms < at <= sent_ms + longest_ms)


def expiries(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    sent_ms = _ms(document["sent_at"])
    if sent_ms is None:
        return None
    invalid = ReasonCode.CAPABILITY_EXPIRY_INVALID
    if _not_after(document["lease"]["expires_at"], sent_ms):
        return invalid, "/lease/expires_at"
    if _not_after(document["stage"]["ends_at"], sent_ms):
        return invalid, "/stage/ends_at"
    source = document["source"]
    if source is not None and not _url_expiry_valid(source["expires_at"], sent_ms, SOURCE_MAX_MS):
        return invalid, "/source/expires_at"
    put = document["put"]
    requested = context.value("expect.capability_request")["put"]
    if put is not None and requested is not None:
        if not _url_expiry_valid(put["expires_at"], sent_ms, requested["lifetime_seconds"] * MS_PER_SECOND):
            return invalid, "/put/expires_at"
    return None


def _url_valid(url: str, context: TrustedContext, bucket: str, key: str, expires_at: str, signed_headers: str) -> bool:
    parsed = split(url, context.value("expect.capability_storage_host"))
    if parsed is None:
        return False
    named_bucket, named_key, params = parsed
    expires_ms = _ms(expires_at)
    return (named_bucket == bucket and named_key == key and set(params) <= SIGNED_QUERY_PARAMETERS
            and params.get("X-Amz-SignedHeaders") == signed_headers
            and expires_ms is not None and signed_until_ms(params, context) == expires_ms)


def urls(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    invalid = ReasonCode.CAPABILITY_URL_INVALID
    source = document["source"]
    if source is not None:
        target = context.value("expect.capability_source")
        if not _url_valid(source["url"], context, target["bucket"], target["object_key"], source["expires_at"],
                          SOURCE_SIGNED_HEADERS):
            return invalid, "/source/url"
    put = document["put"]
    if put is not None:
        output = context.value("expect.capability_output")
        for index, item in enumerate(put["objects"]):
            if not _url_valid(item["url"], context, output["bucket"], f"{output['prefix']}/{item['path']}",
                              put["expires_at"], PUT_SIGNED_HEADERS):
                return invalid, f"/put/objects/{index}/url"
    return None


CHECKS = {"K1": sequence, "K2": unique_paths, "K3": answers_the_request, "K4": expiries, "K5": urls}
