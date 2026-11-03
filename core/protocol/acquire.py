"""Acquire semantics and path-style PUT authority; no URL is ever included in a failure."""

from __future__ import annotations

import re
from urllib.parse import unquote_plus

from .reasons import ReasonCode
from .registry import Role

PART_SIZE = 67_108_864
MS_PER_SECOND = 1_000
SIGV4_MAX_SECONDS = 604_800
BUCKET = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
BAD_PERCENT = re.compile(r"%(?![0-9a-fA-F]{2})")
SIGV4_DATE = re.compile(r"[0-9]{8}T[0-9]{6}Z")
DECIMAL_SECONDS = re.compile(r"[1-9][0-9]{0,5}")
REQUEST = "source.acquire.request"
GRANT = "source.acquire.granted"


def _snapshot(document, kind, context, member):
    return document[member] if kind == REQUEST else context.value("expect.acquire_" + member)


def destination(document, kind, context, registry):
    target = _snapshot(document, kind, context, "destination")
    video = document["identity"]["video_uuid"]
    stem = f"sources/{video}/{target['source_id']}/original."
    if target["source_id"] == video or target["object_key"] not in {stem + ext for ext in ("mp4", "mov", "avi", "mkv", "webm")}:
        return ReasonCode.ACQUIRE_DESTINATION_MISMATCH, "/destination/object_key" if kind == REQUEST else "/output/prefix"
    return None


def snapshot(document, kind, context, registry):
    for member in ("destination", "external", "plan", "limits"):
        if document[member] != context.value("expect.acquire_" + member):
            return ReasonCode.ACQUIRE_SNAPSHOT_MISMATCH, "/" + member
    if document["claim"]["expires_at"] != context.value("expect.acquire_expires_at"):
        return ReasonCode.ACQUIRE_SNAPSHOT_MISMATCH, "/claim/expires_at"
    return None


def plan(document, kind, context, registry):
    size = _snapshot(document, kind, context, "external")["expected"]["size"]
    parts = _snapshot(document, kind, context, "plan")["parts_total"]
    limit = _snapshot(document, kind, context, "limits")["max_source_bytes"]
    pointer = "/upload/parts" if kind == GRANT else "/parts"
    if parts != (size + PART_SIZE - 1) // PART_SIZE:
        return ReasonCode.ACQUIRE_PLAN_MISMATCH, "/plan/parts_total" if kind == REQUEST else pointer
    if size > limit:
        return ReasonCode.ACQUIRE_PLAN_MISMATCH, "/limits/max_source_bytes" if kind == REQUEST else pointer
    return None


def prefix(document, kind, context, registry):
    target = context.value("expect.acquire_destination")
    if document["output"]["location_id"] != target["location_id"]:
        return ReasonCode.STORAGE_LOCATION_MISMATCH, "/output/location_id"
    if document["output"]["prefix"] != target["object_key"].rsplit("/", 1)[0]:
        return ReasonCode.ACQUIRE_DESTINATION_MISMATCH, "/output/prefix"
    return None


def _split(url: str, host: str):
    # Compare the complete authority and raw path. Decode each query name/value exactly once.
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


def _expiry_valid(params, deadline_ms, context):
    from .semantic import parse_timestamp_ms

    date, seconds = params.get("X-Amz-Date", ""), params.get("X-Amz-Expires", "")
    if not SIGV4_DATE.fullmatch(date) or not DECIMAL_SECONDS.fullmatch(seconds):
        return False
    stamp = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[9:11]}:{date[11:13]}:{date[13:15]}.000Z"
    date_ms = parse_timestamp_ms(stamp)
    duration = int(seconds)
    return (date_ms is not None and date_ms <= context.now_ms + context.skew_ms
            and duration <= SIGV4_MAX_SECONDS and date_ms + duration * MS_PER_SECOND == deadline_ms
            and deadline_ms > context.now_ms)


def urls(document, kind, context, registry):
    from .semantic import parse_timestamp_ms

    parts = document["upload"]["parts"]
    total = context.value("expect.acquire_plan")["parts_total"]
    if len(parts) != total or any(part["n"] != index + 1 for index, part in enumerate(parts)):
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/parts"
    deadline = (parse_timestamp_ms(context.value("expect.acquire_expires_at"))
                + context.value("expect.acquire_limits")["max_wall_time_ms"]
                + (2 * registry.clock_skew_seconds + registry.limits["storage_call_margin_seconds"]) * MS_PER_SECOND)
    bucket = upload_id = None
    host = context.value("expect.acquire_storage_host")
    target = context.value("expect.acquire_destination")["object_key"]
    for index, part in enumerate(parts):
        parsed = _split(part["url"], host)
        if parsed is None:
            return ReasonCode.ACQUIRE_URL_INVALID, f"/upload/parts/{index}/url"
        named_bucket, key, params = parsed
        named_upload = params.get("uploadId")
        valid = (key == target and params.get("partNumber") == str(part["n"]) and bool(named_upload)
                 and (bucket is None or named_bucket == bucket) and (upload_id is None or named_upload == upload_id)
                 and _expiry_valid(params, deadline, context))
        if context.role == Role.HUB_SENDER.value:
            valid = valid and named_upload == context.value("expect.acquire_upload_id")
        if not valid:
            return ReasonCode.ACQUIRE_URL_INVALID, f"/upload/parts/{index}/url"
        bucket, upload_id = named_bucket, named_upload
    parsed = _split(document["upload"]["result_url"], host)
    result_key = f"{document['output']['prefix']}/acquire-result-{document['identity']['execution_id']}.json"
    if parsed is None:
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/result_url"
    named_bucket, key, params = parsed
    if (named_bucket != bucket or key != result_key or "uploadId" in params or "partNumber" in params
            or not _expiry_valid(params, deadline, context)):
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/result_url"
    return None


def result_parts(document, kind, context, registry):
    parts = document["parts"]
    if len(parts) != context.value("expect.acquire_plan")["parts_total"]:
        return ReasonCode.ACQUIRE_PARTS_MISMATCH, "/parts"
    for index, part in enumerate(parts):
        if part["n"] != index + 1:
            return ReasonCode.ACQUIRE_PARTS_MISMATCH, f"/parts/{index}/n"
    return None


def integrity(document, kind, context, registry):
    expected = context.value("expect.acquire_external")["expected"]
    if document["bytes"] != expected["size"]:
        return ReasonCode.ACQUIRE_INTEGRITY_MISMATCH, "/bytes"
    checksum = "sha256" if expected["sha256"] is not None else "md5" if expected["md5"] is not None else "none"
    if document["checksum_verified"] != checksum:
        return ReasonCode.ACQUIRE_INTEGRITY_MISMATCH, "/checksum_verified"
    return None


CHECKS = {"A1": destination, "A2": snapshot, "A3": plan, "A4": prefix, "A5": urls,
          "A6": result_parts, "A7": integrity}
