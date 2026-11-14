"""Acquire part plans and path-style PUT authority, without transfer dependencies."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from .presigned_urls import MS_PER_SECOND, signed_until_ms, split
from .reasons import ReasonCode
from .registry import ContractRegistry, Role

if TYPE_CHECKING:
    from .context import TrustedContext

PART_SIZE = 67_108_864
REQUEST = "source.acquire.request"
GRANT = "source.acquire.granted"


def _snapshot(document: Mapping[str, Any], kind: str, context: TrustedContext, member: str) -> Any:
    return document[member] if kind == REQUEST else context.value("expect.acquire_" + member)


def plan(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    size = _snapshot(document, kind, context, "external")["expected"]["size"]
    parts = _snapshot(document, kind, context, "plan")["parts_total"]
    limit = _snapshot(document, kind, context, "limits")["max_source_bytes"]
    pointer = "/upload/parts" if kind == GRANT else "/parts"
    if parts != (size + PART_SIZE - 1) // PART_SIZE:
        return ReasonCode.ACQUIRE_PLAN_MISMATCH, "/plan/parts_total" if kind == REQUEST else pointer
    if size > limit:
        return ReasonCode.ACQUIRE_PLAN_MISMATCH, "/limits/max_source_bytes" if kind == REQUEST else pointer
    return None


def _expiry_valid(params: Mapping[str, str], deadline_ms: int, context: TrustedContext) -> bool:
    return signed_until_ms(params, context) == deadline_ms and deadline_ms > context.now_ms


def urls(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    from .semantic import parse_timestamp_ms

    parts = document["upload"]["parts"]
    total = context.value("expect.acquire_plan")["parts_total"]
    if len(parts) != total or any(part["n"] != index + 1 for index, part in enumerate(parts)):
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/parts"
    deadline = (parse_timestamp_ms(context.value("expect.acquire_expires_at"))
                + context.value("expect.acquire_limits")["max_wall_time_ms"]
                + (2 * registry.clock_skew_seconds + registry.limits["storage_call_margin_seconds"]) * MS_PER_SECOND)
    # The Hub pins the trusted location's bucket; the worker learns it from the first URL.
    bucket = context.value("expect.acquire_bucket") if context.role == Role.HUB_SENDER.value else None
    upload_id = None
    host = context.value("expect.acquire_storage_host")
    target = context.value("expect.acquire_destination")["object_key"]
    for index, part in enumerate(parts):
        parsed = split(part["url"], host)
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
    parsed = split(document["upload"]["result_url"], host)
    result_key = f"{document['output']['prefix']}/acquire-result-{document['identity']['execution_id']}.json"
    if parsed is None:
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/result_url"
    named_bucket, key, params = parsed
    if (named_bucket != bucket or key != result_key or "uploadId" in params or "partNumber" in params
            or not _expiry_valid(params, deadline, context)):
        return ReasonCode.ACQUIRE_URL_INVALID, "/upload/result_url"
    return None


