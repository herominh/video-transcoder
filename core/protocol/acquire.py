"""Acquire semantics and path-style PUT authority; no URL is ever included in a failure."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from .acquire_urls import PART_SIZE, plan, urls
from .reasons import ReasonCode
from .registry import ContractRegistry

if TYPE_CHECKING:
    from .context import TrustedContext

REQUEST = "source.acquire.request"
GRANT = "source.acquire.granted"


def _snapshot(document: Mapping[str, Any], kind: str, context: TrustedContext, member: str) -> Any:
    return document[member] if kind == REQUEST else context.value("expect.acquire_" + member)


def destination(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    target = _snapshot(document, kind, context, "destination")
    video = document["identity"]["video_uuid"]
    stem = f"sources/{video}/{target['source_id']}/original."
    if target["source_id"] == video or target["object_key"] not in {stem + ext for ext in ("mp4", "mov", "avi", "mkv", "webm")}:
        return ReasonCode.ACQUIRE_DESTINATION_MISMATCH, "/destination/object_key" if kind == REQUEST else "/output/prefix"
    return None


def snapshot(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    for member in ("destination", "external", "plan", "limits"):
        if document[member] != context.value("expect.acquire_" + member):
            return ReasonCode.ACQUIRE_SNAPSHOT_MISMATCH, "/" + member
    if document["claim"]["expires_at"] != context.value("expect.acquire_expires_at"):
        return ReasonCode.ACQUIRE_SNAPSHOT_MISMATCH, "/claim/expires_at"
    return None


def prefix(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    target = context.value("expect.acquire_destination")
    if document["output"]["location_id"] != target["location_id"]:
        return ReasonCode.STORAGE_LOCATION_MISMATCH, "/output/location_id"
    if document["output"]["prefix"] != target["object_key"].rsplit("/", 1)[0]:
        return ReasonCode.ACQUIRE_DESTINATION_MISMATCH, "/output/prefix"
    return None


def result_parts(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    parts = document["parts"]
    if len(parts) != context.value("expect.acquire_plan")["parts_total"]:
        return ReasonCode.ACQUIRE_PARTS_MISMATCH, "/parts"
    for index, part in enumerate(parts):
        if part["n"] != index + 1:
            return ReasonCode.ACQUIRE_PARTS_MISMATCH, f"/parts/{index}/n"
    return None


def integrity(document: Mapping[str, Any], kind: str, context: TrustedContext, registry: ContractRegistry) -> tuple[ReasonCode, str] | None:
    expected = context.value("expect.acquire_external")["expected"]
    if document["bytes"] != expected["size"]:
        return ReasonCode.ACQUIRE_INTEGRITY_MISMATCH, "/bytes"
    checksum = "sha256" if expected["sha256"] is not None else "md5" if expected["md5"] is not None else "none"
    if document["checksum_verified"] != checksum:
        return ReasonCode.ACQUIRE_INTEGRITY_MISMATCH, "/checksum_verified"
    return None


CHECKS = {"A1": destination, "A2": snapshot, "A3": plan, "A4": prefix, "A5": urls,
          "A6": result_parts, "A7": integrity}
