"""Layer L4: the semantic checks S1-S30 against the trusted context (standard library only).

The rule table below says, per (role, kind), which channels are allowed and which checks
run, in order; the first failing check wins. It also yields the context fields each
(role, kind) requires (a check that runs needs what it reads), which context.py enforces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

from .reasons import ReasonCode
from .registry import Channel, ContractRegistry, MessageKind, Role

if TYPE_CHECKING:
    from .context import TrustedContext

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_TIMESTAMP_SHAPE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z")
_MS_PER_SECOND = 1_000
_MS_PER_DAY = 86_400_000

KIND_MASTER_PLAYLIST = "hls_master_playlist"
KIND_MEDIA_PLAYLIST = "hls_media_playlist"
KIND_SEGMENT = "hls_segment"
KIND_THUMBNAIL = "thumbnail"
DIAGNOSTIC_THUMBNAIL_UNAVAILABLE = "thumbnail_unavailable"

Finding = Optional[tuple[ReasonCode, str]]
Document = Mapping[str, Any]


@dataclass(frozen=True)
class SemanticFailure:
    check: str
    reason: ReasonCode
    instance_path: str


def parse_timestamp_ms(value: object) -> int | None:
    """Epoch milliseconds of a contract timestamp, or None when it names no real calendar instant.

    Proleptic Gregorian calendar, UTC; year 0000 is not an instant.
    """
    if not isinstance(value, str) or _TIMESTAMP_SHAPE.fullmatch(value) is None:
        return None
    year, month, day = int(value[0:4]), int(value[5:7]), int(value[8:10])
    hour, minute, second, millisecond = int(value[11:13]), int(value[14:16]), int(value[17:19]), int(value[20:23])
    if year < 1:
        return None
    try:
        instant = datetime(year, month, day, hour, minute, second, millisecond * 1_000, tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = instant - _EPOCH
    return delta.days * _MS_PER_DAY + delta.seconds * _MS_PER_SECOND + delta.microseconds // 1_000


# --- the rule table -------------------------------------------------------------------------

_REQUEST = MessageKind.TRANSCODE_REQUEST.value
_PROGRESS = MessageKind.TRANSCODE_PROGRESS.value
_COMPLETED = MessageKind.TRANSCODE_RESULT_COMPLETED.value
_FAILED = MessageKind.TRANSCODE_RESULT_FAILED.value
_HUB_ERROR = MessageKind.HUB_ERROR.value
_MANIFEST = MessageKind.GENERATION_MANIFEST.value
_CLAIM = MessageKind.TRANSCODE_CLAIM.value
_GRANT = MessageKind.TRANSCODE_CLAIM_GRANTED.value
_UNCLAIMED = MessageKind.TRANSCODE_UNCLAIMED.value

_RESULT_CHANNELS = (Channel.PUSH.value, Channel.POLL.value)
# A claim and its grant are one HTTP exchange: neither is ever read from provider output.
_CLAIM_CHANNELS = (Channel.PUSH.value,)
# A claim and an unclaimed report name a dispatch, never an execution.
_DISPATCH_ONLY_CHECKS = ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14")
_RECEIVER_ROWS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    _PROGRESS: (
        _RESULT_CHANNELS,
        ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S22"),
    ),
    _COMPLETED: (
        _RESULT_CHANNELS,
        ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
         "S18", "S19", "S21"),
    ),
    _FAILED: (
        _RESULT_CHANNELS,
        ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17"),
    ),
    _MANIFEST: (
        (Channel.STORAGE.value,),
        ("S1", "S7", "S8", "S9", "S13", "S14", "S15", "S17", "S19", "S20", "S21", "S23", "S24", "S25", "S26",
         "S27", "S28"),
    ),
    _CLAIM: (_CLAIM_CHANNELS, _DISPATCH_ONLY_CHECKS),
    _UNCLAIMED: (_RESULT_CHANNELS, _DISPATCH_ONLY_CHECKS),
}

# (role, kind) -> (allowed channels, checks in evaluation order). worker_sender mirrors hub_receiver.
RULES: Mapping[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {
    (Role.WORKER_RECEIVER.value, _REQUEST): ((Channel.DISPATCH.value,), ("S1", "S2", "S3", "S4", "S6", "S7")),
    (Role.WORKER_RECEIVER.value, _HUB_ERROR): ((Channel.PUSH.value,), ("S1",)),
    # The worker learns the execution id and its fence from the grant, so it compares neither (no S15, no S16).
    (Role.WORKER_RECEIVER.value, _GRANT): (
        _CLAIM_CHANNELS,
        ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S18", "S21", "S29",
         "S30"),
    ),
    (Role.HUB_SENDER.value, _REQUEST): (
        (Channel.DISPATCH.value,),
        ("S1", "S2", "S3", "S4", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S17", "S18", "S19",
         "S20"),
    ),
    (Role.HUB_SENDER.value, _GRANT): (
        _CLAIM_CHANNELS,
        ("S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S18",
         "S21", "S29", "S30"),
    ),
    **{(Role.HUB_RECEIVER.value, kind): row for kind, row in _RECEIVER_ROWS.items()},
    **{(Role.WORKER_SENDER.value, kind): row for kind, row in _RECEIVER_ROWS.items()},
}

_CHECK_CONTEXT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "S2": ("expected_audience",),
    "S3": ("known_key_ids",),
    "S4": ("now", "clock_skew_s"),
    "S5": ("now", "clock_skew_s"),
    "S6": ("now", "clock_skew_s"),
    "S8": ("expect.org_uuid",),
    "S9": ("expect.video_uuid",),
    "S10": ("expect.owning_cell_id",),
    "S11": ("expect.placement_epoch",),
    "S12": ("expect.lifecycle_revision",),
    "S13": ("expect.attempt_id",),
    "S14": ("expect.dispatch_id",),
    "S15": ("expect.execution_id",),
    "S16": ("expect.execution_fence",),
    "S17": ("expect.source_id",),
    "S19": ("expect.profile",),
    "S20": ("expect.encryption",),
    "S22": ("last_event_seq",),
    "S27": ("expect.renditions",),
    "S28": ("expect.max_artifact_count",),
    "S29": ("expect.runtime_id",),
}
_S18_CONTEXT_FIELDS: Mapping[str, tuple[str, ...]] = {
    _REQUEST: ("expect.source_location_id", "expect.output_location_id"),
    _COMPLETED: ("expect.output_location_id",),
    _GRANT: ("expect.output_location_id",),
}


def supported_channels(role: str, kind: str) -> tuple[str, ...] | None:
    """Channels on which `role` may validate `kind`; None when the pair is not part of the contract."""
    row = RULES.get((role, kind))
    return None if row is None else row[0]


def applicable_checks(role: str, kind: str, channel: str) -> tuple[str, ...]:
    """Checks that run for (role, kind) on `channel`, in evaluation order (S5 runs on push only)."""
    row = RULES.get((role, kind))
    if row is None:
        raise KeyError(f"no rule set for role {role!r} and kind {kind!r}")
    return tuple(check for check in row[1] if check != "S5" or channel == Channel.PUSH.value)


def required_context_fields(role: str, kind: str) -> tuple[str, ...]:
    """Context fields (dotted for expect.*) that (role, kind) requires: L2's version list plus what each check reads.

    accepted_message_kinds is not listed: the context schema requires it of every context.
    """
    row = RULES.get((role, kind))
    if row is None:
        raise KeyError(f"no rule set for role {role!r} and kind {kind!r}")
    versions_field = "accepted_manifest_versions" if kind == _MANIFEST else "accepted_protocol_versions"
    fields: list[str] = [versions_field]
    for check in row[1]:
        check_fields = _S18_CONTEXT_FIELDS.get(kind, ()) if check == "S18" else _CHECK_CONTEXT_FIELDS.get(check, ())
        fields.extend(field for field in check_fields if field not in fields)
    return tuple(fields)


# --- helpers --------------------------------------------------------------------------------


def _identity(document: Mapping[str, Any]) -> Mapping[str, Any]:
    return document["identity"]


def _source_reference(document: Mapping[str, Any], kind: str) -> tuple[str, Any] | None:
    """Pointer and value of the message's source id; None when the kind (or an optional block) has none."""
    if kind == _MANIFEST:
        return "/identity/source_id", _identity(document)["source_id"]
    source = document.get("source")
    if not isinstance(source, Mapping):
        return None
    return "/source/source_id", source["source_id"]


def _runtime_reference(document: Mapping[str, Any], kind: str) -> tuple[str, Any] | None:
    """Pointer and value of the message's runtime id; None when the kind carries none."""
    if kind == _CLAIM:
        return "/claim/runtime_id", document["claim"]["runtime_id"]
    if kind in (_GRANT, _UNCLAIMED):
        return "/runtime_id", document["runtime_id"]
    return None


def _timestamps(document: Mapping[str, Any], kind: str) -> Iterator[tuple[str, Any]]:
    if kind != _MANIFEST:
        yield "/sent_at", document["sent_at"]
    if kind == _REQUEST:
        yield "/claim/expires_at", document["claim"]["expires_at"]
    if kind == _MANIFEST:
        yield "/created_at", document["created_at"]


def _count(artifacts: list[Mapping[str, Any]], artifact_kind: str, rendition: str | None = None) -> int:
    return sum(
        1
        for artifact in artifacts
        if artifact["kind"] == artifact_kind and (rendition is None or artifact.get("rendition") == rendition)
    )


# --- the checks -----------------------------------------------------------------------------

CheckFunction = Callable[[Mapping[str, Any], str, "TrustedContext", ContractRegistry], Finding]


def _s1_timestamps(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    for pointer, value in _timestamps(document, kind):
        if parse_timestamp_ms(value) is None:
            return ReasonCode.INVALID_TIMESTAMP, pointer
    return None


def _s2_audience(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if document["audience"] != context.expected_audience:
        return ReasonCode.AUDIENCE_MISMATCH, "/audience"
    return None


def _s3_key_id(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if document["key_id"] not in context.known_key_ids:
        return ReasonCode.UNKNOWN_KEY_ID, "/key_id"
    return None


def _s4_not_in_future(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    sent_ms = parse_timestamp_ms(document["sent_at"])
    if sent_ms is not None and sent_ms > context.now_ms + context.skew_ms:
        return ReasonCode.TIMESTAMP_IN_FUTURE, "/sent_at"
    return None


def _s5_within_tolerance(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    sent_ms = parse_timestamp_ms(document["sent_at"])
    if sent_ms is not None and sent_ms < context.now_ms - context.skew_ms:
        return ReasonCode.TIMESTAMP_OUT_OF_TOLERANCE, "/sent_at"
    return None


def _s6_dispatch_live(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    sent_ms = parse_timestamp_ms(document["sent_at"])
    expires_ms = parse_timestamp_ms(document["claim"]["expires_at"])
    if sent_ms is None or expires_ms is None:
        return None
    if expires_ms <= sent_ms or context.now_ms > expires_ms + context.skew_ms:
        return ReasonCode.DISPATCH_EXPIRED, "/claim/expires_at"
    return None


def _s7_no_alias_of_video(
    document: Document, kind: str, context: TrustedContext, registry: ContractRegistry
) -> Finding:
    identity = _identity(document)
    video_uuid = identity["video_uuid"]
    candidates: list[tuple[str, Any]] = [
        ("/identity/attempt_id", identity.get("attempt_id")),
        ("/identity/dispatch_id", identity.get("dispatch_id")),
        ("/identity/execution_id", identity.get("execution_id")),
    ]
    source_reference = _source_reference(document, kind)
    if source_reference is not None:
        candidates.append(source_reference)
    candidates.append(("/identity/org_uuid", identity.get("org_uuid")))
    runtime_reference = _runtime_reference(document, kind)
    if runtime_reference is not None:
        candidates.append(runtime_reference)
    for pointer, value in candidates:
        if value is not None and value == video_uuid:
            return ReasonCode.IDENTITY_ALIASES_VIDEO_UUID, pointer
    return None


def _equal_identity(field: str, reason: ReasonCode) -> CheckFunction:
    def check(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
        if _identity(document)[field] != context.value(f"expect.{field}"):
            return reason, f"/identity/{field}"
        return None

    return check


def _fenced_identity(field: str, lower: ReasonCode, higher: ReasonCode) -> CheckFunction:
    def check(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
        received = _identity(document)[field]
        recorded = context.value(f"expect.{field}")
        if received < recorded:
            return lower, f"/identity/{field}"
        if received > recorded:
            return higher, f"/identity/{field}"
        return None

    return check


def _s17_source(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    source_reference = _source_reference(document, kind)
    if source_reference is not None and source_reference[1] != context.value("expect.source_id"):
        return ReasonCode.SOURCE_MISMATCH, source_reference[0]
    return None


def _s18_locations(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if kind == _REQUEST:
        comparisons = [
            ("/source/location_id", document["source"]["location_id"], "expect.source_location_id"),
            ("/output/location_id", document["output"]["location_id"], "expect.output_location_id"),
        ]
    elif kind == _GRANT:
        comparisons = [("/output/location_id", document["output"]["location_id"], "expect.output_location_id")]
    else:
        comparisons = [("/manifest/location_id", document["manifest"]["location_id"], "expect.output_location_id")]
    for pointer, received, recorded_field in comparisons:
        if received != context.value(recorded_field):
            return ReasonCode.STORAGE_LOCATION_MISMATCH, pointer
    return None


def _s19_profile(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    received = document["profile"]
    recorded = context.value("expect.profile")
    if any(received[part] != recorded[part] for part in ("id", "version", "sha256")):
        return ReasonCode.PROFILE_MISMATCH, "/profile"
    return None


def _s20_encryption(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    received = document["encryption"]
    recorded = context.value("expect.encryption")
    if received["mode"] != recorded["mode"] or received.get("media_key_id") != recorded.get("media_key_id"):
        return ReasonCode.ENCRYPTION_MISMATCH, "/encryption"
    return None


def _s21_generation_is_execution(
    document: Document, kind: str, context: TrustedContext, registry: ContractRegistry
) -> Finding:
    if document["generation_id"] != _identity(document)["execution_id"]:
        return ReasonCode.GENERATION_EXECUTION_MISMATCH, "/generation_id"
    return None


def _s22_event_sequence(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if not document["event_seq"] > context.last_event_seq:
        return ReasonCode.STALE_EVENT_SEQUENCE, "/event_seq"
    return None


def _s23_unique_paths(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    seen: set[str] = set()
    for index, artifact in enumerate(document["artifacts"]):
        if artifact["path"] in seen:
            return ReasonCode.DUPLICATE_ARTIFACT_PATH, f"/artifacts/{index}/path"
        seen.add(artifact["path"])
    return None


def _s24_inventory(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    artifacts = document["artifacts"]
    renditions = document["renditions"]
    mismatch = ReasonCode.ARTIFACT_INVENTORY_MISMATCH
    if document["artifact_count"] != len(artifacts):
        return mismatch, "/artifact_count"
    if document["total_bytes"] != sum(artifact["size_bytes"] for artifact in artifacts):
        return mismatch, "/total_bytes"
    if document["media"]["rendition_count"] != len(renditions):
        return mismatch, "/media/rendition_count"
    seen_names: set[str] = set()
    for index, rendition in enumerate(renditions):
        if rendition["name"] in seen_names:
            return mismatch, f"/renditions/{index}/name"
        seen_names.add(rendition["name"])
    if _count(artifacts, KIND_MASTER_PLAYLIST) != 1:
        return mismatch, "/artifacts"
    for index, rendition in enumerate(renditions):
        playlists = _count(artifacts, KIND_MEDIA_PLAYLIST, rendition["name"])
        segments = _count(artifacts, KIND_SEGMENT, rendition["name"])
        if playlists != 1 or segments < 1:
            return mismatch, f"/renditions/{index}"
    return None


def _s25_references(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    artifacts = document["artifacts"]
    by_path = {artifact["path"]: artifact for artifact in artifacts}
    dangling = ReasonCode.DANGLING_REFERENCE
    master = by_path.get(document["master_playlist_path"])
    if master is None or master["kind"] != KIND_MASTER_PLAYLIST:
        return dangling, "/master_playlist_path"
    for index, rendition in enumerate(document["renditions"]):
        playlist = by_path.get(rendition["playlist_path"])
        names_playlist = playlist is not None and playlist["kind"] == KIND_MEDIA_PLAYLIST
        if not names_playlist or playlist.get("rendition") != rendition["name"]:
            return dangling, f"/renditions/{index}/playlist_path"
    thumbnail_path = document["thumbnail_path"]
    if thumbnail_path is not None:
        thumbnail = by_path.get(thumbnail_path)
        if thumbnail is None or thumbnail["kind"] != KIND_THUMBNAIL:
            return dangling, "/thumbnail_path"
    rendition_names = {rendition["name"] for rendition in document["renditions"]}
    for index, artifact in enumerate(artifacts):
        if "rendition" in artifact and artifact["rendition"] not in rendition_names:
            return dangling, f"/artifacts/{index}/rendition"
    return None


def _s26_thumbnail(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    thumbnails = _count(document["artifacts"], KIND_THUMBNAIL)
    explained = any(
        diagnostic["code"] == DIAGNOSTIC_THUMBNAIL_UNAVAILABLE for diagnostic in document.get("diagnostics", [])
    )
    if document["thumbnail_path"] is None:
        consistent = explained and thumbnails == 0
    else:
        consistent = thumbnails == 1
    return None if consistent else (ReasonCode.THUMBNAIL_DIAGNOSTIC_MISSING, "/thumbnail_path")


def _s27_requested(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    requested = set(context.value("expect.renditions"))
    for index, rendition in enumerate(document["renditions"]):
        if rendition["name"] not in requested:
            return ReasonCode.RENDITION_NOT_REQUESTED, f"/renditions/{index}/name"
    return None


def _s28_limits(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if document["artifact_count"] > context.value("expect.max_artifact_count"):
        return ReasonCode.LIMIT_EXCEEDED, "/artifact_count"
    for index, rendition in enumerate(document["renditions"]):
        if _count(document["artifacts"], KIND_SEGMENT, rendition["name"]) > registry.max_segments_per_rendition:
            return ReasonCode.LIMIT_EXCEEDED, f"/renditions/{index}"
    return None


def _s29_runtime(document: Document, kind: str, context: TrustedContext, registry: ContractRegistry) -> Finding:
    if document["runtime_id"] != context.value("expect.runtime_id"):
        return ReasonCode.RUNTIME_MISMATCH, "/runtime_id"
    return None


def _s30_prefix_ends_in_generation(
    document: Document, kind: str, context: TrustedContext, registry: ContractRegistry
) -> Finding:
    """The last segment of the granted prefix is the generation; a prefix without a `/` is its own last segment."""
    last_segment = document["output"]["prefix"].rsplit("/", 1)[-1]
    if last_segment != document["generation_id"]:
        return ReasonCode.GENERATION_PREFIX_MISMATCH, "/output/prefix"
    return None


CHECKS: Mapping[str, CheckFunction] = {
    "S1": _s1_timestamps,
    "S2": _s2_audience,
    "S3": _s3_key_id,
    "S4": _s4_not_in_future,
    "S5": _s5_within_tolerance,
    "S6": _s6_dispatch_live,
    "S7": _s7_no_alias_of_video,
    "S8": _equal_identity("org_uuid", ReasonCode.ORG_MISMATCH),
    "S9": _equal_identity("video_uuid", ReasonCode.VIDEO_MISMATCH),
    "S10": _equal_identity("owning_cell_id", ReasonCode.CELL_MISMATCH),
    "S11": _fenced_identity("placement_epoch", ReasonCode.STALE_PLACEMENT_EPOCH, ReasonCode.PLACEMENT_EPOCH_AHEAD),
    "S12": _fenced_identity(
        "lifecycle_revision", ReasonCode.STALE_LIFECYCLE_REVISION, ReasonCode.LIFECYCLE_REVISION_AHEAD
    ),
    "S13": _equal_identity("attempt_id", ReasonCode.ATTEMPT_MISMATCH),
    "S14": _equal_identity("dispatch_id", ReasonCode.DISPATCH_MISMATCH),
    "S15": _equal_identity("execution_id", ReasonCode.EXECUTION_MISMATCH),
    "S16": _fenced_identity("execution_fence", ReasonCode.STALE_EXECUTION_FENCE, ReasonCode.EXECUTION_MISMATCH),
    "S17": _s17_source,
    "S18": _s18_locations,
    "S19": _s19_profile,
    "S20": _s20_encryption,
    "S21": _s21_generation_is_execution,
    "S22": _s22_event_sequence,
    "S23": _s23_unique_paths,
    "S24": _s24_inventory,
    "S25": _s25_references,
    "S26": _s26_thumbnail,
    "S27": _s27_requested,
    "S28": _s28_limits,
    "S29": _s29_runtime,
    "S30": _s30_prefix_ends_in_generation,
}


def failures(
    kind: MessageKind | str, document: Mapping[str, Any], context: TrustedContext, registry: ContractRegistry
) -> list[SemanticFailure]:
    """Every failing check, in evaluation order (one finding per check). The document must have passed L3."""
    kind_value = MessageKind(kind).value
    found: list[SemanticFailure] = []
    for check in applicable_checks(context.role, kind_value, context.channel):
        finding = CHECKS[check](document, kind_value, context, registry)
        if finding is not None:
            found.append(SemanticFailure(check=check, reason=finding[0], instance_path=finding[1]))
    return found


def first_failure(
    kind: MessageKind | str, document: Mapping[str, Any], context: TrustedContext, registry: ContractRegistry
) -> SemanticFailure | None:
    """The first failing check (the L4 verdict), or None when every applicable check passes."""
    kind_value = MessageKind(kind).value
    for check in applicable_checks(context.role, kind_value, context.channel):
        finding = CHECKS[check](document, kind_value, context, registry)
        if finding is not None:
            return SemanticFailure(check=check, reason=finding[0], instance_path=finding[1])
    return None
