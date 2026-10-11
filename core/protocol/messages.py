"""Build, encode and seal what the worker sends under contract v2, and the contexts it validates with.

Builders are pure functions: every input is an explicit keyword argument, the identities and the other
structured parts are typed values, and each builder returns the message as a dict in its schema's key
order, envelope first (`audience` is `hub:` plus the identity's `owning_cell_id`). Every typed value and
every finished message is checked against the contract's own schema definitions, so a wrong value is a
wiring bug: everything a schema judges raises ValueError naming the pointer, and a typed part, a moment,
`ext` or a diagnostic of the wrong Python type raises TypeError naming the parameter; neither shows the
value. The semantic checks belong to the self-check: pipeline.validate(encode(message),
worker_sender_context(...)). No builder takes or returns key material: the claim grant, the one kind
that carries a key, is received, never built. The generation manifest's own context (channel `storage`)
is not built here: the job runner that writes the manifest builds it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping

from core.failure import Diagnostic, Failure

from . import schema, signing
from .context import TrustedContext
from .registry import Channel, MessageKind, Role, load_registry
from .settings import WorkerSettings

HUB_AUDIENCE_PREFIX = "hub:"
CONTEXT_VERSION = 1
_MICROSECONDS_PER_MILLISECOND = 1_000
_SCHEMAS = schema.SCHEMA_BASE_URI + "schemas/"
_COMMON = _SCHEMAS + "common.schema.json#/$defs/"


class MessageTooLarge(ValueError):
    """The encoded message is above its kind's limits.json byte limit."""

    def __init__(self, kind: str, size: int, limit: int) -> None:
        super().__init__(f"an encoded {kind} of {size} bytes is above its limit of {limit} bytes")
        self.kind = kind
        self.size = size
        self.limit = limit


def _refuse_floats(document: Any, what: str) -> None:
    """The writer convention: every number is an integer literal (a float, even 7.0, is refused)."""
    if isinstance(document, float):
        raise ValueError(f"{what} holds a float; contract numbers are integers")
    if isinstance(document, Mapping):
        for value in document.values():
            _refuse_floats(value, what)
    elif isinstance(document, (list, tuple)):
        for value in document:
            _refuse_floats(value, what)


def _require_schema(schema_uri: str, document: Any, what: str) -> None:
    _refuse_floats(document, what)
    pointers = schema.error_pointers_for_schema({"$ref": schema_uri}, document)
    if pointers:
        raise ValueError(f"{what} violates {schema_uri.removeprefix(_SCHEMAS)} at {pointers[0]!r}")


def _require_type(name: str, value: object, expected: type) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}")


class _ContractValue:
    """A frozen value mirroring one contract definition; it refuses to exist unless it satisfies it."""

    SCHEMA: ClassVar[str]

    def __post_init__(self) -> None:
        _require_schema(self.SCHEMA, self.to_dict(), type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        """The definition's object, members in its schema order (the field order)."""
        return {item.name: getattr(self, item.name) for item in fields(self)}  # type: ignore[arg-type]


@dataclass(frozen=True)
class DispatchIdentity(_ContractValue):
    """`identity_dispatch`: the identity of a request, a claim and an unclaimed report."""

    SCHEMA: ClassVar[str] = _COMMON + "identity_dispatch"

    org_uuid: str
    video_uuid: str
    owning_cell_id: str
    placement_epoch: int
    lifecycle_revision: int
    attempt_id: str
    dispatch_id: str


@dataclass(frozen=True)
class ExecutionIdentity(_ContractValue):
    """`identity_execution`: the dispatch identity plus the granted execution, on progress and results."""

    SCHEMA: ClassVar[str] = _COMMON + "identity_execution"

    org_uuid: str
    video_uuid: str
    owning_cell_id: str
    placement_epoch: int
    lifecycle_revision: int
    attempt_id: str
    dispatch_id: str
    execution_id: str
    execution_fence: int

    @classmethod
    def of(cls, dispatch: DispatchIdentity, *, execution_id: str, execution_fence: int) -> ExecutionIdentity:
        """The identity of the execution a grant gave this dispatch."""
        _require_type("dispatch", dispatch, DispatchIdentity)
        return cls(**dispatch.to_dict(), execution_id=execution_id, execution_fence=execution_fence)


@dataclass(frozen=True)
class Profile(_ContractValue):
    """`profile`: the media profile's id, version and SHA-256."""

    SCHEMA: ClassVar[str] = _COMMON + "profile"

    id: str
    version: int
    sha256: str


@dataclass(frozen=True)
class Encryption(_ContractValue):
    """`encryption`: the mode and, for `aes-128` only, the media key's id (never the key)."""

    SCHEMA: ClassVar[str] = _COMMON + "encryption"

    mode: str
    media_key_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        encryption: dict[str, Any] = {"mode": self.mode}
        if self.media_key_id is not None:
            encryption["media_key_id"] = self.media_key_id
        return encryption


@dataclass(frozen=True)
class SourceRead(_ContractValue):
    """`source_read`: what the worker read of the source."""

    SCHEMA: ClassVar[str] = _COMMON + "source_read"

    source_id: str
    bytes_read: int
    etag_observed: str


@dataclass(frozen=True)
class Media(_ContractValue):
    """`media`: the facts of the produced generation."""

    SCHEMA: ClassVar[str] = _COMMON + "media"

    duration_ms: int
    max_width: int
    max_height: int
    has_audio: bool
    rendition_count: int


@dataclass(frozen=True)
class Encoder(_ContractValue):
    """`encoder`: the image, ffmpeg and encoder that produced the generation."""

    SCHEMA: ClassVar[str] = _COMMON + "encoder"

    image_digest: str
    ffmpeg_version: str
    video_encoder: str
    preset: str


@dataclass(frozen=True)
class ManifestLocator(_ContractValue):
    """A completed result's `manifest`: where the generation manifest is and what it lists."""

    SCHEMA: ClassVar[str] = _SCHEMAS + "transcode-result-completed.schema.json#/properties/manifest"

    location_id: str
    path: str
    sha256: str
    size_bytes: int
    artifact_count: int
    total_bytes: int


@dataclass(frozen=True)
class Counters(_ContractValue):
    """A progress message's `counters`."""

    SCHEMA: ClassVar[str] = _SCHEMAS + "transcode-progress.schema.json#/properties/counters"

    renditions_total: int
    renditions_done: int
    bytes_downloaded: int
    bytes_uploaded: int


# --- encoding and sealing ---------------------------------------------------------------------


def timestamp(moment: datetime) -> str:
    """A contract timestamp, `YYYY-MM-DDTHH:MM:SS.mmmZ` in UTC (sub-millisecond digits dropped); naive refused."""
    _require_type("moment", moment, datetime)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("a contract timestamp needs an aware datetime")
    try:
        utc = moment.astimezone(timezone.utc)
    except OverflowError:
        raise ValueError("the moment has no UTC instant within the years 1 to 9999") from None
    # Explicit widths: strftime's %Y does not pad a year below 1000 on every platform.
    return (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
        f".{utc.microsecond // _MICROSECONDS_PER_MILLISECOND:03d}Z"
    )


def encode(message: Mapping[str, Any]) -> bytes:
    """Compact ASCII JSON in the message's own key order; MessageTooLarge above its kind's byte limit."""
    _require_type("message", message, Mapping)
    kind = message.get("message_kind")
    if kind not in {item.value for item in MessageKind} or kind == MessageKind.GENERATION_MANIFEST.value:
        raise ValueError("encode() takes a message whose message_kind the contract defines")
    _refuse_floats(message, kind)
    encoded = json.dumps(message, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    limit = load_registry().max_bytes(kind)
    if len(encoded) > limit:
        raise MessageTooLarge(kind, len(encoded), limit)
    return encoded


def seal(message_bytes: bytes, key: bytes) -> str:
    """The `X-VH-Signature` value of the exact message bytes (core.protocol.signing.sign)."""
    return signing.sign(message_bytes, key)


# --- builders ---------------------------------------------------------------------------------


def _envelope(
    kind: MessageKind, *, protocol_version: str, message_id: str, sent_at: datetime, key_id: str, cell_id: str
) -> dict[str, Any]:
    return {
        "protocol_version": protocol_version,
        "message_kind": kind.value,
        "message_id": message_id,
        "sent_at": timestamp(sent_at),
        "audience": HUB_AUDIENCE_PREFIX + cell_id,
        "key_id": key_id,
    }


def _checked(kind: MessageKind, message: dict[str, Any]) -> dict[str, Any]:
    """The finished message, refused unless it satisfies its kind's schema."""
    _refuse_floats(message, kind.value)
    pointer = schema.first_error_pointer(kind.schema_file, message)
    if pointer is not None:
        raise ValueError(f"{kind.value} violates {kind.schema_file} at {pointer!r}")
    return message


def _optional_parts(
    *, diagnostics: Sequence[Diagnostic], ext: Mapping[str, str | int | bool] | None
) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    # A one-shot iterable would be spent by the checks and leave nothing to send.
    if isinstance(diagnostics, (str, bytes)) or not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a sequence of Diagnostic")
    if diagnostics:
        for diagnostic in diagnostics:
            _require_type("each diagnostic", diagnostic, Diagnostic)
        parts["diagnostics"] = [{"code": item.code, "detail": item.detail} for item in diagnostics]
    if ext is not None:
        _require_type("ext", ext, Mapping)
        parts["ext"] = dict(ext)
    return parts


def claim(
    *,
    protocol_version: str,
    message_id: str,
    sent_at: datetime,
    key_id: str,
    identity: DispatchIdentity,
    bootstrap_token: str,
    runtime_id: str,
) -> dict[str, Any]:
    """`transcode.claim`: the accepted dispatch's identity and bootstrap token, and this runtime's id."""
    _require_type("identity", identity, DispatchIdentity)
    kind = MessageKind.TRANSCODE_CLAIM
    message = {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at, key_id=key_id,
                    cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "claim": {"bootstrap_token": bootstrap_token, "runtime_id": runtime_id},
    }
    return _checked(kind, message)


def progress(
    *,
    protocol_version: str,
    message_id: str,
    sent_at: datetime,
    key_id: str,
    identity: ExecutionIdentity,
    event_seq: int,
    stage: str,
    stage_progress_pct: int,
    message: str | None = None,
    counters: Counters | None = None,
    ext: Mapping[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """`transcode.progress`; `message`, `counters` and `ext` appear only when given."""
    _require_type("identity", identity, ExecutionIdentity)
    kind = MessageKind.TRANSCODE_PROGRESS
    document: dict[str, Any] = {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at, key_id=key_id,
                    cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "event_seq": event_seq,
        "stage": stage,
        "stage_progress_pct": stage_progress_pct,
    }
    if message is not None:
        document["message"] = message
    if counters is not None:
        _require_type("counters", counters, Counters)
        document["counters"] = counters.to_dict()
    document.update(_optional_parts(diagnostics=(), ext=ext))
    return _checked(kind, document)


def result_completed(
    *,
    protocol_version: str,
    message_id: str,
    sent_at: datetime,
    key_id: str,
    identity: ExecutionIdentity,
    generation_id: str,
    manifest: ManifestLocator,
    source: SourceRead,
    profile: Profile,
    media: Media,
    encoder: Encoder,
    diagnostics: Sequence[Diagnostic] = (),
    ext: Mapping[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """`transcode.result.completed`; `diagnostics` appears only when not empty, `ext` only when given."""
    _require_type("identity", identity, ExecutionIdentity)
    for name, value, expected in (
        ("manifest", manifest, ManifestLocator),
        ("source", source, SourceRead),
        ("profile", profile, Profile),
        ("media", media, Media),
        ("encoder", encoder, Encoder),
    ):
        _require_type(name, value, expected)
    kind = MessageKind.TRANSCODE_RESULT_COMPLETED
    document: dict[str, Any] = {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at, key_id=key_id,
                    cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "generation_id": generation_id,
        "manifest": manifest.to_dict(),
        "source": source.to_dict(),
        "profile": profile.to_dict(),
        "media": media.to_dict(),
        "encoder": encoder.to_dict(),
        **_optional_parts(diagnostics=diagnostics, ext=ext),
    }
    return _checked(kind, document)


def result_failed(
    *,
    protocol_version: str,
    message_id: str,
    sent_at: datetime,
    key_id: str,
    identity: ExecutionIdentity,
    failure: Failure,
    stage: str | None,
    source: SourceRead | None = None,
    encoder: Encoder | None = None,
    diagnostics: Sequence[Diagnostic] = (),
    ext: Mapping[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """`transcode.result.failed`: the typed failure with the stage it ended in (None before any stage)."""
    _require_type("identity", identity, ExecutionIdentity)
    _require_type("failure", failure, Failure)
    kind = MessageKind.TRANSCODE_RESULT_FAILED
    document: dict[str, Any] = {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at, key_id=key_id,
                    cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "error": {
            "class": failure.error_class,
            "code": failure.code,
            "retryable": failure.retryable,
            "stage": stage,
            "detail": failure.detail,
        },
    }
    if source is not None:
        _require_type("source", source, SourceRead)
        document["source"] = source.to_dict()
    if encoder is not None:
        _require_type("encoder", encoder, Encoder)
        document["encoder"] = encoder.to_dict()
    document.update(_optional_parts(diagnostics=diagnostics, ext=ext))
    return _checked(kind, document)


def unclaimed(
    *,
    protocol_version: str,
    message_id: str,
    sent_at: datetime,
    key_id: str,
    identity: DispatchIdentity,
    runtime_id: str,
    cause: str,
    hub_error_code: str | None = None,
) -> dict[str, Any]:
    """`transcode.unclaimed`: this start did no work; `hub_error_code` only with the cause `refused`."""
    _require_type("identity", identity, DispatchIdentity)
    kind = MessageKind.TRANSCODE_UNCLAIMED
    document: dict[str, Any] = {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at, key_id=key_id,
                    cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "runtime_id": runtime_id,
        "cause": cause,
    }
    if hub_error_code is not None:
        document["hub_error_code"] = hub_error_code
    return _checked(kind, document)


# --- trusted contexts -------------------------------------------------------------------------

# The kinds the worker sends and self-checks, with the identity each carries.
_SENT_IDENTITIES: Mapping[MessageKind, type] = {
    MessageKind.TRANSCODE_CLAIM: DispatchIdentity,
    MessageKind.TRANSCODE_PROGRESS: ExecutionIdentity,
    MessageKind.TRANSCODE_RESULT_COMPLETED: ExecutionIdentity,
    MessageKind.TRANSCODE_RESULT_FAILED: ExecutionIdentity,
    MessageKind.TRANSCODE_UNCLAIMED: DispatchIdentity,
}


def worker_sender_context(
    kind: MessageKind | str,
    identity: DispatchIdentity | ExecutionIdentity,
    *,
    settings: WorkerSettings,
    now: datetime,
    channel: Channel | str = Channel.PUSH,
    last_event_seq: int | None = None,
    source_id: str | None = None,
    output_location_id: str | None = None,
    profile: Profile | None = None,
) -> TrustedContext:
    """The worker_sender context (README 9.3) of one message about to be sent, from the worker's own records.

    `expected_audience` is the Hub's (`hub:` plus the identity's cell), `known_key_ids` the worker's own
    signing key id. A field the kind requires and the caller left None raises ContextIncomplete.
    """
    message_kind = MessageKind(kind)
    expected_identity = _SENT_IDENTITIES.get(message_kind)
    if expected_identity is None:
        raise ValueError(f"the worker does not send {message_kind.value}")
    _require_type("identity", identity, expected_identity)
    _require_type("settings", settings, WorkerSettings)
    expect: dict[str, Any] = identity.to_dict()
    for name, value in (("source_id", source_id), ("output_location_id", output_location_id)):
        if value is not None:
            expect[name] = value
    if profile is not None:
        _require_type("profile", profile, Profile)
        expect["profile"] = profile.to_dict()
    return TrustedContext.from_dict(
        {
            "context_version": CONTEXT_VERSION,
            "role": Role.WORKER_SENDER.value,
            "channel": Channel(channel).value,
            "accepted_message_kinds": [message_kind.value],
            "now": timestamp(now),
            "clock_skew_s": settings.clock_skew_s,
            "accepted_protocol_versions": list(settings.accepted_protocol_versions),
            "expected_audience": HUB_AUDIENCE_PREFIX + identity.owning_cell_id,
            "known_key_ids": [settings.worker_to_hub_key_id],
            "expect": expect,
            "last_event_seq": last_event_seq,
        }
    )


def receiver_context(
    kind: MessageKind | str,
    *,
    settings: WorkerSettings,
    now: datetime | None = None,
    dispatch: DispatchIdentity | None = None,
    output_location_id: str | None = None,
    runtime_id: str | None = None,
    encryption: Encryption | None = None,
) -> TrustedContext:
    """The worker_receiver context (README 9.3) of `transcode.request`, `transcode.claim.granted` or `hub.error`.

    The request needs `now`; the grant needs `now` and the accepted dispatch's identity, its output location,
    this runtime's id and the dispatch's encryption; `hub.error` needs only the settings. A field the kind
    requires and the caller left None raises ContextIncomplete.
    """
    message_kind = MessageKind(kind)
    _require_type("settings", settings, WorkerSettings)
    data: dict[str, Any] = {
        "context_version": CONTEXT_VERSION,
        "role": Role.WORKER_RECEIVER.value,
        "accepted_message_kinds": [message_kind.value],
        "accepted_protocol_versions": list(settings.accepted_protocol_versions),
    }
    if message_kind is MessageKind.HUB_ERROR:
        data["channel"] = Channel.PUSH.value
        return TrustedContext.from_dict(data)
    if message_kind is MessageKind.TRANSCODE_REQUEST:
        data["channel"] = Channel.DISPATCH.value
    elif message_kind is MessageKind.TRANSCODE_CLAIM_GRANTED:
        data["channel"] = Channel.PUSH.value
        data["expect"] = _grant_expect(dispatch, output_location_id, runtime_id, encryption)
    else:
        raise ValueError(f"the worker does not receive {message_kind.value}")
    data.update(
        {
            "now": None if now is None else timestamp(now),
            "clock_skew_s": settings.clock_skew_s,
            "expected_audience": settings.worker_audience,
            "known_key_ids": list(settings.hub_to_worker_key_ids),
        }
    )
    return TrustedContext.from_dict(data)


def _grant_expect(
    dispatch: DispatchIdentity | None,
    output_location_id: str | None,
    runtime_id: str | None,
    encryption: Encryption | None,
) -> dict[str, Any]:
    expect: dict[str, Any] = {}
    if dispatch is not None:
        _require_type("dispatch", dispatch, DispatchIdentity)
        expect.update(dispatch.to_dict())
    if output_location_id is not None:
        expect["output_location_id"] = output_location_id
    if runtime_id is not None:
        expect["runtime_id"] = runtime_id
    if encryption is not None:
        _require_type("encryption", encryption, Encryption)
        expect["encryption"] = encryption.to_dict()
    return expect
