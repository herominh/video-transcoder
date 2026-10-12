"""Accept a dispatch: the value of a RunPod job's `input` under contract v2 (README sections 12 and 15.1).

The carrier is `{"vh_message": <the request text>, "vh_signature": "hmac-sha256=<hex>"}`. A dispatch is
accepted only when the carrier has that shape, the message bytes verify under a Hub-to-worker key, and the
message is an accepted `transcode.request` for this worker (`worker_receiver`, channel `dispatch`). Anything
else raises DispatchRefused, which names the step, never the content: nothing of a refused dispatch is
trusted, so there is no identity to report it under. The accepted dispatch is a typed, immutable value;
its bootstrap token is kept out of every repr (a claim body can fetch the media key, README 15.2).
The provider hands the handler a decoded value, so the carrier's byte limit (limits.json
`carrier_max_bytes`) is the sender's to keep: here the carrier schema bounds both members. A value that
is not an object of text members is refused before the schema sees it: the schema could only refuse it
too, and its error first renders the value, which fails for a deeply nested one (RecursionError).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, cast

from . import schema
from .inbound import NotAccepted, accept_signed
from .messages import DispatchIdentity, Encryption, Profile, receiver_context
from .registry import MessageKind
from .settings import WorkerSettings
from .verdict import Verdict

RUNPOD_INPUT_SCHEMA = "carriers/runpod-input.schema.json"
MESSAGE_MEMBER = "vh_message"
SIGNATURE_MEMBER = "vh_signature"
STEP_CARRIER = "carrier"
_MICROSECONDS_PER_MILLISECOND = 1_000


class DispatchRefused(Exception):
    """The job input is not a dispatch this worker may act on; the text names the step, the layer and the reason."""

    def __init__(self, step: str, verdict: Verdict | None = None) -> None:
        detail = step
        if verdict is not None and verdict.layer is not None and verdict.reason is not None:
            detail = f"{step}: {verdict.layer.value}/{verdict.reason.value}"
        super().__init__(f"the dispatch is refused at {detail}")
        self.step = step
        self.verdict = verdict


@dataclass(frozen=True)
class SourceLocator:
    """Where the source object is and what the Hub recorded of it."""

    source_id: str
    location_id: str
    bucket: str
    credential_ref: str
    object_key: str
    size_bytes: int
    etag: str


@dataclass(frozen=True)
class OutputLocation:
    """Where the generation is written; the prefix inside it comes with the grant."""

    location_id: str
    bucket: str
    credential_ref: str


@dataclass(frozen=True)
class DispatchLimits:
    max_source_bytes: int
    max_source_duration_ms: int
    max_output_bytes: int
    max_artifact_count: int
    max_wall_time_ms: int


@dataclass(frozen=True)
class AcceptedDispatch:
    """An accepted `transcode.request`; repr() does not show the bootstrap token."""

    protocol_version: str
    identity: DispatchIdentity
    source: SourceLocator
    output: OutputLocation
    profile: Profile
    renditions: tuple[str, ...]
    encryption: Encryption
    limits: DispatchLimits
    bootstrap_token: str = field(repr=False)
    claim_expires_at: datetime


def instant(timestamp: str) -> datetime:
    """The aware datetime of a contract timestamp the validator already accepted (`YYYY-MM-DDTHH:MM:SS.mmmZ`)."""
    return datetime(
        int(timestamp[0:4]),
        int(timestamp[5:7]),
        int(timestamp[8:10]),
        int(timestamp[11:13]),
        int(timestamp[14:16]),
        int(timestamp[17:19]),
        int(timestamp[20:23]) * _MICROSECONDS_PER_MILLISECOND,
        tzinfo=timezone.utc,
    )


def accept_dispatch(job_input: object, *, settings: WorkerSettings, now: datetime) -> AcceptedDispatch:
    """The dispatch a RunPod job input carries, or DispatchRefused. `now` is the worker's clock at the start."""
    if not _is_object_of_text(job_input):
        raise DispatchRefused(STEP_CARRIER)
    if schema.first_error_pointer(RUNPOD_INPUT_SCHEMA, job_input) is not None:
        raise DispatchRefused(STEP_CARRIER)
    carrier = cast(Mapping[str, str], job_input)  # the carrier schema: an object of two strings
    message = _ascii_bytes(carrier[MESSAGE_MEMBER])
    if message is None:
        raise DispatchRefused(STEP_CARRIER)
    context = receiver_context(MessageKind.TRANSCODE_REQUEST, settings=settings, now=now)
    try:
        request = accept_signed(
            message, carrier[SIGNATURE_MEMBER], context=context, keys=settings.hub_to_worker_keys
        )
    except NotAccepted as refusal:
        # A NotAccepted holds a step and a verdict only, so chaining it shows nothing of the message.
        raise DispatchRefused(refusal.step, refusal.verdict) from None
    return _typed(request)


def _is_object_of_text(value: object) -> bool:
    """Whether the value is a JSON object whose members are all text: one level deep, nothing is descended into."""
    return isinstance(value, dict) and all(
        isinstance(name, str) and isinstance(member, str) for name, member in value.items()
    )


def _ascii_bytes(text: str) -> bytes | None:
    """The text's bytes; None when it is not ASCII (the encoder's own error would quote the text)."""
    try:
        return text.encode("ascii")
    except UnicodeEncodeError:
        return None


def _typed(request: Mapping[str, Any]) -> AcceptedDispatch:
    claim = request["claim"]
    return AcceptedDispatch(
        protocol_version=request["protocol_version"],
        identity=DispatchIdentity(**request["identity"]),
        source=SourceLocator(**request["source"]),
        output=OutputLocation(**request["output"]),
        profile=Profile(**request["profile"]),
        renditions=tuple(request["renditions"]),
        encryption=Encryption(**request["encryption"]),
        limits=DispatchLimits(**request["limits"]),
        bootstrap_token=claim["bootstrap_token"],
        claim_expires_at=instant(claim["expires_at"]),
    )
