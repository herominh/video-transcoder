"""Accept only signed acquire requests and retain their immutable authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, cast

from . import schema
from .context import TrustedContext
from .dispatch import (
    MESSAGE_MEMBER, RUNPOD_INPUT_SCHEMA, SIGNATURE_MEMBER, STEP_CARRIER,
    DispatchRefused, _ascii_bytes, _is_object_of_text, instant,
)
from .inbound import NotAccepted, accept_signed
from .messages import CONTEXT_VERSION, DispatchIdentity, timestamp
from .registry import Channel, MessageKind, Role
from .settings import AcquireSettings


@dataclass(frozen=True)
class Destination:
    source_id: str
    location_id: str
    object_key: str


@dataclass(frozen=True)
class Expected:
    size: int
    md5: str | None
    sha1: str | None
    sha256: str | None
    head_revision_id: str | None = field(repr=False)


@dataclass(frozen=True)
class External:
    provider: str
    file_id: str = field(repr=False)
    resource_key: str | None = field(repr=False)
    expected: Expected


@dataclass(frozen=True)
class Plan:
    part_size: int
    parts_total: int


@dataclass(frozen=True)
class AcquireLimits:
    max_source_bytes: int
    max_wall_time_ms: int


@dataclass(frozen=True)
class AcceptedAcquire:
    protocol_version: str
    identity: DispatchIdentity
    destination: Destination
    external: External
    plan: Plan
    limits: AcquireLimits
    bootstrap_token: str = field(repr=False)
    claim_expires_at: datetime

    def expectations(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "acquire_destination": asdict(self.destination),
            "acquire_external": asdict(self.external),
            "acquire_plan": asdict(self.plan),
            "acquire_limits": asdict(self.limits),
            "acquire_expires_at": timestamp(self.claim_expires_at),
        }


def receiver_context(
    *, settings: AcquireSettings, now: datetime, dispatch: AcceptedAcquire | None = None,
    runtime_id: str | None = None,
) -> TrustedContext:
    grant = dispatch is not None
    data: dict[str, Any] = {
        "context_version": CONTEXT_VERSION,
        "role": Role.WORKER_RECEIVER.value,
        "channel": Channel.PUSH.value if grant else Channel.DISPATCH.value,
        "accepted_message_kinds": [
            MessageKind.SOURCE_ACQUIRE_GRANTED.value if grant else MessageKind.SOURCE_ACQUIRE_REQUEST.value
        ],
        "accepted_protocol_versions": [dispatch.protocol_version] if dispatch else list(settings.accepted_protocol_versions),
        "expected_audience": settings.worker_audience,
        "known_key_ids": list(settings.hub_to_worker_key_ids),
        "now": timestamp(now),
        "clock_skew_s": settings.clock_skew_s,
    }
    if dispatch is not None:
        data["expect"] = {
            **dispatch.expectations(), "runtime_id": runtime_id, "acquire_storage_host": settings.storage_host,
        }
    return TrustedContext.from_dict(data)


def accept_dispatch(job_input: object, *, settings: AcquireSettings, now: datetime) -> AcceptedAcquire:
    if not _is_object_of_text(job_input) or schema.first_error_pointer(RUNPOD_INPUT_SCHEMA, job_input) is not None:
        raise DispatchRefused(STEP_CARRIER)
    carrier = cast(Mapping[str, str], job_input)
    body = _ascii_bytes(carrier[MESSAGE_MEMBER])
    if body is None:
        raise DispatchRefused(STEP_CARRIER)
    try:
        request = accept_signed(
            body, carrier[SIGNATURE_MEMBER], context=receiver_context(settings=settings, now=now),
            keys=settings.hub_to_worker_keys,
        )
    except NotAccepted as refusal:
        raise DispatchRefused(refusal.step, refusal.verdict) from None
    external = request["external"]
    return AcceptedAcquire(
        protocol_version=request["protocol_version"], identity=DispatchIdentity(**request["identity"]),
        destination=Destination(**request["destination"]),
        external=External(external["provider"], external["file_id"], external["resource_key"], Expected(**external["expected"])),
        plan=Plan(**request["plan"]), limits=AcquireLimits(**request["limits"]),
        bootstrap_token=request["claim"]["bootstrap_token"], claim_expires_at=instant(request["claim"]["expires_at"]),
    )
