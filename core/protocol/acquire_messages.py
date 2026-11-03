"""Acquire terminal builders and contexts from accepted authority only."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence, TypedDict

from .acquire_dispatch import AcceptedAcquire
from .context import TrustedContext
from .messages import CONTEXT_VERSION, HUB_AUDIENCE_PREFIX, ExecutionIdentity, _checked, _envelope, timestamp
from .registry import Channel, MessageKind, Role
from .settings import AcquireSettings


class Envelope(TypedDict):
    protocol_version: str
    message_id: str
    sent_at: datetime
    key_id: str
    identity: ExecutionIdentity


# The complete relation in the contract; deadline alone uses the stage where time ran out.
ERRORS: Mapping[str, tuple[str, bool, str | None]] = {
    "file_removed": ("source_unavailable", False, "downloading"),
    "link_not_shared": ("source_unavailable", False, "downloading"),
    "provider_quota": ("source_unavailable", True, "downloading"),
    "provider_refused": ("source_unavailable", False, "downloading"),
    "not_configured": ("configuration_error", False, None),
    "size_exceeded": ("input_limits_exceeded", False, "downloading"),
    "size_short": ("source_integrity_mismatch", False, "downloading"),
    "integrity_failed": ("source_integrity_mismatch", False, "finalizing"),
    "source_changed_at_provider": ("source_integrity_mismatch", False, "downloading"),
    "deadline": ("deadline_exceeded", False, None),
    "download_failed": ("source_unavailable", True, "downloading"),
    "upload_failed": ("output_write_failed", True, "uploading"),
    "input_refused": ("input_invalid", False, None),
}


def result_completed(
    *, protocol_version: str, message_id: str, sent_at: datetime, key_id: str, identity: ExecutionIdentity,
    bytes_read: int, checksum_verified: str, parts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kind = MessageKind.SOURCE_ACQUIRE_RESULT_COMPLETED
    return _checked(kind, {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at,
                    key_id=key_id, cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(), "bytes": bytes_read, "checksum_verified": checksum_verified,
        "parts": [dict(part) for part in parts],
    })


def result_failed(
    *, protocol_version: str, message_id: str, sent_at: datetime, key_id: str, identity: ExecutionIdentity,
    code: str, stage: str | None,
) -> dict[str, Any]:
    if code not in ERRORS:
        raise ValueError("unknown acquire failure code")
    error_class, retryable, fixed_stage = ERRORS[code]
    kind = MessageKind.SOURCE_ACQUIRE_RESULT_FAILED
    return _checked(kind, {
        **_envelope(kind, protocol_version=protocol_version, message_id=message_id, sent_at=sent_at,
                    key_id=key_id, cell_id=identity.owning_cell_id),
        "identity": identity.to_dict(),
        "error": {"class": error_class, "code": code, "retryable": retryable,
                  "stage": stage if code == "deadline" else fixed_stage, "detail": "Acquisition ended with " + code + "."},
    })


def sender_context(
    kind: MessageKind, dispatch: AcceptedAcquire, identity: ExecutionIdentity, *,
    settings: AcquireSettings, now: datetime, channel: Channel = Channel.PUSH,
) -> TrustedContext:
    expect = identity.to_dict()
    if kind is MessageKind.SOURCE_ACQUIRE_RESULT_COMPLETED:
        expect.update(dispatch.expectations())
    return TrustedContext.from_dict({
        "context_version": CONTEXT_VERSION, "role": Role.WORKER_SENDER.value, "channel": channel.value,
        "accepted_message_kinds": [kind.value], "accepted_protocol_versions": [dispatch.protocol_version],
        "expected_audience": HUB_AUDIENCE_PREFIX + identity.owning_cell_id,
        "known_key_ids": [settings.worker_to_hub_key_id], "now": timestamp(now),
        "clock_skew_s": settings.clock_skew_s, "expect": expect,
    })
