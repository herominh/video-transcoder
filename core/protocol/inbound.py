"""Accept a signed message the Hub sent: the signature first, then the validation, then the key the body names.

The order is stricter than the live receiver order of README section 12: nothing of the body is parsed
before the exact bytes verified under one of the worker's Hub-to-worker keys, and an accepted message
must name, in its `key_id`, a key under which its bytes verified: a body that names one key and is
signed with another is refused (the settings refuse two key ids that hold the same key bytes). A refusal names
the step and, for a verdict, its layer and reason: never the pointer, the body, the signature or a key.

A `hub.error` is unsigned (README section 12): accept_hub_error() only validates it, and the worker never
acts on one except by not working, or by sending again what the Hub called retryable.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import gate, pipeline, signing
from .context import TrustedContext
from .registry import load_registry
from .verdict import Verdict

STEP_SIGNATURE = "signature"
STEP_VALIDATION = "validation"
STEP_KEY_ID = "key_id"


class NotAccepted(Exception):
    """The message is not one the worker may act on; `step` says where it failed, `verdict` what the validator said."""

    def __init__(self, step: str, verdict: Verdict | None = None) -> None:
        detail = step
        if verdict is not None and verdict.layer is not None and verdict.reason is not None:
            detail = f"{step}: {verdict.layer.value}/{verdict.reason.value}"
        super().__init__(detail)
        self.step = step
        self.verdict = verdict


def accept_signed(
    raw: bytes, signature: object, *, context: TrustedContext, keys: Mapping[str, bytes]
) -> dict[str, Any]:
    """The decoded message when `raw` verifies, validates under `context` and names a key that verified it.

    `keys` maps each Hub-to-worker key id to its key. Raises NotAccepted otherwise.
    """
    if not isinstance(raw, bytes):
        raise TypeError("accept_signed() takes the exact received bytes")
    verifying_key_ids = frozenset(key_id for key_id, key in keys.items() if signing.verify(raw, key, signature))
    if not verifying_key_ids:
        raise NotAccepted(STEP_SIGNATURE)
    verdict = pipeline.validate(raw, context)
    if not verdict.accepted:
        raise NotAccepted(STEP_VALIDATION, verdict)
    document = _decoded(raw)
    if document.get("key_id") not in verifying_key_ids:
        raise NotAccepted(STEP_KEY_ID)
    return document


def accept_hub_error(raw: bytes, *, context: TrustedContext) -> Mapping[str, Any] | None:
    """The `error` object of a `hub.error` accepted under `context` (`worker_receiver`), or None for any other answer."""
    if not isinstance(raw, bytes):
        raise TypeError("accept_hub_error() takes the exact received bytes")
    if not pipeline.validate(raw, context).accepted:
        return None
    return _decoded(raw)["error"]


def _decoded(accepted: bytes) -> dict[str, Any]:
    """The object of bytes the validator accepted, read by the validator's own parser."""
    registry = load_registry()
    document = gate.parse(accepted, registry.max_json_depth, registry.max_tokens)
    if not isinstance(document, dict):
        raise RuntimeError("an accepted message did not decode")
    return document
