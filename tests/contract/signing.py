"""Exact-bytes message signing of the transcode contract v2 (HMAC-SHA256, one key per key id and direction).

Pure: no key storage, no key lookup, no header parsing beyond the `hmac-sha256=` prefix.
The live v1 signer (core/signing.py) is unrelated and untouched.
"""

from __future__ import annotations

import hashlib
import hmac
import re

SIGNING_PREFIX = b"video-hub.transcode.v2\n"
SIGNATURE_SCHEME = "hmac-sha256="
_SIGNATURE_VALUE = re.compile(r"hmac-sha256=[0-9a-f]{64}")


def _require_bytes(message: object, key: object) -> None:
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("the message must be the exact bytes")
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise ValueError("the key must be non-empty bytes")


def sign(message: bytes, key: bytes) -> str:
    """`hmac-sha256=<64 lowercase hex>` over the domain prefix followed by the exact message bytes."""
    _require_bytes(message, key)
    digest = hmac.new(bytes(key), SIGNING_PREFIX + bytes(message), hashlib.sha256).hexdigest()
    return SIGNATURE_SCHEME + digest


def verify(message: bytes, key: bytes, value: object) -> bool:
    """True only when `value` is a well-formed signature of these bytes under this key; never raises for a bad value."""
    _require_bytes(message, key)
    if not isinstance(value, str) or _SIGNATURE_VALUE.fullmatch(value) is None:
        return False
    return hmac.compare_digest(sign(message, key), value)
