from __future__ import annotations

import hashlib
import hmac
import time


def sign_request(body: bytes, secret: str) -> dict[str, str]:
    """Sign a request body with HMAC-SHA256.

    Returns headers dict with X-Signature and X-Timestamp.
    """
    timestamp = str(int(time.time()))
    message = timestamp.encode() + b"." + body
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()

    return {
        "X-Signature": f"sha256={signature}",
        "X-Timestamp": timestamp,
    }


def verify_request(
    body: bytes,
    signature_header: str | None,
    timestamp_header: str | None,
    secret: str,
    tolerance: int = 300,
) -> bool:
    """Verify an incoming request's HMAC-SHA256 signature.

    Returns False if headers are missing, timestamp is stale, or signature
    doesn't match.
    """
    if not signature_header or not timestamp_header:
        return False

    # Reject stale timestamps.
    try:
        ts = int(timestamp_header)
    except (ValueError, TypeError):
        return False

    if abs(int(time.time()) - ts) > tolerance:
        return False

    message = timestamp_header.encode() + b"." + body
    expected = "sha256=" + hmac.new(
        secret.encode(), message, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header)
