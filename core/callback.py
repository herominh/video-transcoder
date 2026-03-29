from __future__ import annotations

import json
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.signing import sign_request

logger = logging.getLogger(__name__)

# Retry config for final callbacks (done/failed).
# 3 retries with exponential backoff: 0s, 1s, 2s.
_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[502, 503, 504],
    allowed_methods=["POST"],
)


def _create_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def send_progress(
    url: str,
    secret: str,
    uuid: str,
    stage: str,
    progress: int,
    message: str = "",
) -> None:
    """Fire-and-forget progress update to Video Hub. Swallows all errors."""
    payload = {
        "uuid": uuid,
        "status": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
    }
    body = json.dumps(payload).encode()
    hmac_headers = sign_request(body, secret)
    headers = {
        **hmac_headers,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        requests.post(url, data=body, headers=headers, timeout=5)
    except Exception:
        logger.debug("Progress update failed for %s (non-critical)", uuid)


def send_callback(
    url: str,
    secret: str,
    uuid: str,
    status: str,
    duration: int | None = None,
    qualities: list[dict] | None = None,
    master_playlist: str | None = None,
    thumbnail: str | None = None,
    error_message: str | None = None,
    encoder: str | None = None,
    preset: str | None = None,
) -> None:
    """POST transcode result back to Video Hub."""
    payload: dict = {
        "uuid": uuid,
        "status": status,
    }

    if status == "ready":
        payload["duration"] = duration
        payload["qualities"] = qualities or []
        payload["master_playlist"] = master_playlist
        payload["thumbnail"] = thumbnail
        if encoder:
            payload["encoder"] = encoder
        if preset:
            payload["preset"] = preset
    elif status == "failed":
        payload["error_message"] = error_message

    body = json.dumps(payload).encode()
    hmac_headers = sign_request(body, secret)

    headers = {
        **hmac_headers,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info("Sending callback to %s for %s (status=%s)", url, uuid, status)

    session = _create_session()
    response = session.post(url, data=body, headers=headers, timeout=30)
    response.raise_for_status()

    logger.info("Callback sent successfully for %s", uuid)
