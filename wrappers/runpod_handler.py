"""RunPod Serverless handler — same core logic, RunPod-managed lifecycle.

RunPod handles inbound auth (API key), so no HMAC verification needed.
WEBHOOK_SECRET is still required for signing outbound callbacks to Video Hub.
"""

import logging
import os
import sys

# Ensure project root is on path so `core` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def handler(job: dict) -> dict:
    """RunPod serverless handler entry point.

    Receives job["input"] as TranscodeRequest fields (dict).
    Runs _process_transcode synchronously — RunPod manages container lifecycle.
    """
    from core.api import TranscodeRequest, _process_transcode
    from core.config import Settings, resolve_encoder

    request = TranscodeRequest.model_validate(job["input"])

    base_settings = Settings.from_env()

    actual_encoder, actual_preset = resolve_encoder(request.encoder, request.preset)

    settings = Settings(
        ffmpeg_encoder=actual_encoder,
        ffmpeg_preset=actual_preset,
        r2_access_key_id=base_settings.r2_access_key_id,
        r2_secret_access_key=base_settings.r2_secret_access_key,
        r2_endpoint=base_settings.r2_endpoint,
        r2_region=base_settings.r2_region,
        webhook_secret=base_settings.webhook_secret,
    )

    logger.info(
        "RunPod transcode %s: requested=%s/%s, using=%s/%s",
        request.uuid, request.encoder, request.preset,
        actual_encoder, actual_preset,
    )

    _process_transcode(request, settings)

    return {"status": "completed", "uuid": request.uuid}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
