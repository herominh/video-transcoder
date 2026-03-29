from __future__ import annotations

import logging
import os
import tempfile

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from core.callback import send_callback, send_progress
from core.config import Settings, _detected_encoder, _detected_preset, resolve_encoder
from core.signing import verify_request
from core.storage import download_source, upload_results
from core.transcoder import cleanup, transcode_to_hls

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Transcoder", version="1.0.0")


class TranscodeRequest(BaseModel):
    uuid: str
    source_url: str
    qualities: list[str]
    encryption_key_hex: str | None = None
    segment_duration: int = 6
    callback_url: str
    r2_bucket: str
    r2_path_prefix: str
    encoder: str | None = None
    preset: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "encoder": _detected_encoder,
        "preset": _detected_preset,
    }


@app.post("/transcode", status_code=202)
async def transcode(raw_request: Request, background_tasks: BackgroundTasks):
    settings = Settings.from_env()

    # Verify HMAC signature.
    body = await raw_request.body()
    sig = raw_request.headers.get("X-Signature")
    ts = raw_request.headers.get("X-Timestamp")

    if not verify_request(body, sig, ts, settings.webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        request = TranscodeRequest.model_validate_json(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    # Resolve encoder: use requested if available, fallback to detected.
    actual_encoder, actual_preset = resolve_encoder(request.encoder, request.preset)
    settings = Settings(
        ffmpeg_encoder=actual_encoder,
        ffmpeg_preset=actual_preset,
        r2_access_key_id=settings.r2_access_key_id,
        r2_secret_access_key=settings.r2_secret_access_key,
        r2_endpoint=settings.r2_endpoint,
        r2_region=settings.r2_region,
        webhook_secret=settings.webhook_secret,
    )

    logger.info(
        "Transcode %s: requested=%s/%s, using=%s/%s",
        request.uuid, request.encoder, request.preset,
        actual_encoder, actual_preset,
    )

    background_tasks.add_task(_process_transcode, request, settings)

    return {"status": "accepted", "uuid": request.uuid}


def _process_transcode(request: TranscodeRequest, settings: Settings) -> None:
    work_dir = tempfile.mkdtemp(prefix=f"transcode-{request.uuid}-")
    input_path = os.path.join(work_dir, f"source-{request.uuid}.mp4")
    output_dir = os.path.join(work_dir, "output")

    def _progress(stage: str, pct: int, msg: str = "") -> None:
        send_progress(request.callback_url, settings.webhook_secret, request.uuid, stage, pct, msg)

    try:
        # 1. Download source from presigned R2 URL.
        logger.info("Starting transcode for %s", request.uuid)
        _progress("downloading", 0, "Downloading source")
        download_source(request.source_url, input_path)

        # 2. Transcode to HLS.
        _progress("transcoding", 0, "Starting transcode")
        result = transcode_to_hls(
            input_path=input_path,
            output_dir=output_dir,
            settings=settings,
            enabled_qualities=request.qualities,
            encryption_key_hex=request.encryption_key_hex,
            segment_duration=request.segment_duration,
            progress_callback=lambda pct, msg: _progress("transcoding", pct, msg),
        )

        # 3. Upload results to R2.
        _progress("uploading", 0, "Uploading files")
        upload_results(
            settings=settings,
            output_dir=output_dir,
            bucket=request.r2_bucket,
            path_prefix=request.r2_path_prefix,
        )

        # 4. Send success callback.
        send_callback(
            url=request.callback_url,
            secret=settings.webhook_secret,
            uuid=request.uuid,
            status="ready",
            duration=result["duration"],
            qualities=result["qualities"],
            master_playlist=result["master_playlist"],
            thumbnail=result["thumbnail"],
            encoder=settings.ffmpeg_encoder,
            preset=settings.ffmpeg_preset,
        )

        logger.info("Transcode completed for %s", request.uuid)

    except Exception as e:
        logger.exception("Transcode failed for %s", request.uuid)

        # Send failure callback.
        try:
            send_callback(
                url=request.callback_url,
                secret=settings.webhook_secret,
                uuid=request.uuid,
                status="failed",
                error_message=str(e)[:1000],
            )
        except Exception:
            logger.exception("Failed to send failure callback for %s", request.uuid)

    finally:
        cleanup(work_dir)
