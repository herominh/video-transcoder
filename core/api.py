from __future__ import annotations

import logging
import os
import tempfile
import uuid as uuid_lib
from time import monotonic

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.callback import build_result_payload, send_progress, send_result
from core.config import Settings, _detected_encoder, _detected_preset, resolve_encoder
from core.signing import verify_request
from core.storage import download_from_s3, upload_results
from core.transcoder import cleanup, transcode_to_hls
from core.v1_admission import ExecutionDisabled, require_v1_execution, v1_execution_admitted

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Transcoder", version="1.0.0")


class TranscodeRequest(BaseModel):
    uuid: str
    qualities: list[str]
    encryption_key_hex: str | None = None
    segment_duration: int = 6
    callback_url: str
    s3_bucket: str = Field(min_length=1)
    s3_path_prefix: str
    s3_original_path: str = Field(min_length=1)
    encoder: str | None = None
    preset: str | None = None
    preset_level: int | None = None

    @field_validator("uuid")
    @classmethod
    def _validate_uuid(cls, value: str) -> str:
        # uuid flows into filesystem paths and S3 keys — reject anything that
        # isn't a canonical UUID (blocks traversal like "../../etc").
        try:
            parsed = uuid_lib.UUID(value)
        except (ValueError, TypeError):
            raise ValueError("uuid must be a valid UUID")
        if str(parsed) != value.lower():
            raise ValueError("uuid must be in canonical UUID form")
        return value


@app.get("/health")
def health():
    return {
        "status": "ok",
        "encoder": _detected_encoder,
        "preset": _detected_preset,
        "v1_execution_admitted": v1_execution_admitted(),
    }


@app.post("/transcode", status_code=202)
async def transcode(raw_request: Request, background_tasks: BackgroundTasks):
    try:
        require_v1_execution()
    except ExecutionDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

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
        raise HTTPException(status_code=422, detail=e.errors(include_context=False))

    # Resolve encoder: use requested if available, fallback to detected.
    actual_encoder, actual_preset = resolve_encoder(
        request.encoder, request.preset, request.preset_level,
    )
    settings = Settings(
        ffmpeg_encoder=actual_encoder,
        ffmpeg_preset=actual_preset,
        s3_access_key_id=settings.s3_access_key_id,
        s3_secret_access_key=settings.s3_secret_access_key,
        s3_endpoint=settings.s3_endpoint,
        s3_region=settings.s3_region,
        webhook_secret=settings.webhook_secret,
    )

    logger.info(
        "Transcode %s: requested=%s/%s level=%s, using=%s/%s",
        request.uuid, request.encoder, request.preset, request.preset_level,
        actual_encoder, actual_preset,
    )

    background_tasks.add_task(_process_transcode, request, settings)

    return {"status": "accepted", "uuid": request.uuid}


def _process_transcode(request: TranscodeRequest, settings: Settings) -> dict:
    """Run the admitted development pipeline and return its ready/failed result.

    Callback delivery is attempted after the outcome is decided and is non-fatal.
    """
    require_v1_execution()
    work_dir = tempfile.mkdtemp(prefix=f"transcode-{request.uuid}-")
    input_path = os.path.join(work_dir, f"source-{request.uuid}.mp4")
    output_dir = os.path.join(work_dir, "output")

    # Back off failed progress delivery without permanently losing heartbeats
    # when the hub recovers. Skip events during backoff instead of sleeping.
    progress_retry_at = 0.0

    def _progress(stage: str, pct: int, msg: str = "") -> None:
        nonlocal progress_retry_at
        if monotonic() < progress_retry_at:
            return
        if not send_progress(
            request.callback_url, settings.webhook_secret,
            request.uuid, stage, pct, msg,
        ):
            progress_retry_at = monotonic() + 30

    try:
        # 1. Download source.
        logger.info("Starting transcode for %s", request.uuid)
        _progress("downloading", 0, "Downloading source")

        download_from_s3(settings, request.s3_bucket, request.s3_original_path, input_path)

        # Get source file size for callback.
        source_filesize = os.path.getsize(input_path)

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

        # 3. Upload results to S3.
        _progress("uploading", 0, "Uploading files")
        upload_results(
            settings=settings,
            output_dir=output_dir,
            bucket=request.s3_bucket,
            path_prefix=request.s3_path_prefix,
        )

        payload = build_result_payload(
            uuid=request.uuid,
            status="ready",
            duration=result["duration"],
            qualities=result["qualities"],
            master_playlist=result["master_playlist"],
            thumbnail=result["thumbnail"],
            encoder=settings.ffmpeg_encoder,
            preset=settings.ffmpeg_preset,
            source_filesize=source_filesize,
        )
        logger.info("Transcode completed for %s", request.uuid)

    except Exception as e:
        logger.exception("Transcode failed for %s", request.uuid)
        payload = build_result_payload(
            uuid=request.uuid,
            status="failed",
            error_message=str(e)[:1000],
        )

    finally:
        try:
            cleanup(work_dir)
        except Exception:
            logger.exception("Cleanup failed for %s; preserving transcode result", request.uuid)

    # Single delivery point AFTER the outcome is decided: a delivery hiccup can
    # no longer flip a successful transcode into a "failed" report.
    try:
        send_result(request.callback_url, settings.webhook_secret, payload)
    except Exception:
        logger.exception(
            "Result delivery failed for %s (status=%s) — Video Hub can still "
            "reconcile via job-status polling",
            request.uuid, payload["status"],
        )

    return payload
