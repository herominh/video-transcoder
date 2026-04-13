"""Modal wrapper — CPU web endpoint + GPU worker with independent lifecycle.

The web endpoint validates HMAC and returns 202 immediately.
The GPU worker runs in its own container with full timeout, fixing the
background task killing issue from the previous single-function approach.
"""

import modal

app = modal.App("video-transcoder")

_secrets = [modal.Secret.from_name("video-transcoder-secrets")]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "software-properties-common",
        "gnupg",
        "curl",
    )
    .run_commands(
        # Install jellyfin-ffmpeg (includes NVENC support).
        "curl -fsSL https://repo.jellyfin.org/jellyfin_team.gpg.key | gpg --dearmor -o /usr/share/keyrings/jellyfin.gpg",
        'echo "deb [signed-by=/usr/share/keyrings/jellyfin.gpg] https://repo.jellyfin.org/ubuntu jammy main" > /etc/apt/sources.list.d/jellyfin.list',
        "apt-get update && apt-get install -y jellyfin-ffmpeg7 && ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg && ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe",
    )
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "requests>=2.31.0",
        "boto3>=1.35.0",
        "pydantic>=2.0",
    )
    .add_local_dir("core", remote_path="/app/core")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    secrets=_secrets,
)
def process_transcode(request_dict: dict) -> None:
    """GPU worker — own container, own lifecycle.

    Runs _process_transcode synchronously. Sends its own callbacks to Video Hub.
    """
    import sys
    sys.path.insert(0, "/app")

    from core.api import TranscodeRequest, _process_transcode
    from core.config import Settings, resolve_encoder

    request = TranscodeRequest.model_validate(request_dict)
    base_settings = Settings.from_env()

    actual_encoder, actual_preset = resolve_encoder(
        request.encoder, request.preset, request.preset_level,
    )
    settings = Settings(
        ffmpeg_encoder=actual_encoder,
        ffmpeg_preset=actual_preset,
        s3_access_key_id=base_settings.s3_access_key_id,
        s3_secret_access_key=base_settings.s3_secret_access_key,
        s3_endpoint=base_settings.s3_endpoint,
        s3_region=base_settings.s3_region,
        webhook_secret=base_settings.webhook_secret,
    )

    _process_transcode(request, settings)


@app.function(
    image=image,
    secrets=_secrets,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def web():
    """CPU web endpoint — validates HMAC, spawns GPU worker."""
    import sys
    sys.path.insert(0, "/app")

    from fastapi import FastAPI, HTTPException, Request
    from pydantic import ValidationError

    from core.api import TranscodeRequest
    from core.config import Settings, _detected_encoder, _detected_preset
    from core.signing import verify_request

    fastapi_app = FastAPI(title="Video Transcoder (Modal)", version="1.0.0")

    @fastapi_app.get("/health")
    def health():
        return {
            "status": "ok",
            "runtime": "modal",
            "encoder": _detected_encoder,
            "preset": _detected_preset,
        }

    @fastapi_app.post("/transcode", status_code=202)
    async def transcode(raw_request: Request):
        settings = Settings.from_env()

        body = await raw_request.body()
        sig = raw_request.headers.get("X-Signature")
        ts = raw_request.headers.get("X-Timestamp")

        if not verify_request(body, sig, ts, settings.webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            request = TranscodeRequest.model_validate_json(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        # Fire-and-forget: GPU worker sends its own callbacks.
        process_transcode.spawn(request.model_dump())

        return {"status": "accepted", "uuid": request.uuid}

    return fastapi_app
