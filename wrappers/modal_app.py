"""Staged Modal candidate — web and GPU definitions retained with execution disabled."""

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
    """Refuse execution until Modal implements the provider adapter contract."""
    import sys
    sys.path.insert(0, "/app")

    from core.v1_admission import refuse_modal_execution

    refuse_modal_execution()


@app.function(
    image=image,
    secrets=_secrets,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def web():
    """Serve health checks and refuse staged execution before reading a request."""
    import sys
    sys.path.insert(0, "/app")

    from fastapi import FastAPI, HTTPException, Request
    from core.config import _detected_encoder, _detected_preset
    from core.v1_admission import ExecutionDisabled, refuse_modal_execution

    fastapi_app = FastAPI(title="Video Transcoder (Modal)", version="1.0.0")

    @fastapi_app.get("/health")
    def health():
        return {
            "status": "ok",
            "runtime": "modal",
            "encoder": _detected_encoder,
            "preset": _detected_preset,
        }

    @fastapi_app.post("/transcode", status_code=503)
    async def transcode(raw_request: Request):
        try:
            refuse_modal_execution()
        except ExecutionDisabled as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None

    return fastapi_app
