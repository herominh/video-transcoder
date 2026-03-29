"""Modal wrapper — mounts the shared FastAPI app on a GPU container."""

import modal

app = modal.App("video-transcoder")

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
    gpu="T4",
    timeout=3600,
    allow_concurrent_inputs=1,
    secrets=[modal.Secret.from_name("video-transcoder-secrets")],
)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app")

    from core.api import app as fastapi_app

    return fastapi_app
