# Video Transcoder

Video processing worker for Video Hub. Supported processing runs through an external provider adapter: RunPod first, with Modal retained as the staged second candidate. A separate development-only container adapter retains the v1 pipeline.

## Architecture

- `wrappers/runpod_handler.py` and `Dockerfile.runpod` — contract v2 transcode entry: claim the execution before processing, then return the signed result. The protocol and job runner live in `core/protocol/` and `core/job/`.
- `wrappers/runpod_import_handler.py`, `core/drive_import/` and `Dockerfile.import` — supported Google Drive intake, separate from transcoding.
- `wrappers/docker_server.py`, `core/api.py` and `Dockerfile` — development-only v1 HTTP adapter, disabled unless `VH_DEVELOPMENT_ADAPTER` is exactly `1`.
- `wrappers/modal_app.py` — staged second candidate. Both its HTTP endpoint and worker function refuse execution until Modal implements the provider adapter contract.
- Stateless — no database; job tracking lives in Video Hub.

## Local Development

Requires Python 3.12+ and FFmpeg (NVENC for GPU, or a standard build for CPU).

```bash
pip install -r requirements.txt
cp .env.example .env
```

Configure the v1 adapter's `FFMPEG_*`, `WEBHOOK_SECRET` and `S3_*` settings in `.env`. Source videos must already be in the organization's store at the request's `s3_bucket` and `s3_original_path`.

Opt in explicitly when starting the development server:

```bash
VH_DEVELOPMENT_ADAPTER=1 python3 wrappers/docker_server.py
```

The server starts at `http://localhost:8000`. Starting it without the opt-in still serves `/health`; `/transcode` returns 503. Direct `uvicorn core.api:app` launches use the same admission rule. Only the exact value `1` admits execution: unset, empty, `0`, `true` and whitespace-padded values remain disabled. The pipeline checks admission again when its background task starts.

## Docker — development-only adapter

`Dockerfile` builds the development image and keeps the `wrappers/docker_server.py` entry point. The image starts with execution disabled; opt in for local development:

```bash
# CPU
docker build -t video-transcoder:cpu .
docker run -p 8000:8000 --env-file .env -e VH_DEVELOPMENT_ADAPTER=1 video-transcoder:cpu

# GPU (requires the NVIDIA container runtime)
docker build --build-arg BASE=gpu -t video-transcoder:gpu .
docker run --gpus all -p 8000:8000 --env-file .env -e VH_DEVELOPMENT_ADAPTER=1 video-transcoder:gpu
```

`docker-compose.yml` is local development wiring. It explicitly sets `VH_DEVELOPMENT_ADAPTER: "1"` for `transcoder` and joins the external `lms-reverse-proxy` network for the local Traefik route `transcoder.lms`.

```bash
docker compose up -d
```

## Modal — staged second candidate

The Modal image, CPU web function and GPU worker definitions remain in `wrappers/modal_app.py` for future adapter work. `/transcode` returns 503 before reading the body or spawning work. A direct call to `process_transcode` raises the same staged refusal. `VH_DEVELOPMENT_ADAPTER=1` does not enable Modal; it is never the development adapter. `/health` remains available.

## Development adapter API (v1)

### `GET /health`

Returns `status: "ok"`, the detected `encoder` and `preset`, and the boolean `v1_execution_admitted`, including when execution is disabled.

### `POST /transcode`

Returns 503 with `{"detail":"The v1 execution path is a development-only adapter and is disabled."}` unless explicitly admitted. An admitted, valid request returns 202 and schedules the development pipeline. Invalid request fields return 422 before scheduling work. Sources are downloaded only from the specified S3 object; a failed download ends the job as failed. Auth: HMAC-SHA256 over `"{timestamp}.{raw_body}"` with the shared `WEBHOOK_SECRET` — headers `X-Signature: sha256=<hex>` + `X-Timestamp: <epoch>` (5-minute tolerance). Callbacks to Video Hub are signed the same way.

**Request:**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "qualities": ["720p", "480p", "360p"],
  "encryption_key_hex": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  "segment_duration": 6,
  "callback_url": "https://videohub.example.com/api/transcode/callback",
  "s3_bucket": "videohub-myorg",
  "s3_path_prefix": "videos/550e8400-...",
  "s3_original_path": "original-videos/550e8400-.../original.mp4"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `uuid` | yes | Video UUID (canonical form — validated) |
| `qualities` | yes | Quality names: `2160p`, `1440p`, `1080p`, `720p`, `480p`, `360p`, `240p` |
| `encryption_key_hex` | no | AES-128 key as hex (32 chars). Omit to skip encryption |
| `segment_duration` | no | HLS segment length in seconds (default: 6) |
| `callback_url` | yes | URL to POST progress + result callbacks to (HMAC-signed) |
| `s3_bucket` | yes | Nonempty S3 bucket name containing the source and outputs |
| `s3_path_prefix` | yes | S3 path prefix for uploaded HLS files |
| `s3_original_path` | yes | Nonempty S3 key of the existing original, downloaded directly |
| `encoder` / `preset` / `preset_level` | no | Encoder override (`h264_nvenc`, `libx264`, …) and preset/level 1-7 |

**Callback (success):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ready",
  "duration": 324,
  "qualities": [
    {"name": "720p", "width": 1280, "height": 720, "bitrate": 2500000, "playlist": "720p/playlist.m3u8"}
  ],
  "master_playlist": "master.m3u8",
  "thumbnail": "thumbnail.jpg"
}
```

**Callback (failure):**
```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error_message": "FFmpeg failed for 720p: ..."
}
```

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -q
```

## Quality Presets

The development adapter uses the presets in `core/config.py`:

| Preset | Resolution | Video Bitrate | Audio Bitrate |
|--------|-----------|---------------|---------------|
| 2160p | 3840×2160 | 15000k | 192k |
| 1440p | 2560×1440 | 10000k | 192k |
| 1080p | 1920×1080 | 5000k | 192k |
| 720p | 1280×720 | 2500k | 128k |
| 480p | 854×480 | 1200k | 96k |
| 360p | 640×360 | 600k | 64k |
| 240p | 426×240 | 300k | 48k |

Quality selection caps at source resolution — a 480p source will never be upscaled to 720p.
