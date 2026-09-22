# Video Transcoder

Serverless video transcoding service. Receives jobs from Video Hub, transcodes with FFmpeg (GPU or CPU), uploads HLS segments to S3, sends callback on completion.

## Architecture

```
Video Hub (Laravel)                    Video Transcoder (Python)
  │                                      │
  │  POST /transcode                     │  FastAPI receives webhook
  │  (source_url, uuid, qualities, ...)  │  Downloads source from S3
  │                                      │  FFmpeg: h264_nvenc (GPU) or libx264 (CPU)
  │                                      │  Uploads HLS to S3
  │  ◄── POST callback ────────────────  │  Sends result back
  │  Updates DB: status=ready            │  Container dies (stateless)
```

- `core/` — provider-agnostic: FFmpeg logic, S3 upload, callback, FastAPI app
- `wrappers/` — one file per deployment target (Modal, Docker)
- Stateless — no database, all job tracking lives in Video Hub

## Requirements

- Python 3.12+
- FFmpeg (with NVENC support for GPU, or standard build for CPU)

## Local Development

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Edit `.env`:
```
FFMPEG_ENCODER=libx264
FFMPEG_PRESET=medium
WEBHOOK_SECRET=test-secret

# S3 credentials (optional for local testing)
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_ENDPOINT=
S3_REGION=auto
```

### 3. Run the server

```bash
python3 wrappers/docker_server.py
```

Server starts at `http://localhost:8000`.

### 4. Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

## Docker

### Build

```bash
# CPU (default — for Mac, VPS without GPU)
docker build -t video-transcoder:cpu .

# GPU (needs nvidia-docker runtime)
docker build --build-arg BASE=gpu -t video-transcoder:gpu .
```

### Run

```bash
# CPU
docker run -p 8000:8000 --env-file .env video-transcoder:cpu

# GPU
docker run --gpus all -p 8000:8000 --env-file .env video-transcoder:gpu
```

### Manual Test with Docker

1. Start the container:

```bash
docker run -p 8000:8000 --env-file .env video-transcoder:cpu
```

2. Serve a test video from your machine (separate terminal):

```bash
# Put a .mp4 file in this directory, then:
python3 -m http.server 9000
```

3. Send a transcode request (separate terminal):

```bash
# Auth is HMAC-SHA256 over "{timestamp}.{raw_body}" with WEBHOOK_SECRET.
# uuid must be a canonical UUID (validated — blocks path traversal).
BODY='{"uuid":"550e8400-e29b-41d4-a716-446655440000","source_url":"http://host.docker.internal:9000/sample.mp4","qualities":["480p","360p"],"segment_duration":6,"callback_url":"http://host.docker.internal:9999/fake","s3_bucket":"fake-bucket","s3_path_prefix":"videos/550e8400-e29b-41d4-a716-446655440000","s3_original_path":"original-videos/550e8400-e29b-41d4-a716-446655440000/original.mp4"}'
TS=$(date +%s)
SIG=$(BODY="$BODY" TS="$TS" python3 -c "import hmac,hashlib,os;print(hmac.new(b'test-secret',(os.environ['TS']+'.'+os.environ['BODY']).encode(),hashlib.sha256).hexdigest())")

curl -X POST http://localhost:8000/transcode \
  -H "X-Signature: sha256=$SIG" \
  -H "X-Timestamp: $TS" \
  -H "Content-Type: application/json" \
  -d "$BODY"
```

> `host.docker.internal` lets the container reach your Mac's localhost.

**Expected result:** Download and transcode succeed (check Docker logs). S3 upload and callback will fail without real credentials — that's fine for testing FFmpeg.

Fill in S3 credentials in `.env` and restart to test the full pipeline.

## Environment Configuration

Three deployment options. Each uses different `.env` values.

### Local Mac (no Traefik, no GPU)

For development and testing. No Traefik, no TLS, CPU encoding.

```env
# Transcoder
FFMPEG_ENCODER=libx264
FFMPEG_PRESET=medium
WEBHOOK_SECRET=test-secret

# S3 (optional — leave empty to test transcoding only)
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_ENDPOINT=
S3_REGION=auto

# Docker Compose (not needed if using `docker run`)
TRANSCODER_DOMAIN=transcoder.lms
TRANSCODER_BASE=cpu
TRAEFIK_ENTRYPOINTS=web
TRAEFIK_TLS_LABEL=traefik.enable=true
```

Run: `docker compose up -d` or `docker run -p 8000:8000 --env-file .env video-transcoder:cpu`

Video Hub reaches transcoder at: `http://localhost:8000` or `http://transcoder.lms` (if using Traefik + `/etc/hosts`)

### Ubuntu VPS (Traefik, optional GPU)

Same VPS as Video Hub. Traefik handles TLS and routing. Joins the shared `lms-reverse-proxy` network.

```env
# Transcoder
FFMPEG_ENCODER=libx264          # or h264_nvenc if VPS has GPU
FFMPEG_PRESET=medium            # or p4 for NVENC
WEBHOOK_SECRET=<generate-a-strong-secret>

# S3
S3_ACCESS_KEY_ID=<your-key>
S3_SECRET_ACCESS_KEY=<your-secret>
S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
S3_REGION=auto

# Docker Compose — Traefik with TLS
TRANSCODER_DOMAIN=transcoder.yourdomain.com
TRANSCODER_BASE=cpu             # or gpu
TRAEFIK_ENTRYPOINTS=websecure
TRAEFIK_TLS_LABEL=traefik.http.routers.transcoder.tls.certresolver=letsencrypt
```

Run: `docker compose up -d`

Video Hub reaches transcoder at: `https://transcoder.yourdomain.com`

DNS: Point `transcoder.yourdomain.com` to your VPS IP. Traefik auto-provisions the TLS cert.

### Modal (GPU — serverless)

No Docker, no Traefik, no VPS. Modal handles everything. Config lives in Modal secrets, not `.env`.

```bash
modal secret create video-transcoder-secrets \
  FFMPEG_ENCODER=h264_nvenc \
  FFMPEG_PRESET=p4 \
  S3_ACCESS_KEY_ID=<your-key> \
  S3_SECRET_ACCESS_KEY=<your-secret> \
  S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com \
  S3_REGION=auto \
  WEBHOOK_SECRET=<generate-a-strong-secret>
```

No `TRANSCODER_DOMAIN`, `TRAEFIK_*`, or `TRANSCODER_BASE` needed — Modal provides the URL and GPU automatically.

Deploy: `modal deploy wrappers/modal_app.py`

Video Hub reaches transcoder at: `https://your-org--video-transcoder.modal.run`

### Summary

| Setting | Local Mac | Ubuntu VPS | Modal |
|---------|-----------|------------|-------|
| `FFMPEG_ENCODER` | `libx264` | `libx264` or `h264_nvenc` | `h264_nvenc` |
| `FFMPEG_PRESET` | `medium` | `medium` or `p4` | `p4` |
| `WEBHOOK_SECRET` | `test-secret` | strong random | strong random |
| S3 creds | optional | required | required |
| `TRANSCODER_DOMAIN` | `transcoder.lms` | `transcoder.yourdomain.com` | n/a (Modal URL) |
| `TRANSCODER_BASE` | `cpu` | `cpu` or `gpu` | n/a |
| `TRAEFIK_ENTRYPOINTS` | `web` | `websecure` | n/a |
| TLS | none | Traefik + Let's Encrypt | Modal auto |
| Run command | `docker compose up -d` | `docker compose up -d` | `modal deploy` |

## Modal (GPU — Production)

### Setup

```bash
pip install modal
modal setup              # one-time auth
```

### Configure secrets

```bash
modal secret create video-transcoder-secrets \
  FFMPEG_ENCODER=h264_nvenc \
  FFMPEG_PRESET=p4 \
  S3_ACCESS_KEY_ID=xxx \
  S3_SECRET_ACCESS_KEY=xxx \
  S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com \
  S3_REGION=auto \
  WEBHOOK_SECRET=your-secret
```

### Deploy

```bash
modal deploy wrappers/modal_app.py
# → https://your-org--video-transcoder.modal.run
```

### Test deployment

```bash
modal serve wrappers/modal_app.py   # temporary dev URL for testing
```

## API

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /transcode`

Starts a transcode job. Auth: HMAC-SHA256 over `"{timestamp}.{raw_body}"` with the shared `WEBHOOK_SECRET` — headers `X-Signature: sha256=<hex>` + `X-Timestamp: <epoch>` (5-minute tolerance). Callbacks to Video Hub are signed the same way.

**Request:**
```json
{
  "uuid": "550e8400-...",
  "source_url": "https://...r2.cloudflarestorage.com/bucket/temp/uuid.mp4?X-Amz-...",
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
| `source_url` | yes | Presigned URL to download source video |
| `qualities` | yes | Quality names: `2160p`, `1440p`, `1080p`, `720p`, `480p`, `360p`, `240p` |
| `encryption_key_hex` | no | AES-128 key as hex (32 chars). Omit to skip encryption |
| `segment_duration` | no | HLS segment length in seconds (default: 6) |
| `callback_url` | yes | URL to POST progress + result callbacks to (HMAC-signed) |
| `s3_bucket` | yes | S3 bucket name |
| `s3_path_prefix` | yes | S3 path prefix for uploaded HLS files |
| `s3_original_path` | yes | S3 key of the original; downloaded directly when present, uploaded here when the source is external |
| `encoder` / `preset` / `preset_level` | no | Encoder override (`h264_nvenc`, `libx264`, …) and preset/level 1-7 |

**Callback (success):**
```json
{
  "uuid": "550e8400-...",
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
  "uuid": "550e8400-...",
  "status": "failed",
  "error_message": "FFmpeg failed for 720p: ..."
}
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Quality Presets

Matches Video Hub's `TranscoderService.php`:

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
