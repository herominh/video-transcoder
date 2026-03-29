# Video Transcoder

Serverless GPU video transcoding service. Receives jobs from Video Hub, transcodes with FFmpeg, uploads HLS segments to R2, callbacks to Video Hub on completion.

## Architecture

- `core/` — provider-agnostic: FFmpeg logic, R2 upload, callback, FastAPI app
- `wrappers/` — one file per provider (Modal, Docker, RunPod)
- Stateless — no database, all job tracking lives in Video Hub's Postgres

## Running Locally

```bash
pip install -r requirements.txt
FFMPEG_ENCODER=libx264 FFMPEG_PRESET=medium WEBHOOK_SECRET=test python wrappers/docker_server.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Deploying to Modal

```bash
modal deploy wrappers/modal_app.py
```

## Webhook Contract

POST /transcode with JSON body (see core/api.py for schema). Auth via `Authorization: Bearer <WEBHOOK_SECRET>`.

## Key Conventions

- Video ID: UUID v4
- Quality presets match Video Hub's TranscoderService.php exactly
- Encryption: AES-128, key provided as hex in request, written to temp file for FFmpeg
- R2 upload skips .keyinfo and enc.key files (keys served from Video Hub DB)
