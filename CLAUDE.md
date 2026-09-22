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

POST /transcode with JSON body (see core/api.py for schema). Auth: HMAC-SHA256 over `"{timestamp}.{raw_body}"` with the shared `WEBHOOK_SECRET` — headers `X-Signature: sha256=<hex>` + `X-Timestamp: <epoch>` (5-min tolerance). The RunPod wrapper skips inbound HMAC (RunPod API key gates dispatch) but still signs outbound callbacks.

`_process_transcode` returns the result payload (status `ready`|`failed`); callback delivery is attempted but non-fatal. The RunPod handler returns that payload as the job output (and raises on failure so RunPod marks the job FAILED) — Video Hub polls job status as the reliable path.

## Key Conventions

- Video ID: UUID v4
- Quality presets match Video Hub's TranscoderService.php exactly
- Encryption: AES-128, key provided as hex in request, written to temp file for FFmpeg
- R2 upload skips .keyinfo and enc.key files (keys served from Video Hub DB)
