# RunPod Serverless Deployment

Deploy the Video Transcoder as a RunPod serverless GPU worker. Scale to zero when idle, pay only during transcoding.

## Architecture

```
Video Hub                        RunPod                           R2 Storage
   │                               │                                  │
   ├─POST /v2/{endpoint}/run──────►│ Queue job                        │
   │   (Bearer: RunPod API key)    │                                  │
   │                               ├─ Spin up GPU worker              │
   │                               │  (runpod_handler.py)             │
   │                               │                                  │
   │                               ├─ Download source ───────────────►│
   │                               ├─ FFmpeg transcode (GPU)          │
   │                               ├─ Upload HLS segments ───────────►│
   │◄─POST /api/transcode/callback─┤  (HMAC-signed)                   │
   │                               │                                  │
   │                               ├─ Container dies                  │
```

**Key difference from Modal/Docker**: RunPod handles inbound auth via API key, not HMAC. Outbound callbacks to Video Hub still use HMAC (shared `WEBHOOK_SECRET`).

## Prerequisites

- RunPod account (console.runpod.io)
- Docker Hub account (or any container registry)
- R2/S3 credentials (same as other deployments)
- `WEBHOOK_SECRET` (same shared secret used by Video Hub)

## Step 1: Build & Push Docker Image

Use the dedicated `Dockerfile.runpod` which includes GPU support (CUDA + jellyfin-ffmpeg + NVENC) and sets the RunPod handler as entrypoint:

```bash
cd ~/work/tbt/video-transcoder

docker build --platform linux/amd64 -f Dockerfile.runpod \
  -t yourdockerhub/video-transcoder-runpod:latest .

docker push yourdockerhub/video-transcoder-runpod:latest
```

## Step 2: Create RunPod Endpoint

1. Go to **console.runpod.io** > **Serverless** > **Endpoints** > **New Endpoint**

2. **Container Image**: `yourdockerhub/video-transcoder-runpod:latest`

3. **GPU Selection**:
   | GPU | VRAM | Cost/hr | Best for |
   |-----|------|---------|----------|
   | **L4** | 24GB | ~$0.30 | Cost-effective, good NVENC |
   | **4090 PRO** | 24GB | ~$0.55 | Faster NVENC |
   | **A4000** | 16GB | ~$0.20 | Budget option |

   Recommended: **L4** (best price/performance for NVENC transcoding)

4. **Scaling**:
   - **Active (Min) Workers**: `0` (scale to zero — $0 when idle)
   - **Max Workers**: `3` (handles 3 concurrent transcodes)
   - **GPUs per Worker**: `1`

5. **Timeouts**:
   - **Idle Timeout**: `5` seconds
   - **Execution Timeout**: `3600` seconds (1 hour — long videos)

6. **Environment Variables**:
   ```
   FFMPEG_ENCODER=h264_nvenc
   FFMPEG_PRESET=p4
   S3_ACCESS_KEY_ID=<your R2 access key>
   S3_SECRET_ACCESS_KEY=<your R2 secret key>
   S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
   S3_REGION=auto
   WEBHOOK_SECRET=<same secret as Video Hub>
   ```

7. Click **Deploy**. Note the **Endpoint ID** (e.g., `abc123xyz`).

## Step 3: Get RunPod API Key

1. Go to **Settings** > **API Keys**
2. Create a new key
3. Copy it — you'll configure it in Video Hub

## Step 4: Configure Video Hub

Add to Video Hub `.env` or admin settings:

```
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_ENDPOINT_ID=abc123xyz
```

### Set transcoder priority

In admin settings (`settings` table), set `transcoder_priority` to include RunPod:

```json
["runpod", "remote", "local"]
```

This tries RunPod first, falls back to remote Docker/VPS, then local CPU.

### Optional: fallback endpoint

If you have a second RunPod endpoint (different GPU, different region):

```
RUNPOD_FALLBACK_ENDPOINT_ID=def456uvw
```

The `RunPodDispatcher` will try the primary first, then the fallback.

## How It Works

### Job Flow

1. **Video Hub** dispatches job via `RunPodDispatcher`:
   ```
   POST https://api.runpod.ai/v2/{endpoint_id}/run
   Authorization: Bearer {RUNPOD_API_KEY}
   Body: {"input": {uuid, source_url, qualities, callback_url, ...}}
   ```

2. **RunPod** queues the job, spins up a GPU worker (cold start: 10-30s)

3. **Worker** (`runpod_handler.py`) runs `_process_transcode()`:
   - Downloads source from R2 presigned URL
   - Transcodes with FFmpeg NVENC (GPU)
   - Uploads HLS segments to R2
   - Sends HMAC-signed callback to Video Hub

4. **Video Hub** receives callback, sets video `status=ready`

5. **RunPod** shuts down worker after idle timeout (5s)

### Auth Model

| Direction | Auth method |
|-----------|-------------|
| Video Hub → RunPod | Bearer token (RunPod API key) |
| RunPod worker → R2 | S3 credentials (env vars) |
| RunPod worker → Video Hub callback | HMAC-SHA256 (`WEBHOOK_SECRET`) |

## Cost Estimate

- **L4 GPU**: ~$0.00031/sec (~$1.12/hr), billed per second
- Scale to zero = $0 when idle
- A 10-min video at 3 qualities takes ~2-5 min GPU time = ~$0.03-0.09
- 100 videos/month ≈ $3-9/month

Set **Active Workers = 1** ($0.30/hr idle cost) to eliminate cold starts if you need instant response.

## Monitoring

### Check job status via RunPod API

```bash
# Check specific job
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID

# Check endpoint health
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$ENDPOINT_ID/health
```

### Logs

- RunPod console: **Endpoints** > your endpoint > **Logs** tab
- Video Hub logs: check for `[runpod]` dispatcher messages

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Job stuck in queue | No workers available | Check max workers setting, GPU availability |
| Cold start too slow | Active workers = 0 | Set active workers to 1 (costs ~$0.30/hr) |
| Callback not received | Wrong WEBHOOK_SECRET | Verify same secret in RunPod env and Video Hub |
| Transcode fails | GPU/FFmpeg issue | Check RunPod logs for FFmpeg errors |
| 401 from RunPod API | Invalid API key | Regenerate key in RunPod settings |

## Files

| File | Location | Purpose |
|------|----------|---------|
| `runpod_handler.py` | `video-transcoder/wrappers/` | RunPod worker entry point |
| `RunPodDispatcher.php` | `video-hub/laravel/app/Services/Dispatchers/` | Video Hub → RunPod dispatcher |
| `TranscoderDispatchManager.php` | `video-hub/laravel/app/Services/Dispatchers/` | Priority-based dispatcher routing |
