# RunPod Serverless Deployment

Deploy the Video Transcoder as a RunPod serverless GPU worker. Scale to zero when idle, pay only during transcoding.

> **Read this first (2026-10-11).** Since D1's last slice the image's entry point (`wrappers/runpod_handler.py`) speaks
> **transcode contract v2 only**. An image built from this commit cannot serve the Video Hub of today, in one of two
> ways: on an endpoint that still has only the v1 variables the worker does not start at all (it stops on the first
> missing `VH_*` variable, so jobs wait in the queue), and with the v2 variables set it refuses the v1 job input the Hub
> still sends. Build and publish it only together with the Hub release that dispatches contract v2. Only the next
> section, "The worker under contract v2", describes what this image really does. EVERYTHING after it (from
> "Architecture" to "Files": the diagram, the prerequisites, the variables of Step 2, the job flow, the auth model,
> the troubleshooting table) is the v1 text, kept until that release rewrites it.

## The worker under contract v2

What one job is (`wrappers/runpod_handler.py`, `core/protocol/handler.py`, `core/job/runner.py`; the rules are the
contract's `tests/contracts/transcode/v2/README.md`):

1. **Start of the process.** Before any job is taken the worker checks the contract tree the image ships and reads its
   whole configuration. A tree or a configuration it cannot trust stops the process; the error names the file or the
   variable and the rule, never a value. It then removes what a killed process left in the scratch directory: every
   directory there that a worker of this host made (its name, `vh-job-*`, carries a tag made from the boot id of
   the host's running kernel), that the worker's own user owns, whose own modification time is at least a minute
   old and that no running job holds locked. Nothing else in the scratch directory is touched: not a directory
   another host made, and not one this host made before its last reboot.
2. **The job input** is `{vh_message, vh_signature}`: a `transcode.request` signed by the Hub. Nothing of it is trusted
   until the signature verifies under a Hub-to-worker key. A job the worker may not act on fails with the refusal's own
   words (step, layer, reason) and nothing of the input.
3. **The claim.** The worker claims the dispatch at the Hub's claim address with a fresh random runtime id. Only a
   signed grant lets it work; every other ending is a signed `transcode.unclaimed` and no work. The grant carries the
   generation's media key; the job input never does.
4. **The job.** Preflight, download, encode, package, upload under the granted prefix, the manifest last.
5. **The result** (`transcode.result.completed` or `.failed`, signed) is the job's output, `{vh_message, vh_signature}`,
   and is also posted to the Hub's callback address (best effort). The Hub can always read the job output.
6. **One wall-time budget.** The dispatch's `limits.max_wall_time_ms` counts from the moment the worker begins to
   handle the job and bounds the claim, the job and the callback. Set RunPod's execution timeout above it, with a
   margin, so that the worker can still report a typed failure.

Variables of the v2 worker (set on the endpoint; the v1 variables of Step 2 are not read by this entry point):

- the `VH_*` protocol variables listed in the docstring of `core/protocol/settings.py` (accepted versions, audience,
  the signing keys of both directions, the claim and callback URLs, the clock skew, and one storage profile per
  `credential_ref`: endpoint, region and credentials);
- `VH_IMAGE_DIGEST` (required): `sha256:` and the 64 hex digits of the image the endpoint runs; a container cannot
  learn its own digest, and a result reports it. Set the endpoint's container image by that same digest
  (`<repository>@sha256:<digest>`, not a tag such as `:latest`) and change the two together: a tag can be moved to
  another image while every result goes on reporting the old digest;
- `VH_SCRATCH_DIR` (optional): the absolute path of an existing directory for the jobs' scratch files, which the
  worker's process can write to and search; it holds printable ASCII only, without `:`, `%`, a double quote or a
  backslash; a usable system temporary directory (`TMPDIR`) is required whatever the scratch directory is. The
  scratch directory (this one, or the system's temporary directory when the variable is not set) must be on a file
  system on which a directory can be locked with `flock`: every job locks its own directory in it, and the worker
  refuses to start where a directory cannot be locked. Keep it on the worker's own disk, never on a volume that
  workers share. A lock is seen only on the host that holds it, so a worker never removes a directory another host
  made, and what a killed worker of another host left there (or this host before a reboot) stays until someone
  removes it by hand. And a shared file system that every worker mounts for itself can keep their locks apart even
  on one host, where a starting worker then cannot tell a sibling's running job from a leftover: the start cannot
  check that;
- `FFMPEG_PATH` and `FFPROBE_PATH` (optional): the tools the stages run, when they are not `ffmpeg` and `ffprobe` on
  the `PATH` (the image links both into `/usr/local/bin`).

Things an operator should know:

- **What RunPod records for a failed job.** Only the worker's own words: `JobRefused` with the refusal's text, or
  `WorkerFault` naming the class of an unexpected error. The worker's log has the class and the stack of a bug, never
  an error's own text.
- **`refresh_worker`.** Once a storage request had to be abandoned, at the job's deadline or at the request's own time
  ceiling (so also in a job that then completed after a repeat), every job of that process asks
  RunPod to stop the pod when it returns (the SDK takes that flag out of the output), so that a stuck request's thread
  and connection end with the process instead of piling up in a warm worker. A stuck request can outlive the job it
  belonged to until then.
- **The SDK version matters.** The entry point relies on three behaviours read from RunPod SDK 1.10.1: the
  `refresh_worker` member is taken out of the output, a failed job is recorded with the error's text and traceback,
  and one job runs at a time. `requirements.txt` allows a range, so run the whole suite in every rebuilt image (five of
  its test cases drive the handler through the SDK's own `run_job`; the command is below), or pin the version when the
  image is built.
- **Logs.** The RunPod SDK prints each job's output at its default DEBUG level. A result or an unclaimed report holds no
  secret. Do not add logging of the job input: the dispatch carries a one-time token. The worker's own lines never name
  an endpoint, a bucket, a key or a path; for the same reason the entry point keeps every line of urllib3, the HTTP
  library under the storage clients, out of the log (its warnings name the storage host, and two of them the request's
  whole URL, so the bucket and the object key). A storage that answers with malformed headers or fills a connection
  pool therefore leaves no line of urllib3's; a request that fails is still logged in the worker's own words.
- **Python.** The image starts the worker with `python3`, which is Python 3.10 there; the suite is run in the image
  for that reason.
- **Running the whole suite in a rebuilt image.** The image ships what the worker runs and none of the tests, so the
  repository's tree is mounted read-only and copied into the container without `.git`, `.claude`, `__pycache__` and
  `.env`, and the suite runs on the copy. From the repository's root, with the rebuilt image in the place of `<image>`:

  ```bash
  docker run --rm --init -v "$PWD":/src:ro <image> bash -c '
    mkdir /work && cd /src &&
    tar --exclude=./.git --exclude=./.claude --exclude=__pycache__ --exclude=./.env -cf - . | tar -xf - -C /work &&
    cd /work && pip install -r requirements-dev.txt &&
    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests -q -p no:cacheprovider'
  ```

  It must end with no failure; the five cases that drive the handler through the SDK run there (they are skipped
  where the SDK is not installed). Keep `--init`: some tests of `tests/preflight/test_bounded_process.py` (those in
  which a killed grandchild must be gone) need a first process in the container that reaps killed children. A
  `bash -c` that stays the container's first process reaps them as well, but bash runs the last command of its line
  in its own place, so here pytest would be the first process; without `--init`, end the line with a pipeline
  (`... 2>&1 | tail -25`), for which bash stays.

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
