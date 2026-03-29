# Video Transcoder — TODO

Serverless video transcoding service on Modal.com. Receives jobs from Video Hub (Laravel), transcodes with GPU (h264_nvenc, H.264) on T4, uploads HLS to R2.

## Phase 1: Project Setup

- [ ] Initialize Python project (pyproject.toml / requirements.txt)
- [ ] Set up Modal account and CLI (`modal setup`)
- [ ] Create Modal app with FFmpeg + CUDA image
- [ ] Set up environment secrets in Modal (R2 credentials, callback auth)

## Phase 2: Core Transcoding

- [ ] Implement webhook endpoint (receives: source_url, uuid, qualities, callback_url)
- [ ] Download source video from R2 presigned URL
- [ ] FFmpeg transcode: `h264_nvenc -preset p4 -rc vbr -cq 23`, multi-quality HLS with AES-128 encryption
- [ ] Generate master.m3u8 playlist
- [ ] Generate thumbnail
- [ ] Upload HLS segments + playlists + thumbnail to R2 final location

## Phase 3: Integration with Video Hub

- [ ] Add presigned R2 upload URL endpoint in Video Hub (browser uploads directly to R2, skipping VPS)
- [ ] Add "upload complete" endpoint in Video Hub → triggers Video Transcoder webhook
- [ ] Add callback endpoint in Video Hub to receive transcoding completion (status: ready/failed)
- [ ] Update Video Hub upload form to use presigned URL flow (JS: get URL → PUT to R2 → notify complete)
- [ ] Clean up R2 temp file after transcoding completes
- [ ] Handle encryption key: Video Hub generates before dispatch, sends to transcoder

## Phase 4: Error Handling & Monitoring

- [ ] Retry logic for transient failures
- [ ] Timeout handling (Modal max 3600s)
- [ ] Error callback to Video Hub on failure (status: failed, error message)
- [ ] Logging for debugging

## Phase 5: Quality & Optimization

- [ ] Match current Video Hub quality presets (2160p–240p, per-library config)
- [ ] Benchmark h264_nvenc (GPU) vs current libx264 (CPU) output (file size, quality)
- [ ] Verify AES-128 encryption compatibility
- [ ] Future: consider H.265 (hevc_nvenc) when Firefox adds HEVC support

## Open Questions

- [ ] Should video transcoder probe video metadata (duration, resolution) and return it in callback?
- [ ] Fallback strategy if Modal is down? (Queue locally and retry later?)
- [ ] Upload progress UI: show progress bar for direct R2 upload from browser?
- [ ] Max file size limit for presigned upload URL?
