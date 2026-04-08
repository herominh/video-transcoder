from __future__ import annotations

import logging
import os

import boto3
import requests

from core.config import Settings

logger = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".key": "application/octet-stream",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
}

# Files to skip during upload — keys are served from Video Hub DB, not S3.
SKIP_SUFFIXES = (".keyinfo", "enc.key")


def download_source(presigned_url: str, dest_path: str) -> None:
    """Download source video from a presigned S3 URL using requests streaming."""
    logger.info("Downloading source to %s", dest_path)

    with requests.get(presigned_url, stream=True, timeout=(30, 3600)) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info("Download complete: %.1f MB", size_mb)


def upload_original(
    settings: Settings,
    local_path: str,
    bucket: str,
    s3_path: str,
) -> None:
    """Upload the original source file to S3."""
    client = _create_s3_client(settings)
    content_type = _get_content_type(os.path.basename(local_path))

    logger.info("Uploading original to %s/%s", bucket, s3_path)
    client.upload_file(
        local_path,
        bucket,
        s3_path,
        ExtraArgs={"ContentType": content_type},
    )

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info("Original uploaded: %.1f MB", size_mb)


def upload_results(
    settings: Settings,
    output_dir: str,
    bucket: str,
    path_prefix: str,
) -> int:
    """Upload transcoded files to S3. Returns count of uploaded files.

    Skips .keyinfo and enc.key files (encryption keys are served from DB).
    """
    client = _create_s3_client(settings)
    uploaded = 0

    for root, _dirs, files in os.walk(output_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, output_dir)

            if any(relative_path.endswith(s) for s in SKIP_SUFFIXES):
                continue

            s3_key = f"{path_prefix}/{relative_path}"
            content_type = _get_content_type(filename)

            client.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )
            uploaded += 1

    logger.info("Uploaded %d files to S3 bucket %s", uploaded, bucket)
    return uploaded


def _create_s3_client(settings: Settings):
    """Create a boto3 S3 client configured for S3-compatible storage."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region,
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )


def _get_content_type(filename: str) -> str:
    """Determine content type from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return CONTENT_TYPES.get(ext, "application/octet-stream")
