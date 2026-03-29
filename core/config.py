from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


QUALITY_PRESETS: dict[str, dict] = {
    "2160p": {
        "width": 3840,
        "height": 2160,
        "bitrate": "15000k",
        "audio_bitrate": "192k",
        "maxrate": "16000k",
        "bufsize": "22500k",
    },
    "1440p": {
        "width": 2560,
        "height": 1440,
        "bitrate": "10000k",
        "audio_bitrate": "192k",
        "maxrate": "10700k",
        "bufsize": "15000k",
    },
    "1080p": {
        "width": 1920,
        "height": 1080,
        "bitrate": "5000k",
        "audio_bitrate": "192k",
        "maxrate": "5350k",
        "bufsize": "7500k",
    },
    "720p": {
        "width": 1280,
        "height": 720,
        "bitrate": "2500k",
        "audio_bitrate": "128k",
        "maxrate": "2675k",
        "bufsize": "3750k",
    },
    "480p": {
        "width": 854,
        "height": 480,
        "bitrate": "1200k",
        "audio_bitrate": "96k",
        "maxrate": "1280k",
        "bufsize": "1800k",
    },
    "360p": {
        "width": 640,
        "height": 360,
        "bitrate": "600k",
        "audio_bitrate": "64k",
        "maxrate": "640k",
        "bufsize": "900k",
    },
    "240p": {
        "width": 426,
        "height": 240,
        "bitrate": "300k",
        "audio_bitrate": "48k",
        "maxrate": "320k",
        "bufsize": "450k",
    },
}


def parse_bitrate(bitrate: str) -> int:
    """Convert bitrate string like '2500k' or '5m' to integer bits/s."""
    bitrate = bitrate.lower().strip()
    if bitrate.endswith("k"):
        return int(bitrate[:-1]) * 1000
    if bitrate.endswith("m"):
        return int(bitrate[:-1]) * 1000000
    return int(bitrate)


def select_qualities(
    requested: list[str], source_height: int
) -> dict[str, dict]:
    """Filter quality presets by source resolution.

    Returns only presets whose height <= source_height.
    Always includes at least the lowest requested quality as fallback.
    """
    selected: dict[str, dict] = {}

    for name in requested:
        preset = QUALITY_PRESETS.get(name)
        if preset and preset["height"] <= source_height:
            selected[name] = preset

    if not selected:
        # Pick the highest preset that doesn't upscale (height <= source).
        # If source is smaller than all presets, use the lowest preset available.
        for name in sorted(
            QUALITY_PRESETS, key=lambda n: QUALITY_PRESETS[n]["height"], reverse=True
        ):
            if QUALITY_PRESETS[name]["height"] <= source_height:
                selected[name] = QUALITY_PRESETS[name]
                break
        if not selected:
            selected["240p"] = QUALITY_PRESETS["240p"]

    return selected


ENCODER_PRIORITY = [
    ("h264_nvenc", "p4"),       # NVIDIA GPU
    ("h264_vaapi", ""),         # Intel/AMD GPU (Linux VA-API)
    ("libx264", "medium"),      # CPU fallback
]


def detect_encoder() -> tuple[str, str]:
    """Detect the best available H.264 encoder.

    Tests encoders in priority order by running a minimal FFmpeg encode.
    Returns (encoder, default_preset).
    """
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")

    for encoder, default_preset in ENCODER_PRIORITY:
        cmd = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.1",
            "-c:v", encoder,
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=10, check=False,
            )
            if result.returncode == 0:
                logger.info("Detected encoder: %s", encoder)
                return encoder, default_preset
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    logger.warning("No hardware encoder found, falling back to libx264")
    return "libx264", "medium"


# Cache detection result at module load so it runs once.
_detected_encoder, _detected_preset = detect_encoder()

# Cache of encoder availability checks.
_available_encoders: dict[str, bool] = {_detected_encoder: True}


def _check_encoder_available(encoder: str) -> bool:
    """Check if a specific encoder works on this machine (cached)."""
    if encoder in _available_encoders:
        return _available_encoders[encoder]

    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg")
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.1",
        "-c:v", encoder,
        "-frames:v", "1",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
        available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        available = False

    _available_encoders[encoder] = available
    return available


def resolve_encoder(
    requested_encoder: str | None,
    requested_preset: str | None,
) -> tuple[str, str]:
    """Resolve requested encoder/preset with fallback.

    If requested encoder is available, use it.
    Otherwise fall back to the auto-detected encoder.
    Preset defaults per encoder if not specified.
    """
    # Default presets per encoder.
    default_presets = {e: p for e, p in ENCODER_PRIORITY}

    if requested_encoder and _check_encoder_available(requested_encoder):
        encoder = requested_encoder
        preset = requested_preset or default_presets.get(encoder, "")
        logger.info("Using requested encoder: %s (preset: %s)", encoder, preset)
    else:
        if requested_encoder:
            logger.warning(
                "Requested encoder %s not available, falling back to %s",
                requested_encoder, _detected_encoder,
            )
        encoder = _detected_encoder
        preset = requested_preset or _detected_preset

    return encoder, preset


@dataclass(frozen=True)
class Settings:
    ffmpeg_encoder: str = "h264_nvenc"
    ffmpeg_preset: str = "p4"
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_endpoint: str = ""
    r2_region: str = "auto"
    webhook_secret: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        encoder = os.environ.get("FFMPEG_ENCODER") or _detected_encoder
        preset = os.environ.get("FFMPEG_PRESET") or _detected_preset

        return cls(
            ffmpeg_encoder=encoder,
            ffmpeg_preset=preset,
            r2_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
            r2_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            r2_endpoint=os.environ.get("R2_ENDPOINT", ""),
            r2_region=os.environ.get("R2_REGION", "auto"),
            webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
        )
