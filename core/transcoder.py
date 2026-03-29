from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core.config import QUALITY_PRESETS, Settings, parse_bitrate, select_qualities

logger = logging.getLogger(__name__)

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")


def probe_video(path: str) -> dict:
    """Get video info via ffprobe. Returns dict with duration, width, height, fps, etc."""
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    info = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream["codec_type"] == "video" and not video_stream:
            video_stream = stream
        elif stream["codec_type"] == "audio" and not audio_stream:
            audio_stream = stream

    if not video_stream:
        raise RuntimeError("No video stream found in file")

    return {
        "duration": int(float(info.get("format", {}).get("duration", 0))),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "fps": _parse_fps(video_stream.get("r_frame_rate", "30/1")),
        "has_audio": audio_stream is not None,
        "filesize": int(info.get("format", {}).get("size", 0)),
    }


def transcode_single_quality(
    input_path: str,
    output_dir: str,
    quality_name: str,
    preset: dict,
    video_info: dict,
    settings: Settings,
    key_info_path: str | None = None,
    segment_duration: int = 6,
) -> None:
    """Transcode input to a single HLS quality variant."""
    playlist_path = os.path.join(output_dir, "playlist.m3u8")
    segment_path = os.path.join(output_dir, "segment_%03d.ts")

    w, h = preset["width"], preset["height"]
    scale = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )

    fps = max(1, round(video_info["fps"] or 30))
    gop_size = segment_duration * fps

    # Build command — encoder-specific flags.
    cmd = [FFMPEG_PATH, "-i", input_path, "-y"]

    if settings.ffmpeg_encoder == "h264_nvenc":
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", settings.ffmpeg_preset,
            "-rc", "vbr",
            "-cq", "23",
            "-vf", scale,
        ]
    elif settings.ffmpeg_encoder == "h264_vaapi":
        cmd = [FFMPEG_PATH, "-vaapi_device", "/dev/dri/renderD128",
               "-i", input_path, "-y"]
        cmd += [
            "-vf", f"format=nv12,hwupload,scale_vaapi=w={w}:h={h}",
            "-c:v", "h264_vaapi",
            "-qp", "23",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", settings.ffmpeg_preset,
            "-profile:v", "main",
            "-level", "4.0",
            "-vf", scale,
        ]

    # Shared flags.
    cmd += [
        "-b:v", preset["bitrate"],
        "-maxrate", preset["maxrate"],
        "-bufsize", preset["bufsize"],
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_duration})",
        "-c:a", "aac",
        "-b:a", preset["audio_bitrate"],
        "-ar", "48000",
        "-ac", "2",
        "-f", "hls",
        "-hls_time", str(segment_duration),
        "-hls_list_size", "0",
        "-hls_segment_filename", segment_path,
        "-hls_playlist_type", "vod",
    ]

    if key_info_path:
        cmd += ["-hls_key_info_file", key_info_path]

    cmd.append(playlist_path)

    logger.info("Transcoding quality: %s", quality_name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # Include last 20 lines of stderr for debugging.
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(f"FFmpeg failed for {quality_name}: {stderr_tail}")


def generate_master_playlist(qualities: dict[str, dict]) -> str:
    """Generate HLS master playlist sorted by bitrate descending."""
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", ""]

    sorted_qualities = sorted(
        qualities.values(), key=lambda q: q["bitrate"], reverse=True
    )

    for q in sorted_qualities:
        bw = q["bitrate"]
        res = f"{q['width']}x{q['height']}"
        name = q["name"]
        lines.append(
            f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={res},NAME="{name}"'
        )
        lines.append(q["playlist"])
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_thumbnail(
    input_path: str, output_path: str, duration: int
) -> None:
    """Extract a thumbnail at 10% of video duration (max 5s)."""
    seek_time = min(duration * 0.1, 5)

    cmd = [
        FFMPEG_PATH,
        "-ss", str(seek_time),
        "-i", input_path,
        "-vframes", "1",
        "-vf", "scale=640:-1",
        "-q:v", "2",
        "-y", output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=False)


def create_key_info_file(
    quality_dir: str, key_hex: str
) -> str:
    """Write enc.key (raw bytes) and enc.keyinfo for FFmpeg HLS encryption."""
    key_bytes = bytes.fromhex(key_hex)
    key_path = os.path.join(quality_dir, "enc.key")
    key_info_path = os.path.join(quality_dir, "enc.keyinfo")

    with open(key_path, "wb") as f:
        f.write(key_bytes)

    # Line 1: URI to embed in m3u8 (rewritten by Video Hub's PlaybackController).
    # Line 2: Local path where FFmpeg reads the actual key.
    with open(key_info_path, "w") as f:
        f.write(f"enc.key\n{key_path}\n")

    return key_info_path


def transcode_to_hls(
    input_path: str,
    output_dir: str,
    settings: Settings,
    enabled_qualities: list[str] | None = None,
    encryption_key_hex: str | None = None,
    segment_duration: int = 6,
    progress_callback: callable | None = None,
) -> dict:
    """Full transcode pipeline: probe → per-quality HLS → master playlist → thumbnail.

    Returns dict with master_playlist, qualities, duration, thumbnail.
    """
    if enabled_qualities is None:
        enabled_qualities = ["1080p", "720p", "480p"]

    video_info = probe_video(input_path)
    source_height = video_info["height"]

    selected = select_qualities(enabled_qualities, source_height)

    os.makedirs(output_dir, exist_ok=True)

    # Transcode each quality.
    quality_info: dict[str, dict] = {}
    quality_names = list(selected.keys())
    total = len(quality_names)

    for i, name in enumerate(quality_names):
        preset = selected[name]

        if progress_callback:
            pct = int((i / total) * 100)
            progress_callback(pct, f"Transcoding {name} ({i + 1}/{total})")

        quality_dir = os.path.join(output_dir, name)
        os.makedirs(quality_dir, exist_ok=True)

        key_info_path = None
        if encryption_key_hex:
            key_info_path = create_key_info_file(quality_dir, encryption_key_hex)

        transcode_single_quality(
            input_path,
            quality_dir,
            name,
            preset,
            video_info,
            settings,
            key_info_path,
            segment_duration,
        )

        quality_info[name] = {
            "name": name,
            "width": preset["width"],
            "height": preset["height"],
            "bitrate": parse_bitrate(preset["bitrate"]),
            "playlist": f"{name}/playlist.m3u8",
        }

    if progress_callback:
        progress_callback(100, "Transcode complete")

    # Generate master playlist.
    master_content = generate_master_playlist(quality_info)
    master_path = os.path.join(output_dir, "master.m3u8")
    with open(master_path, "w") as f:
        f.write(master_content)

    # Generate thumbnail.
    thumbnail_path = os.path.join(output_dir, "thumbnail.jpg")
    generate_thumbnail(input_path, thumbnail_path, video_info["duration"])

    return {
        "master_playlist": "master.m3u8",
        "qualities": list(quality_info.values()),
        "duration": video_info["duration"],
        "thumbnail": "thumbnail.jpg",
    }


def cleanup(directory: str) -> None:
    """Remove a temp directory and all contents."""
    if os.path.isdir(directory):
        shutil.rmtree(directory)


def _parse_fps(fps_str: str) -> float:
    """Parse ffprobe frame rate string like '30/1' or '29.97'."""
    if "/" in fps_str:
        parts = fps_str.split("/")
        num, den = float(parts[0]), float(parts[1])
        return num / den if den > 0 else 30.0
    return float(fps_str) if fps_str else 30.0
