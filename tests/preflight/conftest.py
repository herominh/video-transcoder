"""Small media files generated once per session with the local ffmpeg (lavfi sources)."""

from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

FFMPEG_TIMEOUT_S = 60
FULL_DURATION_S = 2
SUBSECOND_DURATION_S = 0.4
HIGH_FPS_DURATION_S = 1
STANDARD_FPS = 25
HIGH_FPS = 120
LANDSCAPE_SIZE = "320x240"
PORTRAIT_SIZE = "240x320"
SMALL_SIZE = "160x120"
TONE_HZ = 440
EXTRA_AUDIO_STREAMS = 5
COVER_DURATION_S = 1
COVER_FPS = 1
VP9_CPU_USED = "8"  # fastest libvpx setting; quality does not matter here
MALFORMED_SEED = 20261009
MALFORMED_SIZE_BYTES = 64 * 1024
X264_FAST = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")
AAC = ("-c:a", "aac")


@dataclass(frozen=True, slots=True)
class MediaFixtures:
    mp4_faststart: Path
    mp4_moov_at_end: Path
    subsecond: Path
    silent: Path
    portrait: Path
    rotated_90: Path
    anamorphic: Path
    interlaced: Path
    hdr_pq: Path
    high_fps: Path
    many_audio: Path
    mkv: Path
    webm: Path | None  # None when the local ffmpeg lacks libvpx-vp9 or libopus
    mpegts: Path
    audio_with_cover: Path
    malformed: Path
    truncated_mp4: Path


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    )


def _video_input(size: str, duration_s: float, fps: int = STANDARD_FPS) -> tuple[str, ...]:
    return ("-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={duration_s}")


def _tone_input(duration_s: float) -> tuple[str, ...]:
    return ("-f", "lavfi", "-i", f"sine=frequency={TONE_HZ}:duration={duration_s}")


def _h264_aac(target: Path, size: str, *extra: str) -> Path:
    _ffmpeg(
        *_video_input(size, FULL_DURATION_S),
        *_tone_input(FULL_DURATION_S),
        *X264_FAST,
        *AAC,
        "-shortest",
        *extra,
        str(target),
    )
    return target


def _h264_only(target: Path, size: str, duration_s: float, *extra: str, fps: int = STANDARD_FPS) -> Path:
    _ffmpeg(*_video_input(size, duration_s, fps), *X264_FAST, "-an", *extra, str(target))
    return target


def _has_encoders(*names: str) -> bool:
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    ).stdout.decode("utf-8", "replace")
    available = {line.split()[1] for line in listing.splitlines() if len(line.split()) > 1}
    return all(name in available for name in names)


def _rotated_90(root: Path) -> Path:
    landscape = _h264_aac(root / "rotation_source.mp4", LANDSCAPE_SIZE)
    target = root / "rotated_90.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(landscape), "-c", "copy", str(target))
    return target


def _hdr_pq(root: Path) -> Path:
    # ffmpeg 8 takes the colour tags from the frames, so the output options alone leave the
    # transfer unset; setparams tags the frames themselves.
    return _h264_only(
        root / "hdr_pq.mp4",
        LANDSCAPE_SIZE,
        FULL_DURATION_S,
        "-vf",
        "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
    )


def _many_audio(root: Path) -> Path:
    target = root / "many_audio.mp4"
    audio_maps = [arg for _ in range(EXTRA_AUDIO_STREAMS) for arg in ("-map", "1:a")]
    _ffmpeg(
        *_video_input(SMALL_SIZE, FULL_DURATION_S),
        *_tone_input(FULL_DURATION_S),
        "-map",
        "0:v",
        *audio_maps,
        *X264_FAST,
        *AAC,
        "-shortest",
        str(target),
    )
    return target


def _webm(root: Path) -> Path | None:
    if not _has_encoders("libvpx-vp9", "libopus"):
        return None
    target = root / "vp9_opus.webm"
    _ffmpeg(
        *_video_input(LANDSCAPE_SIZE, FULL_DURATION_S),
        *_tone_input(FULL_DURATION_S),
        "-c:v",
        "libvpx-vp9",
        "-deadline",
        "realtime",
        "-cpu-used",
        VP9_CPU_USED,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "libopus",
        "-shortest",
        str(target),
    )
    return target


def _audio_with_cover(root: Path) -> Path:
    cover = root / "cover.jpg"
    _ffmpeg(*_video_input(SMALL_SIZE, COVER_DURATION_S, fps=COVER_FPS), "-frames:v", "1", str(cover))
    target = root / "audio_with_cover.mp3"
    _ffmpeg(
        *_tone_input(FULL_DURATION_S),
        "-i",
        str(cover),
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-c:a",
        "libmp3lame",
        "-c:v",
        "copy",
        "-id3v2_version",
        "3",
        "-disposition:v:0",
        "attached_pic",
        str(target),
    )
    return target


def _malformed(root: Path) -> Path:
    target = root / "malformed.mp4"
    target.write_bytes(random.Random(MALFORMED_SEED).randbytes(MALFORMED_SIZE_BYTES))
    return target


def _truncated(source: Path, root: Path) -> Path:
    target = root / "truncated.mp4"
    data = source.read_bytes()
    target.write_bytes(data[: len(data) // 2])
    return target


def _build_media(root: Path) -> MediaFixtures:
    moov_at_end = _h264_aac(root / "moov_at_end.mp4", LANDSCAPE_SIZE)
    return MediaFixtures(
        mp4_faststart=_h264_aac(root / "faststart.mp4", LANDSCAPE_SIZE, "-movflags", "+faststart"),
        mp4_moov_at_end=moov_at_end,
        subsecond=_h264_only(root / "subsecond.mp4", SMALL_SIZE, SUBSECOND_DURATION_S),
        silent=_h264_only(root / "silent.mp4", LANDSCAPE_SIZE, FULL_DURATION_S),
        portrait=_h264_aac(root / "portrait.mp4", PORTRAIT_SIZE),
        rotated_90=_rotated_90(root),
        anamorphic=_h264_only(root / "anamorphic.mp4", LANDSCAPE_SIZE, FULL_DURATION_S, "-vf", "setsar=4/3"),
        interlaced=_h264_only(
            root / "interlaced.mp4", LANDSCAPE_SIZE, FULL_DURATION_S, "-flags", "+ildct+ilme", "-x264-params", "tff=1"
        ),
        hdr_pq=_hdr_pq(root),
        high_fps=_h264_only(root / "high_fps.mp4", SMALL_SIZE, HIGH_FPS_DURATION_S, fps=HIGH_FPS),
        many_audio=_many_audio(root),
        mkv=_h264_aac(root / "h264_aac.mkv", LANDSCAPE_SIZE),
        webm=_webm(root),
        mpegts=_h264_aac(root / "h264_aac.ts", LANDSCAPE_SIZE),
        audio_with_cover=_audio_with_cover(root),
        malformed=_malformed(root),
        truncated_mp4=_truncated(moov_at_end, root),
    )


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> MediaFixtures:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required to generate the media fixtures")
    return _build_media(tmp_path_factory.mktemp("media"))
