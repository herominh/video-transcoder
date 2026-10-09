"""The versioned media profile: the worker's ceilings for a source and for probing it."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

PROFILE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{1,31}")
ALLOW_LIST_ENTRY_PATTERN = re.compile(r"[a-z0-9_]{1,32}")
PROFILE_VERSION_MIN = 1
PROFILE_VERSION_MAX = 65535

_INT_ANNOTATIONS = (int, "int")


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int_fields(instance: SourceCaps | ProbeBudget) -> None:
    """Every int field is a real int and at least 1: zero never means unlimited."""
    for field in fields(instance):
        if field.type not in _INT_ANNOTATIONS:
            continue
        value = getattr(instance, field.name)
        if not _is_strict_int(value) or value < 1:
            raise ValueError(f"{field.name} must be an int of at least 1, got {value!r}")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")


def _require_allow_list(name: str, entries: object) -> None:
    if not isinstance(entries, tuple) or not entries:
        raise ValueError(f"{name} must be a non-empty tuple")
    for entry in entries:
        if not isinstance(entry, str) or ALLOW_LIST_ENTRY_PATTERN.fullmatch(entry) is None:
            raise ValueError(f"{name} holds an invalid entry: {entry!r}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"{name} holds duplicate entries")


@dataclass(frozen=True, slots=True)
class SourceCaps:
    max_source_bytes: int
    max_duration_ms: int
    max_display_width: int
    max_display_height: int
    max_display_pixels: int
    max_frame_rate_milli: int  # frames per 1000 s; 60 fps = 60_000
    max_video_streams: int  # excluding attached pictures
    max_audio_streams: int
    max_total_streams: int  # every stream ffprobe lists
    allowed_demuxers: tuple[str, ...]  # libavformat demuxer short names, e.g. "mov"
    allowed_video_codecs: tuple[str, ...]  # ffprobe codec_name values
    reject_interlaced: bool
    reject_hdr: bool

    def __post_init__(self) -> None:
        _require_positive_int_fields(self)
        if self.max_display_pixels > self.max_display_width * self.max_display_height:
            raise ValueError("max_display_pixels exceeds max_display_width * max_display_height")
        if self.max_total_streams < self.max_video_streams + self.max_audio_streams:
            raise ValueError("max_total_streams is below max_video_streams + max_audio_streams")
        _require_allow_list("allowed_demuxers", self.allowed_demuxers)
        _require_allow_list("allowed_video_codecs", self.allowed_video_codecs)
        _require_bool("reject_interlaced", self.reject_interlaced)
        _require_bool("reject_hdr", self.reject_hdr)


@dataclass(frozen=True, slots=True)
class ProbeBudget:
    max_bytes: int  # aggregate bytes fetched from storage by one preflight
    max_requests: int  # aggregate storage requests
    max_wall_ms: int  # wall time of the whole preflight
    chunk_bytes: int  # storage fetch granularity
    max_output_bytes: int  # cap on ffprobe's stdout
    max_stderr_bytes: int  # tail of ffprobe's stderr kept
    probesize_bytes: int  # ffprobe -probesize
    analyze_duration_us: int  # ffprobe -analyzeduration

    def __post_init__(self) -> None:
        _require_positive_int_fields(self)
        if self.chunk_bytes > self.max_bytes:
            raise ValueError("chunk_bytes exceeds max_bytes")
        if self.probesize_bytes > self.max_bytes:
            raise ValueError("probesize_bytes exceeds max_bytes")


@dataclass(frozen=True, slots=True)
class MediaProfile:
    profile_id: str
    version: int
    source: SourceCaps
    probe: ProbeBudget

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or PROFILE_ID_PATTERN.fullmatch(self.profile_id) is None:
            raise ValueError(f"invalid profile_id: {self.profile_id!r}")
        if not _is_strict_int(self.version) or not PROFILE_VERSION_MIN <= self.version <= PROFILE_VERSION_MAX:
            raise ValueError(f"version must be an int from {PROFILE_VERSION_MIN} to {PROFILE_VERSION_MAX}")
        if not isinstance(self.source, SourceCaps):
            raise ValueError("source must be a SourceCaps")
        if not isinstance(self.probe, ProbeBudget):
            raise ValueError("probe must be a ProbeBudget")


PILOT_MAX_SOURCE_BYTES = 10 * GIB
PILOT_MAX_DURATION_MS = 21_600_000  # 6 hours
PILOT_MAX_DISPLAY_EDGE = 3840
PILOT_MAX_DISPLAY_PIXELS = 8_294_400  # 3840 x 2160
# 60 fps plus the jitter of variable-rate phone recordings.
PILOT_MAX_FRAME_RATE_MILLI = 61_000
PILOT_MAX_VIDEO_STREAMS = 1
PILOT_MAX_AUDIO_STREAMS = 4
PILOT_MAX_TOTAL_STREAMS = 16
PILOT_PROBE_MAX_BYTES = 32 * MIB
PILOT_PROBE_MAX_REQUESTS = 64
PILOT_PROBE_MAX_WALL_MS = 60_000
PILOT_PROBE_CHUNK_BYTES = 1 * MIB
PILOT_PROBE_MAX_OUTPUT_BYTES = 1 * MIB
# The probe fails closed when ffprobe's diagnostics overflow this cap, so it must hold an honest
# file's warnings.
PILOT_PROBE_MAX_STDERR_BYTES = 64 * KIB
PILOT_PROBESIZE_BYTES = 5_000_000
PILOT_ANALYZE_DURATION_US = 5_000_000

# The worker's ceilings. A dispatch's own limits carry the measured envelope of issue #50,
# and the effective limit is always the smaller of the two.
PILOT_PROFILE = MediaProfile(
    profile_id="pilot-h264-sdr",
    version=1,
    source=SourceCaps(
        max_source_bytes=PILOT_MAX_SOURCE_BYTES,
        max_duration_ms=PILOT_MAX_DURATION_MS,
        max_display_width=PILOT_MAX_DISPLAY_EDGE,
        max_display_height=PILOT_MAX_DISPLAY_EDGE,
        max_display_pixels=PILOT_MAX_DISPLAY_PIXELS,
        max_frame_rate_milli=PILOT_MAX_FRAME_RATE_MILLI,
        max_video_streams=PILOT_MAX_VIDEO_STREAMS,
        max_audio_streams=PILOT_MAX_AUDIO_STREAMS,
        max_total_streams=PILOT_MAX_TOTAL_STREAMS,
        allowed_demuxers=("avi", "matroska", "mov", "mpegts"),
        allowed_video_codecs=("av1", "h264", "hevc", "mpeg2video", "mpeg4", "prores", "vp8", "vp9"),
        reject_interlaced=True,
        reject_hdr=True,
    ),
    probe=ProbeBudget(
        max_bytes=PILOT_PROBE_MAX_BYTES,
        max_requests=PILOT_PROBE_MAX_REQUESTS,
        max_wall_ms=PILOT_PROBE_MAX_WALL_MS,
        chunk_bytes=PILOT_PROBE_CHUNK_BYTES,
        max_output_bytes=PILOT_PROBE_MAX_OUTPUT_BYTES,
        max_stderr_bytes=PILOT_PROBE_MAX_STDERR_BYTES,
        probesize_bytes=PILOT_PROBESIZE_BYTES,
        analyze_duration_us=PILOT_ANALYZE_DURATION_US,
    ),
)
