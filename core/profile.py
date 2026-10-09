"""The versioned media profile: the worker's ceilings for a source and for probing it."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from fractions import Fraction

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


def _require_positive_int_fields(instance: SourceCaps | ProbeBudget | LadderRung | EncodeProfile) -> None:
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


# Contract v2's rendition_name enum, largest first.
RENDITION_NAMES = ("2160p", "1440p", "1080p", "720p", "480p", "360p", "240p")
# Contract v2's limit on segments per rendition (limits.json, max_segments_per_rendition).
CONTRACT_MAX_SEGMENTS_PER_RENDITION = 3600
# What the encode stage can produce today. A GPU encoder joins only when Q2/D7 qualifies its resource.
SUPPORTED_VIDEO_ENCODERS = frozenset({"libx264"})
SUPPORTED_H264_PROFILES = frozenset({"high"})
SUPPORTED_PIXEL_FORMATS = frozenset({"yuv420p"})
SUPPORTED_AUDIO_CODECS = frozenset({"aac"})
X264_PRESETS = frozenset(
    {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}
)
MILLI_PER_UNIT = 1000
# mjpeg's -q:v scale: 2 is its best quality, 31 its worst.
MJPEG_QUALITY_MIN = 2
MJPEG_QUALITY_MAX = 31


def _require_member(name: str, value: object, allowed: frozenset[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")


@dataclass(frozen=True, slots=True)
class LadderRung:
    """One rendition the profile can produce, named by the short edge of its display frame."""

    name: str  # a contract v2 rendition name, e.g. "1080p"
    short_edge: int  # 1080 for "1080p": the shorter side of the output, portrait or landscape
    video_bitrate: int  # bits per second (-b:v)
    video_maxrate: int  # bits per second (-maxrate)
    video_bufsize: int  # bits (-bufsize)
    audio_bitrate: int  # bits per second (-b:a)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in RENDITION_NAMES:
            raise ValueError(f"name must be one of {RENDITION_NAMES}, got {self.name!r}")
        _require_positive_int_fields(self)
        if self.short_edge % 2 != 0:
            raise ValueError("short_edge must be even")
        if self.video_bitrate > self.video_maxrate:
            raise ValueError("video_bitrate exceeds video_maxrate")


@dataclass(frozen=True, slots=True)
class EncodeProfile:
    """How every rendition is encoded: the output format, the ladder and the stage's own bounds."""

    video_encoder: str
    encoder_preset: str
    h264_profile: str
    pixel_format: str
    max_frame_rate_num: int  # the output frame rate ceiling, as a fraction: 60/1
    max_frame_rate_den: int
    # The output frame rate floor: a slide show declaring 1/10 fps still gets a frame, and so a
    # keyframe, at every segment boundary.
    min_frame_rate_num: int
    min_frame_rate_den: int
    segment_duration_s: int  # the platform-wide HLS segment duration
    max_segments_per_rendition: int
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    ladder: tuple[LadderRung, ...]  # largest first
    max_scratch_bytes: int  # the source plus every output of the job
    max_playlist_bytes: int  # cap on reading back a media playlist the encoder wrote
    max_stderr_bytes: int  # tail of the encoder's stderr kept
    output_check_interval_ms: int  # how often the running encode's output is measured
    # Each rendition's wall-time budget: a base plus so much per second of admitted media. No count
    # of packets or frames bounds what a decoder does (an AV1 temporal unit or a VP9 superframe
    # carries any number of frames, hidden ones included), so time is the decoder's bound, and the
    # one the provider bills: a source that lies about its density costs at most this.
    encode_base_wall_ms: int
    encode_wall_ms_per_media_s: int
    # The thumbnail's requirements (`core/thumbnail.py`).
    thumbnail_short_edge: int  # the thumbnail's short edge (even); a smaller rendition is never upscaled
    thumbnail_quality: int  # mjpeg -q:v, from 2 (best) to 31
    thumbnail_max_bytes: int  # the room reserved for the thumbnail; a larger JPEG is refused
    thumbnail_wall_ms: int  # the thumbnail's own time budget, never past the caller's deadline
    # Taken at this share of the admitted duration, below 1000 (a time at the very end names an output
    # frame that does not exist). Below 1000 is not enough on its own: the admitted duration is the
    # longest stream's, often the audio's, so a share close to 1000 can still fall after the video's
    # last frame and leave the job without a thumbnail. The pilot's 10 % keeps far from that end...
    thumbnail_at_per_mille: int
    thumbnail_at_max_ms: int  # ...but never later than this

    def __post_init__(self) -> None:
        _require_positive_int_fields(self)
        _require_member("video_encoder", self.video_encoder, SUPPORTED_VIDEO_ENCODERS)
        _require_member("encoder_preset", self.encoder_preset, X264_PRESETS)
        _require_member("h264_profile", self.h264_profile, SUPPORTED_H264_PROFILES)
        _require_member("pixel_format", self.pixel_format, SUPPORTED_PIXEL_FORMATS)
        _require_member("audio_codec", self.audio_codec, SUPPORTED_AUDIO_CODECS)
        if self.max_segments_per_rendition > CONTRACT_MAX_SEGMENTS_PER_RENDITION:
            raise ValueError(
                f"max_segments_per_rendition exceeds the contract's {CONTRACT_MAX_SEGMENTS_PER_RENDITION}"
            )
        if not isinstance(self.ladder, tuple) or not self.ladder:
            raise ValueError("ladder must be a non-empty tuple")
        if not all(isinstance(rung, LadderRung) for rung in self.ladder):
            raise ValueError("ladder must hold LadderRung entries")
        if len({rung.name for rung in self.ladder}) != len(self.ladder):
            raise ValueError("ladder holds duplicate rendition names")
        floor = Fraction(self.min_frame_rate_num, self.min_frame_rate_den)
        if floor > Fraction(self.max_frame_rate_num, self.max_frame_rate_den):
            raise ValueError("the output frame rate floor exceeds its ceiling")
        if floor * self.segment_duration_s < 1:
            raise ValueError("the output frame rate floor leaves a segment without a frame")
        edges = [rung.short_edge for rung in self.ladder]
        if any(larger <= smaller for larger, smaller in zip(edges, edges[1:])):
            raise ValueError("ladder must be ordered by short_edge, largest first, without ties")
        if self.thumbnail_short_edge % 2 != 0:
            raise ValueError("thumbnail_short_edge must be even")
        if not MJPEG_QUALITY_MIN <= self.thumbnail_quality <= MJPEG_QUALITY_MAX:
            raise ValueError(f"thumbnail_quality must be from {MJPEG_QUALITY_MIN} to {MJPEG_QUALITY_MAX}")
        if self.thumbnail_at_per_mille >= MILLI_PER_UNIT:
            raise ValueError(f"thumbnail_at_per_mille must be below {MILLI_PER_UNIT}")

    @property
    def max_frame_rate_milli(self) -> int:
        return -(-self.max_frame_rate_num * MILLI_PER_UNIT // self.max_frame_rate_den)

    def rung(self, name: str) -> LadderRung | None:
        return next((rung for rung in self.ladder if rung.name == name), None)


@dataclass(frozen=True, slots=True)
class MediaProfile:
    profile_id: str
    version: int
    source: SourceCaps
    probe: ProbeBudget
    encode: EncodeProfile

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or PROFILE_ID_PATTERN.fullmatch(self.profile_id) is None:
            raise ValueError(f"invalid profile_id: {self.profile_id!r}")
        if not _is_strict_int(self.version) or not PROFILE_VERSION_MIN <= self.version <= PROFILE_VERSION_MAX:
            raise ValueError(f"version must be an int from {PROFILE_VERSION_MIN} to {PROFILE_VERSION_MAX}")
        if not isinstance(self.source, SourceCaps):
            raise ValueError("source must be a SourceCaps")
        if not isinstance(self.probe, ProbeBudget):
            raise ValueError("probe must be a ProbeBudget")
        if not isinstance(self.encode, EncodeProfile):
            raise ValueError("encode must be an EncodeProfile")
        # The longest admitted source must fit the segment limit, and the output frame rate
        # ceiling must not exceed what admission lets in.
        max_encoded_ms = self.encode.segment_duration_s * MILLI_PER_UNIT * self.encode.max_segments_per_rendition
        if self.source.max_duration_ms > max_encoded_ms:
            raise ValueError("max_duration_ms exceeds segment_duration_s * max_segments_per_rendition")
        if self.encode.max_frame_rate_milli > self.source.max_frame_rate_milli:
            raise ValueError("the output frame rate ceiling exceeds the source frame rate cap")
        if self.encode.max_scratch_bytes <= self.source.max_source_bytes:
            raise ValueError("max_scratch_bytes leaves no room for output beside the largest source")


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
PILOT_MAX_OUTPUT_FRAME_RATE = 60
PILOT_MIN_OUTPUT_FRAME_RATE = 1
PILOT_SEGMENT_DURATION_S = 6
PILOT_AUDIO_SAMPLE_RATE = 48_000
PILOT_AUDIO_CHANNELS = 2
# The worker's own ceiling on scratch use; the execution resource Q2 qualifies needs at least this
# much disk, or a full disk ends the job first.
PILOT_MAX_SCRATCH_BYTES = 64 * GIB
# 3,600 segments, each with its own key line, take about 420 KiB.
PILOT_MAX_PLAYLIST_BYTES = 1 * MIB
PILOT_ENCODE_MAX_STDERR_BYTES = 64 * KIB
PILOT_OUTPUT_CHECK_INTERVAL_MS = 250
# Unmeasured until Q2 qualifies the execution resource: x264 "medium" at 2160p60 on 8 vCPUs runs
# at roughly 5 to 7.5 s of wall time per second of media, so 20 s leaves about 3x headroom for the
# largest rung and far more for the others; the dispatch's own deadline still caps the whole job.
PILOT_ENCODE_BASE_WALL_MS = 120_000
PILOT_ENCODE_WALL_MS_PER_MEDIA_S = 20_000
# The thumbnail: 360 lines (640x360 for a 16:9 landscape source) at mjpeg's best quality, taken a
# tenth into the video but never later than 5 s, so a long source is not decoded for minutes to
# reach it. A 640x360 JPEG at that quality takes tens of KiB; 1 MiB leaves room for noisy frames
# and extreme aspect ratios.
PILOT_THUMBNAIL_SHORT_EDGE = 360
PILOT_THUMBNAIL_QUALITY = 2
PILOT_THUMBNAIL_MAX_BYTES = 1 * MIB
PILOT_THUMBNAIL_WALL_MS = 60_000
PILOT_THUMBNAIL_AT_PER_MILLE = 100
PILOT_THUMBNAIL_AT_MAX_MS = 5_000
# The bitrates of the Hub's quality presets (and the draft worker's), one rung per contract name.
PILOT_LADDER = (
    LadderRung("2160p", 2160, 15_000_000, 16_000_000, 22_500_000, 192_000),
    LadderRung("1440p", 1440, 10_000_000, 10_700_000, 15_000_000, 192_000),
    LadderRung("1080p", 1080, 5_000_000, 5_350_000, 7_500_000, 192_000),
    LadderRung("720p", 720, 2_500_000, 2_675_000, 3_750_000, 128_000),
    LadderRung("480p", 480, 1_200_000, 1_280_000, 1_800_000, 96_000),
    LadderRung("360p", 360, 600_000, 640_000, 900_000, 64_000),
    LadderRung("240p", 240, 300_000, 320_000, 450_000, 48_000),
)

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
    encode=EncodeProfile(
        video_encoder="libx264",
        encoder_preset="medium",
        h264_profile="high",
        pixel_format="yuv420p",
        max_frame_rate_num=PILOT_MAX_OUTPUT_FRAME_RATE,
        max_frame_rate_den=1,
        min_frame_rate_num=PILOT_MIN_OUTPUT_FRAME_RATE,
        min_frame_rate_den=1,
        segment_duration_s=PILOT_SEGMENT_DURATION_S,
        max_segments_per_rendition=CONTRACT_MAX_SEGMENTS_PER_RENDITION,
        audio_codec="aac",
        audio_sample_rate=PILOT_AUDIO_SAMPLE_RATE,
        audio_channels=PILOT_AUDIO_CHANNELS,
        ladder=PILOT_LADDER,
        max_scratch_bytes=PILOT_MAX_SCRATCH_BYTES,
        max_playlist_bytes=PILOT_MAX_PLAYLIST_BYTES,
        max_stderr_bytes=PILOT_ENCODE_MAX_STDERR_BYTES,
        output_check_interval_ms=PILOT_OUTPUT_CHECK_INTERVAL_MS,
        encode_base_wall_ms=PILOT_ENCODE_BASE_WALL_MS,
        encode_wall_ms_per_media_s=PILOT_ENCODE_WALL_MS_PER_MEDIA_S,
        thumbnail_short_edge=PILOT_THUMBNAIL_SHORT_EDGE,
        thumbnail_quality=PILOT_THUMBNAIL_QUALITY,
        thumbnail_max_bytes=PILOT_THUMBNAIL_MAX_BYTES,
        thumbnail_wall_ms=PILOT_THUMBNAIL_WALL_MS,
        thumbnail_at_per_mille=PILOT_THUMBNAIL_AT_PER_MILLE,
        thumbnail_at_max_ms=PILOT_THUMBNAIL_AT_MAX_MS,
    ),
)
