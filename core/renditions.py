"""Which renditions a source gets, and the exact geometry, frame rate and H.264 level of each.

A rendition is named by the short edge of its display frame ("1080p" is 1920x1080 landscape or
1080x1920 portrait); its long edge follows the source's display aspect ratio, after the sample
aspect ratio and the rotation. A source is never upscaled: a rung above the source's short edge is
dropped, and when every requested rung is above it, the smallest requested rung is encoded at the
source's own size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from core.failure import Failure, WorkerFailure, make_detail
from core.profile import LadderRung, MediaProfile
from core.source_facts import VideoStreamFacts

MIN_OUTPUT_EDGE = 2  # 4:2:0 chroma needs even sides
MACROBLOCK_EDGE = 16
MILLI_PER_UNIT = 1000
H264_HIGH_PROFILE_IDC = 0x64
AAC_LC_CODECS_TAG = "mp4a.40.2"
# High profile may use 1.25 times the level's bit rate and coded picture buffer (H.264 Table A-2).
HIGH_PROFILE_RATE_FACTOR = Fraction(5, 4)
BITS_PER_KILOBIT = 1000
# A picture side may hold at most sqrt(8 * MaxFS) macroblocks (H.264 A.3.1).
LEVEL_SIDE_FACTOR = 8


@dataclass(frozen=True, slots=True)
class H264Level:
    name: str  # as x264 takes it, e.g. "4.2"
    level_idc: int  # 42
    max_macroblocks_per_s: int
    max_frame_macroblocks: int
    max_bitrate_kbps: int  # the Baseline/Main value; High may use 1.25 times
    max_cpb_kbits: int


# Levels 3.0 to 5.2 of H.264 Table A-1. Nothing below 3.0 is declared: every player decodes it.
H264_LEVELS: tuple[H264Level, ...] = (
    H264Level("3.0", 30, 40_500, 1_620, 10_000, 10_000),
    H264Level("3.1", 31, 108_000, 3_600, 14_000, 14_000),
    H264Level("3.2", 32, 216_000, 5_120, 20_000, 20_000),
    H264Level("4.0", 40, 245_760, 8_192, 20_000, 25_000),
    H264Level("4.1", 41, 245_760, 8_192, 50_000, 62_500),
    H264Level("4.2", 42, 522_240, 8_704, 50_000, 62_500),
    H264Level("5.0", 50, 589_824, 22_080, 135_000, 135_000),
    H264Level("5.1", 51, 983_040, 36_864, 240_000, 240_000),
    H264Level("5.2", 52, 2_073_600, 36_864, 240_000, 240_000),
)


@dataclass(frozen=True, slots=True)
class RenditionPlan:
    name: str
    width: int
    height: int
    frame_rate_num: int  # the constant output frame rate
    frame_rate_den: int
    gop_frames: int  # frames per segment, rounded up: one keyframe opens every segment
    video_bitrate: int
    video_maxrate: int
    video_bufsize: int
    audio_bitrate: int | None  # None when the source has no audio
    level: H264Level

    @property
    def codecs(self) -> str:
        """The RFC 6381 CODECS value of the rendition, e.g. "avc1.64002a,mp4a.40.2"."""
        video = f"avc1.{H264_HIGH_PROFILE_IDC:02x}00{self.level.level_idc:02x}"
        return video if self.audio_bitrate is None else f"{video},{AAC_LC_CODECS_TAG}"


def _even_floor(value: Fraction | int) -> int:
    return max(MIN_OUTPUT_EDGE, math.floor(value) // 2 * 2)


def even_round(value: Fraction) -> int:
    """The even integer nearest to `value`, at least 2: an output edge 4:2:0 chroma can carry."""
    return max(MIN_OUTPUT_EDGE, 2 * round(value / 2))


def _output_rate(video: VideoStreamFacts, profile: MediaProfile) -> Fraction:
    """The declared rate, held between the profile's floor and ceiling."""
    declared = Fraction(video.frame_rate_num, video.frame_rate_den)
    ceiling = Fraction(profile.encode.max_frame_rate_num, profile.encode.max_frame_rate_den)
    floor = Fraction(profile.encode.min_frame_rate_num, profile.encode.min_frame_rate_den)
    return max(min(declared, ceiling), floor)


def _macroblocks(edge: int) -> int:
    return -(-edge // MACROBLOCK_EDGE)


def _fits(level: H264Level, width: int, height: int, rate: Fraction, rung: LadderRung) -> bool:
    width_mbs, height_mbs = _macroblocks(width), _macroblocks(height)
    frame_mbs = width_mbs * height_mbs
    side_limit = LEVEL_SIDE_FACTOR * level.max_frame_macroblocks
    max_bitrate = level.max_bitrate_kbps * BITS_PER_KILOBIT * HIGH_PROFILE_RATE_FACTOR
    max_cpb = level.max_cpb_kbits * BITS_PER_KILOBIT * HIGH_PROFILE_RATE_FACTOR
    return (
        frame_mbs <= level.max_frame_macroblocks
        and width_mbs * width_mbs <= side_limit
        and height_mbs * height_mbs <= side_limit
        and frame_mbs * rate <= level.max_macroblocks_per_s
        and rung.video_maxrate <= max_bitrate
        and rung.video_bufsize <= max_cpb
    )


def _level_for(width: int, height: int, rate: Fraction, rung: LadderRung) -> H264Level:
    for level in H264_LEVELS:
        if _fits(level, width, height, rate, rung):
            return level
    raise WorkerFailure(
        Failure(
            error_class="input_limits_exceeded",
            code="h264_level_exceeded",
            retryable=False,
            detail=f"rendition {rung.name} at {width}x{height} exceeds every supported H.264 level",
        )
    )


def _geometry(video: VideoStreamFacts, short_edge: int | None) -> tuple[int, int]:
    """(width, height): scaled to `short_edge`, or the source's own size rounded down to even."""
    display_width, display_height = video.display_width, video.display_height
    source_short = min(display_width, display_height)
    source_long = max(display_width, display_height)
    if short_edge is None:
        return _even_floor(display_width), _even_floor(display_height)
    long_edge = min(even_round(Fraction(short_edge * source_long, source_short)), _even_floor(source_long))
    if display_width >= display_height:
        return long_edge, short_edge
    return short_edge, long_edge


def _requested_rungs(requested: Sequence[str], profile: MediaProfile) -> list[LadderRung]:
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence) or not requested:
        raise ValueError("requested must be a non-empty sequence of rendition names")
    if len(set(requested)) != len(requested):
        raise ValueError("requested holds duplicate rendition names")
    for name in requested:
        if not isinstance(name, str) or profile.encode.rung(name) is None:
            raise WorkerFailure(
                Failure(
                    error_class="configuration_error",
                    code="rendition_not_in_profile",
                    retryable=False,
                    detail=make_detail(f"the profile has no rendition named {name}"),
                )
            )
    wanted = set(requested)
    return [rung for rung in profile.encode.ladder if rung.name in wanted]


def plan_renditions(
    video: VideoStreamFacts, *, has_audio: bool, requested: Sequence[str], profile: MediaProfile
) -> tuple[RenditionPlan, ...]:
    """The renditions to encode, largest first, from the admitted video stream and the dispatch's list."""
    if not isinstance(video, VideoStreamFacts):
        raise TypeError("video must be a VideoStreamFacts")
    if not isinstance(has_audio, bool):
        raise TypeError("has_audio must be a bool")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    rungs = _requested_rungs(requested, profile)
    source_short = min(video.display_width, video.display_height)
    fitting = [(rung, rung.short_edge) for rung in rungs if rung.short_edge <= source_short]
    # Every requested rung would upscale: the smallest one is encoded at the source's own size.
    chosen: list[tuple[LadderRung, int | None]] = list(fitting) or [(rungs[-1], None)]

    rate = _output_rate(video, profile)
    gop_frames = max(1, math.ceil(profile.encode.segment_duration_s * rate))
    plans: list[RenditionPlan] = []
    for rung, short_edge in chosen:
        width, height = _geometry(video, short_edge)
        plans.append(
            RenditionPlan(
                name=rung.name,
                width=width,
                height=height,
                frame_rate_num=rate.numerator,
                frame_rate_den=rate.denominator,
                gop_frames=gop_frames,
                video_bitrate=rung.video_bitrate,
                video_maxrate=rung.video_maxrate,
                video_bufsize=rung.video_bufsize,
                audio_bitrate=rung.audio_bitrate if has_audio else None,
                level=_level_for(width, height, rate, rung),
            )
        )
    return tuple(plans)
