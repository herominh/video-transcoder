"""Decide whether a probed source fits the media profile and the dispatch's own limits."""

from __future__ import annotations

from dataclasses import dataclass

from core.failure import Failure, make_detail
from core.profile import MediaProfile
from core.source_facts import SourceFacts, VideoStreamFacts

INPUT_INVALID = "input_invalid"
INPUT_LIMITS_EXCEEDED = "input_limits_exceeded"
DEMUXER_NAME_SEPARATOR = ","
MILLI_PER_UNIT = 1000
INTERLACED_FIELD_ORDERS = frozenset({"tt", "bb", "tb", "bt"})
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class RequestLimits:
    max_source_bytes: int  # from the dispatch's limits
    max_source_duration_ms: int

    def __post_init__(self) -> None:
        for name in ("max_source_bytes", "max_source_duration_ms"):
            value = getattr(self, name)
            if not _is_strict_int(value) or value < 1:
                raise ValueError(f"{name} must be an int of at least 1, got {value!r}")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    refusals: tuple[Failure, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.refusals, tuple) or not all(isinstance(item, Failure) for item in self.refusals):
            raise ValueError("refusals must be a tuple of Failure")

    @property
    def admissible(self) -> bool:
        return not self.refusals


def _refusal(error_class: str, code: str, detail: str) -> Failure:
    return Failure(error_class=error_class, code=code, retryable=False, detail=make_detail(detail))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _source_refusals(
    facts: SourceFacts, size_bytes: int, profile: MediaProfile, limits: RequestLimits
) -> list[Failure]:
    caps = profile.source
    refusals: list[Failure] = []

    size_limit = min(caps.max_source_bytes, limits.max_source_bytes)
    if size_bytes > size_limit:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "source_too_large",
                f"source size {size_bytes} bytes exceeds the limit of {size_limit} bytes",
            )
        )

    demuxer_names = {name.strip() for name in facts.demuxer.split(DEMUXER_NAME_SEPARATOR)}
    if demuxer_names.isdisjoint(caps.allowed_demuxers):
        refusals.append(
            _refusal(
                INPUT_INVALID,
                "unsupported_container",
                f"container {facts.demuxer} is not one of the allowed containers {', '.join(caps.allowed_demuxers)}",
            )
        )

    if facts.duration_estimated:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "duration_unknown",
                f"duration {facts.duration_ms} ms is an estimate, not a duration the container records",
            )
        )

    duration_limit = min(caps.max_duration_ms, limits.max_source_duration_ms)
    if facts.duration_ms > duration_limit:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "duration_exceeded",
                f"duration {facts.duration_ms} ms exceeds the limit of {duration_limit} ms",
            )
        )
    return refusals


def _stream_count_refusals(facts: SourceFacts, profile: MediaProfile) -> list[Failure]:
    caps = profile.source
    refusals: list[Failure] = []
    video_count = len(facts.video_streams)
    audio_count = len(facts.audio_streams)

    if video_count == 0:
        refusals.append(_refusal(INPUT_INVALID, "no_video_stream", "the source has no video stream"))
    if video_count > caps.max_video_streams:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "too_many_video_streams",
                f"{video_count} video streams exceed the limit of {caps.max_video_streams} video streams",
            )
        )
    if audio_count > caps.max_audio_streams:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "too_many_audio_streams",
                f"{audio_count} audio streams exceed the limit of {caps.max_audio_streams} audio streams",
            )
        )
    if facts.total_stream_count > caps.max_total_streams:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "too_many_streams",
                f"{facts.total_stream_count} streams exceed the limit of {caps.max_total_streams} streams",
            )
        )
    return refusals


def _hdr_reasons(video: VideoStreamFacts) -> list[str]:
    reasons: list[str] = []
    if video.color_transfer in HDR_TRANSFERS:
        reasons.append(f"color transfer {video.color_transfer}")
    if video.dolby_vision_profile is not None:
        reasons.append(f"Dolby Vision profile {video.dolby_vision_profile}")
    return reasons


def _video_stream_refusals(video: VideoStreamFacts, profile: MediaProfile) -> list[Failure]:
    """The per-stream rows for one non-attached video stream; each detail names the stream."""
    caps = profile.source
    stream = f"video stream {video.index}:"
    refusals: list[Failure] = []

    if video.codec_name not in caps.allowed_video_codecs:
        refusals.append(
            _refusal(
                INPUT_INVALID,
                "unsupported_video_codec",
                f"{stream} video codec {video.codec_name} is not one of the allowed codecs "
                f"{', '.join(caps.allowed_video_codecs)}",
            )
        )

    if video.display_width > caps.max_display_width or video.display_height > caps.max_display_height:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "resolution_exceeded",
                f"{stream} display size {video.display_width}x{video.display_height} exceeds the limit of "
                f"{caps.max_display_width}x{caps.max_display_height}",
            )
        )

    display_pixels = video.display_width * video.display_height
    if display_pixels > caps.max_display_pixels:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "display_pixels_exceeded",
                f"{stream} display area {display_pixels} pixels exceeds the limit of {caps.max_display_pixels} pixels",
            )
        )

    # The decoder works on the stored frame, before the aspect ratio and the rotation are applied.
    max_frame_edge = max(caps.max_display_width, caps.max_display_height)
    stored_pixels = video.width * video.height
    if max(video.width, video.height) > max_frame_edge or stored_pixels > caps.max_display_pixels:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "frame_size_exceeded",
                f"{stream} stored frame {video.width}x{video.height} ({stored_pixels} pixels) exceeds the limit "
                f"of {max_frame_edge} pixels per side and {caps.max_display_pixels} pixels",
            )
        )

    frame_rate_milli = _ceil_div(video.frame_rate_num * MILLI_PER_UNIT, video.frame_rate_den)
    if frame_rate_milli > caps.max_frame_rate_milli:
        refusals.append(
            _refusal(
                INPUT_LIMITS_EXCEEDED,
                "frame_rate_exceeded",
                f"{stream} frame rate {frame_rate_milli} frames per 1000 s exceeds the limit of "
                f"{caps.max_frame_rate_milli} frames per 1000 s",
            )
        )

    if caps.reject_interlaced and video.field_order in INTERLACED_FIELD_ORDERS:
        refusals.append(
            _refusal(
                INPUT_INVALID,
                "interlaced_unsupported",
                f"{stream} field order {video.field_order} is interlaced; the profile accepts progressive video only",
            )
        )

    hdr_reasons = _hdr_reasons(video)
    if caps.reject_hdr and hdr_reasons:
        refusals.append(
            _refusal(
                INPUT_INVALID,
                "hdr_unsupported",
                f"{stream} HDR signalled by {' and '.join(hdr_reasons)}; the profile accepts SDR video only",
            )
        )
    return refusals


def decide_admission(
    facts: SourceFacts, *, size_bytes: int, profile: MediaProfile, limits: RequestLimits
) -> AdmissionDecision:
    """Every refusal that applies, in a fixed order; the effective limit is min(profile, request).

    The source rows come first, then the stream counts, then the per-stream rows for every
    non-attached video stream (by stream, then by row).

    Duration, frame rate and geometry are what the container and the probe window declare, not what
    decoding would produce. The encode stage re-enforces the admitted bounds on the decoded stream:
    it cuts the output at the admitted duration, bounds the frame rate and bounds the decoded pixels.
    """
    if not isinstance(facts, SourceFacts):
        raise TypeError("facts must be a SourceFacts")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    if not isinstance(limits, RequestLimits):
        raise TypeError("limits must be a RequestLimits")
    if not _is_strict_int(size_bytes) or size_bytes < 1:
        raise ValueError(f"size_bytes must be an int of at least 1, got {size_bytes!r}")

    refusals = _source_refusals(facts, size_bytes, profile, limits)
    refusals += _stream_count_refusals(facts, profile)
    for video in facts.video_streams:
        refusals += _video_stream_refusals(video, profile)
    return AdmissionDecision(refusals=tuple(refusals))
