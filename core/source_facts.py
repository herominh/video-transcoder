"""Parse ffprobe's JSON output into the facts admission decides on."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from core.failure import CODE_PATTERN, make_detail

MS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
NOT_AVAILABLE = "N/A"
# Matroska declares durations as tags: "DURATION" or "DURATION-<language>", e.g. "00:00:40.000000000".
DURATION_TAG = "DURATION"
DURATION_TAG_PREFIX = "DURATION-"
DURATION_TAG_PATTERN = re.compile(r"([0-9]+):([0-5][0-9]):([0-5][0-9](?:\.[0-9]+)?)")
SUPPORTED_ROTATIONS = frozenset({0, 90, 180, 270})
SIDEWAYS_ROTATIONS = frozenset({90, 270})
DISPLAY_MATRIX_SIDE_DATA = "Display Matrix"
DISPLAY_MATRIX_TEXT_FIELD = "displaymatrix"
DISPLAY_MATRIX_ROW_COUNT = 3
DISPLAY_MATRIX_ROW_PATTERN = re.compile(
    r"[ \t]*[0-9A-Fa-f]+:[ \t]*(-?[0-9]{1,10})[ \t]+(-?[0-9]{1,10})[ \t]+(-?[0-9]{1,10})[ \t]*"
)
DISPLAY_MATRIX_UNIT = 65536  # 1.0 in the matrix's 16.16 fixed point
# (m[0], m[1], m[3], m[4]) of the exact quarter turns, as ffprobe prints them for -display_rotation.
QUARTER_TURNS: dict[tuple[int, int, int, int], int] = {
    (DISPLAY_MATRIX_UNIT, 0, 0, DISPLAY_MATRIX_UNIT): 0,
    (0, -DISPLAY_MATRIX_UNIT, DISPLAY_MATRIX_UNIT, 0): 90,
    (-DISPLAY_MATRIX_UNIT, 0, 0, -DISPLAY_MATRIX_UNIT): 180,
    (0, DISPLAY_MATRIX_UNIT, -DISPLAY_MATRIX_UNIT, 0): 270,
}
DOLBY_VISION_SIDE_DATA = "DOVI configuration record"
DOLBY_VISION_PROFILE_FIELD = "dv_profile"
VIDEO_CODEC_TYPE = "video"
AUDIO_CODEC_TYPE = "audio"
ATTACHED_PICTURE_FLAG = 1
UNKNOWN_FIELD_ORDER = "unknown"
UNKNOWN_CODEC = "unknown"
FRAME_RATE_SEPARATOR = "/"
ASPECT_RATIO_SEPARATOR = ":"
# ffprobe prints plain decimals of modest size. A number with more digits or a larger or smaller
# magnitude than these is no real duration, dimension or rate, and refusing it keeps the
# exact integer arithmetic below (and every message built from its result) small.
MAX_NUMBER_DIGITS = 36
MAX_NUMBER_MAGNITUDE_EXPONENT = 18


class ProbeInconclusive(Exception):
    """ffprobe's output does not establish a fact admission needs."""

    def __init__(self, code: str, detail: str) -> None:
        if not isinstance(code, str) or CODE_PATTERN.fullmatch(code) is None:
            raise ValueError(f"invalid diagnostic code: {code!r}")
        safe_detail = make_detail(detail)
        super().__init__(f"{code}: {safe_detail}")
        self.code = code
        self.detail = safe_detail


def _require_int(name: str, value: object, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an int of at least {minimum}, got {value!r}")


def _require_optional_str(name: str, value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")


@dataclass(frozen=True, slots=True)
class VideoStreamFacts:
    """One video stream as ffprobe reports it.

    rotation_degrees is the container's orientation, read only from its exact display matrix. The
    encode stage runs with autorotation off and applies exactly this rotation, so an orientation
    carried only in the coded stream (an H.264/HEVC display-orientation SEI) is ignored by contract.
    """

    index: int
    codec_name: str
    width: int  # ffprobe "width" (frame width, before SAR)
    height: int
    sample_aspect_num: int  # 1/1 when ffprobe gives none, "0:1", or "N/A"
    sample_aspect_den: int
    rotation_degrees: int  # 0, 90, 180 or 270
    display_width: int  # after SAR, then rotation
    display_height: int
    frame_rate_num: int
    frame_rate_den: int
    field_order: str  # ffprobe field_order, "unknown" when absent
    color_transfer: str | None
    pix_fmt: str | None
    dolby_vision_profile: int | None  # from a "DOVI configuration record"; None when the stream has none

    def __post_init__(self) -> None:
        _require_int("index", self.index, 0)
        for name in (
            "width",
            "height",
            "sample_aspect_num",
            "sample_aspect_den",
            "display_width",
            "display_height",
            "frame_rate_num",
            "frame_rate_den",
        ):
            _require_int(name, getattr(self, name), 1)
        _require_int("rotation_degrees", self.rotation_degrees, 0)
        if self.rotation_degrees not in SUPPORTED_ROTATIONS:
            raise ValueError(f"rotation_degrees must be one of 0, 90, 180, 270, got {self.rotation_degrees!r}")
        if not isinstance(self.codec_name, str) or not isinstance(self.field_order, str):
            raise ValueError("codec_name and field_order must be strings")
        _require_optional_str("color_transfer", self.color_transfer)
        _require_optional_str("pix_fmt", self.pix_fmt)
        if self.dolby_vision_profile is not None:
            _require_int("dolby_vision_profile", self.dolby_vision_profile, 0)


@dataclass(frozen=True, slots=True)
class AudioStreamFacts:
    index: int
    codec_name: str
    channels: int  # 0 when unknown

    def __post_init__(self) -> None:
        _require_int("index", self.index, 0)
        _require_int("channels", self.channels, 0)
        if not isinstance(self.codec_name, str):
            raise ValueError("codec_name must be a string")


@dataclass(frozen=True, slots=True)
class SourceFacts:
    """What the container and the probe window declare about a source.

    duration_ms is the largest duration declared anywhere (container, streams, duration tags).
    Rotation comes only from the container's exact display matrix; the encode stage applies exactly
    that rotation with autorotation off, so an orientation carried only in the coded stream (an
    H.264/HEVC display-orientation SEI) is ignored by contract.
    """

    demuxer: str  # ffprobe format.format_name, e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    duration_ms: int  # >= 1
    duration_estimated: bool
    video_streams: tuple[VideoStreamFacts, ...]  # attached pictures excluded
    audio_streams: tuple[AudioStreamFacts, ...]
    attached_picture_count: int
    total_stream_count: int  # every stream in ffprobe's list

    def __post_init__(self) -> None:
        if not isinstance(self.demuxer, str) or not self.demuxer:
            raise ValueError("demuxer must be a non-empty string")
        _require_int("duration_ms", self.duration_ms, 1)
        if not isinstance(self.duration_estimated, bool):
            raise ValueError("duration_estimated must be a bool")
        if not isinstance(self.video_streams, tuple) or not all(
            isinstance(stream, VideoStreamFacts) for stream in self.video_streams
        ):
            raise ValueError("video_streams must be a tuple of VideoStreamFacts")
        if not isinstance(self.audio_streams, tuple) or not all(
            isinstance(stream, AudioStreamFacts) for stream in self.audio_streams
        ):
            raise ValueError("audio_streams must be a tuple of AudioStreamFacts")
        _require_int("attached_picture_count", self.attached_picture_count, 0)
        listed = len(self.video_streams) + len(self.audio_streams) + self.attached_picture_count
        _require_int("total_stream_count", self.total_stream_count, listed)


def _decimal(value: object) -> Decimal | None:
    """A finite number of plausible size from an ffprobe field (number or numeric string), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        # repr gives the shortest text that round-trips, so 0.4 stays 0.4 rather than its binary value.
        number = Decimal(repr(value))
    elif isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            return None
    else:
        return None
    if not number.is_finite():
        return None
    if number.is_zero():
        return number
    if len(number.as_tuple().digits) > MAX_NUMBER_DIGITS:
        return None
    if abs(number.adjusted()) > MAX_NUMBER_MAGNITUDE_EXPONENT:
        return None
    return number


def _int(value: object) -> int | None:
    number = _decimal(value)
    if number is None or number != number.to_integral_value():
        return None
    return int(number)


def _positive_int(value: object) -> int | None:
    number = _int(value)
    return number if number is not None and number > 0 else None


def _positive_rational(value: object, separator: str) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    parts = value.split(separator)
    if len(parts) != 2:
        return None
    numerator = _positive_int(parts[0])
    denominator = _positive_int(parts[1])
    if numerator is None or denominator is None:
        return None
    return numerator, denominator


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _load_document(stdout: bytes) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    try:
        document = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ProbeInconclusive("probe_output_invalid", "ffprobe output is not UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ProbeInconclusive("probe_output_invalid", "ffprobe output is not a JSON object")
    format_section = document.get("format")
    streams = document.get("streams")
    if not isinstance(format_section, dict) or not isinstance(streams, list):
        raise ProbeInconclusive("probe_output_invalid", "ffprobe output lacks the format object or the stream list")
    if not all(isinstance(stream, dict) for stream in streams):
        raise ProbeInconclusive("probe_output_invalid", "ffprobe stream list holds a non-object entry")
    return format_section, streams


def _declared_seconds(value: object, owner: str) -> Fraction | None:
    """A positive declared duration; None when absent, "N/A" or not positive."""
    if value is None or value == NOT_AVAILABLE:
        return None
    seconds = _decimal(value)
    if seconds is None:
        raise ProbeInconclusive("duration_unknown", f"the duration of {owner} is not a number")
    return Fraction(seconds) if seconds > 0 else None


def _tag_seconds(value: object, owner: str) -> Fraction | None:
    """A positive H+:MM:SS(.fraction) duration tag; None when it is zero."""
    match = DURATION_TAG_PATTERN.fullmatch(value) if isinstance(value, str) else None
    hours = _decimal(match.group(1)) if match is not None else None
    seconds = _decimal(match.group(3)) if match is not None else None
    if match is None or hours is None or seconds is None:
        raise ProbeInconclusive("duration_unknown", f"a duration tag of {owner} is malformed")
    total = Fraction(hours) * SECONDS_PER_HOUR + int(match.group(2)) * SECONDS_PER_MINUTE + Fraction(seconds)
    return total if total > 0 else None


def _is_duration_tag(key: object) -> bool:
    if not isinstance(key, str):
        return False
    upper = key.upper()
    return upper == DURATION_TAG or upper.startswith(DURATION_TAG_PREFIX)


def _declared_durations(section: Mapping[str, object], owner: str) -> list[Fraction | None]:
    declared = [_declared_seconds(section.get("duration"), owner)]
    for key, value in _mapping(section.get("tags")).items():
        if _is_duration_tag(key):
            declared.append(_tag_seconds(value, owner))
    return declared


def _duration_ms(format_section: Mapping[str, object], streams: Sequence[Mapping[str, object]]) -> int:
    """The largest duration declared anywhere, rounded up to whole milliseconds."""
    declared = _declared_durations(format_section, "the container")
    for position, stream in enumerate(streams):
        declared += _declared_durations(stream, f"stream {position}")
    candidates = [seconds for seconds in declared if seconds is not None]
    if not candidates:
        raise ProbeInconclusive("duration_unknown", "neither the container nor any stream declares a positive duration")
    # Exact rational arithmetic: "0.400000" s is exactly 400 ms, never 401 through binary rounding.
    return math.ceil(max(candidates) * MS_PER_SECOND)


def _side_data_entry(stream: Mapping[str, object], side_data_type: str) -> Mapping[str, object] | None:
    side_data = stream.get("side_data_list")
    if not isinstance(side_data, list):
        return None
    for entry in side_data:
        if isinstance(entry, Mapping) and entry.get("side_data_type") == side_data_type:
            return entry
    return None


def _display_matrix(text: object) -> list[int] | None:
    """The 3x3 matrix, row-major, from ffprobe's text; None when the text is missing or malformed."""
    if not isinstance(text, str):
        return None
    rows = [line for line in text.split("\n") if line.strip()]
    if len(rows) != DISPLAY_MATRIX_ROW_COUNT:
        return None
    matrix: list[int] = []
    for row in rows:
        match = DISPLAY_MATRIX_ROW_PATTERN.fullmatch(row)
        if match is None:
            return None
        matrix.extend(int(value) for value in match.groups())
    return matrix


def _rotation_degrees(stream: Mapping[str, object]) -> int:
    """The container's rotation, accepted only as an exact quarter turn of its display matrix.

    ffprobe's printed "rotation" is truncated to whole degrees and blind to flips, so only the
    matrix itself counts; tags.rotate is ignored, as ffmpeg 8 ignores it.
    """
    entry = _side_data_entry(stream, DISPLAY_MATRIX_SIDE_DATA)
    if entry is None:
        return 0
    matrix = _display_matrix(entry.get(DISPLAY_MATRIX_TEXT_FIELD))
    if matrix is None:
        raise ProbeInconclusive("unsupported_rotation", "the display matrix is missing or malformed")
    degrees = QUARTER_TURNS.get((matrix[0], matrix[1], matrix[3], matrix[4]))
    if degrees is None:
        raise ProbeInconclusive("unsupported_rotation", "the display matrix is not an exact quarter turn")
    return degrees


def _dolby_vision_profile(stream: Mapping[str, object]) -> int | None:
    entry = _side_data_entry(stream, DOLBY_VISION_SIDE_DATA)
    if entry is None:
        return None
    profile = entry.get(DOLBY_VISION_PROFILE_FIELD)
    if not isinstance(profile, int) or isinstance(profile, bool) or profile < 0:
        raise ProbeInconclusive("probe_output_invalid", "the Dolby Vision record carries no valid profile")
    return profile


def _frame_rate(stream: Mapping[str, object]) -> tuple[int, int]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = _positive_rational(stream.get(key), FRAME_RATE_SEPARATOR)
        if rate is not None:
            return rate
    raise ProbeInconclusive("frame_rate_unknown", "the video stream reports no positive frame rate")


def _stream_index(stream: Mapping[str, object], position: int) -> int:
    index = _int(stream.get("index"))
    return index if index is not None and index >= 0 else position


def _codec_name(stream: Mapping[str, object]) -> str:
    return _non_empty_str(stream.get("codec_name")) or UNKNOWN_CODEC


def _video_stream(stream: Mapping[str, object], position: int) -> VideoStreamFacts:
    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    if width is None or height is None:
        raise ProbeInconclusive("video_geometry_unknown", "the video stream reports no positive width and height")
    sar_num, sar_den = _positive_rational(stream.get("sample_aspect_ratio"), ASPECT_RATIO_SEPARATOR) or (1, 1)
    rotation = _rotation_degrees(stream)
    display_width = _ceil_div(width * sar_num, sar_den)
    display_height = height
    if rotation in SIDEWAYS_ROTATIONS:
        display_width, display_height = display_height, display_width
    frame_rate_num, frame_rate_den = _frame_rate(stream)
    field_order = _non_empty_str(stream.get("field_order"))
    return VideoStreamFacts(
        index=_stream_index(stream, position),
        codec_name=_codec_name(stream),
        width=width,
        height=height,
        sample_aspect_num=sar_num,
        sample_aspect_den=sar_den,
        rotation_degrees=rotation,
        display_width=display_width,
        display_height=display_height,
        frame_rate_num=frame_rate_num,
        frame_rate_den=frame_rate_den,
        field_order=field_order.lower() if field_order is not None else UNKNOWN_FIELD_ORDER,
        color_transfer=_non_empty_str(stream.get("color_transfer")),
        pix_fmt=_non_empty_str(stream.get("pix_fmt")),
        dolby_vision_profile=_dolby_vision_profile(stream),
    )


def _audio_stream(stream: Mapping[str, object], position: int) -> AudioStreamFacts:
    return AudioStreamFacts(
        index=_stream_index(stream, position),
        codec_name=_codec_name(stream),
        channels=_positive_int(stream.get("channels")) or 0,
    )


def _is_attached_picture(stream: Mapping[str, object]) -> bool:
    return _int(_mapping(stream.get("disposition")).get("attached_pic")) == ATTACHED_PICTURE_FLAG


def parse_ffprobe_output(stdout: bytes, *, duration_estimated: bool) -> SourceFacts:
    """Read `ffprobe -print_format json -show_format -show_streams` output defensively.

    Raises ProbeInconclusive when the output does not establish a fact admission needs.
    """
    if not isinstance(stdout, bytes):
        raise TypeError("stdout must be bytes")
    if not isinstance(duration_estimated, bool):
        raise TypeError("duration_estimated must be a bool")
    format_section, streams = _load_document(stdout)
    demuxer = _non_empty_str(format_section.get("format_name"))
    if demuxer is None:
        raise ProbeInconclusive("probe_output_invalid", "ffprobe output names no container format")
    duration_ms = _duration_ms(format_section, streams)

    video_streams: list[VideoStreamFacts] = []
    audio_streams: list[AudioStreamFacts] = []
    attached_picture_count = 0
    for position, stream in enumerate(streams):
        codec_type = stream.get("codec_type")
        if codec_type == VIDEO_CODEC_TYPE and _is_attached_picture(stream):
            attached_picture_count += 1
        elif codec_type == VIDEO_CODEC_TYPE:
            video_streams.append(_video_stream(stream, position))
        elif codec_type == AUDIO_CODEC_TYPE:
            audio_streams.append(_audio_stream(stream, position))

    return SourceFacts(
        demuxer=demuxer,
        duration_ms=duration_ms,
        duration_estimated=duration_estimated,
        video_streams=tuple(video_streams),
        audio_streams=tuple(audio_streams),
        attached_picture_count=attached_picture_count,
        total_stream_count=len(streams),
    )
