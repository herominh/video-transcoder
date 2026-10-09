from __future__ import annotations

import json
import math
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from core.source_facts import ProbeInconclusive, SourceFacts, parse_ffprobe_output
from tests.preflight.conftest import MediaFixtures

FFPROBE_TIMEOUT_S = 30
FFMPEG_TIMEOUT_S = 60
MS_PER_SECOND = 1000
UNIT = 65536  # 1.0 in the display matrix's 16.16 fixed point
PERSPECTIVE_UNIT = 1 << 30  # the matrix's bottom-right entry, as ffprobe prints it
GUARDED_PARSE_LIMIT_S = 1.0
# ffprobe 8.0.1's text for -display_rotation 90, verbatim.
FFPROBE_ROTATION_90_TEXT = (
    "\n00000000:            0      -65536           0"
    "\n00000001:        65536           0           0"
    "\n00000002:            0           0  1073741824\n"
)


def _video(**overrides: Any) -> dict[str, Any]:
    stream: dict[str, Any] = {
        "index": 0,
        "codec_name": "h264",
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "sample_aspect_ratio": "1:1",
        "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001",
        "field_order": "progressive",
        "color_transfer": "bt709",
        "pix_fmt": "yuv420p",
        "disposition": {"attached_pic": 0},
    }
    stream.update(overrides)
    return stream


def _audio(**overrides: Any) -> dict[str, Any]:
    stream: dict[str, Any] = {
        "index": 1,
        "codec_name": "aac",
        "codec_type": "audio",
        "channels": 2,
        "disposition": {"attached_pic": 0},
    }
    stream.update(overrides)
    return stream


def _document(*, format_overrides: dict[str, Any] | None = None, streams: list[Any] | None = None) -> bytes:
    format_section: dict[str, Any] = {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.000000"}
    format_section.update(format_overrides or {})
    stream_list = streams if streams is not None else [_video(), _audio()]
    return json.dumps({"format": format_section, "streams": stream_list}).encode()


def _parse(document: bytes, *, duration_estimated: bool = False) -> SourceFacts:
    return parse_ffprobe_output(document, duration_estimated=duration_estimated)


def _inconclusive_code(document: bytes) -> str:
    with pytest.raises(ProbeInconclusive) as raised:
        _parse(document)
    return raised.value.code


def _matrix_text(a: int, b: int, c: int, d: int) -> str:
    """ffprobe's displaymatrix text for a matrix whose 2x2 part is (a, b / c, d)."""
    rows = ((a, b, 0), (c, d, 0), (0, 0, PERSPECTIVE_UNIT))
    return "\n" + "".join(
        f"{number:08x}: " + "".join(f"{value:12d}" for value in row) + "\n" for number, row in enumerate(rows)
    )


def _display_matrix(text: object, rotation: object = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"side_data_type": "Display Matrix"}
    if text is not None:
        entry["displaymatrix"] = text
    if rotation is not None:
        entry["rotation"] = rotation
    return {"side_data_list": [entry]}


def _dolby_vision(**fields: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "side_data_type": "DOVI configuration record",
        "dv_version_major": 1,
        "dv_version_minor": 0,
        "dv_level": 6,
        "rpu_present_flag": 1,
        "el_present_flag": 0,
        "bl_present_flag": 1,
        "dv_bl_signal_compatibility_id": 1,
    }
    entry.update(fields)
    return {"side_data_list": [entry]}


def _timed_inconclusive_code(document: bytes) -> tuple[str, float]:
    started = time.monotonic()
    code = _inconclusive_code(document)
    return code, time.monotonic() - started


def _real_ffprobe(path: Path) -> bytes:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        check=True,
        timeout=FFPROBE_TIMEOUT_S,
    )
    return completed.stdout


# --- the document as a whole ---


def test_parse_when_the_document_is_ordinary_should_report_every_fact() -> None:
    # Arrange
    document = _document()

    # Act
    facts = _parse(document)

    # Assert
    assert facts.demuxer == "mov,mp4,m4a,3gp,3g2,mj2"
    assert facts.duration_ms == 10_000
    assert facts.duration_estimated is False
    assert facts.total_stream_count == 2
    assert facts.attached_picture_count == 0
    video = facts.video_streams[0]
    assert (video.index, video.codec_name, video.width, video.height) == (0, "h264", 1920, 1080)
    assert (video.display_width, video.display_height, video.rotation_degrees) == (1920, 1080, 0)
    assert (video.frame_rate_num, video.frame_rate_den) == (30000, 1001)
    assert (video.field_order, video.color_transfer, video.pix_fmt) == ("progressive", "bt709", "yuv420p")
    audio = facts.audio_streams[0]
    assert (audio.index, audio.codec_name, audio.channels) == (1, "aac", 2)


def test_parse_when_the_duration_is_flagged_as_estimated_should_carry_the_flag() -> None:
    # Arrange
    document = _document()

    # Act
    facts = _parse(document, duration_estimated=True)

    # Assert
    assert facts.duration_estimated is True


@pytest.mark.parametrize(
    "stdout",
    [
        b"\xff\xfe not utf-8",
        b"{not json",
        b"",
        b"[]",
        b'"just a string"',
        b'{"streams": []}',
        b'{"format": {"format_name": "mov", "duration": "1.0"}}',
        b'{"format": [], "streams": []}',
        b'{"format": {"format_name": "mov", "duration": "1.0"}, "streams": {}}',
        b'{"format": {"format_name": "mov", "duration": "1.0"}, "streams": [1]}',
    ],
)
def test_parse_when_the_output_is_not_a_probe_document_should_be_inconclusive(stdout: bytes) -> None:
    # Arrange / Act
    code = _inconclusive_code(stdout)

    # Assert
    assert code == "probe_output_invalid"


@pytest.mark.parametrize("format_name", [None, "", 7])
def test_parse_when_the_container_name_is_missing_should_be_inconclusive(format_name: Any) -> None:
    # Arrange
    document = _document(format_overrides={"format_name": format_name})

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "probe_output_invalid"


def test_parse_when_inconclusive_should_never_echo_raw_probe_text_in_its_detail() -> None:
    # Arrange
    document = b"\x00https://evil.example/\xff" * 50

    # Act
    with pytest.raises(ProbeInconclusive) as raised:
        _parse(document)

    # Assert
    assert "evil" not in raised.value.detail
    assert "://" not in raised.value.detail


# --- duration ---


@pytest.mark.parametrize(
    ("duration", "expected_ms"),
    [("0.400000", 400), ("0.0001", 1), ("12.345000", 12_345), ("1.0000001", 1_001), (7, 7_000), (2.5, 2_500)],
)
def test_parse_when_the_container_reports_a_duration_should_round_it_up_to_whole_ms(
    duration: Any, expected_ms: int
) -> None:
    # Arrange
    document = _document(format_overrides={"duration": duration})

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == expected_ms


@pytest.mark.parametrize("container_duration", ["N/A", None, "0.000000", "-3.0", 0])
def test_parse_when_the_container_duration_is_absent_or_not_positive_should_take_the_longest_stream_duration(
    container_duration: Any,
) -> None:
    # Arrange
    document = _document(
        format_overrides={"duration": container_duration},
        streams=[_video(duration="3.200000"), _audio(duration="3.250000"), _audio(index=2, duration="N/A")],
    )

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 3_250


@pytest.mark.parametrize("container_duration", ["garbage", "nan", "inf", "", "1e19", "1e-19", "1" * 37])
def test_parse_when_the_container_duration_does_not_parse_should_be_inconclusive(container_duration: str) -> None:
    # Arrange
    document = _document(format_overrides={"duration": container_duration}, streams=[_video(duration="3.2")])

    # Act
    code, elapsed = _timed_inconclusive_code(document)

    # Assert
    assert code == "duration_unknown"
    assert elapsed < GUARDED_PARSE_LIMIT_S


def test_parse_when_the_container_omits_the_duration_key_should_take_the_longest_stream_duration() -> None:
    # Arrange
    format_section = {"format_name": "matroska,webm"}
    document = json.dumps({"format": format_section, "streams": [_video(duration="4.5")]}).encode()

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 4_500


def test_parse_when_a_stream_declares_more_than_the_container_should_take_the_larger_duration() -> None:
    # Arrange
    document = _document(
        format_overrides={"duration": "2.000000"}, streams=[_video(duration="2.000000"), _audio(duration="2.023000")]
    )

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 2_023


@pytest.mark.parametrize("stream_duration", ["-1.000000", "0", 0, -2.5, "N/A", None])
def test_parse_when_a_stream_duration_is_absent_or_not_positive_should_skip_it(stream_duration: Any) -> None:
    # Arrange
    document = _document(format_overrides={"duration": "5.000000"}, streams=[_video(duration=stream_duration)])

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 5_000


@pytest.mark.parametrize("stream_duration", ["garbage", "inf", "1e19", "1" * 37, [1]])
def test_parse_when_a_stream_duration_does_not_parse_should_be_inconclusive(stream_duration: Any) -> None:
    # Arrange
    document = _document(
        format_overrides={"duration": "5.000000"}, streams=[_video(), _audio(duration=stream_duration)]
    )

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "duration_unknown"


@pytest.mark.parametrize("tag_key", ["DURATION", "DURATION-eng", "duration", "Duration-fre"])
def test_parse_when_a_matroska_duration_tag_is_larger_than_the_container_duration_should_take_the_tag(
    tag_key: str,
) -> None:
    # Arrange
    document = _document(
        format_overrides={"format_name": "matroska,webm", "duration": "10.000000"},
        streams=[_video(tags={tag_key: "00:00:40.000000000"}), _audio(tags={"DURATION": "00:00:39.500000000"})],
    )

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 40_000


def test_parse_when_the_container_tags_declare_a_duration_should_count_it() -> None:
    # Arrange
    document = _document(format_overrides={"duration": "N/A", "tags": {"DURATION": "123:04:05.0006"}})

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == (123 * 3600 + 4 * 60 + 5) * MS_PER_SECOND + 1


def test_parse_when_a_duration_tag_is_zero_should_skip_it() -> None:
    # Arrange
    document = _document(format_overrides={"duration": "3.000000"}, streams=[_video(tags={"DURATION": "00:00:00.000"})])

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 3_000


def test_parse_when_a_tag_merely_starts_with_duration_should_ignore_it() -> None:
    # Arrange
    document = _document(
        format_overrides={"duration": "3.000000"}, streams=[_video(tags={"DURATIONAL": "garbage", "BPS": "12345"})]
    )

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 3_000


@pytest.mark.parametrize(
    "tag_value",
    ["00:61:00", "00:00:60", "1:2:3", "abc", "", "00:00:05.", "-00:00:05", "00:00", "00:00:05.0 ", 5, None],
)
def test_parse_when_a_duration_tag_is_malformed_should_be_inconclusive(tag_value: Any) -> None:
    # Arrange
    document = _document(streams=[_video(tags={"DURATION": tag_value})])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "duration_unknown"


def test_parse_when_no_duration_is_positive_anywhere_should_be_inconclusive() -> None:
    # Arrange
    document = _document(
        format_overrides={"duration": "N/A"},
        streams=[_video(duration="N/A"), _audio(duration="0.000000")],
    )

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "duration_unknown"


def test_parse_when_the_duration_is_at_the_numeric_guard_boundary_should_parse_it_quickly() -> None:
    # Arrange
    document = _document(format_overrides={"duration": "1e18"})
    started = time.monotonic()

    # Act
    facts = _parse(document)

    # Assert
    assert facts.duration_ms == 10**18 * MS_PER_SECOND
    assert time.monotonic() - started < GUARDED_PARSE_LIMIT_S


# --- stream classification ---


def test_parse_when_a_video_stream_is_an_attached_picture_should_count_it_apart_from_video() -> None:
    # Arrange
    cover = _video(index=1, codec_name="mjpeg", disposition={"attached_pic": 1})
    document = _document(streams=[_audio(index=0), cover])

    # Act
    facts = _parse(document)

    # Assert
    assert facts.video_streams == ()
    assert facts.attached_picture_count == 1
    assert facts.total_stream_count == 2


def test_parse_when_streams_of_other_types_exist_should_count_them_only_in_the_total() -> None:
    # Arrange
    document = _document(
        streams=[
            _video(),
            _audio(),
            {"index": 2, "codec_type": "subtitle", "codec_name": "mov_text"},
            {"index": 3, "codec_type": "data"},
            {"index": 4},
        ]
    )

    # Act
    facts = _parse(document)

    # Assert
    assert len(facts.video_streams) == 1
    assert len(facts.audio_streams) == 1
    assert facts.total_stream_count == 5


@pytest.mark.parametrize("channels", [None, "N/A", -2, "two"])
def test_parse_when_audio_channels_are_unknown_should_report_zero(channels: Any) -> None:
    # Arrange
    document = _document(streams=[_video(), _audio(channels=channels)])

    # Act
    facts = _parse(document)

    # Assert
    assert facts.audio_streams[0].channels == 0


# --- video geometry ---


@pytest.mark.parametrize(("width", "height"), [(0, 1080), (1920, None), ("N/A", 1080), (1920, -1), (1920.5, 1080)])
def test_parse_when_the_video_geometry_is_not_positive_should_be_inconclusive(width: Any, height: Any) -> None:
    # Arrange
    document = _document(streams=[_video(width=width, height=height)])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "video_geometry_unknown"


@pytest.mark.parametrize("sar", ["0:1", "N/A", None, "1:0", "-4:3", "garbage", "4/3"])
def test_parse_when_the_sample_aspect_ratio_is_absent_or_degenerate_should_use_square_pixels(sar: Any) -> None:
    # Arrange
    document = _document(streams=[_video(width=320, height=240, sample_aspect_ratio=sar)])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert (video.sample_aspect_num, video.sample_aspect_den) == (1, 1)
    assert (video.display_width, video.display_height) == (320, 240)


def test_parse_when_the_sample_aspect_ratio_is_four_by_three_should_widen_the_display_rounding_up() -> None:
    # Arrange
    document = _document(streams=[_video(width=320, height=240, sample_aspect_ratio="4:3")])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert (video.sample_aspect_num, video.sample_aspect_den) == (4, 3)
    assert (video.display_width, video.display_height) == (427, 240)


@pytest.mark.parametrize(
    ("rotation_fields", "expected_degrees", "expected_display"),
    [
        pytest.param(_display_matrix(FFPROBE_ROTATION_90_TEXT, rotation=90), 90, (240, 320), id="ffprobe-text-90"),
        pytest.param(_display_matrix(_matrix_text(UNIT, 0, 0, UNIT)), 0, (320, 240), id="identity"),
        pytest.param(_display_matrix(_matrix_text(0, -UNIT, UNIT, 0)), 90, (240, 320), id="quarter-turn"),
        pytest.param(_display_matrix(_matrix_text(-UNIT, 0, 0, -UNIT)), 180, (320, 240), id="half-turn"),
        pytest.param(_display_matrix(_matrix_text(0, UNIT, -UNIT, 0)), 270, (240, 320), id="three-quarter-turn"),
        pytest.param({"side_data_list": [{"side_data_type": "Mastering display metadata"}]}, 0, (320, 240), id="other"),
        pytest.param({}, 0, (320, 240), id="none"),
    ],
)
def test_parse_when_the_display_matrix_is_an_exact_quarter_turn_should_rotate_the_display_geometry(
    rotation_fields: dict[str, Any], expected_degrees: int, expected_display: tuple[int, int]
) -> None:
    # Arrange
    document = _document(streams=[_video(width=320, height=240, **rotation_fields)])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.rotation_degrees == expected_degrees
    assert (video.display_width, video.display_height) == expected_display


def test_parse_when_only_tags_rotate_is_set_should_not_rotate() -> None:
    # Arrange
    document = _document(streams=[_video(width=320, height=240, tags={"rotate": "90"})])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.rotation_degrees == 0
    assert (video.display_width, video.display_height) == (320, 240)


def test_parse_when_the_printed_rotation_disagrees_with_the_matrix_should_trust_the_matrix() -> None:
    # Arrange
    stream = _video(width=320, height=240, **_display_matrix(_matrix_text(-UNIT, 0, 0, -UNIT), rotation=90))
    document = _document(streams=[stream])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.rotation_degrees == 180


def test_parse_when_rotated_and_anamorphic_should_apply_the_aspect_ratio_before_the_rotation() -> None:
    # Arrange
    stream = _video(width=320, height=240, sample_aspect_ratio="4:3", **_display_matrix(FFPROBE_ROTATION_90_TEXT))
    document = _document(streams=[stream])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert (video.display_width, video.display_height) == (240, 427)


@pytest.mark.parametrize(
    "rotation_fields",
    [
        pytest.param(_display_matrix(_matrix_text(-686, -65532, 65532, -686), rotation=90), id="90.6-degrees"),
        pytest.param(_display_matrix(_matrix_text(-UNIT, 0, 0, UNIT), rotation=-180), id="horizontal-flip"),
        pytest.param(_display_matrix(_matrix_text(UNIT, 0, 0, -UNIT), rotation=0), id="vertical-flip"),
        pytest.param(_display_matrix(_matrix_text(2 * UNIT, 0, 0, 2 * UNIT)), id="scaled"),
        pytest.param(_display_matrix(None, rotation=90), id="no-text"),
        pytest.param(_display_matrix(42, rotation=90), id="text-not-a-string"),
        pytest.param(_display_matrix("\n00000000: 0 -65536 0\n00000001: 65536 0 0\n"), id="two-rows"),
        pytest.param(
            _display_matrix("\n00000000: 0 -65536 0 0\n00000001: 65536 0 0\n00000002: 0 0 1073741824\n"),
            id="four-values",
        ),
        pytest.param(
            _display_matrix("\n00000000: 0 -65536.0 0\n00000001: 65536 0 0\n00000002: 0 0 1073741824\n"),
            id="not-an-integer",
        ),
        pytest.param(
            _display_matrix("\nrow: 0 -65536 0\n00000001: 65536 0 0\n00000002: 0 0 1073741824\n"),
            id="offset-not-hex",
        ),
    ],
)
def test_parse_when_the_display_matrix_is_not_an_exact_quarter_turn_should_be_inconclusive(
    rotation_fields: dict[str, Any],
) -> None:
    # Arrange
    document = _document(streams=[_video(**rotation_fields)])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "unsupported_rotation"


# --- Dolby Vision ---


@pytest.mark.parametrize("profile", [0, 5, 8])
def test_parse_when_the_stream_carries_a_dolby_vision_record_should_report_its_profile(profile: int) -> None:
    # Arrange
    document = _document(streams=[_video(**_dolby_vision(dv_profile=profile))])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.dolby_vision_profile == profile


def test_parse_when_the_stream_has_no_dolby_vision_record_should_report_none() -> None:
    # Arrange
    document = _document()

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.dolby_vision_profile is None


@pytest.mark.parametrize("profile", ["8", -1, True, 8.0, None])
def test_parse_when_the_dolby_vision_profile_is_not_a_non_negative_int_should_be_inconclusive(profile: Any) -> None:
    # Arrange
    document = _document(streams=[_video(**_dolby_vision(dv_profile=profile))])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "probe_output_invalid"


def test_parse_when_the_dolby_vision_record_lacks_its_profile_should_be_inconclusive() -> None:
    # Arrange
    record = _dolby_vision()
    document = _document(streams=[_video(**record)])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "probe_output_invalid"


# --- frame rate and colour ---


def test_parse_when_the_average_frame_rate_is_zero_over_zero_should_use_the_real_base_rate() -> None:
    # Arrange
    document = _document(streams=[_video(avg_frame_rate="0/0", r_frame_rate="25/1")])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert (video.frame_rate_num, video.frame_rate_den) == (25, 1)


@pytest.mark.parametrize(("avg_rate", "real_rate"), [("0/0", "0/0"), ("N/A", None), (None, "30/0"), ("abc", "-25/1")])
def test_parse_when_no_frame_rate_is_positive_should_be_inconclusive(avg_rate: Any, real_rate: Any) -> None:
    # Arrange
    document = _document(streams=[_video(avg_frame_rate=avg_rate, r_frame_rate=real_rate)])

    # Act
    code = _inconclusive_code(document)

    # Assert
    assert code == "frame_rate_unknown"


def test_parse_when_both_frame_rates_have_a_40_digit_numerator_should_be_inconclusive_quickly() -> None:
    # Arrange
    huge_rate = "1" * 40 + "/1"
    document = _document(streams=[_video(avg_frame_rate=huge_rate, r_frame_rate=huge_rate)])

    # Act
    code, elapsed = _timed_inconclusive_code(document)

    # Assert
    assert code == "frame_rate_unknown"
    assert elapsed < GUARDED_PARSE_LIMIT_S


def test_parse_when_the_width_is_beyond_the_numeric_guard_should_be_inconclusive_quickly() -> None:
    # Arrange
    document = _document(streams=[_video(width="1e19")])

    # Act
    code, elapsed = _timed_inconclusive_code(document)

    # Assert
    assert code == "video_geometry_unknown"
    assert elapsed < GUARDED_PARSE_LIMIT_S


def test_parse_when_field_order_and_colour_are_absent_should_report_unknown_and_none() -> None:
    # Arrange
    stream = _video()
    for key in ("field_order", "color_transfer", "pix_fmt"):
        del stream[key]
    document = _document(streams=[stream])

    # Act
    video = _parse(document).video_streams[0]

    # Assert
    assert video.field_order == "unknown"
    assert video.color_transfer is None
    assert video.pix_fmt is None


# --- real ffprobe over the generated media ---


def test_parse_when_real_ffprobe_reads_rotated_90_should_report_a_portrait_display(media: MediaFixtures) -> None:
    # Arrange
    stdout = _real_ffprobe(media.rotated_90)

    # Act
    video = _parse(stdout).video_streams[0]

    # Assert
    assert (video.width, video.height) == (320, 240)
    assert video.rotation_degrees == 90
    assert (video.display_width, video.display_height) == (240, 320)


@pytest.mark.parametrize(
    "display_options",
    [
        pytest.param(("-display_rotation", "90.6"), id="90.6-degrees"),
        pytest.param(("-display_hflip",), id="horizontal-flip"),
    ],
)
def test_parse_when_real_ffprobe_reads_a_non_quarter_turn_matrix_should_be_inconclusive(
    media: MediaFixtures, tmp_path: Path, display_options: tuple[str, ...]
) -> None:
    # Arrange
    target = tmp_path / "oriented.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            *display_options, "-i", str(media.silent), "-c", "copy", str(target),
        ],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    )
    stdout = _real_ffprobe(target)

    # Act
    code = _inconclusive_code(stdout)

    # Assert
    assert code == "unsupported_rotation"


def test_parse_when_real_ffprobe_reads_anamorphic_should_report_the_widened_display(media: MediaFixtures) -> None:
    # Arrange
    stdout = _real_ffprobe(media.anamorphic)

    # Act
    video = _parse(stdout).video_streams[0]

    # Assert
    assert (video.display_width, video.display_height) == (427, 240)


def test_parse_when_real_ffprobe_reads_subsecond_should_keep_the_exact_fractional_duration(
    media: MediaFixtures,
) -> None:
    # Arrange
    stdout = _real_ffprobe(media.subsecond)
    probe = json.loads(stdout)
    printed = [probe["format"]["duration"], *(stream["duration"] for stream in probe["streams"])]
    longest_printed = max(Decimal(seconds) for seconds in printed)

    # Act
    facts = _parse(stdout)

    # Assert
    assert facts.duration_ms == math.ceil(longest_printed * MS_PER_SECOND)
    if longest_printed == Decimal("0.4"):
        assert facts.duration_ms == 400


def test_parse_when_real_ffprobe_reads_audio_with_cover_should_find_no_video_and_one_picture(
    media: MediaFixtures,
) -> None:
    # Arrange
    stdout = _real_ffprobe(media.audio_with_cover)

    # Act
    facts = _parse(stdout)

    # Assert
    assert facts.video_streams == ()
    assert facts.attached_picture_count == 1
    assert len(facts.audio_streams) == 1
