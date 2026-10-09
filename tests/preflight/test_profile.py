from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from core.profile import (
    CONTRACT_MAX_SEGMENTS_PER_RENDITION,
    PILOT_PROFILE,
    EncodeProfile,
    LadderRung,
    MediaProfile,
    ProbeBudget,
    SourceCaps,
)

SOURCE_INT_FIELDS = [
    "max_source_bytes",
    "max_duration_ms",
    "max_display_width",
    "max_display_height",
    "max_display_pixels",
    "max_frame_rate_milli",
    "max_video_streams",
    "max_audio_streams",
    "max_total_streams",
]
PROBE_INT_FIELDS = [
    "max_bytes",
    "max_requests",
    "max_wall_ms",
    "chunk_bytes",
    "max_output_bytes",
    "max_stderr_bytes",
    "probesize_bytes",
    "analyze_duration_us",
]
LADDER_RUNG_INT_FIELDS = ["short_edge", "video_bitrate", "video_maxrate", "video_bufsize", "audio_bitrate"]
ENCODE_INT_FIELDS = [
    "max_frame_rate_num",
    "max_frame_rate_den",
    "min_frame_rate_num",
    "min_frame_rate_den",
    "segment_duration_s",
    "max_segments_per_rendition",
    "audio_sample_rate",
    "audio_channels",
    "max_scratch_bytes",
    "max_playlist_bytes",
    "max_stderr_bytes",
    "output_check_interval_ms",
    "encode_base_wall_ms",
    "encode_wall_ms_per_media_s",
    "thumbnail_short_edge",
    "thumbnail_quality",
    "thumbnail_max_bytes",
    "thumbnail_wall_ms",
    "thumbnail_at_per_mille",
    "thumbnail_at_max_ms",
]
SIXTY_FPS_MILLI = 60_000
UHD_PIXELS = 3840 * 2160
MS_PER_SECOND = 1000


def _source(**overrides: Any) -> SourceCaps:
    return dataclasses.replace(PILOT_PROFILE.source, **overrides)


def _probe(**overrides: Any) -> ProbeBudget:
    return dataclasses.replace(PILOT_PROFILE.probe, **overrides)


def _encode(**overrides: Any) -> EncodeProfile:
    return dataclasses.replace(PILOT_PROFILE.encode, **overrides)


def _rung(**overrides: Any) -> LadderRung:
    return dataclasses.replace(PILOT_PROFILE.encode.ladder[0], **overrides)


def _profile(*, source: SourceCaps | None = None, encode: EncodeProfile | None = None) -> MediaProfile:
    return dataclasses.replace(PILOT_PROFILE, source=source or _source(), encode=encode or _encode())


def test_pilot_profile_when_rebuilt_from_its_own_values_should_be_valid() -> None:
    # Arrange / Act
    rebuilt = MediaProfile(
        profile_id=PILOT_PROFILE.profile_id,
        version=PILOT_PROFILE.version,
        source=_source(),
        probe=_probe(),
        encode=_encode(),
    )

    # Assert
    assert rebuilt == PILOT_PROFILE
    assert (rebuilt.profile_id, rebuilt.version) == ("pilot-h264-sdr", 1)


def test_pilot_profile_when_checked_against_its_purpose_should_admit_60_fps_uhd_and_refuse_hdr_and_interlace() -> None:
    # Arrange
    source = PILOT_PROFILE.source

    # Act / Assert
    assert source.max_frame_rate_milli >= SIXTY_FPS_MILLI
    assert source.max_display_pixels >= UHD_PIXELS
    assert source.reject_hdr is True
    assert source.reject_interlaced is True
    assert PILOT_PROFILE.probe.chunk_bytes <= PILOT_PROFILE.probe.max_bytes


@pytest.mark.parametrize("field_name", SOURCE_INT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5, "10"])
def test_source_caps_when_an_int_field_is_not_a_positive_int_should_raise(field_name: str, bad_value: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _source(**{field_name: bad_value})


@pytest.mark.parametrize("field_name", PROBE_INT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5, "10"])
def test_probe_budget_when_an_int_field_is_not_a_positive_int_should_raise(field_name: str, bad_value: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _probe(**{field_name: bad_value})


def test_source_caps_when_the_pixel_cap_exceeds_width_times_height_should_raise() -> None:
    # Arrange
    width, height = 1920, 1080

    # Act / Assert
    with pytest.raises(ValueError):
        _source(max_display_width=width, max_display_height=height, max_display_pixels=width * height + 1)


def test_source_caps_when_the_pixel_cap_equals_width_times_height_should_be_accepted() -> None:
    # Arrange
    width, height = 1920, 1080

    # Act
    caps = _source(max_display_width=width, max_display_height=height, max_display_pixels=width * height)

    # Assert
    assert caps.max_display_pixels == width * height


def test_source_caps_when_total_streams_are_fewer_than_video_plus_audio_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _source(max_video_streams=2, max_audio_streams=4, max_total_streams=5)


def test_probe_budget_when_the_chunk_exceeds_the_byte_budget_should_raise() -> None:
    # Arrange
    budget = PILOT_PROFILE.probe.max_bytes

    # Act / Assert
    with pytest.raises(ValueError):
        _probe(chunk_bytes=budget + 1)


def test_probe_budget_when_the_probesize_exceeds_the_byte_budget_should_raise() -> None:
    # Arrange
    budget = PILOT_PROFILE.probe.max_bytes

    # Act / Assert
    with pytest.raises(ValueError):
        _probe(probesize_bytes=budget + 1)


@pytest.mark.parametrize("field_name", ["allowed_demuxers", "allowed_video_codecs"])
@pytest.mark.parametrize(
    "entries",
    [
        (),
        ("mov", "mov"),
        ("MOV",),
        ("mov", "H264"),
        ("matroska webm",),
        ("x" * 33,),
        ["mov"],
        (1,),
    ],
)
def test_source_caps_when_an_allow_list_is_empty_duplicated_or_malformed_should_raise(
    field_name: str, entries: Any
) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _source(**{field_name: entries})


@pytest.mark.parametrize("field_name", ["reject_interlaced", "reject_hdr"])
def test_source_caps_when_a_reject_flag_is_not_a_bool_should_raise(field_name: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _source(**{field_name: 1})


@pytest.mark.parametrize("profile_id", ["p", "Pilot", "1pilot", "pilot h264", "x" * 33, ""])
def test_media_profile_when_the_id_is_malformed_should_raise(profile_id: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        dataclasses.replace(PILOT_PROFILE, profile_id=profile_id)


@pytest.mark.parametrize("version", [0, 65536, -1, True, 1.0])
def test_media_profile_when_the_version_is_not_an_int_from_1_to_65535_should_raise(version: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        dataclasses.replace(PILOT_PROFILE, version=version)


def test_media_profile_when_source_and_probe_are_swapped_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        dataclasses.replace(PILOT_PROFILE, source=PILOT_PROFILE.probe, probe=PILOT_PROFILE.source)


def test_pilot_profile_when_checked_should_encode_h264_high_and_stereo_aac_in_6_s_segments() -> None:
    # Arrange
    encode = PILOT_PROFILE.encode

    # Act / Assert
    assert encode.segment_duration_s == 6
    assert encode.max_segments_per_rendition == CONTRACT_MAX_SEGMENTS_PER_RENDITION == 3600
    assert (encode.max_frame_rate_num, encode.max_frame_rate_den) == (60, 1)
    assert (encode.min_frame_rate_num, encode.min_frame_rate_den) == (1, 1)
    assert (encode.video_encoder, encode.h264_profile, encode.pixel_format) == ("libx264", "high", "yuv420p")
    assert (encode.audio_codec, encode.audio_channels, encode.audio_sample_rate) == ("aac", 2, 48_000)


@pytest.mark.parametrize("field_name", LADDER_RUNG_INT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -2, True, 1.5, "10"])
def test_ladder_rung_when_an_int_field_is_not_a_positive_int_should_raise(field_name: str, bad_value: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _rung(**{field_name: bad_value})


@pytest.mark.parametrize("field_name", ENCODE_INT_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5, "10"])
def test_encode_profile_when_an_int_field_is_not_a_positive_int_should_raise(field_name: str, bad_value: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(**{field_name: bad_value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("video_encoder", "h264_nvenc"),
        ("video_encoder", "libx265"),
        ("h264_profile", "main"),
        ("pixel_format", "yuv444p"),
        ("audio_codec", "opus"),
        ("encoder_preset", "placebo"),
        ("encoder_preset", "ULTRAFAST"),
        ("encoder_preset", None),
    ],
)
def test_encode_profile_when_a_format_choice_is_not_supported_should_raise(field_name: str, value: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(**{field_name: value})


def test_encode_profile_when_the_ladder_is_not_largest_first_should_raise() -> None:
    # Arrange
    ladder = tuple(reversed(PILOT_PROFILE.encode.ladder))

    # Act / Assert
    with pytest.raises(ValueError):
        _encode(ladder=ladder)


def test_encode_profile_when_two_rungs_share_a_short_edge_should_raise() -> None:
    # Arrange
    first, second = PILOT_PROFILE.encode.ladder[:2]
    ladder = (first, dataclasses.replace(second, short_edge=first.short_edge))

    # Act / Assert
    with pytest.raises(ValueError):
        _encode(ladder=ladder)


def test_encode_profile_when_two_rungs_share_a_name_should_raise() -> None:
    # Arrange
    first, second = PILOT_PROFILE.encode.ladder[:2]
    ladder = (first, dataclasses.replace(second, name=first.name))

    # Act / Assert
    with pytest.raises(ValueError):
        _encode(ladder=ladder)


@pytest.mark.parametrize("ladder", [(), [PILOT_PROFILE.encode.ladder[0]], ("1080p",)])
def test_encode_profile_when_the_ladder_is_empty_or_not_a_tuple_of_rungs_should_raise(ladder: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(ladder=ladder)


@pytest.mark.parametrize("name", ["4320p", "1080i", "", None])
def test_ladder_rung_when_the_name_is_not_a_contract_rendition_name_should_raise(name: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _rung(name=name)


def test_ladder_rung_when_the_short_edge_is_odd_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _rung(short_edge=1079)


def test_ladder_rung_when_the_bitrate_exceeds_the_maxrate_should_raise() -> None:
    # Arrange
    rung = PILOT_PROFILE.encode.ladder[0]

    # Act / Assert
    with pytest.raises(ValueError):
        _rung(video_bitrate=rung.video_maxrate + 1)


def test_ladder_rung_when_the_bitrate_equals_the_maxrate_should_be_accepted() -> None:
    # Arrange
    rung = PILOT_PROFILE.encode.ladder[0]

    # Act
    capped = _rung(video_bitrate=rung.video_maxrate)

    # Assert
    assert capped.video_bitrate == capped.video_maxrate


def test_encode_profile_when_segments_per_rendition_exceed_the_contracts_3600_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(max_segments_per_rendition=CONTRACT_MAX_SEGMENTS_PER_RENDITION + 1)


def test_media_profile_when_the_longest_source_needs_more_segments_than_allowed_should_raise() -> None:
    # Arrange
    encode = _encode(segment_duration_s=1)
    longest_encodable_ms = MS_PER_SECOND * encode.max_segments_per_rendition

    # Act / Assert
    with pytest.raises(ValueError):
        _profile(source=_source(max_duration_ms=longest_encodable_ms + 1), encode=encode)


def test_media_profile_when_the_longest_source_fills_the_segment_limit_exactly_should_be_accepted() -> None:
    # Arrange
    encode = _encode(segment_duration_s=1)
    longest_encodable_ms = MS_PER_SECOND * encode.max_segments_per_rendition

    # Act
    profile = _profile(source=_source(max_duration_ms=longest_encodable_ms), encode=encode)

    # Assert
    assert profile.source.max_duration_ms == longest_encodable_ms


def test_media_profile_when_the_output_frame_rate_ceiling_exceeds_the_source_cap_should_raise() -> None:
    # Arrange
    source = _source(max_frame_rate_milli=SIXTY_FPS_MILLI - 1)

    # Act / Assert
    with pytest.raises(ValueError):
        _profile(source=source, encode=_encode(max_frame_rate_num=60, max_frame_rate_den=1))


def test_media_profile_when_the_output_frame_rate_ceiling_equals_the_source_cap_should_be_accepted() -> None:
    # Arrange
    source = _source(max_frame_rate_milli=SIXTY_FPS_MILLI)

    # Act
    profile = _profile(source=source, encode=_encode(max_frame_rate_num=60, max_frame_rate_den=1))

    # Assert
    assert profile.encode.max_frame_rate_milli == SIXTY_FPS_MILLI


@pytest.mark.parametrize("scratch_minus_source", [0, -1])
def test_media_profile_when_the_scratch_ceiling_is_not_above_the_largest_source_should_raise(
    scratch_minus_source: int,
) -> None:
    # Arrange
    source = _source()
    scratch = source.max_source_bytes + scratch_minus_source

    # Act / Assert
    with pytest.raises(ValueError):
        _profile(source=source, encode=_encode(max_scratch_bytes=scratch))


def test_media_profile_when_the_encode_section_is_not_an_encode_profile_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        dataclasses.replace(PILOT_PROFILE, encode=PILOT_PROFILE.probe)


def test_encode_profile_when_the_frame_rate_floor_exceeds_the_ceiling_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(min_frame_rate_num=61, min_frame_rate_den=1, max_frame_rate_num=60, max_frame_rate_den=1)


def test_encode_profile_when_the_frame_rate_floor_equals_the_ceiling_should_be_accepted() -> None:
    # Arrange / Act
    encode = _encode(min_frame_rate_num=30, min_frame_rate_den=1, max_frame_rate_num=30, max_frame_rate_den=1)

    # Assert
    assert (encode.min_frame_rate_num, encode.max_frame_rate_num) == (30, 30)


def test_encode_profile_when_the_frame_rate_floor_leaves_a_segment_without_a_frame_should_raise() -> None:
    # Arrange: one frame every 7 s against 6 s segments
    segment_s = 6

    # Act / Assert
    with pytest.raises(ValueError):
        _encode(segment_duration_s=segment_s, min_frame_rate_num=1, min_frame_rate_den=segment_s + 1)


def test_encode_profile_when_the_frame_rate_floor_gives_each_segment_exactly_one_frame_should_be_accepted() -> None:
    # Arrange
    segment_s = 6

    # Act
    encode = _encode(segment_duration_s=segment_s, min_frame_rate_num=1, min_frame_rate_den=segment_s)

    # Assert
    assert encode.min_frame_rate_den == segment_s


def test_pilot_profile_when_checked_should_take_a_360_line_thumbnail_a_tenth_in_within_5_s() -> None:
    # Arrange
    encode = PILOT_PROFILE.encode

    # Act / Assert
    assert (encode.thumbnail_short_edge, encode.thumbnail_quality) == (360, 2)
    assert encode.thumbnail_max_bytes == 1024 * 1024
    assert encode.thumbnail_wall_ms == 60_000
    assert (encode.thumbnail_at_per_mille, encode.thumbnail_at_max_ms) == (100, 5_000)


def test_encode_profile_when_the_thumbnail_short_edge_is_odd_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(thumbnail_short_edge=359)


@pytest.mark.parametrize("quality", [1, 32])
def test_encode_profile_when_the_thumbnail_quality_is_outside_mjpegs_2_to_31_should_raise(quality: int) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _encode(thumbnail_quality=quality)


@pytest.mark.parametrize("quality", [2, 31])
def test_encode_profile_when_the_thumbnail_quality_is_at_either_end_of_2_to_31_should_be_accepted(
    quality: int,
) -> None:
    # Arrange / Act
    encode = _encode(thumbnail_quality=quality)

    # Assert
    assert encode.thumbnail_quality == quality


@pytest.mark.parametrize("per_mille", [1000, 1001], ids=["at the very end", "past the end"])
def test_encode_profile_when_the_thumbnail_is_taken_at_or_past_the_end_of_the_source_should_raise(
    per_mille: int,
) -> None:
    # Arrange / Act / Assert: a time at the very end can name an output frame that does not exist
    with pytest.raises(ValueError):
        _encode(thumbnail_at_per_mille=per_mille)


def test_encode_profile_when_the_thumbnail_is_taken_just_before_the_end_of_the_source_should_be_accepted() -> None:
    # Arrange / Act
    encode = _encode(thumbnail_at_per_mille=999)

    # Assert
    assert encode.thumbnail_at_per_mille == 999
