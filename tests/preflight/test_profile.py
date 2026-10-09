from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from core.profile import PILOT_PROFILE, MediaProfile, ProbeBudget, SourceCaps

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
SIXTY_FPS_MILLI = 60_000
UHD_PIXELS = 3840 * 2160


def _source(**overrides: Any) -> SourceCaps:
    return dataclasses.replace(PILOT_PROFILE.source, **overrides)


def _probe(**overrides: Any) -> ProbeBudget:
    return dataclasses.replace(PILOT_PROFILE.probe, **overrides)


def test_pilot_profile_when_rebuilt_from_its_own_values_should_be_valid() -> None:
    # Arrange / Act
    rebuilt = MediaProfile(
        profile_id=PILOT_PROFILE.profile_id,
        version=PILOT_PROFILE.version,
        source=_source(),
        probe=_probe(),
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
