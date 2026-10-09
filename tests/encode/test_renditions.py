from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

from core.failure import WorkerFailure
from core.profile import PILOT_PROFILE, RENDITION_NAMES
from core.renditions import RenditionPlan, plan_renditions
from core.source_facts import VideoStreamFacts

LANDSCAPE_NAMES = ["1080p", "720p", "480p", "360p", "240p"]
SEGMENT_DURATION_S = PILOT_PROFILE.encode.segment_duration_s


def _video(
    width: int,
    height: int,
    *,
    sar: tuple[int, int] = (1, 1),
    rotation: int = 0,
    rate: tuple[int, int] = (30, 1),
) -> VideoStreamFacts:
    """A video stream as the preflight reports it: display size after the SAR, then the rotation."""
    display_width, display_height = math.ceil(width * sar[0] / sar[1]), height
    if rotation in (90, 270):
        display_width, display_height = display_height, display_width
    return VideoStreamFacts(
        index=0,
        codec_name="h264",
        width=width,
        height=height,
        sample_aspect_num=sar[0],
        sample_aspect_den=sar[1],
        rotation_degrees=rotation,
        display_width=display_width,
        display_height=display_height,
        frame_rate_num=rate[0],
        frame_rate_den=rate[1],
        field_order="progressive",
        color_transfer=None,
        pix_fmt="yuv420p",
        dolby_vision_profile=None,
    )


def _plans(video: VideoStreamFacts, requested: list[str], *, has_audio: bool = True) -> tuple[RenditionPlan, ...]:
    return plan_renditions(video, has_audio=has_audio, requested=requested, profile=PILOT_PROFILE)


def _geometry(plans: tuple[RenditionPlan, ...]) -> list[tuple[str, int, int]]:
    return [(plan.name, plan.width, plan.height) for plan in plans]


def test_plan_renditions_when_1080p_landscape_asks_for_every_name_should_plan_1080p_down_largest_first() -> None:
    # Arrange
    video = _video(1920, 1080)

    # Act
    plans = _plans(video, list(RENDITION_NAMES))

    # Assert
    assert _geometry(plans) == [
        ("1080p", 1920, 1080),
        ("720p", 1280, 720),
        ("480p", 854, 480),
        ("360p", 640, 360),
        ("240p", 426, 240),
    ]


def test_plan_renditions_when_the_source_is_portrait_should_name_renditions_by_their_short_edge() -> None:
    # Arrange
    video = _video(1080, 1920)

    # Act
    plans = _plans(video, ["1080p"])

    # Assert
    assert _geometry(plans) == [("1080p", 1080, 1920)]


def test_plan_renditions_when_a_landscape_frame_is_rotated_a_quarter_turn_should_plan_the_portrait_display() -> None:
    # Arrange
    rotated = _video(1920, 1080, rotation=90)

    # Act
    plans = _plans(rotated, ["1080p"])

    # Assert
    assert _geometry(plans) == [("1080p", 1080, 1920)]


def test_plan_renditions_when_the_display_is_ultra_wide_should_keep_its_aspect_ratio() -> None:
    # Arrange
    video = _video(2560, 1080)

    # Act
    plans = _plans(video, ["1080p"])

    # Assert
    assert _geometry(plans) == [("1080p", 2560, 1080)]


def test_plan_renditions_when_the_source_is_anamorphic_should_plan_its_display_geometry_with_square_pixels() -> None:
    # Arrange
    video = _video(1440, 1080, sar=(4, 3))

    # Act
    plans = _plans(video, ["1080p"])

    # Assert
    assert _geometry(plans) == [("1080p", 1920, 1080)]


def test_plan_renditions_when_every_rung_is_above_the_source_should_encode_the_smallest_at_source_size() -> None:
    # Arrange
    video = _video(320, 240)

    # Act
    plans = _plans(video, ["1080p", "720p"])

    # Assert
    assert _geometry(plans) == [("720p", 320, 240)]


def test_plan_renditions_when_the_source_is_one_pixel_should_plan_the_smallest_even_frame() -> None:
    # Arrange
    video = _video(1, 1)

    # Act
    plans = _plans(video, ["240p"])

    # Assert
    assert (plans[0].width, plans[0].height) == (2, 2)


@pytest.mark.parametrize(
    ("declared", "expected_rate", "expected_gop"),
    [
        ((120, 1), (60, 1), 360),
        ((30000, 1001), (30000, 1001), 180),
        ((25, 1), (25, 1), 150),
        ((1, 10), (1, 1), 6),
    ],
)
def test_plan_renditions_when_a_frame_rate_is_declared_should_keep_it_exactly_between_the_profile_floor_and_ceiling(
    declared: tuple[int, int], expected_rate: tuple[int, int], expected_gop: int
) -> None:
    # Arrange
    video = _video(1920, 1080, rate=declared)

    # Act
    plan = _plans(video, ["720p"])[0]

    # Assert
    assert (plan.frame_rate_num, plan.frame_rate_den) == expected_rate
    assert plan.gop_frames == expected_gop
    assert plan.gop_frames >= SEGMENT_DURATION_S * expected_rate[0] / expected_rate[1]


@pytest.mark.parametrize(
    ("size", "rate", "name", "expected_level"),
    [
        ((1920, 1080), (30, 1), "1080p", "4.0"),
        ((1920, 1080), (60, 1), "1080p", "4.2"),
        ((1280, 720), (30, 1), "720p", "3.1"),
        ((3840, 2160), (60, 1), "2160p", "5.2"),
        ((3840, 240), (30, 1), "240p", "4.0"),
    ],
)
def test_plan_renditions_when_sized_and_timed_should_declare_the_smallest_h264_level_that_holds(
    size: tuple[int, int], rate: tuple[int, int], name: str, expected_level: str
) -> None:
    # Arrange
    video = _video(*size, rate=rate)

    # Act
    plan = _plans(video, [name])[0]

    # Assert
    assert plan.name == name
    assert plan.level.name == expected_level


@pytest.mark.parametrize(("has_audio", "expected_codecs"), [(True, "avc1.640028,mp4a.40.2"), (False, "avc1.640028")])
def test_plan_renditions_when_level_4_0_should_name_high_profile_and_aac_only_with_audio(
    has_audio: bool, expected_codecs: str
) -> None:
    # Arrange
    video = _video(1920, 1080, rate=(30, 1))

    # Act
    plan = _plans(video, ["1080p"], has_audio=has_audio)[0]

    # Assert
    assert plan.codecs == expected_codecs


@pytest.mark.parametrize("requested", [["4320p"], ["720p", "1080i"]])
def test_plan_renditions_when_a_name_is_not_in_the_profile_should_fail_as_a_configuration_error(
    requested: list[str],
) -> None:
    # Arrange
    video = _video(1920, 1080)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        _plans(video, requested)

    # Assert
    assert raised.value.failure.error_class == "configuration_error"
    assert raised.value.failure.code == "rendition_not_in_profile"
    assert raised.value.failure.retryable is False


def test_plan_renditions_when_the_profile_ladder_lacks_a_contract_name_should_fail_as_a_configuration_error() -> None:
    # Arrange
    ladder = tuple(rung for rung in PILOT_PROFILE.encode.ladder if rung.name != "720p")
    profile = dataclasses.replace(PILOT_PROFILE, encode=dataclasses.replace(PILOT_PROFILE.encode, ladder=ladder))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        plan_renditions(_video(1920, 1080), has_audio=True, requested=["720p"], profile=profile)

    # Assert
    assert raised.value.failure.code == "rendition_not_in_profile"


@pytest.mark.parametrize("requested", [[], ["720p", "720p"], "720p"])
def test_plan_renditions_when_the_request_is_empty_duplicated_or_not_a_list_should_raise(requested: Any) -> None:
    # Arrange
    video = _video(1920, 1080)

    # Act / Assert
    with pytest.raises(ValueError):
        _plans(video, requested)
