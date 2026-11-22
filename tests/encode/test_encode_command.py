"""A rendition's rate control: the pilot's fixed bitrate, or a quality level's capped constant rate factor."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.encode_command import EncodeSource, rendition_argv
from core.profile import PILOT_PROFILE, QUALITY_PROFILES, MediaProfile
from core.renditions import RenditionPlan, plan_renditions
from core.source_facts import VideoStreamFacts

EXECUTABLE = "ffmpeg"
SOURCE_PATH = Path("/scratch/source/original.mp4")
OUTPUT_DIR = Path("/scratch/output/1080p")
KEY_INFO_PATH = Path("/scratch/work/key.info")
SOURCE_DURATION_MS = 12_345
# A 1080p landscape source plans every landscape rung from 1080p down.
LANDSCAPE_NAMES = ["1080p", "720p", "540p", "360p", "240p"]
RATE_CONTROL_OPTIONS = ("-b:v", "-crf", "-maxrate", "-bufsize", "-preset")


def _video() -> VideoStreamFacts:
    return VideoStreamFacts(
        index=0,
        codec_name="h264",
        width=1920,
        height=1080,
        sample_aspect_num=1,
        sample_aspect_den=1,
        rotation_degrees=0,
        display_width=1920,
        display_height=1080,
        frame_rate_num=30,
        frame_rate_den=1,
        field_order="progressive",
        color_transfer=None,
        pix_fmt="yuv420p",
        dolby_vision_profile=None,
    )


def _source() -> EncodeSource:
    return EncodeSource(
        path=SOURCE_PATH,
        demuxer="mov",
        video_index=0,
        audio_index=1,
        rotation_degrees=0,
        duration_ms=SOURCE_DURATION_MS,
    )


def _plan(name: str, profile: MediaProfile) -> RenditionPlan:
    plans = plan_renditions(_video(), has_audio=True, requested=LANDSCAPE_NAMES, profile=profile)
    return next(plan for plan in plans if plan.name == name)


def _argv(name: str, profile: MediaProfile, *, key_info_path: Path | None = None) -> list[str]:
    return rendition_argv(
        EXECUTABLE, _source(), _plan(name, profile), profile, output_dir=OUTPUT_DIR, key_info_path=key_info_path
    )


def _value_of(argv: list[str], option: str) -> str:
    """The value that follows `option`, which must occur exactly once."""
    assert argv.count(option) == 1, f"{option} occurs {argv.count(option)} times"
    return argv[argv.index(option) + 1]


def _without_rate_control(argv: list[str]) -> list[str]:
    """The argv with every rate control option and its value taken out."""
    kept: list[str] = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
        elif argument in RATE_CONTROL_OPTIONS:
            skip_value = True
        else:
            kept.append(argument)
    return kept


@pytest.mark.parametrize("name", LANDSCAPE_NAMES)
def test_rendition_argv_when_the_profile_is_the_pilot_should_encode_at_the_rungs_fixed_bitrate(name: str) -> None:
    # Arrange
    rung = PILOT_PROFILE.encode.rung(name)
    assert rung is not None

    # Act
    argv = _argv(name, PILOT_PROFILE)

    # Assert
    assert _value_of(argv, "-b:v") == str(rung.video_bitrate)
    assert _value_of(argv, "-maxrate") == str(rung.video_maxrate)
    assert _value_of(argv, "-bufsize") == str(rung.video_bufsize)
    assert _value_of(argv, "-preset") == "medium"
    assert "-crf" not in argv


@pytest.mark.parametrize("level", QUALITY_PROFILES, ids=lambda level: level.profile_id)
@pytest.mark.parametrize("name", LANDSCAPE_NAMES)
def test_rendition_argv_when_the_profile_is_a_quality_level_should_encode_at_its_crf_capped_at_the_rungs_bitrate(
    level: MediaProfile, name: str
) -> None:
    # Arrange
    rung = level.encode.rung(name)
    assert rung is not None

    # Act
    argv = _argv(name, level)

    # Assert
    assert _value_of(argv, "-crf") == str(level.encode.crf)
    assert _value_of(argv, "-maxrate") == str(rung.video_bitrate)
    assert _value_of(argv, "-bufsize") == str(rung.video_bufsize)
    assert _value_of(argv, "-preset") == "fast"
    assert "-b:v" not in argv


@pytest.mark.parametrize("key_info_path", [None, KEY_INFO_PATH], ids=["plain", "encrypted"])
@pytest.mark.parametrize("level", QUALITY_PROFILES, ids=lambda level: level.profile_id)
def test_rendition_argv_when_the_profile_is_a_quality_level_should_differ_from_the_pilots_only_in_rate_control(
    level: MediaProfile, key_info_path: Path | None
) -> None:
    # Arrange
    name = LANDSCAPE_NAMES[0]

    # Act
    pilot_argv = _argv(name, PILOT_PROFILE, key_info_path=key_info_path)
    level_argv = _argv(name, level, key_info_path=key_info_path)

    # Assert: keyframes, segments, audio, filters, isolation and encryption are the pilot's
    assert _without_rate_control(level_argv) == _without_rate_control(pilot_argv)
