"""core/job/reservation.py: the room each planned rendition is given up front, and when a job's renditions may be
encoded in parallel. Pure, but for the free space read from a real directory."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.encode import MIN_FREE_BYTES, OutputAllowance
from core.job.reservation import encode_mode, rendition_reservation, scratch_free_bytes
from core.profile import MIB, PILOT_PROFILE, QUALITY_PROFILES
from core.renditions import RenditionPlan, plan_renditions
from tests.job.test_runner import _video

FORTY_MINUTES_MS = 40 * 60 * 1000
SIX_HOURS_MS = 6 * 60 * 60 * 1000
# The pilot's 1080p rung: a 5,350,000 bps video ceiling, a 7,500,000-bit buffer and 192,000 bps of audio; its
# playlists may take 1 MiB.
PLAYLIST_BYTES = 1 * MIB
# Whatever the duration, a 1080p rendition holds room for its whole buffer (937,500 bytes) and one second at its
# video ceiling and audio bitrate (692,750 bytes), the frame at the 1 fps floor the cut lets run past the admitted
# duration.
BUFFER_AND_FRAME_BYTES_1080P = 937_500 + 692_750
# 40 minutes of 6-second segments.
FORTY_MINUTES_SEGMENTS = 400
# The contract's most segments of one rendition.
CONTRACT_MAX_SEGMENTS = 3600
ROOMY = OutputAllowance(max_bytes=10**12, max_artifacts=100_000)


def _plan_1080p(*, has_audio: bool) -> RenditionPlan:
    video = _video(width=1920, height=1080, display_width=1920, display_height=1080)
    (plan,) = plan_renditions(video, has_audio=has_audio, requested=["1080p"], profile=PILOT_PROFILE)
    return plan


def _plan_2160p() -> RenditionPlan:
    video = _video(width=3840, height=2160, display_width=3840, display_height=2160)
    (plan,) = plan_renditions(video, has_audio=True, requested=["2160p"], profile=PILOT_PROFILE)
    return plan


def _three(each: OutputAllowance) -> tuple[OutputAllowance, ...]:
    return (each, each, each)


# --- the reservation of one rendition -------------------------------------------------------------------


def test_rendition_reservation_when_a_40_minute_source_has_audio_should_reserve_twice_its_rates_and_a_playlist():
    # Arrange
    plan = _plan_1080p(has_audio=True)

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, FORTY_MINUTES_MS)

    # Assert: (5,350,000 + 192,000) bps over 2,400 s is 1,662,600,000 bytes, twice that, then the buffer, the
    # frame past the cut and the playlist's room; 400 segments, two more, and the playlist
    assert reservation.max_bytes == 3_325_200_000 + BUFFER_AND_FRAME_BYTES_1080P + PLAYLIST_BYTES
    assert reservation.max_artifacts == FORTY_MINUTES_SEGMENTS + 2 + 1


def test_rendition_reservation_when_a_40_minute_source_has_no_audio_should_reserve_twice_the_video_alone():
    # Arrange
    plan = _plan_1080p(has_audio=False)

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, FORTY_MINUTES_MS)

    # Assert: 5,350,000 bps over 2,400 s is 1,605,000,000 bytes, twice that, then the buffer, the frame past the
    # cut without audio (one second at 5,350,000 bps) and the playlist's room
    assert reservation.max_bytes == 3_210_000_000 + 937_500 + 668_750 + PLAYLIST_BYTES
    assert reservation.max_artifacts == FORTY_MINUTES_SEGMENTS + 2 + 1


def test_rendition_reservation_when_the_profile_is_a_quality_level_should_cover_twice_its_ceiling():
    # Arrange: a quality level holds the video to the rung's video_bitrate (-maxrate), 5,000,000 bps for 1080p
    plan = _plan_1080p(has_audio=True)
    quality_level = QUALITY_PROFILES[0]

    # Act
    reservation = rendition_reservation(plan, quality_level, FORTY_MINUTES_MS)

    # Assert: at least twice (5,000,000 + 192,000) bps over 2,400 s, the 7,500,000-bit buffer, one second at
    # 5,192,000 bps and the playlist's room
    assert reservation.max_bytes >= 2 * 1_557_600_000 + 937_500 + 649_000 + PLAYLIST_BYTES


def test_rendition_reservation_when_the_source_is_six_hours_long_should_reserve_no_more_files_than_the_cap():
    # Arrange: six hours of 6-second segments would be 3,600, and two more
    plan = _plan_1080p(has_audio=True)

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, SIX_HOURS_MS)

    # Assert
    assert reservation.max_artifacts == CONTRACT_MAX_SEGMENTS + 1


def test_rendition_reservation_when_the_duration_is_zero_should_reserve_the_buffer_a_frame_a_playlist_and_three_files():
    # Arrange
    plan = _plan_1080p(has_audio=True)

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, 0)

    # Assert
    assert reservation == OutputAllowance(max_bytes=BUFFER_AND_FRAME_BYTES_1080P + PLAYLIST_BYTES, max_artifacts=3)


@pytest.mark.parametrize("duration_ms", [0, 500])
def test_rendition_reservation_when_the_source_is_shorter_than_a_second_should_hold_what_vbv_lets_the_encode_write(
    duration_ms,
):
    # Arrange: 2160p, the largest buffer: 22,500,000 bits, a 16,000,000 bps video ceiling and 192,000 bps of audio
    plan = _plan_2160p()
    bufsize_bits, maxrate_bps, audio_bps = 22_500_000, 16_000_000, 192_000
    # What runs, at most: the admitted duration and one frame of a second at the 1 fps floor past the cut
    runs_ms = duration_ms + 1000

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, duration_ms)

    # Assert: at least the buffer and a second at the ceiling; and at least what VBV lets the video write over
    # what runs (the buffer and the ceiling over that time), with the audio over the same time
    assert reservation.max_bytes >= bufsize_bits // 8 + maxrate_bps // 8 + PLAYLIST_BYTES
    vbv_bound_bytes = (bufsize_bits + (maxrate_bps + audio_bps) * runs_ms // 1000) // 8
    assert reservation.max_bytes >= vbv_bound_bytes + PLAYLIST_BYTES


def test_rendition_reservation_when_the_duration_is_one_millisecond_should_round_its_bytes_and_segments_up():
    # Arrange
    plan = _plan_1080p(has_audio=True)

    # Act
    reservation = rendition_reservation(plan, PILOT_PROFILE, 1)

    # Assert: 5,542,000 bps over 1 ms is 692.75 bytes, twice that 1,385.5, taken as 1,386; one segment, two more
    assert reservation == OutputAllowance(
        max_bytes=1_386 + BUFFER_AND_FRAME_BYTES_1080P + PLAYLIST_BYTES, max_artifacts=1 + 2 + 1
    )


def test_rendition_reservation_when_the_profiles_floor_is_half_a_frame_a_second_should_reserve_a_two_second_frame():
    # Arrange: a profile whose slowest output is one frame every two seconds
    half_fps = dataclasses.replace(
        PILOT_PROFILE, encode=dataclasses.replace(PILOT_PROFILE.encode, min_frame_rate_num=1, min_frame_rate_den=2)
    )
    plan = _plan_1080p(has_audio=True)

    # Act
    reservation = rendition_reservation(plan, half_fps, 0)

    # Assert: the buffer's 937,500 bytes, and two seconds at 5,542,000 bps, 1,385,500 bytes
    assert reservation.max_bytes == 937_500 + 1_385_500 + PLAYLIST_BYTES


@pytest.mark.parametrize("duration_ms", [-1, True, 1.5, None])
def test_rendition_reservation_when_the_duration_is_not_a_whole_count_of_milliseconds_should_refuse_it(duration_ms):
    # Arrange
    plan = _plan_1080p(has_audio=True)

    # Act / Assert
    with pytest.raises(ValueError):
        rendition_reservation(plan, PILOT_PROFILE, duration_ms)


# --- the choice between parallel and one at a time -----------------------------------------------------


def test_encode_mode_when_everything_fits_should_be_parallel():
    # Arrange
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode(_three(each), OutputAllowance(max_bytes=3_000, max_artifacts=30), 3_000 + MIN_FREE_BYTES)

    # Assert
    assert mode.parallel is True
    assert mode.reason == "parallel: 3 renditions"


def test_encode_mode_when_there_is_one_rendition_should_be_sequential():
    # Arrange
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode((each,), ROOMY, 10**12)

    # Assert
    assert mode.parallel is False


def test_encode_mode_when_the_reservations_exceed_the_allowances_bytes_should_be_sequential():
    # Arrange
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode(_three(each), OutputAllowance(max_bytes=2_999, max_artifacts=30), 10**12)

    # Assert
    assert mode.parallel is False
    assert mode.reason == "sequential: 3000 bytes reserved exceed the allowance's 2999"


def test_encode_mode_when_the_reservations_exceed_the_allowances_files_should_be_sequential():
    # Arrange
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode(_three(each), OutputAllowance(max_bytes=3_000, max_artifacts=29), 10**12)

    # Assert
    assert mode.parallel is False
    assert mode.reason == "sequential: 30 files reserved exceed the allowance's 29"


def test_encode_mode_when_the_free_disk_is_one_byte_short_should_be_sequential():
    # Arrange: the reservations together and the margin a full disk is told by, less one byte
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode(_three(each), ROOMY, 3_000 + MIN_FREE_BYTES - 1)

    # Assert
    assert mode.parallel is False
    assert mode.reason == f"sequential: {3_000 + MIN_FREE_BYTES - 1} bytes free on the scratch disk, " \
        f"{3_000 + MIN_FREE_BYTES} needed"


def test_encode_mode_when_the_free_disk_is_unknown_should_be_sequential():
    # Arrange
    each = OutputAllowance(max_bytes=1_000, max_artifacts=10)

    # Act
    mode = encode_mode(_three(each), ROOMY, None)

    # Assert
    assert mode.parallel is False


def test_scratch_free_bytes_when_statvfs_raises_should_be_unknown(tmp_path: Path):
    # Arrange: statvfs of a directory that does not exist raises FileNotFoundError
    missing = tmp_path / "missing"

    # Act
    free_bytes = scratch_free_bytes(missing)

    # Assert
    assert free_bytes is None
    assert encode_mode(_three(OutputAllowance(max_bytes=1, max_artifacts=1)), ROOMY, free_bytes).parallel is False


def test_scratch_free_bytes_when_the_directory_exists_should_tell_what_may_still_be_written(tmp_path: Path):
    # Act
    free_bytes = scratch_free_bytes(tmp_path)

    # Assert
    assert isinstance(free_bytes, int) and free_bytes >= 0

