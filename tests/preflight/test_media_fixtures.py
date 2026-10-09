"""Smoke tests: each generated media fixture really has the property it stands for."""

from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.preflight.conftest import MALFORMED_SIZE_BYTES, MediaFixtures

FFPROBE_TIMEOUT_S = 30
BOX_HEADER_BYTES = 8
EXTENDED_SIZE_BYTES = 8
EXTENDED_SIZE_MARKER = 1
TO_END_OF_FILE_MARKER = 0


def _ffprobe(path: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        timeout=FFPROBE_TIMEOUT_S,
    )


def _probe(path: Path) -> dict[str, Any]:
    completed = _ffprobe(path)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return json.loads(completed.stdout)


def _streams_of(probe: dict[str, Any], codec_type: str) -> list[dict[str, Any]]:
    return [stream for stream in probe["streams"] if stream.get("codec_type") == codec_type]


def _codec_names(probe: dict[str, Any]) -> list[str]:
    return [stream["codec_name"] for stream in probe["streams"]]


def _top_level_boxes(path: Path) -> list[str]:
    """The ISO BMFF top-level box types in file order."""
    data = path.read_bytes()
    boxes: list[str] = []
    offset = 0
    while offset + BOX_HEADER_BYTES <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        boxes.append(data[offset + 4 : offset + BOX_HEADER_BYTES].decode("latin-1"))
        if size == EXTENDED_SIZE_MARKER:
            extended_end = offset + BOX_HEADER_BYTES + EXTENDED_SIZE_BYTES
            size = int.from_bytes(data[offset + BOX_HEADER_BYTES : extended_end], "big")
        if size == TO_END_OF_FILE_MARKER:
            break
        offset += size
    return boxes


def test_mp4_faststart_when_probed_should_hold_h264_and_aac_with_moov_before_mdat(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.mp4_faststart)
    boxes = _top_level_boxes(media.mp4_faststart)

    # Assert
    assert _codec_names(probe) == ["h264", "aac"]
    assert boxes.index("moov") < boxes.index("mdat")


def test_mp4_moov_at_end_when_probed_should_hold_h264_and_aac_with_moov_after_mdat(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.mp4_moov_at_end)
    boxes = _top_level_boxes(media.mp4_moov_at_end)

    # Assert
    assert _codec_names(probe) == ["h264", "aac"]
    assert boxes.index("mdat") < boxes.index("moov")


def test_subsecond_when_probed_should_last_less_than_one_second(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.subsecond)

    # Assert
    assert Decimal(0) < Decimal(probe["format"]["duration"]) < Decimal(1)
    assert _streams_of(probe, "audio") == []


def test_silent_when_probed_should_have_video_and_no_audio(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.silent)

    # Assert
    assert len(_streams_of(probe, "video")) == 1
    assert _streams_of(probe, "audio") == []


def test_portrait_when_probed_should_be_taller_than_wide(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.portrait), "video")[0]

    # Assert
    assert (video["width"], video["height"]) == (240, 320)


def test_rotated_90_when_probed_should_carry_a_quarter_turn_display_matrix(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.rotated_90), "video")[0]
    rotations = [
        entry["rotation"]
        for entry in video.get("side_data_list", [])
        if entry.get("side_data_type") == "Display Matrix"
    ]

    # Assert
    assert (video["width"], video["height"]) == (320, 240)
    assert len(rotations) == 1
    assert abs(rotations[0]) == 90


def test_anamorphic_when_probed_should_have_a_four_by_three_sample_aspect_ratio(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.anamorphic), "video")[0]

    # Assert
    assert video["sample_aspect_ratio"] == "4:3"


def test_interlaced_when_probed_should_report_top_field_first(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.interlaced), "video")[0]

    # Assert
    assert video["field_order"] == "tt"


def test_hdr_pq_when_probed_should_report_the_pq_transfer(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.hdr_pq), "video")[0]

    # Assert
    assert video["color_transfer"] == "smpte2084"


def test_high_fps_when_probed_should_run_at_120_frames_per_second(media: MediaFixtures) -> None:
    # Arrange / Act
    video = _streams_of(_probe(media.high_fps), "video")[0]

    # Assert
    assert video["avg_frame_rate"] == "120/1"


def test_many_audio_when_probed_should_hold_five_audio_streams(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.many_audio)

    # Assert
    assert len(_streams_of(probe, "video")) == 1
    assert len(_streams_of(probe, "audio")) == 5


def test_mkv_when_probed_should_be_matroska_with_h264_and_aac(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.mkv)

    # Assert
    assert "matroska" in probe["format"]["format_name"].split(",")
    assert _codec_names(probe) == ["h264", "aac"]


def test_webm_when_probed_should_be_webm_with_vp9_and_opus(media: MediaFixtures) -> None:
    # Arrange
    if media.webm is None:
        pytest.skip("the local ffmpeg lacks libvpx-vp9 or libopus")

    # Act
    probe = _probe(media.webm)

    # Assert
    assert "webm" in probe["format"]["format_name"].split(",")
    assert _codec_names(probe) == ["vp9", "opus"]


def test_mpegts_when_probed_should_be_mpegts_with_h264_and_aac(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.mpegts)

    # Assert
    assert probe["format"]["format_name"] == "mpegts"
    assert _codec_names(probe) == ["h264", "aac"]


def test_audio_with_cover_when_probed_should_hold_audio_and_only_an_attached_picture(media: MediaFixtures) -> None:
    # Arrange / Act
    probe = _probe(media.audio_with_cover)
    videos = _streams_of(probe, "video")

    # Assert
    assert len(_streams_of(probe, "audio")) == 1
    assert len(videos) == 1
    assert videos[0]["disposition"]["attached_pic"] == 1


def test_malformed_when_probed_should_be_refused_by_ffprobe(media: MediaFixtures) -> None:
    # Arrange / Act
    completed = _ffprobe(media.malformed)

    # Assert
    assert media.malformed.stat().st_size == MALFORMED_SIZE_BYTES
    assert completed.returncode != 0


def test_truncated_mp4_when_probed_should_be_refused_for_its_missing_index(media: MediaFixtures) -> None:
    # Arrange
    original = media.mp4_moov_at_end.read_bytes()
    truncated = media.truncated_mp4.read_bytes()

    # Act
    completed = _ffprobe(media.truncated_mp4)

    # Assert
    assert original.startswith(truncated)
    assert len(truncated) == len(original) // 2
    assert "moov" not in _top_level_boxes(media.truncated_mp4)
    assert completed.returncode != 0
