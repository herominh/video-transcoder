from __future__ import annotations

import contextlib
import dataclasses
import errno
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import core.encode
import core.hls_key
from core.bounded_process import BoundedResult, Deadline
from core.encode import (
    DURATION_SLACK_MS,
    ArtifactFile,
    OutputAllowance,
    RenditionOutput,
    encode_rendition,
)
from core.encode_command import EncodeSource, decoder_pixel_limit
from core.failure import WorkerFailure
from core.profile import GIB, PILOT_PROFILE, MediaProfile
from core.renditions import RenditionPlan, plan_renditions
from core.source_facts import SourceFacts, VideoStreamFacts
from tests.encode.conftest import (
    LOCATION_MARKER,
    ONE_SECOND_SEGMENTS,
    QUADRANT_COLOURS,
    TITLE_MARKER,
    EncodeMedia,
    fast_profile,
    probe_facts,
)
from tests.failing_log import reported_log_failures
from tests.preflight.conftest import FFMPEG_TIMEOUT_S

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the encode runs under POSIX process groups")

MEDIA_KEY = bytes.fromhex("8f3a5c71e2d94b06a1c7e35f90b28d4e")
GENEROUS_DEADLINE_MS = 60_000
FAR_TOO_SHORT_DEADLINE_MS = 20
GENEROUS_ALLOWANCE = OutputAllowance(max_bytes=1 * GIB, max_artifacts=10_000)
FAST_PROFILE = fast_profile()
SMALLEST_RENDITION = "240p"
TS_SYNC_BYTE = 0x47
MS_PER_SECOND = 1000
FRAME_COUNT_TOLERANCE = 1
ADMITTED_DURATION_MS = 1200
LOWERED_FRAME_RATE = (10, 1)
TWO_ARTIFACTS = 2
TINY_BYTE_ALLOWANCE = 1000
RETURN_WITHIN_S = 5.0
# Mean RGB of a quadrant after lossy coding stays this close to the reference decode.
QUADRANT_TOLERANCE = 40
# The reference quadrants are this far apart in at least one channel, so a swap cannot pass.
QUADRANT_SEPARATION = 100
IV_PATTERN = re.compile(r"IV=0x([0-9a-f]{32})")
MAX_PIXEL_TEXT = "Picture size 320x240 exceeds specified max pixel count 76799"
NO_SPACE_TEXT = "Error writing trailer: No space left on device"
TS_PACKET_BYTES = 188
AES_BLOCK_BITS = 128
CRC32_MPEG2_POLYNOMIAL = 0x04C11DB7
CRC32_MASK = 0xFFFFFFFF
CRC32_TOP_BIT = 0x80000000
TS_PAYLOAD_UNIT_START = 0x40
TS_HEADER_BYTES = 4
TS_ADAPTATION_ONLY = 2
TS_ADAPTATION_AND_PAYLOAD = 3
PSI_SECTION_HEADER_BYTES = 3
PSI_SECTION_LENGTH_HIGH_MASK = 0x0F
# A profile whose decoder limit (10,000 + 63 x 200 + 63 x 63 = 26,569) is below 320x240.
SMALL_PIXEL_CAP_EDGE = 100
# The claimed source of the fake-ffmpeg tests: 2 s at 25 fps.
CLAIMED_DURATION_MS = 2000


@dataclass(frozen=True, slots=True)
class _Job:
    source: EncodeSource
    plan: RenditionPlan
    profile: MediaProfile
    output_root: Path
    work_dir: Path

    @property
    def rendition_dir(self) -> Path:
        return self.output_root / self.plan.name

    def run(
        self,
        *,
        media_key: bytes | None = None,
        allowance: OutputAllowance = GENEROUS_ALLOWANCE,
        deadline: Deadline | None = None,
        ffmpeg_path: str | None = None,
    ) -> RenditionOutput:
        return encode_rendition(
            self.source,
            self.plan,
            profile=self.profile,
            output_root=self.output_root,
            work_dir=self.work_dir,
            media_key=media_key,
            allowance=allowance,
            deadline=deadline or Deadline.after_ms(GENEROUS_DEADLINE_MS),
            ffmpeg_path=ffmpeg_path,
        )


def _directories(root: Path) -> tuple[Path, Path]:
    output_root = root / "output"
    work_dir = root / "work"
    output_root.mkdir()
    work_dir.mkdir()
    return output_root, work_dir


def _job(
    root: Path,
    media_path: Path,
    *,
    profile: MediaProfile = FAST_PROFILE,
    facts: SourceFacts | None = None,
    duration_ms: int | None = None,
) -> _Job:
    facts = facts or probe_facts(media_path)
    output_root, work_dir = _directories(root)
    source = EncodeSource.from_facts(media_path, facts, profile)
    if duration_ms is not None:
        source = dataclasses.replace(source, duration_ms=duration_ms)
    plan = plan_renditions(
        facts.video_streams[0], has_audio=bool(facts.audio_streams), requested=[SMALLEST_RENDITION], profile=profile
    )[0]
    return _Job(source=source, plan=plan, profile=profile, output_root=output_root, work_dir=work_dir)


def _run_tool(*argv: str) -> bytes:
    return subprocess.run(list(argv), check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S).stdout


def _probe(path: Path, *extra: str) -> dict[str, Any]:
    output = _run_tool(
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", *extra, str(path)
    )
    return json.loads(output)


def _segment_paths(job: _Job, output: RenditionOutput) -> list[Path]:
    return [job.output_root / artifact.path for artifact in output.artifacts if artifact.kind == "hls_segment"]


def _decoded_frame_count(segments: list[Path]) -> int:
    total = 0
    for segment in segments:
        stream = _probe(segment, "-count_frames", "-select_streams", "v:0")["streams"][0]
        total += int(stream["nb_read_frames"])
    return total


def _all_files(*roots: Path) -> list[Path]:
    return [path for root in roots for path in root.rglob("*") if path.is_file()]


def _assert_failure(raised: pytest.ExceptionInfo[WorkerFailure], expected: tuple[str, str, bool]) -> None:
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == expected


@dataclass(frozen=True, slots=True)
class _EncryptedRun:
    job: _Job
    output: RenditionOutput


@pytest.fixture(scope="module")
def encrypted_run(encode_media: EncodeMedia, tmp_path_factory: pytest.TempPathFactory) -> _EncryptedRun:
    """One encrypted encode in 1 s segments, so the landscape fixture spans several segments."""
    profile = fast_profile(segment_duration_s=ONE_SECOND_SEGMENTS)
    job = _job(tmp_path_factory.mktemp("encrypted"), encode_media.landscape, profile=profile)
    return _EncryptedRun(job=job, output=job.run(media_key=MEDIA_KEY))


def test_encode_rendition_when_encrypting_should_report_the_planned_rendition(encrypted_run: _EncryptedRun) -> None:
    # Arrange
    plan, output = encrypted_run.job.plan, encrypted_run.output

    # Act
    reported = (output.name, output.width, output.height, output.codecs, output.playlist_path)
    frame_rate = (output.frame_rate_num, output.frame_rate_den)

    # Assert
    assert reported == (plan.name, plan.width, plan.height, plan.codecs, f"{plan.name}/playlist.m3u8")
    assert frame_rate == (plan.frame_rate_num, plan.frame_rate_den)
    assert output.codecs.endswith(",mp4a.40.2")
    assert (output.has_audio, output.encrypted) == (True, True)


def test_encode_rendition_when_encrypting_should_list_the_playlist_then_every_segment_with_its_size_and_digest(
    encrypted_run: _EncryptedRun,
) -> None:
    # Arrange
    job, output = encrypted_run.job, encrypted_run.output
    segment_paths = [f"{job.plan.name}/segment_{number:04d}.ts" for number in range(output.segment_count)]

    # Act
    on_disk = {
        artifact.path: (job.output_root / artifact.path).read_bytes() for artifact in output.artifacts
    }

    # Assert
    assert output.segment_count >= 2
    assert [(artifact.path, artifact.kind) for artifact in output.artifacts] == [
        (f"{job.plan.name}/playlist.m3u8", "hls_media_playlist"),
        *[(path, "hls_segment") for path in segment_paths],
    ]
    for artifact in output.artifacts:
        assert artifact.size_bytes == len(on_disk[artifact.path])
        assert artifact.sha256 == hashlib.sha256(on_disk[artifact.path]).hexdigest()
    assert output.total_bytes == sum(len(data) for data in on_disk.values())
    assert sorted(path.name for path in job.rendition_dir.iterdir()) == sorted(
        Path(artifact.path).name for artifact in output.artifacts
    )


def _decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    unpadder = padding.PKCS7(AES_BLOCK_BITS).unpadder()
    return unpadder.update(decryptor.update(data) + decryptor.finalize()) + unpadder.finalize()


def _encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    padder = padding.PKCS7(AES_BLOCK_BITS).padder()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return encryptor.update(padder.update(data) + padder.finalize()) + encryptor.finalize()


def _crc32_mpeg2(data: bytes) -> int:
    crc = CRC32_MASK
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ CRC32_MPEG2_POLYNOMIAL if crc & CRC32_TOP_BIT else crc << 1) & CRC32_MASK
    return crc


def _first_packet_holds_a_whole_psi_section(transport_stream: bytes) -> bool:
    """Whether the first TS packet starts a PSI section (PAT, PMT, SDT) whose MPEG-2 CRC32 holds:
    over the whole section, its CRC included, the CRC is 0."""
    packet = transport_stream[:TS_PACKET_BYTES]
    if len(packet) < TS_PACKET_BYTES or packet[0] != TS_SYNC_BYTE or not packet[1] & TS_PAYLOAD_UNIT_START:
        return False
    adaptation = (packet[3] >> 4) & 0b11
    if adaptation == TS_ADAPTATION_ONLY:
        return False
    payload_start = TS_HEADER_BYTES + (1 + packet[TS_HEADER_BYTES] if adaptation == TS_ADAPTATION_AND_PAYLOAD else 0)
    section_start = payload_start + 1 + packet[payload_start]
    if section_start + PSI_SECTION_HEADER_BYTES > TS_PACKET_BYTES:
        return False
    section_length = (packet[section_start + 1] & PSI_SECTION_LENGTH_HIGH_MASK) << 8 | packet[section_start + 2]
    section_end = section_start + PSI_SECTION_HEADER_BYTES + section_length
    return section_end <= TS_PACKET_BYTES and _crc32_mpeg2(packet[section_start:section_end]) == 0


def test_encode_rendition_when_encrypting_should_encrypt_each_segment_with_the_key_and_the_iv_its_playlist_names(
    encrypted_run: _EncryptedRun,
) -> None:
    # Arrange
    job, output = encrypted_run.job, encrypted_run.output
    ivs = [bytes.fromhex(iv) for iv in IV_PATTERN.findall((job.rendition_dir / "playlist.m3u8").read_text("ascii"))]
    segments = [segment.read_bytes() for segment in _segment_paths(job, output)]

    # Act
    decrypted = [_decrypt(segment, MEDIA_KEY, iv) for segment, iv in zip(segments, ivs, strict=True)]
    second_with_the_first_iv = _decrypt(segments[1], MEDIA_KEY, ivs[0])

    # Assert
    assert len(set(ivs)) == len(segments) >= 2
    for segment, plain in zip(segments, decrypted, strict=True):
        assert segment != plain
        assert _first_packet_holds_a_whole_psi_section(plain)
    assert not _first_packet_holds_a_whole_psi_section(second_with_the_first_iv)


def test_encode_rendition_when_encrypting_should_leave_no_key_material_under_the_output_or_work_directories(
    encrypted_run: _EncryptedRun,
) -> None:
    # Arrange
    job, output = encrypted_run.job, encrypted_run.output
    artifact_paths = {job.output_root / artifact.path for artifact in output.artifacts}

    # Act
    files = _all_files(job.output_root, job.work_dir)

    # Assert
    assert set(files) == artifact_paths
    assert list(job.work_dir.iterdir()) == []
    assert all(path.read_bytes() != MEDIA_KEY for path in files)


def test_encode_rendition_when_not_encrypting_should_write_plain_transport_stream_segments_and_no_key_line(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape)

    # Act
    output = job.run()

    # Assert
    assert "#EXT-X-KEY" not in (job.rendition_dir / "playlist.m3u8").read_text(encoding="ascii")
    assert output.encrypted is False
    for segment in _segment_paths(job, output):
        assert segment.read_bytes()[0] == TS_SYNC_BYTE
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_the_admitted_duration_is_shorter_than_the_stream_should_cut_the_output_at_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape, duration_ms=ADMITTED_DURATION_MS)
    frame_ms = math.ceil(Fraction(MS_PER_SECOND * job.plan.frame_rate_den, job.plan.frame_rate_num))
    admitted_frames = ADMITTED_DURATION_MS * job.plan.frame_rate_num / (MS_PER_SECOND * job.plan.frame_rate_den)

    # Act
    output = job.run()

    # Assert
    assert ADMITTED_DURATION_MS - frame_ms <= output.duration_ms <= ADMITTED_DURATION_MS + frame_ms + DURATION_SLACK_MS
    assert abs(_decoded_frame_count(_segment_paths(job, output)) - admitted_frames) <= FRAME_COUNT_TOLERANCE


def test_encode_rendition_when_the_admitted_frame_rate_is_below_the_streams_should_encode_at_the_admitted_rate(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    facts = probe_facts(encode_media.landscape)
    lowered = dataclasses.replace(
        facts.video_streams[0], frame_rate_num=LOWERED_FRAME_RATE[0], frame_rate_den=LOWERED_FRAME_RATE[1]
    )
    job = _job(tmp_path, encode_media.landscape, facts=dataclasses.replace(facts, video_streams=(lowered,)))
    expected_frames = job.source.duration_ms * LOWERED_FRAME_RATE[0] / (MS_PER_SECOND * LOWERED_FRAME_RATE[1])
    frame_ms = MS_PER_SECOND * LOWERED_FRAME_RATE[1] // LOWERED_FRAME_RATE[0]

    # Act
    output = job.run()

    # Assert
    segments = _segment_paths(job, output)
    stream = _probe(segments[0], "-select_streams", "v:0")["streams"][0]
    assert (job.plan.frame_rate_num, job.plan.frame_rate_den) == LOWERED_FRAME_RATE
    assert stream["avg_frame_rate"] == f"{LOWERED_FRAME_RATE[0]}/{LOWERED_FRAME_RATE[1]}"
    assert abs(_decoded_frame_count(segments) - expected_frames) <= FRAME_COUNT_TOLERANCE
    # The fewer frames still span the whole admitted duration: they were dropped, not cut off.
    assert output.duration_ms >= job.source.duration_ms - frame_ms


def test_encode_rendition_when_a_decoded_frame_exceeds_the_pixel_limit_should_fail_and_leave_no_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    facts = probe_facts(encode_media.landscape)
    video = facts.video_streams[0]
    profile = _profile_capped_at(SMALL_PIXEL_CAP_EDGE, SMALL_PIXEL_CAP_EDGE, SMALL_PIXEL_CAP_EDGE**2)
    job = _job(tmp_path, encode_media.landscape, profile=profile, facts=facts)
    assert decoder_pixel_limit(profile) < video.width * video.height

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run()

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "input_limits_exceeded", "decoded_frame_too_large", False,
    )
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def _profile_capped_at(max_width: int, max_height: int, max_pixels: int) -> MediaProfile:
    source_caps = dataclasses.replace(
        FAST_PROFILE.source,
        max_display_width=max_width,
        max_display_height=max_height,
        max_display_pixels=max_pixels,
    )
    return dataclasses.replace(FAST_PROFILE, source=source_caps)


def _first_frame_rgb(path: Path, width: int, height: int) -> bytes:
    """The first frame as ffmpeg shows it (autorotation on), scaled to width x height, as RGB24."""
    return _run_tool(
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    )


def _quadrant_means(rgb: bytes, width: int, height: int) -> list[tuple[float, ...]]:
    """Mean colour of the central half of each quadrant: top left, top right, bottom left, bottom right."""
    half_width, half_height = width // 2, height // 2
    means: list[tuple[float, ...]] = []
    for top, left in ((0, 0), (0, half_width), (half_height, 0), (half_height, half_width)):
        rows = range(top + half_height // 4, top + 3 * half_height // 4)
        columns = range(left + half_width // 4, left + 3 * half_width // 4)
        pixels = [rgb[3 * (row * width + column): 3 * (row * width + column) + 3] for row in rows for column in columns]
        means.append(tuple(sum(pixel[channel] for pixel in pixels) / len(pixels) for channel in range(3)))
    return means


def _close(first: tuple[float, ...], second: tuple[float, ...], tolerance: float) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_encode_rendition_when_the_display_is_rotated_should_output_the_frame_as_ffmpeg_shows_it_without_rotation(
    degrees: int, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    source = encode_media.rotated[degrees]
    job = _job(tmp_path, source)
    plan = job.plan
    reference = _quadrant_means(_first_frame_rgb(source, plan.width, plan.height), plan.width, plan.height)

    # Act
    output = job.run()

    # Assert
    first_segment = _segment_paths(job, output)[0]
    stream = _probe(first_segment, "-select_streams", "v:0")["streams"][0]
    assert job.source.rotation_degrees == degrees
    assert (stream["width"], stream["height"]) == (plan.width, plan.height)
    assert "side_data_list" not in stream
    assert "rotate" not in stream.get("tags", {})
    encoded = _quadrant_means(_first_frame_rgb(first_segment, plan.width, plan.height), plan.width, plan.height)
    assert all(
        not _close(reference[i], reference[j], QUADRANT_SEPARATION)
        for i in range(len(QUADRANT_COLOURS))
        for j in range(i + 1, len(QUADRANT_COLOURS))
    )
    assert all(_close(got, want, QUADRANT_TOLERANCE) for got, want in zip(encoded, reference, strict=True))


def test_encode_rendition_when_the_source_carries_title_and_location_tags_should_drop_them(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    source_tags = _probe(encode_media.tagged)["format"]["tags"]
    job = _job(tmp_path, encode_media.tagged)

    # Act
    output = job.run()

    # Assert
    assert source_tags["title"] == TITLE_MARKER
    for segment in _segment_paths(job, output):
        probed = _probe(segment)
        tags = [probed["format"].get("tags", {})] + [stream.get("tags", {}) for stream in probed["streams"]]
        assert all("title" not in found and "location" not in found for found in tags)
        data = segment.read_bytes()
        assert TITLE_MARKER.encode() not in data
        assert LOCATION_MARKER.encode() not in data


def test_encode_rendition_when_the_source_is_silent_should_write_video_only_segments(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.silent)

    # Act
    output = job.run()

    # Assert
    assert output.codecs == job.plan.codecs
    assert "mp4a" not in output.codecs
    assert output.has_audio is False
    for segment in _segment_paths(job, output):
        assert [stream["codec_type"] for stream in _probe(segment)["streams"]] == ["video"]


@pytest.mark.parametrize(("fixture_name", "expected_geometry"), [("portrait", (240, 320)), ("anamorphic", (426, 240))])
def test_encode_rendition_when_the_source_is_portrait_or_anamorphic_should_write_the_planned_square_pixel_geometry(
    fixture_name: str, expected_geometry: tuple[int, int], encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, getattr(encode_media, fixture_name))

    # Act
    output = job.run()

    # Assert
    stream = _probe(_segment_paths(job, output)[0], "-select_streams", "v:0")["streams"][0]
    assert (job.plan.width, job.plan.height) == expected_geometry
    assert (output.width, output.height) == expected_geometry
    assert (stream["width"], stream["height"]) == expected_geometry
    assert stream.get("sample_aspect_ratio", "1:1") == "1:1"


def test_encode_rendition_when_the_source_is_shorter_than_one_segment_should_write_one_segment(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.subsecond)

    # Act
    output = job.run()

    # Assert
    assert output.segment_count == 1
    assert len(output.artifacts) == 2
    assert output.duration_ms < MS_PER_SECOND


def test_encode_rendition_when_the_source_is_shorter_than_half_a_second_should_write_one_segment_of_target_0(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.below_half_second)

    # Act
    output = job.run()

    # Assert
    playlist = (job.rendition_dir / "playlist.m3u8").read_text(encoding="ascii")
    assert "#EXT-X-TARGETDURATION:0\n" in playlist
    assert output.segment_count == 1
    assert [artifact.kind for artifact in output.artifacts] == ["hls_media_playlist", "hls_segment"]
    assert 1 <= output.duration_ms < MS_PER_SECOND // 2
    assert output.bandwidth_bps >= 1
    assert output.average_bandwidth_bps >= 1


def test_encode_rendition_when_the_output_passes_the_byte_allowance_should_fail_and_remove_the_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape)
    allowance = OutputAllowance(max_bytes=TINY_BYTE_ALLOWANCE, max_artifacts=GENEROUS_ALLOWANCE.max_artifacts)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(allowance=allowance)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "output_too_large", False))
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize(("spare_bytes", "accepted"), [(0, True), (-1, False)], ids=["exactly", "one byte short"])
def test_encode_rendition_when_the_allowance_is_exactly_the_output_should_accept_it_and_refuse_one_byte_less(
    spare_bytes: int, accepted: bool, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: the encoder writes its whole output at once and exits, before the monitor's first look
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment)
    output_bytes = len(playlist.encode("ascii")) + len(segment)
    allowance = OutputAllowance(max_bytes=output_bytes + spare_bytes, max_artifacts=TWO_ARTIFACTS)

    # Act
    if accepted:
        output = job.run(ffmpeg_path=executable, allowance=allowance)
    else:
        with pytest.raises(WorkerFailure) as raised:
            job.run(ffmpeg_path=executable, allowance=allowance)

    # Assert
    if accepted:
        assert output.total_bytes == output_bytes
    else:
        _assert_failure(raised, ("input_limits_exceeded", "output_too_large", False))
        assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_output_passes_the_file_allowance_should_fail_and_remove_the_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape, profile=fast_profile(segment_duration_s=ONE_SECOND_SEGMENTS))
    allowance = OutputAllowance(max_bytes=GENEROUS_ALLOWANCE.max_bytes, max_artifacts=TWO_ARTIFACTS)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(allowance=allowance)

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "input_limits_exceeded", "too_many_artifacts", False,
    )
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_deadline_has_passed_should_fail_before_anything_runs(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape)
    expired = Deadline(expires_at=time.monotonic() - 1)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(deadline=expired)

    # Assert
    assert raised.value.failure.error_class == "deadline_exceeded"
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_the_deadline_is_far_too_short_should_stop_the_encode_and_remove_the_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.landscape)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(deadline=Deadline.after_ms(FAR_TOO_SHORT_DEADLINE_MS))

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "encode_deadline_exceeded", False,
    )
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def _claimed_video(frame_rate: tuple[int, int] = (25, 1)) -> VideoStreamFacts:
    """A video stream as a source might claim it, without any media behind the claim."""
    return VideoStreamFacts(
        index=0,
        codec_name="h264",
        width=320,
        height=240,
        sample_aspect_num=1,
        sample_aspect_den=1,
        rotation_degrees=0,
        display_width=320,
        display_height=240,
        frame_rate_num=frame_rate[0],
        frame_rate_den=frame_rate[1],
        field_order="progressive",
        color_transfer=None,
        pix_fmt="yuv420p",
        dolby_vision_profile=None,
    )


def _claimed_job(
    root: Path,
    source_path: Path,
    *,
    profile: MediaProfile = PILOT_PROFILE,
    duration_ms: int = CLAIMED_DURATION_MS,
    frame_rate: tuple[int, int] = (25, 1),
) -> _Job:
    source = EncodeSource(
        path=source_path,
        demuxer="mov",
        video_index=0,
        audio_index=None,
        rotation_degrees=0,
        duration_ms=duration_ms,
    )
    plan = plan_renditions(
        _claimed_video(frame_rate), has_audio=False, requested=[SMALLEST_RENDITION], profile=profile
    )[0]
    output_root, work_dir = _directories(root)
    return _Job(source=source, plan=plan, profile=profile, output_root=output_root, work_dir=work_dir)


def test_encode_rendition_when_the_source_is_a_playlist_disguised_as_mp4_should_fail_without_following_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, encode_media.disguised_playlist, profile=FAST_PROFILE)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run()

    # Assert
    assert raised.value.failure.error_class == "encoder_failed"
    assert list(job.output_root.iterdir()) == []


# The encode's self-decode runs the same executable with -xerror: a stand-in hands that call to
# the real ffmpeg, so a stand-in's output is checked by real decoding like any other.
REAL_FFMPEG = shutil.which("ffmpeg") or "/nonexistent/ffmpeg"
REAL_SELF_DECODE = f'exec "{REAL_FFMPEG}" "$@"'
FAKE_ARGUMENTS = 'for last; do :; done\ndir="${last%/*}"\n'
ENCRYPTED_KEY_LINE = '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",IV=0x' + "0" * 32
# One whole MPEG-TS packet: the sync byte, then 187 bytes.
ONE_PACKET_SEGMENT = 'printf G > "$dir/segment_0000.ts"\nhead -c 187 /dev/zero >> "$dir/segment_0000.ts"\n'


def _playlist_script(*, target: int, extinf: str, key_line: str | None = None) -> str:
    """A shell step writing a one-segment playlist in exactly the shape ffmpeg writes."""
    return "cat > \"$dir/playlist.m3u8\" <<'PLAYLIST'\n" + _playlist_text(
        target=target, extinf=extinf, key_line=key_line
    ) + "PLAYLIST\n"


def _playlist_text(*, target: int, extinf: str, key_line: str | None = None) -> str:
    """A one-segment playlist in exactly the shape ffmpeg writes."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:6",
        f"#EXT-X-TARGETDURATION:{target}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        *([] if key_line is None else [key_line]),
        f"#EXTINF:{extinf},",
        "segment_0000.ts",
        "#EXT-X-ENDLIST",
    ]
    return "\n".join(lines) + "\n"


def _sparse_segment(size_bytes: int) -> str:
    return f'dd if=/dev/zero of="$dir/segment_0000.ts" bs=1 count=0 seek={size_bytes} 2>/dev/null\n'


# A playlist in exactly ffmpeg's shape and one whole TS packet: the right files, but no video.
SHAPED_OUTPUT = _playlist_script(target=1, extinf="1.000000") + ONE_PACKET_SEGMENT


def _real_output(media: EncodeMedia) -> tuple[str, bytes]:
    """The playlist text and the one segment of a real plain rendition ffmpeg wrote (2 s at 25 fps)."""
    rendition = media.plain_rendition
    return (rendition / "playlist.m3u8").read_text(encoding="ascii"), (rendition / "segment_0000.ts").read_bytes()


def _copying_fake(
    root: Path, playlist: str, segment: bytes, *, before: str = "", self_decode: str = REAL_SELF_DECODE
) -> str:
    """A stand-in encoder that runs `before`, writes exactly `playlist` and `segment` and exits 0."""
    prepared = root / "prepared"
    prepared.mkdir(exist_ok=True)
    (prepared / "playlist.m3u8").write_text(playlist, encoding="ascii")
    (prepared / "segment_0000.ts").write_bytes(segment)
    copy = f'cp "{prepared}/playlist.m3u8" "{prepared}/segment_0000.ts" "$dir/"\n'
    return _fake_ffmpeg(root, before + copy + "exit 0\n", self_decode=self_decode)


def _rendition_copying_fake(root: Path, rendition: Path) -> str:
    """A stand-in encoder that writes exactly the files of a real rendition and exits 0."""
    return _fake_ffmpeg(root, f'cp "{rendition}"/* "$dir/"\nexit 0\n')


def _fake_ffmpeg(root: Path, body: str, *, self_decode: str = REAL_SELF_DECODE) -> str:
    """A stand-in executable; it finds the rendition directory from its last argument, the playlist.
    Called as the self-decode (with -xerror), it runs `self_decode`, by default the real ffmpeg."""
    directory = root / "bin"
    directory.mkdir(exist_ok=True)
    script = directory / "ffmpeg"
    preamble = f'#!/bin/sh\ncase " $* " in *" -xerror "*)\n{self_decode}\n;; esac\n' + FAKE_ARGUMENTS
    script.write_text(preamble + body, encoding="ascii")
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("exit 0\n", ("encoder_failed", "rendition_output_invalid", False)),
        (f"echo '{MAX_PIXEL_TEXT}' >&2\nexit 1\n", ("input_limits_exceeded", "decoded_frame_too_large", False)),
        (
            SHAPED_OUTPUT + f"echo '{MAX_PIXEL_TEXT}' >&2\nexit 0\n",
            ("input_limits_exceeded", "decoded_frame_too_large", False),
        ),
        (f"echo '{NO_SPACE_TEXT}' >&2\nexit 1\n", ("resource_exhausted", "scratch_full", True)),
        ("echo 'Conversion failed!' >&2\nexit 1\n", ("encoder_failed", "encode_failed", False)),
        ("kill -9 $$\n", ("internal_error", "encoder_crashed", True)),
        ("exit 126\n", ("configuration_error", "ffmpeg_unavailable", False)),
        ("exit 127\n", ("configuration_error", "ffmpeg_unavailable", False)),
        (
            SHAPED_OUTPUT + 'printf x > "$dir/notes.txt"\nexit 0\n',
            ("encoder_failed", "rendition_output_invalid", False),
        ),
        (SHAPED_OUTPUT + 'mkdir "$dir/extra"\nexit 0\n', ("encoder_failed", "rendition_output_invalid", False)),
        (
            SHAPED_OUTPUT + 'rm "$dir/segment_0000.ts"\nln -s /etc/hosts "$dir/segment_0000.ts"\nexit 0\n',
            ("encoder_failed", "rendition_output_invalid", False),
        ),
        (SHAPED_OUTPUT + ': > "$dir/segment_0000.ts"\nexit 0\n', ("encoder_failed", "rendition_output_invalid", False)),
        ("head -c 70000 /dev/zero\nexit 0\n", ("internal_error", "encoder_output_unexpected", False)),
        (
            _playlist_script(target=2, extinf="2.100000") + ONE_PACKET_SEGMENT + "exit 0\n",
            ("encoder_failed", "duration_cut_failed", False),
        ),
        (
            _playlist_script(target=0, extinf="0.001000") + _sparse_segment(200_000_000) + "exit 0\n",
            ("encoder_failed", "bandwidth_out_of_range", False),
        ),
    ],
    ids=[
        "exits 0 without writing",
        "refused a frame above the pixel cap",
        "refused a frame above the pixel cap yet exited 0",
        "ran out of disk",
        "failed",
        "killed by a signal",
        "found but not executable",
        "not found by the shell",
        "wrote an extra file",
        "wrote an extra directory",
        "left a symbolic link as a segment",
        "wrote an empty segment",
        "wrote more than expected to stdout",
        "outlasted the admitted duration",
        "wrote a bandwidth beyond the contract",
    ],
)
def test_encode_rendition_when_ffmpeg_ends_this_way_should_report_the_matching_typed_failure(
    body: str, expected: tuple[str, str, bool], tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, body)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == expected
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_ffmpeg_writes_exactly_a_valid_rendition_should_accept_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment)

    # Act
    output = job.run(ffmpeg_path=executable)

    # Assert
    assert output.segment_count == 1
    assert output.duration_ms == CLAIMED_DURATION_MS
    assert output.total_bytes == len(playlist.encode("ascii")) + len(segment)
    assert [artifact.sha256 for artifact in output.artifacts][1] == hashlib.sha256(segment).hexdigest()


def test_encode_rendition_when_the_executable_is_missing_should_fail_as_unavailable_and_create_nothing(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=str(tmp_path / "no-such-ffmpeg"))

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "configuration_error", "ffmpeg_unavailable", False,
    )
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_executable_is_not_a_program_should_fail_as_unavailable(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    not_a_program = tmp_path / "ffmpeg-garbage"
    not_a_program.write_bytes(b"\x00\x01\x02 not an executable format")
    not_a_program.chmod(0o755)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=str(not_a_program))

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code) == ("configuration_error", "ffmpeg_unavailable")
    assert list(job.output_root.iterdir()) == []


RUNAWAY_BYTES = 'head -c 200000 /dev/zero > "$dir/segment_0000.ts"\nexec sleep 30\n'
# Above the claimed rendition's planned 75,000 bytes, below what the runaway encoder writes.
RUNAWAY_BYTE_ALLOWANCE = 100_000
RUNAWAY_FILES = (
    'i=0\nwhile [ $i -lt 6 ]; do printf G > "$dir/segment_000$i.ts"; i=$((i+1)); done\nexec sleep 30\n'
)


def test_encode_rendition_when_a_running_encoder_passes_the_byte_allowance_should_stop_it_at_once(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, RUNAWAY_BYTES)
    allowance = OutputAllowance(max_bytes=RUNAWAY_BYTE_ALLOWANCE, max_artifacts=GENEROUS_ALLOWANCE.max_artifacts)
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, allowance=allowance)
    elapsed = time.monotonic() - started

    # Assert
    assert raised.value.failure.code == "output_too_large"
    assert elapsed < RETURN_WITHIN_S
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize(
    ("max_segments", "max_artifacts", "expected_code"),
    [
        (2, GENEROUS_ALLOWANCE.max_artifacts, "too_many_segments"),
        (PILOT_PROFILE.encode.max_segments_per_rendition, 3, "too_many_artifacts"),
    ],
    ids=["the segment limit is the smaller", "the file allowance is the smaller"],
)
def test_encode_rendition_when_a_running_encoder_writes_too_many_files_should_stop_it_naming_the_smaller_limit(
    max_segments: int, max_artifacts: int, expected_code: str, tmp_path: Path
) -> None:
    # Arrange
    profile = fast_profile(max_segments_per_rendition=max_segments)
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=profile)
    executable = _fake_ffmpeg(tmp_path, RUNAWAY_FILES)
    allowance = OutputAllowance(max_bytes=GENEROUS_ALLOWANCE.max_bytes, max_artifacts=max_artifacts)
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, allowance=allowance)
    elapsed = time.monotonic() - started

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("input_limits_exceeded", expected_code, False)
    assert elapsed < RETURN_WITHIN_S
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_rendition_directory_vanishes_while_encoding_should_fail_retryably(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, 'rmdir "$dir"\nexec sleep 30\n')
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)
    elapsed = time.monotonic() - started

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("internal_error", "output_unreadable", True)
    assert elapsed < RETURN_WITHIN_S


@pytest.mark.parametrize(
    ("allowance", "expected_code"),
    [
        (OutputAllowance(max_bytes=0, max_artifacts=10), "output_too_large"),
        (OutputAllowance(max_bytes=1000, max_artifacts=1), "too_many_artifacts"),
    ],
)
def test_encode_rendition_when_the_allowance_cannot_hold_one_segment_should_fail_before_anything_runs(
    allowance: OutputAllowance, expected_code: str, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    marker = tmp_path / "ran"
    executable = _fake_ffmpeg(tmp_path, f'touch "{marker}"\nexit 0\n')

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, allowance=allowance)

    # Assert
    assert (raised.value.failure.error_class, raised.value.failure.code) == ("input_limits_exceeded", expected_code)
    assert not marker.exists()
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_rendition_directory_already_exists_should_raise_and_leave_it_alone(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    job.rendition_dir.mkdir()
    earlier = job.rendition_dir / "playlist.m3u8"
    earlier.write_text("kept", encoding="ascii")

    # Act
    with pytest.raises(ValueError):
        job.run(ffmpeg_path=_fake_ffmpeg(tmp_path, SHAPED_OUTPUT + "exit 0\n"))

    # Assert
    assert earlier.read_text(encoding="ascii") == "kept"


@pytest.mark.parametrize("layout", ["output inside work", "work inside output", "same directory"])
def test_encode_rendition_when_output_and_work_directories_overlap_should_raise(layout: str, tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    if layout == "output inside work":
        output_root, work_dir = job.work_dir / "output", job.work_dir
        output_root.mkdir()
    elif layout == "work inside output":
        output_root, work_dir = job.output_root, job.output_root / "work"
        work_dir.mkdir()
    else:
        output_root, work_dir = job.output_root, job.output_root
    overlapping = dataclasses.replace(job, output_root=output_root, work_dir=work_dir)

    # Act / Assert
    with pytest.raises(ValueError):
        overlapping.run()


@pytest.mark.parametrize("media_key", [bytes(15), bytes(17), MEDIA_KEY.hex()])
def test_encode_rendition_when_the_media_key_is_not_16_bytes_should_raise_and_create_nothing(
    media_key: Any, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")

    # Act
    with pytest.raises(ValueError):
        job.run(media_key=media_key)

    # Assert
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_output_allowance_for_job_when_the_output_limit_is_smaller_than_the_scratch_room_should_take_it() -> None:
    # Arrange
    source_bytes = 10 * GIB
    output_limit = 1 * GIB

    # Act
    allowance = OutputAllowance.for_job(
        max_output_bytes=output_limit, max_artifact_count=500, source_size_bytes=source_bytes, profile=PILOT_PROFILE
    )

    # Assert
    assert allowance == OutputAllowance(max_bytes=output_limit, max_artifacts=500)


def test_output_allowance_for_job_when_the_scratch_room_is_smaller_than_the_output_limit_should_take_it() -> None:
    # Arrange
    source_bytes = 10 * GIB
    scratch_room = PILOT_PROFILE.encode.max_scratch_bytes - source_bytes

    # Act
    allowance = OutputAllowance.for_job(
        max_output_bytes=scratch_room + 1, max_artifact_count=500, source_size_bytes=source_bytes, profile=PILOT_PROFILE
    )

    # Assert
    assert allowance.max_bytes == scratch_room


@pytest.mark.parametrize("excess_bytes", [0, 1])
def test_output_allowance_for_job_when_the_source_fills_the_scratch_ceiling_should_fail_as_scratch_exceeded(
    excess_bytes: int,
) -> None:
    # Arrange
    source_bytes = PILOT_PROFILE.encode.max_scratch_bytes + excess_bytes

    # Act
    with pytest.raises(WorkerFailure) as raised:
        OutputAllowance.for_job(
            max_output_bytes=GIB, max_artifact_count=500, source_size_bytes=source_bytes, profile=PILOT_PROFILE
        )

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "input_limits_exceeded", "scratch_exceeded", False,
    )


@pytest.mark.parametrize(
    "overrides",
    [{"max_output_bytes": 0}, {"max_artifact_count": 0}, {"source_size_bytes": 0}, {"max_output_bytes": True}],
)
def test_output_allowance_for_job_when_a_limit_is_not_a_positive_int_should_raise(overrides: dict[str, Any]) -> None:
    # Arrange
    arguments: dict[str, Any] = {
        "max_output_bytes": GIB, "max_artifact_count": 500, "source_size_bytes": GIB, "profile": PILOT_PROFILE,
    }
    arguments.update(overrides)

    # Act / Assert
    with pytest.raises(ValueError):
        OutputAllowance.for_job(**arguments)


def _output(total_bytes: int, artifact_count: int) -> RenditionOutput:
    artifact = ArtifactFile(path="240p/segment_0000.ts", kind="hls_segment", size_bytes=1, sha256="0" * 64)
    return RenditionOutput(
        name="240p", width=320, height=240, frame_rate_num=25, frame_rate_den=1, codecs="avc1.64001e",
        has_audio=False, encrypted=False, playlist_path="240p/playlist.m3u8",
        segment_count=artifact_count - 1, duration_ms=MS_PER_SECOND, bandwidth_bps=8, average_bandwidth_bps=8,
        total_bytes=total_bytes, artifacts=(artifact,) * artifact_count,
    )


def test_output_allowance_after_when_an_output_is_kept_should_reduce_bytes_and_files() -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=1000, max_artifacts=10)

    # Act
    remaining = allowance.after(_output(total_bytes=400, artifact_count=3))

    # Assert
    assert remaining == OutputAllowance(max_bytes=600, max_artifacts=7)


def test_output_allowance_after_when_an_output_uses_it_up_exactly_should_leave_zero() -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=400, max_artifacts=3)

    # Act
    remaining = allowance.after(_output(total_bytes=400, artifact_count=3))

    # Assert
    assert remaining == OutputAllowance(max_bytes=0, max_artifacts=0)


@pytest.mark.parametrize(("total_bytes", "artifact_count"), [(401, 3), (400, 4)])
def test_output_allowance_after_when_an_output_does_not_fit_should_raise(total_bytes: int, artifact_count: int) -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=400, max_artifacts=3)

    # Act / Assert
    with pytest.raises(ValueError):
        allowance.after(_output(total_bytes=total_bytes, artifact_count=artifact_count))


RESIZE_CAP_EDGE = 64
RESIZED_LARGE_PIXELS = 192 * 192
DAV1D_REFUSAL_TEXT = "Frame size 128x128 exceeds limit 4096"
SMALL_STDERR_TAIL_BYTES = 1024
# About 12 KB of later warnings: the refusal is far before a 1 KiB kept tail.
LATER_WARNINGS = (
    'i=0\nwhile [ $i -lt 200 ]; do echo "a later warning that pushes earlier lines out of the tail" >&2; '
    "i=$((i+1)); done\n"
)
AT_CAP_WIDTH, AT_CAP_HEIGHT = 200, 100
CROP_TOP_COLOUR = (255.0, 0.0, 0.0)
CROP_BOTTOM_COLOUR = (0.0, 0.0, 255.0)
TARGET_DURATION_PATTERN = re.compile(r"#EXT-X-TARGETDURATION:([0-9]+)")
SLIDESHOW_DURATION_S = 20
TIE_DURATION_MS = 1500
FIRST_IV = bytes(16)


@pytest.mark.parametrize("fixture_name", ["resized_h264", "resized_av1"])
def test_encode_rendition_when_frames_grow_past_the_pixel_limit_mid_stream_should_refuse_the_source(
    fixture_name: str, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    source_path = getattr(encode_media, fixture_name)
    if source_path is None:
        pytest.skip("the local ffmpeg lacks libsvtav1 or libdav1d")
    profile = _profile_capped_at(RESIZE_CAP_EDGE, RESIZE_CAP_EDGE, RESIZE_CAP_EDGE * RESIZE_CAP_EDGE)
    job = _job(tmp_path, source_path, profile=profile)
    assert decoder_pixel_limit(profile) < RESIZED_LARGE_PIXELS

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run()

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "decoded_frame_too_large", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_a_refusal_is_far_before_the_kept_stderr_tail_should_still_refuse_the_source(
    tmp_path: Path,
) -> None:
    # Arrange
    profile = fast_profile(max_stderr_bytes=SMALL_STDERR_TAIL_BYTES)
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=profile)
    executable = _fake_ffmpeg(tmp_path, f"echo '{DAV1D_REFUSAL_TEXT}' >&2\n" + LATER_WARNINGS + "exit 0\n")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "decoded_frame_too_large", False))


def test_encode_rendition_when_a_frame_sits_exactly_at_the_pixel_cap_with_an_unaligned_width_should_encode_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    profile = _profile_capped_at(AT_CAP_WIDTH, AT_CAP_WIDTH, AT_CAP_WIDTH * AT_CAP_HEIGHT)
    job = _job(tmp_path, encode_media.at_pixel_cap, profile=profile)

    # Act
    output = job.run()

    # Assert
    assert (output.width, output.height) == (AT_CAP_WIDTH, AT_CAP_HEIGHT)


def test_encode_rendition_when_the_container_crops_the_frame_should_encode_the_whole_coded_frame(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.cropped)
    plan = job.plan

    # Act
    output = job.run()

    # Assert
    rgb = _first_frame_rgb(_segment_paths(job, output)[0], plan.width, plan.height)
    top_left, _, bottom_left, _ = _quadrant_means(rgb, plan.width, plan.height)
    assert (output.width, output.height) == (plan.width, plan.height)
    assert _close(top_left, CROP_TOP_COLOUR, QUADRANT_TOLERANCE)
    assert _close(bottom_left, CROP_BOTTOM_COLOUR, QUADRANT_TOLERANCE)


def test_encode_rendition_when_the_source_shows_a_picture_every_ten_seconds_should_keep_segments_within_target(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.slideshow)
    segment_s = job.profile.encode.segment_duration_s

    # Act
    output = job.run()

    # Assert
    playlist = (job.rendition_dir / "playlist.m3u8").read_text(encoding="ascii")
    match = TARGET_DURATION_PATTERN.search(playlist)
    assert match is not None
    assert (job.plan.frame_rate_num, job.plan.frame_rate_den) == (1, 1)
    assert int(match.group(1)) <= segment_s
    assert output.segment_count >= math.ceil(SLIDESHOW_DURATION_S / segment_s)


def test_encode_rendition_when_the_source_lasts_one_and_a_half_seconds_at_60_fps_should_encode_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.tie)

    # Act
    output = job.run()

    # Assert
    assert output.segment_count == 1
    assert output.duration_ms == TIE_DURATION_MS


def test_encode_rendition_when_a_segment_exceeds_the_contracts_4_gib_should_refuse_it(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    allowance = OutputAllowance(max_bytes=8 * GIB, max_artifacts=GENEROUS_ALLOWANCE.max_artifacts)
    executable = _fake_ffmpeg(
        tmp_path, _playlist_script(target=1, extinf="1.000000") + _sparse_segment(4 * GIB + 1) + "exit 0\n"
    )

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, allowance=allowance)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "artifact_too_large", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_ffmpeg_cannot_be_started_for_lack_of_resources_should_fail_retryably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, SHAPED_OUTPUT + "exit 0\n")

    def no_process(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(core.encode, "run_bounded", no_process)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "encoder_start_failed", True))
    assert "EAGAIN" in raised.value.failure.detail
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_key_cannot_be_written_should_fail_retryably_and_leave_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, SHAPED_OUTPUT + "exit 0\n")

    def disk_full(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(core.encode, "hls_key_info", disk_full)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert "ENOSPC" in raised.value.failure.detail
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_an_encrypted_encode_fails_should_leave_nothing_under_the_work_directory(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, "exit 1\n")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    _assert_failure(raised, ("encoder_failed", "encode_failed", False))
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []




class _SwitchedDeadline(Deadline):
    """A caller's deadline that holds until its switch is set; the stage deadlines it gives out
    (`within_ms`) hold until the stage switch is set. No clock decides either."""

    def __init__(self, switch: threading.Event, stage_switch: threading.Event | None = None) -> None:
        super().__init__(expires_at=time.monotonic() + GENEROUS_DEADLINE_MS / MS_PER_SECOND)
        # Deadline is frozen; this subclass's own attributes live in its instance dictionary.
        object.__setattr__(self, "_switch", switch)
        object.__setattr__(self, "_stage_switch", stage_switch or switch)

    def expired(self) -> bool:
        return self._switch.is_set() or super().expired()

    def remaining_s(self) -> float:
        return 0.0 if self._switch.is_set() else super().remaining_s()

    def within_ms(self, ms: int) -> Deadline:
        return _SwitchedDeadline(self._stage_switch)


class _HashSettingSwitch:
    """A sha256 that sets `switch` once the verification has hashed `after_bytes` in all."""

    def __init__(self, switch: threading.Event, after_bytes: int, hashed: list[int]) -> None:
        self._digest = hashlib.sha256()
        self._switch = switch
        self._after_bytes = after_bytes
        self._hashed = hashed

    def update(self, data: bytes) -> None:
        self._digest.update(data)
        self._hashed[0] += len(data)
        if self._hashed[0] >= self._after_bytes:
            self._switch.set()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


TINY_BUDGET_PROFILE = fast_profile(encode_base_wall_ms=300, encode_wall_ms_per_media_s=1)
TINY_BUDGET_MS = 302  # 300 ms + 1 ms for each of the claimed 2 s
SHORTER_CALLER_DEADLINE_MS = 150
# 499 ms + 1 ms per admitted second: 500 ms for the dense source's 1 s; it takes ~9 s to decode here.
DENSE_BUDGET_PROFILE = fast_profile(encode_base_wall_ms=499, encode_wall_ms_per_media_s=1)
DENSE_DECLARED_MS = 1000
SMALL_READ_BYTES = 4096
GROWN_REFUSAL_TEXT = "Picture size 40000x40000 is invalid"
CASCADE_TEXT = "Picture size 0x0 is invalid"
VIDEO_STREAM_IDS = range(0xE0, 0xF0)
CODEX_PREFIX_BYTES = 1880
SLIDESHOW_SEGMENT_MS = 6000
SLIDESHOW_SEGMENT_FRAMES = 6
SEGMENTED_FILES = 5  # the playlist and four 1 s segments
SEGMENTED_DURATION_MS = 4000
NEVER_CHECKED_BYTES = 1 << 40
PLENTY_OF_ROOM = SimpleNamespace(f_bavail=1 << 20, f_frsize=4096)
NO_ROOM = SimpleNamespace(f_bavail=0, f_frsize=4096)


def _wait_until_gone(pid: int) -> bool:
    give_up_at = time.monotonic() + RETURN_WITHIN_S
    while time.monotonic() < give_up_at:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_encode_rendition_when_the_encode_outlasts_its_time_budget_should_stop_it_as_over_budget(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=TINY_BUDGET_PROFILE)
    pid_file = tmp_path / "encoder.pid"
    executable = _fake_ffmpeg(tmp_path, f'echo $$ > "{pid_file}"\nexec sleep 30\n')
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)
    elapsed = time.monotonic() - started

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert f"{TINY_BUDGET_MS} ms for {CLAIMED_DURATION_MS} ms" in raised.value.failure.detail
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(int(pid_file.read_text(encoding="ascii")))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_callers_deadline_comes_before_the_budget_should_report_the_deadline(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=TINY_BUDGET_PROFILE)
    executable = _fake_ffmpeg(tmp_path, "exec sleep 30\n")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=Deadline.after_ms(SHORTER_CALLER_DEADLINE_MS))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_deadline_exceeded", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_a_source_decodes_far_more_frames_than_it_declares_should_stop_at_the_budget(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    if encode_media.dense_av1 is None:
        pytest.skip("the local ffmpeg lacks libsvtav1 or libdav1d")
    job = _job(tmp_path, encode_media.dense_av1, profile=DENSE_BUDGET_PROFILE)
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run()
    elapsed = time.monotonic() - started

    # Assert
    assert job.source.duration_ms == DENSE_DECLARED_MS
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert elapsed < RETURN_WITHIN_S
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize(
    ("whose", "expected_code"),
    [("budget", "encode_budget_exceeded"), ("caller", "encode_deadline_exceeded")],
    ids=["the time budget runs out", "the caller's deadline runs out"],
)
def test_encode_rendition_when_time_runs_out_while_the_segment_is_being_read_should_stop_the_verification(
    whose: str, expected_code: str, encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: time runs out halfway through the last (only) segment, read in small pieces
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment)
    stage_switch = threading.Event()
    caller = _SwitchedDeadline(stage_switch if whose == "caller" else threading.Event(), stage_switch)
    after_bytes = len(playlist.encode("ascii")) + len(segment) // 2
    hashed = [0]
    monkeypatch.setattr(core.encode, "HASH_CHUNK_BYTES", SMALL_READ_BYTES)
    monkeypatch.setattr(core.encode, "DEADLINE_CHECK_BYTES", SMALL_READ_BYTES)
    monkeypatch.setattr(
        core.encode, "hashlib", SimpleNamespace(sha256=lambda: _HashSettingSwitch(stage_switch, after_bytes, hashed))
    )

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=caller)

    # Assert
    _assert_failure(raised, ("deadline_exceeded", expected_code, False))
    assert hashed[0] < len(playlist.encode("ascii")) + len(segment)
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_time_runs_out_once_ffmpeg_has_exited_should_not_verify_the_output(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment)
    stage_switch = threading.Event()
    real_run_bounded = core.encode.run_bounded

    def run_then_expire(*args: Any, **kwargs: Any) -> Any:
        result = real_run_bounded(*args, **kwargs)
        stage_switch.set()
        return result

    monkeypatch.setattr(core.encode, "run_bounded", run_then_expire)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(threading.Event(), stage_switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_time_runs_out_on_a_disk_that_fills_up_should_keep_the_deadline_failure(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: room after the run, none once the verification has started
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment)
    stage_switch = threading.Event()
    hashed = [0]
    free_space = iter([PLENTY_OF_ROOM])
    monkeypatch.setattr(os, "statvfs", lambda path: next(free_space, NO_ROOM))
    monkeypatch.setattr(core.encode, "HASH_CHUNK_BYTES", SMALL_READ_BYTES)
    monkeypatch.setattr(core.encode, "DEADLINE_CHECK_BYTES", SMALL_READ_BYTES)
    monkeypatch.setattr(
        core.encode, "hashlib", SimpleNamespace(sha256=lambda: _HashSettingSwitch(stage_switch, 1, hashed))
    )

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(threading.Event(), stage_switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))


def test_encode_rendition_when_a_decoder_finds_a_frame_size_invalid_should_refuse_the_source(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, f"echo '{GROWN_REFUSAL_TEXT}' >&2\nexit 0\n")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "decoded_frame_too_large", False))


def test_encode_rendition_when_a_decoder_only_reports_a_zero_size_frame_should_not_take_it_for_a_refusal(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment, before=f"echo '{CASCADE_TEXT}' >&2\n")

    # Act
    output = job.run(ffmpeg_path=executable)

    # Assert
    assert output.segment_count == 1


def test_encode_rendition_when_a_vp9_keyframe_declares_a_frame_too_large_to_be_valid_should_refuse_the_source(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    if encode_media.oversized_vp9 is None:
        pytest.skip("the local ffmpeg lacks libvpx-vp9")
    job = _job(tmp_path, encode_media.oversized_vp9)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run()

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", "decoded_frame_too_large", False))
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize("fixture_name", ["landscape", "ntsc", "tie", "subsecond", "below_half_second", "slideshow"])
@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_encode_rendition_when_encoding_a_real_source_should_prove_every_segment_covers_its_duration(
    fixture_name: str, encrypted: bool, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: the pilot's own profile, x264 "medium" with B-frames
    job = _job(tmp_path, getattr(encode_media, fixture_name), profile=PILOT_PROFILE)

    # Act
    output = job.run(media_key=MEDIA_KEY if encrypted else None)

    # Assert
    assert output.segment_count >= 1


def _video_pes_start_offsets(transport_stream: bytes) -> list[int]:
    """The offsets of the TS packets that open a video PES packet, one per frame."""
    offsets: list[int] = []
    for offset in range(0, len(transport_stream), TS_PACKET_BYTES):
        packet = transport_stream[offset: offset + TS_PACKET_BYTES]
        has_adaptation_field = (packet[3] >> 4) & 0b11 == TS_ADAPTATION_AND_PAYLOAD
        payload = packet[TS_HEADER_BYTES + (1 + packet[TS_HEADER_BYTES] if has_adaptation_field else 0):]
        if packet[1] & TS_PAYLOAD_UNIT_START and payload[:3] == b"\x00\x00\x01" and payload[3] in VIDEO_STREAM_IDS:
            offsets.append(offset)
    return offsets


def _without_the_last_frames(segment: bytes, frames: int) -> bytes:
    """`segment` cut at the start of its last `frames` frames in decode order."""
    return segment[: _video_pes_start_offsets(segment)[-frames]]


def _cut_inside_the_last_frame(segment: bytes) -> bytes:
    """`segment` cut two packets into its last frame's payload, after its PES header and PTS."""
    return segment[: _video_pes_start_offsets(segment)[-1] + 2 * TS_PACKET_BYTES]


def _packet_pid(packet: bytes) -> int:
    return (packet[1] & 0x1F) << 8 | packet[2]


def _without_the_trailing_audio(segment: bytes) -> bytes:
    """`segment` cut right after its last video packet: only audio is lost."""
    video_pid = _packet_pid(segment[_video_pes_start_offsets(segment)[0]:])
    last_video = max(
        offset for offset in range(0, len(segment), TS_PACKET_BYTES) if _packet_pid(segment[offset:]) == video_pid
    )
    cut = segment[: last_video + TS_PACKET_BYTES]
    assert len(cut) < len(segment)
    return cut


@pytest.mark.parametrize(
    "truncate",
    [
        lambda segment: segment[:TS_PACKET_BYTES],
        lambda segment: segment[: 3 * TS_PACKET_BYTES],
        lambda segment: segment[:CODEX_PREFIX_BYTES],
        lambda segment: _without_the_last_frames(segment, 1),
        lambda segment: _without_the_last_frames(segment, 2),
        lambda segment: _without_the_last_frames(segment, 3),
        _cut_inside_the_last_frame,
    ],
    ids=[
        "its first 188 bytes",
        "its first 564 bytes",
        "its first 1,880 bytes",
        "without its last frame in decode order",
        "without its last two frames in decode order",
        "without its last three frames in decode order",
        "cut inside its last frame",
    ],
)
def test_encode_rendition_when_a_segment_was_cut_on_a_packet_boundary_should_refuse_the_rendition(
    truncate: Any, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: a segment x264 "medium" wrote, with B-frames
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, truncate(segment))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert list(job.output_root.iterdir()) == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap: the cut removes whole audio PES packets, so the audio merely ends early and decodes "
        "clean; nothing in the segment tells how long its audio should run"
    ),
)
def test_encode_rendition_when_the_last_segment_lost_its_trailing_audio_should_refuse_the_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, _without_the_trailing_audio(segment))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_encode_rendition_when_a_middle_segment_was_cut_inside_its_last_frame_should_refuse_the_rendition(
    encrypted: bool, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: segment 1 of 4, cut two packets into its last frame (re-padded when encrypted)
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SEGMENTED_DURATION_MS)
    rendition = tmp_path / "rendition"
    rendition.mkdir()
    playlist = (encode_media.segmented_rendition / "playlist.m3u8").read_text(encoding="ascii")
    for number in range(SEGMENTED_FILES - 1):
        name = f"segment_{number:04d}.ts"
        segment = (encode_media.segmented_rendition / name).read_bytes()
        if number == 1:
            segment = _cut_inside_the_last_frame(segment)
        if encrypted:
            segment = _encrypt(segment, MEDIA_KEY, number.to_bytes(16, "big"))
            playlist = playlist.replace(
                f"#EXTINF:1.000000,\n{name}", f"{ENCRYPTED_KEY_LINE[:-32]}{number:032x}\n#EXTINF:1.000000,\n{name}"
            )
        (rendition / name).write_bytes(segment)
    (rendition / "playlist.m3u8").write_text(playlist, encoding="ascii")
    executable = _rendition_copying_fake(tmp_path, rendition)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY if encrypted else None)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))


def test_encode_rendition_when_an_encrypted_segment_was_cut_inside_its_last_frame_and_re_padded_should_refuse_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: valid padding, whole packets, the last frame's PES header intact
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    keyed = playlist.replace("#EXTINF:", ENCRYPTED_KEY_LINE + "\n#EXTINF:", 1)
    executable = _copying_fake(tmp_path, keyed, _encrypt(_cut_inside_the_last_frame(segment), MEDIA_KEY, FIRST_IV))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_a_one_fps_segment_holds_four_of_its_six_frames_should_refuse_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: the slideshow's first segment, 6 frames at 1 fps, losing its last two in decode order
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SLIDESHOW_SEGMENT_MS, frame_rate=(1, 1))
    segment = (encode_media.slideshow_rendition / "segment_0000.ts").read_bytes()
    playlist = _playlist_text(target=SLIDESHOW_SEGMENT_MS // MS_PER_SECOND, extinf="6.000000")
    assert len(_video_pes_start_offsets(segment)) == SLIDESHOW_SEGMENT_FRAMES
    executable = _copying_fake(tmp_path, playlist, _without_the_last_frames(segment, 2))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))


def test_encode_rendition_when_a_one_fps_segment_is_whole_should_accept_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SLIDESHOW_SEGMENT_MS, frame_rate=(1, 1))
    segment = (encode_media.slideshow_rendition / "segment_0000.ts").read_bytes()
    playlist = _playlist_text(target=SLIDESHOW_SEGMENT_MS // MS_PER_SECOND, extinf="6.000000")
    executable = _copying_fake(tmp_path, playlist, segment)

    # Act
    output = job.run(ffmpeg_path=executable)

    # Assert
    assert output.duration_ms == SLIDESHOW_SEGMENT_MS


@pytest.mark.parametrize(
    ("target", "extinf"),
    [("4", "4.000000"), ("1", "1.000000")],
    ids=["twice as long as its video", "half as long as its video"],
)
def test_encode_rendition_when_a_segment_declares_a_duration_its_video_does_not_have_should_refuse_it(
    target: str, extinf: str, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: the real segment holds 2 s of video
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=6000)
    playlist, segment = _real_output(encode_media)
    declared = playlist.replace("#EXT-X-TARGETDURATION:2", f"#EXT-X-TARGETDURATION:{target}").replace(
        "#EXTINF:2.000000,", f"#EXTINF:{extinf},"
    )
    executable = _copying_fake(tmp_path, declared, segment)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))


def test_encode_rendition_when_a_plain_segment_gains_a_byte_should_refuse_it_as_not_whole_packets(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    executable = _copying_fake(tmp_path, playlist, segment + b"\x47")

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert "not MPEG-TS" in raised.value.failure.detail


def test_encode_rendition_when_a_later_packet_lost_its_sync_byte_should_refuse_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    playlist, segment = _real_output(encode_media)
    damaged = bytearray(segment)
    damaged[len(segment) // TS_PACKET_BYTES // 2 * TS_PACKET_BYTES] = 0x48
    executable = _copying_fake(tmp_path, playlist, bytes(damaged))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert "not MPEG-TS" in raised.value.failure.detail


def _encrypted_output(media: EncodeMedia, *, drop_last_block: bool) -> tuple[str, bytes]:
    playlist, segment = _real_output(media)
    keyed = playlist.replace("#EXTINF:", ENCRYPTED_KEY_LINE + "\n#EXTINF:", 1)
    encrypted = _encrypt(segment, MEDIA_KEY, FIRST_IV)
    return keyed, encrypted[: -AES_BLOCK_BITS // 8] if drop_last_block else encrypted


def test_encode_rendition_when_an_encrypted_segment_is_whole_should_accept_it(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))

    # Act
    output = job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    assert output.segment_count == 1


@pytest.mark.parametrize(
    "work_dir_name",
    [pytest.param('work-"quoted"', id="a-double-quote"), pytest.param("work-back\\slash", id="a-backslash")],
)
def test_encode_rendition_when_the_key_line_cannot_carry_the_key_files_path_should_raise_and_leave_nothing(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, work_dir_name: str
) -> None:
    # Arrange: without a memory file the key is a file under the work directory, and the playlist the
    # rendition is decoded through names that file between double quotes
    monkeypatch.setattr(core.hls_key, "memory_key_supported", lambda: False)
    work_dir = tmp_path / work_dir_name
    work_dir.mkdir()
    job = dataclasses.replace(_claimed_job(tmp_path, tmp_path / "source.mp4"), work_dir=work_dir)
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))

    # Act: the encode refuses the path itself; a decoder that does not find the key under the path
    # it read would have a whole rendition refused as one that does not decode
    with pytest.raises(ValueError, match="key line"):
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    assert list(job.output_root.iterdir()) == []
    assert list(work_dir.iterdir()) == []


def test_encode_rendition_when_an_encrypted_segment_lost_its_last_block_should_refuse_the_rendition(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: still a multiple of 16 bytes, but the last block is no longer padding
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=True))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_disk_is_full_after_a_clean_exit_should_report_scratch_full_retryably(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: an output that would pass every check
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_real_output(encode_media))
    monkeypatch.setattr(os, "statvfs", lambda path: NO_ROOM)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "scratch_full", True))
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize(
    ("body", "free_space"),
    [
        (': > "$dir/playlist.m3u8"\nexit 0\n', [PLENTY_OF_ROOM]),
        ("exit 1\n", []),
    ],
    ids=["exits 0 with an empty playlist as the disk fills", "exits 1 without saying why"],
)
def test_encode_rendition_when_a_failure_meets_a_full_disk_should_report_scratch_full_retryably(
    body: str, free_space: list[SimpleNamespace], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, body)
    readings = iter(free_space)
    monkeypatch.setattr(os, "statvfs", lambda path: next(readings, NO_ROOM))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "scratch_full", True))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_playlist_is_empty_and_the_disk_has_room_should_report_invalid_output(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _fake_ffmpeg(tmp_path, ': > "$dir/playlist.m3u8"\nexit 0\n')

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))


class _DigestCountingSwitch:
    """A sha256 that counts the files finalized and sets `switch` once `after_files` were, or once
    `after_bytes` were hashed in all."""

    def __init__(self, switch: threading.Event, finished: list[int], hashed: list[int], *, after_files: int,
                 after_bytes: int) -> None:
        self._digest = hashlib.sha256()
        self._switch = switch
        self._finished = finished
        self._hashed = hashed
        self._after_files = after_files
        self._after_bytes = after_bytes

    def update(self, data: bytes) -> None:
        self._digest.update(data)
        self._hashed[0] += len(data)
        if self._hashed[0] >= self._after_bytes:
            self._switch.set()

    def hexdigest(self) -> str:
        self._finished[0] += 1
        if self._finished[0] >= self._after_files:
            self._switch.set()
        return self._digest.hexdigest()


def _rendition_bytes(rendition: Path) -> int:
    return sum(path.stat().st_size for path in rendition.iterdir())


def _switch_on_digests(
    monkeypatch: pytest.MonkeyPatch, switch: threading.Event, *, after_files: int, after_bytes: int
) -> list[int]:
    """Routes the verification's hashing through a counting sha256; returns the finalized-file count."""
    finished, hashed = [0], [0]
    monkeypatch.setattr(core.encode, "DEADLINE_CHECK_BYTES", NEVER_CHECKED_BYTES)
    monkeypatch.setattr(
        core.encode,
        "hashlib",
        SimpleNamespace(
            sha256=lambda: _DigestCountingSwitch(
                switch, finished, hashed, after_files=after_files, after_bytes=after_bytes
            )
        ),
    )
    return finished


def test_encode_rendition_when_the_budget_runs_out_after_the_first_file_should_stop_before_reading_the_last(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: several small segments, each below the byte count between checks
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SEGMENTED_DURATION_MS)
    executable = _rendition_copying_fake(tmp_path, encode_media.segmented_rendition)
    stage_switch = threading.Event()
    finished = _switch_on_digests(monkeypatch, stage_switch, after_files=1, after_bytes=NEVER_CHECKED_BYTES)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(threading.Event(), stage_switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert finished[0] < SEGMENTED_FILES
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_the_budget_runs_out_while_the_last_file_is_read_should_not_succeed(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: time runs out once the last byte of the last segment is hashed, no check inside files
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SEGMENTED_DURATION_MS)
    executable = _rendition_copying_fake(tmp_path, encode_media.segmented_rendition)
    stage_switch = threading.Event()
    total = _rendition_bytes(encode_media.segmented_rendition)
    finished = _switch_on_digests(monkeypatch, stage_switch, after_files=NEVER_CHECKED_BYTES, after_bytes=total)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(threading.Event(), stage_switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert finished[0] == SEGMENTED_FILES


def test_encode_rendition_when_the_budget_runs_out_during_the_self_decode_should_not_succeed(
    encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_real_output(encode_media))
    stage_switch = threading.Event()
    real_self_decode = core.encode._self_decode

    def decode_then_expire(*args: Any, **kwargs: Any) -> None:
        real_self_decode(*args, **kwargs)
        stage_switch.set()

    monkeypatch.setattr(core.encode, "_self_decode", decode_then_expire)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(threading.Event(), stage_switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_every_segment_of_a_real_rendition_is_whole_should_count_every_frame(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", duration_ms=SEGMENTED_DURATION_MS)
    executable = _rendition_copying_fake(tmp_path, encode_media.segmented_rendition)

    # Act
    output = job.run(ffmpeg_path=executable)

    # Assert
    assert output.segment_count == SEGMENTED_FILES - 1
    assert output.duration_ms == SEGMENTED_DURATION_MS


PROGRESS_END_TEXT = "progress=end"


@pytest.mark.parametrize(
    ("self_decode", "expected"),
    [
        ("kill -9 $$", ("internal_error", "self_check_crashed", True)),
        ("exit 127", ("configuration_error", "ffmpeg_unavailable", False)),
        ("exit 1", ("encoder_failed", "rendition_output_invalid", False)),
        (f"echo 'frame=1'; echo '{PROGRESS_END_TEXT}'; exit 0", ("encoder_failed", "rendition_output_invalid", False)),
        ("echo 'frame=50'; exit 0", ("internal_error", "self_check_output_invalid", False)),
        (
            f"echo 'frame=fifty'; echo '{PROGRESS_END_TEXT}'; exit 0",
            ("internal_error", "self_check_output_invalid", False),
        ),
        (f"echo '{PROGRESS_END_TEXT}'; exit 0", ("internal_error", "self_check_output_invalid", False)),
        ("head -c 2000000 /dev/zero; exit 0", ("internal_error", "self_check_output_invalid", False)),
    ],
    ids=[
        "killed by a signal",
        "not found by the shell",
        "decoding error",
        "fewer frames than the segments carry",
        "no end of progress",
        "a frame count that is not a number",
        "no frame count",
        "too much output",
    ],
)
def test_encode_rendition_when_the_self_decode_ends_this_way_should_report_the_matching_typed_failure(
    self_decode: str, expected: tuple[str, str, bool], encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: an output that passes every structural check
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_real_output(encode_media), self_decode=self_decode)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, expected)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_the_self_decode_outlasts_the_budget_should_stop_it_as_over_budget(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=TINY_BUDGET_PROFILE)
    executable = _copying_fake(tmp_path, *_real_output(encode_media), self_decode="exec sleep 30")
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)
    elapsed = time.monotonic() - started

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_budget_exceeded", False))
    assert elapsed < RETURN_WITHIN_S


def test_encode_rendition_when_the_self_decode_of_an_encrypted_rendition_ends_should_leave_no_key_behind(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))

    # Act
    output = job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    assert output.segment_count == 1
    assert list(job.work_dir.iterdir()) == []
    assert all(path.read_bytes() != MEDIA_KEY for path in _all_files(job.output_root, job.work_dir))


# --- a log line whose handler fails ------------------------------------------------------------------
# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while an error is still
# being handled would put that error's own text into the report. `failing_log` makes every line's
# handler fail; how the rendition ends is how it ends without it. Unless a test takes `encode_media`,
# no ffmpeg runs: a stand-in for run_bounded ends as the test says, having written what the test says.

# What the text of an error holds that must never reach the worker's output: the path an operating
# system's error names, the path a removal that failed names, and what a bug choked on.
PATH_CANARY = "/CANARY/path/in/an/os/errors/text"
REMOVAL_CANARY = "/CANARY/path/of/what/could/not/be/removed"
BUG_CANARY = "CANARYtextOfABugUnderTheEncode0123"
EXITED_CLEANLY = BoundedResult(
    returncode=0, stdout=b"", stderr_tail=b"", timed_out=False, stdout_overflow=False, stderr_truncated=False
)
EXITED_WITH_AN_ERROR = dataclasses.replace(EXITED_CLEANLY, returncode=1)
STOPPED_BY_ITS_MONITOR = dataclasses.replace(EXITED_CLEANLY, returncode=None, stopped_by_monitor=True)
ONE_TS_PACKET = bytes([TS_SYNC_BYTE]) + bytes(TS_PACKET_BYTES - 1)
TWO_SEGMENT_PLAYLIST = _playlist_text(target=1, extinf="1.000000").replace(
    "#EXT-X-ENDLIST\n", "#EXTINF:1.000000,\nsegment_0001.ts\n#EXT-X-ENDLIST\n"
)
SELF_DECODE_CWD = "cwd"


def _raising(error: BaseException) -> Callable[..., Any]:
    """A stand-in for a function that fails with `error`."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    return fail


def _encode_os(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Replace what core.encode sees of `os`, leaving the real module alone for everything else."""
    view = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if not name.startswith("__")})
    for name, replacement in overrides.items():
        setattr(view, name, replacement)
    monkeypatch.setattr(core.encode, "os", view)


def _ffmpeg_stand_in(monkeypatch: pytest.MonkeyPatch, run: Callable[[Path, Any], BoundedResult]) -> None:
    """Put `run` where the encode starts ffmpeg: it is handed the rendition's directory and the output
    monitor, and returns how ffmpeg ended, or raises what starting it raised."""

    def run_bounded(argv: list[str], **limits: Any) -> BoundedResult:
        return run(Path(argv[-1]).parent, limits.get("monitor"))

    monkeypatch.setattr(core.encode, "run_bounded", run_bounded)


def _wrote_a_shaped_rendition(rendition_dir: Path, monitor: Any) -> BoundedResult:
    """An ffmpeg that exited well, having written a playlist in exactly its shape and one whole TS
    packet: the right files, but no video."""
    (rendition_dir / "playlist.m3u8").write_text(_playlist_text(target=1, extinf="1.000000"), encoding="ascii")
    (rendition_dir / "segment_0000.ts").write_bytes(ONE_TS_PACKET)
    return EXITED_CLEANLY


def _wrote_nothing(rendition_dir: Path, monitor: Any) -> BoundedResult:
    return EXITED_CLEANLY


def _self_decode_stand_in(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Let the encode's own run of ffmpeg through, and have the self-decode's run raise `error`."""
    _self_decode_raising(monkeypatch, lambda: error)


def _self_decode_raising(monkeypatch: pytest.MonkeyPatch, make_error: Callable[[], BaseException]) -> None:
    """Let the encode's own run of ffmpeg through, and have the self-decode's run raise what
    `make_error` makes for it."""
    real_run_bounded = core.encode.run_bounded
    runs: list[list[str]] = []

    def run_bounded(argv: list[str], **limits: Any) -> BoundedResult:
        runs.append(argv)
        if len(runs) > 1:
            raise make_error()
        return real_run_bounded(argv, **limits)

    monkeypatch.setattr(core.encode, "run_bounded", run_bounded)


def _fail_the_making_of(monkeypatch: pytest.MonkeyPatch, name: str, error: OSError) -> None:
    """Make the making of a directory called `name` fail with `error`; every other one is made."""
    real_mkdir = Path.mkdir

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == name:
            raise error
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)


def _refuse_the_removal_of(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Make the removal of `directory` fail, on an error whose own text names a path; every other
    tree is removed."""
    real_rmtree = shutil.rmtree

    def rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path) == directory:
            raise PermissionError(errno.EACCES, "Permission denied", REMOVAL_CANARY)
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(core.encode.shutil, "rmtree", rmtree)


def _refuse_the_removal_of_key_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the removal of the key's files fail, on an error whose own text names a path."""

    def refuse(path: Path, missing_ok: bool = False) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", REMOVAL_CANARY)

    monkeypatch.setattr(Path, "unlink", refuse)


def _carries_nothing_of_another_error(raised: pytest.ExceptionInfo[WorkerFailure]) -> bool:
    return raised.value.__cause__ is None and raised.value.__context__ is None


def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_a_monitor_that_cannot_look_should_keep_the_path_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failing_log: None
) -> None:
    # Arrange: the rendition directory is gone when the monitor looks, so its look fails on the
    # operating system's own error, whose text names the directory
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")

    def lost_its_directory(rendition_dir: Path, monitor: Any) -> BoundedResult:
        rendition_dir.rmdir()
        return EXITED_CLEANLY if monitor() else STOPPED_BY_ITS_MONITOR

    _ffmpeg_stand_in(monkeypatch, lost_its_directory)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert: logging did report its failure on that line, with nothing of the path the error named
    written = reported_log_failures(capsys, "the rendition output could not be read")
    assert str(tmp_path) not in written
    _assert_failure(raised, ("internal_error", "output_unreadable", True))
    assert raised.value.failure.detail == "the rendition output could not be read: ENOENT"


def _an_unreadable_playlist(monkeypatch: pytest.MonkeyPatch) -> None:
    unreadable = OSError(errno.EIO, "Input/output error", PATH_CANARY)
    monkeypatch.setattr(core.encode, "read_media_playlist", _raising(unreadable))


def _an_unlistable_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    _encode_os(monkeypatch, scandir=_raising(OSError(errno.EIO, "Input/output error", PATH_CANARY)))


def _a_file_that_cannot_be_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    _encode_os(monkeypatch, open=_raising(OSError(errno.EIO, "Input/output error", PATH_CANARY)))


def _a_file_that_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _encode_os(monkeypatch, read=_raising(OSError(errno.EIO, "Input/output error", PATH_CANARY)))


@pytest.mark.parametrize(
    "make_unreadable",
    [
        pytest.param(_an_unreadable_playlist, id="its-playlist-cannot-be-read"),
        pytest.param(_an_unlistable_directory, id="its-directory-cannot-be-listed"),
        pytest.param(_a_file_that_cannot_be_opened, id="a-file-cannot-be-opened"),
        pytest.param(_a_file_that_cannot_be_read, id="a-file-cannot-be-read"),
    ],
)
def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_an_unreadable_output_should_keep_the_path_out(
    make_unreadable: Callable[[pytest.MonkeyPatch], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: ffmpeg exited well; what it wrote then cannot be read, on an error whose own text names a path
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, _wrote_a_shaped_rendition)
    make_unreadable(monkeypatch)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert: nothing of the path, in the report as in the typed failure, which carries no other error
    written = reported_log_failures(capsys, "the rendition output could not be read")
    assert PATH_CANARY not in written
    _assert_failure(raised, ("internal_error", "output_unreadable", True))
    assert raised.value.failure.detail == "the rendition output could not be read: EIO"
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize("vanishing_under", ["scandir", "open"])
def test_encode_rendition_when_a_rendition_file_vanishes_while_verified_should_fail_typed_with_no_cause_and_no_context(
    vanishing_under: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a file is gone by the time the directory is listed, or by the time the file is opened
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, _wrote_a_shaped_rendition)
    gone = FileNotFoundError(errno.ENOENT, "No such file or directory", PATH_CANARY)
    _encode_os(monkeypatch, **{vanishing_under: _raising(gone)})

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert
    _assert_failure(raised, ("encoder_failed", "rendition_output_invalid", False))
    assert raised.value.failure.detail == "a rendition file vanished while it was verified"
    assert _carries_nothing_of_another_error(raised)


def _no_room_for_the_rendition_directory(monkeypatch: pytest.MonkeyPatch, job: _Job) -> None:
    full = OSError(errno.ENOSPC, "No space left on device", PATH_CANARY)
    _fail_the_making_of(monkeypatch, job.plan.name, full)


def _no_room_for_the_key(monkeypatch: pytest.MonkeyPatch, job: _Job) -> None:
    monkeypatch.setattr(core.encode, "hls_key_info", _raising(OSError(errno.ENOSPC, "No space left", PATH_CANARY)))


def _no_room_while_the_key_is_held(monkeypatch: pytest.MonkeyPatch, job: _Job) -> None:
    """An error of the operating system from inside the block that holds the key, before ffmpeg is started."""
    full = OSError(errno.ENOSPC, "No space left on device", PATH_CANARY)
    monkeypatch.setattr(core.encode, "rendition_argv", _raising(full))


@pytest.mark.parametrize(
    "fail_the_setup",
    [
        pytest.param(_no_room_for_the_rendition_directory, id="the-rendition-directory-cannot-be-made"),
        pytest.param(_no_room_for_the_key, id="the-key-cannot-be-held"),
        pytest.param(_no_room_while_the_key_is_held, id="an-os-error-while-the-key-is-held"),
    ],
)
def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_a_setup_that_failed_should_keep_the_path_out(
    fail_the_setup: Callable[[pytest.MonkeyPatch, _Job], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, _wrote_a_shaped_rendition)
    fail_the_setup(monkeypatch, job)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, "the encode could not be set up")
    assert PATH_CANARY not in written
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert raised.value.failure.detail == "the encode could not be set up: ENOSPC"
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def _no_room_for_the_self_decodes_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_the_making_of(monkeypatch, SELF_DECODE_CWD, OSError(errno.ENOSPC, "No space left on device", PATH_CANARY))


def _no_room_for_the_self_decodes_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the encode hold its key, and have the self-decode's own hold of it fail."""
    hold_the_key = core.encode.hls_key_info
    holds: list[Path] = []

    def hold_the_first_time(key: bytes, *, parent: Path) -> Any:
        holds.append(parent)
        if len(holds) > 1:
            raise OSError(errno.ENOSPC, "No space left on device", PATH_CANARY)
        return hold_the_key(key, parent=parent)

    monkeypatch.setattr(core.encode, "hls_key_info", hold_the_first_time)


@pytest.mark.parametrize(
    "fail_the_setup",
    [
        pytest.param(_no_room_for_the_self_decodes_directory, id="its-working-directory-cannot-be-made"),
        pytest.param(_no_room_for_the_self_decodes_key, id="its-key-cannot-be-held"),
    ],
)
def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_a_self_decode_not_set_up_should_keep_the_path_out(
    fail_the_setup: Callable[[pytest.MonkeyPatch], None],
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: an output that passes every structural check, and no room for the self-decode's own files
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    fail_the_setup(monkeypatch)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, "the encode could not be set up")
    assert PATH_CANARY not in written
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert raised.value.failure.detail == "the encode could not be set up: ENOSPC"
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


NOT_STARTED = [
    pytest.param(
        BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable", PATH_CANARY),
        "ffmpeg could not be started: %s errno=%s",
        ("resource_exhausted", "encoder_start_failed", True),
        "ffmpeg could not be started: EAGAIN",
        id="no-process-could-be-started",
    ),
    pytest.param(
        PermissionError(errno.EACCES, "Permission denied", PATH_CANARY),
        "ffmpeg could not be started: %s errno=%s",
        ("configuration_error", "ffmpeg_unavailable", False),
        "ffmpeg could not be started",
        id="the-program-may-not-be-run",
    ),
    pytest.param(
        RuntimeError(BUG_CANARY),
        "a reader thread did not start",
        ("resource_exhausted", "encoder_start_failed", True),
        "ffmpeg could not be started: unknown",
        id="a-reader-thread-did-not-start",
    ),
]


@pytest.mark.parametrize(("start_error", "line_words", "expected", "expected_detail"), NOT_STARTED)
def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_an_ffmpeg_not_started_should_keep_the_errors_text_out(
    start_error: Exception,
    line_words: str,
    expected: tuple[str, str, bool],
    expected_detail: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: ffmpeg cannot be started, on an error whose own text names a path or quotes what it met
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    monkeypatch.setattr(core.encode, "run_bounded", _raising(start_error))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, line_words)
    assert [canary for canary in (PATH_CANARY, BUG_CANARY) if canary in written] == []
    _assert_failure(raised, expected)
    assert raised.value.failure.detail == expected_detail
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


@pytest.mark.parametrize(("start_error", "line_words", "expected", "expected_detail"), NOT_STARTED)
def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_a_self_decode_not_started_should_keep_the_text_out(
    start_error: Exception,
    line_words: str,
    expected: tuple[str, str, bool],
    expected_detail: str,
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: an output that passes every structural check, and a self-decode that cannot be started
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    _self_decode_stand_in(monkeypatch, start_error)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, line_words)
    assert [canary for canary in (PATH_CANARY, BUG_CANARY) if canary in written] == []
    _assert_failure(raised, expected)
    assert raised.value.failure.detail == expected_detail
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_a_log_handler_fails_on_the_line_of_a_rendition_not_removed_should_keep_its_path_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failing_log: None
) -> None:
    # Arrange: ffmpeg failed, and then the rendition's directory cannot be removed
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, lambda rendition_dir, monitor: EXITED_WITH_AN_ERROR)
    _refuse_the_removal_of(monkeypatch, job.rendition_dir)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert: the failure is ffmpeg's, as without the failing log, and the directory is where it was left
    written = reported_log_failures(capsys, "a failed rendition's directory could not be removed")
    assert REMOVAL_CANARY not in written
    _assert_failure(raised, ("encoder_failed", "encode_failed", False))
    assert _carries_nothing_of_another_error(raised)
    assert job.rendition_dir.is_dir()


def test_encode_rendition_when_a_log_handler_fails_while_a_rendition_is_removed_behind_a_bug_should_keep_its_text_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failing_log: None
) -> None:
    # Arrange: the encode ends on a bug, in words that quote what it choked on, and then the rendition's
    # directory cannot be removed
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    bug = ValueError(BUG_CANARY)
    monkeypatch.setattr(core.encode, "run_bounded", _raising(bug))
    _refuse_the_removal_of(monkeypatch, job.rendition_dir)

    # Act
    with pytest.raises(ValueError) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert: nothing of the bug, which is the error that is raised, as it is, nor of the removal's error
    written = reported_log_failures(capsys, "a failed rendition's directory could not be removed")
    assert [canary for canary in (BUG_CANARY, REMOVAL_CANARY) if canary in written] == []
    assert raised.value is bug


def test_encode_rendition_when_a_log_handler_fails_while_the_key_is_released_behind_a_bug_should_keep_its_text_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failing_log: None
) -> None:
    # Arrange: the encode ends on a bug while it holds the key, and then the key's files cannot be removed
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    bug = ValueError(BUG_CANARY)
    monkeypatch.setattr(core.encode, "run_bounded", _raising(bug))
    _refuse_the_removal_of_key_files(monkeypatch)

    # Act
    with pytest.raises(ValueError) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert: the key's lines hold nothing of the bug the key was released behind, nor of their own error
    written = reported_log_failures(capsys, "an HLS key file could not be removed")
    assert [canary for canary in (BUG_CANARY, REMOVAL_CANARY) if canary in written] == []
    assert raised.value is bug
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_a_log_handler_fails_while_the_self_decodes_key_is_released_should_keep_the_path_out(
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: an encrypted output that passes every structural check; the self-decode's own files
    # cannot be made, on an error whose text names a path, and then its key's files cannot be removed
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    _fail_the_making_of(monkeypatch, SELF_DECODE_CWD, OSError(errno.ENOSPC, "No space left on device", PATH_CANARY))
    _refuse_the_removal_of_key_files(monkeypatch)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, "an HLS key file could not be removed")
    assert [canary for canary in (PATH_CANARY, REMOVAL_CANARY) if canary in written] == []
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []


def test_encode_rendition_when_a_log_handler_fails_while_a_self_decode_ends_on_a_bug_should_keep_the_bugs_text_out(
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: an encrypted output that passes every structural check; the self-decode ends on a bug
    # while it holds the key, and then the key's files cannot be removed
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    bug = ValueError(BUG_CANARY)
    _self_decode_stand_in(monkeypatch, bug)
    _refuse_the_removal_of_key_files(monkeypatch)

    # Act
    with pytest.raises(ValueError) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert: the key's lines hold nothing of the bug the key was released behind, nor of their own error
    written = reported_log_failures(capsys, "an HLS key file could not be removed")
    assert [canary for canary in (BUG_CANARY, REMOVAL_CANARY) if canary in written] == []
    assert raised.value is bug
    assert list(job.output_root.iterdir()) == []


def _fail_the_release_of_the_key(monkeypatch: pytest.MonkeyPatch, *, hold: int) -> None:
    """Have the release of the key fail for the hold numbered `hold` (1 is the encode's, 2 the
    self-decode's), on an error of the operating system whose own text names a path. The key is held
    and released as ever; the error comes once that is done."""
    hold_the_key = core.encode.hls_key_info
    holds: list[Path] = []

    @contextlib.contextmanager
    def hold_and_fail_the_release(key: bytes, *, parent: Path) -> Iterator[Any]:
        holds.append(parent)
        release_fails = len(holds) == hold
        try:
            with hold_the_key(key, parent=parent) as key_info:
                yield key_info
        finally:
            if release_fails:
                raise OSError(errno.EBADF, "Bad file descriptor", PATH_CANARY)

    monkeypatch.setattr(core.encode, "hls_key_info", hold_and_fail_the_release)


def _ended_on_a_bug(rendition_dir: Path, monitor: Any) -> BoundedResult:
    raise ValueError(BUG_CANARY)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(_ended_on_a_bug, id="behind-a-bug"),
        pytest.param(_wrote_a_shaped_rendition, id="behind-a-run-that-ended-well"),
    ],
)
def test_encode_rendition_when_the_keys_release_fails_under_a_failing_log_should_keep_every_text_out(
    run: Callable[[Path, Any], BoundedResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: the release of the key fails on an error of the operating system, which stands whatever
    # the block that held the key ended on
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, run)
    _fail_the_release_of_the_key(monkeypatch, hold=1)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert: the release's error ends the rendition, typed, and neither its text nor the bug's is told
    written = reported_log_failures(capsys, "the encode could not be set up")
    assert [canary for canary in (PATH_CANARY, BUG_CANARY) if canary in written] == []
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert raised.value.failure.detail == "the encode could not be set up: EBADF"
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []


def _a_self_decode_that_ends_on_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    _self_decode_stand_in(monkeypatch, ValueError(BUG_CANARY))


def _a_self_decode_that_ends_well(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to replace: the stand-in encoder hands the self-decode to the real ffmpeg."""


@pytest.mark.parametrize(
    "end_the_self_decode",
    [
        pytest.param(_a_self_decode_that_ends_on_a_bug, id="behind-a-bug"),
        pytest.param(_a_self_decode_that_ends_well, id="behind-a-run-that-ended-well"),
    ],
)
def test_encode_rendition_when_the_self_decodes_key_release_fails_under_a_failing_log_should_keep_every_text_out(
    end_the_self_decode: Callable[[pytest.MonkeyPatch], None],
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: an encrypted output that passes every structural check; the release of the key the
    # self-decode held fails on an error of the operating system, whatever the decode ended on
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    end_the_self_decode(monkeypatch)
    _fail_the_release_of_the_key(monkeypatch, hold=2)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    written = reported_log_failures(capsys, "the encode could not be set up")
    assert [canary for canary in (PATH_CANARY, BUG_CANARY) if canary in written] == []
    _assert_failure(raised, ("resource_exhausted", "encode_setup_failed", True))
    assert raised.value.failure.detail == "the encode could not be set up: EBADF"
    assert _carries_nothing_of_another_error(raised)
    assert list(job.output_root.iterdir()) == []


def _wrote_two_segments(rendition_dir: Path, monitor: Any) -> BoundedResult:
    (rendition_dir / "playlist.m3u8").write_text(TWO_SEGMENT_PLAYLIST, encoding="ascii")
    for name in ("segment_0000.ts", "segment_0001.ts"):
        (rendition_dir / name).write_bytes(ONE_TS_PACKET)
    return EXITED_CLEANLY


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        pytest.param(_wrote_nothing, ("encoder_failed", "rendition_output_invalid", False), id="no-playlist"),
        pytest.param(
            _wrote_two_segments, ("input_limits_exceeded", "too_many_segments", False), id="too-many-segments"
        ),
    ],
)
def test_encode_rendition_when_the_playlist_is_refused_should_fail_typed_with_no_cause_and_no_context(
    run: Callable[[Path, Any], BoundedResult],
    expected: tuple[str, str, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4", profile=fast_profile(max_segments_per_rendition=1))
    _ffmpeg_stand_in(monkeypatch, run)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert: the typed failure stands in the playlist reader's refusal's place and carries nothing of it
    _assert_failure(raised, expected)
    assert _carries_nothing_of_another_error(raised)


def test_encode_rendition_when_a_full_disk_explains_a_refused_output_should_fail_with_no_cause_and_no_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: room after the run, none once the output has been refused
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    _ffmpeg_stand_in(monkeypatch, _wrote_nothing)
    readings = iter([PLENTY_OF_ROOM])
    monkeypatch.setattr(os, "statvfs", lambda path: next(readings, NO_ROOM))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "scratch_full", True))
    assert _carries_nothing_of_another_error(raised)


def test_decoded_frames_when_the_progress_is_not_ascii_should_fail_typed_with_no_cause_and_no_context() -> None:
    # Arrange
    progress = b"frame=\xff\nprogress=end\n"

    # Act
    with pytest.raises(WorkerFailure) as raised:
        core.encode._decoded_frames(progress)

    # Assert
    _assert_failure(raised, ("internal_error", "self_check_output_invalid", False))
    assert _carries_nothing_of_another_error(raised)


INTERRUPTS = [pytest.param(KeyboardInterrupt(), id="a-keyboard-interrupt"), pytest.param(SystemExit(3), id="an-exit")]


@pytest.mark.parametrize("interrupt", INTERRUPTS)
def test_encode_rendition_when_ffmpeg_is_interrupted_should_leave_nothing_and_raise_the_interrupt_as_it_is(
    interrupt: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the interrupt arrives while ffmpeg runs, with a segment written and the key held
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")

    def interrupted(rendition_dir: Path, monitor: Any) -> BoundedResult:
        (rendition_dir / "segment_0000.ts").write_bytes(ONE_TS_PACKET)
        raise interrupt

    _ffmpeg_stand_in(monkeypatch, interrupted)

    # Act
    with pytest.raises(type(interrupt)) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert: the partial rendition, the key and the working directory are gone
    assert raised.value is interrupt
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


@pytest.mark.parametrize("interrupt", INTERRUPTS)
def test_encode_rendition_when_the_self_decode_is_interrupted_should_leave_nothing_and_raise_the_interrupt_as_it_is(
    interrupt: BaseException, encode_media: EncodeMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: an encrypted output that passes every structural check; the interrupt arrives while it is decoded
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    _self_decode_stand_in(monkeypatch, interrupt)

    # Act
    with pytest.raises(type(interrupt)) as raised:
        job.run(ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert: the rendition, the check playlist and the key are gone
    assert raised.value is interrupt
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def test_encode_rendition_when_ffmpeg_cannot_be_started_should_log_that_before_the_key_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange: ffmpeg cannot be started, and then the key's files cannot be removed, which is logged too
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    monkeypatch.setattr(core.encode, "run_bounded", _raising(BlockingIOError(errno.EAGAIN, "No more processes")))
    _refuse_the_removal_of_key_files(monkeypatch)

    # Act
    with caplog.at_level(logging.WARNING), pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert: the line of the start that failed comes first, the lines of the key's release after it
    logged = [record.getMessage() for record in caplog.records if record.name in ("core.encode", "core.hls_key")]
    _assert_failure(raised, ("resource_exhausted", "encoder_start_failed", True))
    assert logged == [
        "ffmpeg could not be started: BlockingIOError errno=EAGAIN",
        "an HLS key file could not be removed: errno=EACCES",
        "an HLS key file could not be removed: errno=EACCES",
        "an HLS key directory could not be removed: errno=ENOTEMPTY",
    ]


# --- what an error keeps alive ------------------------------------------------------------------------
# The frames of the encode hold the media key. An error keeps the frames on its stack alive, so a frame
# that still held the error it raised would live on, with the key, after everyone had let go of the
# error: until a pass of the cyclic collector, which these tests switch off.


class _ABug(Exception):
    """A bug of a class of its own: a weak reference can be taken of it, as of a typed failure."""


def _no_process() -> BaseException:
    return BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")


def _raising_a_fresh(make_error: Callable[[], BaseException]) -> Callable[..., Any]:
    """A stand-in for a function that fails on an error made for that call, which nothing else holds."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise make_error()

    return fail


@contextlib.contextmanager
def _cyclic_collector_off() -> Iterator[None]:
    was_on = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_on:
            gc.enable()


def _freed_once_let_go(job: _Job, raised_class: type[BaseException], **run_arguments: Any) -> bool:
    """Run the encode to the error of `raised_class` it raises, let go of that error, and tell whether
    it was freed at once, with no pass of the cyclic collector."""
    with _cyclic_collector_off():
        try:
            job.run(**run_arguments)
        except raised_class as error:
            caught = weakref.ref(error)
        return caught() is None


LIFETIMES = [
    pytest.param(_no_process, WorkerFailure, id="ffmpeg-could-not-be-started"),
    pytest.param(_ABug, _ABug, id="a-bug-in-the-block"),
]


@pytest.mark.parametrize(("make_error", "raised_class"), LIFETIMES)
def test_encode_rendition_when_what_it_raised_is_let_go_of_should_keep_no_frame_that_holds_the_key_alive(
    make_error: Callable[[], BaseException],
    raised_class: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    monkeypatch.setattr(core.encode, "run_bounded", _raising_a_fresh(make_error))

    # Act
    freed = _freed_once_let_go(job, raised_class, ffmpeg_path=sys.executable, media_key=MEDIA_KEY)

    # Assert
    assert freed


@pytest.mark.parametrize(("make_error", "raised_class"), LIFETIMES)
def test_encode_rendition_when_what_the_self_decode_raised_is_let_go_of_should_keep_no_frame_that_holds_the_key_alive(
    make_error: Callable[[], BaseException],
    raised_class: type[BaseException],
    encode_media: EncodeMedia,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: an encrypted output that passes every structural check, so that the self-decode is reached
    job = _claimed_job(tmp_path, tmp_path / "source.mp4")
    executable = _copying_fake(tmp_path, *_encrypted_output(encode_media, drop_last_block=False))
    _self_decode_raising(monkeypatch, make_error)

    # Act
    freed = _freed_once_let_go(job, raised_class, ffmpeg_path=executable, media_key=MEDIA_KEY)

    # Assert
    assert freed
