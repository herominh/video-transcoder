from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import core.thumbnail
from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.encode_command import EncodeSource, input_arguments
from core.failure import Diagnostic, WorkerFailure
from core.profile import PILOT_PROFILE, RENDITION_NAMES, MediaProfile
from core.renditions import RenditionPlan, plan_renditions
from core.source_facts import VideoStreamFacts
from core.thumbnail import ThumbnailResult, jpeg_frame_size, make_thumbnail, thumbnail_geometry
from tests.encode.conftest import (
    ICC_MARKER,
    LOCATION_MARKER,
    QUADRANT_COLOURS,
    TITLE_MARKER,
    EncodeMedia,
    fast_profile,
    probe_facts,
    require_media_tools,
)
from tests.preflight.conftest import FFMPEG_TIMEOUT_S

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the thumbnail runs under POSIX process groups")

GENEROUS_DEADLINE_MS = 60_000
SHORT_CALLER_DEADLINE_MS = 300
RETURN_WITHIN_S = 5.0
FAST_PROFILE = fast_profile()
# Below the 240-line fixtures, so the thumbnail is scaled down.
SMALL_EDGE_PROFILE = fast_profile(thumbnail_short_edge=120)
TINY_BYTES = 1000
TINY_BYTES_PROFILE = fast_profile(thumbnail_max_bytes=TINY_BYTES)
TINY_WALL_MS = 200
TINY_WALL_PROFILE = fast_profile(thumbnail_wall_ms=TINY_WALL_MS)
# Mean RGB of a quadrant after lossy coding stays this close to the reference decode.
QUADRANT_TOLERANCE = 40
# The reference quadrants are this far apart in at least one channel, so a swap cannot pass.
QUADRANT_SEPARATION = 100
CLAIMED_DURATION_MS = 2000
CLAIMED_SIZE = (320, 240)
WRONG_SIZE = "160x120"
MAX_PIXEL_TEXT = "Picture size 320x240 exceeds specified max pixel count 76799"
NO_ROOM = SimpleNamespace(f_bavail=0, f_frsize=4096)
NOT_THE_PLANNED_JPEG = "the thumbnail is not a JPEG of the planned size"
FAKE_ARGUMENTS = "for last; do :; done\n"
RED = (255.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 255.0)
# Mean RGB of a solid picture after lossy coding stays this close to its colour.
COLOUR_TOLERANCE = 40
# What mjpeg writes with +bitexact, on ffmpeg 7.1 and 8 alike: JFIF, tables, frame, scan.
CLEAN_MARKERS = [0xE0, 0xDB, 0xC4, 0xC0, 0xDA]
JFIF_IDENTIFIER = b"JFIF\x00"
CARRIES_METADATA = "the thumbnail carries metadata"
# A profile whose decoder limit (10,000 + 63 x 200 + 63 x 63 = 26,569) is below 320x240.
SMALL_PIXEL_CAP_EDGE = 100
THUMBNAIL_UNAVAILABLE = "thumbnail_unavailable"


@dataclass(frozen=True, slots=True)
class _Job:
    source: EncodeSource
    reference: RenditionPlan
    profile: MediaProfile
    output_root: Path
    work_dir: Path

    @property
    def thumbnail(self) -> Path:
        return self.output_root / "thumbnail.jpg"

    def run(self, *, deadline: Deadline | None = None, ffmpeg_path: str | None = None) -> ThumbnailResult:
        return make_thumbnail(
            self.source,
            self.reference,
            profile=self.profile,
            output_root=self.output_root,
            work_dir=self.work_dir,
            deadline=deadline or Deadline.after_ms(GENEROUS_DEADLINE_MS),
            ffmpeg_path=ffmpeg_path,
        )


def _directories(root: Path) -> tuple[Path, Path]:
    output_root = root / "output"
    work_dir = root / "work"
    output_root.mkdir()
    work_dir.mkdir()
    return output_root, work_dir


def _job(root: Path, media_path: Path, *, profile: MediaProfile = FAST_PROFILE) -> _Job:
    """The thumbnail of a real source, sized after its largest planned rendition."""
    facts = probe_facts(media_path)
    output_root, work_dir = _directories(root)
    reference = plan_renditions(
        facts.video_streams[0], has_audio=bool(facts.audio_streams), requested=list(RENDITION_NAMES), profile=profile
    )[0]
    return _Job(
        source=EncodeSource.from_facts(media_path, facts, profile),
        reference=reference,
        profile=profile,
        output_root=output_root,
        work_dir=work_dir,
    )


def _claimed_video() -> VideoStreamFacts:
    """A 320x240 video stream as a source might claim it, without any media behind the claim."""
    return VideoStreamFacts(
        index=0,
        codec_name="h264",
        width=CLAIMED_SIZE[0],
        height=CLAIMED_SIZE[1],
        sample_aspect_num=1,
        sample_aspect_den=1,
        rotation_degrees=0,
        display_width=CLAIMED_SIZE[0],
        display_height=CLAIMED_SIZE[1],
        frame_rate_num=25,
        frame_rate_den=1,
        field_order="progressive",
        color_transfer=None,
        pix_fmt="yuv420p",
        dolby_vision_profile=None,
    )


def _claimed_job(root: Path, *, profile: MediaProfile = FAST_PROFILE, source_path: Path | None = None) -> _Job:
    source = EncodeSource(
        path=source_path or root / "source.mp4",
        demuxer="mov",
        video_index=0,
        audio_index=None,
        rotation_degrees=0,
        duration_ms=CLAIMED_DURATION_MS,
    )
    reference = plan_renditions(_claimed_video(), has_audio=False, requested=list(RENDITION_NAMES), profile=profile)[0]
    output_root, work_dir = _directories(root)
    return _Job(source=source, reference=reference, profile=profile, output_root=output_root, work_dir=work_dir)


def _fake_ffmpeg(root: Path, body: str) -> str:
    """A stand-in executable; its last argument, `$last`, is the thumbnail's path."""
    directory = root / "bin"
    directory.mkdir(exist_ok=True)
    script = directory / "ffmpeg"
    script.write_text("#!/bin/sh\n" + FAKE_ARGUMENTS + body, encoding="ascii")
    script.chmod(0o755)
    return str(script)


def _run_tool(*argv: str) -> bytes:
    return subprocess.run(list(argv), check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S).stdout


def _probe_image(path: Path) -> dict[str, Any]:
    output = _run_tool("ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path))
    return json.loads(output)["streams"][0]


@dataclass(frozen=True, slots=True)
class _Jpegs:
    planned: Path  # a real JPEG of the claimed job's planned 320x240
    other_size: Path  # a real JPEG of 160x120


def _real_jpeg(target: Path, size: str) -> Path:
    _run_tool(
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25", "-frames:v", "1",
        "-c:v", "mjpeg", "-flags:v", "+bitexact", "-pix_fmt", "yuvj420p", str(target),
    )
    return target


@pytest.fixture(scope="module")
def jpegs(tmp_path_factory: pytest.TempPathFactory) -> _Jpegs:
    require_media_tools()
    root = tmp_path_factory.mktemp("jpegs")
    return _Jpegs(
        planned=_real_jpeg(root / "planned.jpg", f"{CLAIMED_SIZE[0]}x{CLAIMED_SIZE[1]}"),
        other_size=_real_jpeg(root / "other_size.jpg", WRONG_SIZE),
    )


def _assert_unavailable(result: ThumbnailResult, detail: str, job: _Job) -> None:
    assert result.artifact is None
    assert result.diagnostic == Diagnostic(code=THUMBNAIL_UNAVAILABLE, detail=detail)
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


def _assert_failure(raised: pytest.ExceptionInfo[WorkerFailure], expected: tuple[str, str, bool]) -> None:
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == expected


# --- with the real ffmpeg --------------------------------------------------------------------------


def test_make_thumbnail_when_the_source_is_larger_than_the_thumbnail_should_write_a_jpeg_of_the_planned_size(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: a 320x240 source against a 120-line thumbnail
    job = _job(tmp_path, encode_media.landscape, profile=SMALL_EDGE_PROFILE)

    # Act
    result = job.run()

    # Assert
    data = job.thumbnail.read_bytes()
    stream = _probe_image(job.thumbnail)
    assert result.diagnostic is None
    assert result.artifact == ArtifactFile(
        path="thumbnail.jpg", kind="thumbnail", size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    assert (stream["codec_name"], stream["pix_fmt"]) == ("mjpeg", "yuvj420p")
    assert (stream["width"], stream["height"]) == (160, 120)
    assert jpeg_frame_size(data) == (160, 120)
    assert os.listdir(job.output_root) == ["thumbnail.jpg"]
    assert list(job.work_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("fixture_name", "profile", "expected_size"),
    [
        ("portrait", SMALL_EDGE_PROFILE, (120, 160)),
        ("landscape", PILOT_PROFILE, (320, 240)),
        ("portrait", PILOT_PROFILE, (240, 320)),
        ("subsecond", PILOT_PROFILE, (160, 120)),
        ("silent", SMALL_EDGE_PROFILE, (160, 120)),
    ],
    ids=[
        "portrait scaled down",
        "landscape smaller than the thumbnail edge",
        "portrait smaller than the thumbnail edge",
        "shorter than a second",
        "without audio",
    ],
)
def test_make_thumbnail_when_the_source_has_this_shape_should_keep_its_aspect_and_never_upscale(
    fixture_name: str, profile: MediaProfile, expected_size: tuple[int, int], encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, getattr(encode_media, fixture_name), profile=profile)

    # Act
    result = job.run()

    # Assert
    stream = _probe_image(job.thumbnail)
    assert result.artifact is not None
    assert (stream["width"], stream["height"]) == expected_size
    assert jpeg_frame_size(job.thumbnail.read_bytes()) == expected_size


def _rgb(path: Path, width: int, height: int, *, first_frame: bool) -> bytes:
    """The (first) frame as ffmpeg shows it (autorotation on), scaled to width x height, as RGB24."""
    frames = ["-frames:v", "1"] if first_frame else []
    return _run_tool(
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), *frames,
        "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
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
def test_make_thumbnail_when_the_display_is_rotated_should_show_the_frame_as_ffmpeg_displays_it(
    degrees: int, encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    source = encode_media.rotated[degrees]
    job = _job(tmp_path, source, profile=SMALL_EDGE_PROFILE)
    width, height = thumbnail_geometry(job.reference, job.profile)
    reference = _quadrant_means(_rgb(source, width, height, first_frame=True), width, height)

    # Act
    result = job.run()

    # Assert
    stream = _probe_image(job.thumbnail)
    shown = _quadrant_means(_rgb(job.thumbnail, width, height, first_frame=False), width, height)
    assert result.artifact is not None
    expected_size = (120, 160) if degrees in (90, 270) else (160, 120)
    assert (width, height) == (stream["width"], stream["height"]) == expected_size
    assert all(
        not _close(reference[i], reference[j], QUADRANT_SEPARATION)
        for i in range(len(QUADRANT_COLOURS))
        for j in range(i + 1, len(QUADRANT_COLOURS))
    )
    assert all(_close(got, want, QUADRANT_TOLERANCE) for got, want in zip(shown, reference, strict=True))


def test_make_thumbnail_when_the_source_carries_title_and_location_tags_should_leave_them_out_of_the_jpeg(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    job = _job(tmp_path, encode_media.tagged)

    # Act
    result = job.run()

    # Assert
    data = job.thumbnail.read_bytes()
    assert result.artifact is not None
    assert TITLE_MARKER.encode() not in data
    assert LOCATION_MARKER.encode() not in data


# --- with stand-in encoders ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "profile", "expected_detail"),
    [
        ("exit 1\n", FAST_PROFILE, "the thumbnail encoder exited with status 1"),
        ("exit 0\n", FAST_PROFILE, "the thumbnail encoder wrote no image"),
        (': > "$last"\nexit 0\n', FAST_PROFILE, "the thumbnail encoder wrote no image"),
        ('printf "not a jpeg" > "$last"\nexit 0\n', FAST_PROFILE, NOT_THE_PLANNED_JPEG),
        ('cp "{other_size}" "$last"\nexit 0\n', FAST_PROFILE, NOT_THE_PLANNED_JPEG),
        ('head -c 1000 "{planned}" > "$last"\nexit 0\n', FAST_PROFILE, NOT_THE_PLANNED_JPEG),
        ('ln -s "{planned}" "$last"\nexit 0\n', FAST_PROFILE, "the thumbnail is not a regular file"),
        ('mkdir "$last"\nexit 0\n', FAST_PROFILE, "the thumbnail is not a regular file"),
        (
            f'head -c {2 * TINY_BYTES} /dev/zero > "$last"\nexec sleep 30\n',
            TINY_BYTES_PROFILE,
            f"the thumbnail exceeds {TINY_BYTES} bytes",
        ),
        (
            f'head -c {2 * TINY_BYTES} /dev/zero > "$last"\nexit 0\n',
            TINY_BYTES_PROFILE,
            f"the thumbnail exceeds {TINY_BYTES} bytes",
        ),
        ("exec sleep 30\n", TINY_WALL_PROFILE, f"the thumbnail exceeded its time budget of {TINY_WALL_MS} ms"),
        (
            f"echo '{MAX_PIXEL_TEXT}' >&2\n" + 'cp "{planned}" "$last"\nexit 0\n',
            FAST_PROFILE,
            "a decoded frame exceeds the profile's pixel limit",
        ),
        ("kill -9 $$\n", FAST_PROFILE, "the thumbnail encoder ended on a signal"),
        ("exit 127\n", FAST_PROFILE, "the thumbnail encoder could not be started"),
        ("head -c 70000 /dev/zero\nexit 0\n", FAST_PROFILE, "the thumbnail encoder wrote more than expected to stdout"),
    ],
    ids=[
        "exits 1",
        "exits 0 without writing",
        "writes an empty file",
        "writes garbage",
        "writes a JPEG of another size",
        "writes a JPEG cut short",
        "leaves a symbolic link",
        "leaves a directory",
        "keeps writing past the byte limit",
        "exits after writing past the byte limit",
        "outlasts its time budget",
        "refuses a frame above the pixel cap",
        "is killed by a signal",
        "is not found by the shell",
        "writes to stdout",
    ],
)
def test_make_thumbnail_when_ffmpeg_ends_this_way_should_report_the_thumbnail_unavailable_and_leave_nothing(
    body: str, profile: MediaProfile, expected_detail: str, jpegs: _Jpegs, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path, profile=profile)
    executable = _fake_ffmpeg(tmp_path, body.replace("{planned}", str(jpegs.planned)).replace(
        "{other_size}", str(jpegs.other_size)
    ))
    started = time.monotonic()

    # Act
    result = job.run(ffmpeg_path=executable)
    elapsed = time.monotonic() - started

    # Assert
    _assert_unavailable(result, expected_detail, job)
    assert elapsed < RETURN_WITHIN_S


def test_make_thumbnail_when_ffmpeg_writes_exactly_a_jpeg_of_the_planned_size_should_accept_it(
    jpegs: _Jpegs, tmp_path: Path
) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, f'cp "{jpegs.planned}" "$last"\nexit 0\n')
    data = jpegs.planned.read_bytes()

    # Act
    result = job.run(ffmpeg_path=executable)

    # Assert
    assert thumbnail_geometry(job.reference, job.profile) == CLAIMED_SIZE
    assert result == ThumbnailResult(
        artifact=ArtifactFile(
            path="thumbnail.jpg", kind="thumbnail", size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        ),
        diagnostic=None,
    )
    assert job.thumbnail.read_bytes() == data


def test_make_thumbnail_when_the_executable_is_missing_should_report_the_thumbnail_unavailable(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path)

    # Act
    result = job.run(ffmpeg_path=str(tmp_path / "no-such-ffmpeg"))

    # Assert
    _assert_unavailable(result, "the thumbnail encoder could not be started", job)


def test_make_thumbnail_when_unavailable_should_log_one_warning_with_the_detail_and_no_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, "exit 1\n")

    # Act
    with caplog.at_level(logging.WARNING, logger="core.thumbnail"):
        result = job.run(ffmpeg_path=executable)

    # Assert
    assert result.diagnostic is not None
    messages = [record.getMessage() for record in caplog.records if record.name == "core.thumbnail"]
    assert len(messages) == 1
    assert result.diagnostic.detail in messages[0]
    assert str(tmp_path) not in messages[0]


def test_make_thumbnail_when_the_callers_deadline_has_passed_should_fail_before_anything_runs(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    marker = tmp_path / "ran"
    executable = _fake_ffmpeg(tmp_path, f'touch "{marker}"\nexit 0\n')

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=Deadline(expires_at=time.monotonic() - 1))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_deadline_exceeded", False))
    assert not marker.exists()
    assert list(job.output_root.iterdir()) == []


def test_make_thumbnail_when_the_callers_deadline_passes_while_ffmpeg_runs_should_fail_and_leave_nothing(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, 'printf x > "$last"\nexec sleep 30\n')
    started = time.monotonic()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=Deadline.after_ms(SHORT_CALLER_DEADLINE_MS))
    elapsed = time.monotonic() - started

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_deadline_exceeded", False))
    assert elapsed < RETURN_WITHIN_S
    assert list(job.output_root.iterdir()) == []
    assert list(job.work_dir.iterdir()) == []


class _SwitchedDeadline(Deadline):
    """A caller's deadline that holds until its switch is set; no clock decides it."""

    def __init__(self, switch: threading.Event) -> None:
        super().__init__(expires_at=time.monotonic() + GENEROUS_DEADLINE_MS / 1000)
        # Deadline is frozen; this subclass's own attribute lives in its instance dictionary.
        object.__setattr__(self, "_switch", switch)

    def expired(self) -> bool:
        return self._switch.is_set() or super().expired()


def test_make_thumbnail_when_the_callers_deadline_passes_once_ffmpeg_has_exited_should_fail_and_remove_the_jpeg(
    jpegs: _Jpegs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a valid thumbnail, written just as the caller runs out of time
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, f'cp "{jpegs.planned}" "$last"\nexit 0\n')
    switch = threading.Event()
    real_run_bounded = core.thumbnail.run_bounded

    def run_then_expire(*args: Any, **kwargs: Any) -> Any:
        result = real_run_bounded(*args, **kwargs)
        switch.set()
        return result

    monkeypatch.setattr(core.thumbnail, "run_bounded", run_then_expire)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable, deadline=_SwitchedDeadline(switch))

    # Assert
    _assert_failure(raised, ("deadline_exceeded", "encode_deadline_exceeded", False))
    assert list(job.output_root.iterdir()) == []


@pytest.mark.parametrize(
    "body",
    ['cp "{planned}" "$last"\nexit 0\n', "exit 1\n"],
    ids=["after a clean exit", "after a failed run"],
)
def test_make_thumbnail_when_the_scratch_disk_is_full_should_fail_retryably_and_leave_nothing(
    body: str, jpegs: _Jpegs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, body.replace("{planned}", str(jpegs.planned)))
    monkeypatch.setattr(os, "statvfs", lambda path: NO_ROOM)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        job.run(ffmpeg_path=executable)

    # Assert
    _assert_failure(raised, ("resource_exhausted", "scratch_full", True))
    assert list(job.output_root.iterdir()) == []


def test_make_thumbnail_when_the_thumbnail_already_exists_should_raise_and_leave_it_alone(tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    job.thumbnail.write_bytes(b"kept")
    marker = tmp_path / "ran"
    executable = _fake_ffmpeg(tmp_path, f'touch "{marker}"\nexit 0\n')

    # Act
    with pytest.raises(ValueError):
        job.run(ffmpeg_path=executable)

    # Assert
    assert job.thumbnail.read_bytes() == b"kept"
    assert not marker.exists()


@pytest.mark.parametrize("layout", ["output inside work", "work inside output", "same directory"])
def test_make_thumbnail_when_output_and_work_directories_overlap_should_raise(layout: str, tmp_path: Path) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
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


@pytest.mark.parametrize("field", ["source", "reference", "profile"])
def test_make_thumbnail_when_an_argument_has_the_wrong_type_should_raise_type_error(field: str, tmp_path: Path) -> None:
    # Arrange
    job = dataclasses.replace(_claimed_job(tmp_path), **{field: "not the right type"})

    # Act / Assert
    with pytest.raises(TypeError):
        job.run()


# --- the pure parts --------------------------------------------------------------------------------


def _reference(width: int, height: int) -> RenditionPlan:
    plan = plan_renditions(_claimed_video(), has_audio=False, requested=["240p"], profile=PILOT_PROFILE)[0]
    return dataclasses.replace(plan, width=width, height=height)


@pytest.mark.parametrize(
    ("reference_size", "expected"),
    [
        ((1920, 1080), (640, 360)),
        ((1080, 1920), (360, 640)),
        ((3840, 2160), (640, 360)),
        ((854, 480), (640, 360)),
        ((320, 240), (320, 240)),
        ((640, 360), (640, 360)),
        ((360, 640), (360, 640)),
        ((3840, 362), (3818, 360)),
    ],
    ids=["1080p", "portrait 1080p", "2160p", "480p", "smaller", "exactly the edge", "portrait at the edge", "wide"],
)
def test_thumbnail_geometry_when_given_the_largest_rendition_should_scale_its_short_edge_down_to_the_profiles(
    reference_size: tuple[int, int], expected: tuple[int, int]
) -> None:
    # Arrange
    reference = _reference(*reference_size)

    # Act
    geometry = thumbnail_geometry(reference, PILOT_PROFILE)

    # Assert
    assert geometry == expected


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _frame_header(marker: int = 0xC0, *, width: int = 320, height: int = 240) -> bytes:
    components = bytes([3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1])
    return _segment(marker, bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + components)


APP0 = _segment(0xE0, b"JFIF\x00\x01\x02\x00\x00\x01\x00\x01\x00\x00")
SCAN_HEADER = _segment(0xDA, bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3F, 0]))
SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
ENTROPY_CODED = b"\x12\x34\xff\x00\x56"


def _jpeg(*segments: bytes) -> bytes:
    return SOI + b"".join(segments) + ENTROPY_CODED + EOI


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_jpeg(APP0, _frame_header(), SCAN_HEADER), (320, 240)),
        (_jpeg(APP0, _frame_header(0xC2, width=640, height=360), SCAN_HEADER), (640, 360)),
        (SOI + b"\xff" + APP0 + _frame_header() + SCAN_HEADER + ENTROPY_CODED + EOI, (320, 240)),
        (_jpeg(APP0, _frame_header(0xC3), SCAN_HEADER), None),
        (_jpeg(APP0, _frame_header(0xC9), SCAN_HEADER), None),
        (_jpeg(APP0, SCAN_HEADER), None),
        (_jpeg(_frame_header(), _frame_header(), SCAN_HEADER), None),
        (_jpeg(_frame_header(height=0), SCAN_HEADER), None),
        (_jpeg(APP0, _frame_header(), SCAN_HEADER)[:-2], None),
        (SOI + b"\xff\xe0\x10\x00" + EOI, None),
        (SOI + b"\x00" + APP0 + _frame_header() + SCAN_HEADER + EOI, None),
        (_jpeg(APP0, _frame_header()), None),
        (SOI + EOI, None),
        (b"", None),
    ],
    ids=[
        "baseline",
        "progressive",
        "a fill byte before a marker",
        "lossless",
        "arithmetic-coded",
        "no frame header",
        "two frame headers",
        "a height defined later",
        "no EOI",
        "a segment longer than the data",
        "no marker after SOI",
        "no scan",
        "SOI then EOI",
        "empty",
    ],
)
def test_jpeg_frame_size_when_walking_the_markers_should_find_exactly_one_readable_frame_header(
    data: bytes, expected: tuple[int, int] | None
) -> None:
    # Arrange / Act
    size = jpeg_frame_size(data)

    # Assert
    assert size == expected


def test_jpeg_frame_size_when_reading_ffmpegs_own_jpeg_should_find_its_size(jpegs: _Jpegs) -> None:
    # Arrange
    data = jpegs.other_size.read_bytes()

    # Act
    size = jpeg_frame_size(data)

    # Assert
    assert size == (160, 120)


THUMBNAIL_ARTIFACT = ArtifactFile(path="thumbnail.jpg", kind="thumbnail", size_bytes=1, sha256="0" * 64)
UNAVAILABLE = Diagnostic(code=THUMBNAIL_UNAVAILABLE, detail="the thumbnail encoder wrote no image")


@pytest.mark.parametrize(
    ("artifact", "diagnostic"),
    [
        (THUMBNAIL_ARTIFACT, UNAVAILABLE),
        (None, None),
        (dataclasses.replace(THUMBNAIL_ARTIFACT, path="poster.jpg"), None),
        (dataclasses.replace(THUMBNAIL_ARTIFACT, kind="hls_segment"), None),
        (None, Diagnostic(code="audio_dropped", detail="not the thumbnail's")),
    ],
    ids=["both", "neither", "another path", "another kind", "another diagnostic"],
)
def test_thumbnail_result_when_not_exactly_the_thumbnail_or_its_diagnostic_should_raise(
    artifact: ArtifactFile | None, diagnostic: Diagnostic | None
) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        ThumbnailResult(artifact=artifact, diagnostic=diagnostic)


# --- review fix pass 1: metadata, the picture on screen, isolation -----------------------------------


def _markers(data: bytes) -> list[tuple[int, bytes]]:
    """(marker, payload) of every segment from SOI up to and including the first scan."""
    markers: list[tuple[int, bytes]] = []
    position = len(SOI)
    while data[position] == 0xFF:
        marker = data[position + 1]
        length = int.from_bytes(data[position + 2: position + 4], "big")
        markers.append((marker, data[position + 4: position + 2 + length]))
        if marker == 0xDA:
            break
        position += 2 + length
    return markers


def _mean_rgb(path: Path) -> tuple[float, ...]:
    rgb = _run_tool(
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    )
    pixels = len(rgb) // 3
    return tuple(sum(rgb[channel::3]) / pixels for channel in range(3))


def test_make_thumbnail_when_the_source_carries_an_icc_profile_should_write_nothing_but_the_image(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    if encode_media.icc_profiled is None:
        pytest.skip("the local ffmpeg does not keep the PNG's ICC profile in the MOV")
    job = _job(tmp_path, encode_media.icc_profiled)

    # Act
    result = job.run()

    # Assert
    data = job.thumbnail.read_bytes()
    markers = _markers(data)
    assert ICC_MARKER in encode_media.icc_profiled.read_bytes()
    assert result.artifact is not None
    assert ICC_MARKER not in data
    assert b"Lavc" not in data
    assert [marker for marker, _ in markers] == CLEAN_MARKERS
    assert markers[0][1].startswith(JFIF_IDENTIFIER)


@pytest.mark.parametrize(
    "segment",
    [
        _segment(0xE1, b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08"),
        _segment(0xE2, b"ICC_PROFILE\x00\x01\x01" + ICC_MARKER),
        _segment(0xED, b"Photoshop 3.0\x00"),
        _segment(0xFE, b"Lavc62.11.100\x00"),
        _segment(0xE0, b"JFXX\x00\x10"),
    ],
    ids=["Exif", "an ICC profile", "IPTC", "a comment", "an APP0 that is not JFIF"],
)
def test_make_thumbnail_when_ffmpeg_writes_a_jpeg_carrying_metadata_should_report_the_thumbnail_unavailable(
    segment: bytes, jpegs: _Jpegs, tmp_path: Path
) -> None:
    # Arrange: a JPEG of the planned size with one metadata segment after its SOI
    planned = jpegs.planned.read_bytes()
    with_metadata = tmp_path / "with_metadata.jpg"
    with_metadata.write_bytes(planned[: len(SOI)] + segment + planned[len(SOI):])
    job = _claimed_job(tmp_path)
    executable = _fake_ffmpeg(tmp_path, f'cp "{with_metadata}" "$last"\nexit 0\n')

    # Act
    result = job.run(ffmpeg_path=executable)

    # Assert
    _assert_unavailable(result, CARRIES_METADATA, job)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_jpeg(APP0, _segment(0xDD, b"\x00\x10"), _frame_header(), SCAN_HEADER), (320, 240)),
        (_jpeg(APP0, _segment(0xE1, b"Exif\x00\x00"), _frame_header(), SCAN_HEADER), None),
        (_jpeg(APP0, _frame_header(), _segment(0xFE, b"comment"), SCAN_HEADER), None),
        (_jpeg(_segment(0xE0, b"JFXX\x00\x10"), _frame_header(), SCAN_HEADER), None),
        (_jpeg(APP0, _segment(0xF0, b"\x00"), _frame_header(), SCAN_HEADER), None),
    ],
    ids=["a restart interval", "Exif", "a comment", "an APP0 that is not JFIF", "a reserved marker"],
)
def test_jpeg_frame_size_when_a_segment_is_not_part_of_the_image_should_refuse_the_jpeg(
    data: bytes, expected: tuple[int, int] | None
) -> None:
    # Arrange / Act
    size = jpeg_frame_size(data)

    # Assert
    assert size == expected


def test_make_thumbnail_when_the_source_is_one_picture_held_for_ten_seconds_should_show_that_picture(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: its only frame starts at 0 s; the thumbnail is taken at 1 s
    job = _job(tmp_path, encode_media.single_picture)

    # Act
    result = job.run()

    # Assert
    assert result.artifact is not None
    assert _close(_mean_rgb(job.thumbnail), RED, COLOUR_TOLERANCE)


@pytest.mark.parametrize(
    ("per_mille", "max_ms", "expected_colour"),
    [(100, 5_000, RED), (600, 60_000, BLUE), (900, 5_000, RED), (475, 60_000, RED)],
    ids=[
        "2 s: the first picture, shown since 0 s",
        "12 s: the share binds, the second picture",
        "18 s capped at 5 s: the cap binds, the first picture",
        "9.5 s: inside the last frame of the first picture, not the next frame",
    ],
)
def test_make_thumbnail_when_the_picture_changes_over_time_should_show_the_one_on_screen_at_its_time(
    per_mille: int, max_ms: int, expected_colour: tuple[float, ...], encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: red from 0 s, blue from 10 s, 20 s in all
    profile = fast_profile(thumbnail_at_per_mille=per_mille, thumbnail_at_max_ms=max_ms)
    job = _job(tmp_path, encode_media.two_pictures, profile=profile)

    # Act
    result = job.run()

    # Assert
    assert result.artifact is not None
    assert _close(_mean_rgb(job.thumbnail), expected_colour, COLOUR_TOLERANCE)


@pytest.mark.parametrize(
    ("time_ms", "frame_rate", "expected_ms"),
    [(1999, (25, 1), 1960), (2000, (30000, 1001), 1968), (60, (1, 1), 0), (2000, (25, 1), 2000)],
    ids=["25 fps", "NTSC, frame 59", "1 fps, before the second frame", "on a frame start"],
)
def test_on_frame_grid_ms_when_given_a_time_should_round_down_to_the_start_of_the_frame_on_screen(
    time_ms: int, frame_rate: tuple[int, int], expected_ms: int
) -> None:
    # Arrange / Act
    snapped = core.thumbnail._on_frame_grid_ms(time_ms, *frame_rate)

    # Assert
    assert snapped == expected_ms


@pytest.mark.parametrize(
    ("duration_ms", "expected_ms"),
    [(20_000, 2_000), (100_000, 5_000), (9, 0)],
    ids=["the share binds", "the cap binds", "shorter than ten milliseconds"],
)
def test_taken_at_ms_when_given_the_admitted_duration_should_take_the_profiles_share_capped(
    duration_ms: int, expected_ms: int, tmp_path: Path
) -> None:
    # Arrange: the pilot's 10 % capped at 5 s
    source = dataclasses.replace(_claimed_job(tmp_path, profile=PILOT_PROFILE).source, duration_ms=duration_ms)

    # Act
    taken_at = core.thumbnail._taken_at_ms(source, PILOT_PROFILE)

    # Assert
    assert taken_at == expected_ms


def test_make_thumbnail_when_the_source_is_a_playlist_disguised_as_mp4_should_report_the_thumbnail_unavailable(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: read as the admitted demuxer (mov), never as the playlist it is
    job = _claimed_job(tmp_path, source_path=encode_media.disguised_playlist)

    # Act
    result = job.run()

    # Assert
    assert result.artifact is None
    assert result.diagnostic is not None
    assert result.diagnostic.detail.startswith("the thumbnail encoder exited with status")
    assert list(job.output_root.iterdir()) == []


def test_make_thumbnail_when_the_container_crops_the_frame_should_show_the_whole_coded_frame(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange: red over blue, the container's crop removing the blue half
    job = _job(tmp_path, encode_media.cropped)
    width, height = thumbnail_geometry(job.reference, job.profile)

    # Act
    result = job.run()

    # Assert
    top_left, _, bottom_left, _ = _quadrant_means(_rgb(job.thumbnail, width, height, first_frame=False), width, height)
    assert result.artifact is not None
    assert _close(top_left, RED, COLOUR_TOLERANCE)
    assert _close(bottom_left, BLUE, COLOUR_TOLERANCE)


def test_make_thumbnail_when_a_decoded_frame_exceeds_the_pixel_cap_should_report_the_decoders_refusal(
    encode_media: EncodeMedia, tmp_path: Path
) -> None:
    # Arrange
    source_caps = dataclasses.replace(
        FAST_PROFILE.source,
        max_display_width=SMALL_PIXEL_CAP_EDGE,
        max_display_height=SMALL_PIXEL_CAP_EDGE,
        max_display_pixels=SMALL_PIXEL_CAP_EDGE**2,
    )
    job = _job(tmp_path, encode_media.landscape, profile=dataclasses.replace(FAST_PROFILE, source=source_caps))

    # Act
    result = job.run()

    # Assert
    _assert_unavailable(result, "a decoded frame exceeds the profile's pixel limit", job)


def test_make_thumbnail_when_it_runs_ffmpeg_should_read_the_source_as_a_rendition_does_and_seek_after_reading(
    tmp_path: Path,
) -> None:
    # Arrange
    job = _claimed_job(tmp_path)
    record = tmp_path / "argv.txt"
    executable = _fake_ffmpeg(tmp_path, f'printf "%s\\n" "$@" > "{record}"\nexit 1\n')

    # Act
    job.run(ffmpeg_path=executable)

    # Assert
    argv = record.read_text(encoding="ascii").splitlines()
    expected_input = input_arguments(job.source, job.profile)
    start = argv.index(expected_input[0])
    assert argv[start: start + len(expected_input)] == expected_input
    assert argv.count("-i") == 1
    assert argv.index("-ss") > argv.index("-i")
