"""The job's optional thumbnail: one JPEG frame of the admitted source, read with the rendition's isolation.

The plan says "an absent optional thumbnail is null plus a diagnostic", so every way a thumbnail can
fail ends in a `thumbnail_unavailable` diagnostic, never in a failed job: the renditions are what a
viewer needs. Two endings are the exception, because nothing after them can finish: the caller's
deadline passing, and a full scratch disk.

ffmpeg reads the source through `core/encode_command.py`'s input arguments (a local file, the one
admitted demuxer, the pixel cap, autorotation off) and seeks on the output side, so it decodes from
the start up to the thumbnail's time instead of trusting the container's seek index; the thumbnail's
own time budget bounds that decode. The `fps` filter first puts the frames on the renditions' constant
grid, holding the last picture to the end of the stream, and the seek is snapped down onto that grid,
so the thumbnail is the picture on screen at its time even for a still image or a slide show. The
frame is turned by exactly the admitted rotation, scaled to the largest rendition's aspect at the
profile's short edge (never upscaled) and written as a baseline 4:2:0 JPEG. Nothing of the source but
its pixels reaches the file: `-map_metadata` drops the container's tags, but an ICC profile travels as
frame side data, so the filter chain deletes it, and `+bitexact` keeps the encoder's version comment
out. A monitor stops ffmpeg once the file passes its byte limit. ffmpeg's exit status is never trusted
alone: the file must be a regular file within the limit and a whole JPEG whose frame header names
exactly the planned size and which carries no segment before its scan but its JFIF header, its tables
and its frame, so a later ffmpeg that wrote metadata there would be refused. The walk stops at the
scan: nothing between the scan data and the end marker is checked, where ffmpeg writes nothing today.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from core.bounded_process import BoundedResult, Deadline, run_bounded
from core.encode import (
    ARTIFACT_KIND_THUMBNAIL,
    FRAME_REFUSAL_PATTERNS,
    MAX_STDOUT_BYTES,
    MS_PER_SECOND,
    SHELL_CANNOT_RUN_EXIT_CODES,
    ArtifactFile,
    _deadline_exceeded,
    _disk_full,
    _errno_name,
    _require_directory,
    _require_int,
    _resolve_ffmpeg,
    _scratch_full,
)
from core.encode_command import (
    FFMPEG_ENV,
    LOG_LEVEL,
    OUTPUT_PROTOCOLS_PLAIN,
    ROTATION_FILTERS,
    EncodeSource,
    input_arguments,
    require_safe_path,
)
from core.failure import Diagnostic, WorkerFailure
from core.profile import MediaProfile
from core.renditions import RenditionPlan, even_round

logger = logging.getLogger(__name__)

THUMBNAIL_PATH = "thumbnail.jpg"
THUMBNAIL_UNAVAILABLE = "thumbnail_unavailable"
THUMBNAIL_CODEC = "mjpeg"
IMAGE_MUXER = "image2"
# Full-range 4:2:0, which mjpeg writes as a baseline JPEG on ffmpeg 7.1 and 8 alike (both log that
# the yuvj formats are deprecated, and both still take them).
THUMBNAIL_PIXEL_FORMAT = "yuvj420p"
# The source's ICC profile (a MOV colr/prof atom, a PNG iCCP chunk) travels as frame side data, which
# -map_metadata does not remove and mjpeg would copy into APP2 segments; ffmpeg 7.1 and 8 both take this.
ICC_PROFILE_REMOVAL = "sidedata=mode=delete:type=ICC_PROFILE"
# Keeps mjpeg's "Lavc<version>" comment (COM) out of the file.
BITEXACT_FLAGS = "+bitexact"
PER_MILLE = 1000
EMPTY_CWD_PREFIX = "thumbnail-cwd-"
READ_CHUNK_BYTES = 64 * 1024
# O_NONBLOCK keeps a FIFO planted in place of the thumbnail from blocking the open.
READ_FLAGS = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

JPEG_MARKER_PREFIX = 0xFF
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
JPEG_MIN_BYTES = len(JPEG_SOI) + len(JPEG_EOI)
JPEG_SOS = 0xDA
# Baseline, extended sequential and progressive Huffman frames: what a JPEG decoder always reads.
JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2})
# TEM and RST0 to RST7 carry no length.
JPEG_STANDALONE_MARKERS = frozenset({0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7})
# Before the scan, besides the frame header, only these may appear: APP0 (a JFIF header only), the
# quantization and Huffman tables, the restart interval, and the standalone markers.
JPEG_APP0 = 0xE0
JPEG_JFIF_IDENTIFIER = b"JFIF\x00"
JPEG_IMAGE_MARKERS = frozenset({JPEG_APP0, 0xDB, 0xC4, 0xDD}) | JPEG_STANDALONE_MARKERS
# APP1 to APP15 (Exif, XMP, ICC profile, ...) and COM: data about the image, never the image.
JPEG_METADATA_MARKERS = frozenset({*range(0xE1, 0xF0), 0xFE})
# A stuffed zero is no marker; SOI and EOI never open a segment before the scan.
JPEG_NOT_A_SEGMENT = frozenset({0x00, 0xD8, 0xD9})
JPEG_LENGTH_BYTES = 2
# A frame header's sample precision, height, width and component count, after its length.
JPEG_FRAME_HEADER_BYTES = 6
JPEG_FRAME_HEIGHT = slice(1, 3)
JPEG_FRAME_WIDTH = slice(3, 5)
# mjpeg writes about six segments before its scan; a walk past this many is not reading its output.
JPEG_MAX_SEGMENTS = 64

NOT_STARTED = "the thumbnail encoder could not be started"
NO_IMAGE = "the thumbnail encoder wrote no image"
NOT_A_FILE = "the thumbnail is not a regular file"
NOT_A_PLANNED_JPEG = "the thumbnail is not a JPEG of the planned size"
CARRIES_METADATA = "the thumbnail carries metadata"
UNREADABLE = "the thumbnail could not be read"
FRAME_TOO_LARGE = "a decoded frame exceeds the profile's pixel limit"
ENDED_ON_SIGNAL = "the thumbnail encoder ended on a signal"
STDOUT_UNEXPECTED = "the thumbnail encoder wrote more than expected to stdout"
# The monitor always records its reason before it stops ffmpeg; this stands in should it ever not.
MONITOR_STOP_WITHOUT_REASON = "the thumbnail's output monitor stopped the encoder"


@dataclass(frozen=True, slots=True)
class ThumbnailResult:
    """The thumbnail, or the diagnostic that says why there is none: exactly one of the two."""

    artifact: ArtifactFile | None  # kind "thumbnail", path "thumbnail.jpg"
    diagnostic: Diagnostic | None  # code "thumbnail_unavailable"

    def __post_init__(self) -> None:
        if (self.artifact is None) == (self.diagnostic is None):
            raise ValueError("exactly one of artifact and diagnostic must be set")
        if self.artifact is not None and not _is_thumbnail_artifact(self.artifact):
            raise ValueError(f"artifact must be the {ARTIFACT_KIND_THUMBNAIL} artifact at {THUMBNAIL_PATH}")
        if self.diagnostic is not None and (
            not isinstance(self.diagnostic, Diagnostic) or self.diagnostic.code != THUMBNAIL_UNAVAILABLE
        ):
            raise ValueError(f"diagnostic must be a {THUMBNAIL_UNAVAILABLE} Diagnostic")


def _is_thumbnail_artifact(artifact: object) -> bool:
    return isinstance(artifact, ArtifactFile) and (artifact.path, artifact.kind) == (
        THUMBNAIL_PATH,
        ARTIFACT_KIND_THUMBNAIL,
    )


def thumbnail_geometry(reference: RenditionPlan, profile: MediaProfile) -> tuple[int, int]:
    """(width, height) of the thumbnail: the reference rendition's aspect with the profile's short
    edge, or the reference's own size when its short edge is no longer (never upscaled)."""
    if not isinstance(reference, RenditionPlan):
        raise TypeError("reference must be a RenditionPlan")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    short_edge = profile.encode.thumbnail_short_edge
    reference_short = min(reference.width, reference.height)
    reference_long = max(reference.width, reference.height)
    if reference_short <= short_edge:
        return reference.width, reference.height
    long_edge = even_round(Fraction(reference_long * short_edge, reference_short))
    if reference.width >= reference.height:
        return long_edge, short_edge
    return short_edge, long_edge


def _segment_at(data: bytes, position: int) -> tuple[int, bytes, int] | None:
    """(marker, payload, next position) of the whole marker segment at `position`, or None."""
    if position >= len(data) or data[position] != JPEG_MARKER_PREFIX:
        return None
    # Fill bytes may precede a marker.
    while position < len(data) and data[position] == JPEG_MARKER_PREFIX:
        position += 1
    if position >= len(data):
        return None
    marker = data[position]
    position += 1
    if marker in JPEG_STANDALONE_MARKERS:
        return marker, b"", position
    if marker in JPEG_NOT_A_SEGMENT:
        return None
    payload_start = position + JPEG_LENGTH_BYTES
    if payload_start > len(data):
        return None
    length = int.from_bytes(data[position:payload_start], "big")
    end = position + length
    if length < JPEG_LENGTH_BYTES or end > len(data):
        return None
    return marker, data[payload_start:end], end


def _frame_size(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < JPEG_FRAME_HEADER_BYTES:
        return None
    height = int.from_bytes(payload[JPEG_FRAME_HEIGHT], "big")
    width = int.from_bytes(payload[JPEG_FRAME_WIDTH], "big")
    # A height of 0 is defined later in the scan (DNL): not a size this header names.
    if width == 0 or height == 0:
        return None
    return width, height


def _is_metadata(marker: int, payload: bytes) -> bool:
    return marker in JPEG_METADATA_MARKERS or (marker == JPEG_APP0 and not payload.startswith(JPEG_JFIF_IDENTIFIER))


def _jpeg_layout(data: bytes) -> tuple[int, int] | str:
    """(width, height) of a whole JPEG that holds nothing but its image; or why it is refused.

    The data must open with SOI followed by a marker and end with EOI. The walk goes from SOI one
    marker segment at a time, at most JPEG_MAX_SEGMENTS of them, up to the first scan (SOS). Before it,
    only a JFIF APP0, the tables, the restart interval, standalone markers and one baseline or
    progressive frame header may appear: any metadata segment (APP1-APP15, COM, an APP0 that is not
    JFIF) is CARRIES_METADATA; a truncated segment, a second frame header, any other kind of frame
    (lossless, hierarchical, arithmetic-coded), any other marker or no frame header is NOT_A_PLANNED_JPEG.
    """
    if len(data) < JPEG_MIN_BYTES or not data.startswith(JPEG_SOI) or not data.endswith(JPEG_EOI):
        return NOT_A_PLANNED_JPEG
    position = len(JPEG_SOI)
    size: tuple[int, int] | None = None
    for _ in range(JPEG_MAX_SEGMENTS):
        segment = _segment_at(data, position)
        if segment is None:
            return NOT_A_PLANNED_JPEG
        marker, payload, position = segment
        if marker == JPEG_SOS:
            return NOT_A_PLANNED_JPEG if size is None else size
        if _is_metadata(marker, payload):
            return CARRIES_METADATA
        if marker in JPEG_FRAME_MARKERS and size is None:
            size = _frame_size(payload)
            if size is None:
                return NOT_A_PLANNED_JPEG
        elif marker not in JPEG_IMAGE_MARKERS:
            return NOT_A_PLANNED_JPEG
    return NOT_A_PLANNED_JPEG


def jpeg_frame_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the frame header of a whole baseline or progressive JPEG that holds
    nothing but its image; None for any other data (see `_jpeg_layout`)."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    layout = _jpeg_layout(data)
    return None if isinstance(layout, str) else layout


def _too_large(max_bytes: int) -> str:
    return f"the thumbnail exceeds {max_bytes} bytes"


def _unreadable(error: OSError) -> str:
    logger.warning("the thumbnail could not be read: %s errno=%s", type(error).__name__, _errno_name(error))
    return UNREADABLE


class _SizeMonitor:
    """Stops ffmpeg once the thumbnail passes its byte limit; records why."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self.problem: str | None = None

    def __call__(self) -> bool:
        try:
            size = os.lstat(self._path).st_size
        except FileNotFoundError:
            # Nothing written yet.
            return True
        except OSError as error:
            self.problem = _unreadable(error)
            return False
        if size > self._max_bytes:
            self.problem = _too_large(self._max_bytes)
            return False
        return True


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds // MS_PER_SECOND}.{milliseconds % MS_PER_SECOND:03d}"


def _taken_at_ms(source: EncodeSource, profile: MediaProfile) -> int:
    """The thumbnail's time: the profile's share of the admitted duration, but never later than its cap."""
    encode = profile.encode
    return min(source.duration_ms * encode.thumbnail_at_per_mille // PER_MILLE, encode.thumbnail_at_max_ms)


def _on_frame_grid_ms(time_ms: int, frame_rate_num: int, frame_rate_den: int) -> int:
    """The start, rounded down to whole ms, of the output frame on screen at `time_ms`.

    `fps` puts frame k at k * den / num s. A seek to that start rounded down keeps frame k and drops
    frame k - 1: one frame lasts at least 1/60 s at the profile's ceiling, far more than the 1 ms the
    rounding can take. Frame 0 exists whenever the renditions have a frame: `fps` rounds the stream's
    end to its grid, so a stream shorter than about half an output frame has none (0.4 s at 1 fps).
    """
    frame = time_ms * frame_rate_num // (MS_PER_SECOND * frame_rate_den)
    return frame * MS_PER_SECOND * frame_rate_den // frame_rate_num


def _seek_ms(source: EncodeSource, reference: RenditionPlan, profile: MediaProfile) -> int:
    return _on_frame_grid_ms(_taken_at_ms(source, profile), reference.frame_rate_num, reference.frame_rate_den)


def _video_filter(source: EncodeSource, reference: RenditionPlan, width: int, height: int) -> str:
    steps = [
        ICC_PROFILE_REMOVAL,
        # The renditions' constant frame rate first: the last picture is held to the end of the stream,
        # so the seek finds the picture on screen at its time, still images and slide shows included.
        f"fps={reference.frame_rate_num}/{reference.frame_rate_den}",
        # Exactly the admitted rotation, as the renditions get it.
        *ROTATION_FILTERS[source.rotation_degrees],
        f"scale={width}:{height}",
        "setsar=1",
        f"format={THUMBNAIL_PIXEL_FORMAT}",
    ]
    return ",".join(steps)


def _thumbnail_argv(
    executable: str,
    source: EncodeSource,
    reference: RenditionPlan,
    profile: MediaProfile,
    *,
    target: Path,
    width: int,
    height: int,
) -> list[str]:
    require_safe_path("the thumbnail's path", target)
    return [
        executable,
        "-hide_banner", "-nostdin", "-nostats", "-loglevel", LOG_LEVEL,
        *input_arguments(source, profile),
        # Output: one local file; the admitted video stream only; no metadata carried over.
        "-protocol_whitelist", OUTPUT_PROTOCOLS_PLAIN,
        "-map", f"0:{source.video_index}",
        "-an", "-sn", "-dn",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        # On the output side: decoded from the start, the container's seek index never trusted.
        "-ss", _seconds(_seek_ms(source, reference, profile)),
        "-frames:v", "1",
        "-vf", _video_filter(source, reference, width, height),
        "-c:v", THUMBNAIL_CODEC,
        "-flags:v", BITEXACT_FLAGS,
        "-q:v", str(profile.encode.thumbnail_quality),
        "-f", IMAGE_MUXER,
        "-update", "1",
        "-n",
        str(target),
    ]


def _run_ffmpeg(
    argv: list[str], profile: MediaProfile, *, work_dir: Path, deadline: Deadline, monitor: _SizeMonitor
) -> BoundedResult | None:
    """ffmpeg's ending, or None when it could not be started. Its working directory is an empty
    temporary directory under `work_dir`, which lives only while it runs."""
    try:
        with tempfile.TemporaryDirectory(prefix=EMPTY_CWD_PREFIX, dir=work_dir, ignore_cleanup_errors=True) as cwd:
            return run_bounded(
                argv,
                deadline=deadline,
                max_stdout_bytes=MAX_STDOUT_BYTES,
                max_stderr_bytes=profile.encode.max_stderr_bytes,
                env=FFMPEG_ENV,
                cwd=cwd,
                monitor=monitor,
                monitor_interval_ms=profile.encode.output_check_interval_ms,
                stderr_patterns=FRAME_REFUSAL_PATTERNS,
            )
    except OSError as error:
        # Also where ffmpeg's output could not be fully read; the group is dead by then.
        logger.warning(
            "the thumbnail encoder could not be started: %s errno=%s", type(error).__name__, _errno_name(error)
        )
        return None
    except RuntimeError:
        logger.warning("the thumbnail encoder could not be started: a reader thread did not start")
        return None


def _ending_problem(result: BoundedResult, monitor: _SizeMonitor, wall_ms: int) -> str | None:
    """Why ffmpeg's ending is not a clean exit, in our words; None when it is. ffmpeg's own words
    are never forwarded."""
    if result.timed_out:
        return f"the thumbnail exceeded its time budget of {wall_ms} ms"
    if result.stopped_by_monitor:
        return monitor.problem or MONITOR_STOP_WITHOUT_REASON
    if result.stdout_overflow:
        return STDOUT_UNEXPECTED
    returncode = result.returncode
    if returncode is None or returncode < 0:
        return ENDED_ON_SIGNAL
    if returncode in SHELL_CANNOT_RUN_EXIT_CODES:
        return NOT_STARTED
    if result.stderr_patterns_seen:
        return FRAME_TOO_LARGE
    if returncode > 0:
        return f"the thumbnail encoder exited with status {returncode}"
    return None


def _run(
    source: EncodeSource,
    reference: RenditionPlan,
    profile: MediaProfile,
    *,
    target: Path,
    width: int,
    height: int,
    work_dir: Path,
    deadline: Deadline,
    ffmpeg_path: str | None,
) -> str | None:
    """Run ffmpeg within the thumbnail's own time budget; why it did not exit cleanly, or None."""
    try:
        executable = _resolve_ffmpeg(ffmpeg_path)
    except WorkerFailure:
        return NOT_STARTED
    wall_ms = profile.encode.thumbnail_wall_ms
    monitor = _SizeMonitor(target, profile.encode.thumbnail_max_bytes)
    argv = _thumbnail_argv(executable, source, reference, profile, target=target, width=width, height=height)
    result = _run_ffmpeg(argv, profile, work_dir=work_dir, deadline=deadline.within_ms(wall_ms), monitor=monitor)
    if result is None:
        return NOT_STARTED
    return _ending_problem(result, monitor, wall_ms)


def _read_at_most(descriptor: int, limit: int) -> bytes:
    data = bytearray()
    while len(data) < limit:
        chunk = os.read(descriptor, min(READ_CHUNK_BYTES, limit - len(data)))
        if not chunk:
            break
        data += chunk
    return bytes(data)


def _read_thumbnail(target: Path, max_bytes: int) -> bytes | str:
    """The bytes of the regular file at `target`, read never through a link; or why they cannot be
    the thumbnail."""
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        return NO_IMAGE
    except OSError as error:
        return _unreadable(error)
    if not stat.S_ISREG(info.st_mode):
        return NOT_A_FILE
    if info.st_size > max_bytes:
        return _too_large(max_bytes)
    # Every step that can fail, the close included, ends in a diagnostic, never an exception.
    try:
        descriptor = os.open(target, READ_FLAGS)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return NOT_A_FILE
            data = _read_at_most(descriptor, max_bytes + 1)
        finally:
            os.close(descriptor)
    except OSError as error:
        return _unreadable(error)
    if not data:
        return NO_IMAGE
    if len(data) > max_bytes:
        return _too_large(max_bytes)
    return data


def _verified_thumbnail(target: Path, width: int, height: int, max_bytes: int) -> ArtifactFile | str:
    """The thumbnail's artifact once the file proves to be a whole JPEG of the planned size holding
    nothing but its image; or why not."""
    data = _read_thumbnail(target, max_bytes)
    if isinstance(data, str):
        return data
    layout = _jpeg_layout(data)
    if isinstance(layout, str):
        return layout
    if layout != (width, height):
        return NOT_A_PLANNED_JPEG
    return ArtifactFile(
        path=THUMBNAIL_PATH,
        kind=ARTIFACT_KIND_THUMBNAIL,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _attempt(
    source: EncodeSource,
    reference: RenditionPlan,
    profile: MediaProfile,
    *,
    target: Path,
    work_dir: Path,
    deadline: Deadline,
    ffmpeg_path: str | None,
) -> ArtifactFile | str:
    """The verified thumbnail, or why there is none. Raises WorkerFailure only when the job cannot
    finish: the caller's deadline has passed, or the scratch disk is full."""
    width, height = thumbnail_geometry(reference, profile)
    problem = _run(
        source,
        reference,
        profile,
        target=target,
        width=width,
        height=height,
        work_dir=work_dir,
        deadline=deadline,
        ffmpeg_path=ffmpeg_path,
    )
    if deadline.expired():
        raise _deadline_exceeded()
    # A full disk explains a failed run, and ffmpeg can exit 0 after a write it could not finish.
    if _disk_full(target.parent):
        raise _scratch_full()
    if problem is not None:
        return problem
    verified = _verified_thumbnail(target, width, height, profile.encode.thumbnail_max_bytes)
    if deadline.expired():
        raise _deadline_exceeded()
    return verified


def _remove_thumbnail(target: Path) -> None:
    """Best effort: the job's scratch cleanup removes whatever is left. Paths are never logged."""
    try:
        if stat.S_ISDIR(os.lstat(target).st_mode):
            shutil.rmtree(target)
        else:
            os.unlink(target)
    except FileNotFoundError:
        return
    except OSError as error:
        logger.warning("a failed thumbnail could not be removed: errno=%s", _errno_name(error))


def _validate_arguments(
    source: object,
    reference: object,
    profile: object,
    output_root: object,
    work_dir: object,
    deadline: object,
    ffmpeg_path: object,
) -> None:
    if not isinstance(source, EncodeSource):
        raise TypeError("source must be an EncodeSource")
    if not isinstance(reference, RenditionPlan):
        raise TypeError("reference must be a RenditionPlan")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if ffmpeg_path is not None and (not isinstance(ffmpeg_path, str) or not ffmpeg_path):
        raise ValueError("ffmpeg_path must be None or a non-empty string")
    _require_int("the reference's width", reference.width, 1)
    _require_int("the reference's height", reference.height, 1)
    resolved_output = _require_directory("output_root", output_root)
    resolved_work = _require_directory("work_dir", work_dir)
    if resolved_output.is_relative_to(resolved_work) or resolved_work.is_relative_to(resolved_output):
        raise ValueError("output_root and work_dir must not lie inside one another")
    if os.path.lexists(resolved_output / THUMBNAIL_PATH):
        raise ValueError("the thumbnail already exists")


def make_thumbnail(
    source: EncodeSource,
    reference: RenditionPlan,
    *,
    profile: MediaProfile,
    output_root: Path,
    work_dir: Path,
    deadline: Deadline,
    ffmpeg_path: str | None = None,
) -> ThumbnailResult:
    """Write `output_root / thumbnail.jpg` from `source`, sized after `reference` (the largest
    planned rendition, whose width and height give the display geometry), and verify it.

    ffmpeg's empty working directory lives under `work_dir` only while it runs; the thumbnail has
    the profile's own time budget, never past `deadline`.
    Returns the thumbnail's artifact, or a `thumbnail_unavailable` diagnostic for every ending but
    two, after removing whatever it left. Raises WorkerFailure when `deadline` has passed (before
    the start or at the end) and when the scratch disk is full, after removing the thumbnail;
    raises TypeError or ValueError for invalid arguments before anything runs.
    """
    _validate_arguments(source, reference, profile, output_root, work_dir, deadline, ffmpeg_path)
    if deadline.expired():
        raise _deadline_exceeded()
    target = output_root / THUMBNAIL_PATH
    try:
        outcome = _attempt(
            source,
            reference,
            profile,
            target=target,
            work_dir=work_dir,
            deadline=deadline,
            ffmpeg_path=ffmpeg_path,
        )
    except BaseException:
        _remove_thumbnail(target)
        raise
    if isinstance(outcome, ArtifactFile):
        return ThumbnailResult(artifact=outcome, diagnostic=None)
    _remove_thumbnail(target)
    logger.warning("the thumbnail is unavailable: %s", outcome)
    return ThumbnailResult(artifact=None, diagnostic=Diagnostic(code=THUMBNAIL_UNAVAILABLE, detail=outcome))
