"""Encode one HLS rendition of an admitted source within the job's output allowance, and verify it.

ffmpeg runs under `core/bounded_process.py` with a monitor that measures the rendition directory
while it writes, and stops it when the output passes its byte or file allowance. The run and the
verification share one wall-time budget proportional to the admitted duration: no count of packets
or frames bounds what a decoder does, so time is the decoder's bound. Once ffmpeg exits, the
output is verified before anything trusts it, because ffmpeg exits 0 even when its output protocol
whitelist stopped it from writing a segment, when a decoder skipped a frame above the pixel cap, or
when a segment write failed: the exact playlist grammar, every file accounted for, the exact
allowance, the duration cut, the bandwidth the segments on disk really take, and every segment's
structure (whole MPEG-TS packets, decrypted first when encrypted), its video's presentation span
against its declared duration and the frames that span holds; then the whole rendition is decoded
once, every error fatal, and must yield exactly the frames its segments carry. Every ending is a
typed failure (`core/failure.py`), and a failed rendition's directory is removed.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from core.bounded_process import BoundedResult, Deadline, run_bounded
from core.encode_command import (
    FFMPEG_ENV,
    PLAYLIST_NAME,
    UNSAFE_PATH_CHARACTERS,
    EncodeSource,
    rendition_argv,
    require_safe_path,
)
from core.failure import Failure, WorkerFailure
from core.hls_key import KEY_BYTES, KEY_URI_PLACEHOLDER, KeyInfo, hls_key_info
from core.hls_playlist import (
    TOO_MANY_SEGMENTS,
    MediaPlaylist,
    PlaylistInvalid,
    PlaylistSegment,
    average_bandwidth_bps,
    peak_bandwidth_bps,
    read_media_playlist,
)
from core.profile import RENDITION_NAMES, MediaProfile
from core.renditions import RenditionPlan

logger = logging.getLogger(__name__)

# ffmpeg writes nothing to stdout with this command; anything there is unexpected.
MAX_STDOUT_BYTES = 64 * 1024
# Contract v2: an artifact's size_bytes and a rendition's bandwidth_bps have these maxima.
MAX_ARTIFACT_BYTES = 4_294_967_296
MAX_BANDWIDTH_BPS = 999_999_999
ARTIFACT_KIND_PLAYLIST = "hls_media_playlist"
ARTIFACT_KIND_SEGMENT = "hls_segment"
ARTIFACT_KIND_MASTER = "hls_master_playlist"
ARTIFACT_KIND_THUMBNAIL = "thumbnail"
# The active artifact kinds of contract v2's manifest 1.0.0-draft.
ARTIFACT_KINDS = frozenset(
    {ARTIFACT_KIND_MASTER, ARTIFACT_KIND_PLAYLIST, ARTIFACT_KIND_SEGMENT, ARTIFACT_KIND_THUMBNAIL}
)
# Contract v2's rel_path: up to four lowercase components, never absolute, never "." or "..".
REL_PATH_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}(/[a-z0-9][a-z0-9_.-]{0,63}){0,3}")
REL_PATH_MAX_CHARS = 128
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
HASH_CHUNK_BYTES = 1024 * 1024
MS_PER_SECOND = 1000
# How far the measured duration may pass the admitted duration plus one output frame (the frame
# that starts before the cut ends after it). On every fixture the measured duration stays within
# that frame; the slack covers only the playlist's own rounding: each EXTINF is printed to six
# decimals (an error of at most 0.5 microseconds), which over 3,600 segments adds up to 1.8 ms.
DURATION_SLACK_MS = 2
PLAYLIST_FILES = 1
# The smallest rendition: a playlist and one segment of at least one byte.
MIN_ARTIFACTS = PLAYLIST_FILES + 1
MIN_OUTPUT_BYTES = 1
MIN_SEGMENT_BYTES = 1
# While ffmpeg runs, one file beyond the limit may be in flight (a segment it has just opened).
TRANSIENT_FILES = 1
# A shell's "found but not executable" and "not found" exit statuses.
SHELL_CANNOT_RUN_EXIT_CODES = frozenset({126, 127})
# Start errors that mean the configured ffmpeg itself cannot run (besides FileNotFoundError and
# PermissionError); any other start error is a passing shortage of the worker's own resources.
ENCODER_UNUSABLE_ERRNOS = frozenset({errno.ENOEXEC})
# What a decoder logs when it refuses a frame above -max_pixels: libavutil's image check (every
# native decoder) and libdav1d's frame size limit (AV1). ffmpeg may still exit 0 after skipping the
# frame, and later warnings may push the line out of the kept tail, so every stderr byte is scanned.
FRAME_REFUSAL_PATTERNS = (
    # libavutil's validity check, which runs before its pixel-count check; nonzero sides only, so
    # the "0x0" line a decoder prints after an earlier refusal is not taken for one.
    re.compile(rb"Picture size [1-9][0-9]{0,9}x[1-9][0-9]{0,9} is invalid"),
    re.compile(rb"exceeds specified max pixel count"),
    re.compile(rb"Frame size [0-9]{1,10}x[0-9]{1,10} exceeds limit"),
)
NO_SPACE_MARKER = b"No space left on device"
# Below this much free space beside the output, a failed or unproven write is a full scratch disk:
# ffmpeg's HLS muxer can drop a segment's write error and still exit 0.
MIN_FREE_BYTES = 1024 * 1024
TS_PACKET_BYTES = 188
TS_SYNC_BYTE = 0x47
AES_BLOCK_BYTES = 16
AES_BLOCK_BITS = 128
# Verification checks the time budget before it reads each file and after every this many bytes,
# counted across the whole verification.
DEADLINE_CHECK_BYTES = 64 * 1024 * 1024
TS_PAYLOAD_UNIT_START = 0x40
TS_HEADER_BYTES = 4
# adaptation_field_control: a payload only, or an adaptation field then a payload.
TS_PAYLOAD_ONLY = 0b01
TS_ADAPTATION_FIELD_AND_PAYLOAD = 0b11
# Maps a TS header's second byte to 1 when its payload_unit_start bit is set: the scan finds the
# packets that open a PES packet with C-level byte operations, and only those reach Python.
PAYLOAD_UNIT_START_TABLE = bytes(1 if byte & TS_PAYLOAD_UNIT_START else 0 for byte in range(256))
PES_START_CODE = b"\x00\x00\x01"
# A PES header up to its PTS: start code, stream_id, length, two flag bytes, header length, PTS.
PES_HEADER_WITH_PTS_BYTES = 14
FIRST_VIDEO_STREAM_ID = 0xE0
LAST_VIDEO_STREAM_ID = 0xEF
PES_PTS_PRESENT = frozenset({0b10, 0b11})  # PTS_DTS_flags: PTS only, or PTS and DTS
PTS_CLOCK_HZ = 90_000
# A fresh encode's presentation timestamps never span this many ticks (13 hours) within a segment;
# a larger span is a wrap or garbage.
MAX_PTS_SPAN_TICKS = 1 << 32
# A segment's video, from its first to its last frame plus one frame, must match its EXTINF within
# half a frame and 2 ms (the worst measured on ffmpeg 7.1 and 8 is 0.00033 ms, EXTINF's rounding),
# and hold one frame for every frame duration of that span.
SPAN_TOLERANCE_FRAMES = Fraction(1, 2)
SPAN_TOLERANCE = Fraction(2, MS_PER_SECOND)
# The self-decode: the rendition's own segments, read back through a private check playlist and
# decoded once with every error fatal (-xerror), video and audio.
CHECK_DIR_PREFIX = "self-check-"
CHECK_PLAYLIST_NAME = "check.m3u8"
CHECK_CWD_NAME = "cwd"
CHECK_MAX_STDOUT_BYTES = 1024 * 1024
CHECK_PROGRESS_PERIOD_S = "5"
CHECK_PLAYLIST_VERSION = 3
PROGRESS_FRAME_PREFIX = "frame="
PROGRESS_END = "progress=end"
FRAME_COUNT_PATTERN = re.compile(r"0|[1-9][0-9]{0,17}")
# EXTINF values were parsed from at most 20 characters of decimal text.
MAX_DECIMAL_DIGITS = 20
DECIMAL_BASE = 10
KEY_PATH_FORBIDDEN = '"'
UNKNOWN_ERRNO_NAME = "unknown"
EMPTY_CWD_PREFIX = "encode-cwd-"
FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

INPUT_LIMITS_EXCEEDED = "input_limits_exceeded"
DEADLINE_EXCEEDED = "deadline_exceeded"
ENCODE_DEADLINE_EXCEEDED = "encode_deadline_exceeded"
ENCODE_BUDGET_EXCEEDED = "encode_budget_exceeded"
ENCODER_FAILED = "encoder_failed"
OUTPUT_TOO_LARGE = "output_too_large"
TOO_MANY_ARTIFACTS = "too_many_artifacts"
RENDITION_OUTPUT_INVALID = "rendition_output_invalid"
# The monitor always records its reason before it stops ffmpeg; this stands in should it ever not.
MONITOR_STOP_WITHOUT_REASON = Failure(
    error_class="internal_error", code="output_unreadable", retryable=True,
    detail="the output monitor stopped ffmpeg without a recorded reason",
)


def _require_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int of at least {minimum}, got {value!r}")


def _failure(error_class: str, code: str, detail: str, *, retryable: bool = False) -> WorkerFailure:
    return WorkerFailure(Failure(error_class=error_class, code=code, retryable=retryable, detail=detail))


def _errno_name(error: BaseException) -> str:
    # The errno's symbolic name only: an OSError's own message may carry a path.
    return errno.errorcode.get(getattr(error, "errno", None) or 0, UNKNOWN_ERRNO_NAME)


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str  # relative to the job's output root: "<rendition>/playlist.m3u8", "master.m3u8", "thumbnail.jpg"
    kind: str  # one of ARTIFACT_KINDS (contract v2 artifact kinds)
    size_bytes: int
    sha256: str  # 64 lowercase hex

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or len(self.path) > REL_PATH_MAX_CHARS
            or REL_PATH_PATTERN.fullmatch(self.path) is None
        ):
            raise ValueError("path must be a contract rel_path")
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"kind must be one of {sorted(ARTIFACT_KINDS)}, got {self.kind!r}")
        _require_int("size_bytes", self.size_bytes, 1)
        if not isinstance(self.sha256, str) or SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True)
class RenditionOutput:
    name: str
    width: int
    height: int
    frame_rate_num: int  # the plan's constant output frame rate
    frame_rate_den: int
    codecs: str
    has_audio: bool
    encrypted: bool  # every segment AES-128 encrypted
    playlist_path: str
    segment_count: int
    duration_ms: int  # measured: ceil(1000 * sum of EXTINF)
    bandwidth_bps: int  # peak, measured
    average_bandwidth_bps: int  # measured
    total_bytes: int
    artifacts: tuple[ArtifactFile, ...]  # the playlist first, then the segments in order

    def __post_init__(self) -> None:
        _require_int("frame_rate_num", self.frame_rate_num, 1)
        _require_int("frame_rate_den", self.frame_rate_den, 1)
        if not isinstance(self.has_audio, bool) or not isinstance(self.encrypted, bool):
            raise ValueError("has_audio and encrypted must be bools")
        if not isinstance(self.artifacts, tuple) or not all(isinstance(item, ArtifactFile) for item in self.artifacts):
            raise ValueError("artifacts must be a tuple of ArtifactFile")


@dataclass(frozen=True, slots=True)
class OutputAllowance:
    """What the job's remaining outputs may still take: bytes and files."""

    max_bytes: int  # >= 0
    max_artifacts: int  # >= 0

    def __post_init__(self) -> None:
        _require_int("max_bytes", self.max_bytes, 0)
        _require_int("max_artifacts", self.max_artifacts, 0)

    @classmethod
    def for_job(
        cls, *, max_output_bytes: int, max_artifact_count: int, source_size_bytes: int, profile: MediaProfile
    ) -> OutputAllowance:
        """The dispatch's output limits, capped by the scratch room the source leaves under the profile's ceiling.

        It covers every output of the job: the caller passes it straight to `core/package.py`
        `reserve_package`, which sets aside what the master playlist, the thumbnail and the
        manifest need, and hands the rest to the renditions.
        """
        _require_int("max_output_bytes", max_output_bytes, 1)
        _require_int("max_artifact_count", max_artifact_count, 1)
        _require_int("source_size_bytes", source_size_bytes, 1)
        if not isinstance(profile, MediaProfile):
            raise TypeError("profile must be a MediaProfile")
        scratch_room = profile.encode.max_scratch_bytes - source_size_bytes
        if scratch_room <= 0:
            raise _failure(
                INPUT_LIMITS_EXCEEDED,
                "scratch_exceeded",
                f"a source of {source_size_bytes} bytes leaves no room under the scratch ceiling of "
                f"{profile.encode.max_scratch_bytes} bytes",
            )
        return cls(max_bytes=min(max_output_bytes, scratch_room), max_artifacts=max_artifact_count)

    def reserve(self, max_bytes: int, max_artifacts: int) -> OutputAllowance:
        """What remains once `max_bytes` and `max_artifacts` are set aside for other outputs.

        Raises the typed limit failure when the room does not fit the allowance.
        """
        _require_int("max_bytes", max_bytes, 0)
        _require_int("max_artifacts", max_artifacts, 0)
        if max_bytes > self.max_bytes:
            raise _failure(
                INPUT_LIMITS_EXCEEDED, OUTPUT_TOO_LARGE,
                f"the job's {self.max_bytes} output bytes cannot hold the {max_bytes} its package needs",
            )
        if max_artifacts > self.max_artifacts:
            raise _failure(
                INPUT_LIMITS_EXCEEDED, TOO_MANY_ARTIFACTS,
                f"the job's {self.max_artifacts} files cannot hold the {max_artifacts} its package needs",
            )
        return OutputAllowance(max_bytes=self.max_bytes - max_bytes, max_artifacts=self.max_artifacts - max_artifacts)

    def after(self, output: RenditionOutput) -> OutputAllowance:
        """What is left once `output` is kept. Raises ValueError when it does not fit."""
        if not isinstance(output, RenditionOutput):
            raise TypeError("output must be a RenditionOutput")
        remaining_bytes = self.max_bytes - output.total_bytes
        remaining_artifacts = self.max_artifacts - len(output.artifacts)
        if remaining_bytes < 0 or remaining_artifacts < 0:
            raise ValueError("the output does not fit the allowance")
        return OutputAllowance(max_bytes=remaining_bytes, max_artifacts=remaining_artifacts)


class _OutputMonitor:
    """Measures the rendition directory while ffmpeg writes; records why it asked to stop."""

    def __init__(self, directory: Path, allowance: OutputAllowance, profile: MediaProfile) -> None:
        self._directory = directory
        self._max_bytes = allowance.max_bytes
        segment_file_limit = profile.encode.max_segments_per_rendition + PLAYLIST_FILES
        self._file_limit = min(allowance.max_artifacts, segment_file_limit)
        self._segment_limit_is_smaller = segment_file_limit < allowance.max_artifacts
        self._max_segments = profile.encode.max_segments_per_rendition
        self.failure: Failure | None = None

    def __call__(self) -> bool:
        try:
            total_bytes, file_count = _measure_directory(self._directory)
        except OSError as error:
            self.failure = _output_unreadable(error).failure
            return False
        if total_bytes > self._max_bytes:
            self.failure = _output_too_large(self._max_bytes).failure
            return False
        if file_count > self._file_limit + TRANSIENT_FILES:
            if self._segment_limit_is_smaller:
                self.failure = _too_many_segments(self._max_segments).failure
            else:
                self.failure = _too_many_artifacts(self._file_limit).failure
            return False
        return True


def _measure_directory(directory: Path) -> tuple[int, int]:
    """(total bytes, entry count) of the directory, sizes by lstat."""
    total_bytes = 0
    entry_count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                # Gone between the listing and the stat (a file the muxer renamed or removed).
                continue
            total_bytes += size
            entry_count += 1
    return total_bytes, entry_count


def _output_unreadable(error: BaseException) -> WorkerFailure:
    logger.warning("the rendition output could not be read: %s errno=%s", type(error).__name__, _errno_name(error))
    return _failure(
        "internal_error", "output_unreadable",
        f"the rendition output could not be read: {_errno_name(error)}", retryable=True,
    )


def _output_too_large(max_bytes: int) -> WorkerFailure:
    return _failure(
        INPUT_LIMITS_EXCEEDED, OUTPUT_TOO_LARGE, f"the rendition output exceeds the remaining {max_bytes} bytes"
    )


def _too_many_artifacts(max_artifacts: int) -> WorkerFailure:
    return _failure(
        INPUT_LIMITS_EXCEEDED, TOO_MANY_ARTIFACTS, f"the rendition output exceeds the remaining {max_artifacts} files"
    )


def _too_many_segments(max_segments: int) -> WorkerFailure:
    return _failure(INPUT_LIMITS_EXCEEDED, TOO_MANY_SEGMENTS, f"the rendition exceeds {max_segments} segments")


def _output_invalid(detail: str) -> WorkerFailure:
    return _failure(ENCODER_FAILED, RENDITION_OUTPUT_INVALID, detail)


def _deadline_exceeded() -> WorkerFailure:
    return _failure(DEADLINE_EXCEEDED, ENCODE_DEADLINE_EXCEEDED, "the encode ran out of time")


class _TimeBudget:
    """The rendition's own wall-time budget, for the run and the verification alike: the profile's
    base plus so much per second of admitted media, never past the caller's deadline."""

    def __init__(self, deadline: Deadline, profile: MediaProfile, source: EncodeSource) -> None:
        encode = profile.encode
        per_media_ms = -(-encode.encode_wall_ms_per_media_s * source.duration_ms // MS_PER_SECOND)
        self._budget_ms = encode.encode_base_wall_ms + per_media_ms
        self._media_ms = source.duration_ms
        self._caller = deadline
        self.deadline = deadline.within_ms(self._budget_ms)

    def failure(self) -> WorkerFailure:
        """Running out of time: the caller's deadline when that has passed, else this budget."""
        if self._caller.expired():
            return _deadline_exceeded()
        return _failure(
            DEADLINE_EXCEEDED,
            ENCODE_BUDGET_EXCEEDED,
            f"the rendition's encode exceeded its time budget of {self._budget_ms} ms "
            f"for {self._media_ms} ms of admitted media",
        )

    def require_time_left(self) -> None:
        if self.deadline.expired():
            raise self.failure()


class _VerificationClock:
    """Checks the time budget before each file, after every DEADLINE_CHECK_BYTES counted across the
    whole verification rather than per file, and whenever the verification asks."""

    def __init__(self, budget: _TimeBudget) -> None:
        self._budget = budget
        self._unchecked_bytes = 0

    def check(self) -> None:
        self._budget.require_time_left()

    def count(self, read_bytes: int) -> None:
        self._unchecked_bytes += read_bytes
        if self._unchecked_bytes >= DEADLINE_CHECK_BYTES:
            self._unchecked_bytes = 0
            self._budget.require_time_left()


def _setup_failure(error: OSError) -> WorkerFailure:
    logger.warning("the encode could not be set up: %s errno=%s", type(error).__name__, _errno_name(error))
    return _failure(
        "resource_exhausted", "encode_setup_failed",
        f"the encode could not be set up: {_errno_name(error)}", retryable=True,
    )


def _encoder_start_failure(error: OSError) -> WorkerFailure:
    """A broken ffmpeg installation is permanent; any other start failure is worth a retry."""
    logger.warning("ffmpeg could not be started: %s errno=%s", type(error).__name__, _errno_name(error))
    if isinstance(error, (FileNotFoundError, PermissionError)) or error.errno in ENCODER_UNUSABLE_ERRNOS:
        return _failure("configuration_error", "ffmpeg_unavailable", "ffmpeg could not be started")
    return _failure(
        "resource_exhausted", "encoder_start_failed",
        f"ffmpeg could not be started: {_errno_name(error)}", retryable=True,
    )


def _encoder_thread_failure() -> WorkerFailure:
    logger.warning("ffmpeg could not be started: a reader thread did not start")
    return _failure(
        "resource_exhausted", "encoder_start_failed",
        f"ffmpeg could not be started: {UNKNOWN_ERRNO_NAME}", retryable=True,
    )


def _require_directory(name: str, directory: object) -> Path:
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ValueError(f"{name} must be an absolute Path")
    if any(character in str(directory) for character in UNSAFE_PATH_CHARACTERS):
        raise ValueError(f"{name} holds a character the command line would interpret")
    if not directory.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return directory.resolve()


def _validate_arguments(
    source: object,
    plan: object,
    profile: object,
    output_root: object,
    work_dir: object,
    media_key: object,
    allowance: object,
    deadline: object,
    ffmpeg_path: object,
) -> None:
    if not isinstance(source, EncodeSource):
        raise TypeError("source must be an EncodeSource")
    if not isinstance(plan, RenditionPlan):
        raise TypeError("plan must be a RenditionPlan")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    if not isinstance(allowance, OutputAllowance):
        raise TypeError("allowance must be an OutputAllowance")
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if media_key is not None and (not isinstance(media_key, bytes) or len(media_key) != KEY_BYTES):
        raise ValueError(f"media_key must be None or {KEY_BYTES} bytes")
    if ffmpeg_path is not None and (not isinstance(ffmpeg_path, str) or not ffmpeg_path):
        raise ValueError("ffmpeg_path must be None or a non-empty string")
    # The plan's name becomes a directory name: only a contract rendition name may.
    if plan.name not in RENDITION_NAMES:
        raise ValueError(f"the plan's name must be one of {RENDITION_NAMES}")
    resolved_output = _require_directory("output_root", output_root)
    resolved_work = _require_directory("work_dir", work_dir)
    if resolved_output.is_relative_to(resolved_work) or resolved_work.is_relative_to(resolved_output):
        raise ValueError("output_root and work_dir must not lie inside one another")
    if os.path.lexists(resolved_output / plan.name):
        raise ValueError("the rendition directory already exists")


def _require_room(allowance: OutputAllowance) -> None:
    if allowance.max_bytes < MIN_OUTPUT_BYTES:
        raise _output_too_large(allowance.max_bytes)
    if allowance.max_artifacts < MIN_ARTIFACTS:
        raise _too_many_artifacts(allowance.max_artifacts)


def _resolve_ffmpeg(ffmpeg_path: str | None) -> str:
    candidate = ffmpeg_path or os.environ.get("FFMPEG_PATH") or "ffmpeg"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise _failure("configuration_error", "ffmpeg_unavailable", "ffmpeg is not installed")
    return resolved


@contextlib.contextmanager
def _key_info(media_key: bytes | None, work_dir: Path) -> Iterator[KeyInfo | None]:
    if media_key is None:
        yield None
        return
    with hls_key_info(media_key, parent=work_dir) as key_info:
        yield key_info


def _run_encoder(
    executable: str,
    source: EncodeSource,
    plan: RenditionPlan,
    profile: MediaProfile,
    *,
    rendition_dir: Path,
    work_dir: Path,
    media_key: bytes | None,
    monitor: _OutputMonitor,
    deadline: Deadline,
) -> BoundedResult:
    # The empty working directory and the key files live only while ffmpeg runs. The start-failure
    # mappings inside raise WorkerFailure, which the setup except lets through.
    try:
        with (
            tempfile.TemporaryDirectory(prefix=EMPTY_CWD_PREFIX, dir=work_dir, ignore_cleanup_errors=True) as empty_dir,
            _key_info(media_key, work_dir) as key_info,
        ):
            argv = rendition_argv(
                executable,
                source,
                plan,
                profile,
                output_dir=rendition_dir,
                key_info_path=None if key_info is None else key_info.path,
            )
            try:
                result = run_bounded(
                    argv,
                    deadline=deadline,
                    max_stdout_bytes=MAX_STDOUT_BYTES,
                    max_stderr_bytes=profile.encode.max_stderr_bytes,
                    env=FFMPEG_ENV,
                    cwd=empty_dir,
                    monitor=monitor,
                    monitor_interval_ms=profile.encode.output_check_interval_ms,
                    pass_fds=() if key_info is None else key_info.pass_fds,
                    stderr_patterns=FRAME_REFUSAL_PATTERNS,
                )
            except OSError as error:
                # Also where ffmpeg's output could not be fully read; the group is dead by then.
                raise _encoder_start_failure(error) from None
            except RuntimeError:
                raise _encoder_thread_failure() from None
    except OSError as error:
        raise _setup_failure(error) from None
    return result


def _scratch_full() -> WorkerFailure:
    return _failure("resource_exhausted", "scratch_full", "the scratch disk is full", retryable=True)


def _disk_full(directory: Path) -> bool:
    """Whether less than MIN_FREE_BYTES is free beside `directory`; False when that cannot be read."""
    try:
        usage = os.statvfs(directory)
    except OSError:
        return False
    return usage.f_bavail * usage.f_frsize < MIN_FREE_BYTES


def _scratch_full_or(failure: WorkerFailure, directory: Path) -> WorkerFailure:
    """A full scratch disk explains a write that failed or that ffmpeg did not report; otherwise
    `failure` stands, and so it does when the free space cannot be read."""
    return _scratch_full() if _disk_full(directory) else failure


def _raise_for_ending(
    result: BoundedResult, monitor: _OutputMonitor, rendition_dir: Path, budget: _TimeBudget
) -> None:
    """Raise the typed failure of every ending but a clean exit. ffmpeg's own words are never forwarded."""
    if result.timed_out:
        raise budget.failure()
    if result.stopped_by_monitor:
        raise WorkerFailure(monitor.failure or MONITOR_STOP_WITHOUT_REASON)
    if result.stdout_overflow:
        raise _failure("internal_error", "encoder_output_unexpected", "ffmpeg wrote more than expected to stdout")
    returncode = result.returncode
    if returncode is None or returncode < 0:
        raise _failure("internal_error", "encoder_crashed", "ffmpeg ended on a signal", retryable=True)
    if returncode in SHELL_CANNOT_RUN_EXIT_CODES:
        raise _failure("configuration_error", "ffmpeg_unavailable", "ffmpeg could not be started")
    if result.stderr_patterns_seen:
        raise _failure(
            INPUT_LIMITS_EXCEEDED, "decoded_frame_too_large", "a decoded frame exceeds the profile's pixel limit"
        )
    if returncode > 0 and NO_SPACE_MARKER in result.stderr_tail:
        raise _scratch_full()
    if returncode > 0:
        raise _scratch_full_or(
            _failure(ENCODER_FAILED, "encode_failed", f"ffmpeg exited with status {returncode}"), rendition_dir
        )


def _read_playlist(rendition_dir: Path, profile: MediaProfile, *, encrypted: bool) -> MediaPlaylist:
    try:
        return read_media_playlist(
            rendition_dir / PLAYLIST_NAME,
            max_bytes=profile.encode.max_playlist_bytes,
            encrypted=encrypted,
            key_uri=KEY_URI_PLACEHOLDER,
            max_segments=profile.encode.max_segments_per_rendition,
        )
    except PlaylistInvalid as invalid:
        if invalid.code == TOO_MANY_SEGMENTS:
            raise _too_many_segments(profile.encode.max_segments_per_rendition) from None
        raise _output_invalid(f"the rendition's playlist is refused: {invalid.detail}") from None
    except OSError as error:
        raise _output_unreadable(error) from None


def _listed_sizes(rendition_dir: Path, names: Sequence[str]) -> dict[str, int]:
    """The size of every file, when the directory holds exactly `names`, each a regular file."""
    try:
        with os.scandir(rendition_dir) as entries:
            found = {entry.name: entry.stat(follow_symlinks=False) for entry in entries}
    except OSError as error:
        raise _output_invalid_or_unreadable(error) from None
    if set(found) != set(names):
        raise _output_invalid("the rendition directory does not hold exactly the files its playlist names")
    if not all(stat.S_ISREG(info.st_mode) for info in found.values()):
        raise _output_invalid("a rendition file is not a regular file")
    return {name: found[name].st_size for name in names}


def _output_invalid_or_unreadable(error: OSError) -> WorkerFailure:
    # A file that vanished after the listing was not ffmpeg's finished output.
    if isinstance(error, FileNotFoundError):
        return _output_invalid("a rendition file vanished while it was verified")
    return _output_unreadable(error)


def _require_within_limits(sizes: dict[str, int], allowance: OutputAllowance) -> None:
    if sum(sizes.values()) > allowance.max_bytes:
        raise _output_too_large(allowance.max_bytes)
    if len(sizes) > allowance.max_artifacts:
        raise _too_many_artifacts(allowance.max_artifacts)
    if any(size > MAX_ARTIFACT_BYTES for size in sizes.values()):
        raise _failure(
            INPUT_LIMITS_EXCEEDED, "artifact_too_large", f"a rendition file exceeds {MAX_ARTIFACT_BYTES} bytes"
        )


def _measured_duration_ms(playlist: MediaPlaylist) -> int:
    total = sum((segment.duration for segment in playlist.segments), Fraction(0))
    return math.ceil(total * MS_PER_SECOND)


def _require_cut_held(duration_ms: int, source: EncodeSource, plan: RenditionPlan) -> None:
    frame_ms = math.ceil(Fraction(MS_PER_SECOND * plan.frame_rate_den, plan.frame_rate_num))
    limit_ms = source.duration_ms + frame_ms + DURATION_SLACK_MS
    if duration_ms > limit_ms:
        raise _failure(
            ENCODER_FAILED, "duration_cut_failed",
            f"the rendition lasts {duration_ms} ms, beyond the admitted {source.duration_ms} ms",
        )


def _require_bandwidth_in_range(*bandwidths: int) -> None:
    if any(bandwidth > MAX_BANDWIDTH_BPS for bandwidth in bandwidths):
        raise _failure(
            ENCODER_FAILED, "bandwidth_out_of_range", f"the rendition's bandwidth exceeds {MAX_BANDWIDTH_BPS} bps"
        )


SEGMENT_NOT_TS = "a segment is incomplete or not MPEG-TS"
SEGMENT_SPAN_MISMATCH = "a segment's video does not cover its declared duration"
SEGMENT_FRAMES_MISSING = "a segment's video is missing frames within its span"
NOT_DECODING_CLEANLY = "the rendition does not decode cleanly"


def _pes_start_pts(packet: bytes) -> int | None:
    """The PTS of the video PES packet this TS packet opens, or None when it opens none or has no PTS."""
    adaptation = (packet[3] >> 4) & 0b11
    if adaptation not in (TS_PAYLOAD_ONLY, TS_ADAPTATION_FIELD_AND_PAYLOAD):
        return None
    payload = TS_HEADER_BYTES + (1 + packet[TS_HEADER_BYTES] if adaptation == TS_ADAPTATION_FIELD_AND_PAYLOAD else 0)
    header = packet[payload: payload + PES_HEADER_WITH_PTS_BYTES]
    if len(header) < PES_HEADER_WITH_PTS_BYTES or header[:3] != PES_START_CODE:
        return None
    if not FIRST_VIDEO_STREAM_ID <= header[3] <= LAST_VIDEO_STREAM_ID or header[7] >> 6 not in PES_PTS_PRESENT:
        return None
    return (
        ((header[9] >> 1) & 0b111) << 30
        | header[10] << 22
        | (header[11] >> 1) << 15
        | header[12] << 7
        | header[13] >> 1
    )


class _TransportStreamCheck:
    """Whether a stream of bytes is whole MPEG-TS packets, each opening with the sync byte, whose
    video presentation timestamps span its declared duration.

    The bytes arrive in chunks of any size; whole packets are scanned with C-level byte operations,
    and only the packets that open a PES packet are read one by one.
    """

    def __init__(self, declared_duration: Fraction, frame_duration: Fraction) -> None:
        self._declared_duration = declared_duration
        self._frame_duration = frame_duration
        self._pending = b""
        self._total_bytes = 0
        self._synced = True
        self._min_pts: int | None = None
        self._max_pts: int | None = None
        self.video_frames = 0  # the video PES packets that carry a PTS: one per frame

    def feed(self, data: bytes) -> None:
        self._total_bytes += len(data)
        if not self._synced or not data:
            return
        buffer = self._pending + data if self._pending else data
        whole = len(buffer) - len(buffer) % TS_PACKET_BYTES
        self._pending = buffer[whole:]
        if whole:
            self._scan(buffer[:whole])

    def _scan(self, packets: bytes) -> None:
        sync_bytes = packets[::TS_PACKET_BYTES]
        if sync_bytes.count(TS_SYNC_BYTE) != len(sync_bytes):
            self._synced = False
            return
        starts = packets[1::TS_PACKET_BYTES].translate(PAYLOAD_UNIT_START_TABLE)
        index = starts.find(1)
        while index != -1:
            offset = index * TS_PACKET_BYTES
            pts = _pes_start_pts(packets[offset: offset + TS_PACKET_BYTES])
            if pts is not None:
                self.video_frames += 1
                self._min_pts = pts if self._min_pts is None else min(self._min_pts, pts)
                self._max_pts = pts if self._max_pts is None else max(self._max_pts, pts)
            index = starts.find(1, index + 1)

    def problem(self) -> str | None:
        """Why the stream fails the check, in our words; None when it passes."""
        if not self._synced or self._pending or self._total_bytes == 0:
            return SEGMENT_NOT_TS
        if self._min_pts is None or self._max_pts is None:
            return SEGMENT_SPAN_MISMATCH
        span_ticks = self._max_pts - self._min_pts
        if span_ticks > MAX_PTS_SPAN_TICKS:
            return SEGMENT_SPAN_MISMATCH
        span = Fraction(span_ticks, PTS_CLOCK_HZ) + self._frame_duration
        tolerance = SPAN_TOLERANCE_FRAMES * self._frame_duration + SPAN_TOLERANCE
        if abs(span - self._declared_duration) > tolerance:
            return SEGMENT_SPAN_MISMATCH
        # A frame lost from the middle of the span (a B-frame last in decode order) leaves the
        # span as it was; the frame count does not.
        if self.video_frames != round(Fraction(span_ticks, PTS_CLOCK_HZ) / self._frame_duration) + 1:
            return SEGMENT_FRAMES_MISSING
        return None


class _EncryptedTransportStreamCheck:
    """The same check behind AES-128-CBC: valid PKCS#7 padding, decrypted with the media key and the
    segment's IV (its media sequence number)."""

    def __init__(self, media_key: bytes, sequence_number: int, plain: _TransportStreamCheck) -> None:
        iv = sequence_number.to_bytes(AES_BLOCK_BYTES, "big")
        self._decryptor = Cipher(algorithms.AES(media_key), modes.CBC(iv)).decryptor()
        self._unpadder = padding.PKCS7(AES_BLOCK_BITS).unpadder()
        self._plain = plain
        self._ciphertext_bytes = 0

    def feed(self, data: bytes) -> None:
        self._ciphertext_bytes += len(data)
        self._plain.feed(self._unpadder.update(self._decryptor.update(data)))

    def problem(self) -> str | None:
        if self._ciphertext_bytes == 0 or self._ciphertext_bytes % AES_BLOCK_BYTES != 0:
            return SEGMENT_NOT_TS
        try:
            remainder = self._unpadder.update(self._decryptor.finalize()) + self._unpadder.finalize()
        except ValueError:
            # Invalid padding: the segment does not end where its encryption ended.
            return SEGMENT_NOT_TS
        self._plain.feed(remainder)
        return self._plain.problem()


_SegmentCheck = _TransportStreamCheck | _EncryptedTransportStreamCheck


def _segment_check(
    media_key: bytes | None, sequence_number: int, segment: PlaylistSegment, plan: RenditionPlan
) -> tuple[_SegmentCheck, _TransportStreamCheck]:
    """The check a segment's bytes go through, and the plain check behind it that counts frames."""
    frame_duration = Fraction(plan.frame_rate_den, plan.frame_rate_num)
    plain = _TransportStreamCheck(segment.duration, frame_duration)
    if media_key is None:
        return plain, plain
    return _EncryptedTransportStreamCheck(media_key, sequence_number, plain), plain


def _digest(path: Path, expected_size: int, *, check: _SegmentCheck | None, clock: _VerificationClock) -> str:
    """The sha256 of the regular file at `path`, which must still hold `expected_size` bytes and,
    with `check`, pass it; one read pass, which the time budget bounds."""
    clock.check()
    try:
        descriptor = os.open(path, FILE_OPEN_FLAGS)
    except OSError as error:
        raise _output_invalid_or_unreadable(error) from None
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _output_invalid("a rendition file is not a regular file")
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)
            if check is not None:
                check.feed(chunk)
            read_bytes += len(chunk)
            clock.count(len(chunk))
    except OSError as error:
        raise _output_unreadable(error) from None
    finally:
        os.close(descriptor)
    if read_bytes != expected_size:
        raise _output_invalid("a rendition file changed while it was verified")
    problem = None if check is None else check.problem()
    if problem is not None:
        raise _output_invalid(problem)
    return digest.hexdigest()


def _decimal_text(value: Fraction) -> str:
    """The exact decimal text of `value`, which was parsed from decimal text."""
    digits = 0
    scaled = value
    while scaled.denominator != 1:
        if digits == MAX_DECIMAL_DIGITS:
            raise ValueError("the value has no short decimal form")
        scaled *= DECIMAL_BASE
        digits += 1
    whole, fraction = divmod(scaled.numerator, DECIMAL_BASE**digits)
    return f"{whole}.{fraction:0{digits}d}" if digits else f"{whole}.0"


def _check_playlist(playlist: MediaPlaylist, rendition_dir: Path, key_path: Path | None) -> str:
    """A playlist naming the rendition's segments by absolute path, and when encrypted its key by
    the path the decoder reads it from; built from the parsed playlist, never from its text."""
    if key_path is not None:
        require_safe_path("key_path", key_path)
        if KEY_PATH_FORBIDDEN in str(key_path):
            raise ValueError("key_path holds a character the key line cannot carry")
    lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{CHECK_PLAYLIST_VERSION}",
        f"#EXT-X-TARGETDURATION:{playlist.target_duration_s}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    for sequence_number, segment in enumerate(playlist.segments):
        segment_path = rendition_dir / segment.name
        require_safe_path("segment path", segment_path)
        if key_path is not None:
            lines.append(f'#EXT-X-KEY:METHOD=AES-128,URI="{key_path}",IV=0x{sequence_number:032x}')
        lines.append(f"#EXTINF:{_decimal_text(segment.duration)},")
        lines.append(str(segment_path))
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def _check_argv(executable: str, check_playlist: Path, *, encrypted: bool) -> list[str]:
    # The HLS demuxer refuses to open a key whose name lacks a media extension, and the key is
    # /proc/self/fd/N or a private file (ffmpeg 7.1 and 8 alike). The check playlist is built from
    # parsed segment names and our own key path, so it names nothing else the filter could stop.
    key_extensions = ["-allowed_extensions", "ALL"] if encrypted else []
    return [
        executable,
        "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-xerror",
        "-protocol_whitelist", "crypto,file" if encrypted else "file",
        "-format_whitelist", "hls,mpegts",
        "-f", "hls",
        *key_extensions,
        # The decoders' own errors are fatal too: without this, H.264 conceals a frame cut short at
        # the very end of the rendition, prints an error and still exits 0.
        "-err_detect", "explode",
        "-i", str(check_playlist),
        # Audio is decoded too, so an audio packet cut short is caught.
        "-map", "0:v:0", "-map", "0:a:0?",
        "-fps_mode", "passthrough",
        "-f", "null",
        "-progress", "pipe:1", "-stats_period", CHECK_PROGRESS_PERIOD_S,
        "-",
    ]


def _check_output_invalid() -> WorkerFailure:
    return _failure("internal_error", "self_check_output_invalid", "the self-decode's progress is not understood")


def _decoded_frames(progress: bytes) -> int:
    """The video frames decoded: the last `frame=` before `progress=end` in ffmpeg's progress output."""
    try:
        text = progress.decode("ascii")
    except UnicodeDecodeError:
        raise _check_output_invalid() from None
    frames: int | None = None
    for line in text.splitlines():
        if line.startswith(PROGRESS_FRAME_PREFIX):
            value = line[len(PROGRESS_FRAME_PREFIX):]
            if FRAME_COUNT_PATTERN.fullmatch(value) is None:
                raise _check_output_invalid()
            frames = int(value)
        elif line == PROGRESS_END:
            if frames is None:
                raise _check_output_invalid()
            return frames
    raise _check_output_invalid()


def _raise_for_check_ending(result: BoundedResult, budget: _TimeBudget) -> None:
    if result.timed_out:
        raise budget.failure()
    if result.stdout_overflow:
        raise _check_output_invalid()
    returncode = result.returncode
    if returncode is None or returncode < 0:
        raise _failure("internal_error", "self_check_crashed", "the self-decode ended on a signal", retryable=True)
    if returncode in SHELL_CANNOT_RUN_EXIT_CODES:
        raise _failure("configuration_error", "ffmpeg_unavailable", "ffmpeg could not be started")
    if returncode != 0:
        raise _output_invalid(NOT_DECODING_CLEANLY)


def _self_decode(
    executable: str,
    playlist: MediaPlaylist,
    rendition_dir: Path,
    *,
    work_dir: Path,
    media_key: bytes | None,
    profile: MediaProfile,
    budget: _TimeBudget,
    expected_frames: int,
) -> None:
    """Decode the whole rendition once, every error fatal, and require exactly `expected_frames`
    video frames. The check playlist and the key live in a private directory only while it runs."""
    # The start-failure mappings inside raise WorkerFailure, which the setup except lets through.
    try:
        with (
            tempfile.TemporaryDirectory(prefix=CHECK_DIR_PREFIX, dir=work_dir, ignore_cleanup_errors=True) as directory,
            _key_info(media_key, Path(directory)) as key_info,
        ):
            check_dir = Path(directory)
            empty_cwd = check_dir / CHECK_CWD_NAME
            empty_cwd.mkdir()
            check_playlist = check_dir / CHECK_PLAYLIST_NAME
            key_path = None if key_info is None else key_info.key_path
            check_playlist.write_text(_check_playlist(playlist, rendition_dir, key_path), encoding="ascii")
            try:
                result = run_bounded(
                    _check_argv(executable, check_playlist, encrypted=media_key is not None),
                    deadline=budget.deadline,
                    max_stdout_bytes=CHECK_MAX_STDOUT_BYTES,
                    max_stderr_bytes=profile.encode.max_stderr_bytes,
                    env=FFMPEG_ENV,
                    cwd=str(empty_cwd),
                    pass_fds=() if key_info is None else key_info.pass_fds,
                )
            except OSError as error:
                raise _encoder_start_failure(error) from None
            except RuntimeError:
                raise _encoder_thread_failure() from None
    except OSError as error:
        raise _setup_failure(error) from None
    _raise_for_check_ending(result, budget)
    if _decoded_frames(result.stdout) != expected_frames:
        raise _output_invalid(NOT_DECODING_CLEANLY)


def _verified_output(
    source: EncodeSource,
    plan: RenditionPlan,
    profile: MediaProfile,
    rendition_dir: Path,
    allowance: OutputAllowance,
    *,
    executable: str,
    work_dir: Path,
    media_key: bytes | None,
    budget: _TimeBudget,
) -> RenditionOutput:
    clock = _VerificationClock(budget)
    playlist = _read_playlist(rendition_dir, profile, encrypted=media_key is not None)
    segment_names = [segment.name for segment in playlist.segments]
    names = [PLAYLIST_NAME, *segment_names]
    sizes = _listed_sizes(rendition_dir, names)
    if any(sizes[name] < MIN_SEGMENT_BYTES for name in segment_names):
        raise _output_invalid("a segment is empty")
    _require_within_limits(sizes, allowance)

    duration_ms = _measured_duration_ms(playlist)
    _require_cut_held(duration_ms, source, plan)
    measurements = [(segment.duration, sizes[segment.name]) for segment in playlist.segments]
    peak = peak_bandwidth_bps(measurements, playlist.target_duration_s)
    average = average_bandwidth_bps(measurements)
    _require_bandwidth_in_range(peak, average)

    artifacts = [
        ArtifactFile(
            path=f"{plan.name}/{PLAYLIST_NAME}",
            kind=ARTIFACT_KIND_PLAYLIST,
            size_bytes=sizes[PLAYLIST_NAME],
            sha256=_digest(rendition_dir / PLAYLIST_NAME, sizes[PLAYLIST_NAME], check=None, clock=clock),
        )
    ]
    video_frames = 0
    for sequence_number, segment in enumerate(playlist.segments):
        check, frame_counter = _segment_check(media_key, sequence_number, segment, plan)
        digest = _digest(rendition_dir / segment.name, sizes[segment.name], check=check, clock=clock)
        video_frames += frame_counter.video_frames
        artifacts.append(
            ArtifactFile(
                path=f"{plan.name}/{segment.name}",
                kind=ARTIFACT_KIND_SEGMENT,
                size_bytes=sizes[segment.name],
                sha256=digest,
            )
        )
    clock.check()
    _self_decode(
        executable,
        playlist,
        rendition_dir,
        work_dir=work_dir,
        media_key=media_key,
        profile=profile,
        budget=budget,
        expected_frames=video_frames,
    )
    clock.check()
    return RenditionOutput(
        name=plan.name,
        width=plan.width,
        height=plan.height,
        frame_rate_num=plan.frame_rate_num,
        frame_rate_den=plan.frame_rate_den,
        codecs=plan.codecs,
        has_audio=plan.audio_bitrate is not None,
        encrypted=media_key is not None,
        playlist_path=f"{plan.name}/{PLAYLIST_NAME}",
        segment_count=len(segment_names),
        duration_ms=duration_ms,
        bandwidth_bps=peak,
        average_bandwidth_bps=average,
        total_bytes=sum(sizes.values()),
        artifacts=tuple(artifacts),
    )


def _remove_rendition(rendition_dir: Path) -> None:
    """Best effort: the job's scratch cleanup removes whatever is left. Paths are never logged.

    (No rmtree error callback: the worker image runs Python 3.10, which lacks `onexc`.)
    """
    try:
        shutil.rmtree(rendition_dir)
    except FileNotFoundError:
        return
    except OSError as error:
        logger.warning("a failed rendition's directory could not be removed: errno=%s", _errno_name(error))


def encode_rendition(
    source: EncodeSource,
    plan: RenditionPlan,
    *,
    profile: MediaProfile,
    output_root: Path,
    work_dir: Path,
    media_key: bytes | None,
    allowance: OutputAllowance,
    deadline: Deadline,
    ffmpeg_path: str | None = None,
) -> RenditionOutput:
    """Encode `source` into the rendition `plan` names, under `output_root / plan.name`, and verify it.

    `media_key` (16 bytes) encrypts every segment with AES-128; None leaves them plain. The key
    files and ffmpeg's empty working directory live under `work_dir` only while ffmpeg runs. The run
    and the verification together get the profile's time budget for the admitted duration, never
    past `deadline`.
    Raises WorkerFailure for every ending but a verified rendition, after removing the rendition
    directory; raises TypeError or ValueError for invalid arguments before anything runs.
    """
    _validate_arguments(source, plan, profile, output_root, work_dir, media_key, allowance, deadline, ffmpeg_path)
    _require_room(allowance)
    if deadline.expired():
        raise _deadline_exceeded()
    budget = _TimeBudget(deadline, profile, source)
    executable = _resolve_ffmpeg(ffmpeg_path)

    rendition_dir = output_root / plan.name
    try:
        rendition_dir.mkdir()
    except OSError as error:
        raise _setup_failure(error) from None
    # A rendition that fails for any reason leaves nothing behind, not even a partial directory.
    try:
        monitor = _OutputMonitor(rendition_dir, allowance, profile)
        result = _run_encoder(
            executable,
            source,
            plan,
            profile,
            rendition_dir=rendition_dir,
            work_dir=work_dir,
            media_key=media_key,
            monitor=monitor,
            deadline=budget.deadline,
        )
        _raise_for_ending(result, monitor, rendition_dir, budget)
        budget.require_time_left()
        # ffmpeg's HLS muxer can drop a segment's write error and exit 0: a full disk is checked
        # after every clean run, before the output is read.
        if _disk_full(rendition_dir):
            raise _scratch_full()
        try:
            return _verified_output(
                source,
                plan,
                profile,
                rendition_dir,
                allowance,
                executable=executable,
                work_dir=work_dir,
                media_key=media_key,
                budget=budget,
            )
        except WorkerFailure as failure:
            # Running out of time is not something the output shows; every other refusal may be.
            if failure.failure.error_class == DEADLINE_EXCEEDED:
                raise
            raise _scratch_full_or(failure, rendition_dir) from None
    except BaseException:
        _remove_rendition(rendition_dir)
        raise
