"""The package around a job's renditions: the HLS master playlist, the generation manifest, and the
room the job's output allowance keeps for them and for the thumbnail.

The master playlist lists the renditions in the planned order (largest first) with the bandwidth
measured from their segments on disk (RFC 8216 4.3.4.2), never a configured bitrate. The generation
manifest is contract v2's `generation.manifest` 1.0.0-draft document: it names every artifact the
job wrote (the master, the thumbnail or the diagnostic that says why there is none, every rendition's
playlist and segments) with its size and SHA-256. It is written last and is never an artifact
itself. Its preconditions fail only on a wiring bug, never on media: every media fact was proven by
the stage that produced it. Both files are created new, never through a link and never over an
existing file, and a failed write leaves nothing behind.

`reserve_package` sets aside what the master, the thumbnail and the manifest may take, so the
renditions can never use their room.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from core.encode import (
    ARTIFACT_KIND_MASTER,
    ARTIFACT_KIND_PLAYLIST,
    ARTIFACT_KIND_SEGMENT,
    ARTIFACT_KIND_THUMBNAIL,
    SHA256_PATTERN,
    ArtifactFile,
    OutputAllowance,
    RenditionOutput,
    _errno_name,
    _failure,
    _require_directory,
    _require_int,
    _scratch_full,
)
from core.encode_command import PLAYLIST_NAME
from core.failure import Diagnostic, WorkerFailure
from core.profile import PROFILE_ID_PATTERN, PROFILE_VERSION_MAX, PROFILE_VERSION_MIN, RENDITION_NAMES, MediaProfile
from core.thumbnail import THUMBNAIL_PATH, THUMBNAIL_UNAVAILABLE, ThumbnailResult

logger = logging.getLogger(__name__)

MASTER_PLAYLIST_PATH = "master.m3u8"
# Seven renditions with every field at its widest take 1,442 bytes.
MASTER_PLAYLIST_MAX_BYTES = 4096
MASTER_PLAYLIST_VERSION = 3
MAX_RENDITIONS = len(RENDITION_NAMES)
# Contract v2's codecs: a video codec, then optionally an audio codec, in at most 64 characters.
CODECS_PATTERN = re.compile(r"[A-Za-z0-9.]{1,32}(,[A-Za-z0-9.]{1,32})?")
CODECS_MAX_CHARS = 64
# FRAME-RATE carries three decimals, rounded half up.
FRAME_RATE_SCALE = 1000
HALF = Fraction(1, 2)

MANIFEST_PATH = "generation-manifest.json"
MANIFEST_VERSION = "1.0.0-draft"
DOCUMENT_KIND = "generation.manifest"
CONTRACT_MANIFEST_MAX_BYTES = 8_388_608  # limits.json max_bytes["generation.manifest"]
CONTRACT_MAX_ARTIFACTS = 25_209
CONTRACT_MAX_TOTAL_BYTES = 1_099_511_627_776
CONTRACT_MAX_DURATION_MS = 21_600_000
CONTRACT_MAX_DIAGNOSTICS = 8
# Everything but the artifact list, worst case: seven renditions, eight diagnostics of 200 escaped
# characters, every field at its longest; 6,287 bytes.
MANIFEST_HEADER_MAX_BYTES = 16 * 1024
# One artifact entry of this writer, worst case: a media playlist at the profile's 1 MiB cap with its
# rendition, 176 bytes with its comma (a 4 GiB segment takes 174).
MANIFEST_ARTIFACT_MAX_BYTES = 256
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
ENCRYPTION_AES_128 = "aes-128"
ENCRYPTION_NONE = "none"
ENCRYPTION_MODES = frozenset({ENCRYPTION_AES_128, ENCRYPTION_NONE})
# S7: every identifier but video_uuid names another entity, so none may equal video_uuid.
NOT_VIDEO_UUID_FIELDS = ("attempt_id", "dispatch_id", "execution_id", "source_id", "org_uuid")
# The master playlist and the thumbnail; the manifest is not an artifact.
PACKAGE_ARTIFACTS = 2
MIN_RENDITION_ARTIFACTS = 2  # its playlist and one segment
NEW_FILE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
NEW_FILE_MODE = 0o644
DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
MICROSECONDS_PER_MILLISECOND = 1000


def _require_pattern(name: str, value: object, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is malformed: {value!r}")


@dataclass(frozen=True, slots=True)
class ManifestIdentity:
    """The manifest's identity block: lowercase UUIDs; none but video_uuid equals video_uuid (S7)."""

    org_uuid: str
    video_uuid: str
    source_id: str
    attempt_id: str
    dispatch_id: str
    execution_id: str

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_pattern(field.name, getattr(self, field.name), UUID_PATTERN)
        for name in NOT_VIDEO_UUID_FIELDS:
            if getattr(self, name) == self.video_uuid:
                raise ValueError(f"{name} must not equal video_uuid")


@dataclass(frozen=True, slots=True)
class ProfileRef:
    """The dispatch's profile reference, echoed in the manifest (S19)."""

    id: str
    version: int
    sha256: str

    def __post_init__(self) -> None:
        _require_pattern("id", self.id, PROFILE_ID_PATTERN)
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("version must be an int")
        if not PROFILE_VERSION_MIN <= self.version <= PROFILE_VERSION_MAX:
            raise ValueError(f"version must be from {PROFILE_VERSION_MIN} to {PROFILE_VERSION_MAX}")
        _require_pattern("sha256", self.sha256, SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class EncryptionRef:
    """The dispatch's encryption: "aes-128" with its media key id, or "none" without one (S20)."""

    mode: str
    media_key_id: str | None

    def __post_init__(self) -> None:
        if self.mode not in ENCRYPTION_MODES:
            raise ValueError(f"mode must be one of {sorted(ENCRYPTION_MODES)}, got {self.mode!r}")
        if self.mode == ENCRYPTION_NONE:
            if self.media_key_id is not None:
                raise ValueError("an unencrypted job has no media key id")
            return
        _require_pattern("media_key_id", self.media_key_id, UUID_PATTERN)

    @property
    def encrypted(self) -> bool:
        return self.mode == ENCRYPTION_AES_128


@dataclass(frozen=True, slots=True)
class ManifestOutput:
    """What `transcode.result.completed`'s manifest locator needs, but its location id."""

    path: str
    sha256: str
    size_bytes: int
    artifact_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        if self.path != MANIFEST_PATH:
            raise ValueError(f"path must be {MANIFEST_PATH}")
        _require_pattern("sha256", self.sha256, SHA256_PATTERN)
        for name in ("size_bytes", "artifact_count", "total_bytes"):
            _require_int(name, getattr(self, name), 1)


def _require_rendition_fields(output: RenditionOutput) -> None:
    if output.name not in RENDITION_NAMES:
        raise ValueError(f"a rendition's name must be one of {RENDITION_NAMES}")
    if output.playlist_path != f"{output.name}/{PLAYLIST_NAME}":
        raise ValueError(f"rendition {output.name}'s playlist path is not {output.name}/{PLAYLIST_NAME}")
    _require_pattern("codecs", output.codecs, CODECS_PATTERN)
    if len(output.codecs) > CODECS_MAX_CHARS:
        raise ValueError(f"codecs must be at most {CODECS_MAX_CHARS} characters")
    for name in ("width", "height", "bandwidth_bps", "average_bandwidth_bps", "duration_ms"):
        _require_int(name, getattr(output, name), 1)


def _validated_renditions(renditions: object) -> tuple[RenditionOutput, ...]:
    """1 to 7 renditions of unique contract names that agree on audio and on encryption."""
    if isinstance(renditions, (str, bytes)) or not isinstance(renditions, Sequence):
        raise ValueError("renditions must be a sequence of RenditionOutput")
    outputs = tuple(renditions)
    if not 1 <= len(outputs) <= MAX_RENDITIONS:
        raise ValueError(f"there must be 1 to {MAX_RENDITIONS} renditions, got {len(outputs)}")
    if not all(isinstance(output, RenditionOutput) for output in outputs):
        raise TypeError("renditions must hold RenditionOutput entries")
    names = [output.name for output in outputs]
    if len(set(names)) != len(names):
        raise ValueError("renditions hold a duplicate name")
    for output in outputs:
        _require_rendition_fields(output)
    if len({output.has_audio for output in outputs}) != 1:
        raise ValueError("the renditions disagree on audio")
    if len({output.encrypted for output in outputs}) != 1:
        raise ValueError("the renditions disagree on encryption")
    return outputs


def _frame_rate_text(output: RenditionOutput) -> str:
    """The constant output frame rate rounded half up to three decimals: 30000/1001 is "29.970"."""
    scaled = math.floor(Fraction(output.frame_rate_num * FRAME_RATE_SCALE, output.frame_rate_den) + HALF)
    return f"{scaled // FRAME_RATE_SCALE}.{scaled % FRAME_RATE_SCALE:03d}"


def _stream_inf(output: RenditionOutput) -> str:
    return (
        f"#EXT-X-STREAM-INF:BANDWIDTH={output.bandwidth_bps},AVERAGE-BANDWIDTH={output.average_bandwidth_bps},"
        f"RESOLUTION={output.width}x{output.height},FRAME-RATE={_frame_rate_text(output)},"
        f'CODECS="{output.codecs}"'
    )


def master_playlist_bytes(renditions: Sequence[RenditionOutput]) -> bytes:
    """The master playlist of `renditions`, in the given order (largest first, as planned).

    Raises ValueError when the renditions break a precondition (only a wiring bug can).
    """
    outputs = _validated_renditions(renditions)
    lines = ["#EXTM3U", f"#EXT-X-VERSION:{MASTER_PLAYLIST_VERSION}", "#EXT-X-INDEPENDENT-SEGMENTS"]
    for output in outputs:
        lines += [_stream_inf(output), output.playlist_path]
    data = ("\n".join(lines) + "\n").encode("ascii")
    if len(data) > MASTER_PLAYLIST_MAX_BYTES:
        raise ValueError(f"the master playlist exceeds {MASTER_PLAYLIST_MAX_BYTES} bytes")
    return data


def _write_failure(error: OSError) -> WorkerFailure:
    logger.warning("a package file could not be written: %s errno=%s", type(error).__name__, _errno_name(error))
    if error.errno in DISK_FULL_ERRNOS:
        return _scratch_full()
    return _failure(
        "output_write_failed", "package_write_failed",
        f"the package file could not be written: {_errno_name(error)}", retryable=True,
    )


def _remove_partial(path: Path) -> None:
    """Best effort: the job's scratch cleanup removes whatever is left. Paths are never logged."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as error:
        logger.warning("a partial package file could not be removed: errno=%s", _errno_name(error))


def _write_all(descriptor: int, data: bytes) -> None:
    """Write every byte of `data`, however few of them each call takes."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "a write made no progress")
        remaining = remaining[written:]


def _write_new_file(path: Path, data: bytes) -> None:
    """Create `path` with exactly `data`, never through a link and never over an existing file.

    Raises the typed write failure; a file it created is removed first.
    """
    try:
        descriptor = os.open(path, NEW_FILE_FLAGS, NEW_FILE_MODE)
    except OSError as error:
        # Nothing of ours to remove: the file was not created, or it was not created by this call.
        raise _write_failure(error) from None
    try:
        try:
            _write_all(descriptor, data)
        finally:
            os.close(descriptor)
    except OSError as error:
        _remove_partial(path)
        raise _write_failure(error) from None


def _require_new_file(output_root: object, name: str) -> Path:
    """The path of `name` in the output root, which must not exist yet."""
    path = _require_directory("output_root", output_root) / name
    if os.path.lexists(path):
        raise ValueError(f"{name} already exists")
    return path


def write_master_playlist(renditions: Sequence[RenditionOutput], *, output_root: Path) -> ArtifactFile:
    """Write `output_root / master.m3u8` and return its artifact.

    Raises ValueError for invalid arguments or an existing file, before anything is written; raises
    WorkerFailure when the write fails, after removing what it wrote.
    """
    path = _require_new_file(output_root, MASTER_PLAYLIST_PATH)
    data = master_playlist_bytes(renditions)
    _write_new_file(path, data)
    return ArtifactFile(
        path=MASTER_PLAYLIST_PATH,
        kind=ARTIFACT_KIND_MASTER,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def manifest_max_bytes(max_artifacts: int) -> int:
    """The most a manifest of this writer listing `max_artifacts` artifacts can take, within the
    contract's limit: the room the job reserves for it."""
    _require_int("max_artifacts", max_artifacts, 0)
    return min(MANIFEST_HEADER_MAX_BYTES + MANIFEST_ARTIFACT_MAX_BYTES * max_artifacts, CONTRACT_MANIFEST_MAX_BYTES)


def _timestamp(created_at: datetime) -> str:
    """Contract v2's timestamp, YYYY-MM-DDTHH:MM:SS.mmmZ, in UTC, the milliseconds truncated."""
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    utc = created_at.astimezone(timezone.utc)
    return (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
        f".{utc.microsecond // MICROSECONDS_PER_MILLISECOND:03d}Z"
    )


def _require_manifest_types(
    identity: object,
    profile_ref: object,
    profile: object,
    encryption: object,
    master: object,
    thumbnail: object,
    created_at: object,
) -> None:
    for name, value, expected in (
        ("identity", identity, ManifestIdentity),
        ("profile_ref", profile_ref, ProfileRef),
        ("profile", profile, MediaProfile),
        ("encryption", encryption, EncryptionRef),
        ("master", master, ArtifactFile),
        ("thumbnail", thumbnail, ThumbnailResult),
        ("created_at", created_at, datetime),
    ):
        if not isinstance(value, expected):
            raise TypeError(f"{name} must be a {expected.__name__}")


def _require_package_files(master: ArtifactFile, thumbnail: ThumbnailResult) -> None:
    if (master.path, master.kind) != (MASTER_PLAYLIST_PATH, ARTIFACT_KIND_MASTER):
        raise ValueError(f"master must be the {ARTIFACT_KIND_MASTER} artifact at {MASTER_PLAYLIST_PATH}")
    artifact = thumbnail.artifact
    if artifact is not None and (artifact.path, artifact.kind) != (THUMBNAIL_PATH, ARTIFACT_KIND_THUMBNAIL):
        raise ValueError(f"the thumbnail must be the {ARTIFACT_KIND_THUMBNAIL} artifact at {THUMBNAIL_PATH}")


def _requested_names(requested_renditions: object) -> frozenset[str]:
    if isinstance(requested_renditions, (str, bytes)) or not isinstance(requested_renditions, Sequence):
        raise ValueError("requested_renditions must be a sequence of rendition names")
    if not all(isinstance(name, str) for name in requested_renditions):
        raise ValueError("requested_renditions must hold strings")
    return frozenset(requested_renditions)


def _require_rendition_artifacts(output: RenditionOutput, max_segments: int) -> None:
    """Exactly its playlist, then at least one and at most `max_segments` segments, all under its name."""
    artifacts = output.artifacts
    if len(artifacts) < MIN_RENDITION_ARTIFACTS:
        raise ValueError(f"rendition {output.name} lacks its playlist or a segment")
    playlist, segments = artifacts[0], artifacts[1:]
    if (playlist.kind, playlist.path) != (ARTIFACT_KIND_PLAYLIST, output.playlist_path):
        raise ValueError(f"rendition {output.name}'s first artifact is not its playlist")
    if any(segment.kind != ARTIFACT_KIND_SEGMENT for segment in segments):
        raise ValueError(f"rendition {output.name} lists something other than segments after its playlist")
    if len(segments) > max_segments:
        raise ValueError(f"rendition {output.name} exceeds {max_segments} segments")
    if not all(artifact.path.startswith(f"{output.name}/") for artifact in artifacts):
        raise ValueError(f"rendition {output.name} lists a file outside its directory")


def _require_renditions_fit(
    outputs: tuple[RenditionOutput, ...],
    requested: frozenset[str],
    encryption: EncryptionRef,
    profile: MediaProfile,
) -> None:
    for output in outputs:
        if output.name not in requested:
            raise ValueError(f"rendition {output.name} was not requested")
        if output.encrypted != encryption.encrypted:
            raise ValueError(f"rendition {output.name}'s encryption differs from the job's")
        _require_rendition_artifacts(output, profile.encode.max_segments_per_rendition)


def _diagnostics(thumbnail: ThumbnailResult, extra_diagnostics: object) -> list[Diagnostic]:
    """The thumbnail's diagnostic first, then the others: at most CONTRACT_MAX_DIAGNOSTICS."""
    if isinstance(extra_diagnostics, (str, bytes)) or not isinstance(extra_diagnostics, Sequence):
        raise ValueError("extra_diagnostics must be a sequence of Diagnostic")
    if not all(isinstance(diagnostic, Diagnostic) for diagnostic in extra_diagnostics):
        raise TypeError("extra_diagnostics must hold Diagnostic entries")
    if any(diagnostic.code == THUMBNAIL_UNAVAILABLE for diagnostic in extra_diagnostics):
        raise ValueError(f"a {THUMBNAIL_UNAVAILABLE} diagnostic comes only from the thumbnail")
    diagnostics = ([] if thumbnail.diagnostic is None else [thumbnail.diagnostic]) + list(extra_diagnostics)
    if len(diagnostics) > CONTRACT_MAX_DIAGNOSTICS:
        raise ValueError(f"a manifest carries at most {CONTRACT_MAX_DIAGNOSTICS} diagnostics")
    return diagnostics


def _artifact_entry(artifact: ArtifactFile, rendition: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": artifact.path,
        "kind": artifact.kind,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
    }
    if rendition is not None:
        entry["rendition"] = rendition
    return entry


def _artifact_entries(
    master: ArtifactFile, thumbnail: ThumbnailResult, outputs: tuple[RenditionOutput, ...]
) -> list[dict[str, Any]]:
    """The master, the thumbnail when there is one, then each rendition's playlist and segments."""
    entries = [_artifact_entry(master, None)]
    if thumbnail.artifact is not None:
        entries.append(_artifact_entry(thumbnail.artifact, None))
    for output in outputs:
        entries += [_artifact_entry(artifact, output.name) for artifact in output.artifacts]
    paths = [entry["path"] for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("two artifacts share a path")
    if len(entries) > CONTRACT_MAX_ARTIFACTS:
        raise ValueError(f"a manifest lists at most {CONTRACT_MAX_ARTIFACTS} artifacts")
    total_bytes = sum(entry["size_bytes"] for entry in entries)
    if not 1 <= total_bytes <= CONTRACT_MAX_TOTAL_BYTES:
        raise ValueError(f"the artifacts' total size must be from 1 to {CONTRACT_MAX_TOTAL_BYTES} bytes")
    return entries


def _rendition_entry(output: RenditionOutput) -> dict[str, Any]:
    return {
        "name": output.name,
        "width": output.width,
        "height": output.height,
        "bandwidth_bps": output.bandwidth_bps,
        "average_bandwidth_bps": output.average_bandwidth_bps,
        "codecs": output.codecs,
        "playlist_path": output.playlist_path,
    }


def _media(outputs: tuple[RenditionOutput, ...]) -> dict[str, Any]:
    # The cut lets a rendition run up to one frame past an admitted 6 hours.
    return {
        "duration_ms": min(max(output.duration_ms for output in outputs), CONTRACT_MAX_DURATION_MS),
        "max_width": max(output.width for output in outputs),
        "max_height": max(output.height for output in outputs),
        "has_audio": outputs[0].has_audio,
        "rendition_count": len(outputs),
    }


def _encryption_entry(encryption: EncryptionRef) -> dict[str, Any]:
    if encryption.media_key_id is None:
        return {"mode": encryption.mode}
    return {"mode": encryption.mode, "media_key_id": encryption.media_key_id}


def _serialized_manifest(
    *,
    identity: ManifestIdentity,
    profile_ref: ProfileRef,
    profile: MediaProfile,
    encryption: EncryptionRef,
    requested_renditions: Sequence[str],
    renditions: Sequence[RenditionOutput],
    master: ArtifactFile,
    thumbnail: ThumbnailResult,
    extra_diagnostics: Sequence[Diagnostic],
    created_at: datetime,
) -> tuple[bytes, int, int]:
    """(the manifest's bytes, its artifact count, its total bytes), every precondition checked."""
    _require_manifest_types(identity, profile_ref, profile, encryption, master, thumbnail, created_at)
    if (profile_ref.id, profile_ref.version) != (profile.profile_id, profile.version):
        raise ValueError("the profile reference does not name the profile the encode used")
    _require_package_files(master, thumbnail)
    outputs = _validated_renditions(renditions)
    _require_renditions_fit(outputs, _requested_names(requested_renditions), encryption, profile)
    diagnostics = _diagnostics(thumbnail, extra_diagnostics)
    artifacts = _artifact_entries(master, thumbnail, outputs)
    total_bytes = sum(entry["size_bytes"] for entry in artifacts)
    document: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "document_kind": DOCUMENT_KIND,
        "generation_id": identity.execution_id,
        "created_at": _timestamp(created_at),
        "identity": {field.name: getattr(identity, field.name) for field in fields(identity)},
        "profile": {"id": profile_ref.id, "version": profile_ref.version, "sha256": profile_ref.sha256},
        "encryption": _encryption_entry(encryption),
        "media": _media(outputs),
        "renditions": [_rendition_entry(output) for output in outputs],
        "master_playlist_path": master.path,
        "thumbnail_path": None if thumbnail.artifact is None else thumbnail.artifact.path,
        "artifact_count": len(artifacts),
        "total_bytes": total_bytes,
        "artifacts": artifacts,
    }
    if diagnostics:
        document["diagnostics"] = [{"code": item.code, "detail": item.detail} for item in diagnostics]
    data = json.dumps(document, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if len(data) > CONTRACT_MANIFEST_MAX_BYTES:
        raise ValueError(f"the manifest exceeds the contract's {CONTRACT_MANIFEST_MAX_BYTES} bytes")
    return data, len(artifacts), total_bytes


def manifest_bytes(
    *,
    identity: ManifestIdentity,
    profile_ref: ProfileRef,
    profile: MediaProfile,
    encryption: EncryptionRef,
    requested_renditions: Sequence[str],
    renditions: Sequence[RenditionOutput],
    master: ArtifactFile,
    thumbnail: ThumbnailResult,
    extra_diagnostics: Sequence[Diagnostic] = (),
    created_at: datetime,
) -> bytes:
    """The generation manifest, compact JSON in the contract's key order, without a final newline.

    Raises TypeError or ValueError when the inputs break a precondition (only a wiring bug can).
    """
    data, _, _ = _serialized_manifest(
        identity=identity,
        profile_ref=profile_ref,
        profile=profile,
        encryption=encryption,
        requested_renditions=requested_renditions,
        renditions=renditions,
        master=master,
        thumbnail=thumbnail,
        extra_diagnostics=extra_diagnostics,
        created_at=created_at,
    )
    return data


def write_manifest(
    *,
    identity: ManifestIdentity,
    profile_ref: ProfileRef,
    profile: MediaProfile,
    encryption: EncryptionRef,
    requested_renditions: Sequence[str],
    renditions: Sequence[RenditionOutput],
    master: ArtifactFile,
    thumbnail: ThumbnailResult,
    extra_diagnostics: Sequence[Diagnostic] = (),
    created_at: datetime,
    output_root: Path,
    max_bytes: int,
) -> ManifestOutput:
    """Write `output_root / generation-manifest.json`, the job's last output, within `max_bytes`
    (the room reserved for it: `manifest_max_bytes` of the job's artifact allowance).

    Raises TypeError or ValueError for invalid arguments or an existing file, before anything is
    written; raises WorkerFailure for a manifest longer than `max_bytes` and when the write fails,
    after removing what it wrote.
    """
    _require_int("max_bytes", max_bytes, 1)
    path = _require_new_file(output_root, MANIFEST_PATH)
    data, artifact_count, total_bytes = _serialized_manifest(
        identity=identity,
        profile_ref=profile_ref,
        profile=profile,
        encryption=encryption,
        requested_renditions=requested_renditions,
        renditions=renditions,
        master=master,
        thumbnail=thumbnail,
        extra_diagnostics=extra_diagnostics,
        created_at=created_at,
    )
    if len(data) > max_bytes:
        raise _failure(
            "internal_error", "manifest_too_large",
            f"the generation manifest of {len(data)} bytes exceeds its reserved {max_bytes} bytes",
        )
    _write_new_file(path, data)
    return ManifestOutput(
        path=MANIFEST_PATH,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        artifact_count=artifact_count,
        total_bytes=total_bytes,
    )


def reserve_package(job: OutputAllowance, profile: MediaProfile) -> OutputAllowance:
    """The renditions' allowance: the job's, less the room of the master playlist, the thumbnail
    and the manifest.

    The job (D1) calls it once, right after `OutputAllowance.for_job`, and gives the remainder to
    the renditions. The master playlist is held to MASTER_PLAYLIST_MAX_BYTES, the thumbnail to the
    profile's `thumbnail_max_bytes`, the manifest to `manifest_max_bytes(job.max_artifacts)`; two
    files are set aside (the master and the thumbnail: the manifest is no artifact, but its bytes
    are output and scratch). Raises the typed limit failure when the job's allowance cannot hold them.
    """
    if not isinstance(job, OutputAllowance):
        raise TypeError("job must be an OutputAllowance")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    package_bytes = (
        MASTER_PLAYLIST_MAX_BYTES + profile.encode.thumbnail_max_bytes + manifest_max_bytes(job.max_artifacts)
    )
    return job.reserve(package_bytes, PACKAGE_ARTIFACTS)
