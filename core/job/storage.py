"""How the job touches a bucket: it reads the source object and stores the generation, nothing else.

Each side has its own client of an access profile: storage_client() reads the source, upload_client()
stores the generation.

download_source() copies the source into the scratch directory through RangeSource.read(), so every byte
arrives through the reader's own checks: in production an S3RangeSource, whose ranged reads are each
conditional on the recorded ETag. A source that changed while it was read fails the job.

upload_generation() stores exactly the files the manifest lists and then the manifest, each under
`<prefix>/<relative path>` (contract README 15.1, rule 3: every object a runtime writes has a key that
starts with the granted prefix and `/`). The manifest is the completion marker, so it goes last, only
after every artifact was stored. It takes its list from the caller, never from a walk of the output
directory, and it only ever puts: it lists nothing, reads nothing back and deletes nothing.

No log line and no failure detail holds an endpoint, a URL, a credential or the storage client's own
error text: a storage error is named by its class and, for an answer of the storage, its error code.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import Failure, WorkerFailure, make_detail
from core.package import ManifestOutput
from core.profile import MIB
from core.protocol import schema
from core.protocol.settings import StorageProfile
from core.source_reader import (
    S3_CONNECT_TIMEOUT_S,
    S3_MAX_ATTEMPTS,
    S3_READ_TIMEOUT_S,
    RangeSource,
    make_s3_client,
)

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_BYTES = 8 * MIB
# How many more times one chunk's read is tried after a failure the source calls retryable, and one
# object's upload after a failed request.
DOWNLOAD_EXTRA_ATTEMPTS = 2
UPLOAD_EXTRA_ATTEMPTS = 2
SOURCE_FILE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
SOURCE_FILE_MODE = 0o600
DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
UNKNOWN_ERRNO_NAME = "unknown"
KEY_SEPARATOR = "/"
# How the upload client signs, addresses and retries: as the source reader's client does (core/source_reader.py).
SIGNATURE_VERSION = "s3v4"
ADDRESSING_STYLE = "path"
RETRY_MODE = "standard"
# botocore's setting for "a checksum only where the operation requires one".
CHECKSUM_WHEN_REQUIRED = "when_required"
# What the Hub's delivery and a player expect of each file the worker writes; the manifest is the one JSON file.
CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".jpg": "image/jpeg",
    ".json": "application/json",
}
_COMMON_DEFINITIONS = schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/"
_REL_PATH_DEFINITION = _COMMON_DEFINITIONS + "rel_path"
_OBJECT_KEY_DEFINITION = _COMMON_DEFINITIONS + "object_key"

DOWNLOAD_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "download_deadline_exceeded", False,
    "the job ran out of time while it downloaded the source",
)
UPLOAD_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "upload_deadline_exceeded", False,
    "the job ran out of time while it uploaded the generation",
)
SCRATCH_FULL = Failure("resource_exhausted", "scratch_full", True, "the scratch disk is full")


def storage_client(profile: StorageProfile) -> Any:
    """The S3 client the source is read with: the endpoint and credentials a dispatch's credential_ref names."""
    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a StorageProfile")
    return make_s3_client(
        endpoint_url=profile.endpoint,
        region=profile.region,
        access_key_id=profile.access_key_id,
        secret_access_key=profile.secret_access_key,
    )


def upload_client(profile: StorageProfile) -> Any:
    """The S3 client the generation is stored with: the access profile's endpoint and credentials, and
    the source reader's signing, path-style addressing, time-outs and single attempt.

    It adds a checksum only where an operation requires one: botocore's default since 1.36 sends a
    PutObject as an `aws-chunked` body with a checksum trailer, which not every S3-compatible store
    accepts, while a plain signed PUT with a Content-Length is accepted by all of them.
    """
    if not isinstance(profile, StorageProfile):
        raise TypeError("profile must be a StorageProfile")
    return boto3.client(
        "s3",
        endpoint_url=profile.endpoint,
        region_name=profile.region,
        aws_access_key_id=profile.access_key_id,
        aws_secret_access_key=profile.secret_access_key,
        config=Config(
            signature_version=SIGNATURE_VERSION,
            connect_timeout=S3_CONNECT_TIMEOUT_S,
            read_timeout=S3_READ_TIMEOUT_S,
            retries={"total_max_attempts": S3_MAX_ATTEMPTS, "mode": RETRY_MODE},
            s3={"addressing_style": ADDRESSING_STYLE},
            request_checksum_calculation=CHECKSUM_WHEN_REQUIRED,
            response_checksum_validation=CHECKSUM_WHEN_REQUIRED,
        ),
    )


def _errno_name(error: OSError) -> str:
    # The errno's symbolic name only: an OSError's own message may carry a path.
    return errno.errorcode.get(error.errno or 0, UNKNOWN_ERRNO_NAME)


def _notify(on_bytes: Callable[[int], None] | None, total_bytes: int) -> None:
    """Tell the caller how far the transfer is; whatever its callback raises never fails the transfer."""
    if on_bytes is None:
        return
    try:
        on_bytes(total_bytes)
    except Exception as error:  # noqa: BLE001 - progress is the caller's convenience, never the transfer's outcome
        logger.warning("a transfer's progress callback failed: %s", type(error).__name__)


def _require_transfer_arguments(deadline: object, on_bytes: object) -> None:
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if on_bytes is not None and not callable(on_bytes):
        raise TypeError("on_bytes must be callable or None")


# --- the source ----------------------------------------------------------------------------------


def download_source(
    source: RangeSource,
    *,
    destination: Path,
    deadline: Deadline,
    on_bytes: Callable[[int], None] | None = None,
) -> int:
    """Copy the whole source object to the new file `destination` and return the bytes written.

    The object is read in order, DOWNLOAD_CHUNK_BYTES at a time, only through `source.read()`. A chunk
    whose read fails with a retryable WorkerFailure is read again, DOWNLOAD_EXTRA_ATTEMPTS times at
    most; every other failure of a read ends the download at once. `on_bytes` is called with the bytes
    written so far after each chunk.

    Raises WorkerFailure for a failed read, a chunk of another length than asked (the object is not the
    recorded size), a passed deadline and a failed write; the partial file is removed first. Raises
    TypeError or ValueError for invalid arguments or an existing destination, before anything is read.
    """
    size_bytes = _validated_download(source, destination)
    _require_transfer_arguments(deadline, on_bytes)
    try:
        descriptor = os.open(destination, SOURCE_FILE_FLAGS, SOURCE_FILE_MODE)
    except OSError as error:
        # Nothing of ours to remove: the file was not created, or it was not created by this call.
        raise _source_write_failure(error) from None
    try:
        try:
            return _copy_source(source, size_bytes, descriptor, deadline, on_bytes)
        finally:
            os.close(descriptor)
    except BaseException:
        _remove_partial(destination)
        raise


def _validated_download(source: object, destination: object) -> int:
    """The source's size, once the source and the destination are what a download can use."""
    if not callable(getattr(source, "read", None)):
        raise TypeError("source must be a RangeSource")
    size_bytes = getattr(source, "size_bytes", None)
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
        raise ValueError("the source's size_bytes must be a positive integer")
    if not isinstance(destination, Path) or not destination.is_absolute():
        raise ValueError("destination must be an absolute Path")
    if os.path.lexists(destination):
        raise ValueError("destination already exists")
    return size_bytes


def _copy_source(
    source: RangeSource,
    size_bytes: int,
    descriptor: int,
    deadline: Deadline,
    on_bytes: Callable[[int], None] | None,
) -> int:
    written = 0
    while written < size_bytes:
        length = min(DOWNLOAD_CHUNK_BYTES, size_bytes - written)
        data = _read_chunk(source, written, length, deadline)
        if len(data) != length:
            raise WorkerFailure(
                Failure(
                    "source_integrity_mismatch", "source_size_mismatch", False,
                    f"the source gave {len(data)} bytes where {length} were read",
                )
            )
        try:
            _write_all(descriptor, data)
        except OSError as error:
            raise _source_write_failure(error) from None
        written += length
        _notify(on_bytes, written)
    return written


def _read_chunk(source: RangeSource, start: int, length: int, deadline: Deadline) -> bytes:
    """One chunk through the source's own checks, read again only after a failure it calls retryable."""
    extra_attempts_left = DOWNLOAD_EXTRA_ATTEMPTS
    while True:
        if deadline.expired():
            raise WorkerFailure(DOWNLOAD_DEADLINE_EXCEEDED)
        try:
            return source.read(start, length)
        except WorkerFailure as failed:
            if not failed.failure.retryable or extra_attempts_left == 0:
                raise
            extra_attempts_left -= 1
            logger.warning(
                "a read of the source failed and is tried again: %s/%s",
                failed.failure.error_class, failed.failure.code,
            )


def _write_all(descriptor: int, data: bytes) -> None:
    """Write every byte of `data`, however few of them each call takes."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "a write made no progress")
        remaining = remaining[written:]


def _source_write_failure(error: OSError) -> WorkerFailure:
    logger.warning("the source file could not be written: %s errno=%s", type(error).__name__, _errno_name(error))
    if error.errno in DISK_FULL_ERRNOS:
        return WorkerFailure(SCRATCH_FULL)
    return WorkerFailure(
        Failure(
            "output_write_failed", "source_write_failed", True,
            f"the source file could not be written: {_errno_name(error)}",
        )
    )


def _remove_partial(path: Path) -> None:
    """Best effort: the job's scratch cleanup removes whatever is left. Paths are never logged."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as error:
        logger.warning("a partial source file could not be removed: errno=%s", _errno_name(error))


# --- the generation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Upload:
    """One object to store: its key, the local file it is read from and what the manifest records of it."""

    relative_path: str
    key: str
    local_path: Path
    size_bytes: int
    content_type: str


def upload_generation(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    output_root: Path,
    artifacts: Sequence[ArtifactFile],
    manifest: ManifestOutput,
    deadline: Deadline,
    on_bytes: Callable[[int], None] | None = None,
) -> int:
    """Store the generation in `bucket` and return the bytes uploaded: `artifacts` in order, then the manifest.

    Every key is `prefix`, `/` and the file's path relative to `output_root`. Each object is one
    PutObject of the file, whose size on disk must still be the recorded one; a failed request is sent
    again, UPLOAD_EXTRA_ATTEMPTS times at most, from the start of the file. `on_bytes` is called with
    the bytes uploaded so far after each object.

    Raises TypeError or ValueError, before any request, for anything that would put an object where the
    generation does not belong (a wiring bug): a prefix that is no contract object key, a path that is
    no contract rel_path, two files with one path, a file that is not a regular file of the output root
    or has no content type here; a ValueError about a file names its relative path only. Raises
    WorkerFailure for a file that changed, a passed deadline and an upload that kept failing: the
    manifest is then not stored.
    """
    if not callable(getattr(client, "put_object", None)):
        raise TypeError("client must be an S3 client")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("bucket must be a non-empty string")
    _require_transfer_arguments(deadline, on_bytes)
    uploads = _planned_uploads(prefix, output_root, artifacts, manifest)
    uploaded = 0
    for upload in uploads:
        _store(client, bucket, upload, deadline)
        uploaded += upload.size_bytes
        _notify(on_bytes, uploaded)
    return uploaded


def _planned_uploads(prefix: object, output_root: object, artifacts: object, manifest: object) -> list[_Upload]:
    """Every object of the generation in upload order, the manifest last; refused as a whole for a wiring bug."""
    if not isinstance(prefix, str) or schema.error_pointers_for_schema({"$ref": _OBJECT_KEY_DEFINITION}, prefix):
        raise ValueError("prefix must be a contract object key: no leading or trailing `/`, no `.` or `..` segment")
    if not isinstance(output_root, Path) or not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("output_root must be an absolute Path of an existing directory")
    # Resolved once, so that a link below the root is the only thing that can still move a file.
    root = output_root.resolve()
    return [
        _planned_upload(prefix, root, relative_path, size_bytes)
        for relative_path, size_bytes in _listed_files(artifacts, manifest)
    ]


def _listed_files(artifacts: object, manifest: object) -> list[tuple[str, int]]:
    """(relative path, recorded size) of every artifact in order, then of the manifest; no path twice."""
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise TypeError("artifacts must be a sequence of ArtifactFile")
    if not all(isinstance(artifact, ArtifactFile) for artifact in artifacts):
        raise TypeError("artifacts must hold ArtifactFile entries")
    if not isinstance(manifest, ManifestOutput):
        raise TypeError("manifest must be a ManifestOutput")
    listed = [(artifact.path, artifact.size_bytes) for artifact in artifacts]
    listed.append((manifest.path, manifest.size_bytes))
    seen: set[str] = set()
    for relative_path, _ in listed:
        if not isinstance(relative_path, str) or schema.error_pointers_for_schema(
            {"$ref": _REL_PATH_DEFINITION}, relative_path
        ):
            raise ValueError(f"{relative_path!r} is not a contract rel_path")
        # The manifest comes last, so an artifact that takes its path is found here too.
        if relative_path in seen:
            raise ValueError(f"{relative_path!r} names two files of the generation")
        seen.add(relative_path)
    return listed


def _planned_upload(prefix: str, root: Path, relative_path: str, size_bytes: int) -> _Upload:
    content_type = CONTENT_TYPES.get(os.path.splitext(relative_path)[1])
    if content_type is None:
        raise ValueError(f"{relative_path!r} has no content type this worker uploads")
    local_path = root / relative_path
    if not _is_regular_file_below(root, local_path):
        raise ValueError(f"{relative_path!r} is not a regular file of the output root")
    return _Upload(
        relative_path=relative_path,
        key=f"{prefix}{KEY_SEPARATOR}{relative_path}",
        local_path=local_path,
        size_bytes=size_bytes,
        content_type=content_type,
    )


def _is_regular_file_below(root: Path, local_path: Path) -> bool:
    """Whether `local_path`, a path below the resolved `root`, is a regular file reached through no link."""
    try:
        is_regular = stat.S_ISREG(os.lstat(local_path).st_mode)
        # `root` is resolved already: resolving changes the path only when a part below it is a link.
        return is_regular and local_path.resolve() == local_path
    except OSError:
        return False


def _store(client: Any, bucket: str, upload: _Upload, deadline: Deadline) -> None:
    """Put one object; a failed request is sent again, UPLOAD_EXTRA_ATTEMPTS times at most."""
    extra_attempts_left = UPLOAD_EXTRA_ATTEMPTS
    while True:
        if deadline.expired():
            raise WorkerFailure(UPLOAD_DEADLINE_EXCEEDED)
        try:
            _put(client, bucket, upload)
            return
        except (BotoCoreError, ClientError) as error:
            name = _storage_error_name(error)
            if extra_attempts_left == 0:
                # `from None`: the client's own error names the endpoint, and must not travel with this one.
                raise _upload_failed(name) from None
            extra_attempts_left -= 1
            logger.warning("an upload failed and is tried again: %s", name)


def _put(client: Any, bucket: str, upload: _Upload) -> None:
    """One PutObject of the file as it is on disk now, opened anew so that every attempt sends it from its start."""
    with _opened_artifact(upload) as body:
        client.put_object(
            Bucket=bucket,
            Key=upload.key,
            Body=body,
            ContentLength=upload.size_bytes,
            ContentType=upload.content_type,
        )


def _open_without_following(path: str, flags: int) -> int:
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))


def _opened_artifact(upload: _Upload) -> BinaryIO:
    """The file to upload, open for reading, once it is still a regular file of the recorded size."""
    try:
        body = open(upload.local_path, "rb", opener=_open_without_following)
    except OSError as error:
        raise _artifact_changed(upload, f"it could not be opened: {_errno_name(error)}") from None
    try:
        status = os.fstat(body.fileno())
    except OSError as error:
        body.close()
        raise _artifact_changed(upload, f"it could not be measured: {_errno_name(error)}") from None
    if not stat.S_ISREG(status.st_mode) or status.st_size != upload.size_bytes:
        body.close()
        raise _artifact_changed(upload, f"it is no longer the file of {upload.size_bytes} bytes the manifest records")
    return body


def _artifact_changed(upload: _Upload, what: str) -> WorkerFailure:
    return WorkerFailure(
        Failure(
            "internal_error", "artifact_changed_before_upload", True,
            make_detail(f"{upload.relative_path} changed before its upload: {what}"),
        )
    )


def _upload_failed(storage_error_name: str) -> WorkerFailure:
    return WorkerFailure(
        Failure(
            "output_write_failed", "upload_failed", True,
            make_detail(f"an object could not be stored: {storage_error_name}"),
        )
    )


def _storage_error_name(error: BotoCoreError | ClientError) -> str:
    """The error's class name and, for an answer of the storage, its error code: never the error's own
    text, which names the endpoint and the object."""
    name = type(error).__name__
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code")
        if isinstance(code, str) and code:
            name = f"{name} {code}"
    return make_detail(name)
