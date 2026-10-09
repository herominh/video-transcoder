"""The AES-128 key the HLS muxer reads, for the length of one encode and never inside the output.

On Linux the key lives only in an anonymous memory file (memfd) that ffmpeg inherits: it never
touches a disk, and it disappears with the last process holding it, even when the worker itself
is killed. The key-info file the muxer reads names that descriptor (`/proc/self/fd/N`, which ffmpeg
resolves in its own process) and holds no secret. Elsewhere (a developer's macOS) the key is a
private file instead. Both sit in a fresh owner-only directory beside, never inside, the output
directory, so no listing or upload of the output can reach them.

The muxer re-reads the key-info file at every segment (`periodic_rekey`), which is what makes it use
each segment's media sequence number as that segment's IV and write the IV into the playlist;
without it, ffmpeg (7.1 and 8) encrypts every segment with the first segment's IV.
"""

from __future__ import annotations

import errno
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

KEY_BYTES = 16  # AES-128
# The URI written into every #EXT-X-KEY line. The Hub rewrites it when it serves a playlist; the
# key itself never leaves the Hub's database except through its key endpoint.
KEY_URI_PLACEHOLDER = "enc.key"
KEY_DIR_PREFIX = "hls-key-"
KEY_FILE_NAME = "media.key"
KEY_INFO_FILE_NAME = "key.info"
MEMFD_NAME = "hls-key"
PROC_SELF_FD = Path("/proc/self/fd")
PRIVATE_FILE_MODE = 0o600
PRIVATE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
# The key-info file is line-based: a path holding one of these would change its meaning.
LINE_BREAKS = ("\n", "\r")
UNKNOWN_ERRNO_NAME = "unknown"


@dataclass(frozen=True, slots=True)
class KeyInfo:
    path: Path  # the key-info file to hand to -hls_key_info_file
    key_path: Path  # where a child process reads the key itself (/proc/self/fd/N, or the private file)
    pass_fds: tuple[int, ...]  # descriptors ffmpeg must inherit (the memfd), empty for a key file


def memory_key_supported() -> bool:
    """True where the key can live in a memfd that a child reaches through /proc/self/fd."""
    return hasattr(os, "memfd_create") and PROC_SELF_FD.is_dir()


def _errno_name(error: OSError) -> str:
    return errno.errorcode.get(error.errno or 0, UNKNOWN_ERRNO_NAME)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, PRIVATE_FILE_FLAGS, PRIVATE_FILE_MODE)
    try:
        _write_all(descriptor, data)
    finally:
        os.close(descriptor)


def _memory_key(key: bytes) -> int:
    """A close-on-exec memfd holding exactly the key; the caller passes it to ffmpeg explicitly."""
    descriptor = os.memfd_create(MEMFD_NAME, os.MFD_CLOEXEC)
    try:
        _write_all(descriptor, key)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        # The job's scratch cleanup still removes it; the path itself is never logged.
        logger.warning("an HLS key file could not be removed: errno=%s", _errno_name(error))


@contextmanager
def hls_key_info(key: bytes, *, parent: Path) -> Iterator[KeyInfo]:
    """Hold `key` for ffmpeg and write its key-info file under a new private directory in `parent`;
    yield both; release the key and remove the directory on exit, whatever happened.

    `parent` must be an existing absolute directory outside every output directory. Raises
    OSError when the key cannot be held or the files cannot be written (nothing is left then).
    """
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise ValueError(f"key must be {KEY_BYTES} bytes")
    if not isinstance(parent, Path) or not parent.is_absolute():
        raise ValueError("parent must be an absolute Path")
    directory = Path(tempfile.mkdtemp(prefix=KEY_DIR_PREFIX, dir=parent))  # created owner-only
    key_path = directory / KEY_FILE_NAME
    info_path = directory / KEY_INFO_FILE_NAME
    memory_descriptor: int | None = None
    try:
        if memory_key_supported():
            memory_descriptor = _memory_key(key)
            key_location = PROC_SELF_FD / str(memory_descriptor)
            pass_fds: tuple[int, ...] = (memory_descriptor,)
        else:
            if any(mark in str(key_path) for mark in LINE_BREAKS):
                raise ValueError("the key directory path holds a line break")
            _write_private(key_path, key)
            key_location = key_path
            pass_fds = ()
        _write_private(info_path, f"{KEY_URI_PLACEHOLDER}\n{key_location}\n".encode())
        yield KeyInfo(path=info_path, key_path=key_location, pass_fds=pass_fds)
    finally:
        if memory_descriptor is not None:
            os.close(memory_descriptor)
        _remove(key_path)
        _remove(info_path)
        try:
            directory.rmdir()
        except OSError as error:
            logger.warning("an HLS key directory could not be removed: errno=%s", _errno_name(error))
