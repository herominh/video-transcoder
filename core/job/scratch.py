"""An execution's scratch directory in the scratch root: the lock a live one holds, and the removal of
those a killed process left behind.

An execution makes one directory of its own in the scratch root, named with SCRATCH_PREFIX, and removes
it whatever its ending (core/job/runner.py). A process that is killed removes nothing: its directory
stays, with a downloaded source and what was encoded by then. remove_leftover_scratch() removes such
directories at the next process start.

What tells a leftover from the directory of an execution that still runs, in another process on the same
root, is a lock. Every execution holds an exclusive advisory lock (flock) on its own directory, from
right after it made the directory until after it removed it (hold_scratch_lock, release_scratch_lock).
The lock lives on an open descriptor, so the kernel lets go of it when the process dies, however it
dies. The removal takes the same lock, without waiting, before it removes a directory, and passes over
one whose lock it does not get. Between making its directory and locking it an execution holds no lock
yet, and it takes its lock without waiting too: so a directory changed less than LEFTOVER_MIN_AGE_S ago
is passed over without a try at its lock.

The scratch root must be on a local file system on which a directory can be locked with flock (nothing
checks that at process start): on a root that cannot lock, every execution ends as the runner's
retryable failure for a scratch directory that could not be set up (resource_exhausted /
scratch_create_failed). One host's kernel keeps the lock, so it tells a live directory from a leftover
only between processes of one host: a root shared between hosts is not supported, because a sibling
host's removal would find a live directory's lock free and remove the directory once it is
LEFTOVER_MIN_AGE_S old.

No log line of this module holds a path or an error's own text, which may name one: an error is named
by its class. And the line that tells of an error is written once the `except` block that caught that
error has ended: a log handler that fails on a line does not raise, Python's logging prints a report of
its own with every error that failure is chained to, and inside the block that would be the caught
error, with its own text.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import stat
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# What the name of every execution's scratch directory starts with.
SCRATCH_PREFIX = "vh-job-"
# How long a scratch directory must have gone unchanged before it can be a leftover, in seconds: far
# longer than the moment between a live execution making its directory and locking it.
LEFTOVER_MIN_AGE_S = 60.0
# How a scratch directory is opened for its lock: the directory itself, through no link at its last
# part, on a descriptor no child process (ffprobe, ffmpeg) inherits, which would keep the lock alive.
LOCK_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Exclusive, and refused at once where another holds it: neither an execution nor a removal waits.
LOCK_WITHOUT_WAITING = fcntl.LOCK_EX | fcntl.LOCK_NB


def _now_s() -> float:
    """The wall-clock time a directory's last change is counted back from."""
    return time.time()


def hold_scratch_lock(directory: Path) -> int:
    """Take the exclusive lock on the scratch directory `directory` and return the descriptor it lives on.

    The lock is held until that descriptor is closed (release_scratch_lock) or the process dies. It is
    taken without waiting. Raises OSError when the directory cannot be opened as a real directory or
    its lock is not free (BlockingIOError), with nothing left open; TypeError for anything but a Path.
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    descriptor = os.open(directory, LOCK_OPEN_FLAGS)
    try:
        fcntl.flock(descriptor, LOCK_WITHOUT_WAITING)
    except BaseException:
        release_scratch_lock(descriptor)
        raise
    return descriptor


def release_scratch_lock(descriptor: int) -> None:
    """Let go of a lock hold_scratch_lock() took, by closing the descriptor it lives on.

    A descriptor that cannot be closed is logged by the error's class and raises nothing: a lock that
    lingers never changes an execution's result. Raises TypeError for anything but a descriptor.
    """
    if isinstance(descriptor, bool) or not isinstance(descriptor, int):
        raise TypeError("descriptor must be an int")
    error_class: str | None = None
    try:
        os.close(descriptor)
    except OSError as error:
        # Only the class name is kept past this block (see the module's last paragraph).
        error_class = type(error).__name__
    if error_class is not None:
        logger.warning("a scratch directory's lock could not be released: %s", error_class)


def remove_leftover_scratch(scratch_root: Path) -> int:
    """Remove the scratch directories a killed process left in `scratch_root`, and return how many.

    It looks only at the direct children of `scratch_root` whose name starts with SCRATCH_PREFIX. Of
    those it passes over, untouched: whatever is no real directory (a symbolic link is neither
    followed nor removed), a directory this process's user does not own, one that was changed less
    than LEFTOVER_MIN_AGE_S ago, and one whose lock is held (an execution of another process runs in
    it). Every other one is removed, under its lock.

    It never raises. A root that cannot be searched (one that is no Path included) and a directory
    that cannot be examined or removed are passed over, each with one warning that names the error's
    class. One line tells how many were removed. No line holds a path.
    """
    removed = 0
    error_class: str | None = None
    try:
        for directory in _prefixed_children(scratch_root):
            removed += int(_removed_as_a_leftover(directory))
    except Exception as error:  # noqa: BLE001 - leftovers that stay never keep a process from starting
        # Only the class name is kept past this block: the error's own text names the root.
        error_class = type(error).__name__
    if error_class is not None:
        logger.warning("the scratch root could not be searched for leftover directories: %s", error_class)
    logger.info("removed %d leftover scratch directories", removed)
    return removed


def _prefixed_children(scratch_root: Path) -> list[Path]:
    """The direct children of the scratch root that bear a scratch directory's name, whatever they are."""
    if not isinstance(scratch_root, Path):
        raise TypeError("scratch_root must be a Path")
    with os.scandir(scratch_root) as children:
        names = sorted(child.name for child in children if child.name.startswith(SCRATCH_PREFIX))
    return [scratch_root / name for name in names]


def _removed_as_a_leftover(directory: Path) -> bool:
    """True once `directory` was removed; False when it was passed over, which an error makes it too."""
    try:
        return _removed_under_its_lock(directory)
    except Exception as error:  # noqa: BLE001 - one directory never keeps the others from being looked at
        # Only the class name is kept past this block: the error's own text names the directory.
        error_class = type(error).__name__
    logger.warning("a leftover scratch directory could not be examined or removed: %s", error_class)
    return False


def _removed_under_its_lock(directory: Path) -> bool:
    """Examine what bears a scratch directory's name, and remove it when it is a leftover whose lock
    is free: True then, False when it is passed over. Raises what examining, locking or removing raises."""
    examined = os.lstat(directory)
    if not _may_be_a_leftover(examined):
        return False
    try:
        descriptor = hold_scratch_lock(directory)
    except BlockingIOError:
        # An execution of another process holds the lock: its directory is no leftover.
        return False
    try:
        # What was opened is what was examined: a directory put in its place since is not removed.
        if not os.path.samestat(examined, os.fstat(descriptor)):
            return False
        shutil.rmtree(directory)
    finally:
        release_scratch_lock(descriptor)
    return True


def _may_be_a_leftover(examined: os.stat_result) -> bool:
    """Whether what bears a scratch directory's name is a real directory of this process's user that
    was not changed for LEFTOVER_MIN_AGE_S. `examined` is of the name itself, never of a link's target."""
    if not stat.S_ISDIR(examined.st_mode):
        return False
    if examined.st_uid != os.geteuid():
        return False
    return _now_s() - examined.st_mtime >= LEFTOVER_MIN_AGE_S
