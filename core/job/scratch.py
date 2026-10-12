"""An execution's scratch directory in the scratch root: its name, the lock a live one holds, and the
removal of those a killed process left behind.

An execution makes one directory of its own in the scratch root, named with scratch_prefix(), and
removes it whatever its ending (core/job/runner.py). A process that is killed removes nothing: its
directory stays, with a downloaded source and what was encoded by then. remove_leftover_scratch()
removes such directories at the next process start.

What tells a leftover from the directory of an execution that still runs, in another process on the same
root, is a lock. Every execution holds an exclusive advisory lock (flock) on its own directory, from
right after it made the directory until after it removed it (hold_scratch_lock, release_scratch_lock).
The lock lives on an open descriptor, so the kernel lets go of it when the process dies, however it
dies. The removal takes the same lock, without waiting, before it removes a directory, and passes over
one whose lock it does not get. Between making its directory and locking it an execution holds no lock
yet, and it takes its lock without waiting too: so a directory changed less than LEFTOVER_MIN_AGE_S ago
is passed over without a try at its lock.

One host's kernel keeps that lock, so it tells a live directory from a leftover only between processes
of one host: on a scratch root shared between hosts, a sibling host finds a live directory's lock free.
So the name of a scratch directory carries a tag of the host that made it (scratch_prefix): a digest of
the id Linux gives every start of a kernel (BOOT_ID_PATH), which is the same for every process and
container of one running host and another on every other host and after every reboot, as far as a
lock is seen on a host's own disk; on a host without that id, a digest of its node name, which a
reboot does not change and two hosts may share. The removal looks only at the
directories that carry this host's tag. It never removes a directory with another host's tag (the
directory of an execution that runs on a sibling host, or one this host made before its last reboot)
nor one with no tag (the earlier name, SCRATCH_PREFIX and a random part, which a worker of an earlier
image makes and holds no lock on), and it tries the lock of neither: removing those is an operator's
task. On a host that tells no identity at all it removes nothing.

The scratch root must be on a file system on which a directory can be locked with flock: on a root
that cannot lock, every execution would end, after its preflight, as the runner's retryable failure for
a scratch directory that could not be set up (resource_exhausted / scratch_create_failed).
lock_refused_in() tries that lock on a directory of its own, and the process start refuses a scratch
root on which it is refused (core/job/config.py). The scratch root belongs on the worker's own disk: a
remote file system that two workers under one kernel each mount for themselves keeps their locks
apart, whatever the tag says.

No log line of this module holds a path or an error's own text, which may name one: an error is named
by its class. And the line that tells of an error is written once the `except` block that caught that
error has ended: a log handler that fails on a line does not raise, Python's logging prints a report of
its own with every error that failure is chained to, and inside the block that would be the caught
error, with its own text.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# What the name of every execution's scratch directory starts with, whatever host made it.
SCRATCH_PREFIX = "vh-job-"
# Where Linux tells the id of the kernel's current start: one for every process and container of a
# running host, another on every other host and after every reboot. A test points it at a file of its own.
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
# How many hex digits of the SHA-256 of a host's identity are its tag in a scratch directory's name.
HOST_TAG_HEX_DIGITS = 16
# What stands where the tag would in a name made on a host that tells no identity. No tag can equal
# it: it is not hex.
NO_HOST_TAG = "nohost"
# What ends the tag in a name: the part that tells one directory of a host from another comes after it.
HOST_TAG_END = "-"
# What a tag's digest holds before each kind of identity, so that a boot id and a node name of the
# same text give two tags.
BOOT_ID_KIND = b"boot-id:"
NODE_NAME_KIND = b"node-name:"
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


def scratch_prefix() -> str:
    """What the name of a scratch directory made by this process starts with: SCRATCH_PREFIX, the tag
    of the host this process runs on, and HOST_TAG_END.

    The tag is the first HOST_TAG_HEX_DIGITS hex digits of the SHA-256 of the host's identity, so the
    prefix holds only `[a-z0-9-]` whatever the identity's own text is. The identity is the id of the
    kernel's current start (BOOT_ID_PATH) or, on a host without one, the node name. On a host that
    tells neither, NO_HOST_TAG stands where the tag would. The identity is read at every call; reading
    it never raises, and nothing of it is logged.
    """
    host_tag = _host_tag()
    return _prefix_with(NO_HOST_TAG if host_tag is None else host_tag)


def _prefix_with(host_tag: str) -> str:
    """What the name of a scratch directory starts with when a host of the tag `host_tag` made it."""
    return f"{SCRATCH_PREFIX}{host_tag}{HOST_TAG_END}"


def _host_tag() -> str | None:
    """The tag of the host this process runs on; None on a host that tells no identity."""
    identity = _host_identity()
    if identity is None:
        return None
    return hashlib.sha256(identity).hexdigest()[:HOST_TAG_HEX_DIGITS]


def _host_identity() -> bytes | None:
    """What tells this host from every other one for as long as its kernel runs, its kind first: the
    boot id, else the node name; None when the host tells neither."""
    boot_id = _boot_id()
    if boot_id:
        return BOOT_ID_KIND + boot_id
    node_name = _node_name()
    if node_name:
        return NODE_NAME_KIND + node_name
    return None


def _boot_id() -> bytes:
    """The id of the kernel's current start, without the white space around it; empty on a system that
    has no such file (one that is not Linux: the development host) or does not let it be read."""
    try:
        return BOOT_ID_PATH.read_bytes().strip()
    except OSError:
        return b""


def _node_name() -> bytes:
    """The host's node name; empty when the system does not tell it. Encoded so that no name fails."""
    try:
        return os.uname().nodename.encode("utf-8", "surrogatepass")
    except OSError:
        return b""


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


def lock_refused_in(scratch_root: Path) -> bool:
    """Whether the system refuses the lock an execution takes on its scratch directory, tried on a
    directory in `scratch_root`.

    It makes one directory in `scratch_root`, named as an execution's (scratch_prefix(): a process
    killed in the middle leaves an ordinary leftover, which a later start removes), opens it and asks
    for its lock as an execution does (LOCK_OPEN_FLAGS, LOCK_WITHOUT_WAITING), then closes the
    descriptor and removes the directory, whatever happened.

    True only when the directory was made and opened and the system refused the lock with an OSError
    other than BlockingIOError: a lock that is held is a lock that works. False when the lock was
    taken, and also when it could not be tried, the directory not made or not opened: that is no
    verdict on the lock (a full disk, a root that became unusable), and an execution meets that
    trouble as its own typed, retryable failure.

    Raises TypeError for anything but a Path, and nothing else: a descriptor that cannot be closed and
    a directory that cannot be removed are passed over, and so is a line of its own that cannot be
    written. An interrupt passes through as it is, once the descriptor is closed and the directory
    removed. One warning tells of a lock that could not be tried and one of a directory that could not
    be removed, each by the error's class. No line holds a path.
    """
    if not isinstance(scratch_root, Path):
        raise TypeError("scratch_root must be a Path")
    directory: Path | None = None
    descriptor: int | None = None
    refused = False
    untried_class: str | None = None
    try:
        try:
            directory = Path(tempfile.mkdtemp(prefix=scratch_prefix(), dir=scratch_root))
            descriptor = os.open(directory, LOCK_OPEN_FLAGS)
            refused = _lock_is_refused(descriptor)
        except Exception as error:  # noqa: BLE001 - whatever kept the lock from being tried is no verdict on it
            # Only the class name is kept past this block: the error's own text names the directory.
            untried_class = type(error).__name__
    finally:
        # Behind an interrupt too, and so without a line: both are written below, once nothing is on its way.
        unremoved_class = _cleared_away(descriptor, directory)
    if untried_class is not None:
        _warn("the lock of a directory in the scratch root could not be tried: %s", untried_class)
    if unremoved_class is not None:
        _warn("the directory the scratch root's lock was tried on could not be removed: %s", unremoved_class)
    return refused


def _lock_is_refused(descriptor: int) -> bool:
    """Ask for the lock on `descriptor` as an execution does: True when the system refuses it, False
    when it is taken or another holds it. Raises what the lock call raises that is no OSError."""
    try:
        fcntl.flock(descriptor, LOCK_WITHOUT_WAITING)
    except BlockingIOError:
        # Another holds it: a lock that is held is a lock that works.
        return False
    except OSError:
        return True
    return False


def _cleared_away(descriptor: int | None, directory: Path | None) -> str | None:
    """Close the descriptor the lock was tried on, which lets go of a lock that was taken, and remove
    the directory it was tried on, each where there is one. Returns the class name of the error that
    kept the directory from being removed, None when it is gone. It raises nothing and logs nothing: it
    runs behind an interrupt too."""
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            # Passed over: the directory is removed all the same, and a descriptor that stays open on
            # it keeps no job from its own directory.
            pass
    if directory is None:
        return None
    try:
        os.rmdir(directory)
    except OSError as error:
        # Only the class name leaves this block: the error's own text names the directory.
        return type(error).__name__
    return None


def _warn(line: str, error_class: str) -> None:
    """Write one warning of the lock check, which names an error by its class. A line that cannot be
    written is dropped: the process start asks only whether the lock was refused."""
    try:
        logger.warning(line, error_class)
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return


def remove_leftover_scratch(scratch_root: Path) -> int:
    """Remove the scratch directories a killed process left in `scratch_root`, and return how many.

    It looks only at the direct children of `scratch_root` whose name starts with this host's
    scratch_prefix(). A directory with another host's tag (a sibling host's execution on a shared
    root, or one this host made before its last reboot) and one with no tag (the earlier name,
    SCRATCH_PREFIX and a random part) are left untouched: not examined, their lock not tried, and with
    no line of their own. On a host that tells no identity that is every directory, so nothing is
    removed. Of this host's own it passes over, untouched: whatever is no real directory (a symbolic
    link is neither followed nor removed), a directory this process's user does not own, one that was
    changed less than LEFTOVER_MIN_AGE_S ago, and one whose lock is held (an execution of another
    process runs in it). Every other one is removed, under its lock.

    No error of the root or of a directory leaves it: a root that cannot be searched (one that is no
    Path included) and a directory that cannot be examined or removed are passed over, each with one
    warning that names the error's class. One line tells how many were removed. No line holds a path.
    What one of those lines raises when it cannot be written is not caught here (the entry point
    drops it).
    """
    removed = 0
    error_class: str | None = None
    try:
        for directory in _this_hosts_children(scratch_root):
            removed += int(_removed_as_a_leftover(directory))
    except Exception as error:  # noqa: BLE001 - leftovers that stay never keep a process from starting
        # Only the class name is kept past this block: the error's own text names the root.
        error_class = type(error).__name__
    if error_class is not None:
        logger.warning("the scratch root could not be searched for leftover directories: %s", error_class)
    logger.info("removed %d leftover scratch directories", removed)
    return removed


def _this_hosts_children(scratch_root: Path) -> list[Path]:
    """The direct children of the scratch root that bear the name of a scratch directory this host
    made, whatever they are; none on a host that tells no identity, where no name tells whose a
    directory is."""
    if not isinstance(scratch_root, Path):
        raise TypeError("scratch_root must be a Path")
    host_tag = _host_tag()
    if host_tag is None:
        return []
    prefix = _prefix_with(host_tag)
    with os.scandir(scratch_root) as children:
        names = sorted(child.name for child in children if child.name.startswith(prefix))
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
