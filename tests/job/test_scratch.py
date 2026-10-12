"""The scratch directories a killed process left behind, the lock that tells them from a live one's,
the tag of its host in a scratch directory's name, and the check that a scratch root can lock.

scratch_prefix() is what the name of a scratch directory starts with on the host that makes it: the
prefix of every scratch directory, a tag of that host's running kernel and a dash.
remove_leftover_scratch() looks at the direct children of the scratch root whose name starts with this
host's prefix and removes those that are real directories of this process's user, were not changed
for LEFTOVER_MIN_AGE_S and are locked by nobody. It removes nothing else (no directory of another host,
none without a tag), follows no link, never raises and logs no path. hold_scratch_lock() is the lock an
execution holds on its own directory, and lock_refused_in() tells a scratch root on which the system
refuses that lock.

A directory of "this host" is named with the module's own prefix for the host the suite runs on
(THIS_HOSTS_PREFIX: from its boot id on Linux, from its node name elsewhere). The tests of the prefix
itself, and those that need another host, give the module a boot id file and a node name of their own
(the `host` fixture, _prefix_on_a_host_booted_as): neither /proc nor the real node name decides them,
but for two that read the host's own on purpose (the instrument test of THIS_HOSTS_PREFIX, and the one
that reads the system's boot id where there is one).

The directories are real ones under pytest's tmp_path. No test waits for an age: a directory's time is
set with os.utime, and the module's clock stands still at the test's start (frozen_now), so an age is
exactly what the test set (one test reads the module's own clock, against a directory made that moment).
The lock "of another process" is held by the test itself on a descriptor of its own: flock belongs to
the open file description, so a second open of a directory contends with the first, in one process as
between two. Where the code under test is then called on the test's own thread, that lock lets go by
itself after a while (conftest's held_at_most_a_while): code that waited for it would otherwise hang
the suite, and so it fails its test. Whether a descriptor was left open is told by what this process
holds open on a directory (_descriptors_open_on), never by a count of all its descriptors.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from core.job import scratch as scratch_module
from core.job.scratch import (
    LEFTOVER_MIN_AGE_S,
    SCRATCH_PREFIX,
    hold_scratch_lock,
    lock_refused_in,
    release_scratch_lock,
    remove_leftover_scratch,
    scratch_prefix,
)
from tests.job.conftest import held_at_most_a_while, reported_log_failures

# How long ago a leftover was last changed, unless a test says otherwise: far longer than the least age.
LONG_AGO_S = 2 * 60 * 60
# What the name of a scratch directory starts with when the host the suite runs on made it: the module's
# own answer, read once here from that host's real identity.
THIS_HOSTS_PREFIX = scratch_prefix()
LEFTOVER_NAME = THIS_HOSTS_PREFIX + "abc123"
OTHER_LEFTOVER_NAME = THIS_HOSTS_PREFIX + "def456"
# The name a worker of an earlier image gave its scratch directory: the prefix and no host's tag.
UNTAGGED_NAME = SCRATCH_PREFIX + "abc123"
# What a host that tells no identity names its scratch directories with.
NO_HOST_PREFIX = SCRATCH_PREFIX + "nohost-"
# The boot ids and the node names of the hosts a test makes up.
BOOT_ID = "0b5a8f3e-6d0c-4e1f-9a47-3c2d1e0f9b8a"
OTHER_BOOT_ID = "f1e2d3c4-b5a6-4978-8a9b-0c1d2e3f4a5b"
NODE_NAME = "worker-a"
OTHER_NODE_NAME = "worker-b"
# Where the system the suite runs on tells its boot id, kept from before any test points the module
# elsewhere: a file on Linux (the worker image), nothing on macOS.
THE_SYSTEMS_BOOT_ID_PATH = scratch_module.BOOT_ID_PATH
# What follows the prefix of every scratch directory in the prefix of a host that has an identity (its
# tag and a dash), and what may follow it on any host.
TAG_AND_DASH = re.compile(r"[0-9a-f]{16}-")
SAFE_IN_A_NAME = re.compile(r"[a-z0-9-]+")
SOURCE_BYTES = b"the source a killed execution had downloaded"
KEPT_BYTES = b"a file no removal of leftovers may touch"
PATH_CANARY = "/CANARY/path/in/an/errors/text"
# The own words of two errors of the system, which no report of a failing log may hold.
FULL_DISK_WORDS = "No space left on device"
NOT_PERMITTED_WORDS = "Permission denied"
# What the one line that tells the count says, around its number.
COUNT_LINE_WORDS = "leftover scratch director"
# The module's own clock, kept from before any test stops it (frozen_now).
THE_MODULES_OWN_CLOCK = scratch_module._now_s
# The user id of root, the worker image's user, and the id of the user and the group `nobody`, to whom
# a suite that runs as root gives a directory away.
ROOT_ID = 0
NOBODY_ID = 65534
# Where macOS and Linux list the descriptors a process holds open.
OPEN_DESCRIPTORS = "/dev/fd"
# The mode of a directory whose owner may list it and may not make or remove anything in it, and the
# mode a test puts back.
LISTED_NOT_CHANGED = 0o500
OWNER_ONLY = 0o700


@pytest.fixture(autouse=True)
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> float:
    """The module's clock stands still at the second the test began, so a directory is exactly as old as
    the test made it, however long the test takes."""
    now_s = float(int(time.time()))
    monkeypatch.setattr(scratch_module, "_now_s", lambda: now_s)
    return now_s


def _set_age(path: Path, now_s: float, age_s: float) -> None:
    """Make `path` itself (a link, not what it points to) last changed `age_s` seconds before `now_s`."""
    changed_at_s = now_s - age_s
    os.utime(path, (changed_at_s, changed_at_s), follow_symlinks=False)


def _leftover(root: Path, now_s: float, name: str = LEFTOVER_NAME, *, age_s: float = LONG_AGO_S) -> Path:
    """A scratch directory as a killed execution leaves it: a downloaded source, part of an output and a
    work directory, last changed `age_s` seconds ago."""
    directory = root / name
    for inside in ("source", "output/720p", "work"):
        (directory / inside).mkdir(parents=True)
    (directory / "source" / "source").write_bytes(SOURCE_BYTES)
    (directory / "output" / "720p" / "segment_0000.ts").write_bytes(b"\x47 a segment")
    _set_age(directory, now_s, age_s)
    return directory


def _locked_as_by_a_live_execution(directory: Path) -> int:
    """Hold the exclusive lock on `directory` on a descriptor of the test's own, and return it. A lock
    that is refused (BlockingIOError) leaves nothing open."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _a_shared_lock_is_free(directory: Path) -> bool:
    """Whether another process could take a shared lock on `directory` at this instant, which it can
    unless someone holds the exclusive one. The test's own try at it, on a descriptor of its own, is let
    go at once."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)
    return True


def _identity_of(path: Path) -> tuple[int, int]:
    """What tells a file or a directory from every other one, whatever becomes of its name: its device
    and its inode."""
    status = os.lstat(path)
    return status.st_dev, status.st_ino


def _descriptors_open_on(identity: tuple[int, int]) -> int:
    """How many descriptors this process holds open on the file or directory `identity` tells. A
    directory that was removed while a descriptor was open on it is still what that descriptor is open on."""
    count = 0
    for name in os.listdir(OPEN_DESCRIPTORS):
        try:
            held = os.fstat(int(name))
        except OSError:
            # The descriptor the listing itself used is gone by now.
            continue
        count += (held.st_dev, held.st_ino) == identity
    return count


def _raising(error: BaseException) -> Callable[..., Any]:
    def stand_in(*arguments: Any, **named_arguments: Any) -> Any:
        raise error

    return stand_in


def _written(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every captured log line as a handler writes it, a traceback included where a record carries one."""
    as_a_handler_writes_it = logging.Formatter()
    return [as_a_handler_writes_it.format(record) for record in caplog.records]


class _StandInHost:
    """The host the module takes itself to run on, for the length of a test: the file it reads the boot
    id from, a file of the test's that is not there until the test writes it, and the node name the
    system tells."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
        self._monkeypatch = monkeypatch
        self.boot_id_file = directory / "boot_id"
        monkeypatch.setattr(scratch_module, "BOOT_ID_PATH", self.boot_id_file)
        self.name_the_node(NODE_NAME)

    def boot_as(self, boot_id: str | bytes) -> None:
        """Write the boot id file: exactly `boot_id`, with no line break added."""
        self.boot_id_file.write_bytes(boot_id.encode("utf-8") if isinstance(boot_id, str) else boot_id)

    def name_the_node(self, node_name: str) -> None:
        """Have the system tell `node_name`, and everything else as it does."""
        real = os.uname()
        told = os.uname_result((real.sysname, node_name, real.release, real.version, real.machine))
        self._monkeypatch.setattr(scratch_module.os, "uname", lambda: told)


@pytest.fixture
def host(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> _StandInHost:
    """A host of the test's own making in the place of the one the suite runs on: without a boot id
    until the test gives it one, and named NODE_NAME. Its boot id file lies outside tmp_path."""
    return _StandInHost(monkeypatch, tmp_path_factory.mktemp("host"))


def _prefix_on_a_host_booted_as(boot_id: str, aside: Path) -> str:
    """What the name of a scratch directory starts with on a host whose kernel start has the id
    `boot_id`: the module's own answer while it reads that id from a file in `aside`. Once this returns
    the module reads again what it read before."""
    boot_id_file = aside / "boot_id_of_another_host"
    boot_id_file.write_text(boot_id + "\n", encoding="ascii")
    with pytest.MonkeyPatch.context() as on_that_host:
        on_that_host.setattr(scratch_module, "BOOT_ID_PATH", boot_id_file)
        return scratch_prefix()


def _tree_of(directory: Path) -> list[tuple[str, bytes | None]]:
    """Everything inside `directory`: each path, with a file's bytes."""
    return sorted(
        (inside.relative_to(directory).as_posix(), inside.read_bytes() if inside.is_file() else None)
        for inside in directory.rglob("*")
    )


@dataclass(frozen=True)
class _LockTry:
    """One try at a lock, as the system was asked for it: the descriptor and what it was open on, how it
    was opened and how the lock was asked for, and what the scratch root held at that moment (each
    name, with what it names)."""

    descriptor: int
    opened_on: tuple[int, int]
    is_a_directory: bool
    inheritable: bool
    operation: int
    in_the_root: dict[str, tuple[int, int]]


def _note_the_lock_tries(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, then_raise: BaseException | None = None
) -> list[_LockTry]:
    """Note every try at a lock made from here on, with what `root` holds at that moment. The system is
    then asked for the lock, unless the try is to end as `then_raise`."""
    tries: list[_LockTry] = []
    lock = fcntl.flock

    def note_then_lock(descriptor: int, operation: int) -> None:
        status = os.fstat(descriptor)
        tries.append(
            _LockTry(
                descriptor=descriptor,
                opened_on=(status.st_dev, status.st_ino),
                is_a_directory=stat.S_ISDIR(status.st_mode),
                inheritable=os.get_inheritable(descriptor),
                operation=operation,
                in_the_root={child.name: _identity_of(child) for child in root.iterdir()} if root.is_dir() else {},
            )
        )
        if then_raise is not None:
            raise then_raise
        lock(descriptor, operation)

    monkeypatch.setattr(scratch_module.fcntl, "flock", note_then_lock)
    return tries


class _RaisingLogHandler(logging.Handler):
    """A log handler whose `emit` raises: unlike a stream that fails, its error comes out of the log
    call itself. It counts the lines it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.lines = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.lines += 1
        raise MemoryError(f"no memory left for a line about {PATH_CANARY}")


# --- the tests' own instruments ---------------------------------------------------------------------
# What tests below rely on to tell them something: each is shown here to be able to tell it.


def test_descriptors_open_on_when_a_descriptor_is_open_on_a_directory_should_count_it_until_it_is_closed(tmp_path):
    # Arrange
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()
    identity = _identity_of(directory)

    # Act: while a descriptor is open on the directory, once the directory was removed under it, and
    # once the descriptor is closed
    descriptor = os.open(directory, os.O_RDONLY)
    while_open = _descriptors_open_on(identity)
    directory.rmdir()
    once_the_directory_is_gone = _descriptors_open_on(identity)
    os.close(descriptor)
    once_closed = _descriptors_open_on(identity)

    # Assert: a test that finds no descriptor left open could have found one, on this platform
    assert (while_open, once_the_directory_is_gone, once_closed) == (1, 1, 0)


def test_a_shared_lock_is_free_when_the_exclusive_lock_is_held_should_say_so_until_it_is_let_go(tmp_path):
    # Arrange
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()

    # Act
    free_before = _a_shared_lock_is_free(directory)
    held = _locked_as_by_a_live_execution(directory)
    free_while_held = _a_shared_lock_is_free(directory)
    os.close(held)
    free_after = _a_shared_lock_is_free(directory)

    # Assert: and the tries themselves left no lock behind
    assert (free_before, free_while_held, free_after) == (True, False, True)


def test_scratch_prefix_when_asked_on_the_host_the_suite_runs_on_should_carry_a_tag_and_stay_as_it_was_read():
    # Act
    prefix = scratch_prefix()

    # Assert: the names the tests below give "this host's" directories are of a host with an identity,
    # and what was read of it when this module was loaded still holds
    assert prefix == THIS_HOSTS_PREFIX
    assert TAG_AND_DASH.fullmatch(prefix.removeprefix(SCRATCH_PREFIX))


def test_prefix_on_a_host_booted_as_when_it_returns_should_leave_the_module_with_this_hosts_own_prefix(tmp_path):
    # Act
    on_another_host = _prefix_on_a_host_booted_as(OTHER_BOOT_ID, tmp_path)

    # Assert: a tagged prefix that is not this host's, and the module is back on this host
    assert TAG_AND_DASH.fullmatch(on_another_host.removeprefix(SCRATCH_PREFIX))
    assert on_another_host != THIS_HOSTS_PREFIX
    assert scratch_prefix() == THIS_HOSTS_PREFIX


# --- the name of a scratch directory: the tag of its host ---------------------------------------------


@pytest.mark.parametrize(
    ("boot_id", "node_name", "has_a_tag"),
    [
        pytest.param(BOOT_ID + "\n", NODE_NAME, True, id="a-host-with-a-boot-id"),
        pytest.param(None, NODE_NAME, True, id="a-host-with-a-node-name-only"),
        pytest.param(None, "", False, id="a-host-with-no-identity"),
    ],
)
def test_scratch_prefix_when_asked_twice_should_give_one_prefix_that_a_name_can_begin_with(
    host, boot_id, node_name, has_a_tag
):
    # Arrange
    if boot_id is not None:
        host.boot_as(boot_id)
    host.name_the_node(node_name)

    # Act
    first, second = scratch_prefix(), scratch_prefix()

    # Assert: the prefix of every scratch directory, then nothing but `[a-z0-9-]`, a dash last; where the
    # host has an identity, a tag of sixteen hex digits
    rest = first.removeprefix(SCRATCH_PREFIX)
    assert first == second
    assert first.startswith(SCRATCH_PREFIX) and first.endswith("-")
    assert SAFE_IN_A_NAME.fullmatch(rest)
    assert (TAG_AND_DASH.fullmatch(rest) is not None) == has_a_tag


@pytest.mark.parametrize("told_by", ["the-boot-id", "the-node-name"])
@pytest.mark.parametrize(
    "identity",
    [
        pytest.param(b"../../../outside/the/root", id="path-separators"),
        pytest.param(b"UPPER case, spaces: and % \"quotes\"", id="characters-a-scratch-path-may-not-hold"),
        pytest.param("\u00e9t\u00e9-\u8996\u983b".encode("utf-8"), id="letters-of-another-script"),
        pytest.param(b"\xff\xfe not text \x01", id="bytes-that-are-no-text"),
        pytest.param(b"x" * 4096, id="a-very-long-text"),
    ],
)
def test_scratch_prefix_when_the_identity_holds_any_text_should_still_hold_only_what_a_name_can(
    host, told_by, identity
):
    # Arrange: the system tells a node name as text, with what is no text escaped
    if told_by == "the-boot-id":
        host.boot_as(identity)
    else:
        host.name_the_node(os.fsdecode(identity))

    # Act
    prefix = scratch_prefix()

    # Assert: nothing of the identity's own text is in the name
    assert TAG_AND_DASH.fullmatch(prefix.removeprefix(SCRATCH_PREFIX))


def test_scratch_prefix_when_two_hosts_have_different_boot_ids_should_give_each_its_own_prefix(host):
    # Act: one host, another, and the first again
    host.boot_as(BOOT_ID + "\n")
    first = scratch_prefix()
    host.boot_as(OTHER_BOOT_ID + "\n")
    other = scratch_prefix()
    host.boot_as(BOOT_ID + "\n")
    first_again = scratch_prefix()

    # Assert
    assert first != other
    assert first_again == first


def test_scratch_prefix_when_the_host_has_a_boot_id_should_not_change_with_its_node_name(host):
    # Arrange: two containers of one running host have a node name each, and one kernel, whose locks both see
    host.boot_as(BOOT_ID + "\n")

    # Act
    in_one_container = scratch_prefix()
    host.name_the_node(OTHER_NODE_NAME)
    in_another_container = scratch_prefix()

    # Assert
    assert in_another_container == in_one_container


@pytest.mark.parametrize(
    "written",
    [
        pytest.param(BOOT_ID + "\n", id="a-line-break-after-it"),
        pytest.param("  " + BOOT_ID + " \t\r\n", id="spaces-around-it"),
        pytest.param("\n\n" + BOOT_ID, id="line-breaks-before-it"),
    ],
)
def test_scratch_prefix_when_white_space_surrounds_the_boot_id_should_give_the_prefix_of_the_bare_id(host, written):
    # Arrange
    host.boot_as(BOOT_ID)
    of_the_bare_id = scratch_prefix()
    host.boot_as(written)

    # Act
    prefix = scratch_prefix()

    # Assert
    assert prefix == of_the_bare_id


def _leave_no_boot_id_file(host: _StandInHost) -> None:
    """The host has no boot id file, as the fixture left it."""


def _put_a_directory_where_the_boot_id_file_is(host: _StandInHost) -> None:
    host.boot_id_file.mkdir()


def _take_the_boot_id_file_from_its_reader(host: _StandInHost) -> None:
    host.boot_as(BOOT_ID + "\n")
    host.boot_id_file.chmod(0)


@pytest.mark.parametrize(
    "without_a_boot_id",
    [
        pytest.param(_leave_no_boot_id_file, id="no-boot-id-file"),
        pytest.param(_put_a_directory_where_the_boot_id_file_is, id="a-directory-in-its-place"),
        pytest.param(
            _take_the_boot_id_file_from_its_reader,
            marks=pytest.mark.skipif(os.geteuid() == ROOT_ID, reason="root reads a file whatever its mode"),
            id="a-file-this-process-may-not-read",
        ),
        pytest.param(lambda host: host.boot_as(b""), id="an-empty-file"),
        pytest.param(lambda host: host.boot_as(b" \n\t\r\n"), id="a-file-of-white-space"),
    ],
)
def test_scratch_prefix_when_the_host_tells_no_boot_id_should_go_by_its_node_name_and_log_nothing(
    host, caplog, without_a_boot_id
):
    # Arrange: what the node name alone gives, on a host that has no boot id file at all
    caplog.set_level(logging.DEBUG)
    by_the_node_name_alone = scratch_prefix()
    without_a_boot_id(host)

    # Act
    prefix = scratch_prefix()
    host.name_the_node(OTHER_NODE_NAME)
    under_another_node_name = scratch_prefix()

    # Assert: a tag of the node name, and another node name has another
    assert TAG_AND_DASH.fullmatch(prefix.removeprefix(SCRATCH_PREFIX))
    assert prefix == by_the_node_name_alone
    assert under_another_node_name != prefix
    assert caplog.records == []


def test_scratch_prefix_when_a_boot_id_and_a_node_name_are_the_same_text_should_give_two_prefixes(host):
    # Arrange: one host is named what another's kernel start is called
    same_text = "one-text-for-both"
    host.name_the_node(same_text)

    # Act
    by_the_node_name = scratch_prefix()
    host.boot_as(same_text)
    by_the_boot_id = scratch_prefix()

    # Assert
    assert by_the_boot_id != by_the_node_name


def _tell_no_node_name(host: _StandInHost, monkeypatch: pytest.MonkeyPatch) -> None:
    host.name_the_node("")


def _fail_to_tell_the_node_name(host: _StandInHost, monkeypatch: pytest.MonkeyPatch) -> None:
    refused = OSError(errno.EPERM, "Operation not permitted", PATH_CANARY)
    monkeypatch.setattr(scratch_module.os, "uname", _raising(refused))


@pytest.mark.parametrize(
    "without_a_node_name",
    [
        pytest.param(_tell_no_node_name, id="an-empty-node-name"),
        pytest.param(_fail_to_tell_the_node_name, id="a-system-that-does-not-tell-it"),
    ],
)
def test_scratch_prefix_when_the_host_tells_no_identity_should_be_the_nohost_prefix_and_log_nothing(
    host, monkeypatch, caplog, without_a_node_name
):
    # Arrange: no boot id file, and no node name either
    caplog.set_level(logging.DEBUG)
    without_a_node_name(host, monkeypatch)

    # Act
    prefix = scratch_prefix()

    # Assert: a fixed word where a tag would be, which is no tag: it is not hex
    assert prefix == NO_HOST_PREFIX
    assert TAG_AND_DASH.fullmatch(prefix.removeprefix(SCRATCH_PREFIX)) is None
    assert caplog.records == []


@pytest.mark.skipif(
    not THE_SYSTEMS_BOOT_ID_PATH.is_file(), reason="this system tells no boot id (it is not Linux)"
)
def test_scratch_prefix_when_the_system_tells_a_boot_id_should_go_by_it_and_not_by_the_node_name(monkeypatch):
    # Arrange: the system's own boot id file, as the worker image has it; only the node name is the test's
    real = os.uname()
    renamed = os.uname_result((real.sysname, real.nodename + "-renamed", real.release, real.version, real.machine))
    monkeypatch.setattr(scratch_module.os, "uname", lambda: renamed)

    # Act
    prefix = scratch_prefix()

    # Assert
    assert prefix == THIS_HOSTS_PREFIX


def test_scratch_prefix_when_the_boot_id_file_is_named_should_be_the_one_linux_keeps():
    # Assert: the id of the running kernel's start, which every container of the host reads the same
    assert THE_SYSTEMS_BOOT_ID_PATH == Path("/proc/sys/kernel/random/boot_id")


# --- what is removed ------------------------------------------------------------------------------


def test_remove_leftover_scratch_when_a_killed_process_left_directories_should_remove_them_and_return_how_many(
    tmp_path, frozen_now
):
    # Arrange
    leftovers = [_leftover(tmp_path, frozen_now, LEFTOVER_NAME), _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)]

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 2
    assert not any(leftover.exists() for leftover in leftovers)
    assert list(tmp_path.iterdir()) == []


def test_remove_leftover_scratch_when_nothing_was_left_should_return_zero(tmp_path):
    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 0


def test_remove_leftover_scratch_when_it_removed_a_leftover_should_leave_no_descriptor_open_on_it(
    tmp_path, frozen_now
):
    # Arrange
    leftover = _leftover(tmp_path, frozen_now)
    identity = _identity_of(leftover)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: the descriptor the removal's own lock lived on is closed, as it is after a removal that failed
    assert removed == 1 and not leftover.exists()
    assert _descriptors_open_on(identity) == 0


def test_remove_leftover_scratch_when_it_removes_a_leftover_should_hold_the_exclusive_lock_while_it_does(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange: note, the moment the leftover is about to be removed, whether another process could take
    # even a shared lock on it
    leftover = _leftover(tmp_path, frozen_now)
    free_for_another: list[bool] = []
    remove_tree = shutil.rmtree

    def note_the_lock_then_remove(path: Any, *arguments: Any, **named_arguments: Any) -> None:
        free_for_another.append(_a_shared_lock_is_free(Path(path)))
        remove_tree(path, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.shutil, "rmtree", note_the_lock_then_remove)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: an execution that took this directory for its own could not have locked it meanwhile
    assert removed == 1 and not leftover.exists()
    assert free_for_another == [False]


@pytest.mark.parametrize(
    ("age_s", "is_removed"),
    [
        pytest.param(LEFTOVER_MIN_AGE_S, True, id="exactly-the-least-age"),
        pytest.param(LEFTOVER_MIN_AGE_S - 1, False, id="a-second-younger"),
        pytest.param(0, False, id="changed-this-second"),
        pytest.param(-5, False, id="changed-in-the-future-of-a-stepped-clock"),
    ],
)
def test_remove_leftover_scratch_when_a_directory_is_of_some_age_should_remove_it_only_at_the_least_age(
    tmp_path, frozen_now, age_s, is_removed
):
    # Arrange
    directory = _leftover(tmp_path, frozen_now, age_s=age_s)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == int(is_removed)
    assert directory.exists() != is_removed


@pytest.mark.parametrize(
    ("changed_s_ago", "read_s_ago", "is_removed"),
    [
        pytest.param(LONG_AGO_S, 0, True, id="changed-long-ago-and-read-this-second"),
        pytest.param(0, LONG_AGO_S, False, id="changed-this-second-and-read-long-ago"),
    ],
)
def test_remove_leftover_scratch_when_a_directory_was_read_at_another_time_than_changed_should_go_by_its_change(
    tmp_path, frozen_now, changed_s_ago, read_s_ago, is_removed
):
    # Arrange: the directory's last read and its last change lie on opposite sides of the least age
    directory = _leftover(tmp_path, frozen_now)
    os.utime(directory, (frozen_now - read_s_ago, frozen_now - changed_s_ago))

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: a leftover is one nothing was made or removed in for long enough, whoever looked into it since
    assert removed == int(is_removed)
    assert directory.exists() != is_removed


def test_remove_leftover_scratch_when_its_own_clock_is_read_should_keep_a_directory_made_now_and_remove_an_old_one(
    tmp_path, monkeypatch
):
    # Arrange: the module's own clock, which every other test stops. A directory made this moment, its
    # time left as the file system set it, beside one last changed two hours ago by the real clock
    monkeypatch.setattr(scratch_module, "_now_s", THE_MODULES_OWN_CLOCK)
    made_now = tmp_path / LEFTOVER_NAME
    (made_now / "source").mkdir(parents=True)
    old = _leftover(tmp_path, time.time(), OTHER_LEFTOVER_NAME)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 1
    assert made_now.is_dir() and not old.exists()


def test_remove_leftover_scratch_when_the_least_age_is_read_should_be_a_minute():
    # Assert: long enough for a live execution to lock the directory it has just made
    assert LEFTOVER_MIN_AGE_S == 60


# --- what is passed over --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["other-abc123", SCRATCH_PREFIX.rstrip("-"), "x" + LEFTOVER_NAME])
def test_remove_leftover_scratch_when_a_name_does_not_start_with_the_scratch_prefix_should_leave_it(
    tmp_path, frozen_now, name
):
    # Arrange: an old directory of this user, locked by nobody, under another name
    directory = _leftover(tmp_path, frozen_now, name)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 0
    assert (directory / "source" / "source").read_bytes() == SOURCE_BYTES


# The names of scratch directories this host did not make, or cannot be known to have made: each takes
# a directory beside the scratch root, where it may put a file of its own.
NOT_THIS_HOSTS = [
    pytest.param(lambda aside: _prefix_on_a_host_booted_as(OTHER_BOOT_ID, aside) + "abc123", id="another-hosts-tag"),
    pytest.param(lambda aside: UNTAGGED_NAME, id="no-tag"),
    pytest.param(lambda aside: SCRATCH_PREFIX + "k3_x9q2a", id="no-tag-as-an-earlier-image-named-it"),
    pytest.param(lambda aside: NO_HOST_PREFIX + "abc123", id="the-word-of-a-host-with-no-identity"),
    pytest.param(lambda aside: THIS_HOSTS_PREFIX.rstrip("-") + "abc123", id="this-hosts-tag-and-no-dash-after-it"),
]


@pytest.mark.parametrize("name_of", NOT_THIS_HOSTS)
def test_remove_leftover_scratch_when_a_leftover_in_every_other_way_is_not_this_hosts_should_leave_it_and_its_files(
    tmp_path, frozen_now, caplog, name_of
):
    # Arrange: a real directory of this process's user, last changed long ago and locked by nobody, with
    # the files of an execution in it. On a root shared between hosts it may be the directory of an
    # execution that runs on another host, whose lock this host's kernel does not see
    caplog.set_level(logging.DEBUG)
    root, aside = tmp_path / "scratch", tmp_path / "aside"
    root.mkdir()
    aside.mkdir()
    directory = _leftover(root, frozen_now, name_of(aside))
    held_before = _tree_of(directory)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert: untouched, its files and its own time included, and without a line of its own
    assert removed == 0
    assert _tree_of(directory) == held_before
    assert (directory / "source" / "source").read_bytes() == SOURCE_BYTES
    assert os.lstat(directory).st_mtime == frozen_now - LONG_AGO_S
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert COUNT_LINE_WORDS in caplog.records[0].getMessage() and " 0 " in caplog.records[0].getMessage()


@pytest.mark.parametrize("name_of", NOT_THIS_HOSTS)
def test_remove_leftover_scratch_when_a_directory_is_not_this_hosts_should_neither_open_it_nor_try_its_lock(
    tmp_path, frozen_now, monkeypatch, name_of
):
    # Arrange: beside it a leftover of this host, which shows that an opening and a try at a lock are
    # noted; both directories are leftovers in every other way
    root, aside = tmp_path / "scratch", tmp_path / "aside"
    root.mkdir()
    aside.mkdir()
    not_this_hosts = _leftover(root, frozen_now, name_of(aside))
    this_hosts = _leftover(root, frozen_now, LEFTOVER_NAME)
    identities = {_identity_of(not_this_hosts): "not this host's", _identity_of(this_hosts): "this host's"}
    opened: list[str] = []
    open_file = os.open

    def note_then_open(opened_path: Any, flags: int, *arguments: Any, **named_arguments: Any) -> int:
        opened.append(os.fspath(opened_path))
        return open_file(opened_path, flags, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.os, "open", note_then_open)
    tries = _note_the_lock_tries(monkeypatch, root)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert: a lock taken, however briefly, on a directory another host's execution works in would
    # tell nothing, and one on a directory of this host's earlier image could fail a job
    assert removed == 1
    assert [identities[tried.opened_on] for tried in tries] == ["this host's"]
    assert str(this_hosts) in opened and str(not_this_hosts) not in opened
    assert not this_hosts.exists()
    assert (not_this_hosts / "source" / "source").read_bytes() == SOURCE_BYTES


def test_remove_leftover_scratch_when_the_host_was_started_anew_since_should_leave_what_it_made_before(
    tmp_path, frozen_now, host
):
    # Arrange: a directory a killed worker left before the host's last reboot, and one a killed worker
    # left since; nothing tells a directory of an earlier start of this host from another host's
    host.boot_as(BOOT_ID + "\n")
    made_before_the_reboot = _leftover(tmp_path, frozen_now, scratch_prefix() + "abc123")
    host.boot_as(OTHER_BOOT_ID + "\n")
    made_since = _leftover(tmp_path, frozen_now, scratch_prefix() + "abc123")

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 1
    assert not made_since.exists()
    assert (made_before_the_reboot / "source" / "source").read_bytes() == SOURCE_BYTES


def test_remove_leftover_scratch_when_the_host_tells_no_identity_should_remove_nothing(
    tmp_path, frozen_now, host, monkeypatch, caplog
):
    # Arrange: a host with neither a boot id nor a node name. In its scratch root, leftovers in every
    # other way: one named as this very process names its own, one with a host's tag and one with none
    caplog.set_level(logging.DEBUG)
    host.name_the_node("")
    own_prefix = scratch_prefix()
    directories = [
        _leftover(tmp_path, frozen_now, name)
        for name in (own_prefix + "abc123", THIS_HOSTS_PREFIX + "abc123", UNTAGGED_NAME)
    ]
    tries = _note_the_lock_tries(monkeypatch, tmp_path)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: no name tells which directories this host made, so none is taken for its leftover
    assert own_prefix == NO_HOST_PREFIX
    assert removed == 0
    assert [(directory / "source" / "source").read_bytes() for directory in directories] == [SOURCE_BYTES] * 3
    assert tries == []
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert " 0 " in caplog.records[0].getMessage()


def test_remove_leftover_scratch_when_a_leftover_lies_deeper_than_the_root_should_leave_it(tmp_path, frozen_now):
    # Arrange: a prefixed directory inside a directory that is no leftover
    holder = tmp_path / "another-tool"
    holder.mkdir()
    nested = _leftover(holder, frozen_now)
    _set_age(holder, frozen_now, LONG_AGO_S)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 0
    assert nested.is_dir()


def test_remove_leftover_scratch_when_a_prefixed_name_is_a_regular_file_should_leave_it(
    tmp_path, frozen_now, caplog
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    regular_file = tmp_path / LEFTOVER_NAME
    regular_file.write_bytes(KEPT_BYTES)
    _set_age(regular_file, frozen_now, LONG_AGO_S)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: passed over as what it is, not as a directory that failed to open
    assert removed == 0
    assert regular_file.read_bytes() == KEPT_BYTES
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_remove_leftover_scratch_when_a_prefixed_name_is_a_symbolic_link_should_neither_follow_nor_remove_it(
    tmp_path, frozen_now, caplog
):
    # Arrange: an old link with a leftover's name, to an old directory of this user outside the root
    caplog.set_level(logging.DEBUG)
    root, elsewhere = tmp_path / "scratch", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (elsewhere / "kept").write_bytes(KEPT_BYTES)
    _set_age(elsewhere, frozen_now, LONG_AGO_S)
    link = root / LEFTOVER_NAME
    link.symlink_to(elsewhere, target_is_directory=True)
    _set_age(link, frozen_now, LONG_AGO_S)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert: the link itself was examined, never its target, so nothing was even tried on it
    assert removed == 0
    assert link.is_symlink()
    assert (elsewhere / "kept").read_bytes() == KEPT_BYTES
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_remove_leftover_scratch_when_a_prefixed_link_points_into_the_root_should_neither_follow_nor_remove_it(
    tmp_path, frozen_now, caplog
):
    # Arrange: an old link with a leftover's name, to an old directory of this user that lies in the
    # scratch root itself, under a name that is no scratch directory's
    caplog.set_level(logging.DEBUG)
    inside = tmp_path / "another-tool"
    inside.mkdir()
    (inside / "kept").write_bytes(KEPT_BYTES)
    _set_age(inside, frozen_now, LONG_AGO_S)
    link = tmp_path / LEFTOVER_NAME
    link.symlink_to(inside, target_is_directory=True)
    _set_age(link, frozen_now, LONG_AGO_S)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: a target beside the link is as little the removal's as one elsewhere
    assert removed == 0
    assert link.is_symlink()
    assert (inside / "kept").read_bytes() == KEPT_BYTES
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_remove_leftover_scratch_when_a_leftover_holds_a_symbolic_link_should_remove_the_link_and_keep_its_target(
    tmp_path, frozen_now
):
    # Arrange: a leftover whose work directory links to a directory outside the root
    root, elsewhere = tmp_path / "scratch", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (elsewhere / "kept").write_bytes(KEPT_BYTES)
    leftover = _leftover(root, frozen_now)
    (leftover / "work" / "link").symlink_to(elsewhere, target_is_directory=True)
    _set_age(leftover, frozen_now, LONG_AGO_S)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert
    assert removed == 1
    assert not leftover.exists()
    assert (elsewhere / "kept").read_bytes() == KEPT_BYTES


def test_remove_leftover_scratch_when_another_user_owns_a_directory_should_leave_it(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange: this process runs as another user than the one who owns the directory. (The process's
    # user is replaced, not the owner: only root may give a directory away. The next test is root's.)
    directory = _leftover(tmp_path, frozen_now)
    owner = directory.stat().st_uid
    monkeypatch.setattr(scratch_module.os, "geteuid", lambda: owner + 1)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 0
    assert (directory / "source" / "source").read_bytes() == SOURCE_BYTES


def test_remove_leftover_scratch_when_the_process_is_root_and_another_user_owns_a_directory_should_leave_it(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange: root, the worker image's user, could remove anybody's directory. A suite that runs as root
    # gives the directory away for real; under any other user the process is taken for root, and the
    # directory stays that user's
    directory = _leftover(tmp_path, frozen_now)
    if os.geteuid() == ROOT_ID:
        try:
            os.chown(directory, NOBODY_ID, NOBODY_ID)
        except OSError:
            pytest.skip("this root may not give a directory away (it runs without the capability to change an owner)")
        _set_age(directory, frozen_now, LONG_AGO_S)
    else:
        monkeypatch.setattr(scratch_module.os, "geteuid", lambda: ROOT_ID)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: no exemption for root
    assert removed == 0
    assert (directory / "source" / "source").read_bytes() == SOURCE_BYTES


def test_remove_leftover_scratch_when_a_directorys_lock_is_held_should_leave_it_until_the_lock_is_gone(
    tmp_path, frozen_now
):
    # Arrange: an old directory in which an execution of another process still runs, and a leftover
    live = _leftover(tmp_path, frozen_now, LEFTOVER_NAME)
    leftover = _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)
    let_go = held_at_most_a_while(_locked_as_by_a_live_execution(live))

    # Act: one removal while the execution runs, one after its process died (the kernel closed the descriptor)
    try:
        removed_while_it_ran = remove_leftover_scratch(tmp_path)
        still_whole = live.is_dir() and (live / "source" / "source").read_bytes() == SOURCE_BYTES
    finally:
        let_go()
    removed_after_it_died = remove_leftover_scratch(tmp_path)

    # Assert
    assert (removed_while_it_ran, still_whole) == (1, True)
    assert not leftover.exists()
    assert removed_after_it_died == 1
    assert not live.exists()


def test_remove_leftover_scratch_when_a_directory_was_just_made_should_not_even_try_its_lock(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange: a live execution has made its directory and not locked it yet; it locks without waiting,
    # so a removal that held the lock at that instant, however briefly, would fail that execution
    _leftover(tmp_path, frozen_now, age_s=0)
    tried: list[int] = []
    lock = fcntl.flock

    def note_the_try(descriptor: int, operation: int) -> None:
        tried.append(operation)
        lock(descriptor, operation)

    monkeypatch.setattr(scratch_module.fcntl, "flock", note_the_try)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    assert removed == 0
    assert tried == []


def test_remove_leftover_scratch_when_a_directory_is_replaced_while_it_is_examined_should_leave_the_new_one(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange: once the old directory was examined, and before it is opened for its lock, another one
    # takes its name: one that was just made
    root, aside = tmp_path / "scratch", tmp_path / "aside"
    root.mkdir()
    examined = _leftover(root, frozen_now)
    open_file = os.open

    def replace_then_open(path: Any, flags: int, *arguments: Any, **named_arguments: Any) -> int:
        if Path(path) == examined and not aside.exists():
            examined.rename(aside)
            _leftover(root, frozen_now, age_s=0)
        return open_file(path, flags, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.os, "open", replace_then_open)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert: what now bears the name is not what was found old, and stays; the descriptor it was opened
    # on for the lock is closed again
    assert removed == 0
    assert examined.is_dir() and aside.is_dir()
    assert _descriptors_open_on(_identity_of(examined)) == 0


# --- it never raises ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_root", "error_class"),
    [
        pytest.param(lambda tmp_path: tmp_path / "missing", "FileNotFoundError", id="a-root-that-does-not-exist"),
        pytest.param(lambda tmp_path: tmp_path / "a-file", "NotADirectoryError", id="a-root-that-is-a-file"),
        pytest.param(lambda tmp_path: str(tmp_path), "TypeError", id="a-root-given-as-text"),
        pytest.param(lambda tmp_path: None, "TypeError", id="no-root"),
    ],
)
def test_remove_leftover_scratch_when_the_root_cannot_be_searched_should_return_zero_and_log_the_errors_class(
    tmp_path, frozen_now, monkeypatch, caplog, make_root, error_class
):
    # Arrange: a leftover in tmp_path, which is not what the removal is given as a Path to search. The
    # working directory is an empty one of the test's own: a search that took no root for "here" would
    # find nothing there to look at
    caplog.set_level(logging.DEBUG)
    (tmp_path / "a-file").write_bytes(KEPT_BYTES)
    leftover = _leftover(tmp_path, frozen_now)
    (tmp_path / "an-empty-working-directory").mkdir()
    monkeypatch.chdir(tmp_path / "an-empty-working-directory")
    root = make_root(tmp_path)

    # Act
    removed = remove_leftover_scratch(root)

    # Assert: one warning, which names the error's class and nothing of the root, not even its last part
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    root_names = [] if root is None else [str(root), Path(root).name]
    assert removed == 0
    assert leftover.is_dir()
    assert len(warnings) == 1 and warnings[0].endswith(f": {error_class}")
    assert all(str(tmp_path) not in line for line in _written(caplog))
    assert [name for name in root_names if any(name in line for line in _written(caplog))] == []


def test_remove_leftover_scratch_when_a_directory_cannot_be_removed_should_pass_over_it_and_remove_the_others(
    tmp_path, frozen_now, monkeypatch, caplog
):
    # Arrange: the removal of one leftover fails, in words that name a path
    caplog.set_level(logging.DEBUG)
    stuck = _leftover(tmp_path, frozen_now, LEFTOVER_NAME)
    removable = _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)
    remove_tree = shutil.rmtree

    def remove_all_but_the_stuck_one(path: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(path) == stuck:
            raise PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)
        remove_tree(path, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.shutil, "rmtree", remove_all_but_the_stuck_one)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: one warning, by the error's class and with no directory's name; the count is of what is gone
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert removed == 1
    assert stuck.is_dir() and not removable.exists()
    assert len(warnings) == 1 and warnings[0].endswith(": PermissionError")
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(SCRATCH_PREFIX not in line for line in _written(caplog))
    assert all(record.exc_info is None for record in caplog.records)


def test_remove_leftover_scratch_when_a_directory_could_not_be_removed_should_leave_it_unlocked(
    tmp_path, frozen_now, monkeypatch
):
    # Arrange
    stuck = _leftover(tmp_path, frozen_now)
    monkeypatch.setattr(scratch_module.shutil, "rmtree", _raising(OSError(errno.EIO, "Input/output error")))

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: the removal's own lock is gone with its failure, so the next one can try again
    assert removed == 0
    release_scratch_lock(hold_scratch_lock(stuck))


def test_remove_leftover_scratch_when_a_directory_cannot_be_examined_should_pass_over_it_and_remove_the_others(
    tmp_path, frozen_now, monkeypatch, caplog
):
    # Arrange: no descriptor is left when one leftover is opened for its lock
    caplog.set_level(logging.DEBUG)
    unopened = _leftover(tmp_path, frozen_now, LEFTOVER_NAME)
    removable = _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)
    open_file = os.open

    def open_all_but_one(path: Any, flags: int, *arguments: Any, **named_arguments: Any) -> int:
        if Path(path) == unopened:
            raise OSError(errno.EMFILE, "Too many open files", PATH_CANARY)
        return open_file(path, flags, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.os, "open", open_all_but_one)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert removed == 1
    assert unopened.is_dir() and not removable.exists()
    assert len(warnings) == 1 and warnings[0].endswith(": OSError")
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(SCRATCH_PREFIX not in line for line in _written(caplog))


def test_remove_leftover_scratch_when_a_leftover_vanished_before_it_is_examined_should_still_look_at_the_others(
    tmp_path, frozen_now, monkeypatch, caplog
):
    # Arrange: of two leftovers, the one that is looked at first is gone the moment it is to be
    # examined, as when another process's removal took it meanwhile
    caplog.set_level(logging.DEBUG)
    vanishing = _leftover(tmp_path, frozen_now, LEFTOVER_NAME)
    other = _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)
    examine = os.lstat
    vanished: list[Path] = []

    def vanish_then_examine(path: Any, *arguments: Any, **named_arguments: Any) -> os.stat_result:
        if not vanished and Path(path) == vanishing:
            vanished.append(vanishing)
            shutil.rmtree(vanishing)
        return examine(path, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.os, "lstat", vanish_then_examine)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert: one warning for the one that could not be examined, and the search went on
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert vanished == [vanishing]
    assert removed == 1
    assert not other.exists()
    assert len(warnings) == 1 and warnings[0].endswith(": FileNotFoundError")
    assert all(str(tmp_path) not in line and SCRATCH_PREFIX not in line for line in _written(caplog))


@pytest.mark.skipif(os.geteuid() == ROOT_ID, reason="root removes what a directory's mode forbids its owner to remove")
def test_remove_leftover_scratch_when_a_leftover_really_cannot_be_removed_should_not_count_it_and_warn_once(
    tmp_path, frozen_now, caplog
):
    # Arrange: a leftover whose source directory its owner may list and may not change, so that the file
    # in it cannot be removed: no stand-in fails here, the operating system does
    caplog.set_level(logging.DEBUG)
    stuck = _leftover(tmp_path, frozen_now)
    os.chmod(stuck / "source", LISTED_NOT_CHANGED)
    _set_age(stuck, frozen_now, LONG_AGO_S)

    # Act: the mode is put back whatever happens, so that pytest can remove its own directory
    try:
        removed = remove_leftover_scratch(tmp_path)
        kept = (stuck / "source" / "source").read_bytes()
    finally:
        os.chmod(stuck / "source", OWNER_ONLY)

    # Assert: what is still there is not counted as removed, and one warning tells of it by the error's class
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert removed == 0
    assert kept == SOURCE_BYTES
    assert len(warnings) == 1 and warnings[0].endswith(": PermissionError")
    assert all(str(tmp_path) not in line and SCRATCH_PREFIX not in line for line in _written(caplog))


# --- what it logs ---------------------------------------------------------------------------------


@pytest.mark.parametrize("leftovers", [0, 2], ids=["nothing-left", "two-left"])
def test_remove_leftover_scratch_when_it_ends_should_log_one_line_with_the_count_and_no_path(
    tmp_path, frozen_now, caplog, leftovers
):
    # Arrange: leftovers to remove, and beside them a live directory and a file that are passed over
    caplog.set_level(logging.DEBUG)
    names = [LEFTOVER_NAME, OTHER_LEFTOVER_NAME][:leftovers]
    for name in names:
        _leftover(tmp_path, frozen_now, name)
    live = _leftover(tmp_path, frozen_now, THIS_HOSTS_PREFIX + "live")
    let_go = held_at_most_a_while(_locked_as_by_a_live_execution(live))
    (tmp_path / (THIS_HOSTS_PREFIX + "file")).write_bytes(KEPT_BYTES)

    # Act
    try:
        removed = remove_leftover_scratch(tmp_path)
    finally:
        let_go()

    # Assert
    lines = _written(caplog)
    assert removed == leftovers
    assert len(lines) == 1 and COUNT_LINE_WORDS in lines[0] and f" {leftovers} " in lines[0]
    assert caplog.records[0].levelno == logging.INFO
    assert all(str(tmp_path) not in line and SCRATCH_PREFIX not in line for line in lines)


# --- the lock an execution holds --------------------------------------------------------------------


def test_hold_scratch_lock_when_the_directory_is_locked_by_nobody_should_hold_it_until_it_is_released(tmp_path):
    # Arrange
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()

    # Act
    held = hold_scratch_lock(directory)
    try:
        with pytest.raises(BlockingIOError):
            _locked_as_by_a_live_execution(directory)
    finally:
        release_scratch_lock(held)
    taken_after_the_release = _locked_as_by_a_live_execution(directory)

    # Assert: nobody else could take it while it was held (above), and it is free again; neither the
    # lock that was released nor the test's own refused try left a descriptor open on the directory
    os.close(taken_after_the_release)
    assert _descriptors_open_on(_identity_of(directory)) == 0


def test_hold_scratch_lock_when_another_holds_the_lock_should_raise_at_once(tmp_path):
    # Arrange
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()
    let_go = held_at_most_a_while(_locked_as_by_a_live_execution(directory))

    # Act and Assert: without waiting for the other to let go (a lock that was got all the same is
    # given back at once)
    try:
        with pytest.raises(BlockingIOError):
            release_scratch_lock(hold_scratch_lock(directory))
    finally:
        let_go()


def test_hold_scratch_lock_when_another_holds_the_lock_should_leave_no_descriptor_of_its_own_open(tmp_path):
    # Arrange
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()
    identity = _identity_of(directory)
    let_go = held_at_most_a_while(_locked_as_by_a_live_execution(directory))

    # Act
    try:
        with pytest.raises(BlockingIOError):
            hold_scratch_lock(directory)
        open_after_the_refusal = _descriptors_open_on(identity)
    finally:
        let_go()

    # Assert: the other's descriptor, and none of the try that was refused
    assert open_after_the_refusal == 1


def test_hold_scratch_lock_when_the_path_is_a_symbolic_link_should_refuse_it_and_lock_nothing(tmp_path):
    # Arrange
    target, link = tmp_path / "target", tmp_path / LEFTOVER_NAME
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)

    # Act and Assert
    with pytest.raises(OSError):
        hold_scratch_lock(link)
    os.close(_locked_as_by_a_live_execution(target))


def test_hold_scratch_lock_when_the_path_is_a_regular_file_should_refuse_it(tmp_path):
    # Arrange
    regular_file = tmp_path / LEFTOVER_NAME
    regular_file.write_bytes(KEPT_BYTES)

    # Act and Assert
    with pytest.raises(NotADirectoryError):
        hold_scratch_lock(regular_file)


@pytest.mark.parametrize(
    "no_path_of",
    [
        pytest.param(lambda tmp_path: str(tmp_path), id="text"),
        pytest.param(lambda tmp_path: None, id="none"),
        pytest.param(lambda tmp_path: 7, id="a-number"),
    ],
)
def test_hold_scratch_lock_when_the_directory_is_no_path_should_raise_type_error(tmp_path, no_path_of):
    # Arrange: the text is the test's own directory: were it ever taken for a path, nothing else is locked
    directory = no_path_of(tmp_path)

    # Act and Assert
    with pytest.raises(TypeError):
        hold_scratch_lock(directory)  # type: ignore[arg-type]


def test_release_scratch_lock_when_the_descriptor_cannot_be_closed_should_log_the_errors_class_and_not_raise(
    tmp_path, monkeypatch, caplog
):
    # Arrange: the close of the lock's own descriptor fails, in words that name a path
    caplog.set_level(logging.DEBUG)
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()
    held = hold_scratch_lock(directory)
    close = os.close

    def close_all_but_the_locks(descriptor: int) -> None:
        if descriptor == held:
            raise OSError(errno.EIO, "Input/output error", PATH_CANARY)
        close(descriptor)

    # Act
    with monkeypatch.context() as while_released:
        while_released.setattr(scratch_module.os, "close", close_all_but_the_locks)
        release_scratch_lock(held)
    close(held)

    # Assert
    assert [record.getMessage() for record in caplog.records] == [
        "a scratch directory's lock could not be released: OSError"
    ]
    assert all(PATH_CANARY not in line for line in _written(caplog))
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("descriptor", [True, "7", None], ids=["a-bool", "text", "none"])
def test_release_scratch_lock_when_given_no_descriptor_should_raise_type_error(monkeypatch, descriptor):
    # Arrange: a recorder stands where the module closes a descriptor. True is the number 1 to the
    # operating system: were it ever closed for real, the suite would lose its standard output
    closed: list[Any] = []

    # Act
    with monkeypatch.context() as while_released:
        while_released.setattr(scratch_module.os, "close", closed.append)
        with pytest.raises(TypeError):
            release_scratch_lock(descriptor)  # type: ignore[arg-type]

    # Assert: refused before anything was closed
    assert closed == []


# --- a scratch root that cannot lock ----------------------------------------------------------------
# lock_refused_in() makes a directory of its own in the root and tries the lock an execution takes. The
# system's refusal is a stand-in's (no file system of the suite refuses a lock): the try is noted with
# what the root held at that moment, and then ends as the test says.


def test_lock_refused_in_when_a_directory_in_the_root_can_be_locked_should_say_no_and_leave_nothing_behind(
    tmp_path, monkeypatch, caplog
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    tries = _note_the_lock_tries(monkeypatch, tmp_path)

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: one lock was asked of the system and given; the directory is gone, its descriptor closed
    (tried,) = tries
    assert refused is False
    assert list(tmp_path.iterdir()) == []
    assert _descriptors_open_on(tried.opened_on) == 0
    assert caplog.records == []


def test_lock_refused_in_when_it_tries_the_lock_should_ask_as_an_execution_does_on_a_new_directory_of_this_host(
    tmp_path, monkeypatch
):
    # Arrange: something of another tool lies in the root
    kept = tmp_path / "kept-by-another-tool"
    kept.write_bytes(KEPT_BYTES)
    tries = _note_the_lock_tries(monkeypatch, tmp_path)

    # Act
    lock_refused_in(tmp_path)

    # Assert: the exclusive lock, without waiting, on a directory it made in the root under this host's
    # prefix (killed there, it leaves a leftover like any other), opened so that no child inherits it;
    # and nothing else in the root was touched
    (tried,) = tries
    (made,) = [name for name in tried.in_the_root if name != kept.name]
    assert made.startswith(scratch_prefix())
    assert tried.in_the_root[made] == tried.opened_on
    assert tried.is_a_directory and not tried.inheritable
    assert tried.operation == fcntl.LOCK_EX | fcntl.LOCK_NB
    assert [child.name for child in tmp_path.iterdir()] == [kept.name]
    assert kept.read_bytes() == KEPT_BYTES


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(OSError(errno.ENOLCK, "No locks available", PATH_CANARY), id="no-locks-available"),
        pytest.param(OSError(errno.EOPNOTSUPP, "Operation not supported", PATH_CANARY), id="not-supported"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument", PATH_CANARY), id="an-invalid-argument"),
        pytest.param(PermissionError(errno.EACCES, "Permission denied", PATH_CANARY), id="not-permitted"),
    ],
)
def test_lock_refused_in_when_the_system_refuses_the_lock_should_say_yes_and_leave_nothing_behind(
    tmp_path, monkeypatch, caplog, refusal
):
    # Arrange: the directory is made and opened, and the system refuses its lock
    caplog.set_level(logging.DEBUG)
    tries = _note_the_lock_tries(monkeypatch, tmp_path, then_raise=refusal)

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: refused, once; the directory is gone and its descriptor closed, and no line tells a path
    (tried,) = tries
    assert refused is True
    assert list(tmp_path.iterdir()) == []
    assert _descriptors_open_on(tried.opened_on) == 0
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(record.exc_info is None for record in caplog.records)


def test_lock_refused_in_when_another_holds_the_lock_of_its_directory_should_say_no_and_leave_nothing_behind(
    tmp_path, monkeypatch
):
    # Arrange: the instant the directory is made, another process holds its lock (for a while at most:
    # a check that waited for the lock would otherwise never return)
    letting_go: list[Callable[[], None]] = []
    locked: list[tuple[int, int]] = []
    make = tempfile.mkdtemp

    def make_a_directory_another_locks(*arguments: Any, **named_arguments: Any) -> str:
        made = Path(make(*arguments, **named_arguments))
        locked.append(_identity_of(made))
        letting_go.append(held_at_most_a_while(_locked_as_by_a_live_execution(made)))
        return str(made)

    monkeypatch.setattr(scratch_module.tempfile, "mkdtemp", make_a_directory_another_locks)

    # Act
    try:
        refused = lock_refused_in(tmp_path)
        open_on_it_afterwards = _descriptors_open_on(locked[0])
    finally:
        for let_go in letting_go:
            let_go()

    # Assert: a lock that is held is a lock that works. The other's descriptor is still open on the
    # directory, and none of the check's own
    assert refused is False
    assert open_on_it_afterwards == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError(errno.ENOSPC, "No space left on device", PATH_CANARY), id="a-full-disk"),
        pytest.param(PermissionError(errno.EACCES, "Permission denied", PATH_CANARY), id="a-root-closed-since"),
        pytest.param(OSError(errno.EIO, "Input/output error", PATH_CANARY), id="an-error-of-the-disk"),
    ],
)
def test_lock_refused_in_when_its_directory_cannot_be_made_should_say_no_and_warn_by_the_errors_class(
    tmp_path, monkeypatch, caplog, error
):
    # Arrange: making the directory fails, in words that name a path
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(scratch_module.tempfile, "mkdtemp", _raising(error))
    tries = _note_the_lock_tries(monkeypatch, tmp_path)

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: no lock was tried, so there is no verdict on it: the start goes on, and a job meets the
    # trouble as its own failure. One warning, by the error's class and with no path
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert refused is False
    assert tries == []
    assert len(caplog.records) == 1
    assert len(warnings) == 1 and warnings[0].endswith(f": {type(error).__name__}")
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(SCRATCH_PREFIX not in line for line in _written(caplog))
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    ("make_root", "error_class"),
    [
        pytest.param(lambda tmp_path: tmp_path / "missing", "FileNotFoundError", id="a-root-that-does-not-exist"),
        pytest.param(lambda tmp_path: tmp_path / "a-file", "NotADirectoryError", id="a-root-that-is-a-file"),
    ],
)
def test_lock_refused_in_when_the_root_is_no_directory_should_say_no_and_name_nothing_of_it(
    tmp_path, caplog, make_root, error_class
):
    # Arrange: no stand-in fails here, the operating system does, and its own error names the root
    caplog.set_level(logging.DEBUG)
    (tmp_path / "a-file").write_bytes(KEPT_BYTES)
    root = make_root(tmp_path)

    # Act
    refused = lock_refused_in(root)

    # Assert
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert refused is False
    assert len(warnings) == 1 and warnings[0].endswith(f": {error_class}")
    assert all(str(tmp_path) not in line and root.name not in line for line in _written(caplog))
    assert (tmp_path / "a-file").read_bytes() == KEPT_BYTES


def test_lock_refused_in_when_its_directory_cannot_be_opened_should_say_no_and_remove_the_directory(
    tmp_path, monkeypatch, caplog
):
    # Arrange: no descriptor is left when the directory it made is opened for its lock
    caplog.set_level(logging.DEBUG)
    made: list[Path] = []
    open_file = os.open

    def open_nothing_in_the_root(opened_path: Any, flags: int, *arguments: Any, **named_arguments: Any) -> int:
        if Path(opened_path).parent == tmp_path:
            made.append(Path(opened_path))
            raise OSError(errno.EMFILE, "Too many open files", PATH_CANARY)
        return open_file(opened_path, flags, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.os, "open", open_nothing_in_the_root)
    tries = _note_the_lock_tries(monkeypatch, tmp_path)

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: the directory was made, no lock was tried, and what was made is gone
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(made) == 1 and made[0].name.startswith(scratch_prefix())
    assert refused is False
    assert tries == []
    assert list(tmp_path.iterdir()) == []
    assert len(warnings) == 1 and warnings[0].endswith(": OSError")
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(SCRATCH_PREFIX not in line for line in _written(caplog))


def test_lock_refused_in_when_a_link_took_the_place_of_its_directory_should_not_follow_it(
    tmp_path, monkeypatch, caplog
):
    # Arrange: the instant the directory is made, a link to a directory outside the root bears its name
    caplog.set_level(logging.DEBUG)
    root, elsewhere = tmp_path / "scratch", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (elsewhere / "kept").write_bytes(KEPT_BYTES)
    make = tempfile.mkdtemp

    def make_a_directory_a_link_replaces(*arguments: Any, **named_arguments: Any) -> str:
        made = make(*arguments, **named_arguments)
        os.rmdir(made)
        os.symlink(elsewhere, made, target_is_directory=True)
        return made

    monkeypatch.setattr(scratch_module.tempfile, "mkdtemp", make_a_directory_a_link_replaces)
    tries = _note_the_lock_tries(monkeypatch, root)

    # Act
    refused = lock_refused_in(root)

    # Assert: opened as an execution opens its own, through no link: no lock was tried on what the link
    # points to, which is neither removed nor changed, and that is no verdict on the lock
    assert refused is False
    assert tries == []
    assert (elsewhere / "kept").read_bytes() == KEPT_BYTES
    assert all(str(tmp_path) not in line and SCRATCH_PREFIX not in line for line in _written(caplog))


def test_lock_refused_in_when_the_lock_call_fails_with_no_os_error_should_say_no_and_leave_nothing_behind(
    tmp_path, monkeypatch, caplog
):
    # Arrange: the lock call itself fails, and not as the system refuses a lock
    caplog.set_level(logging.DEBUG)
    tries = _note_the_lock_tries(monkeypatch, tmp_path, then_raise=ValueError(f"no lock on {PATH_CANARY}"))

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: only the system's own refusal of the lock is a verdict on it
    (tried,) = tries
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert refused is False
    assert list(tmp_path.iterdir()) == []
    assert _descriptors_open_on(tried.opened_on) == 0
    assert len(warnings) == 1 and warnings[0].endswith(": ValueError")
    assert all(PATH_CANARY not in line for line in _written(caplog))


# How the system refuses a lock on a file system that keeps none, in words that name a path.
LOCK_REFUSED = OSError(errno.ENOLCK, "No locks available", PATH_CANARY)


@pytest.mark.parametrize(
    ("lock_ends_as", "verdict"),
    [pytest.param(None, False, id="a-root-that-locks"), pytest.param(LOCK_REFUSED, True, id="a-root-that-refuses")],
)
def test_lock_refused_in_when_its_directory_cannot_be_removed_should_give_its_verdict_and_leave_a_leftover(
    tmp_path, frozen_now, monkeypatch, caplog, lock_ends_as, verdict
):
    # Arrange: the removal of the directory fails, in words that name a path
    caplog.set_level(logging.DEBUG)
    remove_directory = os.rmdir

    def remove_nothing_in_the_root(removed_path: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(removed_path).parent == tmp_path:
            raise PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)
        remove_directory(removed_path, *arguments, **named_arguments)

    # Act
    with monkeypatch.context() as during_the_check:
        during_the_check.setattr(scratch_module.os, "rmdir", remove_nothing_in_the_root)
        tries = _note_the_lock_tries(during_the_check, tmp_path, then_raise=lock_ends_as)
        refused = lock_refused_in(tmp_path)

    # Assert: the verdict is the lock's. One warning, by the error's class and with no path. What stays
    # is a leftover like any other: of this host, locked by nobody and with no descriptor open on it,
    # which a later start removes once it is old
    (tried,) = tries
    (left,) = tmp_path.iterdir()
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert refused is verdict
    assert len(warnings) == 1 and warnings[0].endswith(": PermissionError")
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))
    assert all(SCRATCH_PREFIX not in line for line in _written(caplog))
    assert _descriptors_open_on(tried.opened_on) == 0
    assert _a_shared_lock_is_free(left)
    _set_age(left, frozen_now, LONG_AGO_S)
    assert remove_leftover_scratch(tmp_path) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == ROOT_ID, reason="root removes a directory whatever its parent's mode forbids")
def test_lock_refused_in_when_its_directory_really_cannot_be_removed_should_say_no_and_warn_once(
    tmp_path, monkeypatch, caplog
):
    # Arrange: once the directory is made, the root is one its owner may list and may not change: no
    # stand-in fails the removal here, the operating system does
    caplog.set_level(logging.DEBUG)
    root = tmp_path / "scratch"
    root.mkdir()
    make = tempfile.mkdtemp

    def make_then_close_the_root(*arguments: Any, **named_arguments: Any) -> str:
        made = make(*arguments, **named_arguments)
        os.chmod(root, LISTED_NOT_CHANGED)
        return made

    monkeypatch.setattr(scratch_module.tempfile, "mkdtemp", make_then_close_the_root)

    # Act: the mode is put back whatever happens, so that pytest can remove its own directory
    try:
        refused = lock_refused_in(root)
        left = sorted(child.name for child in root.iterdir())
    finally:
        os.chmod(root, OWNER_ONLY)

    # Assert: the lock was given, and one warning tells of the directory that stays, by the error's class
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert refused is False
    assert len(left) == 1 and left[0].startswith(scratch_prefix())
    assert len(warnings) == 1 and warnings[0].endswith(": PermissionError")
    assert all(str(tmp_path) not in line and SCRATCH_PREFIX not in line for line in _written(caplog))


@pytest.mark.parametrize(
    ("lock_ends_as", "verdict"),
    [pytest.param(None, False, id="a-root-that-locks"), pytest.param(LOCK_REFUSED, True, id="a-root-that-refuses")],
)
def test_lock_refused_in_when_its_descriptor_cannot_be_closed_should_give_its_verdict_and_remove_its_directory(
    tmp_path, monkeypatch, caplog, lock_ends_as, verdict
):
    # Arrange: the close of the descriptor the lock was tried on fails, in words that name a path
    caplog.set_level(logging.DEBUG)
    close = os.close
    tries: list[_LockTry] = []

    def close_all_but_the_one_the_lock_was_tried_on(descriptor: int) -> None:
        if tries and descriptor == tries[0].descriptor:
            raise OSError(errno.EIO, "Input/output error", PATH_CANARY)
        close(descriptor)

    # Act
    with monkeypatch.context() as during_the_check:
        tries = _note_the_lock_tries(during_the_check, tmp_path, then_raise=lock_ends_as)
        during_the_check.setattr(scratch_module.os, "close", close_all_but_the_one_the_lock_was_tried_on)
        refused = lock_refused_in(tmp_path)
    (tried,) = tries
    close(tried.descriptor)

    # Assert: nothing was raised (above); the verdict is the lock's and the directory is gone
    assert refused is verdict
    assert list(tmp_path.iterdir()) == []
    assert all(PATH_CANARY not in line and str(tmp_path) not in line for line in _written(caplog))


@pytest.mark.parametrize(
    "interrupt_class", [KeyboardInterrupt, SystemExit], ids=["an-interrupt", "the-interpreters-exit"]
)
@pytest.mark.parametrize("arrives_at", ["the-opening", "the-lock"])
def test_lock_refused_in_when_an_interrupt_arrives_should_let_it_through_as_it_is_and_leave_nothing_behind(
    tmp_path, monkeypatch, caplog, interrupt_class, arrives_at
):
    # Arrange: the directory is made, and the interrupt arrives while it is opened or while its lock is tried
    caplog.set_level(logging.DEBUG)
    interrupt = interrupt_class()
    open_file = os.open

    def be_interrupted_at_the_opening(opened_path: Any, flags: int, *arguments: Any, **named_arguments: Any) -> int:
        if Path(opened_path).parent == tmp_path:
            raise interrupt
        return open_file(opened_path, flags, *arguments, **named_arguments)

    # Act
    with monkeypatch.context() as during_the_check:
        tries = _note_the_lock_tries(
            during_the_check, tmp_path, then_raise=interrupt if arrives_at == "the-lock" else None
        )
        if arrives_at == "the-opening":
            during_the_check.setattr(scratch_module.os, "open", be_interrupted_at_the_opening)
        with pytest.raises(interrupt_class) as raised:
            lock_refused_in(tmp_path)

    # Assert: the very interrupt, with nothing chained to it; the directory is gone, the descriptor
    # closed, and no line was written behind it
    assert raised.value is interrupt
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert list(tmp_path.iterdir()) == []
    assert [_descriptors_open_on(tried.opened_on) for tried in tries] == [0] * (arrives_at == "the-lock")
    assert caplog.records == []


@pytest.mark.parametrize(
    "no_path_of",
    [
        pytest.param(lambda tmp_path: str(tmp_path), id="text"),
        pytest.param(lambda tmp_path: None, id="none"),
        pytest.param(lambda tmp_path: 7, id="a-number"),
    ],
)
def test_lock_refused_in_when_the_root_is_no_path_should_raise_type_error_and_try_nothing(
    tmp_path, monkeypatch, no_path_of
):
    # Arrange: the text is the test's own directory, and so is the working directory: were either ever
    # taken for a root, the try is made there and nowhere else
    monkeypatch.chdir(tmp_path)
    tries = _note_the_lock_tries(monkeypatch, tmp_path)
    root = no_path_of(tmp_path)

    # Act and Assert
    with pytest.raises(TypeError):
        lock_refused_in(root)  # type: ignore[arg-type]
    assert tries == []
    assert list(tmp_path.iterdir()) == []


def test_lock_refused_in_when_a_line_of_its_own_raises_should_still_give_its_verdict(tmp_path, monkeypatch):
    # Arrange: the directory cannot be made, which the check tells in a warning, and the module's
    # logger hands its lines to a handler whose emit raises
    monkeypatch.setattr(
        scratch_module.tempfile, "mkdtemp", _raising(OSError(errno.ENOSPC, "No space left on device", PATH_CANARY))
    )
    raising = _RaisingLogHandler()
    scratch_module.logger.addHandler(raising)

    # Act
    try:
        refused = lock_refused_in(tmp_path)
    finally:
        scratch_module.logger.removeHandler(raising)
        raising.close()

    # Assert: the line did meet the failing log, and the check answers all the same: a start is not
    # stopped by a line that cannot be written
    assert raising.lines == 1
    assert refused is False


# --- a log line whose handler fails -------------------------------------------------------------------
# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while the error it tells of
# is still being handled would put that error's own text, which names a path, into the report.
# `failing_log` makes every line's handler fail; what the call returns is what it returns without it.


def test_remove_leftover_scratch_when_a_log_handler_fails_on_the_line_of_an_unsearchable_root_should_keep_its_path_out(
    tmp_path, capsys, failing_log
):
    # Arrange: a root that does not exist: the operating system's own error names its path
    missing = tmp_path / "missing"

    # Act
    removed = remove_leftover_scratch(missing)

    # Assert: logging did report its failure on that line, with nothing of the path
    written = reported_log_failures(capsys, "scratch root could not be searched")
    assert str(missing) not in written
    assert removed == 0


def test_remove_leftover_scratch_when_a_log_handler_fails_on_the_line_of_a_stuck_leftover_should_keep_its_path_out(
    tmp_path, frozen_now, monkeypatch, capsys, failing_log
):
    # Arrange: the removal of one leftover fails, in words that name a path
    stuck = _leftover(tmp_path, frozen_now, LEFTOVER_NAME)
    removable = _leftover(tmp_path, frozen_now, OTHER_LEFTOVER_NAME)
    remove_tree = shutil.rmtree

    def remove_all_but_the_stuck_one(path: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(path) == stuck:
            raise PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)
        remove_tree(path, *arguments, **named_arguments)

    monkeypatch.setattr(scratch_module.shutil, "rmtree", remove_all_but_the_stuck_one)

    # Act
    removed = remove_leftover_scratch(tmp_path)

    # Assert
    written = reported_log_failures(capsys, "could not be examined or removed")
    assert PATH_CANARY not in written
    assert removed == 1
    assert stuck.is_dir() and not removable.exists()


def test_release_scratch_lock_when_a_log_handler_fails_on_the_line_of_a_lock_not_released_should_keep_the_error_out(
    tmp_path, monkeypatch, capsys, failing_log
):
    # Arrange: the close of the lock's own descriptor fails, in words that name a path
    directory = tmp_path / LEFTOVER_NAME
    directory.mkdir()
    held = hold_scratch_lock(directory)
    close = os.close

    def close_all_but_the_locks(descriptor: int) -> None:
        if descriptor == held:
            raise OSError(errno.EIO, "Input/output error", PATH_CANARY)
        close(descriptor)

    # Act
    with monkeypatch.context() as while_released:
        while_released.setattr(scratch_module.os, "close", close_all_but_the_locks)
        release_scratch_lock(held)
    close(held)

    # Assert: nothing was raised (above), and the report holds nothing of the error's text
    written = reported_log_failures(capsys, "lock could not be released")
    assert PATH_CANARY not in written


def test_lock_refused_in_when_a_log_handler_fails_on_the_line_of_a_lock_not_tried_should_keep_the_errors_text_out(
    tmp_path, monkeypatch, capsys, failing_log
):
    # Arrange: the directory cannot be made, in words that name a path
    monkeypatch.setattr(
        scratch_module.tempfile, "mkdtemp", _raising(OSError(errno.ENOSPC, FULL_DISK_WORDS, PATH_CANARY))
    )

    # Act
    refused = lock_refused_in(tmp_path)

    # Assert: logging did report its failure on that line, with nothing of the error that was caught:
    # neither the path it names nor its own words
    written = reported_log_failures(capsys, "could not be tried")
    assert PATH_CANARY not in written and FULL_DISK_WORDS not in written
    assert refused is False


def test_lock_refused_in_when_a_log_handler_fails_on_the_line_of_a_directory_left_should_keep_the_errors_text_out(
    tmp_path, monkeypatch, capsys, failing_log
):
    # Arrange: the directory is made and locked, and its removal fails, in words that name a path
    remove_directory = os.rmdir

    def remove_nothing_in_the_root(removed_path: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(removed_path).parent == tmp_path:
            raise PermissionError(errno.EACCES, NOT_PERMITTED_WORDS, PATH_CANARY)
        remove_directory(removed_path, *arguments, **named_arguments)

    # Act
    with monkeypatch.context() as during_the_check:
        during_the_check.setattr(scratch_module.os, "rmdir", remove_nothing_in_the_root)
        refused = lock_refused_in(tmp_path)

    # Assert
    written = reported_log_failures(capsys, "could not be removed")
    assert PATH_CANARY not in written and NOT_PERMITTED_WORDS not in written
    assert refused is False
