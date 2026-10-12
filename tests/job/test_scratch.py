"""The scratch directories a killed process left behind, and the lock that tells them from a live one's.

remove_leftover_scratch() looks at the direct children of the scratch root whose name starts with the
runner's prefix and removes those that are real directories of this process's user, were not changed
for LEFTOVER_MIN_AGE_S and are locked by nobody. It removes nothing else, follows no link, never raises
and logs no path. hold_scratch_lock() is the lock an execution holds on its own directory.

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
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from core.job import scratch as scratch_module
from core.job.scratch import (
    LEFTOVER_MIN_AGE_S,
    SCRATCH_PREFIX,
    hold_scratch_lock,
    release_scratch_lock,
    remove_leftover_scratch,
)
from tests.job.conftest import held_at_most_a_while, reported_log_failures

# How long ago a leftover was last changed, unless a test says otherwise: far longer than the least age.
LONG_AGO_S = 2 * 60 * 60
LEFTOVER_NAME = SCRATCH_PREFIX + "abc123"
OTHER_LEFTOVER_NAME = SCRATCH_PREFIX + "def456"
SOURCE_BYTES = b"the source a killed execution had downloaded"
KEPT_BYTES = b"a file no removal of leftovers may touch"
PATH_CANARY = "/CANARY/path/in/an/errors/text"
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


@pytest.mark.parametrize("name", ["other-abc123", SCRATCH_PREFIX.rstrip("-"), "x" + SCRATCH_PREFIX + "abc123"])
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
    live = _leftover(tmp_path, frozen_now, SCRATCH_PREFIX + "live")
    let_go = held_at_most_a_while(_locked_as_by_a_live_execution(live))
    (tmp_path / (SCRATCH_PREFIX + "file")).write_bytes(KEPT_BYTES)

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
