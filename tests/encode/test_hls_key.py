from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import core.hls_key
from core.hls_key import KEY_URI_PLACEHOLDER, PROC_SELF_FD, hls_key_info, memory_key_supported
from tests.failing_log import reported_log_failures

pytestmark = pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX only")

KEY = bytes(range(16))
OWNER_ONLY_FILE = 0o600
OWNER_ONLY_DIRECTORY = 0o700
requires_memory_key = pytest.mark.skipif(
    not memory_key_supported(), reason="the key lives in a memfd only where Linux offers one"
)
# What the text of an error holds that must never reach the worker's output: the path a removal that
# failed names, and what a key-info file that could not be written names.
PATH_CANARY = "/CANARY/path/of/a/key/file"
SETUP_CANARY = "/CANARY/path/of/a/key-info/file/not/written"


class _EncodeFailed(Exception):
    pass


@pytest.fixture
def file_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private-file fallback, as on a system without memfd."""
    monkeypatch.setattr(core.hls_key, "memory_key_supported", lambda: False)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _info_lines(info_path: Path) -> list[str]:
    return info_path.read_text(encoding="ascii").splitlines()


def _files_under(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def _descriptor_open(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            return False
        raise
    return True


@requires_memory_key
def test_hls_key_info_when_a_memfd_is_available_should_keep_the_key_out_of_every_file_under_parent(
    tmp_path: Path,
) -> None:
    # Arrange / Act
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        file_contents = [path.read_bytes() for path in _files_under(tmp_path)]

    # Assert
    assert file_contents
    assert all(KEY not in content for content in file_contents)


@requires_memory_key
def test_hls_key_info_when_a_memfd_is_available_should_name_a_passed_descriptor_that_holds_the_key(
    tmp_path: Path,
) -> None:
    # Arrange / Act
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        uri, key_location = _info_lines(key_info.path)
        descriptor = int(Path(key_location).name)
        key_bytes = Path(key_location).read_bytes()
        passed = key_info.pass_fds

    # Assert
    assert uri == KEY_URI_PLACEHOLDER
    assert Path(key_location).parent == PROC_SELF_FD
    assert passed == (descriptor,)
    assert key_bytes == KEY
    assert not _descriptor_open(descriptor)


@requires_memory_key
def test_hls_key_info_when_a_memfd_is_available_should_close_it_and_remove_the_directory_after_an_error(
    tmp_path: Path,
) -> None:
    # Arrange
    descriptors: tuple[int, ...] = ()

    # Act
    with pytest.raises(_EncodeFailed):
        with hls_key_info(KEY, parent=tmp_path) as key_info:
            descriptors = key_info.pass_fds
            raise _EncodeFailed

    # Assert
    assert descriptors and not any(_descriptor_open(descriptor) for descriptor in descriptors)
    assert list(tmp_path.iterdir()) == []


def test_hls_key_info_when_falling_back_to_a_file_should_hold_exactly_the_key_owner_only_under_parent(
    tmp_path: Path, file_key: None
) -> None:
    # Arrange / Act
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        uri, key_path_text = _info_lines(key_info.path)
        key_path = Path(key_path_text)
        key_bytes = key_path.read_bytes()
        modes = (_mode(key_path), _mode(key_info.path), _mode(key_path.parent))
        directory = key_path.parent
        passed = key_info.pass_fds

    # Assert
    assert key_bytes == KEY
    assert uri == KEY_URI_PLACEHOLDER
    assert modes == (OWNER_ONLY_FILE, OWNER_ONLY_FILE, OWNER_ONLY_DIRECTORY)
    assert directory.parent == tmp_path
    assert key_info.path.parent == directory
    assert passed == ()


def test_hls_key_info_when_falling_back_to_a_file_should_remove_it_and_its_directory_after_the_block(
    tmp_path: Path, file_key: None
) -> None:
    # Arrange / Act
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        directory = key_info.path.parent

    # Assert
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_hls_key_info_when_falling_back_to_a_file_should_still_remove_it_after_an_error(
    tmp_path: Path, file_key: None
) -> None:
    # Arrange
    directory: Path | None = None

    # Act
    with pytest.raises(_EncodeFailed):
        with hls_key_info(KEY, parent=tmp_path) as key_info:
            directory = key_info.path.parent
            raise _EncodeFailed

    # Assert
    assert directory is not None and not directory.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key", [bytes(15), bytes(17), b"", "0" * 16, bytearray(16)])
def test_hls_key_info_when_the_key_is_not_16_bytes_should_raise_and_write_nothing(key: Any, tmp_path: Path) -> None:
    # Arrange / Act
    with pytest.raises(ValueError):
        with hls_key_info(key, parent=tmp_path):
            pass

    # Assert
    assert list(tmp_path.iterdir()) == []


def test_hls_key_info_when_parent_is_relative_should_raise() -> None:
    # Arrange
    relative = Path("scratch")

    # Act / Assert
    with pytest.raises(ValueError):
        with hls_key_info(KEY, parent=relative):
            pass


# --- a log line whose handler fails ------------------------------------------------------------------
# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while an error is still
# being handled would put that error's own text into the report. `failing_log` makes every line's
# handler fail; the key is released as it is without it.


def _refuse_every_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the removal of a file fail, on an error whose own text names a path."""

    def refuse(path: Path, missing_ok: bool = False) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)

    monkeypatch.setattr(Path, "unlink", refuse)


def test_hls_key_info_when_a_log_handler_fails_on_the_line_of_a_key_file_not_removed_should_keep_its_path_out(
    tmp_path: Path, file_key: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange
    _refuse_every_removal(monkeypatch)

    # Act
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        directory = key_info.path.parent

    # Assert: logging did report its failure on that line, with nothing of the path the removal's error named
    written = reported_log_failures(capsys, "an HLS key file could not be removed")
    assert PATH_CANARY not in written
    assert sorted(path.name for path in directory.iterdir()) == ["key.info", "media.key"]


def test_hls_key_info_when_a_log_handler_fails_on_the_line_of_a_key_directory_not_removed_should_keep_its_path_out(
    tmp_path: Path, file_key: None, capsys: pytest.CaptureFixture[str], failing_log: None
) -> None:
    # Arrange / Act: a file the block leaves in the key's directory, so that the directory cannot be
    # removed, on the operating system's own error, whose text names the directory
    with hls_key_info(KEY, parent=tmp_path) as key_info:
        directory = key_info.path.parent
        (directory / "left-behind").write_bytes(b"x")

    # Assert
    written = reported_log_failures(capsys, "an HLS key directory could not be removed")
    assert str(tmp_path) not in written
    assert [path.name for path in directory.iterdir()] == ["left-behind"]


def test_hls_key_info_when_a_log_handler_fails_while_an_unwritten_key_is_released_should_keep_the_errors_text_out(
    tmp_path: Path, file_key: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    failing_log: None,
) -> None:
    # Arrange: the key is written, the key-info file is not, on an error whose text names its path, and
    # then neither file can be removed
    not_written = OSError(errno.ENOSPC, "No space left on device", SETUP_CANARY)
    write_private = core.hls_key._write_private
    written_paths: list[Path] = []

    def write_the_key_only(path: Path, data: bytes) -> None:
        written_paths.append(path)
        if len(written_paths) > 1:
            raise not_written
        write_private(path, data)

    monkeypatch.setattr(core.hls_key, "_write_private", write_the_key_only)
    _refuse_every_removal(monkeypatch)

    # Act
    with pytest.raises(OSError) as raised:
        with hls_key_info(KEY, parent=tmp_path):
            pass

    # Assert: nothing of the error the release ran behind, which is the one that is raised, as it is
    written = reported_log_failures(capsys, "an HLS key file could not be removed")
    assert [canary for canary in (SETUP_CANARY, PATH_CANARY) if canary in written] == []
    assert raised.value is not_written


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(3)], ids=["a-keyboard-interrupt", "an-exit"])
def test_hls_key_info_when_the_block_is_interrupted_should_remove_the_key_and_raise_the_interrupt_as_it_is(
    tmp_path: Path, file_key: None, interrupt: BaseException
) -> None:
    # Arrange
    directory: Path | None = None

    # Act
    with pytest.raises(type(interrupt)) as raised:
        with hls_key_info(KEY, parent=tmp_path) as key_info:
            directory = key_info.path.parent
            raise interrupt

    # Assert
    assert raised.value is interrupt
    assert directory is not None and not directory.exists()
    assert list(tmp_path.iterdir()) == []
