from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import core.hls_key
from core.hls_key import KEY_URI_PLACEHOLDER, PROC_SELF_FD, hls_key_info, memory_key_supported

pytestmark = pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX only")

KEY = bytes(range(16))
OWNER_ONLY_FILE = 0o600
OWNER_ONLY_DIRECTORY = 0o700
requires_memory_key = pytest.mark.skipif(
    not memory_key_supported(), reason="the key lives in a memfd only where Linux offers one"
)


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
