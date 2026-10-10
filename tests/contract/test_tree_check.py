"""The start-up check accepts only the pinned tree: every way a copy can drift from it is refused."""

import os
import re
import shutil
from pathlib import Path

import pytest

from core.protocol import files
from core.protocol.files import ContractTreeInvalid, verify_contract_tree

TAMPERED_FILE = "limits.json"
REMOVED_FILE = "versions.json"
UNLISTED_FILE = "schemas/extra.schema.json"


@pytest.fixture
def tree_copy(tmp_path: Path) -> Path:
    copy_root = tmp_path / "v2"
    shutil.copytree(files.CONTRACT_ROOT, copy_root)
    return copy_root


def _append_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b" ")


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform cannot create a symbolic link: {error}")


def _named(relative_path: str) -> str:
    """The pattern of an entry as the refusal names it (its repr, so no name can break a log line)."""
    return re.escape(repr(relative_path))


def test_verify_contract_tree_when_the_shipped_tree_is_checked_should_return():
    # Act
    outcome = verify_contract_tree()

    # Assert
    assert outcome is None


def test_verify_contract_tree_when_an_untouched_copy_is_checked_should_return(tree_copy: Path):
    # Act
    outcome = verify_contract_tree(root=tree_copy)

    # Assert
    assert outcome is None


def test_verify_contract_tree_when_the_tree_is_missing_should_refuse(tmp_path: Path):
    # Arrange
    absent_root = tmp_path / "absent"

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match="contract tree not found"):
        verify_contract_tree(root=absent_root)


def test_verify_contract_tree_when_a_file_differs_from_its_checksum_should_refuse(tree_copy: Path):
    # Arrange
    _append_byte(tree_copy / TAMPERED_FILE)

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=f"SHA-256 of {TAMPERED_FILE} differs"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_a_file_is_not_listed_should_refuse(tree_copy: Path):
    # Arrange
    (tree_copy / UNLISTED_FILE).write_bytes(b"{}\n")

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=_named(UNLISTED_FILE) + " is in the tree but not listed"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_a_listed_file_is_missing_should_refuse(tree_copy: Path):
    # Arrange
    (tree_copy / REMOVED_FILE).unlink()

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=f"{REMOVED_FILE} is listed in SHA256SUMS but missing"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_sha256sums_is_not_the_pinned_one_should_refuse(tree_copy: Path):
    # Arrange: a self-consistent tree (the tampered file's line rewritten to its new digest), so only
    # the pin on SHA256SUMS itself can tell it from the contract.
    tampered = tree_copy / TAMPERED_FILE
    original_digest = files.sha256_hex(tampered.read_bytes())
    _append_byte(tampered)
    sums = tree_copy / files.CHECKSUM_FILE
    rewritten = sums.read_bytes().replace(
        f"{original_digest}  {TAMPERED_FILE}\n".encode("ascii"),
        f"{files.sha256_hex(tampered.read_bytes())}  {TAMPERED_FILE}\n".encode("ascii"),
    )
    assert rewritten != sums.read_bytes()
    sums.write_bytes(rewritten)

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match="differs from EXPECTED_CONTRACT_DIGEST"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_sha256sums_is_missing_should_refuse(tree_copy: Path):
    # Arrange
    (tree_copy / files.CHECKSUM_FILE).unlink()

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match="SHA256SUMS is missing"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_an_entry_name_is_not_utf8_should_refuse(tree_copy: Path):
    # Arrange: POSIX names are bytes; Python hands this one over as a lone surrogate.
    raw_name = b"bad\xff.json"
    tree_bytes = os.fsencode(tree_copy)
    try:
        with open(os.path.join(tree_bytes, raw_name), "wb") as handle:
            handle.write(b"{}\n")
    except OSError as error:
        pytest.skip(f"this filesystem refuses a name that is not UTF-8: {error}")
    if raw_name not in os.listdir(tree_bytes):
        pytest.skip("this filesystem stored the name under other bytes")

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=_named(os.fsdecode(raw_name)) + " is not a UTF-8 name"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_a_listed_file_is_a_symlink_should_refuse(tree_copy: Path, tmp_path: Path):
    # Arrange: the link reaches identical bytes, so only the entry's type can refuse it.
    listed = tree_copy / TAMPERED_FILE
    outside = tmp_path / "outside.json"
    outside.write_bytes(listed.read_bytes())
    listed.unlink()
    _symlink_or_skip(listed, outside)

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=_named(TAMPERED_FILE) + " is a symbolic link"):
        verify_contract_tree(root=tree_copy)


def test_verify_contract_tree_when_a_directory_is_a_symlink_should_refuse(tree_copy: Path, tmp_path: Path):
    # Arrange: an empty target, so nothing reached through the link could be refused instead.
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(tree_copy / "linked", outside)

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=_named("linked") + " is a symbolic link"):
        verify_contract_tree(root=tree_copy)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="this platform has no named pipes")
def test_verify_contract_tree_when_an_entry_is_a_fifo_should_refuse(tree_copy: Path):
    # Arrange
    os.mkfifo(tree_copy / "schemas" / "pipe")

    # Act / Assert
    with pytest.raises(ContractTreeInvalid, match=_named("schemas/pipe") + " is neither a directory nor a regular file"):
        verify_contract_tree(root=tree_copy)
