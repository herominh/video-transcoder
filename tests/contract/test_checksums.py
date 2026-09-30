"""The contract copy is exactly the tree SHA256SUMS describes, and its digest is the pinned one."""

import re

from tests.contract import files

CHECKSUM_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>[^\n]+)")


def _checksum_entries() -> list[tuple[str, str]]:
    text = files.read_bytes(files.CHECKSUM_FILE).decode("ascii")
    entries = []
    for line in text.split("\n")[:-1]:
        match = CHECKSUM_LINE.fullmatch(line)
        assert match is not None, f"malformed SHA256SUMS line: {line!r}"
        entries.append((match["path"], match["digest"]))
    return entries


def test_checksum_file_when_read_should_hold_lf_terminated_lines_sorted_by_path_bytes():
    # Arrange
    raw = files.read_bytes(files.CHECKSUM_FILE)

    # Act
    paths = [path for path, _ in _checksum_entries()]

    # Assert
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert paths == sorted(paths, key=lambda path: path.encode("utf-8"))
    assert len(paths) == len(set(paths))


def test_contract_files_when_hashed_should_match_sha256sums():
    # Arrange
    entries = _checksum_entries()

    # Act
    mismatched = [path for path, digest in entries if files.sha256_hex(files.read_bytes(path)) != digest]

    # Assert
    assert mismatched == []


def test_contract_tree_when_listed_should_hold_exactly_the_files_of_sha256sums():
    # Arrange
    listed = {path for path, _ in _checksum_entries()}

    # Act
    present = set(files.list_tree_files()) - {files.CHECKSUM_FILE}

    # Assert
    assert present == listed


def test_contract_digest_when_computed_should_equal_pinned_constant():
    # Act
    digest = files.aggregate_digest()

    # Assert
    assert digest == files.EXPECTED_CONTRACT_DIGEST
