"""Locate the transcode contract v2 tree, read its files, pin its aggregate digest and verify it.

The tree stays at the repository root's tests/contracts/transcode/v2/ (the RunPod image copies it to
the same relative path). Any edit to the tree changes EXPECTED_CONTRACT_DIGEST here and the Hub's
ContractFiles::EXPECTED_DIGEST in the same change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

CONTRACT_ROOT: Path = Path(__file__).resolve().parents[2] / "tests" / "contracts" / "transcode" / "v2"
CHECKSUM_FILE = "SHA256SUMS"
EXPECTED_CONTRACT_DIGEST = "e31fefb6b3230eb52f598b209adae83f4a38b9ba2bfa8e664a63febeda13f31e"

_FORBIDDEN_SEGMENTS = frozenset({"", ".", ".."})
# One SHA256SUMS line without its LF: the lowercase digest, two spaces, the relative path.
_CHECKSUM_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>[^\n]+)")


class ContractTreeInvalid(RuntimeError):
    """The contract tree is missing, or differs from what SHA256SUMS and EXPECTED_CONTRACT_DIGEST pin."""


def _path_in(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or relative_path == "":
        raise ValueError("a contract path must be a non-empty string")
    segments = relative_path.split("/")
    if "\\" in relative_path or any(segment in _FORBIDDEN_SEGMENTS for segment in segments):
        raise ValueError(f"not a path inside the contract tree: {relative_path!r}")
    return root.joinpath(*segments)


def contract_path(relative_path: str) -> Path:
    """Return the absolute path of a file named relative to the tree root; refuse any path that leaves the tree."""
    return _path_in(CONTRACT_ROOT, relative_path)


def read_bytes(relative_path: str) -> bytes:
    """Return the exact bytes of a tree file."""
    return contract_path(relative_path).read_bytes()


def read_json(relative_path: str) -> Any:
    """Decode a contract data file (strict UTF-8, standard JSON)."""
    return json.loads(read_bytes(relative_path).decode("utf-8"))


def list_tree_files() -> list[str]:
    """Every file of the tree as a relative POSIX path, sorted by its UTF-8 bytes."""
    if not CONTRACT_ROOT.is_dir():
        raise FileNotFoundError(f"contract tree not found at {CONTRACT_ROOT}")
    relative_paths = [
        path.relative_to(CONTRACT_ROOT).as_posix() for path in CONTRACT_ROOT.rglob("*") if path.is_file()
    ]
    return sorted(relative_paths, key=lambda relative_path: relative_path.encode("utf-8"))


def sha256_hex(data: bytes) -> str:
    """Lowercase hexadecimal SHA-256 of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def aggregate_digest() -> str:
    """SHA-256 of the SHA256SUMS bytes: the one value both sides pin."""
    return sha256_hex(read_bytes(CHECKSUM_FILE))


def verify_contract_tree(root: Path | None = None) -> None:
    """The start-up check: return only when the tree is exactly the pinned one, else raise ContractTreeInvalid.

    In this order, the first problem found is the one named: the tree is missing; an entry is anything but a
    real directory or a regular file (a symbolic link, never followed; a FIFO; a socket; a device) or its
    name is not UTF-8; SHA256SUMS is missing or its SHA-256 differs from EXPECTED_CONTRACT_DIGEST (the anchor
    every later check trusts); a listed file is missing or its SHA-256 differs from its line; a file of the
    tree is not listed. Any error while reading the tree refuses it too. `root` replaces CONTRACT_ROOT for
    tests only.
    """
    tree_root = CONTRACT_ROOT if root is None else Path(root)
    try:
        _verify_tree(tree_root)
    except (OSError, ValueError) as error:
        # str() of an OSError names the failing file by its repr, so no name can break a log line.
        raise ContractTreeInvalid(f"contract tree at {tree_root} cannot be read: {error}") from error


def _verify_tree(tree_root: Path) -> None:
    if not tree_root.is_dir():
        raise ContractTreeInvalid(f"contract tree not found at {tree_root}")
    regular_files = _regular_files(tree_root)
    present = frozenset(regular_files)
    if CHECKSUM_FILE not in present:
        raise ContractTreeInvalid(f"{CHECKSUM_FILE} is missing from the contract tree at {tree_root}")
    checksum_bytes = (tree_root / CHECKSUM_FILE).read_bytes()
    if sha256_hex(checksum_bytes) != EXPECTED_CONTRACT_DIGEST:
        raise ContractTreeInvalid(f"sha256({CHECKSUM_FILE}) differs from EXPECTED_CONTRACT_DIGEST")

    entries = _checksum_entries(tree_root, checksum_bytes)
    for relative_path, (path, expected_digest) in entries.items():
        if relative_path not in present:
            raise ContractTreeInvalid(f"{relative_path} is listed in {CHECKSUM_FILE} but missing from the tree")
        if sha256_hex(path.read_bytes()) != expected_digest:
            raise ContractTreeInvalid(f"the SHA-256 of {relative_path} differs from its line in {CHECKSUM_FILE}")

    for relative_path in regular_files:
        if relative_path != CHECKSUM_FILE and relative_path not in entries:
            raise ContractTreeInvalid(f"{relative_path!r} is in the tree but not listed in {CHECKSUM_FILE}")


def _regular_files(tree_root: Path) -> list[str]:
    """Every regular file of the tree as a relative POSIX path, sorted by its UTF-8 bytes.

    Symbolic links are never followed: any entry but a real directory or a regular file refuses the tree,
    as does a name that is not UTF-8. A refusal names the entry by its repr, which any log line can hold.
    """
    found: list[str] = []
    # Directories still to read, as (path, prefix of their entries): a loop, so no depth exhausts the stack.
    pending: list[tuple[Path, str]] = [(tree_root, "")]
    while pending:
        directory, prefix = pending.pop()
        for entry in _sorted_entries(directory):
            relative_path = _checked_entry(entry, prefix)
            if entry.is_dir(follow_symlinks=False):
                pending.append((Path(entry.path), relative_path + "/"))
            else:
                found.append(relative_path)
    return sorted(found, key=lambda relative_path: relative_path.encode("utf-8"))


def _sorted_entries(directory: Path) -> list[os.DirEntry[str]]:
    with os.scandir(directory) as scan:
        return sorted(scan, key=lambda entry: os.fsencode(entry.name))


def _checked_entry(entry: os.DirEntry[str], prefix: str) -> str:
    """The entry's relative path when it is a real directory or a regular file with a UTF-8 name; else refuse."""
    relative_path = prefix + entry.name
    try:
        relative_path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ContractTreeInvalid(f"{relative_path!r} is not a UTF-8 name") from error
    if entry.is_symlink():
        raise ContractTreeInvalid(f"{relative_path!r} is a symbolic link; the tree holds no links")
    if not entry.is_dir(follow_symlinks=False) and not entry.is_file(follow_symlinks=False):
        raise ContractTreeInvalid(f"{relative_path!r} is neither a directory nor a regular file")
    return relative_path


def _checksum_entries(tree_root: Path, checksum_bytes: bytes) -> dict[str, tuple[Path, str]]:
    """Each listed relative path with its absolute path and digest, in file order; a malformed line refuses."""
    try:
        text = checksum_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractTreeInvalid(f"{CHECKSUM_FILE} is not ASCII") from error
    if not text.endswith("\n"):
        raise ContractTreeInvalid(f"{CHECKSUM_FILE} does not end with a line feed")
    entries: dict[str, tuple[Path, str]] = {}
    for line in text.split("\n")[:-1]:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ContractTreeInvalid(f"{CHECKSUM_FILE} holds a malformed line: {line!r}")
        relative_path = match["path"]
        if relative_path in entries:
            raise ContractTreeInvalid(f"{CHECKSUM_FILE} lists {relative_path} twice")
        try:
            path = _path_in(tree_root, relative_path)
        except ValueError as error:
            raise ContractTreeInvalid(f"{CHECKSUM_FILE} lists a path outside the tree: {relative_path!r}") from error
        entries[relative_path] = (path, match["digest"])
    return entries
