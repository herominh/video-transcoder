"""Locate the transcode contract v2 tree, read its files and pin its aggregate digest.

Any edit to the tree changes EXPECTED_CONTRACT_DIGEST here and the Hub's
ContractFiles::EXPECTED_DIGEST in the same change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_ROOT: Path = Path(__file__).resolve().parent.parent / "contracts" / "transcode" / "v2"
CHECKSUM_FILE = "SHA256SUMS"
EXPECTED_CONTRACT_DIGEST = "6daef3dc2a79137590b226f8f17df1ebb1075d1d66a87b221694462805c44891"

_FORBIDDEN_SEGMENTS = frozenset({"", ".", ".."})


def contract_path(relative_path: str) -> Path:
    """Return the absolute path of a file named relative to the tree root; refuse any path that leaves the tree."""
    if not isinstance(relative_path, str) or relative_path == "":
        raise ValueError("a contract path must be a non-empty string")
    segments = relative_path.split("/")
    if "\\" in relative_path or any(segment in _FORBIDDEN_SEGMENTS for segment in segments):
        raise ValueError(f"not a path inside the contract tree: {relative_path!r}")
    return CONTRACT_ROOT.joinpath(*segments)


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
