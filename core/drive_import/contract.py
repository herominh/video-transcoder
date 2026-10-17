"""The import contract v1 tree: locate it, pin its aggregate digest, verify it, validate against its schemas, and
compute the receipt's MAC.

The tree stays at the repository root's tests/contracts/import/v1/ (the import image copies it to the same relative
path). Any edit to the tree changes EXPECTED_IMPORT_CONTRACT_DIGEST here and the Hub's
ImportContractFiles::EXPECTED_DIGEST in the same change (README section 1).

This module is the import's own: it imports nothing of core.protocol, whose tree check and schema layer it mirrors
for the transcode contract, so that the import entry point reaches no transcode module (tests/test_runtime_scope.py).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import DRAFT202012

IMPORT_CONTRACT_ROOT: Path = Path(__file__).resolve().parents[2] / "tests" / "contracts" / "import" / "v1"
CHECKSUM_FILE = "SHA256SUMS"
EXPECTED_IMPORT_CONTRACT_DIGEST = "39ece40a7f7798dc66a46aafdd80321bdfc9cb0ad4e3ac71617459afc050fa9e"

CONTRACT = "import/v1"
SCHEMA_BASE_URI = "https://schemas.video-hub.invalid/import/v1/"
INPUT_SCHEMA = "input.schema.json"
OUTPUT_SCHEMA = "output.schema.json"
RECEIPT_SCHEMA = "receipt.schema.json"
SCHEMAS = (INPUT_SCHEMA, OUTPUT_SCHEMA, RECEIPT_SCHEMA)
# The member of the input and of the receipt that holds its MAC, left out of what the MAC is computed over (README
# sections 3.1 and 5.1).
MAC_MEMBER = "mac"
# The bytes an input's MAC is computed over before the input's canonical form, so that an input's MAC is never a
# receipt's (README section 3.1).
INPUT_MAC_PREFIX = "import/v1 input\n"
# The largest integer a JSON number carries exactly in both languages (2^53).
LARGEST_EXACT_INTEGER = 2**53

_FORBIDDEN_SEGMENTS = frozenset({"", ".", ".."})
# One SHA256SUMS line without its LF: the lowercase digest, two spaces, the relative path.
_CHECKSUM_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>[^\n]+)")


class ImportContractInvalid(RuntimeError):
    """The import contract tree is missing, or differs from what SHA256SUMS and EXPECTED_IMPORT_CONTRACT_DIGEST pin."""


class NoCanonicalForm(ValueError):
    """A receipt holds a value the canonical form has no writing for (a number that is not an integer)."""


# --- the tree ------------------------------------------------------------------------------------------------------


def _path_in(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or relative_path == "":
        raise ValueError("a contract path must be a non-empty string")
    segments = relative_path.split("/")
    if "\\" in relative_path or any(segment in _FORBIDDEN_SEGMENTS for segment in segments):
        raise ValueError(f"not a path inside the import contract tree: {relative_path!r}")
    return root.joinpath(*segments)


def read_bytes(relative_path: str) -> bytes:
    """The exact bytes of a file of the tree, named relative to its root; a path that leaves the tree is refused."""
    return _path_in(IMPORT_CONTRACT_ROOT, relative_path).read_bytes()


def read_json(relative_path: str) -> Any:
    """A data file of the tree, decoded (strict UTF-8, standard JSON)."""
    return json.loads(read_bytes(relative_path).decode("utf-8"))


def sha256_hex(data: bytes) -> str:
    """Lowercase hexadecimal SHA-256 of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def verify_import_contract_tree(root: Path | None = None) -> None:
    """The start-up check: return only when the tree is exactly the pinned one, else raise ImportContractInvalid.

    As core/protocol/files.py checks the transcode tree, in this order, the first problem found being the one
    named: the tree is missing; an entry is anything but a real directory or a regular file (a symbolic link, never
    followed; a FIFO; a socket; a device) or its name is not UTF-8; SHA256SUMS is missing or its SHA-256 differs
    from EXPECTED_IMPORT_CONTRACT_DIGEST (the anchor every later check trusts); a listed file is missing or its
    SHA-256 differs from its line; a file of the tree is not listed. Any error while reading the tree refuses it
    too. `root` replaces IMPORT_CONTRACT_ROOT for tests only.
    """
    tree_root = IMPORT_CONTRACT_ROOT if root is None else Path(root)
    refusal: str | None = None
    try:
        _verify_tree(tree_root)
    except (OSError, ValueError) as error:
        # str() of an OSError names the failing file by its repr, so no name can break a log line.
        refusal = f"import contract tree at {tree_root} cannot be read: {error}"
    if refusal is not None:
        # Raised once the error is no longer being handled: the refusal carries its words, not the error.
        raise ImportContractInvalid(refusal)


def _verify_tree(tree_root: Path) -> None:
    if not tree_root.is_dir():
        raise ImportContractInvalid(f"import contract tree not found at {tree_root}")
    regular_files = _regular_files(tree_root)
    present = frozenset(regular_files)
    if CHECKSUM_FILE not in present:
        raise ImportContractInvalid(f"{CHECKSUM_FILE} is missing from the import contract tree at {tree_root}")
    checksum_bytes = (tree_root / CHECKSUM_FILE).read_bytes()
    if sha256_hex(checksum_bytes) != EXPECTED_IMPORT_CONTRACT_DIGEST:
        raise ImportContractInvalid(f"sha256({CHECKSUM_FILE}) differs from EXPECTED_IMPORT_CONTRACT_DIGEST")

    entries = _checksum_entries(tree_root, checksum_bytes)
    for relative_path, (path, expected_digest) in entries.items():
        if relative_path not in present:
            raise ImportContractInvalid(f"{relative_path} is listed in {CHECKSUM_FILE} but missing from the tree")
        if sha256_hex(path.read_bytes()) != expected_digest:
            raise ImportContractInvalid(f"the SHA-256 of {relative_path} differs from its line in {CHECKSUM_FILE}")

    for relative_path in regular_files:
        if relative_path != CHECKSUM_FILE and relative_path not in entries:
            raise ImportContractInvalid(f"{relative_path!r} is in the tree but not listed in {CHECKSUM_FILE}")


def _regular_files(tree_root: Path) -> list[str]:
    """Every regular file of the tree as a relative POSIX path, sorted by its UTF-8 bytes.

    Symbolic links are never followed: any entry but a real directory or a regular file refuses the tree, as does
    a name that is not UTF-8. A refusal names the entry by its repr, which any log line can hold.
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
    if not _is_utf8(relative_path):
        raise ImportContractInvalid(f"{relative_path!r} is not a UTF-8 name")
    if entry.is_symlink():
        raise ImportContractInvalid(f"{relative_path!r} is a symbolic link; the tree holds no links")
    if not entry.is_dir(follow_symlinks=False) and not entry.is_file(follow_symlinks=False):
        raise ImportContractInvalid(f"{relative_path!r} is neither a directory nor a regular file")
    return relative_path


def _is_utf8(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _checksum_entries(tree_root: Path, checksum_bytes: bytes) -> dict[str, tuple[Path, str]]:
    """Each listed relative path with its absolute path and digest, in file order; a malformed line refuses."""
    if not checksum_bytes.isascii():
        raise ImportContractInvalid(f"{CHECKSUM_FILE} is not ASCII")
    text = checksum_bytes.decode("ascii")
    if not text.endswith("\n"):
        raise ImportContractInvalid(f"{CHECKSUM_FILE} does not end with a line feed")
    entries: dict[str, tuple[Path, str]] = {}
    for line in text.split("\n")[:-1]:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ImportContractInvalid(f"{CHECKSUM_FILE} holds a malformed line: {line!r}")
        relative_path = match["path"]
        if relative_path in entries:
            raise ImportContractInvalid(f"{CHECKSUM_FILE} lists {relative_path} twice")
        segments = relative_path.split("/")
        if "\\" in relative_path or any(segment in _FORBIDDEN_SEGMENTS for segment in segments):
            raise ImportContractInvalid(f"{CHECKSUM_FILE} lists a path outside the tree: {relative_path!r}")
        entries[relative_path] = (tree_root.joinpath(*segments), match["digest"])
    return entries


# --- the schemas ---------------------------------------------------------------------------------------------------


def _refuse_retrieval(uri: str) -> Resource:
    raise NoSuchResource(ref=uri)


@lru_cache(maxsize=1)
def local_registry() -> Registry:
    """The three schemas of the tree, each keyed by its $id, which must be the base URI plus its file name.

    jsonschema combines this registry with its bundled copies of the json-schema.org metaschemas, which therefore
    resolve without the network; a $ref that names anything else fails, because the registry refuses to retrieve.
    """
    resources: list[tuple[str, Resource]] = []
    for name in SCHEMAS:
        contents = read_json(name)
        expected_id = SCHEMA_BASE_URI + name
        if not isinstance(contents, dict) or contents.get("$id") != expected_id:
            raise ImportContractInvalid(f"{name} must declare $id {expected_id}")
        resources.append((expected_id, Resource.from_contents(contents, default_specification=DRAFT202012)))
    return Registry(retrieve=_refuse_retrieval).with_resources(resources).crawl()


@lru_cache(maxsize=None)
def validator_for(name: str) -> Draft202012Validator:
    """A validator for one of the three schemas of the tree."""
    if name not in SCHEMAS:
        raise ValueError(f"no schema {name!r} in the import contract")
    return Draft202012Validator(local_registry().contents(SCHEMA_BASE_URI + name), registry=local_registry())


def json_pointer(path: Iterable[Any]) -> str:
    """RFC 6901 pointer of an instance path ("" for the root)."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def first_error_pointer(name: str, instance: Any) -> str | None:
    """The instance pointer of the first error jsonschema reports against schema `name`, or None when the instance
    is valid. Validation stops at that error: nothing after it is evaluated (bounded work on hostile input)."""
    for error in validator_for(name).iter_errors(instance):
        return json_pointer(error.absolute_path)
    return None


def error_pointers(name: str, instance: Any) -> list[str]:
    """Pointers of every reported error, sub-errors flattened, in report order; empty when valid. For tests."""
    pointers: list[str] = []
    pending = list(validator_for(name).iter_errors(instance))
    while pending:
        error = pending.pop(0)
        pointers.append(json_pointer(error.absolute_path))
        pending[:0] = list(error.context or ())
    return pointers


# --- the MACs of the input (README section 3.1) and of the receipt (section 5.1) -----------------------------------


def canonical_form(receipt: Mapping[str, Any]) -> str:
    """The canonical form of `receipt` (or of an input) without its `mac`: the members of every object in the order of
    their names, no whitespace, every string with its characters as themselves (only `"`, `\\` and the control
    characters escaped, as JSON writes them), `null` as `null`, every number an integer in decimal. A number whose
    value is an integer that a float carries exactly (as JSON decoders read `15.0`) is written as that integer, as the
    Hub writes it; any other number has no canonical form: NoCanonicalForm.

    The names of a valid receipt or input are ASCII, so the order of Python's strings is the order of their bytes.
    """
    without_mac = {name: value for name, value in receipt.items() if name != MAC_MEMBER}
    return json.dumps(_with_integers(without_mac), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _with_integers(value: Any) -> Any:
    """A copy of `value` whose integral floats are ints; NoCanonicalForm for any other float."""
    if isinstance(value, dict):
        return {name: _with_integers(member) for name, member in value.items()}
    if isinstance(value, list):
        return [_with_integers(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or abs(value) > LARGEST_EXACT_INTEGER:
            raise NoCanonicalForm("a receipt holds a number that is not an integer")
        return int(value)
    return value


def receipt_mac(receipt: Mapping[str, Any], receipt_key: str) -> str:
    """HMAC-SHA256 of the canonical form's UTF-8 bytes under the receipt key's UTF-8 bytes, in lowercase hex."""
    if receipt_key == "":
        raise ValueError("a receipt is signed with a key that is not empty")
    return hmac.new(
        receipt_key.encode("utf-8"), canonical_form(receipt).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def input_mac(document: Mapping[str, Any], import_key: str) -> str:
    """The input's MAC (README section 3.1): HMAC-SHA256 under the import key's UTF-8 bytes (the receipt key) of
    INPUT_MAC_PREFIX followed by the canonical form of the input without its `mac`, in lowercase hex."""
    if import_key == "":
        raise ValueError("an input is signed with a key that is not empty")
    signed = INPUT_MAC_PREFIX + canonical_form(document)
    return hmac.new(import_key.encode("utf-8"), signed.encode("utf-8"), hashlib.sha256).hexdigest()


def input_mac_verifies(document: Mapping[str, Any], import_key: str) -> bool:
    """Whether the input's own `mac` is its MAC under `import_key`, compared in constant time. An input with no
    `mac`, a `mac` that is no ASCII text, an empty key and an input with no canonical form do not verify."""
    given = document.get(MAC_MEMBER)
    if not isinstance(given, str) or not given.isascii() or import_key == "":
        return False
    try:
        expected: str | None = input_mac(document, import_key)
    except NoCanonicalForm:
        # A number that is not an integer: such an input has no MAC, so none of its own can verify.
        expected = None
    return expected is not None and hmac.compare_digest(expected, given)
