"""The import contract copy: exactly the pinned tree (and the Hub's, byte for byte), its start-up check refusing every
way a copy can drift, every fixture of its catalog given its verdict, and the receipt's MAC as its known answer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing.exceptions import Unresolvable

from core.drive_import import contract
from core.drive_import.contract import ImportContractInvalid, NoCanonicalForm, verify_import_contract_tree

# The Hub's tree to compare the copy with, byte for byte, when the variable names one (a checkout of the Hub's branch
# the copy was taken from: the checkout beside this repository may be on another branch).
HUB_TREE_VARIABLE = "HUB_IMPORT_CONTRACT_TREE"
HUB_TREE = Path(os.environ[HUB_TREE_VARIABLE]) if os.environ.get(HUB_TREE_VARIABLE) else None
CATALOG = "fixtures/cases.json"
MAC_VECTOR = "fixtures/receipt-mac-vector.json"
TAMPERED_FILE = "input.schema.json"
REMOVED_FILE = "output.schema.json"
UNLISTED_FILE = "fixtures/input/valid-extra.json"
FIXTURE_DIRECTORIES = ("fixtures/input/", "fixtures/output/", "fixtures/receipt/")


@pytest.fixture
def tree_copy(tmp_path: Path) -> Path:
    copy_root = tmp_path / "v1"
    shutil.copytree(contract.IMPORT_CONTRACT_ROOT, copy_root)
    return copy_root


def _listed() -> dict[str, str]:
    text = contract.read_bytes(contract.CHECKSUM_FILE).decode("ascii")
    return {line[66:]: line[:64] for line in text.split("\n")[:-1]}


def _cases() -> list[dict]:
    return contract.read_json(CATALOG)["cases"]


# --- the tree ------------------------------------------------------------------------------------------------------


def test_verify_import_contract_tree_when_the_shipped_tree_is_checked_should_return():
    # Act
    outcome = verify_import_contract_tree()

    # Assert
    assert outcome is None


def test_import_contract_digest_when_computed_should_equal_the_pinned_constant():
    # Act
    digest = hashlib.sha256(contract.read_bytes(contract.CHECKSUM_FILE)).hexdigest()

    # Assert
    assert digest == contract.EXPECTED_IMPORT_CONTRACT_DIGEST
    assert digest == "39ece40a7f7798dc66a46aafdd80321bdfc9cb0ad4e3ac71617459afc050fa9e"


def test_import_contract_files_when_hashed_should_match_their_lines_in_sha256sums():
    # Act
    mismatched = [
        path for path, digest in _listed().items() if hashlib.sha256(contract.read_bytes(path)).hexdigest() != digest
    ]

    # Assert
    assert mismatched == []


@pytest.mark.skipif(HUB_TREE is None or not HUB_TREE.is_dir(), reason=f"{HUB_TREE_VARIABLE} names no Hub tree")
def test_import_contract_copy_when_compared_with_the_hubs_tree_should_hold_the_same_bytes():
    # Arrange
    hub_files = sorted(path.relative_to(HUB_TREE).as_posix() for path in HUB_TREE.rglob("*") if path.is_file())
    own_files = sorted(
        path.relative_to(contract.IMPORT_CONTRACT_ROOT).as_posix()
        for path in contract.IMPORT_CONTRACT_ROOT.rglob("*")
        if path.is_file()
    )

    # Act
    differing = [path for path in hub_files if (HUB_TREE / path).read_bytes() != contract.read_bytes(path)]

    # Assert
    assert own_files == hub_files
    assert differing == []


def test_verify_import_contract_tree_when_an_untouched_copy_is_checked_should_return(tree_copy):
    # Act
    outcome = verify_import_contract_tree(root=tree_copy)

    # Assert
    assert outcome is None


def test_verify_import_contract_tree_when_the_tree_is_missing_should_refuse(tmp_path):
    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="not found"):
        verify_import_contract_tree(root=tmp_path / "absent")


def test_verify_import_contract_tree_when_a_file_differs_from_its_checksum_should_refuse(tree_copy):
    # Arrange
    tampered = tree_copy / TAMPERED_FILE
    tampered.write_bytes(tampered.read_bytes() + b" ")

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match=f"SHA-256 of {TAMPERED_FILE} differs"):
        verify_import_contract_tree(root=tree_copy)


def test_verify_import_contract_tree_when_a_file_is_not_listed_should_refuse(tree_copy):
    # Arrange
    (tree_copy / UNLISTED_FILE).write_bytes(b"{}\n")

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="is in the tree but not listed"):
        verify_import_contract_tree(root=tree_copy)


def test_verify_import_contract_tree_when_a_listed_file_is_missing_should_refuse(tree_copy):
    # Arrange
    (tree_copy / REMOVED_FILE).unlink()

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match=f"{REMOVED_FILE} is listed in SHA256SUMS but missing"):
        verify_import_contract_tree(root=tree_copy)


def test_verify_import_contract_tree_when_sha256sums_is_self_consistent_but_not_the_pinned_one_should_refuse(tree_copy):
    # Arrange: the tampered file's line rewritten to its new digest, so only the pin can tell the copy from the tree
    tampered = tree_copy / TAMPERED_FILE
    before = hashlib.sha256(tampered.read_bytes()).hexdigest()
    tampered.write_bytes(tampered.read_bytes() + b" ")
    after = hashlib.sha256(tampered.read_bytes()).hexdigest()
    sums = tree_copy / contract.CHECKSUM_FILE
    sums.write_bytes(sums.read_bytes().replace(before.encode("ascii"), after.encode("ascii")))

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="differs from EXPECTED_IMPORT_CONTRACT_DIGEST"):
        verify_import_contract_tree(root=tree_copy)


def test_verify_import_contract_tree_when_sha256sums_is_missing_should_refuse(tree_copy):
    # Arrange
    (tree_copy / contract.CHECKSUM_FILE).unlink()

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="SHA256SUMS is missing"):
        verify_import_contract_tree(root=tree_copy)


def test_verify_import_contract_tree_when_a_file_is_replaced_by_a_symbolic_link_should_refuse(tree_copy):
    # Arrange: a link to a file of the same bytes, so only the link itself is wrong
    target = tree_copy.parent / "outside.json"
    shutil.copyfile(tree_copy / TAMPERED_FILE, target)
    (tree_copy / TAMPERED_FILE).unlink()
    try:
        (tree_copy / TAMPERED_FILE).symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform cannot create a symbolic link: {error}")

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="symbolic link"):
        verify_import_contract_tree(root=tree_copy)


# --- the schemas and the catalog -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_fixture_when_validated_against_its_schema_should_get_its_verdict_at_its_instance_path(case):
    # Arrange
    document = contract.read_json(case["fixture"])

    # Act
    first = contract.first_error_pointer(case["schema"], document)
    every = contract.error_pointers(case["schema"], document)

    # Assert: an accepted fixture has no error; a refused one is refused at its one violation, and only there
    if case["verdict"] == "accept":
        assert first is None and every == []
    else:
        assert first == case["instance_path"]
        assert [pointer for pointer in every if not pointer.startswith(case["instance_path"])] == []


def test_catalog_when_scanned_should_name_every_fixture_file_once():
    # Arrange
    named = [case["fixture"] for case in _cases()]

    # Act
    files = sorted(path for path in _listed() if path.startswith(FIXTURE_DIRECTORIES))

    # Assert
    assert sorted(named) == files
    assert len(set(named)) == len(named)
    assert len({case["id"] for case in _cases()}) == len(_cases())


def test_validator_when_a_schema_is_not_one_of_the_contract_should_refuse_to_make_one():
    # Act / Assert
    with pytest.raises(ValueError):
        contract.validator_for("carrier.schema.json")


def test_registry_when_a_schema_refers_outside_the_tree_should_not_fetch_it():
    # Arrange: a schema of the test's own whose $ref names a document nobody registered
    schema = {"$ref": "https://schemas.video-hub.invalid/import/v1/absent.schema.json"}
    validator = Draft202012Validator(schema, registry=contract.local_registry())

    # Act / Assert
    with pytest.raises(Unresolvable):
        list(validator.iter_errors({}))


# --- the receipt's MAC (README section 5) --------------------------------------------------------------------------


def test_receipt_mac_when_computed_for_the_known_answer_should_give_its_canonical_form_and_its_mac():
    # Arrange
    vector = contract.read_json(MAC_VECTOR)

    # Act
    canonical = contract.canonical_form(vector["receipt"])
    mac = contract.receipt_mac(vector["receipt"], vector["key"])

    # Assert
    assert canonical == vector["canonical"]
    assert mac == vector["mac"] == vector["receipt"]["mac"]


def test_receipt_mac_when_the_receipt_has_no_mac_yet_should_give_the_same_mac():
    # Arrange
    vector = contract.read_json(MAC_VECTOR)
    unsigned = {name: value for name, value in vector["receipt"].items() if name != "mac"}

    # Act
    mac = contract.receipt_mac(unsigned, vector["key"])

    # Assert
    assert mac == vector["mac"]


def test_receipt_mac_when_one_etag_changes_should_change():
    # Arrange
    vector = contract.read_json(MAC_VECTOR)
    changed = json.loads(json.dumps(vector["receipt"]))
    changed["parts"][0]["etag"] = changed["parts"][0]["etag"][:-1] + "0"

    # Act
    mac = contract.receipt_mac(changed, vector["key"])

    # Assert
    assert mac != vector["mac"]


def test_canonical_form_when_a_string_holds_characters_outside_ascii_and_slashes_should_write_them_as_themselves():
    # Arrange
    receipt = {"b": "a/b é", "a": [{"z": 1, "y": '"\\'}]}

    # Act
    canonical = contract.canonical_form(receipt)

    # Assert
    assert canonical == '{"a":[{"y":"\\"\\\\","z":1}],"b":"a/b é"}'


def test_canonical_form_when_a_number_is_an_integral_float_should_write_it_as_an_integer():
    # Act
    canonical = contract.canonical_form({"bytes": 943718400.0})

    # Assert
    assert canonical == '{"bytes":943718400}'


def test_canonical_form_when_a_number_is_not_an_integer_should_refuse_the_receipt():
    # Act / Assert
    with pytest.raises(NoCanonicalForm):
        contract.canonical_form({"bytes": 1.5})


def test_receipt_mac_when_the_key_is_empty_should_refuse():
    # Act / Assert
    with pytest.raises(ValueError):
        contract.receipt_mac({"bytes": 1}, "")


# --- the input's MAC (README section 3.1) --------------------------------------------------------------------------

INPUT_MAC_VECTOR = "fixtures/input-mac-vector.json"
VALID_INPUTS = (
    "fixtures/input/valid-running-example.json",
    "fixtures/input/valid-one-part-with-resource-key.json",
    "fixtures/input/valid-largest.json",
)


def test_input_mac_when_computed_for_the_known_answer_should_give_its_canonical_form_and_its_mac():
    # Arrange
    vector = contract.read_json(INPUT_MAC_VECTOR)

    # Act
    canonical = contract.canonical_form(vector["input"])
    mac = contract.input_mac(vector["input"], vector["key"])

    # Assert
    assert vector["prefix"] == contract.INPUT_MAC_PREFIX == "import/v1 input\n"
    assert canonical == vector["canonical"]
    assert mac == vector["mac"] == vector["input"]["mac"]
    assert contract.input_mac_verifies(vector["input"], vector["key"])


@pytest.mark.parametrize("fixture", VALID_INPUTS)
def test_input_mac_when_a_valid_input_fixture_is_checked_under_the_vectors_key_should_verify(fixture):
    # Arrange
    key = contract.read_json(INPUT_MAC_VECTOR)["key"]

    # Act
    verifies = contract.input_mac_verifies(contract.read_json(fixture), key)

    # Assert
    assert verifies


def test_input_mac_when_computed_should_never_be_the_receipts_mac_of_the_same_members():
    # Arrange
    vector = contract.read_json(INPUT_MAC_VECTOR)

    # Act
    as_input = contract.input_mac(vector["input"], vector["key"])
    as_receipt = contract.receipt_mac(vector["input"], vector["key"])

    # Assert: the prefix is part of what is signed
    assert as_input != as_receipt


def _vector_input(**changes: object) -> dict:
    document = json.loads(json.dumps(contract.read_json(INPUT_MAC_VECTOR)["input"]))
    document.update(changes)
    return document


@pytest.mark.parametrize(
    ("document", "key"),
    [
        pytest.param(_vector_input(file_id="1AnotherFileOfTheSameSize0123456"), None, id="another-file-id"),
        pytest.param(_vector_input(mac="0" * 64), None, id="another-mac"),
        pytest.param(_vector_input(mac=contract.read_json(INPUT_MAC_VECTOR)["mac"].upper()), None,
                     id="mac-in-capitals"),
        pytest.param(_vector_input(mac="é" * 64), None, id="a-mac-outside-ascii"),
        pytest.param(_vector_input(mac=None), None, id="a-mac-that-is-no-text"),
        pytest.param({**_vector_input(), "expected": {**_vector_input()["expected"], "size": 1.5}}, None,
                     id="a-number-that-is-no-integer"),
        pytest.param(_vector_input(), "another-import-key-of-32-characters", id="another-key"),
        pytest.param(_vector_input(), "", id="an-empty-key"),
    ],
)
def test_input_mac_verifies_when_the_input_or_the_key_is_not_the_signed_ones_should_say_no(document, key):
    # Arrange
    import_key = contract.read_json(INPUT_MAC_VECTOR)["key"] if key is None else key

    # Act
    verifies = contract.input_mac_verifies(document, import_key)

    # Assert
    assert verifies is False


def test_input_mac_verifies_when_the_input_has_no_mac_should_say_no():
    # Arrange
    document = {name: value for name, value in _vector_input().items() if name != "mac"}

    # Act / Assert
    assert contract.input_mac_verifies(document, contract.read_json(INPUT_MAC_VECTOR)["key"]) is False
