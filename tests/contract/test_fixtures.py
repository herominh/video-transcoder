"""Every catalog case yields its pinned verdict through the Python pipeline (the Hub runs the same catalog)."""

from typing import Any

import pytest

from tests.contract import files, gate, pipeline, schema, semantic
from tests.contract.context import TrustedContext
from tests.contract.reasons import Layer, ReasonCode
from tests.contract.registry import load_registry

CATALOG_FILE = "fixtures/cases.json"
CATALOG_SCHEMA_FILE = "schemas/fixture-catalog.schema.json"
PAYLOAD_DIRECTORIES = (
    "request", "progress", "result-completed", "result-failed", "hub-error", "manifest", "claim", "claim-granted",
    "unclaimed",
)
EXPECTED_POSITIVE_IDS = {f"P{number:02d}" for number in range(1, 27)} - {"P17"}  # P17 retired into N76
EXPECTED_NEGATIVE_IDS = {f"N{number:02d}" for number in range(1, 133)}

# README section 14: an S7 case also fails the equality check of the aliased id where its row runs one (S13 for
# N45, S15 for N91); N126, whose row runs no S15, fails S21 instead. Every other negative fails its pinned check
# alone: the S7 cases of a row that compares no identity (N90, N92, N93) and N132, whose runtime id no equality
# check of its row reads.
S7_COMPANION_CHECK = {"N45": "S13", "N91": "S15", "N126": "S21"}

CATALOG: dict[str, Any] = files.read_json(CATALOG_FILE)
POSITIVE_CASES = [case for case in CATALOG["cases"] if case["expect"]["outcome"] == "accept"]
NEGATIVE_CASES = [case for case in CATALOG["cases"] if case["expect"]["outcome"] == "reject"]


def _context(case: dict[str, Any]) -> TrustedContext:
    return TrustedContext.from_dict(files.read_json(case["context"]))


def _selected(case: dict[str, Any]):
    raw = files.read_bytes(case["payload"])
    context = _context(case)
    registry = load_registry()
    document = gate.parse(raw, registry.max_json_depth, registry.max_tokens)
    return document, gate.select_kind(document, context), context


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=[case["id"] for case in POSITIVE_CASES])
def test_positive_fixture_when_validated_should_be_accepted(case):
    # Arrange
    raw = files.read_bytes(case["payload"])
    context = _context(case)

    # Act
    verdict = pipeline.validate(raw, context)

    # Assert
    assert verdict.accepted, verdict


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=[case["id"] for case in NEGATIVE_CASES])
def test_negative_fixture_when_validated_should_be_rejected_at_expected_layer_with_expected_reason(case):
    # Arrange
    raw = files.read_bytes(case["payload"])
    context = _context(case)
    expected = case["expect"]

    # Act
    verdict = pipeline.validate(raw, context)

    # Assert: at L3 too the pinned pointer is the first error reported (README section 6).
    assert not verdict.accepted
    assert verdict.layer == Layer(expected["layer"])
    assert verdict.reason == ReasonCode(expected["reason"])
    assert verdict.instance_path == expected["instance_path"]


@pytest.mark.parametrize(
    "case",
    [case for case in NEGATIVE_CASES if case["expect"]["layer"] in (Layer.SCHEMA.value, Layer.SEMANTIC.value)],
    ids=lambda case: case["id"],
)
def test_negative_fixture_when_evaluated_should_violate_only_its_own_rule(case):
    # Arrange
    document, kind, context = _selected(case)
    pinned = case["expect"]["instance_path"]

    # Act
    if case["expect"]["layer"] == Layer.SCHEMA.value:
        reported = set(schema.error_pointers(kind.schema_file, document))
        found = None
    else:
        reported = None
        found = semantic.failures(kind, document, context, load_registry())

    # Assert
    if found is None:
        assert reported == {pinned}
    else:
        first, others = found[0], found[1:]
        assert (first.reason.value, first.instance_path) == (case["expect"]["reason"], pinned)
        companion = S7_COMPANION_CHECK.get(case["id"])
        assert companion is None or first.check == "S7"
        assert [other.check for other in others] == ([] if companion is None else [companion]), others


def test_catalog_when_validated_should_satisfy_catalog_schema():
    # Act
    pointers = schema.error_pointers(CATALOG_SCHEMA_FILE, CATALOG)

    # Assert
    assert pointers == []


def test_catalog_when_counted_should_hold_25_positive_and_132_negative_cases():
    # Act
    counts = (len(POSITIVE_CASES), len(NEGATIVE_CASES))

    # Assert
    assert counts == (25, 132)


def test_catalog_when_listed_should_hold_exactly_the_expected_case_ids():
    # Act
    ids = ({case["id"] for case in POSITIVE_CASES}, {case["id"] for case in NEGATIVE_CASES})

    # Assert
    assert ids == (EXPECTED_POSITIVE_IDS, EXPECTED_NEGATIVE_IDS)


def test_catalog_when_scanned_should_use_each_case_id_once():
    # Act
    ids = [case["id"] for case in CATALOG["cases"]]

    # Assert
    assert len(ids) == len(set(ids))


def test_catalog_when_scanned_should_cover_every_non_reserved_reason_code():
    # Arrange
    non_reserved = {entry["code"] for entry in load_registry().reason_codes if not entry["reserved"]}

    # Act
    covered = {case["expect"]["reason"] for case in NEGATIVE_CASES}

    # Assert
    assert non_reserved - covered == set()
    assert covered <= non_reserved


def test_catalog_when_scanned_should_reference_only_existing_files():
    # Arrange
    present = set(files.list_tree_files())

    # Act
    referenced = {case["payload"] for case in CATALOG["cases"]} | {case["context"] for case in CATALOG["cases"]}

    # Assert
    assert referenced - present == set()


def test_fixture_files_when_listed_should_each_be_used_by_a_case():
    # Arrange
    referenced = {case["payload"] for case in CATALOG["cases"]} | {case["context"] for case in CATALOG["cases"]}

    # Act
    fixture_files = {
        path
        for path in files.list_tree_files()
        if path.startswith("fixtures/") and path != CATALOG_FILE
    }

    # Assert
    assert fixture_files - referenced == set()
    assert {path.split("/")[1] for path in fixture_files} == {*PAYLOAD_DIRECTORIES, "contexts"}
