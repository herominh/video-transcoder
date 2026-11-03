"""Every catalog case yields its pinned verdict through the Python pipeline (the Hub runs the same catalog)."""

from typing import Any

import pytest

from core.protocol import files, gate, pipeline, schema, semantic
from core.protocol.context import TrustedContext, ContextInvalid, ContextIncomplete
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import MessageKind, load_registry

CATALOG_FILE = "fixtures/cases.json"
CATALOG_SCHEMA_FILE = "schemas/fixture-catalog.schema.json"
PAYLOAD_DIRECTORIES = (
    "request", "progress", "result-completed", "result-failed", "hub-error", "manifest", "claim", "claim-granted",
    "unclaimed", "acquire-request", "acquire-grant", "acquire-results",
)
# README section 14: an S7 case also fails the equality check of the aliased id where its row runs one (S13 for
# N45, S15 for N91); N126, whose row runs no S15, fails S21 instead. Every other negative fails its pinned check
# alone: the S7 cases of a row that compares no identity (N90, N92, N93) and N132, whose runtime id no equality
# check of its row reads.
S7_COMPANION_CHECK = {"N45": "S13", "N91": "S15", "N126": "S21", "N747": "A1"}

CATALOG: dict[str, Any] = files.read_json(CATALOG_FILE)
POSITIVE_CASES = [case for case in CATALOG["cases"] if case["expect"]["outcome"] == "accept"]
NEGATIVE_CASES = [case for case in CATALOG["cases"] if case["expect"]["outcome"] == "reject"]


def _validate_case(case, raw, context):
    if "carrier" in case:
        registry = load_registry()
        if len(raw) > registry.carrier_max_bytes(case["carrier"]):
            from core.protocol.verdict import Verdict
            return Verdict.reject(Layer.SIZE, ReasonCode.PAYLOAD_TOO_LARGE, None)
        outer = gate.parse(raw, registry.max_json_depth, registry.max_tokens)
        schema_file = "carriers/" + case["carrier"].replace(".", "-") + ".schema.json"
        pointer = schema.first_error_pointer(schema_file, outer)
        if pointer is not None:
            from core.protocol.verdict import Verdict
            return Verdict.reject(Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION, pointer)
        from core.protocol.signing import verify
        assert verify(outer["vh_message"].encode("ascii"), b"test-only-wk2hub-k1-not-a-secret", outer["vh_signature"])
        raw = outer["vh_message"].encode("ascii")
    return pipeline.validate(raw, context)


CONTEXT_CASES = [case for case in CATALOG["cases"] if case["expect"]["layer"] == "context"]


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda case: case["id"])
def test_context_fixture_when_constructed_should_raise_the_pinned_error(case):
    # Arrange
    expected = case["expect"]
    error_type = ContextIncomplete if expected["outcome"] == "context_incomplete" else ContextInvalid
    # Act / Assert
    with pytest.raises(error_type) as caught:
        _context(case)
    if error_type is ContextIncomplete:
        assert caught.value.field == expected["instance_path"][1:].replace("/", ".")
        data = files.read_json(case["context"])
        assert caught.value.role == data["role"]
        assert caught.value.kind == data["accepted_message_kinds"][0]


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
    verdict = _validate_case(case, raw, context)

    # Assert
    assert verdict.accepted, verdict


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=[case["id"] for case in NEGATIVE_CASES])
def test_negative_fixture_when_validated_should_be_rejected_at_expected_layer_with_expected_reason(case):
    # Arrange
    raw = files.read_bytes(case["payload"])
    context = _context(case)
    expected = case["expect"]

    # Act
    verdict = _validate_case(case, raw, context)

    # Assert: at L3 too the pinned pointer is the first error reported (README section 6).
    assert not verdict.accepted
    assert verdict.layer == Layer(expected["layer"])
    assert verdict.reason == ReasonCode(expected["reason"])
    assert verdict.instance_path == expected["instance_path"]


def _schema_error_pointers(kind, document):
    def reported(errors):
        for error in errors:
            yield schema.json_pointer(error.absolute_path)
            # A8's failed alternatives explain the oneOf error at /error. Suppress only those
            # descendants, preserving every independent sibling and all other schema sub-errors.
            if (kind is MessageKind.SOURCE_ACQUIRE_RESULT_FAILED
                    and error.validator == "oneOf"
                    and tuple(error.absolute_schema_path) == ("properties", "error", "oneOf")
                    and tuple(error.absolute_path) == ("error",)):
                continue
            yield from reported(error.context)

    return set(reported(schema.validator_for(kind.schema_file).iter_errors(document)))


@pytest.mark.parametrize(
    "case",
    [case for case in NEGATIVE_CASES if case["expect"]["layer"] in (Layer.SCHEMA.value, Layer.SEMANTIC.value)],
    ids=lambda case: case["id"],
)
def test_negative_fixture_when_evaluated_should_violate_only_its_own_rule(case):
    # Arrange
    pinned = case["expect"]["instance_path"]
    if "carrier" in case:
        carrier_schema = "carriers/" + case["carrier"].replace(".", "-") + ".schema.json"
        assert set(schema.error_pointers(carrier_schema, files.read_json(case["payload"]))) == {pinned}
        return
    document, kind, context = _selected(case)

    # Act
    if case["expect"]["layer"] == Layer.SCHEMA.value:
        reported = _schema_error_pointers(kind, document)
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
        if case["id"] == "N767":
            assert first.check == "S18" and [other.check for other in others] == ["A4"]
            return
        assert companion is None or first.check == "S7"
        assert [other.check for other in others] == ([] if companion is None else [companion]), others


@pytest.mark.parametrize("case_id,field,value", [
    ("N809", "external.resource_key", "a" * 129),
    ("N775", "error.detail", "a" * 1001),
    ("N775", "identity.placement_epoch", 0),
], ids=["request-second-field", "A8-error-sibling", "A8-envelope-sibling"])
def test_single_rule_assertion_when_acquire_negative_has_a_second_error_should_fail(monkeypatch, case_id, field, value):
    # Arrange: exercise the catalog assertion itself, including siblings of A8's oneOf.
    case = next(case for case in NEGATIVE_CASES if case["id"] == case_id)
    document, kind, context = _selected(case)
    cursor = document
    members = field.split(".")
    for member in members[:-1]:
        cursor = cursor[member]
    cursor[members[-1]] = value
    monkeypatch.setattr(f"{__name__}._selected", lambda _: (document, kind, context))
    # Act / Assert
    with pytest.raises(AssertionError):
        test_negative_fixture_when_evaluated_should_violate_only_its_own_rule(case)


def test_catalog_when_validated_should_satisfy_catalog_schema():
    # Act
    pointers = schema.error_pointers(CATALOG_SCHEMA_FILE, CATALOG)

    # Assert
    assert pointers == []


def test_catalog_when_counted_should_have_contiguous_ids_with_p17_retired():
    # Arrange
    ids = {case["id"] for case in CATALOG["cases"]}
    # Act
    maxima = {prefix: max(int(value[1:]) for value in ids if value.startswith(prefix)) for prefix in ("P", "N")}
    expected = {f"{prefix}{number:02d}" for prefix, maximum in maxima.items() for number in range(1, maximum + 1)} - {"P17"}
    # Assert
    assert ids == expected
    assert len(POSITIVE_CASES) + len(NEGATIVE_CASES) + len(CONTEXT_CASES) == len(CATALOG["cases"])


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
