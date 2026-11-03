"""Acquire minor conformance beyond the paired catalog: escaped maxima, signatures and context order."""

import copy
import hashlib
import hmac
import json

import pytest

from core.protocol import files, gate, pipeline, schema, signing, semantic
from core.protocol.context import TrustedContext, ContextIncomplete, ContextInvalid
from core.protocol.registry import MessageKind, load_registry
from tests.contract.test_maximum_size import _maximality_gaps

CATALOG = files.read_json("fixtures/cases.json")["cases"]
MAXIMA = [case for case in CATALOG if case["purpose"].startswith("maximal envelope source acquire")]
ACQUIRE_POSITIVES = [case for case in CATALOG if case["expect"]["outcome"] == "accept"
                     and case["rule"] in ("acquire-request", "acquire-grant", "acquire-results")]
ACQUIRE_INTEGER_FIELDS = (
    "acquire_external.expected.size", "acquire_plan.part_size", "acquire_plan.parts_total",
    "acquire_limits.max_source_bytes", "acquire_limits.max_wall_time_ms",
)


@pytest.mark.parametrize("case_id", ["P35", "P38"])
@pytest.mark.parametrize("field", ACQUIRE_INTEGER_FIELDS)
def test_acquire_context_when_integer_valued_floats_are_used_should_normalize_and_accept_the_grant(case_id, field):
    # Arrange: contexts retain the existing numeric-value rule; wire messages use integer tokens.
    case = next(case for case in CATALOG if case["id"] == case_id)
    data = files.read_json(case["context"])
    data["clock_skew_s"] = float(data["clock_skew_s"])
    data["expect"]["placement_epoch"] = float(data["expect"]["placement_epoch"])
    cursor = data["expect"]
    members = field.split(".")
    for member in members[:-1]:
        cursor = cursor[member]
    expected = cursor[members[-1]]
    cursor[members[-1]] = float(expected)
    # Act
    context = TrustedContext.from_dict(json.loads(json.dumps(data)))
    verdict = pipeline.validate(files.read_bytes(case["payload"]), context)
    # Assert
    assert verdict.accepted, verdict
    actual = context.expect
    for member in members:
        actual = actual[member]
    assert type(actual) is int
    assert actual == expected
    assert context.clock_skew_s == 300
    assert context.expect["placement_epoch"] == 7


@pytest.mark.parametrize("field", ACQUIRE_INTEGER_FIELDS)
@pytest.mark.parametrize("invalid", [1.5, "1", True], ids=["fraction", "string", "boolean"])
def test_acquire_context_when_integer_members_are_invalid_should_refuse_before_normalizing(field, invalid):
    # Arrange
    data = files.read_json("fixtures/contexts/ctx-acquire-p35.json")
    cursor = data["expect"]
    members = field.split(".")
    for member in members[:-1]:
        cursor = cursor[member]
    cursor[members[-1]] = invalid
    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(data)


def _wire(document):
    return json.dumps(document, separators=(",", ":")).replace("/", "\\/").encode("ascii")


def _maximal_schema_instance(case):
    document = files.read_json(case["payload"])
    document["protocol_version"] = "9999.9999.9999-draft"
    kind = MessageKind(document["message_kind"])
    if kind is MessageKind.SOURCE_ACQUIRE_REQUEST:
        document["destination"]["object_key"] = "a/" * 255 + "aa"
    if kind is MessageKind.SOURCE_ACQUIRE_GRANTED:
        document["output"]["prefix"] = "a/" * 255 + "aa"
        for part in document["upload"]["parts"]:
            part["n"] = 160
    if kind is MessageKind.SOURCE_ACQUIRE_RESULT_COMPLETED:
        for part in document["parts"]:
            part["n"] = 160
    return document, kind


@pytest.mark.parametrize("case", MAXIMA, ids=lambda case: case["id"])
def test_acquire_maximum_when_serialized_with_worst_escaping_should_fit_wire_and_carrier_caps(case):
    # Arrange
    document, kind = _maximal_schema_instance(case)
    registry = load_registry()
    raw = _wire(document)
    carrier = {"vh_message": raw.decode("ascii"), "vh_signature": signing.sign(raw, b"test-only-key")}
    family = "runpod.input" if kind is MessageKind.SOURCE_ACQUIRE_REQUEST else "runpod.acquire.output"
    # Act
    pointers = schema.error_pointers(kind.schema_file, document)
    parsed = gate.parse(raw, registry.max_json_depth, registry.max_tokens)
    # Assert
    assert pointers == []
    assert isinstance(parsed, dict)
    assert len(raw) <= registry.max_bytes(kind)
    if kind is not MessageKind.SOURCE_ACQUIRE_GRANTED:
        assert len(_wire(carrier)) <= registry.carrier_max_bytes(family)
        assert schema.first_error_pointer("carriers/" + family.replace(".", "-") + ".schema.json", carrier) is None
    else:
        assert len(raw) > 1_000_000
        assert all(len(part["url"]) == 4096 for part in document["upload"]["parts"])
        assert len(document["upload"]["result_url"]) == 4096


@pytest.mark.parametrize("case", MAXIMA, ids=lambda case: case["id"])
def test_acquire_maximum_when_compared_with_schema_should_reach_each_bound(case):
    # Arrange
    document, kind = _maximal_schema_instance(case)
    uri = schema.SCHEMA_BASE_URI + kind.schema_file
    gaps = []
    # Act
    _maximality_gaps(document, schema.resolve_reference(uri, uri), uri, "", gaps)
    # Assert: error.code is narrowed by its enum to thirteen codes, shorter than diagnostic_code's bound.
    assert gaps == (["/error/code: string of 26 below maxLength 48"]
                    if kind is MessageKind.SOURCE_ACQUIRE_RESULT_FAILED else [])


@pytest.mark.parametrize("case", ACQUIRE_POSITIVES, ids=lambda case: case["id"])
def test_acquire_signature_when_direction_domain_or_exact_bytes_change_should_refuse(case):
    # Arrange
    raw = files.read_bytes(case["payload"])
    key, other_key = b"test-only-one-direction", b"test-only-other-direction"
    # Act
    signature = signing.sign(raw, key)
    wrong_domain = "hmac-sha256=" + hmac.new(key, b"video-hub.transcode.v2.1\n" + raw, hashlib.sha256).hexdigest()
    # Assert
    assert signing.verify(raw, key, signature)
    assert not signing.verify(raw, other_key, signature)
    assert not signing.verify(raw + b" ", key, signature)
    assert not signing.verify(raw, key, wrong_domain)


def test_fixture_sizes_when_measured_should_apply_the_exception_only_to_acquire_grant_payloads():
    # Arrange
    grant_payloads = {case["payload"] for case in CATALOG if case["payload"].startswith("fixtures/acquire-grant/")}
    # Act / Assert
    for path in files.list_tree_files():
        if path.startswith("fixtures/"):
            assert len(files.read_bytes(path)) <= (2_097_153 if path in grant_payloads else 999_999), path


@pytest.mark.parametrize("member", ["external.file_id", "external.resource_key", "external.expected.head_revision_id"])
def test_acquire_request_when_a_url_is_in_an_external_fact_should_refuse(member):
    # Arrange
    case = next(case for case in CATALOG if case["id"] == "P30")
    document = files.read_json(case["payload"])
    cursor = document
    parts = member.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = "https://example.invalid/opaque"
    context = TrustedContext.from_dict(files.read_json(case["context"]))
    # Act
    verdict = pipeline.validate(_wire(document), context)
    # Assert
    assert verdict.reason.value == "schema_violation"


def test_encode_caps_when_acquire_is_added_should_remain_unchanged():
    # Arrange
    registry = load_registry()
    # Act / Assert
    assert registry.max_bytes(MessageKind.TRANSCODE_CLAIM_GRANTED) == 4096
    assert registry.max_bytes(MessageKind.TRANSCODE_RESULT_COMPLETED) == 16384
    assert registry.carrier_max_bytes("runpod.output") == 33024
    assert registry.carrier_max_bytes("runpod.input") == 8448


@pytest.mark.parametrize("case", [case for case in CATALOG if case["rule"] == "acquire-contexts"
                                 and case["purpose"].endswith("valid")], ids=lambda case: case["id"])
def test_acquire_context_when_fields_are_removed_in_order_should_report_the_first_missing(case):
    # Arrange: expected order is the catalog row's literal field order, independent of validator code.
    data = files.read_json(case["context"])
    ordered = ["accepted_protocol_versions", "expected_audience", "known_key_ids", "now", "clock_skew_s"]
    ordered += ["expect." + name for name in data.get("expect", {})]
    # Act / Assert
    assert semantic.required_context_fields(data["role"], data["accepted_message_kinds"][0]) == tuple(ordered)
    for index, first in enumerate(ordered):
        changed = copy.deepcopy(data)
        for field in ordered[index:]:
            if field.startswith("expect."):
                del changed["expect"][field.split(".", 1)[1]]
            else:
                del changed[field]
        with pytest.raises(ContextIncomplete) as caught:
            TrustedContext.from_dict(changed)
        assert caught.value.field == first


@pytest.mark.parametrize("case", ACQUIRE_POSITIVES, ids=lambda case: case["id"])
def test_acquire_context_when_a_role_has_no_row_should_raise_context_invalid(case):
    # Arrange
    data = files.read_json(case["context"])
    allowed = {"worker_receiver", "hub_sender"} if data["accepted_message_kinds"][0] in {
        "source.acquire.request", "source.acquire.granted"
    } else {"worker_sender", "hub_receiver"}
    # Act / Assert
    for role in {"worker_receiver", "hub_sender", "worker_sender", "hub_receiver"} - allowed:
        with pytest.raises(ContextInvalid):
            TrustedContext.from_dict({**data, "role": role})
