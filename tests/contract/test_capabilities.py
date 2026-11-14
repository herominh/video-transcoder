"""Capability exchange (protocol 2.2.0, a draft) conformance beyond the paired catalog: measured maxima, the
version gate both ways, context normalization and order, and that no live list holds the draft."""

import copy
import json

import pytest

from core.protocol import files, gate, pipeline, schema, semantic
from core.protocol.context import ContextIncomplete, ContextInvalid, TrustedContext
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import MessageKind, VersionChannel, VersionLine, load_registry
from tests.contract.test_maximum_size import _maximality_gaps

CATALOG = files.read_json("fixtures/cases.json")["cases"]
MAXIMA = [case for case in CATALOG
          if case["rule"].startswith("capabilities-") and case["purpose"].startswith("maximal escaped")]
CONTEXT_ROWS = [case for case in CATALOG if case["rule"] == "capabilities-contexts" and case["purpose"].endswith("valid")]
# The measured bytes of the catalog's maximal escaped instances, which pass K5 too (README section 13).
MEASURED_MAXIMA = {
    MessageKind.EXECUTION_CAPABILITIES_REQUEST: 29_062,
    MessageKind.EXECUTION_CAPABILITIES_GRANTED: 959_619,
}
# The schema-only worst cases: the longest version and audience strings, and every URL of the answer 4,096
# characters of the costliest schema-valid character after `https://` (`/`, written `\/`). They need not pass K5.
SCHEMA_MAXIMA = {
    MessageKind.EXECUTION_CAPABILITIES_REQUEST: 29_080,
    MessageKind.EXECUTION_CAPABILITIES_GRANTED: 1_076_500,
}
URL_SCHEME = "https://"
COSTLIEST_URL_CHARACTER = "/"
MD5_LENGTH = 24
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_SEQ = 2_147_483_647
MAX_DURATION_MS = 21_600_000
MAX_LIFETIME_SECONDS = 300
MAX_OBJECT_BYTES = 67_108_864
LONGEST_VERSION = "9999.9999.9999-draft"
LONGEST_AUDIENCE = "worker:a" + "9" * 39
URL_LENGTH = 4_096
PATH_LENGTH = 128
OBJECTS = 128
KNOWN_VERSIONS = ("2.0.0", "2.1.0", "2.2.0")


def _wire(document):
    return json.dumps(document, separators=(",", ":")).replace("/", "\\/").encode("ascii")


def _case(purpose):
    return next(case for case in CATALOG if case["purpose"] == purpose)


def _schema_maximal(case):
    document = files.read_json(case["payload"])
    document["protocol_version"] = LONGEST_VERSION
    document["audience"] = LONGEST_AUDIENCE
    kind = MessageKind(document["message_kind"])
    if kind is MessageKind.EXECUTION_CAPABILITIES_GRANTED:
        costliest = URL_SCHEME + COSTLIEST_URL_CHARACTER * (URL_LENGTH - len(URL_SCHEME))
        document["source"]["url"] = costliest
        for item in document["put"]["objects"]:
            item["url"] = costliest
    return document, kind


@pytest.mark.parametrize("case", MAXIMA, ids=lambda case: case["id"])
def test_capability_maximum_when_measured_should_hold_every_member_at_its_bound_and_fit_its_cap(case):
    # Arrange
    raw = files.read_bytes(case["payload"])
    registry = load_registry()
    # Act
    document = gate.parse(raw, registry.max_json_depth, registry.max_tokens)
    kind = MessageKind(document["message_kind"])
    # Assert
    assert document["seq"] == MAX_SEQ
    assert all(document["identity"][fence] == MAX_SAFE_INTEGER
               for fence in ("placement_epoch", "lifecycle_revision", "execution_fence"))
    assert len(document["put"]["objects"]) == OBJECTS
    assert all(len(item["path"]) == PATH_LENGTH for item in document["put"]["objects"])
    if kind is MessageKind.EXECUTION_CAPABILITIES_GRANTED:
        assert all(len(item["url"]) == URL_LENGTH for item in document["put"]["objects"])
        assert len(document["source"]["url"]) == URL_LENGTH
    else:
        assert document["media_duration_ms"] == MAX_DURATION_MS
        assert document["put"]["lifetime_seconds"] == MAX_LIFETIME_SECONDS
        assert all(item["size_bytes"] == MAX_OBJECT_BYTES and len(item["md5"]) == MD5_LENGTH
                   for item in document["put"]["objects"])
    assert len(raw) == MEASURED_MAXIMA[kind]
    assert len(raw) <= registry.max_bytes(kind)


@pytest.mark.parametrize("case", MAXIMA, ids=lambda case: case["id"])
def test_capability_maximum_when_widened_to_the_schema_alone_should_reach_each_bound_and_fit_its_cap(case):
    # Arrange
    document, kind = _schema_maximal(case)
    uri = schema.SCHEMA_BASE_URI + kind.schema_file
    gaps = []
    # Act
    _maximality_gaps(document, schema.resolve_reference(uri, uri), uri, "", gaps)
    raw = _wire(document)
    # Assert
    assert gaps == []
    assert schema.error_pointers(kind.schema_file, document) == []
    assert len(raw) == SCHEMA_MAXIMA[kind]
    assert len(raw) <= load_registry().max_bytes(kind)


ALL_MAX_BYTES = {
    "transcode.request": 4_096, "transcode.progress": 4_096, "transcode.result.completed": 16_384,
    "transcode.result.failed": 16_384, "hub.error": 2_048, "generation.manifest": 8_388_608,
    "transcode.claim": 4_096, "transcode.claim.granted": 4_096, "transcode.unclaimed": 4_096,
    "source.acquire.request": 4_096, "source.acquire.granted": 2_097_152, "source.acquire.result.completed": 65_536,
    "source.acquire.result.failed": 4_096, "execution.capabilities.request": 32_768,
    "execution.capabilities.granted": 2_097_152,
}
ALL_CARRIER_MAX_BYTES = {"runpod.input": 8_448, "runpod.output": 33_024, "runpod.acquire.output": 131_328}


def test_byte_caps_when_read_should_be_every_cap_the_contract_states():
    # Arrange
    registry = load_registry()
    # Act
    max_bytes = {kind.value: registry.max_bytes(kind) for kind in MessageKind}
    # Assert
    assert max_bytes == ALL_MAX_BYTES
    assert dict(registry.limits["max_bytes"]) == ALL_MAX_BYTES
    assert dict(registry.limits["carrier_max_bytes"]) == ALL_CARRIER_MAX_BYTES
    assert registry.max_tokens == 277_507
    assert registry.limits["capability_closure_margin_seconds"] == 60


# The kinds that exist at one protocol version only, by their literal names (README section 5).
SINGLE_VERSION_KINDS = {
    "source.acquire.request": "2.1.0", "source.acquire.granted": "2.1.0",
    "source.acquire.result.completed": "2.1.0", "source.acquire.result.failed": "2.1.0",
    "execution.capabilities.request": "2.2.0", "execution.capabilities.granted": "2.2.0",
}


def _existence(kind, version):
    return SINGLE_VERSION_KINDS.get(kind.value, version) == version


VERSIONED_KINDS = [(kind, version) for kind in MessageKind if kind is not MessageKind.GENERATION_MANIFEST
                   for version in KNOWN_VERSIONS]


@pytest.mark.parametrize("kind,version", VERSIONED_KINDS, ids=[f"{kind.value}@{version}" for kind, version in VERSIONED_KINDS])
def test_version_gate_when_a_kind_is_named_at_a_known_version_should_select_it_only_where_it_exists(kind, version):
    # Arrange: a receiver that lists every known protocol version and the kind.
    data = files.read_json(_case("row request-hub valid")["context"])
    data["accepted_protocol_versions"] = list(KNOWN_VERSIONS)
    context = TrustedContext.from_dict(data)
    header = {"protocol_version": version, "message_kind": kind.value}
    # Act
    selected = gate.select_kind(header, context)
    # Assert
    if not _existence(kind, version):
        assert (selected.layer, selected.reason, selected.instance_path) == (
            Layer.VERSION, ReasonCode.UNKNOWN_MESSAGE_KIND, "/message_kind")
    elif kind is MessageKind.EXECUTION_CAPABILITIES_REQUEST:
        assert selected is kind
    else:
        assert (selected.layer, selected.reason) == (Layer.VERSION, ReasonCode.UNEXPECTED_MESSAGE_KIND)


def test_capability_request_when_the_context_lists_only_the_live_versions_should_be_refused_before_its_kind():
    # Arrange
    case = _case("heartbeat at transcoding")
    data = files.read_json(case["context"])
    data["accepted_protocol_versions"] = list(load_registry().accepted_versions(VersionLine.PROTOCOL, VersionChannel.LIVE))
    # Act
    verdict = pipeline.validate(files.read_bytes(case["payload"]), TrustedContext.from_dict(data))
    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.VERSION, ReasonCode.UNSUPPORTED_VERSION, "/protocol_version")


def test_versions_when_read_should_list_two_two_zero_as_a_draft_and_never_as_live():
    # Arrange
    registry = load_registry()
    # Act
    live = registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.LIVE)
    draft = registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.DRAFT)
    # Assert
    assert "2.2.0" not in live
    assert draft == ("2.2.0",)
    assert registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.LIVE) == ("1.0.0",)


def test_capability_context_when_integers_are_written_as_integer_valued_floats_should_hold_the_integers():
    # Arrange
    granted = files.read_json(_case("row granted-worker valid")["context"])
    granted["expect"]["capability_request"]["seq"] = 5.0
    granted["expect"]["capability_request"]["put"] = {"lifetime_seconds": 180.0, "paths": ["thumbnail.jpg"]}
    # Act
    context = TrustedContext.from_dict(granted)
    # Assert
    answered = context.value("expect.capability_request")
    assert type(answered["seq"]) is int and answered["seq"] == 5
    assert type(answered["put"]["lifetime_seconds"]) is int and answered["put"]["lifetime_seconds"] == 180
    assert answered["put"]["paths"] == ("thumbnail.jpg",)


@pytest.mark.parametrize("path,value", [
    ("seq", 1.5), ("seq", "5"), ("put", {"lifetime_seconds": 180.5, "paths": ["a.ts"]}),
    ("put", {"lifetime_seconds": True, "paths": ["a.ts"]}), ("put", {"lifetime_seconds": 180, "paths": {"0": "a.ts"}}),
], ids=["seq-fraction", "seq-string", "lifetime-fraction", "lifetime-boolean", "paths-as-a-map"])
def test_capability_context_when_the_answered_request_is_malformed_should_refuse_before_normalizing(path, value):
    # Arrange
    granted = files.read_json(_case("row granted-worker valid")["context"])
    granted["expect"]["capability_request"][path] = value
    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(granted)


@pytest.mark.parametrize("case", CONTEXT_ROWS, ids=lambda case: case["id"])
def test_capability_context_when_fields_are_missing_should_report_them_in_table_order(case):
    # Arrange
    data = files.read_json(case["context"])
    ordered = ["accepted_protocol_versions", "expected_audience", "known_key_ids", "now", "clock_skew_s",
               *("expect." + name for name in data["expect"])]
    if "last_capability_seq" in data:
        ordered.append("last_capability_seq")
    # Act / Assert
    assert list(semantic.required_context_fields(data["role"], data["accepted_message_kinds"][0])) == ordered
    for index, first in enumerate(ordered):
        changed = copy.deepcopy(data)
        for field in ordered[index:]:
            if field.startswith("expect."):
                del changed["expect"][field.removeprefix("expect.")]
            else:
                del changed[field]
        with pytest.raises(ContextIncomplete) as caught:
            TrustedContext.from_dict(changed)
        assert caught.value.field == first
