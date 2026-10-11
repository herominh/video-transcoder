"""The worker's builders write exactly the catalog's positive messages, which its own self-check accepts.

Each builder is fed the values of a positive fixture of its kind and must produce that fixture: the
same members in the same order, byte for byte once encoded compactly. The receiver contexts must
equal the catalog's worker contexts.
"""

import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from core.failure import Diagnostic, Failure
from core.protocol import files, messages, pipeline, signing
from core.protocol.context import ContextIncomplete, TrustedContext
from core.protocol.messages import (
    Counters,
    DispatchIdentity,
    Encoder,
    Encryption,
    ExecutionIdentity,
    ManifestLocator,
    Media,
    MessageTooLarge,
    Profile,
    SourceRead,
    encode,
    receiver_context,
    seal,
    timestamp,
    worker_sender_context,
)
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import load_registry
from core.protocol.settings import WorkerSettings

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
VECTORS = files.read_json("signing/vectors.json")
TEST_KEYS = {entry["key_id"]: entry["key_hex"] for entry in VECTORS["keys"]}
# The catalog's test-only media key (README section 14): no message the worker builds may hold it.
CATALOG_MEDIA_KEY_HEX = "00112233445566778899aabbccddeeff"
HUB_PUSH_EXPECT = files.read_json("fixtures/contexts/ctx-hub-push.json")["expect"]
LAST_EVENT_SEQ = files.read_json("fixtures/contexts/ctx-hub-push.json")["last_event_seq"]
BUILDERS = (
    messages.claim,
    messages.progress,
    messages.result_completed,
    messages.result_failed,
    messages.unclaimed,
)


def _settings(worker_key_id: str = "test-wk2hub-k1") -> WorkerSettings:
    return WorkerSettings.from_env(
        {
            "VH_ACCEPTED_PROTOCOL_VERSIONS": "2.0.0-draft",
            "VH_WORKER_AUDIENCE": "worker:runpod-test",
            "VH_HUB_TO_WORKER_KEYS": f"test-hub2wk-k1={TEST_KEYS['test-hub2wk-k1']}",
            "VH_WORKER_TO_HUB_KEY_ID": worker_key_id,
            "VH_WORKER_TO_HUB_KEY": TEST_KEYS[worker_key_id],
            "VH_CLAIM_URL": "https://hub.test.invalid/transcode/claim",
            "VH_CALLBACK_URL": "https://hub.test.invalid/transcode/callback",
            "VH_STORAGE_PROFILES": "default",
            "VH_STORAGE_DEFAULT_ENDPOINT": "https://storage.test.invalid",
            "VH_STORAGE_DEFAULT_ACCESS_KEY_ID": "test-access-key-id",
            "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY": "test-secret-access-key",
        }
    )


def _moment(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def _envelope(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": fixture["protocol_version"],
        "message_id": fixture["message_id"],
        "sent_at": _moment(fixture["sent_at"]),
        "key_id": fixture["key_id"],
    }


def _diagnostics(fixture: dict[str, Any]) -> list[Diagnostic]:
    return [Diagnostic(**entry) for entry in fixture.get("diagnostics", [])]


def _build_claim(fixture: dict[str, Any]) -> dict[str, Any]:
    return messages.claim(
        **_envelope(fixture),
        identity=DispatchIdentity(**fixture["identity"]),
        bootstrap_token=fixture["claim"]["bootstrap_token"],
        runtime_id=fixture["claim"]["runtime_id"],
    )


def _build_progress(fixture: dict[str, Any]) -> dict[str, Any]:
    return messages.progress(
        **_envelope(fixture),
        identity=ExecutionIdentity(**fixture["identity"]),
        event_seq=fixture["event_seq"],
        stage=fixture["stage"],
        stage_progress_pct=fixture["stage_progress_pct"],
        message=fixture.get("message"),
        counters=Counters(**fixture["counters"]) if "counters" in fixture else None,
        ext=fixture.get("ext"),
    )


def _build_completed(fixture: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The fixture's completed result; `changes` replace the values the fixture gives."""
    values = dict(
        **_envelope(fixture),
        identity=ExecutionIdentity(**fixture["identity"]),
        generation_id=fixture["generation_id"],
        manifest=ManifestLocator(**fixture["manifest"]),
        source=SourceRead(**fixture["source"]),
        profile=Profile(**fixture["profile"]),
        media=Media(**fixture["media"]),
        encoder=Encoder(**fixture["encoder"]),
        diagnostics=_diagnostics(fixture),
        ext=fixture.get("ext"),
    )
    return messages.result_completed(**{**values, **changes})


def _build_failed(fixture: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The fixture's failed result; `changes` replace the values the fixture gives."""
    error = fixture["error"]
    values = dict(
        **_envelope(fixture),
        identity=ExecutionIdentity(**fixture["identity"]),
        failure=Failure(error_class=error["class"], code=error["code"], retryable=error["retryable"], detail=error["detail"]),
        stage=error["stage"],
        source=SourceRead(**fixture["source"]) if "source" in fixture else None,
        encoder=Encoder(**fixture["encoder"]) if "encoder" in fixture else None,
        diagnostics=_diagnostics(fixture),
        ext=fixture.get("ext"),
    )
    return messages.result_failed(**{**values, **changes})


def _build_unclaimed(fixture: dict[str, Any]) -> dict[str, Any]:
    return messages.unclaimed(
        **_envelope(fixture),
        identity=DispatchIdentity(**fixture["identity"]),
        runtime_id=fixture["runtime_id"],
        cause=fixture["cause"],
        hub_error_code=fixture.get("hub_error_code"),
    )


# (fixture, builder, self-check channel); P09 and P26 are provider output polled hours later.
POSITIVE_CASES = [
    ("fixtures/claim/P23-claim-of-the-accepted-dispatch.json", _build_claim, "push"),
    ("fixtures/progress/P04-downloading.json", _build_progress, "push"),
    ("fixtures/progress/P05-counters-and-unknown-ext-member.json", _build_progress, "push"),
    ("fixtures/progress/P06-sent-exactly-300s-ago.json", _build_progress, "push"),
    ("fixtures/progress/P19-sent-exactly-300s-ahead.json", _build_progress, "push"),
    ("fixtures/result-completed/P07-full.json", _build_completed, "push"),
    ("fixtures/result-completed/P08-thumbnail-unavailable-diagnostic.json", _build_completed, "push"),
    ("fixtures/result-completed/P09-polled-three-hours-late.json", _build_completed, "poll"),
    ("fixtures/result-failed/P10-source-unavailable-retryable.json", _build_failed, "push"),
    ("fixtures/result-failed/P11-minimal-stage-null.json", _build_failed, "push"),
    ("fixtures/unclaimed/P25-refused-with-claim-conflict.json", _build_unclaimed, "push"),
    ("fixtures/unclaimed/P26-unreachable-polled-three-hours-late.json", _build_unclaimed, "poll"),
]
CASE_IDS = [path.rsplit("/", 1)[-1].split("-", 1)[0] for path, _, _ in POSITIVE_CASES]


def _fixture(path: str) -> dict[str, Any]:
    return json.loads(files.read_bytes(path))


def _sender_context(message: dict[str, Any], channel: str) -> TrustedContext:
    identity_type = ExecutionIdentity if "execution_id" in message["identity"] else DispatchIdentity
    return worker_sender_context(
        message["message_kind"],
        identity_type(**message["identity"]),
        settings=_settings(message["key_id"]),
        now=NOW,
        channel=channel,
        last_event_seq=LAST_EVENT_SEQ,
        source_id=HUB_PUSH_EXPECT["source_id"],
        output_location_id=HUB_PUSH_EXPECT["output_location_id"],
        profile=Profile(**HUB_PUSH_EXPECT["profile"]),
    )


# --- the builders write the catalog's messages -------------------------------------------------


@pytest.mark.parametrize(("path", "build", "channel"), POSITIVE_CASES, ids=CASE_IDS)
def test_builder_when_fed_a_positive_fixtures_values_should_return_that_fixture(path, build, channel):
    # Arrange
    fixture = _fixture(path)

    # Act
    built = build(fixture)

    # Assert
    assert built == fixture
    assert encode(built) == json.dumps(fixture, separators=(",", ":")).encode("ascii")


@pytest.mark.parametrize(("path", "build", "channel"), POSITIVE_CASES, ids=CASE_IDS)
def test_builder_output_when_self_checked_as_worker_sender_should_be_accepted(path, build, channel):
    # Arrange
    built = build(_fixture(path))

    # Act
    verdict = pipeline.validate(encode(built), _sender_context(built, channel))

    # Assert
    assert verdict.accepted, verdict


def test_self_check_when_the_message_names_another_key_than_the_workers_should_refuse_it():
    # Arrange: P05 is signed under test-wk2hub-k2; this worker signs with test-wk2hub-k1.
    built = _build_progress(_fixture("fixtures/progress/P05-counters-and-unknown-ext-member.json"))
    context = worker_sender_context(
        "transcode.progress",
        ExecutionIdentity(**built["identity"]),
        settings=_settings("test-wk2hub-k1"),
        now=NOW,
        last_event_seq=LAST_EVENT_SEQ,
    )

    # Act
    verdict = pipeline.validate(encode(built), context)

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (Layer.SEMANTIC, ReasonCode.UNKNOWN_KEY_ID, "/key_id")


@pytest.mark.parametrize(("path", "build", "channel"), POSITIVE_CASES, ids=CASE_IDS)
def test_builder_output_when_encoded_should_hold_no_key_material(path, build, channel):
    # Arrange
    settings = _settings()
    key_hexes = [*TEST_KEYS.values(), CATALOG_MEDIA_KEY_HEX, settings.worker_to_hub_key.hex()]

    # Act
    encoded = encode(build(_fixture(path))).decode("ascii")

    # Assert
    assert [key for key in key_hexes if key in encoded] == []


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.__name__)
def test_builder_when_its_parameters_are_listed_should_take_no_key(builder):
    # Act
    parameters = inspect.signature(builder).parameters

    # Assert
    assert [name for name in parameters if "key" in name and name != "key_id"] == []
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())


# --- sealing -----------------------------------------------------------------------------------


def test_seal_when_a_built_message_is_signed_should_verify_under_the_workers_key_only():
    # Arrange
    settings = _settings()
    encoded = encode(_build_claim(_fixture("fixtures/claim/P23-claim-of-the-accepted-dispatch.json")))

    # Act
    value = seal(encoded, settings.worker_to_hub_key)

    # Assert
    assert signing.verify(encoded, settings.worker_to_hub_key, value)
    assert not signing.verify(encoded + b" ", settings.worker_to_hub_key, value)
    assert not signing.verify(encoded, bytes.fromhex(TEST_KEYS["test-hub2wk-k1"]), value)


@pytest.mark.parametrize(
    "vector",
    [vector for vector in VECTORS["vectors"] if vector["category"] == "valid"],
    ids=lambda vector: vector["id"],
)
def test_seal_when_given_a_valid_signing_vector_should_produce_its_value(vector):
    # Arrange
    message = files.read_bytes(vector["message_file"])

    # Act
    value = seal(message, bytes.fromhex(TEST_KEYS[vector["key_id"]]))

    # Assert
    assert value == vector["value"]


# --- encode and timestamp ----------------------------------------------------------------------


def _padded_message(total_bytes: int) -> dict[str, Any]:
    """A progress-kind mapping whose compact encoding is exactly `total_bytes` long."""
    message: dict[str, Any] = {"message_kind": "transcode.progress", "pad": ""}
    overhead = len(json.dumps(message, separators=(",", ":")))
    message["pad"] = "x" * (total_bytes - overhead)
    return message


def test_encode_when_the_message_is_exactly_at_its_kinds_limit_should_return_it():
    # Arrange
    limit = load_registry().max_bytes("transcode.progress")

    # Act
    encoded = encode(_padded_message(limit))

    # Assert
    assert len(encoded) == limit


def test_encode_when_the_message_is_one_byte_above_its_kinds_limit_should_refuse():
    # Arrange
    limit = load_registry().max_bytes("transcode.progress")

    # Act / Assert
    with pytest.raises(MessageTooLarge) as raised:
        encode(_padded_message(limit + 1))
    assert (raised.value.size, raised.value.limit) == (limit + 1, limit)


def test_encode_when_a_string_is_not_ascii_should_escape_it():
    # Act
    encoded = encode({"message_kind": "transcode.progress", "message": "café"})

    # Assert
    assert encoded == b'{"message_kind":"transcode.progress","message":"caf\\u00e9"}'


@pytest.mark.parametrize(
    "message",
    [
        {"message_kind": "generation.manifest"},
        {"message_kind": "transcode.unknown"},
        {"identity": {}},
        {"message_kind": "transcode.progress", "stage_progress_pct": 10.0},
    ],
    ids=["a-manifest", "an-unknown-kind", "no-kind", "a-float"],
)
def test_encode_when_the_mapping_is_not_an_encodable_message_should_refuse(message):
    # Act / Assert
    with pytest.raises(ValueError):
        encode(message)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2029, 12, 31, 23, 59, 55, tzinfo=timezone.utc), "2029-12-31T23:59:55.000Z"),
        (datetime(2030, 1, 1, 7, 0, 0, 123999, tzinfo=timezone(timedelta(hours=7))), "2030-01-01T00:00:00.123Z"),
        (datetime(999, 1, 2, 3, 4, 5, tzinfo=timezone.utc), "0999-01-02T03:04:05.000Z"),
    ],
    ids=["utc", "another-zone-and-sub-millisecond-digits", "a-year-below-1000"],
)
def test_timestamp_when_given_an_aware_datetime_should_write_the_contract_form_in_utc(moment, expected):
    # Act
    text = timestamp(moment)

    # Assert
    assert text == expected


def test_timestamp_when_given_a_naive_datetime_should_refuse():
    # Act / Assert
    with pytest.raises(ValueError):
        timestamp(datetime(2030, 1, 1))


# --- wiring bugs are refused -------------------------------------------------------------------


def _claim_values(**changes: Any) -> dict[str, Any]:
    fixture = _fixture("fixtures/claim/P23-claim-of-the-accepted-dispatch.json")
    values = {
        **_envelope(fixture),
        "identity": DispatchIdentity(**fixture["identity"]),
        "bootstrap_token": fixture["claim"]["bootstrap_token"],
        "runtime_id": fixture["claim"]["runtime_id"],
    }
    values.update(changes)
    return values


IDENTITY = _fixture("fixtures/progress/P04-downloading.json")["identity"]
DISPATCH_FIELDS = {name: value for name, value in IDENTITY.items() if name not in ("execution_id", "execution_fence")}


@pytest.mark.parametrize(
    "changes",
    [
        {"org_uuid": "AAAAAAAA-AAAA-7AAA-8AAA-AAAAAAAAAAAA"},
        {"placement_epoch": 0},
        {"placement_epoch": 7.0},
        {"placement_epoch": True},
        {"lifecycle_revision": -1},
        {"owning_cell_id": "Cell-A"},
    ],
    ids=["uuid-upper-case", "epoch-zero", "epoch-float", "epoch-bool", "revision-negative", "cell-upper-case"],
)
def test_dispatch_identity_when_a_field_breaks_its_definition_should_refuse(changes):
    # Act / Assert
    with pytest.raises(ValueError):
        DispatchIdentity(**{**DISPATCH_FIELDS, **changes})


def test_execution_identity_when_made_of_a_dispatch_and_its_grant_should_equal_the_full_identity():
    # Act
    identity = ExecutionIdentity.of(
        DispatchIdentity(**DISPATCH_FIELDS),
        execution_id=IDENTITY["execution_id"],
        execution_fence=IDENTITY["execution_fence"],
    )

    # Assert
    assert identity.to_dict() == IDENTITY


def test_encryption_when_aes_128_lacks_its_media_key_id_should_refuse():
    # Act / Assert
    with pytest.raises(ValueError):
        Encryption(mode="aes-128")


@pytest.mark.parametrize(
    ("builder", "values", "error"),
    [
        (messages.claim, lambda: _claim_values(identity=ExecutionIdentity(**IDENTITY)), TypeError),
        (messages.claim, lambda: _claim_values(sent_at=datetime(2029, 12, 31, 23, 59, 55)), ValueError),
        (messages.claim, lambda: _claim_values(runtime_id="not-a-uuid"), ValueError),
        (messages.claim, lambda: _claim_values(protocol_version="two"), ValueError),
    ],
    ids=["claim-with-an-execution-identity", "naive-sent-at", "runtime-id-not-a-uuid", "version-not-a-version"],
)
def test_builder_when_a_value_is_a_wiring_bug_should_refuse(builder: Callable[..., Any], values, error):
    # Act / Assert
    with pytest.raises(error):
        builder(**values())


def test_claim_when_the_bootstrap_token_is_malformed_should_refuse_without_showing_it():
    # Arrange
    token = "SECRETbootstrapCanary" + "0" * 21  # 42 characters, one short

    # Act / Assert
    with pytest.raises(ValueError) as raised:
        messages.claim(**_claim_values(bootstrap_token=token))
    assert "/claim/bootstrap_token" in str(raised.value)
    assert token not in str(raised.value)


def test_progress_when_the_event_sequence_is_zero_should_refuse_naming_it():
    # Arrange
    fixture = _fixture("fixtures/progress/P04-downloading.json")

    # Act / Assert
    with pytest.raises(ValueError, match="/event_seq"):
        _build_progress({**fixture, "event_seq": 0})


@pytest.mark.parametrize(
    ("cause", "hub_error_code"),
    [("refused", None), ("unreachable", "claim_conflict")],
    ids=["refused-without-code", "unreachable-with-code"],
)
def test_unclaimed_when_the_cause_and_hub_error_code_disagree_should_refuse(cause, hub_error_code):
    # Arrange
    fixture = _fixture("fixtures/unclaimed/P25-refused-with-claim-conflict.json")

    # Act / Assert
    with pytest.raises(ValueError):
        messages.unclaimed(
            **_envelope(fixture),
            identity=DispatchIdentity(**fixture["identity"]),
            runtime_id=fixture["runtime_id"],
            cause=cause,
            hub_error_code=hub_error_code,
        )


# The two builders that take diagnostics, each with a fixture for its other values.
RESULT_BUILDS = [
    (_build_completed, "fixtures/result-completed/P08-thumbnail-unavailable-diagnostic.json"),
    (_build_failed, "fixtures/result-failed/P10-source-unavailable-retryable.json"),
]
RESULT_IDS = ["completed", "failed"]
DIAGNOSTIC_ENTRIES = [
    {"code": "thumbnail_unavailable", "detail": "No frame could be extracted for the thumbnail."},
    {"code": "audio_track_skipped", "detail": "The second audio track was not encoded."},
]
NOT_A_SEQUENCE = "diagnostics must be a sequence of Diagnostic"


@pytest.mark.parametrize(("build", "path"), RESULT_BUILDS, ids=RESULT_IDS)
@pytest.mark.parametrize(
    "not_a_sequence",
    [
        lambda diagnostics: (diagnostic for diagnostic in diagnostics),
        lambda diagnostics: diagnostics[0],
        lambda diagnostics: diagnostics[0].code,
    ],
    ids=["a-generator", "a-single-diagnostic", "a-string"],
)
def test_result_builder_when_the_diagnostics_are_not_a_sequence_should_refuse(build, path, not_a_sequence):
    # Arrange: valid diagnostics, handed over in a form that cannot be read twice or is not a list of them.
    diagnostics = not_a_sequence([Diagnostic(**entry) for entry in DIAGNOSTIC_ENTRIES])

    # Act / Assert
    with pytest.raises(TypeError, match=NOT_A_SEQUENCE):
        build(_fixture(path), diagnostics=diagnostics)


@pytest.mark.parametrize(("build", "path"), RESULT_BUILDS, ids=RESULT_IDS)
def test_result_builder_when_the_diagnostics_are_a_list_or_a_tuple_should_build_the_same_message(build, path):
    # Arrange
    fixture = _fixture(path)
    diagnostics = [Diagnostic(**entry) for entry in DIAGNOSTIC_ENTRIES]

    # Act
    from_list = build(fixture, diagnostics=diagnostics)
    from_tuple = build(fixture, diagnostics=tuple(diagnostics))

    # Assert
    assert from_list["diagnostics"] == DIAGNOSTIC_ENTRIES
    assert encode(from_tuple) == encode(from_list)


# --- the contexts ------------------------------------------------------------------------------


GRANT_CONTEXT = files.read_json("fixtures/contexts/ctx-worker-claim-grant.json")


@pytest.mark.parametrize(
    ("context_file", "build"),
    [
        ("fixtures/contexts/ctx-worker-dispatch.json", lambda: receiver_context("transcode.request", settings=_settings(), now=NOW)),
        ("fixtures/contexts/ctx-worker-error.json", lambda: receiver_context("hub.error", settings=_settings())),
        (
            "fixtures/contexts/ctx-worker-claim-grant.json",
            lambda: receiver_context(
                "transcode.claim.granted",
                settings=_settings(),
                now=NOW,
                dispatch=DispatchIdentity(**DISPATCH_FIELDS),
                output_location_id=GRANT_CONTEXT["expect"]["output_location_id"],
                runtime_id=GRANT_CONTEXT["expect"]["runtime_id"],
                encryption=Encryption(**GRANT_CONTEXT["expect"]["encryption"]),
            ),
        ),
        (
            "fixtures/contexts/ctx-worker-claim-grant-unencrypted.json",
            lambda: receiver_context(
                "transcode.claim.granted",
                settings=_settings(),
                now=NOW,
                dispatch=DispatchIdentity(**DISPATCH_FIELDS),
                output_location_id=GRANT_CONTEXT["expect"]["output_location_id"],
                runtime_id=GRANT_CONTEXT["expect"]["runtime_id"],
                encryption=Encryption(mode="none"),
            ),
        ),
    ],
    ids=["request", "hub-error", "claim-grant", "claim-grant-unencrypted"],
)
def test_receiver_context_when_built_from_the_settings_should_equal_the_catalogs_worker_context(context_file, build):
    # Act
    built = build()

    # Assert
    assert built == TrustedContext.from_dict(files.read_json(context_file))


@pytest.mark.parametrize(
    "payload",
    [
        "fixtures/request/P01-encrypted-three-renditions.json",
        "fixtures/request/P02-unencrypted.json",
    ],
)
def test_receiver_context_for_a_request_when_validating_a_positive_request_should_accept_it(payload):
    # Act
    verdict = pipeline.validate(files.read_bytes(payload), receiver_context("transcode.request", settings=_settings(), now=NOW))

    # Assert
    assert verdict.accepted, verdict


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: receiver_context("transcode.request", settings=_settings()), "now"),
        (
            lambda: receiver_context(
                "transcode.claim.granted",
                settings=_settings(),
                now=NOW,
                dispatch=DispatchIdentity(**DISPATCH_FIELDS),
                output_location_id=GRANT_CONTEXT["expect"]["output_location_id"],
                encryption=Encryption(mode="none"),
            ),
            "expect.runtime_id",
        ),
        (
            lambda: worker_sender_context(
                "transcode.progress", ExecutionIdentity(**IDENTITY), settings=_settings(), now=NOW
            ),
            "last_event_seq",
        ),
        (
            lambda: worker_sender_context(
                "transcode.result.completed", ExecutionIdentity(**IDENTITY), settings=_settings(), now=NOW
            ),
            "expect.source_id",
        ),
    ],
    ids=["request-without-now", "grant-without-runtime-id", "progress-without-last-event-seq", "completed-without-source"],
)
def test_context_factory_when_a_required_value_is_missing_should_raise_context_incomplete(build, field):
    # Act / Assert
    with pytest.raises(ContextIncomplete) as raised:
        build()
    assert raised.value.field == field


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda: worker_sender_context("transcode.claim", ExecutionIdentity(**IDENTITY), settings=_settings(), now=NOW), TypeError),
        (lambda: worker_sender_context("transcode.request", DispatchIdentity(**DISPATCH_FIELDS), settings=_settings(), now=NOW), ValueError),
        (lambda: receiver_context("transcode.progress", settings=_settings(), now=NOW), ValueError),
        (lambda: worker_sender_context("transcode.claim", DispatchIdentity(**DISPATCH_FIELDS), settings=None, now=NOW), TypeError),
    ],
    ids=["claim-with-an-execution-identity", "a-kind-the-worker-does-not-send", "a-kind-the-worker-does-not-receive", "no-settings"],
)
def test_context_factory_when_called_with_the_wrong_kind_or_type_should_refuse(build, error):
    # Act / Assert
    with pytest.raises(error):
        build()
