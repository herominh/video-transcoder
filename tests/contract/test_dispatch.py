"""A RunPod job input is a dispatch only when it is the carrier, verifies and is an accepted request for this worker.

accept_dispatch() returns the request as a typed, immutable value, or raises DispatchRefused naming the
step (carrier, signature, validation) and the verdict's layer and reason: never the content of what it
refused. The bootstrap token of an accepted dispatch stays out of its repr. The fixtures carry no
signature: each test signs the exact bytes it puts in the carrier.
"""

import dataclasses
import json
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable

import pytest

from core.protocol import files, gate, pipeline, signing
from core.protocol.dispatch import AcceptedDispatch, DispatchRefused, accept_dispatch
from core.protocol.reasons import Layer, ReasonCode
from tests.contract.v2_support import (
    FIXTURE_NOW,
    GRANT_ENCRYPTED,
    HUB_KEY,
    HUB_KEY_ID,
    REMOVED,
    REQUEST_ENCRYPTED,
    REQUEST_UNENCRYPTED,
    SECOND_HUB_KEY,
    UNCLAIMED_REFUSED,
    UNCONFIGURED_KEY,
    modified_fixture,
    moment,
    padded_to,
    seconds_from_now,
    signed_carrier,
    two_hub_keys,
    worker_settings,
)

SETTINGS = worker_settings()
SKEW_S = SETTINGS.clock_skew_s
STEP_CARRIER = "carrier"
STEP_SIGNATURE = "signature"
STEP_VALIDATION = "validation"
MESSAGE_MEMBER = "vh_message"
SIGNATURE_MEMBER = "vh_signature"
# The carrier bounds the message text (carriers/runpod-input.schema.json, README section 12).
CARRIER_MESSAGE_MAX_CHARS = 4096
SIGNATURE_SCHEME = "hmac-sha256="
OTHER_SCHEME = "hmac-sha512="
NON_ASCII_CHARACTER = "\u00e9"
LONE_SURROGATE = "\ud800"
ONE_DAY = timedelta(days=1)
REQUEST_WITH_BOUNDED_EXPIRY = "fixtures/request/P20-claim-expired-exactly-300s-ago.json"
OTHER_WORKER_AUDIENCE = "worker:another-worker"
UNHELD_KEY_ID = "test-hub2wk-k9"
RELEASE_VERSION_NOT_LISTED = "2.0.0"
MILLISECOND_EXPIRY = "2030-01-01T00:50:00.123Z"
MILLISECOND_EXPIRY_MICROSECONDS = 123_000
# A value no refusal may show, placed in the request's object key.
CANARY = "CANARY-dispatch-object-key"
CANARY_OBJECT_KEY = f"test/sources/{CANARY}.mp4"
EXPIRY_POINTER = "/claim/expires_at"
# A piece of the fixtures' object key, replaced to put one other character in the message text.
OBJECT_KEY_PIECE = "source-0001"
STEP_KEY_ID = "key_id"


def _accept(job_input: object, now: datetime = FIXTURE_NOW) -> AcceptedDispatch:
    return accept_dispatch(job_input, settings=SETTINGS, now=now)


def _refusal(job_input: object, now: datetime = FIXTURE_NOW) -> DispatchRefused:
    with pytest.raises(DispatchRefused) as raised:
        _accept(job_input, now)
    return raised.value


def _request() -> bytes:
    return files.read_bytes(REQUEST_ENCRYPTED)


def _carrier(**changes: Any) -> dict[str, Any]:
    """The acceptable carrier of the encrypted request, with some members replaced."""
    return {**signed_carrier(_request(), HUB_KEY), **changes}


def _carrier_of_text(text: str) -> dict[str, str]:
    """A carrier whose signature is the Hub's over the text's own bytes, whatever characters it holds."""
    return {
        MESSAGE_MEMBER: text,
        SIGNATURE_MEMBER: signing.sign(text.encode("utf-8", errors="surrogatepass"), HUB_KEY),
    }


# --- accepted -----------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [REQUEST_ENCRYPTED, REQUEST_UNENCRYPTED], ids=["P01-encrypted", "P02-unencrypted"])
def test_accept_dispatch_when_a_signed_fixture_arrives_at_the_fixtures_now_should_return_every_field_typed(path):
    # Arrange
    raw = files.read_bytes(path)
    fixture = json.loads(raw)

    # Act
    dispatch = _accept(signed_carrier(raw, HUB_KEY))

    # Assert
    assert dispatch.protocol_version == fixture["protocol_version"]
    assert dataclasses.asdict(dispatch.identity) == fixture["identity"]
    assert dataclasses.asdict(dispatch.source) == fixture["source"]
    assert dataclasses.asdict(dispatch.output) == fixture["output"]
    assert dataclasses.asdict(dispatch.profile) == fixture["profile"]
    assert dispatch.renditions == tuple(fixture["renditions"])
    assert isinstance(dispatch.renditions, tuple)
    assert dispatch.encryption.mode == fixture["encryption"]["mode"]
    assert dispatch.encryption.media_key_id == fixture["encryption"].get("media_key_id")
    assert dataclasses.asdict(dispatch.limits) == fixture["limits"]
    assert dispatch.bootstrap_token == fixture["claim"]["bootstrap_token"]
    assert dispatch.claim_expires_at == moment(fixture["claim"]["expires_at"])
    assert dispatch.claim_expires_at.utcoffset() == timedelta(0)


def test_accept_dispatch_when_the_request_is_unencrypted_should_hold_no_media_key_id():
    # Act
    encrypted = _accept(signed_carrier(files.read_bytes(REQUEST_ENCRYPTED), HUB_KEY))
    unencrypted = _accept(signed_carrier(files.read_bytes(REQUEST_UNENCRYPTED), HUB_KEY))

    # Assert
    assert encrypted.encryption.media_key_id is not None
    assert unencrypted.encryption.media_key_id is None


def test_accept_dispatch_when_the_claim_expiry_has_milliseconds_should_keep_them():
    # Arrange
    raw = modified_fixture(REQUEST_ENCRYPTED, {"/claim/expires_at": MILLISECOND_EXPIRY})

    # Act
    dispatch = _accept(signed_carrier(raw, HUB_KEY))

    # Assert
    assert dispatch.claim_expires_at == moment(MILLISECOND_EXPIRY)
    assert dispatch.claim_expires_at.microsecond == MILLISECOND_EXPIRY_MICROSECONDS


def test_accept_dispatch_when_the_message_text_is_exactly_at_the_carrier_bound_should_accept_it():
    # Arrange: trailing spaces are insignificant whitespace, and the request's own byte limit is the same number.
    raw = padded_to(_request(), CARRIER_MESSAGE_MAX_CHARS)

    # Act
    dispatch = _accept(signed_carrier(raw, HUB_KEY))

    # Assert
    assert dispatch.identity.dispatch_id == json.loads(raw)["identity"]["dispatch_id"]


def test_accepted_dispatch_when_shown_with_repr_should_not_show_the_bootstrap_token():
    # Arrange
    dispatch = _accept(_carrier())

    # Act
    shown = repr(dispatch)

    # Assert
    assert dispatch.bootstrap_token == json.loads(_request())["claim"]["bootstrap_token"]
    assert dispatch.bootstrap_token not in shown
    assert dispatch.identity.dispatch_id in shown


# --- the carrier --------------------------------------------------------------------------------


def _signature() -> str:
    return _carrier()[SIGNATURE_MEMBER]


NOT_THE_CARRIER: list[Any] = [
    pytest.param(lambda: None, id="none"),
    pytest.param(lambda: [_carrier()[MESSAGE_MEMBER], _signature()], id="a-list"),
    pytest.param(lambda: _carrier()[MESSAGE_MEMBER], id="the-message-text-alone"),
    pytest.param(lambda: {MESSAGE_MEMBER: _carrier()[MESSAGE_MEMBER]}, id="no-signature-member"),
    pytest.param(lambda: {SIGNATURE_MEMBER: _signature()}, id="no-message-member"),
    pytest.param(lambda: _carrier(vh_job_id="job-0001"), id="an-extra-member"),
    pytest.param(lambda: _carrier(vh_message=json.loads(_request())), id="the-message-as-an-object"),
    pytest.param(lambda: _carrier(vh_message=_request()), id="the-message-as-bytes"),
    pytest.param(lambda: _carrier(vh_signature=None), id="the-signature-null"),
    pytest.param(lambda: _carrier(vh_signature=_signature().replace(SIGNATURE_SCHEME, OTHER_SCHEME)), id="signature-wrong-prefix"),
    pytest.param(
        lambda: _carrier(vh_signature=SIGNATURE_SCHEME + _signature()[len(SIGNATURE_SCHEME):].upper()),
        id="signature-upper-case-hex",
    ),
    pytest.param(lambda: _carrier(vh_signature=_signature()[:-1]), id="signature-one-digit-short"),
    pytest.param(
        lambda: signed_carrier(padded_to(_request(), CARRIER_MESSAGE_MAX_CHARS + 1), HUB_KEY),
        id="message-one-character-above-the-bound",
    ),
    pytest.param(
        lambda: _carrier_of_text(_request().decode("ascii").replace(OBJECT_KEY_PIECE, OBJECT_KEY_PIECE + NON_ASCII_CHARACTER)),
        id="message-with-a-non-ascii-character",
    ),
    pytest.param(
        lambda: _carrier_of_text(_request().decode("ascii").replace(OBJECT_KEY_PIECE, OBJECT_KEY_PIECE + LONE_SURROGATE)),
        id="message-with-a-lone-surrogate",
    ),
]


@pytest.mark.parametrize("job_input", NOT_THE_CARRIER)
def test_accept_dispatch_when_the_job_input_is_not_the_carrier_should_refuse_at_the_carrier(
    job_input: Callable[[], object]
):
    # Act
    refusal = _refusal(job_input())

    # Assert
    assert refusal.step == STEP_CARRIER
    assert refusal.verdict is None


# --- the signature, then the validation ---------------------------------------------------------


def test_accept_dispatch_when_a_key_this_worker_does_not_hold_signed_it_should_refuse_at_the_signature():
    # Arrange
    job_input = signed_carrier(_request(), UNCONFIGURED_KEY)

    # Act
    refusal = _refusal(job_input)

    # Assert
    assert refusal.step == STEP_SIGNATURE


def test_accept_dispatch_when_no_configured_key_signed_it_should_refuse_without_reading_the_message(monkeypatch):
    # Arrange: a validator and a parser that fail the test the moment anything is handed to them.
    def reached(*args: object, **kwargs: object) -> None:
        raise AssertionError("the message was read before a signature verified")

    monkeypatch.setattr(pipeline, "validate", reached)
    monkeypatch.setattr(gate, "parse", reached)
    job_input = signed_carrier(_request(), UNCONFIGURED_KEY)

    # Act
    refusal = _refusal(job_input)

    # Assert
    assert refusal.step == STEP_SIGNATURE


def test_accept_dispatch_when_the_message_changed_after_signing_should_refuse_at_the_signature():
    # Arrange: re-serialized, the same request no longer has the signed bytes.
    job_input = _carrier(vh_message=modified_fixture(REQUEST_ENCRYPTED, {}).decode("ascii"))

    # Act
    refusal = _refusal(job_input)

    # Assert
    assert refusal.step == STEP_SIGNATURE


@pytest.mark.parametrize(
    ("changes", "layer", "reason"),
    [
        pytest.param({"/audience": OTHER_WORKER_AUDIENCE}, Layer.SEMANTIC, ReasonCode.AUDIENCE_MISMATCH, id="another-workers-audience"),
        pytest.param({"/key_id": UNHELD_KEY_ID}, Layer.SEMANTIC, ReasonCode.UNKNOWN_KEY_ID, id="a-key-id-the-worker-does-not-hold"),
        pytest.param(
            {"/sent_at": seconds_from_now(SKEW_S + 1)},
            Layer.SEMANTIC,
            ReasonCode.TIMESTAMP_IN_FUTURE,
            id="sent-one-second-beyond-the-skew",
        ),
        pytest.param(
            {"/claim/expires_at": seconds_from_now(-(SKEW_S + 1))},
            Layer.SEMANTIC,
            ReasonCode.DISPATCH_EXPIRED,
            id="claim-expired-one-second-beyond-the-skew",
        ),
        pytest.param(
            {"/protocol_version": RELEASE_VERSION_NOT_LISTED},
            Layer.VERSION,
            ReasonCode.UNSUPPORTED_VERSION,
            id="a-version-the-worker-does-not-accept",
        ),
        pytest.param({"/claim/bootstrap_token": REMOVED}, Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION, id="no-bootstrap-token"),
    ],
)
def test_accept_dispatch_when_a_correctly_signed_request_fails_validation_should_refuse_with_the_verdict(
    changes, layer, reason
):
    # Arrange
    job_input = signed_carrier(modified_fixture(REQUEST_ENCRYPTED, changes), HUB_KEY)

    # Act
    refusal = _refusal(job_input)

    # Assert
    assert refusal.step == STEP_VALIDATION
    assert (refusal.verdict.layer, refusal.verdict.reason) == (layer, reason)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(modified_fixture(REQUEST_ENCRYPTED, {"/sent_at": seconds_from_now(SKEW_S)}), id="sent-exactly-the-skew-ahead"),
        pytest.param(
            modified_fixture(REQUEST_ENCRYPTED, {"/claim/expires_at": seconds_from_now(-SKEW_S)}),
            id="claim-expired-exactly-the-skew-ago",
        ),
        pytest.param(files.read_bytes(REQUEST_WITH_BOUNDED_EXPIRY), id="P20-of-the-catalog"),
    ],
)
def test_accept_dispatch_when_a_request_is_exactly_at_a_freshness_bound_should_accept_it(raw):
    # Act
    dispatch = _accept(signed_carrier(raw, HUB_KEY))

    # Assert
    assert dispatch.claim_expires_at == moment(json.loads(raw)["claim"]["expires_at"])


@pytest.mark.parametrize("path", [GRANT_ENCRYPTED, UNCLAIMED_REFUSED], ids=["a-claim-grant", "a-worker-sent-report"])
def test_accept_dispatch_when_a_correctly_signed_message_is_not_a_request_should_refuse_it(path):
    # Arrange
    job_input = signed_carrier(files.read_bytes(path), HUB_KEY)

    # Act
    refusal = _refusal(job_input)

    # Assert
    assert refusal.step == STEP_VALIDATION
    assert refusal.verdict.reason == ReasonCode.UNEXPECTED_MESSAGE_KIND


def test_accept_dispatch_when_the_request_names_one_configured_key_and_another_signed_it_should_refuse_at_the_key_id():
    # Arrange: a worker holding two Hub-to-worker keys; the request names the first, the second signed it.
    settings = worker_settings(VH_HUB_TO_WORKER_KEYS=two_hub_keys())
    signed_by_the_named_key = signed_carrier(_request(), HUB_KEY)
    signed_by_the_other_key = signed_carrier(_request(), SECOND_HUB_KEY)
    assert json.loads(_request())["key_id"] == HUB_KEY_ID

    # Act
    accepted = accept_dispatch(signed_by_the_named_key, settings=settings, now=FIXTURE_NOW)
    with pytest.raises(DispatchRefused) as raised:
        accept_dispatch(signed_by_the_other_key, settings=settings, now=FIXTURE_NOW)

    # Assert
    assert accepted.identity.dispatch_id == json.loads(_request())["identity"]["dispatch_id"]
    assert raised.value.step == STEP_KEY_ID


# --- a refusal shows the step and the verdict's layer and reason only ---------------------------


def _canary_request(**changes: Any) -> bytes:
    return modified_fixture(REQUEST_ENCRYPTED, {"/source/object_key": CANARY_OBJECT_KEY, **changes})


@pytest.mark.parametrize(
    ("job_input", "step", "layer_and_reason"),
    [
        pytest.param(lambda: {**signed_carrier(_canary_request(), HUB_KEY), "vh_job_id": CANARY}, STEP_CARRIER, "", id="carrier"),
        pytest.param(lambda: signed_carrier(_canary_request(), UNCONFIGURED_KEY), STEP_SIGNATURE, "", id="signature"),
        pytest.param(
            lambda: signed_carrier(_canary_request(**{EXPIRY_POINTER: seconds_from_now(-(SKEW_S + 1))}), HUB_KEY),
            STEP_VALIDATION,
            "semantic/dispatch_expired",
            id="validation",
        ),
    ],
)
def test_dispatch_refused_when_turned_into_text_should_name_the_step_and_reason_and_show_nothing_of_the_message(
    job_input: Callable[[], dict[str, str]], step: str, layer_and_reason: str
):
    # Arrange
    carrier = job_input()
    token = json.loads(carrier[MESSAGE_MEMBER])["claim"]["bootstrap_token"]
    signature = carrier[SIGNATURE_MEMBER]
    hidden = [token, CANARY, signature, signature[len(SIGNATURE_SCHEME):], HUB_KEY.hex(), EXPIRY_POINTER]

    # Act
    refusal = _refusal(carrier)
    shown = str(refusal) + repr(refusal) + "".join(traceback.format_exception(type(refusal), refusal, refusal.__traceback__))

    # Assert
    assert refusal.step == step
    assert step in str(refusal)
    assert layer_and_reason in str(refusal)
    assert (refusal.verdict is None) == (layer_and_reason == "")
    assert [value for value in hidden if value in shown] == []


# --- the clock ----------------------------------------------------------------------------------


def test_accept_dispatch_when_the_same_signed_request_arrives_a_day_later_should_refuse_it_as_expired():
    # Arrange
    job_input = _carrier()

    # Act
    accepted = _accept(job_input, now=FIXTURE_NOW)
    refusal = _refusal(job_input, now=FIXTURE_NOW + ONE_DAY)

    # Assert
    assert accepted.identity.dispatch_id == json.loads(_request())["identity"]["dispatch_id"]
    assert refusal.step == STEP_VALIDATION
    assert (refusal.verdict.layer, refusal.verdict.reason) == (Layer.SEMANTIC, ReasonCode.DISPATCH_EXPIRED)
