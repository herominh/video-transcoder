"""What the Hub sent is acted on only when its exact bytes verify, validate and name the key that verified them.

accept_signed() refuses in that order (signature, validation, key_id), so nothing of a body is parsed
before a configured Hub-to-worker key verified it, and a refusal's text names the step and the verdict's
layer and reason only. accept_hub_error() validates an unsigned `hub.error` and gives its `error` object,
or None for anything else. The fixtures carry no signature: each test signs the bytes it sends.
"""

import json
from typing import Any, Mapping

import pytest

from core.protocol import files, gate, pipeline, signing
from core.protocol.context import TrustedContext
from core.protocol.inbound import NotAccepted, accept_hub_error, accept_signed
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import MessageKind, load_registry
from tests.contract.v2_support import (
    GRANT_ENCRYPTED,
    HUB_ERROR_FIXTURES,
    HUB_KEY,
    HUB_KEY_ID,
    REQUEST_ENCRYPTED,
    SECOND_HUB_KEY,
    SECOND_HUB_KEY_ID,
    UNCONFIGURED_KEY,
    WORKER_DISPATCH_CONTEXT,
    WORKER_ERROR_CONTEXT,
    WORKER_GRANT_CONTEXT,
    catalog_context,
    compact,
    hub_error_body,
    modified_fixture,
    padded_to,
    seconds_from_now,
    sign_message,
)

STEP_SIGNATURE = "signature"
STEP_VALIDATION = "validation"
STEP_KEY_ID = "key_id"
ONE_KEY: Mapping[str, bytes] = {HUB_KEY_ID: HUB_KEY}
TWO_KEYS: Mapping[str, bytes] = {HUB_KEY_ID: HUB_KEY, SECOND_HUB_KEY_ID: SECOND_HUB_KEY}
NOT_JSON = b"this is not JSON at all"
# A value no refusal may show, placed in the body's object key.
CANARY = "CANARY-inbound-object-key"
CANARY_OBJECT_KEY = f"test/sources/{CANARY}.mp4"
OTHER_WORKER_AUDIENCE = "worker:another-worker"
UNKNOWN_HUB_ERROR_CODE = "made_up_refusal"
SIGNATURE_SCHEME = "hmac-sha256="
OTHER_SCHEME = "hmac-sha512="
HUB_ERROR_MAX_BYTES = load_registry().max_bytes(MessageKind.HUB_ERROR)
EXPIRED_BEYOND_THE_SKEW_S = -301
EXPIRY_POINTER = "/claim/expires_at"
RETRYABLE_CODE = "service_unavailable"
RETRY_AFTER_S = 30


def _dispatch_context(**changes: Any) -> TrustedContext:
    return catalog_context(WORKER_DISPATCH_CONTEXT, **changes)


def _refusal(raw: bytes, signature: object, *, keys: Mapping[str, bytes] = ONE_KEY, **context: Any) -> NotAccepted:
    with pytest.raises(NotAccepted) as raised:
        accept_signed(raw, signature, context=_dispatch_context(**context), keys=keys)
    return raised.value


# --- accepted -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "context"),
    [(REQUEST_ENCRYPTED, WORKER_DISPATCH_CONTEXT), (GRANT_ENCRYPTED, WORKER_GRANT_CONTEXT)],
    ids=["a-request", "a-claim-grant"],
)
def test_accept_signed_when_the_bytes_verify_validate_and_name_the_key_should_return_the_decoded_message(
    fixture, context
):
    # Arrange
    raw = files.read_bytes(fixture)
    signature = signing.sign(raw, HUB_KEY)

    # Act
    message = accept_signed(raw, signature, context=catalog_context(context), keys=ONE_KEY)

    # Assert
    assert message == json.loads(raw)
    assert message["key_id"] == HUB_KEY_ID


def test_accept_signed_when_the_second_configured_key_signed_a_body_that_names_it_should_accept():
    # Arrange: the control of the key binding test below: naming the key that signed is accepted.
    raw = modified_fixture(REQUEST_ENCRYPTED, {"/key_id": SECOND_HUB_KEY_ID})
    signature = signing.sign(raw, SECOND_HUB_KEY)

    # Act
    message = accept_signed(
        raw, signature, context=_dispatch_context(known_key_ids=[HUB_KEY_ID, SECOND_HUB_KEY_ID]), keys=TWO_KEYS
    )

    # Assert
    assert message["key_id"] == SECOND_HUB_KEY_ID


# --- the signature comes first ------------------------------------------------------------------


def _valid_signature() -> str:
    return signing.sign(files.read_bytes(REQUEST_ENCRYPTED), HUB_KEY)


@pytest.mark.parametrize(
    "signature",
    [
        pytest.param(lambda: signing.sign(files.read_bytes(REQUEST_ENCRYPTED) + b" ", HUB_KEY), id="of-other-bytes"),
        pytest.param(
            lambda: signing.sign(files.read_bytes(REQUEST_ENCRYPTED), UNCONFIGURED_KEY), id="of-an-unconfigured-key"
        ),
        pytest.param(lambda: _valid_signature().replace(SIGNATURE_SCHEME, OTHER_SCHEME), id="wrong-prefix"),
        pytest.param(lambda: SIGNATURE_SCHEME + _valid_signature()[len(SIGNATURE_SCHEME):].upper(), id="upper-case-hex"),
        pytest.param(lambda: _valid_signature()[:-1], id="one-digit-short"),
        pytest.param(lambda: _valid_signature() + "0", id="one-digit-long"),
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: _valid_signature().encode("ascii"), id="bytes-not-a-string"),
        pytest.param(lambda: 0, id="a-number"),
    ],
)
def test_accept_signed_when_no_configured_key_verifies_the_bytes_should_refuse_at_the_signature(signature):
    # Arrange: the body itself is the acceptable request.
    raw = files.read_bytes(REQUEST_ENCRYPTED)

    # Act
    refusal = _refusal(raw, signature())

    # Assert
    assert refusal.step == STEP_SIGNATURE
    assert refusal.verdict is None


def test_accept_signed_when_the_signature_is_wrong_and_the_body_is_not_json_should_refuse_before_validating():
    # Arrange: correctly signed, these bytes are refused by the validator (see the validation test below).
    signature = signing.sign(NOT_JSON, UNCONFIGURED_KEY)

    # Act
    refusal = _refusal(NOT_JSON, signature)

    # Assert
    assert refusal.step == STEP_SIGNATURE


def _fail_when_the_body_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the validator and its parser fail the test the moment anything is handed to them."""

    def reached(*args: object, **kwargs: object) -> None:
        raise AssertionError("the body was read before a signature verified")

    monkeypatch.setattr(pipeline, "validate", reached)
    monkeypatch.setattr(gate, "parse", reached)


def test_accept_signed_when_no_configured_key_verifies_the_bytes_should_refuse_without_reading_the_body(monkeypatch):
    # Arrange: the body is the acceptable request, signed by a key this worker does not hold.
    raw = files.read_bytes(REQUEST_ENCRYPTED)
    _fail_when_the_body_is_read(monkeypatch)

    # Act
    refusal = _refusal(raw, signing.sign(raw, UNCONFIGURED_KEY))

    # Assert
    assert refusal.step == STEP_SIGNATURE


# --- then the validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "layer", "reason"),
    [
        pytest.param(
            modified_fixture(REQUEST_ENCRYPTED, {"/audience": OTHER_WORKER_AUDIENCE}),
            Layer.SEMANTIC,
            ReasonCode.AUDIENCE_MISMATCH,
            id="another-workers-audience",
        ),
        pytest.param(NOT_JSON, Layer.PARSE, ReasonCode.MALFORMED_JSON, id="not-json"),
    ],
)
def test_accept_signed_when_a_correctly_signed_message_fails_validation_should_refuse_with_the_verdict(
    raw, layer, reason
):
    # Arrange
    signature = signing.sign(raw, HUB_KEY)

    # Act
    refusal = _refusal(raw, signature)

    # Assert
    assert refusal.step == STEP_VALIDATION
    assert (refusal.verdict.layer, refusal.verdict.reason) == (layer, reason)


# --- then the key the body names ----------------------------------------------------------------


def test_accept_signed_when_the_body_names_one_key_and_another_configured_key_signed_it_should_refuse_at_the_key_id():
    # Arrange: the body names the first key; the second key's bytes signed it.
    raw = files.read_bytes(REQUEST_ENCRYPTED)
    signature = signing.sign(raw, SECOND_HUB_KEY)
    context = _dispatch_context(known_key_ids=[HUB_KEY_ID, SECOND_HUB_KEY_ID])
    assert json.loads(raw)["key_id"] == HUB_KEY_ID
    assert signing.verify(raw, TWO_KEYS[SECOND_HUB_KEY_ID], signature)
    assert pipeline.validate(raw, context).accepted

    # Act
    with pytest.raises(NotAccepted) as raised:
        accept_signed(raw, signature, context=context, keys=TWO_KEYS)

    # Assert
    assert raised.value.step == STEP_KEY_ID


# --- a refusal shows the step and the verdict's layer and reason only ---------------------------


def _canary_request(**changes: Any) -> bytes:
    return modified_fixture(REQUEST_ENCRYPTED, {"/source/object_key": CANARY_OBJECT_KEY, **changes})


def _expired_canary_request() -> bytes:
    return _canary_request(**{"/claim/expires_at": seconds_from_now(EXPIRED_BEYOND_THE_SKEW_S)})


@pytest.mark.parametrize(
    ("raw", "signing_key", "step"),
    [
        pytest.param(_canary_request(), UNCONFIGURED_KEY, STEP_SIGNATURE, id="signature"),
        pytest.param(_expired_canary_request(), HUB_KEY, STEP_VALIDATION, id="validation"),
        pytest.param(_canary_request(), SECOND_HUB_KEY, STEP_KEY_ID, id="key-id"),
    ],
)
def test_not_accepted_when_turned_into_text_should_name_its_step_and_show_nothing_of_the_message(
    raw, signing_key, step
):
    # Arrange
    signature = signing.sign(raw, signing_key)
    hidden = [CANARY, raw.decode("ascii"), signature, signature[len(SIGNATURE_SCHEME):]]
    hidden += [key.hex() for key in TWO_KEYS.values()]

    # Act
    refusal = _refusal(raw, signature, keys=TWO_KEYS, known_key_ids=[HUB_KEY_ID, SECOND_HUB_KEY_ID])

    # Assert
    assert refusal.step == step
    assert step in str(refusal)
    assert [value for value in hidden if value in str(refusal) or value in repr(refusal)] == []


def test_not_accepted_when_it_holds_a_verdict_should_show_its_layer_and_reason_but_not_its_pointer():
    # Arrange
    raw = _expired_canary_request()

    # Act
    refusal = _refusal(raw, signing.sign(raw, HUB_KEY))

    # Assert
    assert refusal.verdict.instance_path == EXPIRY_POINTER
    assert f"{Layer.SEMANTIC.value}/{ReasonCode.DISPATCH_EXPIRED.value}" in str(refusal)
    assert EXPIRY_POINTER not in str(refusal)


# --- hub.error ----------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", HUB_ERROR_FIXTURES, ids=["P12", "P13"])
def test_accept_hub_error_when_the_body_is_an_accepted_hub_error_should_return_its_error_object(fixture):
    # Arrange
    raw = files.read_bytes(fixture)

    # Act
    error = accept_hub_error(raw, context=catalog_context(WORKER_ERROR_CONTEXT))

    # Assert
    assert error == json.loads(raw)["error"]
    assert error["retryable"] is False


def test_accept_hub_error_when_the_body_is_exactly_at_the_byte_limit_should_return_its_error_object():
    # Arrange
    raw = padded_to(hub_error_body(RETRYABLE_CODE, retryable=True, retry_after_s=RETRY_AFTER_S), HUB_ERROR_MAX_BYTES)

    # Act
    error = accept_hub_error(raw, context=catalog_context(WORKER_ERROR_CONTEXT))

    # Assert
    assert (error["code"], error["retryable"], error["retry_after_s"]) == (RETRYABLE_CODE, True, RETRY_AFTER_S)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(NOT_JSON, id="not-json"),
        pytest.param(hub_error_body(UNKNOWN_HUB_ERROR_CODE, retryable=True), id="an-unknown-code"),
        pytest.param(files.read_bytes(GRANT_ENCRYPTED), id="a-claim-grant"),
        pytest.param(b"", id="empty"),
        pytest.param(
            padded_to(hub_error_body(RETRYABLE_CODE, retryable=True), HUB_ERROR_MAX_BYTES + 1),
            id="one-byte-above-the-limit",
        ),
        pytest.param(
            compact({"error": {"code": RETRYABLE_CODE, "message": "No envelope around it.", "retryable": True}}),
            id="an-error-object-without-its-envelope",
        ),
    ],
)
def test_accept_hub_error_when_the_body_is_anything_else_should_return_none(raw):
    # Act
    error = accept_hub_error(raw, context=catalog_context(WORKER_ERROR_CONTEXT))

    # Assert
    assert error is None


# --- only the exact bytes -----------------------------------------------------------------------


def test_accept_signed_when_given_text_instead_of_bytes_should_raise_type_error():
    # Arrange
    signed = sign_message(files.read_bytes(REQUEST_ENCRYPTED), HUB_KEY)

    # Act / Assert
    with pytest.raises(TypeError):
        accept_signed(signed.body.decode("ascii"), signed.signature, context=_dispatch_context(), keys=ONE_KEY)


def test_accept_hub_error_when_given_text_instead_of_bytes_should_raise_type_error():
    # Arrange
    text = files.read_bytes(HUB_ERROR_FIXTURES[0]).decode("ascii")

    # Act / Assert
    with pytest.raises(TypeError):
        accept_hub_error(text, context=catalog_context(WORKER_ERROR_CONTEXT))
