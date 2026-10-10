"""What the worker sends leaves only self-checked and signed, and travels as the job output or to the callback address.

seal_message() encodes a built message, validates those exact bytes under the sender context and signs
them: a message its own validator refuses raises SelfCheckFailed and is never sealed. runpod_output() is
the job's output carrier of a result or an unclaimed report; deliver() posts the same bytes to the
configured callback address, sends again only after a transport failure or a retryable `hub.error`, and
never raises for a delivery that failed. A claim travels neither way.

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Iterator

import pytest

from core.failure import Failure
from core.protocol import files, messages, schema, signing
from core.protocol.context import TrustedContext
from core.protocol.messages import DispatchIdentity, ExecutionIdentity
from core.protocol.outbox import CALLBACK_ATTEMPTS, SealedMessage, SelfCheckFailed, deliver, runpod_output, seal_message
from core.protocol.registry import MessageKind
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.contract.v2_support import (
    CALLBACK_URL,
    FIXTURE_NOW,
    HUB_KEY,
    UNCLAIMED_REFUSED,
    WORKER_KEY,
    FakeClock,
    FakeTransport,
    compact,
    hub_error_answer,
    moment,
    worker_settings,
)

SETTINGS = worker_settings()
CLAIM = "fixtures/claim/P23-claim-of-the-accepted-dispatch.json"
PROGRESS = "fixtures/progress/P04-downloading.json"
FAILED_RESULT = "fixtures/result-failed/P11-minimal-stage-null.json"
RUNPOD_OUTPUT_SCHEMA = "carriers/runpod-output.schema.json"
# What the Hub records for the fixtures' execution (fixtures/contexts/ctx-hub-push.json).
HUB_PUSH_CONTEXT = files.read_json("fixtures/contexts/ctx-hub-push.json")
SOURCE_ID = HUB_PUSH_CONTEXT["expect"]["source_id"]
LAST_EVENT_SEQ = HUB_PUSH_CONTEXT["last_event_seq"]
OTHER_UUID = "9a9a9a9a-9a9a-79a9-89a9-9a9a9a9a9a9a"
ONE_HOUR = timedelta(hours=1)
KIND_MEMBER = "message_kind"
SIGNATURE_HEX_DIGITS = 64
HTTP_OK = 200
HTTP_NO_CONTENT = 204
HTTP_FOUND = 302
HTTP_CONFLICT = 409
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
HTML_BODY = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
NO_ANSWER = TransportFailed("ReadTimeout")
DELIVERED = HttpAnswer(HTTP_OK, b"", None)
RETRY_LATER = hub_error_answer(HTTP_UNAVAILABLE, "service_unavailable", retryable=True)


def _fixture(path: str) -> dict[str, Any]:
    return json.loads(files.read_bytes(path))


def _envelope(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": fixture["protocol_version"],
        "message_id": fixture["message_id"],
        "sent_at": moment(fixture["sent_at"]),
        "key_id": fixture["key_id"],
    }


def _unclaimed() -> dict[str, Any]:
    """The catalog's unclaimed report (P25), built by the worker's own builder."""
    fixture = _fixture(UNCLAIMED_REFUSED)
    return messages.unclaimed(
        **_envelope(fixture),
        identity=DispatchIdentity(**fixture["identity"]),
        runtime_id=fixture["runtime_id"],
        cause=fixture["cause"],
        hub_error_code=fixture["hub_error_code"],
    )


def _unclaimed_context(
    *, now: datetime = FIXTURE_NOW, identity: DispatchIdentity | None = None
) -> TrustedContext:
    recorded = DispatchIdentity(**_fixture(UNCLAIMED_REFUSED)["identity"]) if identity is None else identity
    return messages.worker_sender_context(MessageKind.TRANSCODE_UNCLAIMED, recorded, settings=SETTINGS, now=now)


def _sealed_unclaimed() -> SealedMessage:
    return seal_message(_unclaimed(), _unclaimed_context(), settings=SETTINGS)


def _sealed_failed_result() -> SealedMessage:
    fixture = _fixture(FAILED_RESULT)
    error = fixture["error"]
    identity = ExecutionIdentity(**fixture["identity"])
    message = messages.result_failed(
        **_envelope(fixture),
        identity=identity,
        failure=Failure(error_class=error["class"], code=error["code"], retryable=error["retryable"], detail=error["detail"]),
        stage=error["stage"],
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_RESULT_FAILED, identity, settings=SETTINGS, now=FIXTURE_NOW, source_id=SOURCE_ID
    )
    return seal_message(message, context, settings=SETTINGS)


def _sealed_claim() -> SealedMessage:
    fixture = _fixture(CLAIM)
    identity = DispatchIdentity(**fixture["identity"])
    message = messages.claim(
        **_envelope(fixture),
        identity=identity,
        bootstrap_token=fixture["claim"]["bootstrap_token"],
        runtime_id=fixture["claim"]["runtime_id"],
    )
    context = messages.worker_sender_context(MessageKind.TRANSCODE_CLAIM, identity, settings=SETTINGS, now=FIXTURE_NOW)
    return seal_message(message, context, settings=SETTINGS)


def _sealed_progress() -> SealedMessage:
    fixture = _fixture(PROGRESS)
    identity = ExecutionIdentity(**fixture["identity"])
    message = messages.progress(
        **_envelope(fixture),
        identity=identity,
        event_seq=fixture["event_seq"],
        stage=fixture["stage"],
        stage_progress_pct=fixture["stage_progress_pct"],
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_PROGRESS, identity, settings=SETTINGS, now=FIXTURE_NOW, last_event_seq=LAST_EVENT_SEQ
    )
    return seal_message(message, context, settings=SETTINGS)


def _text_values(document: Any) -> Iterator[str]:
    """Every string value of a message, at any depth."""
    if isinstance(document, str):
        yield document
    elif isinstance(document, dict):
        for value in document.values():
            yield from _text_values(value)


def _deliver(sealed: SealedMessage, transport: FakeTransport, **options: Any) -> bool:
    return deliver(sealed, settings=SETTINGS, transport=transport, sleep=FakeClock().sleep, **options)


# --- sealing ------------------------------------------------------------------------------------


def test_seal_message_when_the_message_passes_its_self_check_should_return_its_kind_bytes_and_signature():
    # Arrange
    message = _unclaimed()

    # Act
    sealed = seal_message(message, _unclaimed_context(), settings=SETTINGS)

    # Assert
    assert sealed.kind == MessageKind.TRANSCODE_UNCLAIMED
    assert sealed.body == messages.encode(message) == compact(_fixture(UNCLAIMED_REFUSED))
    assert signing.verify(sealed.body, WORKER_KEY, sealed.signature)
    assert not signing.verify(sealed.body, HUB_KEY, sealed.signature)


@pytest.mark.parametrize(
    ("context", "layer_and_reason"),
    [
        pytest.param(
            lambda: _unclaimed_context(now=moment(_fixture(UNCLAIMED_REFUSED)["sent_at"]) + ONE_HOUR),
            "semantic/timestamp_out_of_tolerance",
            id="sent-an-hour-before-the-contexts-now",
        ),
        pytest.param(
            lambda: _unclaimed_context(
                identity=DispatchIdentity(**{**_fixture(UNCLAIMED_REFUSED)["identity"], "dispatch_id": OTHER_UUID})
            ),
            "semantic/dispatch_mismatch",
            id="another-dispatch-on-record",
        ),
    ],
)
def test_seal_message_when_the_context_refuses_the_message_should_raise_self_check_failed_naming_only_the_reason(
    context, layer_and_reason
):
    # Arrange
    message = _unclaimed()
    values = [value for name, value in message.items() if name != KIND_MEMBER for value in _text_values(value)]

    # Act
    with pytest.raises(SelfCheckFailed) as raised:
        seal_message(message, context(), settings=SETTINGS)

    # Assert
    text = str(raised.value)
    assert layer_and_reason in text
    assert [value for value in values if value in text] == []


# --- the job output -----------------------------------------------------------------------------


@pytest.mark.parametrize("sealed", [_sealed_unclaimed, _sealed_failed_result], ids=["an-unclaimed-report", "a-result"])
def test_runpod_output_when_given_a_report_or_a_result_should_return_the_carrier_of_its_bytes_and_signature(sealed):
    # Arrange
    message = sealed()

    # Act
    output = runpod_output(message)

    # Assert
    assert output == {"vh_message": message.body.decode("ascii"), "vh_signature": message.signature}
    assert schema.first_error_pointer(RUNPOD_OUTPUT_SCHEMA, output) is None
    assert signing.verify(output["vh_message"].encode("ascii"), WORKER_KEY, output["vh_signature"])


@pytest.mark.parametrize("sealed", [_sealed_claim, _sealed_progress], ids=["a-claim", "a-progress-message"])
def test_runpod_output_when_given_a_message_that_is_never_a_job_output_should_raise_value_error(sealed):
    # Arrange
    message = sealed()

    # Act / Assert
    with pytest.raises(ValueError):
        runpod_output(message)


# --- the callback -------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [HTTP_OK, HTTP_NO_CONTENT])
@pytest.mark.parametrize("sealed", [_sealed_unclaimed, _sealed_failed_result], ids=["an-unclaimed-report", "a-result"])
def test_deliver_when_the_hub_answers_2xx_should_post_once_to_the_callback_address_and_return_true(sealed, status):
    # Arrange
    message = sealed()
    transport = FakeTransport(HttpAnswer(status, b"", None))

    # Act
    delivered = _deliver(message, transport)

    # Assert
    (call,) = transport.calls
    assert delivered is True
    assert (call.url, call.body, call.signature) == (CALLBACK_URL, message.body, message.signature)


@pytest.mark.parametrize("first_answer", [NO_ANSWER, RETRY_LATER], ids=["no-answer", "a-retryable-hub-error"])
def test_deliver_when_the_first_attempt_may_be_sent_again_should_send_the_same_bytes_again_and_return_true(
    first_answer,
):
    # Arrange
    message = _sealed_unclaimed()
    transport = FakeTransport(first_answer, DELIVERED)

    # Act
    delivered = _deliver(message, transport)

    # Assert
    assert delivered is True
    assert [(call.url, call.body, call.signature) for call in transport.calls] == [
        (CALLBACK_URL, message.body, message.signature)
    ] * 2


@pytest.mark.parametrize("every_answer", [NO_ANSWER, RETRY_LATER], ids=["no-answer", "a-retryable-hub-error"])
def test_deliver_when_every_attempt_may_be_sent_again_should_stop_after_the_allowed_attempts_and_return_false(
    every_answer,
):
    # Arrange
    transport = FakeTransport(then=every_answer)

    # Act
    delivered = _deliver(_sealed_unclaimed(), transport)

    # Assert
    assert delivered is False
    assert len(transport.calls) == CALLBACK_ATTEMPTS
    assert CALLBACK_ATTEMPTS > 1


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(hub_error_answer(HTTP_CONFLICT, "conflicting_terminal_result", retryable=False), id="a-final-hub-error"),
        pytest.param(HttpAnswer(HTTP_BAD_GATEWAY, HTML_BODY, None), id="an-html-502"),
        pytest.param(HttpAnswer(HTTP_FOUND, b"", None), id="a-redirect"),
    ],
)
def test_deliver_when_the_answer_is_a_final_refusal_or_no_contract_answer_should_stop_at_once_and_return_false(answer):
    # Arrange
    transport = FakeTransport(answer, then=DELIVERED)

    # Act
    delivered = _deliver(_sealed_unclaimed(), transport)

    # Assert
    assert delivered is False
    assert len(transport.calls) == 1


def test_deliver_when_allowed_one_attempt_should_post_exactly_once():
    # Arrange
    transport = FakeTransport(NO_ANSWER, then=DELIVERED)

    # Act
    delivered = _deliver(_sealed_unclaimed(), transport, attempts=1)

    # Assert
    assert delivered is False
    assert len(transport.calls) == 1


def test_deliver_when_given_a_claim_should_raise_value_error_without_posting():
    # Arrange
    transport = FakeTransport(then=DELIVERED)

    # Act / Assert
    with pytest.raises(ValueError):
        _deliver(_sealed_claim(), transport)
    assert transport.calls == []


# --- what a sealed message shows ----------------------------------------------------------------


def test_sealed_message_when_shown_with_repr_should_show_neither_its_body_nor_its_signature():
    # Arrange
    sealed = _sealed_unclaimed()
    fixture = _fixture(UNCLAIMED_REFUSED)
    hidden = [sealed.signature, sealed.signature[-SIGNATURE_HEX_DIGITS:], sealed.body.decode("ascii"), fixture["runtime_id"], fixture["message_id"]]

    # Act
    shown = repr(sealed)

    # Assert
    assert MessageKind.TRANSCODE_UNCLAIMED.value in shown
    assert [value for value in hidden if value in shown] == []
