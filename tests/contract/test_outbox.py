"""What the worker sends leaves only self-checked and signed, and travels as the job output or to the callback address.

seal_message() encodes a built message, validates those exact bytes under the sender context and signs
them: a message its own validator refuses raises SelfCheckFailed and is never sealed, and a SealedMessage
is made in no other way. runpod_output() is the job's output carrier of a result or an unclaimed report;
deliver() posts the same bytes to the configured callback address, sends again only after a transport
failure, a gateway's 502, 503 or 504 that carries no accepted `hub.error`, or a retryable `hub.error`
(read only under a 4xx or 5xx status), and never raises for a delivery that failed. A claim travels
neither way. Given the instant it must give up at, deliver() starts neither an attempt nor a wait that
would reach it, and shortens each POST's time-out to the time left; given none, it behaves as it always
did. Nothing of a message or of an answer is logged.

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import dataclasses
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Iterator

import pytest

from core.failure import Failure
from core.protocol import files, messages, schema, signing
from core.protocol import outbox as outbox_module
from core.protocol.context import TrustedContext
from core.protocol.messages import DispatchIdentity, ExecutionIdentity
from core.protocol.outbox import (
    CALLBACK_ATTEMPTS,
    CALLBACK_BACKOFF_S,
    CALLBACK_TIMEOUT_S,
    SealedMessage,
    SelfCheckFailed,
    deliver,
    runpod_output,
    seal_message,
)
from core.protocol.registry import MessageKind, load_registry
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.contract.v2_support import (
    CALLBACK_URL,
    FIXTURE_NOW,
    HUB_ERROR_MESSAGE_ID,
    HUB_ERROR_TEXT,
    HUB_KEY,
    LOG_PROBE,
    MEDIA_KEY_HEX,
    TOO_LARGE_FOR_A_FLOAT,
    UNCLAIMED_REFUSED,
    WORKER_KEY,
    FakeClock,
    FakeTransport,
    Outcome,
    compact,
    hub_error_answer,
    key_forms,
    moment,
    slices_shown,
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
# The last status that is a success: every 2xx is a delivery.
HTTP_LAST_SUCCESS = 299
HTTP_FOUND = 302
HTTP_NOT_MODIFIED = 304
HTTP_CONFLICT = 409
HTTP_INTERNAL_ERROR = 500
HTTP_NOT_IMPLEMENTED = 501
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504
HTTP_VERSION_NOT_SUPPORTED = 505
# The last status that is no error status, and the first and the last that are: a `hub.error` is read
# under an error status only.
HTTP_LAST_BEFORE_THE_ERRORS = 399
HTTP_BAD_REQUEST = 400
HTTP_LAST_ERROR = 599
# The first status after the error statuses, and the last one a status line can hold.
HTTP_FIRST_AFTER_THE_ERRORS = 600
HTTP_LAST_STATUS = 999
# What a gateway in front of the Hub answers while it cannot reach it.
GATEWAY_STATUSES = [HTTP_BAD_GATEWAY, HTTP_UNAVAILABLE, HTTP_GATEWAY_TIMEOUT]
HTML_BODY = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
RETRYABLE_CODE = "service_unavailable"
FINAL_CODE = "conflicting_terminal_result"
UNKNOWN_HUB_ERROR_CODE = "made_up_refusal"
HUB_ERROR_MAX_BYTES = load_registry().max_bytes(MessageKind.HUB_ERROR)
NO_ANSWER = TransportFailed("ReadTimeout")
DELIVERED = HttpAnswer(HTTP_OK, b"", None)
RETRY_LATER = hub_error_answer(HTTP_UNAVAILABLE, RETRYABLE_CODE, retryable=True)
FINAL_REFUSAL = hub_error_answer(HTTP_CONFLICT, FINAL_CODE, retryable=False)
GATEWAY_FAILURE = HttpAnswer(HTTP_BAD_GATEWAY, HTML_BODY, None)
# What a gateway answers while it cannot reach the Hub: one of its own statuses, and no accepted `hub.error`
# (a body that says it is one and is not accepted, whatever it says of sending again, is none).
GATEWAY_FAILURES = [
    pytest.param(GATEWAY_FAILURE, id="an-html-502"),
    pytest.param(HttpAnswer(HTTP_UNAVAILABLE, HTML_BODY, None), id="an-html-503"),
    pytest.param(HttpAnswer(HTTP_GATEWAY_TIMEOUT, HTML_BODY, None), id="an-html-504"),
    pytest.param(HttpAnswer(HTTP_GATEWAY_TIMEOUT, b"", None), id="an-empty-504"),
    pytest.param(
        hub_error_answer(HTTP_UNAVAILABLE, UNKNOWN_HUB_ERROR_CODE, retryable=False), id="a-503-hub-error-with-an-unknown-code"
    ),
]
# What a delivery waits before its second and its third attempt, and the least time-out of a POST that
# started before the delivery's end.
FIRST_WAIT_S = 1.0
SECOND_WAIT_S = 2.0
LEAST_POST_TIMEOUT_S = 1.0
FAILED_RESULT_KIND = "transcode.result.failed"
STOPPED_FOR_WANT_OF_TIME = "no time is left"


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


def _deliver_on(sealed: SealedMessage, transport: FakeTransport, clock: FakeClock, **options: Any) -> bool:
    """Deliver with nothing to give up at, waiting on the test's clock, which records each wait."""
    return deliver(sealed, settings=SETTINGS, transport=transport, sleep=clock.sleep, **options)


def _deliver_within(
    sealed: SealedMessage, transport: FakeTransport, clock: FakeClock, seconds: float, **options: Any
) -> bool:
    """Deliver on the test's clock with `seconds` left until the delivery must give up."""
    return deliver(
        sealed,
        settings=SETTINGS,
        transport=transport,
        sleep=clock.sleep,
        give_up_at=clock.monotonic() + seconds,
        monotonic=clock.monotonic,
        **options,
    )


def _slow_transport(clock: FakeClock, seconds_per_post: float) -> FakeTransport:
    """A transport no POST is answered on, each taking `seconds_per_post` of the test's clock."""
    return FakeTransport(then=NO_ANSWER, clock=clock, seconds_per_post=seconds_per_post)


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


def test_self_check_failed_when_turned_into_text_should_say_the_kind_the_layer_and_the_reason_and_nothing_more():
    # Arrange: a report sent an hour before the context's now
    context = _unclaimed_context(now=moment(_fixture(UNCLAIMED_REFUSED)["sent_at"]) + ONE_HOUR)

    # Act
    with pytest.raises(SelfCheckFailed) as raised:
        seal_message(_unclaimed(), context, settings=SETTINGS)

    # Assert
    assert str(raised.value) == "a transcode.unclaimed fails its self-check at semantic/timestamp_out_of_tolerance"


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


@pytest.mark.parametrize("status", [HTTP_OK, HTTP_NO_CONTENT, HTTP_LAST_SUCCESS])
@pytest.mark.parametrize(
    "sealed",
    [_sealed_unclaimed, _sealed_failed_result, _sealed_progress],
    ids=["an-unclaimed-report", "a-result", "a-progress-message"],
)
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


# What a delivery is sent again after: no answer, a gateway that did not reach the Hub, and the Hub's own
# word that it may be, under any error status.
SENT_AGAIN_AFTER = [
    pytest.param(NO_ANSWER, id="no-answer"),
    pytest.param(RETRY_LATER, id="a-retryable-hub-error"),
    pytest.param(
        hub_error_answer(HTTP_BAD_REQUEST, RETRYABLE_CODE, retryable=True),
        id="a-retryable-hub-error-under-the-first-error-status",
    ),
    pytest.param(
        hub_error_answer(HTTP_LAST_ERROR, RETRYABLE_CODE, retryable=True),
        id="a-retryable-hub-error-under-the-last-error-status",
    ),
    pytest.param(GATEWAY_FAILURE, id="an-html-502"),
]


@pytest.mark.parametrize("first_answer", SENT_AGAIN_AFTER)
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


@pytest.mark.parametrize("every_answer", SENT_AGAIN_AFTER)
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
        pytest.param(FINAL_REFUSAL, id="a-final-hub-error"),
        pytest.param(HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None), id="an-html-500"),
        pytest.param(HttpAnswer(HTTP_NOT_IMPLEMENTED, HTML_BODY, None), id="an-html-501-just-below-the-gateway-statuses"),
        pytest.param(
            HttpAnswer(HTTP_VERSION_NOT_SUPPORTED, HTML_BODY, None), id="an-html-505-just-above-the-gateway-statuses"
        ),
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


@pytest.mark.parametrize(
    "status",
    [HTTP_FOUND, HTTP_NOT_MODIFIED, HTTP_LAST_BEFORE_THE_ERRORS, HTTP_FIRST_AFTER_THE_ERRORS, HTTP_LAST_STATUS],
    ids=[
        "a-redirect",
        "not-modified",
        "the-last-status-before-the-error-statuses",
        "the-first-status-after-the-error-statuses",
        "the-last-status-there-is",
    ],
)
def test_deliver_when_a_retryable_hub_error_comes_under_a_status_that_is_no_error_should_not_send_again(status):
    # Arrange: the body alone would have the message sent again (RETRY_LATER); the Hub would then take it
    clock = FakeClock()
    transport = FakeTransport(hub_error_answer(status, RETRYABLE_CODE, retryable=True), then=DELIVERED, clock=clock)

    # Act
    delivered = _deliver_on(_sealed_unclaimed(), transport, clock)

    # Assert: a `hub.error` is read only under a 4xx or 5xx status, so this is an answer like any other
    assert delivered is False
    assert len(transport.calls) == 1
    assert clock.waits == []


# --- a gateway's 502, 503 or 504 ----------------------------------------------------------------


@pytest.mark.parametrize("gateway_failure", GATEWAY_FAILURES)
def test_deliver_when_a_gateway_failure_precedes_the_hubs_2xx_should_wait_the_first_back_off_then_send_again(
    gateway_failure,
):
    # Arrange
    clock = FakeClock()
    message = _sealed_failed_result()
    transport = FakeTransport(gateway_failure, DELIVERED, clock=clock)

    # Act
    delivered = _deliver_on(message, transport, clock)

    # Assert: the gateway's answer is no answer from the Hub, so it is treated as a failure in transport
    assert delivered is True
    assert [(call.body, call.signature) for call in transport.calls] == [(message.body, message.signature)] * 2
    assert clock.waits == [FIRST_WAIT_S]


@pytest.mark.parametrize(
    ("attempts", "waits"),
    [
        (2, [FIRST_WAIT_S]),
        (3, [FIRST_WAIT_S, SECOND_WAIT_S]),
        # More attempts than the back-off has waits: the last wait is made again.
        (4, [FIRST_WAIT_S, SECOND_WAIT_S, SECOND_WAIT_S]),
    ],
    ids=["two-attempts", "three-attempts", "four-attempts"],
)
@pytest.mark.parametrize("status", GATEWAY_STATUSES)
def test_deliver_when_a_gateway_answers_every_attempt_should_post_exactly_the_allowed_attempts_and_return_false(
    status, attempts, waits
):
    # Arrange
    clock = FakeClock()
    transport = FakeTransport(then=HttpAnswer(status, HTML_BODY, None), clock=clock)

    # Act
    delivered = _deliver_on(_sealed_failed_result(), transport, clock, attempts=attempts)

    # Assert
    assert delivered is False
    assert len(transport.calls) == attempts
    assert clock.waits == waits


@pytest.mark.parametrize("status", GATEWAY_STATUSES)
def test_deliver_when_a_gateway_status_carries_a_final_hub_error_should_stop_at_once_and_return_false(status):
    # Arrange: the Hub itself answered, whatever the status: its refusal counts, not the status
    clock = FakeClock()
    transport = FakeTransport(hub_error_answer(status, FINAL_CODE, retryable=False), then=DELIVERED, clock=clock)

    # Act
    delivered = _deliver_on(_sealed_failed_result(), transport, clock)

    # Assert
    assert delivered is False
    assert len(transport.calls) == 1
    assert clock.waits == []


@pytest.mark.parametrize("status", GATEWAY_STATUSES)
def test_deliver_when_a_gateway_status_carries_a_retryable_hub_error_should_send_again_and_return_true(status):
    # Arrange
    clock = FakeClock()
    transport = FakeTransport(hub_error_answer(status, RETRYABLE_CODE, retryable=True), DELIVERED, clock=clock)

    # Act
    delivered = _deliver_on(_sealed_failed_result(), transport, clock)

    # Assert
    assert delivered is True
    assert len(transport.calls) == 2
    assert clock.waits == [FIRST_WAIT_S]


# --- the attempts and each POST -----------------------------------------------------------------


def test_deliver_when_allowed_one_attempt_should_post_exactly_once():
    # Arrange
    transport = FakeTransport(NO_ANSWER, then=DELIVERED)

    # Act
    delivered = _deliver(_sealed_unclaimed(), transport, attempts=1)

    # Assert
    assert delivered is False
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "attempts",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="a-bool"),
        pytest.param(1.5, id="a-fraction"),
        pytest.param(2.0, id="a-whole-float"),
        pytest.param("3", id="text"),
        pytest.param(None, id="none"),
    ],
)
def test_deliver_when_the_attempts_are_no_whole_number_of_one_or_more_should_raise_value_error_without_posting(
    attempts,
):
    # Arrange
    transport = FakeTransport(then=DELIVERED)

    # Act / Assert
    with pytest.raises(ValueError):
        _deliver(_sealed_failed_result(), transport, attempts=attempts)
    assert transport.calls == []


def test_deliver_when_no_attempt_is_answered_should_wait_its_back_off_in_order_and_bound_every_post_alike():
    # Arrange
    clock = FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock)

    # Act
    delivered = _deliver_on(_sealed_failed_result(), transport, clock)

    # Assert: one second, then two; every POST with the callback's own time-out, and an answer cut at
    # the size of the only one a delivery reads, a `hub.error`
    assert delivered is False
    assert clock.waits == [FIRST_WAIT_S, SECOND_WAIT_S]
    assert tuple(clock.waits) == CALLBACK_BACKOFF_S
    assert [(call.timeout_s, call.max_body_bytes) for call in transport.calls] == [
        (CALLBACK_TIMEOUT_S, HUB_ERROR_MAX_BYTES)
    ] * CALLBACK_ATTEMPTS


def test_deliver_when_given_no_time_out_should_post_with_the_callbacks_own():
    # Arrange
    transport = FakeTransport(NO_ANSWER, then=DELIVERED)

    # Act
    _deliver(_sealed_failed_result(), transport)

    # Assert
    assert [call.timeout_s for call in transport.calls] == [CALLBACK_TIMEOUT_S, CALLBACK_TIMEOUT_S]
    assert CALLBACK_TIMEOUT_S == 10.0


def test_deliver_when_given_a_time_out_should_post_every_attempt_with_it():
    # Arrange
    shorter_than_the_callbacks_own = 3.0
    transport = FakeTransport(NO_ANSWER, then=DELIVERED)

    # Act
    delivered = _deliver(_sealed_progress(), transport, timeout_s=shorter_than_the_callbacks_own)

    # Assert
    assert delivered is True
    assert [call.timeout_s for call in transport.calls] == [3.0, 3.0]


@pytest.mark.parametrize(
    "timeout_s",
    [
        pytest.param(0, id="zero"),
        pytest.param(0.0, id="zero-as-a-float"),
        pytest.param(-3.0, id="negative"),
        pytest.param(float("nan"), id="not-a-number"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("-inf"), id="minus-infinite"),
        pytest.param(True, id="a-bool"),
        pytest.param("3.0", id="text"),
        pytest.param(None, id="none"),
    ],
)
def test_deliver_when_the_time_out_is_no_positive_finite_number_should_raise_value_error_without_posting(timeout_s):
    # Arrange
    transport = FakeTransport(then=DELIVERED)

    # Act / Assert
    with pytest.raises(ValueError):
        _deliver(_sealed_progress(), transport, timeout_s=timeout_s)
    assert transport.calls == []


def test_deliver_when_given_a_claim_should_raise_value_error_without_posting():
    # Arrange
    transport = FakeTransport(then=DELIVERED)

    # Act / Assert
    with pytest.raises(ValueError):
        _deliver(_sealed_claim(), transport)
    assert transport.calls == []


# --- a delivery that must give up at an instant -------------------------------------------------


def test_deliver_when_the_hub_answers_2xx_with_time_left_should_return_true_after_one_post():
    # Arrange
    clock = FakeClock()
    transport = FakeTransport(DELIVERED, clock=clock)
    message = _sealed_failed_result()

    # Act
    delivered = _deliver_within(message, transport, clock, seconds=30.0)

    # Assert
    (call,) = transport.calls
    assert delivered is True
    assert (call.url, call.body, call.signature) == (CALLBACK_URL, message.body, message.signature)


@pytest.mark.parametrize("seconds_left", [0.0, -5.0], ids=["at-that-very-instant", "five-seconds-late"])
def test_deliver_when_its_time_is_already_spent_should_send_nothing_and_return_false(seconds_left):
    # Arrange: a Hub that would take the message at once
    clock = FakeClock()
    transport = FakeTransport(then=DELIVERED, clock=clock)

    # Act
    delivered = _deliver_within(_sealed_failed_result(), transport, clock, seconds=seconds_left)

    # Assert
    assert delivered is False
    assert transport.calls == []
    assert clock.waits == []


def test_deliver_when_its_time_runs_out_between_attempts_should_start_no_attempt_at_or_after_the_end():
    # Arrange: no POST is answered and each takes 4 s; the delivery must give up 6 s from now
    clock = FakeClock()
    transport = _slow_transport(clock, seconds_per_post=4.0)
    gives_up_at = clock() + timedelta(seconds=6.0)

    # Act
    delivered = _deliver_within(_sealed_failed_result(), transport, clock, seconds=6.0)

    # Assert: one attempt at once, one after the first wait, and no third: it would start after the end
    assert delivered is False
    assert [call.at - clock.start for call in transport.calls] == [timedelta(0), timedelta(seconds=5.0)]
    assert [call.at for call in transport.calls if call.at >= gives_up_at] == []
    assert len(transport.calls) < CALLBACK_ATTEMPTS


@pytest.mark.parametrize(
    ("seconds_left", "waits", "posts"),
    [
        pytest.param(FIRST_WAIT_S, [], 1, id="the-first-wait-would-end-at-the-end"),
        pytest.param(FIRST_WAIT_S + 0.5, [FIRST_WAIT_S], 2, id="the-first-wait-ends-before-the-end"),
        pytest.param(FIRST_WAIT_S + SECOND_WAIT_S, [FIRST_WAIT_S], 2, id="the-second-wait-would-end-at-the-end"),
        pytest.param(
            FIRST_WAIT_S + SECOND_WAIT_S + 0.5, [FIRST_WAIT_S, SECOND_WAIT_S], 3, id="both-waits-end-before-the-end"
        ),
    ],
)
def test_deliver_when_a_wait_would_end_at_or_after_the_end_should_not_start_it(seconds_left, waits, posts):
    # Arrange: no POST is answered, each fails at once, so only the waits take time
    clock = FakeClock()
    transport = _slow_transport(clock, seconds_per_post=0.0)

    # Act
    delivered = _deliver_within(_sealed_failed_result(), transport, clock, seconds=seconds_left)

    # Assert
    assert delivered is False
    assert clock.waits == waits
    assert len(transport.calls) == posts
    assert clock.elapsed_s < seconds_left


def test_deliver_when_its_time_runs_out_between_gateway_failures_should_start_no_attempt_at_or_after_the_end():
    # Arrange: a gateway answers each POST after 4 s; the delivery must give up 6 s from now
    clock = FakeClock()
    transport = FakeTransport(then=GATEWAY_FAILURE, clock=clock, seconds_per_post=4.0)
    gives_up_at = clock() + timedelta(seconds=6.0)

    # Act
    delivered = _deliver_within(_sealed_failed_result(), transport, clock, seconds=6.0)

    # Assert: sent again once, after the first wait, and no third time: it would start after the end
    assert delivered is False
    assert [call.at - clock.start for call in transport.calls] == [timedelta(0), timedelta(seconds=5.0)]
    assert [call.at for call in transport.calls if call.at >= gives_up_at] == []
    assert len(transport.calls) < CALLBACK_ATTEMPTS


@pytest.mark.parametrize(
    ("seconds_left", "waits", "posts"),
    [
        pytest.param(FIRST_WAIT_S + 0.5, [FIRST_WAIT_S], 2, id="the-first-wait-ends-before-the-end"),
        pytest.param(FIRST_WAIT_S + SECOND_WAIT_S, [FIRST_WAIT_S], 2, id="the-second-wait-would-end-at-the-end"),
        pytest.param(
            FIRST_WAIT_S + SECOND_WAIT_S + 0.5, [FIRST_WAIT_S, SECOND_WAIT_S], 3, id="both-waits-end-before-the-end"
        ),
    ],
)
def test_deliver_when_a_wait_after_a_gateway_failure_would_end_at_or_after_the_end_should_not_start_it(
    seconds_left, waits, posts
):
    # Arrange: a gateway answers every POST at once, so only the waits take time
    clock = FakeClock()
    transport = FakeTransport(then=GATEWAY_FAILURE, clock=clock)

    # Act
    delivered = _deliver_within(_sealed_failed_result(), transport, clock, seconds=seconds_left)

    # Assert: sent again as after a failure in transport, and held to the same end
    assert delivered is False
    assert clock.waits == waits
    assert len(transport.calls) == posts
    assert clock.elapsed_s < seconds_left


@pytest.mark.parametrize(
    ("seconds_left", "options", "expected_timeout_s"),
    [
        pytest.param(30.0, {}, 10.0, id="more-time-left-than-the-callbacks-own-time-out"),
        pytest.param(4.0, {}, 4.0, id="less-time-left-than-the-callbacks-own-time-out"),
        pytest.param(LEAST_POST_TIMEOUT_S, {}, 1.0, id="exactly-the-least-time-out-left"),
        pytest.param(0.25, {}, 1.0, id="less-left-than-the-least-time-out"),
        pytest.param(30.0, {"timeout_s": 3.0}, 3.0, id="more-time-left-than-a-given-time-out"),
        pytest.param(2.0, {"timeout_s": 3.0}, 2.0, id="less-time-left-than-a-given-time-out"),
        pytest.param(0.2, {"timeout_s": 0.5}, 0.5, id="a-given-time-out-below-the-least-one"),
    ],
)
def test_deliver_when_it_must_give_up_should_cap_a_posts_time_out_at_the_time_left_counted_as_a_second_at_least(
    seconds_left, options, expected_timeout_s
):
    # Arrange: less time left than a second counts as a second, and a POST never gets more than its own
    # time-out, even one below a second
    clock = FakeClock()
    transport = FakeTransport(DELIVERED, clock=clock)

    # Act
    _deliver_within(_sealed_failed_result(), transport, clock, seconds=seconds_left, **options)

    # Assert
    assert [call.timeout_s for call in transport.calls] == [expected_timeout_s]
    assert CALLBACK_TIMEOUT_S == 10.0


def test_deliver_when_time_passes_between_attempts_should_give_each_post_only_what_is_then_left():
    # Arrange: no POST is answered and each takes 3 s; the delivery must give up 12 s from now
    clock = FakeClock()
    transport = _slow_transport(clock, seconds_per_post=3.0)

    # Act
    _deliver_within(_sealed_failed_result(), transport, clock, seconds=12.0)

    # Assert: 12 s left at first, 8 s after a post and a wait, then 3 s
    assert [call.timeout_s for call in transport.calls] == [10.0, 8.0, 3.0]


def test_deliver_when_it_stops_for_want_of_time_should_log_once_and_name_the_kind_only(caplog):
    # Arrange
    caplog.set_level(logging.DEBUG, logger=outbox_module.__name__)
    clock = FakeClock()
    message = _sealed_failed_result()
    fixture = _fixture(FAILED_RESULT)

    # Act
    _deliver_within(message, FakeTransport(then=DELIVERED, clock=clock), clock, seconds=0.0)

    # Assert
    (logged,) = [record for record in caplog.records if record.name == outbox_module.__name__]
    line = logged.getMessage()
    assert logged.levelno == logging.INFO
    assert STOPPED_FOR_WANT_OF_TIME in line and FAILED_RESULT_KIND in line
    assert [hidden for hidden in (fixture["message_id"], message.signature, CALLBACK_URL) if hidden in line] == []


@pytest.mark.parametrize(
    "end",
    [
        pytest.param({"give_up_at": 5.0}, id="an-instant-without-a-clock"),
        pytest.param({"monotonic": FakeClock().monotonic}, id="a-clock-without-an-instant"),
        pytest.param({"give_up_at": float("nan"), "monotonic": FakeClock().monotonic}, id="not-a-number"),
        pytest.param({"give_up_at": float("inf"), "monotonic": FakeClock().monotonic}, id="infinite"),
        pytest.param({"give_up_at": float("-inf"), "monotonic": FakeClock().monotonic}, id="minus-infinite"),
        pytest.param(
            {"give_up_at": TOO_LARGE_FOR_A_FLOAT, "monotonic": FakeClock().monotonic},
            id="an-integer-too-large-to-be-a-float",
        ),
        pytest.param(
            {"give_up_at": -TOO_LARGE_FOR_A_FLOAT, "monotonic": FakeClock().monotonic},
            id="a-negative-integer-too-large-to-be-a-float",
        ),
        pytest.param({"give_up_at": True, "monotonic": FakeClock().monotonic}, id="a-bool"),
        pytest.param({"give_up_at": "5.0", "monotonic": FakeClock().monotonic}, id="text"),
    ],
)
def test_deliver_when_its_end_is_given_by_half_or_is_no_finite_number_should_raise_value_error_without_posting(end):
    # Arrange
    transport = FakeTransport(then=DELIVERED)

    # Act / Assert
    with pytest.raises(ValueError):
        _deliver(_sealed_failed_result(), transport, **end)
    assert transport.calls == []


def test_deliver_when_its_end_is_a_whole_number_should_take_it_as_an_instant():
    # Arrange: five whole seconds of the clock, as an int
    clock = FakeClock()
    transport = FakeTransport(DELIVERED, clock=clock)

    # Act
    delivered = deliver(
        _sealed_failed_result(),
        settings=SETTINGS, transport=transport, sleep=clock.sleep, give_up_at=5, monotonic=clock.monotonic,
    )

    # Assert
    assert delivered is True
    assert [call.timeout_s for call in transport.calls] == [5.0]


# --- a delivery given no such instant behaves as it always did ----------------------------------


def test_deliver_when_given_no_end_should_wait_before_each_further_attempt_however_long_the_posts_take():
    # Arrange: no POST is answered and each takes a minute
    clock = FakeClock()
    transport = _slow_transport(clock, seconds_per_post=60.0)

    # Act
    delivered = deliver(_sealed_failed_result(), settings=SETTINGS, transport=transport, sleep=clock.sleep)

    # Assert: every attempt is made, each with the callback's own time-out, after the same two waits
    assert delivered is False
    assert len(transport.calls) == CALLBACK_ATTEMPTS == 3
    assert clock.waits == [FIRST_WAIT_S, SECOND_WAIT_S]
    assert [call.timeout_s for call in transport.calls] == [10.0, 10.0, 10.0]


def test_deliver_when_given_no_end_should_post_with_a_time_out_below_one_second_as_given():
    # Arrange: the least time-out belongs to a delivery that must give up, not to this one
    transport = FakeTransport(DELIVERED)

    # Act
    _deliver(_sealed_progress(), transport, timeout_s=0.5)

    # Assert
    assert [call.timeout_s for call in transport.calls] == [0.5]


# --- what a delivery logs -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        pytest.param(NO_ANSWER, "got no answer: ReadTimeout", id="no-answer"),
        pytest.param(RETRY_LATER, f"refused for now: {RETRYABLE_CODE}", id="a-retryable-hub-error"),
        pytest.param(FINAL_REFUSAL, f"refused: status {HTTP_CONFLICT}, code {FINAL_CODE}", id="a-final-hub-error"),
        pytest.param(
            HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None),
            f"refused: status {HTTP_INTERNAL_ERROR}, code none",
            id="an-html-500",
        ),
        pytest.param(GATEWAY_FAILURE, f"did not reach the Hub: status {HTTP_BAD_GATEWAY}", id="a-gateway-failure"),
    ],
)
def test_deliver_when_an_attempt_is_not_delivered_should_log_one_line_naming_the_kind_and_the_outcome(
    caplog, answer, outcome
):
    # Arrange: the Hub takes the message at the next attempt, where there is one
    caplog.set_level(logging.DEBUG, logger=outbox_module.__name__)
    transport = FakeTransport(answer, then=DELIVERED)

    # Act
    _deliver(_sealed_failed_result(), transport)

    # Assert
    (logged,) = [record for record in caplog.records if record.name == outbox_module.__name__]
    line = logged.getMessage()
    assert logged.levelno == logging.INFO
    assert FAILED_RESULT_KIND in line and outcome in line


@pytest.mark.parametrize(
    ("outcomes", "then"),
    [
        pytest.param((DELIVERED,), None, id="delivered"),
        pytest.param((), NO_ANSWER, id="never-answered"),
        pytest.param((RETRY_LATER, NO_ANSWER, DELIVERED), None, id="delivered-after-retries"),
        pytest.param((), RETRY_LATER, id="refused-for-now-until-the-end"),
        pytest.param((FINAL_REFUSAL,), None, id="refused"),
        pytest.param((HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None),), None, id="an-html-500"),
        pytest.param((GATEWAY_FAILURE,), DELIVERED, id="a-gateway-failure-first"),
        pytest.param((), GATEWAY_FAILURE, id="gateway-failures-until-the-end"),
    ],
)
def test_deliver_when_it_ends_any_way_should_log_nothing_of_the_message_or_of_an_answer(
    caplog, outcomes: tuple[Outcome, ...], then: Outcome | None
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    message = _sealed_failed_result()
    transport = FakeTransport(*outcomes, then=then)
    answers = [answer.body.decode("ascii") for answer in (*outcomes, then) if isinstance(answer, HttpAnswer)]
    # A line may name the message's kind and the code of the Hub's refusal, and nothing else of either.
    nameable = (FAILED_RESULT_KIND, RETRYABLE_CODE, FINAL_CODE)

    # Act
    _deliver(message, transport)
    logging.getLogger(outbox_module.__name__).debug(LOG_PROBE)
    logged = caplog.text

    # Assert
    # A key is held as bytes, so it is looked for as bytes are printed too, not as hex alone.
    keys = [bytes.fromhex(MEDIA_KEY_HEX), WORKER_KEY, HUB_KEY]
    secrets = [message.signature, message.signature[-SIGNATURE_HEX_DIGITS:], CALLBACK_URL, HUB_ERROR_TEXT]
    secrets += [HUB_ERROR_MESSAGE_ID, *(form for key in keys for form in key_forms(key))]
    assert LOG_PROBE in logged
    assert [secret for secret in secrets if secret in logged] == []
    assert slices_shown(logged, message.body.decode("ascii"), nameable) == []
    assert [piece for answer in answers for piece in slices_shown(logged, answer, nameable)] == []


# --- a sealed message is made by seal_message() only --------------------------------------------


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(
            lambda sealed: SealedMessage(kind=sealed.kind, body=sealed.body, signature=sealed.signature), id="by-name"
        ),
        pytest.param(lambda sealed: SealedMessage(sealed.kind, sealed.body, sealed.signature), id="by-position"),
        pytest.param(
            lambda sealed: SealedMessage(kind=sealed.kind, body=b'{"never":"self-checked"}', signature=sealed.signature),
            id="of-bytes-that-passed-no-self-check",
        ),
        pytest.param(
            lambda sealed: SealedMessage(sealed.kind, sealed.body, sealed.signature, object()),
            id="with-a-made-up-fourth-argument",
        ),
        pytest.param(
            lambda sealed: SealedMessage(
                kind=sealed.kind, body=sealed.body, signature=sealed.signature, _construction=True
            ),
            id="with-true-as-its-fourth-argument",
        ),
    ],
)
def test_sealed_message_when_constructed_directly_should_raise_type_error(construct):
    # Arrange: the kind, the bytes and the signature of a message that did pass its self-check
    sealed = _sealed_unclaimed()

    # Act / Assert
    with pytest.raises(TypeError):
        construct(sealed)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({}, id="nothing-changed"),
        pytest.param({"body": b'{"never":"self-checked"}'}, id="another-body"),
        pytest.param({"kind": MessageKind.TRANSCODE_RESULT_COMPLETED}, id="another-kind"),
    ],
)
def test_sealed_message_when_copied_with_dataclasses_replace_should_raise_type_error(changes):
    # Arrange
    sealed = _sealed_unclaimed()

    # Act / Assert
    with pytest.raises(TypeError):
        dataclasses.replace(sealed, **changes)


def test_sealed_message_when_made_by_seal_message_should_hold_its_three_fields_and_equal_one_of_the_same_bytes():
    # Arrange
    message = _unclaimed()

    # Act
    sealed = seal_message(message, _unclaimed_context(), settings=SETTINGS)
    sealed_again = seal_message(message, _unclaimed_context(), settings=SETTINGS)

    # Assert: refusing every other construction changed nothing else of the value
    assert [field.name for field in dataclasses.fields(sealed)] == ["kind", "body", "signature"]
    assert (sealed.kind, sealed.body) == (MessageKind.TRANSCODE_UNCLAIMED, messages.encode(message))
    assert signing.verify(sealed.body, WORKER_KEY, sealed.signature)
    assert sealed == sealed_again and hash(sealed) == hash(sealed_again)
    assert sealed != _sealed_failed_result()
    assert repr(sealed) == f"SealedMessage(kind={MessageKind.TRANSCODE_UNCLAIMED!r})"
    with pytest.raises(dataclasses.FrozenInstanceError):
        sealed.body = b""  # type: ignore[misc]


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
