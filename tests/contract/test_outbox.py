"""What the worker sends leaves only self-checked and signed, and travels as the job output or to the callback address.

seal_message() encodes a built message, validates those exact bytes under the sender context and signs
them: a message its own validator refuses raises SelfCheckFailed and is never sealed. runpod_output() is
the job's output carrier of a result or an unclaimed report; deliver() posts the same bytes to the
configured callback address, sends again only after a transport failure or a retryable `hub.error`, and
never raises for a delivery that failed. A claim travels neither way. Given the instant it must give up
at, deliver() starts neither an attempt nor a wait that would reach it, and shortens each POST's
time-out to the time left; given none, it behaves as it always did.

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

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
    CALLBACK_TIMEOUT_S,
    SealedMessage,
    SelfCheckFailed,
    deliver,
    runpod_output,
    seal_message,
)
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
