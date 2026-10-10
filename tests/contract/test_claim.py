"""The claim exchange, the worker's side (README section 15.1): a runtime works only on an accepted, signed grant.

claim_execution() sends `transcode.claim` to the configured claim address and returns a Grant only for a
`200` whose body verifies under a Hub-to-worker key and is an accepted `transcode.claim.granted` for this
dispatch, this runtime and this encryption. Every other ending is Unclaimed: `refused` (a `hub.error`),
`unreachable` (no answer) or `invalid_answer` (anything else, never retried). It claims again only after
a transport failure or a retryable `hub.error`, never at or after the dispatch's `claim.expires_at`, and
never for longer than its patience. Nothing of a claim or of an answer is logged.

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import dataclasses
import json
import logging
from datetime import timedelta
from typing import Any, Mapping, Sequence

import pytest

from core.protocol import claim as claim_module
from core.protocol import files, pipeline, signing
from core.protocol.claim import CLAIM_BACKOFF_S, CLAIM_PATIENCE_S, Grant, Unclaimed, claim_execution
from core.protocol.dispatch import AcceptedDispatch
from core.protocol.reasons import ReasonCode
from core.protocol.registry import MessageKind, load_registry
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.contract.v2_support import (
    CELL_ID,
    CLAIM_URL,
    FIXTURE_NOW,
    GRANT_ENCRYPTED,
    GRANT_UNENCRYPTED,
    HUB_CLAIM_CONTEXT,
    HUB_KEY,
    MEDIA_KEY_HEX,
    REQUEST_ENCRYPTED,
    REQUEST_UNENCRYPTED,
    RUNTIME_ID,
    UNCONFIGURED_KEY,
    WORKER_GRANT_CONTEXT,
    WORKER_GRANT_CONTEXT_UNENCRYPTED,
    WORKER_KEY,
    WORKER_KEY_ID,
    FakeClock,
    FakeTransport,
    accepted_dispatch,
    catalog_context,
    hub_error_answer,
    hub_error_body,
    modified_fixture,
    moment,
    padded_to,
    resigned_fixture,
    seconds_from_now,
    signed_answer,
    uuid_source,
    worker_settings,
)

SETTINGS = worker_settings()
SKEW_S = SETTINGS.clock_skew_s
CAUSE_REFUSED = "refused"
CAUSE_UNREACHABLE = "unreachable"
CAUSE_INVALID_ANSWER = "invalid_answer"
CLAIM_KIND = "transcode.claim"
GRANT_MAX_BYTES = load_registry().max_bytes(MessageKind.TRANSCODE_CLAIM_GRANTED)
HUB_ERROR_MAX_BYTES = load_registry().max_bytes(MessageKind.HUB_ERROR)
# The worker context of the catalog under which each dispatch's grant is judged.
GRANT_CONTEXT_OF = {
    REQUEST_ENCRYPTED: WORKER_GRANT_CONTEXT,
    REQUEST_UNENCRYPTED: WORKER_GRANT_CONTEXT_UNENCRYPTED,
}
# A billed runtime gives up within two minutes, whatever is left of a bootstrap that can last an hour: the
# module's own patience, and never more than that.
LONGEST_PATIENCE_S = 120
PATIENCE_S = min(CLAIM_PATIENCE_S, LONGEST_PATIENCE_S)
PATIENCE_ENDS = FIXTURE_NOW + timedelta(seconds=PATIENCE_S)
FIRST_BACKOFF_S = CLAIM_BACKOFF_S[0]
RETRY_AFTER_EXCESS_S = 6
RETRY_AFTER_LONGER_THAN_THE_BACKOFF_S = FIRST_BACKOFF_S + RETRY_AFTER_EXCESS_S
RETRY_AFTER_AN_HOUR_S = 3600
# How long a POST that times out takes on the fake clock in the slow-transport cases.
SLOW_POST_S = 15.0
TWO_SECOND_POST_S = 2.0
AES_128_KEY_BYTES = 16
HTTP_OK = 200
EXPIRES_SOON_S = 5
EXPIRED_ALREADY_S = -10
OTHER_UUID = "9a9a9a9a-9a9a-79a9-89a9-9a9a9a9a9a9a"
UNKNOWN_HUB_ERROR_CODE = "made_up_refusal"
HTML_BODY = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
HTTP_CREATED = 201
HTTP_NO_CONTENT = 204
HTTP_FOUND = 302
HTTP_UNAUTHORIZED = 401
HTTP_CONFLICT = 409
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
NO_ANSWER = TransportFailed("ConnectTimeout")
RETRYABLE_REFUSAL_CODE = "service_unavailable"
FINAL_REFUSAL_CODE = "claim_conflict"
# Steps of the wall clock while the Hub does not answer: set back an hour, set forward a minute.
WALL_CLOCK_STEPS_S = {"set-back-an-hour": -3600.0, "set-forward-a-minute": 60.0}
# A log line may name these two ids; any longer slice of a claim or of a grant is a leak.
LEAK_SLICE_CHARS = 16
LOG_PROBE = "probe: the capture sees this module's logger at DEBUG"


def _request(path: str = REQUEST_ENCRYPTED) -> dict[str, Any]:
    return json.loads(files.read_bytes(path))


def _dispatch(path: str = REQUEST_ENCRYPTED, **changes: Any) -> AcceptedDispatch:
    """The accepted dispatch of a request fixture, signed by the Hub after `changes` (JSON pointers) were applied."""
    request = modified_fixture(path, changes) if changes else files.read_bytes(path)
    return accepted_dispatch(request, settings=SETTINGS)


def _claim(dispatch: AcceptedDispatch, transport: FakeTransport, clock: FakeClock) -> Grant | Unclaimed:
    return claim_execution(
        dispatch,
        settings=SETTINGS,
        runtime_id=RUNTIME_ID,
        transport=transport,
        clock=clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        new_message_id=uuid_source(),
    )


def _granted(path: str = GRANT_ENCRYPTED) -> HttpAnswer:
    """The fixture's grant as the Hub answers it: status 200, signed with the configured Hub-to-worker key."""
    return signed_answer(files.read_bytes(path), HUB_KEY)


def _changed_grant(changes: Mapping[str, Any]) -> HttpAnswer:
    """The encrypted grant with some members changed, correctly signed after the change."""
    signed = resigned_fixture(GRANT_ENCRYPTED, changes, HUB_KEY)
    return HttpAnswer(status=HTTP_OK, body=signed.body, signature=signed.signature)


def _retryable_refusal(retry_after_s: int | None = None, code: str = RETRYABLE_REFUSAL_CODE) -> HttpAnswer:
    return hub_error_answer(HTTP_UNAVAILABLE, code, retryable=True, retry_after_s=retry_after_s)


def _sent(transport: FakeTransport) -> list[dict[str, Any]]:
    """The claims that were POSTed, decoded."""
    return [json.loads(call.body) for call in transport.calls]


# --- granted ------------------------------------------------------------------------------------


def test_claim_execution_when_the_hub_grants_an_encrypted_dispatch_should_return_the_grant_with_its_media_key():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    fixture = json.loads(files.read_bytes(GRANT_ENCRYPTED))
    transport = FakeTransport(_granted(), clock=clock)

    # Act
    grant = _claim(dispatch, transport, clock)

    # Assert
    assert isinstance(grant, Grant)
    assert dataclasses.asdict(grant.identity) == {
        **dataclasses.asdict(dispatch.identity),
        "execution_id": fixture["identity"]["execution_id"],
        "execution_fence": fixture["identity"]["execution_fence"],
    }
    assert grant.generation_id == fixture["generation_id"]
    assert grant.output_location_id == fixture["output"]["location_id"]
    assert grant.output_prefix == fixture["output"]["prefix"]
    assert grant.media_key_id == fixture["media_key"]["media_key_id"]
    assert grant.media_key == bytes.fromhex(fixture["media_key"]["key_hex"])
    assert len(grant.media_key) == AES_128_KEY_BYTES
    assert len(transport.calls) == 1


def test_claim_execution_when_the_hub_grants_an_unencrypted_dispatch_should_return_a_grant_without_a_key():
    # Arrange
    dispatch, clock = _dispatch(REQUEST_UNENCRYPTED), FakeClock()
    fixture = json.loads(files.read_bytes(GRANT_UNENCRYPTED))
    transport = FakeTransport(_granted(GRANT_UNENCRYPTED), clock=clock)

    # Act
    grant = _claim(dispatch, transport, clock)

    # Assert
    assert isinstance(grant, Grant)
    assert grant.identity.execution_id == fixture["identity"]["execution_id"]
    assert grant.generation_id == fixture["generation_id"]
    assert grant.media_key is None
    assert grant.media_key_id is None


def test_grant_when_shown_with_repr_should_not_show_the_media_key():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    grant = _claim(dispatch, FakeTransport(_granted(), clock=clock), clock)
    key = bytes.fromhex(MEDIA_KEY_HEX)

    # Act
    shown = repr(grant) + str(grant)

    # Assert
    assert grant.media_key == key
    assert [value for value in (MEDIA_KEY_HEX, MEDIA_KEY_HEX.upper(), repr(key), repr(key)[2:-1]) if value in shown] == []
    assert grant.generation_id in shown


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(signed_answer(padded_to(files.read_bytes(GRANT_ENCRYPTED), GRANT_MAX_BYTES), HUB_KEY), id="exactly-at-the-byte-limit"),
        pytest.param(_changed_grant({"/sent_at": seconds_from_now(-SKEW_S)}), id="sent-exactly-the-skew-ago"),
    ],
)
def test_claim_execution_when_the_grant_is_exactly_at_a_bound_should_accept_it(answer):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()

    # Act
    grant = _claim(dispatch, FakeTransport(answer, clock=clock), clock)

    # Assert
    assert isinstance(grant, Grant)


# --- the claim that is sent ---------------------------------------------------------------------


def test_claim_execution_when_it_claims_should_post_one_signed_accepted_claim_to_the_configured_address():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    request = _request()
    transport = FakeTransport(_granted(), clock=clock)

    # Act
    _claim(dispatch, transport, clock)

    # Assert
    (call,) = transport.calls
    sent = json.loads(call.body)
    hub_context = catalog_context(HUB_CLAIM_CONTEXT, now=call.at, known_key_ids=[WORKER_KEY_ID])
    assert call.url == CLAIM_URL
    assert pipeline.validate(call.body, hub_context).accepted
    assert sent["message_kind"] == CLAIM_KIND
    assert sent["audience"] == f"hub:{CELL_ID}"
    assert sent["identity"] == request["identity"]
    assert sent["claim"] == {"bootstrap_token": request["claim"]["bootstrap_token"], "runtime_id": RUNTIME_ID}
    assert signing.verify(call.body, WORKER_KEY, call.signature)
    assert not signing.verify(call.body, HUB_KEY, call.signature)
    assert call.max_body_bytes >= GRANT_MAX_BYTES


# --- invalid_answer: never retried --------------------------------------------------------------

# (the dispatch, the 200 answer, what the validator says of its body under the worker's grant context:
# None when the body itself is acceptable and only its signature is at fault)
NOT_A_GRANT_FOR_THIS_RUNTIME = [
    pytest.param(
        REQUEST_ENCRYPTED, HttpAnswer(HTTP_OK, files.read_bytes(GRANT_ENCRYPTED), None), None, id="no-signature-header"
    ),
    pytest.param(
        REQUEST_ENCRYPTED, signed_answer(files.read_bytes(GRANT_ENCRYPTED), UNCONFIGURED_KEY), None, id="signed-with-an-unconfigured-key"
    ),
    pytest.param(
        REQUEST_ENCRYPTED, signed_answer(files.read_bytes(GRANT_ENCRYPTED), WORKER_KEY), None, id="signed-with-the-workers-own-key"
    ),
    pytest.param(
        REQUEST_ENCRYPTED, _changed_grant({"/runtime_id": OTHER_UUID}), ReasonCode.RUNTIME_MISMATCH, id="for-another-runtime"
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        _changed_grant({"/identity/dispatch_id": OTHER_UUID}),
        ReasonCode.DISPATCH_MISMATCH,
        id="for-another-dispatch",
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        _changed_grant({"/output/location_id": OTHER_UUID}),
        ReasonCode.STORAGE_LOCATION_MISMATCH,
        id="another-output-location",
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        _changed_grant({"/generation_id": OTHER_UUID, "/output/prefix": f"test/media/{OTHER_UUID}"}),
        ReasonCode.GENERATION_EXECUTION_MISMATCH,
        id="generation-is-not-the-execution",
    ),
    pytest.param(
        REQUEST_ENCRYPTED, _granted(GRANT_UNENCRYPTED), ReasonCode.ENCRYPTION_MISMATCH, id="encrypted-dispatch-grant-without-a-key"
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        _changed_grant({"/media_key/media_key_id": OTHER_UUID}),
        ReasonCode.ENCRYPTION_MISMATCH,
        id="another-media-key-id",
    ),
    pytest.param(
        REQUEST_UNENCRYPTED, _granted(GRANT_ENCRYPTED), ReasonCode.ENCRYPTION_MISMATCH, id="unencrypted-dispatch-grant-with-a-key"
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        _changed_grant({"/sent_at": seconds_from_now(-(SKEW_S + 1))}),
        ReasonCode.TIMESTAMP_OUT_OF_TOLERANCE,
        id="sent-one-second-beyond-the-skew",
    ),
    pytest.param(
        REQUEST_ENCRYPTED,
        signed_answer(hub_error_body(RETRYABLE_REFUSAL_CODE, retryable=True), HUB_KEY),
        ReasonCode.UNEXPECTED_MESSAGE_KIND,
        id="a-signed-hub-error",
    ),
    pytest.param(REQUEST_ENCRYPTED, signed_answer(b"", HUB_KEY), ReasonCode.MALFORMED_JSON, id="an-empty-body"),
    pytest.param(
        REQUEST_ENCRYPTED,
        signed_answer(padded_to(files.read_bytes(GRANT_ENCRYPTED), GRANT_MAX_BYTES + 1), HUB_KEY),
        ReasonCode.PAYLOAD_TOO_LARGE,
        id="one-byte-above-the-limit",
    ),
]


@pytest.mark.parametrize(("request_path", "answer", "reason"), NOT_A_GRANT_FOR_THIS_RUNTIME)
def test_claim_execution_when_a_200_is_not_an_accepted_grant_for_this_runtime_should_end_invalid_after_one_post(
    request_path, answer, reason
):
    # Arrange: the body is at fault for exactly `reason` (None: the body is acceptable, its signature is not).
    dispatch, clock = _dispatch(request_path), FakeClock()
    transport = FakeTransport(answer, clock=clock)
    assert pipeline.validate(answer.body, catalog_context(GRANT_CONTEXT_OF[request_path])).reason == reason

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_INVALID_ANSWER)
    assert len(transport.calls) == 1
    assert clock.waits == []


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(HttpAnswer(HTTP_BAD_GATEWAY, HTML_BODY, None), id="an-html-502"),
        pytest.param(HttpAnswer(HTTP_FOUND, b"", None), id="a-redirect"),
        pytest.param(HttpAnswer(HTTP_NO_CONTENT, b"", None), id="no-content"),
        pytest.param(
            hub_error_answer(HTTP_FOUND, RETRYABLE_REFUSAL_CODE, retryable=True), id="a-retryable-hub-error-under-a-redirect"
        ),
        pytest.param(
            hub_error_answer(HTTP_NO_CONTENT, FINAL_REFUSAL_CODE, retryable=False), id="a-final-hub-error-under-a-204"
        ),
        pytest.param(signed_answer(files.read_bytes(GRANT_ENCRYPTED), HUB_KEY, status=HTTP_CREATED), id="a-grant-with-status-201"),
        pytest.param(
            hub_error_answer(HTTP_UNAVAILABLE, UNKNOWN_HUB_ERROR_CODE, retryable=True), id="a-hub-error-with-an-unknown-code"
        ),
        pytest.param(
            HttpAnswer(
                HTTP_UNAVAILABLE,
                padded_to(hub_error_body(RETRYABLE_REFUSAL_CODE, retryable=True), HUB_ERROR_MAX_BYTES + 1),
                None,
            ),
            id="a-hub-error-above-its-byte-limit",
        ),
    ],
)
def test_claim_execution_when_a_non_200_is_not_an_accepted_hub_error_should_end_invalid_after_one_post(answer):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(answer, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_INVALID_ANSWER)
    assert len(transport.calls) == 1
    assert clock.waits == []


# --- refused ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [(HTTP_CONFLICT, "claim_conflict"), (HTTP_UNAUTHORIZED, "invalid_bootstrap_token")],
    ids=["claim-conflict", "invalid-bootstrap-token"],
)
def test_claim_execution_when_the_hub_refuses_for_good_should_end_refused_with_its_code_after_one_post(status, code):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(hub_error_answer(status, code, retryable=False), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, code)
    assert len(transport.calls) == 1
    assert clock.waits == []


# --- claiming again -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retry_after_s", "least_wait_s"),
    [
        pytest.param(None, FIRST_BACKOFF_S, id="no-retry-after"),
        pytest.param(FIRST_BACKOFF_S, FIRST_BACKOFF_S, id="retry-after-no-longer-than-the-back-off"),
        pytest.param(
            RETRY_AFTER_LONGER_THAN_THE_BACKOFF_S, RETRY_AFTER_LONGER_THAN_THE_BACKOFF_S, id="retry-after-longer-than-the-back-off"
        ),
    ],
)
def test_claim_execution_when_a_retryable_refusal_precedes_the_grant_should_wait_then_claim_again_as_the_same_runtime(
    retry_after_s, least_wait_s
):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(_retryable_refusal(retry_after_s), _granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    first, second = _sent(transport)
    assert isinstance(outcome, Grant)
    assert first["message_id"] != second["message_id"]
    assert moment(second["sent_at"]) > moment(first["sent_at"])
    assert first["claim"] == second["claim"] == {"bootstrap_token": dispatch.bootstrap_token, "runtime_id": RUNTIME_ID}
    assert len(clock.waits) == 1
    assert clock.waits[0] >= least_wait_s
    assert transport.calls[1].at - transport.calls[0].at >= timedelta(seconds=least_wait_s)


def test_claim_execution_when_no_retry_after_is_given_should_wait_the_first_back_off():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(_retryable_refusal(), _granted(), clock=clock)

    # Act
    _claim(dispatch, transport, clock)

    # Assert
    assert clock.waits == [FIRST_BACKOFF_S]


def test_claim_execution_when_a_transport_failure_precedes_the_grant_should_claim_again_and_return_the_grant():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(NO_ANSWER, _granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    first, second = _sent(transport)
    assert isinstance(outcome, Grant)
    assert first["message_id"] != second["message_id"]
    assert moment(second["sent_at"]) > moment(first["sent_at"])
    assert first["claim"] == second["claim"] == {"bootstrap_token": dispatch.bootstrap_token, "runtime_id": RUNTIME_ID}
    assert all(signing.verify(call.body, WORKER_KEY, call.signature) for call in transport.calls)


# --- giving up ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seconds_per_post", [0.0, SLOW_POST_S], ids=["failing-at-once", "timing-out"])
def test_claim_execution_when_no_claim_is_ever_answered_should_end_unreachable_within_its_patience(seconds_per_post):
    # Arrange: the bootstrap itself lasts far longer than the patience.
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock, seconds_per_post=seconds_per_post)
    assert dispatch.claim_expires_at > PATIENCE_ENDS

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) > 1
    assert [call.at for call in transport.calls if call.at >= PATIENCE_ENDS] == []
    assert sum(clock.waits) < PATIENCE_S


class _SteppedOnItsFirstPost(FakeTransport):
    """A transport nothing answers on, whose first POST coincides with a step of the wall clock."""

    def __init__(self, clock: FakeClock, step_s: float) -> None:
        super().__init__(then=NO_ANSWER, clock=clock)
        self._stepped_clock = clock
        self._step_s: float | None = step_s

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        if self._step_s is not None:
            self._stepped_clock.step_wall(self._step_s)
            self._step_s = None
        return super().post(url, body, signature, timeout_s=timeout_s, max_body_bytes=max_body_bytes)


@pytest.mark.parametrize("step_s", list(WALL_CLOCK_STEPS_S.values()), ids=list(WALL_CLOCK_STEPS_S))
def test_claim_execution_when_the_wall_clock_is_stepped_while_claiming_should_keep_its_patience(step_s):
    # Arrange: what an undisturbed clock gives; the bootstrap outlasts the patience even after a step forward.
    steady_clock = FakeClock()
    steady = FakeTransport(then=NO_ANSWER, clock=steady_clock)
    _claim(_dispatch(), steady, steady_clock)
    dispatch, clock = _dispatch(), FakeClock()
    transport = _SteppedOnItsFirstPost(clock, step_s)
    assert dispatch.claim_expires_at > PATIENCE_ENDS + timedelta(seconds=max(step_s, 0.0))

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: the step neither stretches the claiming nor cuts it short.
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert clock.waits == steady_clock.waits
    assert len(transport.calls) == len(steady.calls)
    assert clock.elapsed_s < PATIENCE_S


def test_claim_execution_when_every_claim_is_refused_for_now_should_end_refused_with_that_code():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=_retryable_refusal(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) > 1
    assert [call.at for call in transport.calls if call.at >= PATIENCE_ENDS] == []


def test_claim_execution_when_one_retryable_refusal_comes_among_transport_failures_should_end_refused_with_its_code():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    failures_before = (NO_ANSWER, NO_ANSWER)
    transport = FakeTransport(*failures_before, _retryable_refusal(code="internal_error"), then=NO_ANSWER, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: it kept claiming after the refusal, and the refusal is what it reports.
    assert outcome == Unclaimed(CAUSE_REFUSED, "internal_error")
    assert len(transport.calls) > len(failures_before) + 1


# --- never at or after the expiry ---------------------------------------------------------------


@pytest.mark.parametrize("seconds_per_post", [0.0, TWO_SECOND_POST_S], ids=["failing-at-once", "failing-after-two-seconds"])
def test_claim_execution_when_the_bootstrap_expires_in_five_seconds_should_never_claim_at_or_after_the_expiry(
    seconds_per_post,
):
    # Arrange
    dispatch, clock = _dispatch(**{"/claim/expires_at": seconds_from_now(EXPIRES_SOON_S)}), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock, seconds_per_post=seconds_per_post)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) >= 1
    assert [call.at for call in transport.calls if call.at >= dispatch.claim_expires_at] == []


@pytest.mark.parametrize("expired_s", [EXPIRED_ALREADY_S, 0], ids=["ten-seconds-ago", "this-very-instant"])
def test_claim_execution_when_the_bootstrap_has_expired_on_arrival_should_not_claim_at_all(expired_s):
    # Arrange: S6 accepts a request whose claim expired no more than the skew ago.
    dispatch, clock = _dispatch(**{"/claim/expires_at": seconds_from_now(expired_s)}), FakeClock()
    transport = FakeTransport(then=_granted(), clock=clock)
    assert dispatch.claim_expires_at <= clock()

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert transport.calls == []
    assert clock.waits == []


@pytest.mark.parametrize(
    ("expires_in_s", "retry_after_s"),
    [(None, RETRY_AFTER_AN_HOUR_S), (EXPIRES_SOON_S, EXPIRES_SOON_S)],
    ids=["longer-than-the-patience", "up-to-the-expiry"],
)
def test_claim_execution_when_the_hub_asks_to_retry_after_the_claiming_ends_should_end_refused_at_once(
    expires_in_s, retry_after_s
):
    # Arrange
    changes = {} if expires_in_s is None else {"/claim/expires_at": seconds_from_now(expires_in_s)}
    dispatch, clock = _dispatch(**changes), FakeClock()
    transport = FakeTransport(_retryable_refusal(retry_after_s), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) == 1
    assert clock.waits == []


# --- nothing of a claim or of an answer is logged -----------------------------------------------


def _slices_shown(logged: str, body: str, nameable: Sequence[str]) -> list[str]:
    """Every LEAK_SLICE_CHARS-character slice of `body` that `logged` shows, the ids a log line may name left out."""
    for value in nameable:
        logged = logged.replace(value, " ")
    slices = (body[start : start + LEAK_SLICE_CHARS] for start in range(len(body) - LEAK_SLICE_CHARS + 1))
    return [piece for piece in slices if piece in logged]


@pytest.mark.parametrize(
    ("outcomes", "then", "expected"),
    [
        pytest.param((_granted(),), None, Grant, id="granted"),
        pytest.param((hub_error_answer(HTTP_CONFLICT, "claim_conflict", retryable=False),), None, Unclaimed, id="refused"),
        pytest.param(
            (signed_answer(files.read_bytes(GRANT_ENCRYPTED), UNCONFIGURED_KEY),), None, Unclaimed, id="invalid-answer"
        ),
        pytest.param((), NO_ANSWER, Unclaimed, id="unreachable"),
        pytest.param((_retryable_refusal(), NO_ANSWER, _granted()), None, Grant, id="granted-after-retries"),
    ],
)
def test_claim_execution_when_it_ends_any_way_should_log_nothing_of_a_claim_or_of_an_answer(
    caplog, outcomes, then, expected
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(*outcomes, then=then, clock=clock)
    grant = json.loads(files.read_bytes(GRANT_ENCRYPTED))
    nameable = (dispatch.identity.dispatch_id, grant["identity"]["execution_id"])
    signatures = [answer.signature for answer in outcomes if isinstance(answer, HttpAnswer) and answer.signature]

    # Act
    outcome = _claim(dispatch, transport, clock)
    logging.getLogger(claim_module.__name__).debug(LOG_PROBE)
    logged = caplog.text

    # Assert
    signatures += [call.signature for call in transport.calls]
    bodies = [call.body.decode("ascii") for call in transport.calls] + [files.read_bytes(GRANT_ENCRYPTED).decode("ascii")]
    secrets = [dispatch.bootstrap_token, MEDIA_KEY_HEX, RUNTIME_ID, "vh_", "bootstrap_token", "key_hex", *signatures]
    assert isinstance(outcome, expected)
    assert LOG_PROBE in logged
    assert [secret for secret in secrets if secret in logged] == []
    assert [piece for body in bodies for piece in _slices_shown(logged, body, nameable)] == []
