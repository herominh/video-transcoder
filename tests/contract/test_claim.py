"""The claim exchange, the worker's side (README section 15.1): a runtime works only on an accepted, signed grant.

claim_execution() sends `transcode.claim` to the configured claim address and returns a Grant only for a
`200` whose body verifies under a Hub-to-worker key and is an accepted `transcode.claim.granted` for this
dispatch, this runtime and this encryption. Every other ending is Unclaimed: `refused` (a `hub.error`),
`unreachable` (no answer from the Hub) or `invalid_answer` (anything else, never retried). A gateway's
502, 503 or 504 that carries no accepted `hub.error` is no answer from the Hub. It claims again only after
a transport failure, such a gateway failure or a retryable `hub.error`, never at or after the dispatch's
`claim.expires_at`, never for longer than its patience, and never at or after the instant its start's
wall time ends; sealing a claim takes time, so those ends are judged again once it is sealed, and a
claim whose time ran out meanwhile is neither sent nor counted. Nothing of a claim or of an answer is
logged.

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import dataclasses
import json
import logging
import re
from datetime import datetime, timedelta
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
# A bootstrap that outlasts the first claim and not the wait before a second.
EXPIRES_BEFORE_A_SECOND_CLAIM_S = FIRST_BACKOFF_S / 2
OTHER_UUID = "9a9a9a9a-9a9a-79a9-89a9-9a9a9a9a9a9a"
UNKNOWN_HUB_ERROR_CODE = "made_up_refusal"
# A value no log line may show, placed in the error page of a server that is not the Hub.
ERROR_PAGE_CANARY = "CANARY-error-page"
HTML_BODY = b"<html><body><h1>The service is unavailable: " + ERROR_PAGE_CANARY.encode("ascii") + b"</h1></body></html>"
HTTP_CREATED = 201
HTTP_NO_CONTENT = 204
HTTP_FOUND = 302
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_INTERNAL_ERROR = 500
HTTP_NOT_IMPLEMENTED = 501
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504
HTTP_VERSION_NOT_SUPPORTED = 505
# What a gateway in front of the Hub answers while it cannot reach it.
GATEWAY_STATUSES = [HTTP_BAD_GATEWAY, HTTP_UNAVAILABLE, HTTP_GATEWAY_TIMEOUT]
GATEWAY_FAILURE = HttpAnswer(HTTP_BAD_GATEWAY, HTML_BODY, None)
NO_ANSWER = TransportFailed("ConnectTimeout")
RETRYABLE_REFUSAL_CODE = "service_unavailable"
OTHER_RETRYABLE_REFUSAL_CODE = "internal_error"
FINAL_REFUSAL_CODE = "claim_conflict"
# What the log says of a claiming that ends unreachable: no claim was sent (and for which of two reasons),
# or so many were.
NOTHING_WAS_SENT = "nothing was sent"
BOOTSTRAP_HAD_EXPIRED = "bootstrap had expired"
WALL_TIME_WAS_SPENT = "wall time was spent"
ENDED_UNREACHABLE = "ended unreachable"
# A start's wall time that ends before the claiming's patience would, and the request's own time-outs.
WALL_TIME_SHORTER_THAN_THE_PATIENCE_S = 30.0
REQUEST_TIMEOUT_S = 15.0
LEAST_REQUEST_TIMEOUT_S = 1.0
# The two ends of a claiming a test brings near, by the names of its cases.
THE_STARTS_WALL_TIME = "the-starts-wall-time"
THE_BOOTSTRAPS_EXPIRY = "the-bootstraps-expiry"
# How long sealing one claim takes on the fake clock in the tests of the second look at the clocks, and a
# time left that holds one such claim, the first wait and less than the sealing of a second claim.
SEALING_TAKES_S = 1.0
TIME_FOR_ONE_SLOWLY_SEALED_CLAIM_S = 2.8
ENDED_UNREACHABLE_AFTER = re.compile(r"ended unreachable after (\d+) claims")
# Steps of the wall clock while the Hub does not answer: set back an hour, set forward a minute.
WALL_CLOCK_STEPS_S = {"set-back-an-hour": -3600.0, "set-forward-a-minute": 60.0}
# A log line may name these two ids; any longer slice of a claim or of a grant is a leak.
LEAK_SLICE_CHARS = 16
LOG_PROBE = "probe: the capture sees this module's logger at DEBUG"
MILLISECONDS_PER_SECOND = 1000


def _request(path: str = REQUEST_ENCRYPTED) -> dict[str, Any]:
    return json.loads(files.read_bytes(path))


def _dispatch(path: str = REQUEST_ENCRYPTED, **changes: Any) -> AcceptedDispatch:
    """The accepted dispatch of a request fixture, signed by the Hub after `changes` (JSON pointers) were applied."""
    request = modified_fixture(path, changes) if changes else files.read_bytes(path)
    return accepted_dispatch(request, settings=SETTINGS)


def _claim(
    dispatch: AcceptedDispatch, transport: FakeTransport, clock: FakeClock, *, wall_time_left_s: float | None = None
) -> Grant | Unclaimed:
    """Claim as a start does that has `wall_time_left_s` of its wall time left when it claims; when none
    is given, the dispatch's whole limit, as for a start that has just begun."""
    if wall_time_left_s is None:
        wall_time_left_s = dispatch.limits.max_wall_time_ms / MILLISECONDS_PER_SECOND
    return claim_execution(
        dispatch,
        settings=SETTINGS,
        runtime_id=RUNTIME_ID,
        transport=transport,
        clock=clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        new_message_id=uuid_source(),
        wall_time_ends_at=clock.monotonic() + wall_time_left_s,
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


def _oversized_hub_error(status: int) -> HttpAnswer:
    """A retryable `hub.error` one byte above its kind's limit: well formed, and not accepted."""
    body = padded_to(hub_error_body(RETRYABLE_REFUSAL_CODE, retryable=True), HUB_ERROR_MAX_BYTES + 1)
    return HttpAnswer(status, body, None)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None), id="an-html-500"),
        pytest.param(HttpAnswer(HTTP_NOT_IMPLEMENTED, HTML_BODY, None), id="an-html-501-just-below-the-gateway-statuses"),
        pytest.param(
            HttpAnswer(HTTP_VERSION_NOT_SUPPORTED, HTML_BODY, None), id="an-html-505-just-above-the-gateway-statuses"
        ),
        pytest.param(HttpAnswer(HTTP_NOT_FOUND, HTML_BODY, None), id="an-html-404"),
        pytest.param(HttpAnswer(HTTP_BAD_REQUEST, HTML_BODY, None), id="an-html-400"),
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
            hub_error_answer(HTTP_CONFLICT, UNKNOWN_HUB_ERROR_CODE, retryable=True), id="a-hub-error-with-an-unknown-code"
        ),
        pytest.param(_oversized_hub_error(HTTP_INTERNAL_ERROR), id="a-hub-error-above-its-byte-limit"),
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


@pytest.mark.parametrize("status", GATEWAY_STATUSES)
def test_claim_execution_when_a_gateway_status_carries_a_final_hub_error_should_end_refused_after_one_post(status):
    # Arrange: the Hub itself answered, whatever the status: its refusal counts, not the status.
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(hub_error_answer(status, FINAL_REFUSAL_CODE, retryable=False), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, FINAL_REFUSAL_CODE)
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


# What a gateway answers while it cannot reach the Hub: one of its own statuses, and no accepted `hub.error`.
GATEWAY_FAILURES = [
    pytest.param(HttpAnswer(HTTP_BAD_GATEWAY, HTML_BODY, None), id="an-html-502"),
    pytest.param(HttpAnswer(HTTP_UNAVAILABLE, HTML_BODY, None), id="an-html-503"),
    pytest.param(HttpAnswer(HTTP_GATEWAY_TIMEOUT, HTML_BODY, None), id="an-html-504"),
    pytest.param(HttpAnswer(HTTP_GATEWAY_TIMEOUT, b"", None), id="an-empty-504"),
    pytest.param(
        hub_error_answer(HTTP_UNAVAILABLE, UNKNOWN_HUB_ERROR_CODE, retryable=True), id="a-503-hub-error-with-an-unknown-code"
    ),
    pytest.param(_oversized_hub_error(HTTP_UNAVAILABLE), id="a-503-hub-error-above-its-byte-limit"),
]


@pytest.mark.parametrize("gateway_failure", GATEWAY_FAILURES)
def test_claim_execution_when_a_gateway_failure_precedes_the_grant_should_wait_the_first_back_off_then_claim_again(
    gateway_failure,
):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(gateway_failure, _granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: the same runtime claims again, in a new message.
    first, second = _sent(transport)
    assert isinstance(outcome, Grant)
    assert first["message_id"] != second["message_id"]
    assert moment(second["sent_at"]) > moment(first["sent_at"])
    assert first["claim"] == second["claim"] == {"bootstrap_token": dispatch.bootstrap_token, "runtime_id": RUNTIME_ID}
    assert all(signing.verify(call.body, WORKER_KEY, call.signature) for call in transport.calls)
    assert clock.waits == [FIRST_BACKOFF_S]


@pytest.mark.parametrize("status", GATEWAY_STATUSES)
def test_claim_execution_when_a_gateway_status_carries_a_retryable_hub_error_should_wait_as_the_hub_asks_then_claim_again(
    status,
):
    # Arrange: the Hub asks for a longer wait than a failure in transport is given.
    dispatch, clock = _dispatch(), FakeClock()
    refusal = hub_error_answer(
        status, RETRYABLE_REFUSAL_CODE, retryable=True, retry_after_s=RETRY_AFTER_LONGER_THAN_THE_BACKOFF_S
    )
    transport = FakeTransport(refusal, _granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert isinstance(outcome, Grant)
    assert len(transport.calls) == 2
    assert clock.waits == [RETRY_AFTER_LONGER_THAN_THE_BACKOFF_S]


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


@pytest.mark.parametrize("gateway_failure", GATEWAY_FAILURES)
def test_claim_execution_when_every_claim_meets_a_gateway_failure_should_end_unreachable_within_its_patience(
    gateway_failure,
):
    # Arrange: the bootstrap itself lasts far longer than the patience.
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=gateway_failure, clock=clock)
    assert dispatch.claim_expires_at > PATIENCE_ENDS

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) > 1
    assert [call.at for call in transport.calls if call.at >= PATIENCE_ENDS] == []
    assert sum(clock.waits) < PATIENCE_S


def test_claim_execution_when_every_claim_meets_a_gateway_failure_should_wait_and_claim_as_after_transport_failures():
    # Arrange: what claims that get no answer at all give.
    unanswered_clock = FakeClock()
    unanswered = FakeTransport(then=NO_ANSWER, clock=unanswered_clock)
    _claim(_dispatch(), unanswered, unanswered_clock)
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=GATEWAY_FAILURE, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert clock.waits == unanswered_clock.waits
    assert [call.at for call in transport.calls] == [call.at for call in unanswered.calls]


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
    refusal = _retryable_refusal(code=OTHER_RETRYABLE_REFUSAL_CODE)
    transport = FakeTransport(*failures_before, refusal, then=NO_ANSWER, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: it kept claiming after the refusal, and the refusal is what it reports.
    assert outcome == Unclaimed(CAUSE_REFUSED, OTHER_RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) > len(failures_before) + 1


def test_claim_execution_when_one_retryable_refusal_comes_among_gateway_failures_should_end_refused_with_its_code():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    failures_before = (GATEWAY_FAILURE,)
    refusal = _retryable_refusal(code=OTHER_RETRYABLE_REFUSAL_CODE)
    transport = FakeTransport(*failures_before, refusal, then=GATEWAY_FAILURE, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: it kept claiming after the refusal, and the refusal is what it reports.
    assert outcome == Unclaimed(CAUSE_REFUSED, OTHER_RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) > len(failures_before) + 1
    assert [call.at for call in transport.calls if call.at >= PATIENCE_ENDS] == []


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


@pytest.mark.parametrize("past_the_expiry_s", [0.0, 60.0], ids=["to-the-expiry", "a-minute-past-the-expiry"])
def test_claim_execution_when_the_wall_clock_is_stepped_to_or_past_the_expiry_while_claiming_should_not_claim_again(
    past_the_expiry_s,
):
    # Arrange: the first claim gets no answer, and while it is out the wall clock is set forward. The
    # patience and the start's wall time, both counted in seconds that passed, are nowhere near their end.
    dispatch, clock = _dispatch(), FakeClock()
    until_the_expiry_s = (dispatch.claim_expires_at - clock()).total_seconds()
    transport = _SteppedOnItsFirstPost(clock, until_the_expiry_s + past_the_expiry_s)
    assert until_the_expiry_s > PATIENCE_S

    # Act
    outcome = _claim(dispatch, transport, clock)

    # Assert: the expiry is judged on the wall clock, as the contract states it
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) == 1
    assert clock.waits == []
    assert clock() >= dispatch.claim_expires_at
    assert clock.elapsed_s < PATIENCE_S


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


# --- never at or after the end of the start's wall time -----------------------------------------


def _wall_time_ends(clock: FakeClock, seconds: float = WALL_TIME_SHORTER_THAN_THE_PATIENCE_S) -> datetime:
    """The instant, on the test's wall clock, `seconds` after the clock began (nothing steps it in these tests)."""
    return clock.start + timedelta(seconds=seconds)


@pytest.mark.parametrize("seconds_per_post", [0.0, SLOW_POST_S], ids=["failing-at-once", "timing-out"])
def test_claim_execution_when_the_wall_time_ends_before_its_patience_should_stop_claiming_there_and_end_unreachable(
    seconds_per_post,
):
    # Arrange: the bootstrap and the patience both outlast the 30 s the start has
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock, seconds_per_post=seconds_per_post)
    assert WALL_TIME_SHORTER_THAN_THE_PATIENCE_S < PATIENCE_S
    assert dispatch.claim_expires_at > PATIENCE_ENDS

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) > 1
    assert [call.at for call in transport.calls if call.at >= _wall_time_ends(clock)] == []
    assert sum(clock.waits) < WALL_TIME_SHORTER_THAN_THE_PATIENCE_S


def test_claim_execution_when_the_wall_time_ends_while_it_is_refused_for_now_should_end_refused_with_that_code():
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=_retryable_refusal(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) > 1
    assert [call.at for call in transport.calls if call.at >= _wall_time_ends(clock)] == []


def test_claim_execution_when_the_hub_asks_to_retry_after_the_wall_time_ends_should_end_refused_at_once():
    # Arrange: the Hub asks for exactly the time the start has left
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(_retryable_refusal(int(WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) == 1
    assert clock.waits == []


def test_claim_execution_when_the_grant_comes_before_the_wall_time_ends_should_return_it():
    # Arrange: one claim is lost, the next is granted, well inside the 30 s
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(NO_ANSWER, _granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert
    assert isinstance(outcome, Grant)
    assert len(transport.calls) == 2


@pytest.mark.parametrize("left_s", [0.0, -10.0], ids=["this-very-instant", "ten-seconds-ago"])
def test_claim_execution_when_the_wall_time_is_already_spent_should_not_claim_at_all(left_s):
    # Arrange: a Hub that would grant, and a bootstrap that has not expired
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=_granted(), clock=clock)
    assert dispatch.claim_expires_at > clock()

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=left_s)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert transport.calls == []
    assert clock.waits == []


@pytest.mark.parametrize(
    ("left_s", "timeout_s"),
    [
        pytest.param(3600.0, REQUEST_TIMEOUT_S, id="far-more-left-than-the-requests-time-out"),
        pytest.param(6.0, 6.0, id="less-left-than-the-requests-time-out"),
        pytest.param(0.4, LEAST_REQUEST_TIMEOUT_S, id="less-left-than-the-least-time-out"),
    ],
)
def test_claim_execution_when_the_wall_time_ends_soon_should_give_a_claim_the_time_left_and_never_below_one_second(
    left_s, timeout_s
):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(_granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=left_s)

    # Assert: a claim sent before the end is given at least a second, as one sent just before the expiry is
    (call,) = transport.calls
    assert isinstance(outcome, Grant)
    assert call.timeout_s == timeout_s
    assert (REQUEST_TIMEOUT_S, LEAST_REQUEST_TIMEOUT_S) == (15.0, 1.0)


def test_claim_execution_when_the_wall_time_ends_after_its_patience_should_claim_exactly_as_with_a_far_one():
    # Arrange: what a start with an hour of wall time gives; this one's ends a second after the patience
    far_clock = FakeClock()
    far = FakeTransport(then=NO_ANSWER, clock=far_clock)
    _claim(_dispatch(), far, far_clock, wall_time_left_s=3600.0)
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=PATIENCE_S + 1.0)

    # Assert: the patience ends the claiming, as before
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert clock.waits == far_clock.waits
    assert [call.at for call in transport.calls] == [call.at for call in far.calls]
    assert [call.timeout_s for call in transport.calls] == [call.timeout_s for call in far.calls]
    assert clock.elapsed_s < PATIENCE_S


def test_claim_execution_when_the_bootstrap_expires_before_the_wall_time_ends_should_still_stop_at_the_expiry():
    # Arrange: the bootstrap expires in 5 s, the start has 30 s
    dispatch, clock = _dispatch(**{"/claim/expires_at": seconds_from_now(EXPIRES_SOON_S)}), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) >= 1
    assert [call.at for call in transport.calls if call.at >= dispatch.claim_expires_at] == []


@pytest.mark.parametrize("step_s", list(WALL_CLOCK_STEPS_S.values()), ids=list(WALL_CLOCK_STEPS_S))
def test_claim_execution_when_the_wall_clock_is_stepped_while_claiming_should_keep_to_the_starts_wall_time(step_s):
    # Arrange: what an undisturbed clock gives; the bootstrap outlasts the 30 s even after a step forward
    steady_clock = FakeClock()
    steady = FakeTransport(then=NO_ANSWER, clock=steady_clock)
    _claim(_dispatch(), steady, steady_clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)
    dispatch, clock = _dispatch(), FakeClock()
    transport = _SteppedOnItsFirstPost(clock, step_s)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=WALL_TIME_SHORTER_THAN_THE_PATIENCE_S)

    # Assert: the wall time is counted in seconds that passed, whatever the wall clock was set to
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert clock.waits == steady_clock.waits
    assert len(transport.calls) == len(steady.calls)
    assert clock.elapsed_s < WALL_TIME_SHORTER_THAN_THE_PATIENCE_S


@pytest.mark.parametrize(
    "wall_time_ends_at",
    [
        pytest.param(float("nan"), id="not-a-number"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("-inf"), id="minus-infinite"),
        pytest.param(True, id="a-bool"),
        pytest.param("30.0", id="text"),
        pytest.param(None, id="none"),
    ],
)
def test_claim_execution_when_the_wall_times_end_is_no_finite_number_should_raise_value_error_without_claiming(
    wall_time_ends_at,
):
    # Arrange
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=_granted(), clock=clock)

    # Act / Assert
    with pytest.raises(ValueError):
        claim_execution(
            dispatch,
            settings=SETTINGS,
            runtime_id=RUNTIME_ID,
            transport=transport,
            clock=clock,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            new_message_id=uuid_source(),
            wall_time_ends_at=wall_time_ends_at,
        )
    assert transport.calls == []


# --- the time left is read again once a claim is sealed, immediately before it is posted --------


def _make_sealing_take(monkeypatch: pytest.MonkeyPatch, clock: FakeClock, seconds: float = SEALING_TAKES_S) -> None:
    """Make every claim take `seconds` of the test's clock to seal: the time that passes, between the
    claiming's look at its clocks and the POST, while a claim is built, validated and signed."""
    seal = claim_module.seal_message

    def seal_slowly(*arguments: Any, **named_arguments: Any) -> Any:
        sealed = seal(*arguments, **named_arguments)
        clock.advance(seconds)
        return sealed

    monkeypatch.setattr(claim_module, "seal_message", seal_slowly)


def _claim_with_one_end_near(
    end: str, left_s: float, transport: FakeTransport, clock: FakeClock
) -> Grant | Unclaimed:
    """Claim `left_s` seconds before one end of the claiming, the other one far away."""
    if end == THE_BOOTSTRAPS_EXPIRY:
        return _claim(_dispatch(**{"/claim/expires_at": seconds_from_now(left_s)}), transport, clock)
    return _claim(_dispatch(), transport, clock, wall_time_left_s=left_s)


@pytest.mark.parametrize("end", [THE_STARTS_WALL_TIME, THE_BOOTSTRAPS_EXPIRY])
@pytest.mark.parametrize(
    "left_s",
    [
        pytest.param(SEALING_TAKES_S / 2, id="the-end-passes-while-the-claim-is-sealed"),
        pytest.param(SEALING_TAKES_S, id="the-end-is-reached-just-as-the-claim-is-sealed"),
    ],
)
def test_claim_execution_when_an_end_is_reached_while_the_first_claim_is_sealed_should_send_nothing(
    monkeypatch, caplog, end, left_s
):
    # Arrange: time is left when the claim is built, none when it is sealed; the Hub would grant
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    clock = FakeClock()
    _make_sealing_take(monkeypatch, clock)
    transport = FakeTransport(then=_granted(), clock=clock)

    # Act
    outcome = _claim_with_one_end_near(end, left_s, transport, clock)
    logged = _claim_log(caplog)

    # Assert: nothing is posted at or after the end, and the log names the end, not a network fault
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert transport.calls == []
    assert clock.waits == []
    assert NOTHING_WAS_SENT in logged and ENDED_UNREACHABLE not in logged
    assert (WALL_TIME_WAS_SPENT in logged) == (end == THE_STARTS_WALL_TIME)
    assert (BOOTSTRAP_HAD_EXPIRED in logged) == (end == THE_BOOTSTRAPS_EXPIRY)


@pytest.mark.parametrize("end", [THE_STARTS_WALL_TIME, THE_BOOTSTRAPS_EXPIRY])
@pytest.mark.parametrize("unanswered", [NO_ANSWER, GATEWAY_FAILURE], ids=["no-answer", "a-gateway-failure"])
def test_claim_execution_when_an_end_is_reached_while_a_later_claim_is_sealed_should_neither_send_nor_count_it(
    monkeypatch, caplog, end, unanswered
):
    # Arrange: sealing takes 1 s and 2.8 s are left: the first claim is posted with 1.8 s left and gets no
    # answer; after the first wait 0.8 s are left, and they pass while the second claim is sealed
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    clock = FakeClock()
    _make_sealing_take(monkeypatch, clock)
    transport = FakeTransport(unanswered, then=_granted(), clock=clock)

    # Act
    outcome = _claim_with_one_end_near(end, TIME_FOR_ONE_SLOWLY_SEALED_CLAIM_S, transport, clock)
    logged = _claim_log(caplog)

    # Assert: one claim was sent, and one is what the log counts
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) == 1
    assert clock.waits == [FIRST_BACKOFF_S]
    assert ENDED_UNREACHABLE_AFTER.findall(logged) == ["1"]
    assert NOTHING_WAS_SENT not in logged


@pytest.mark.parametrize("end", [THE_STARTS_WALL_TIME, THE_BOOTSTRAPS_EXPIRY])
def test_claim_execution_when_an_end_is_reached_while_a_claim_is_sealed_after_a_refusal_for_now_should_end_refused(
    monkeypatch, end
):
    # Arrange: as above, with a Hub that refuses the one claim it gets for now
    clock = FakeClock()
    _make_sealing_take(monkeypatch, clock)
    transport = FakeTransport(_retryable_refusal(), then=_granted(), clock=clock)

    # Act
    outcome = _claim_with_one_end_near(end, TIME_FOR_ONE_SLOWLY_SEALED_CLAIM_S, transport, clock)

    # Assert
    assert outcome == Unclaimed(CAUSE_REFUSED, RETRYABLE_REFUSAL_CODE)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("end", [THE_STARTS_WALL_TIME, THE_BOOTSTRAPS_EXPIRY])
@pytest.mark.parametrize(
    ("left_s", "timeout_s"),
    [
        pytest.param(7.0, 6.0, id="six-seconds-left-once-sealed"),
        pytest.param(1.4, LEAST_REQUEST_TIMEOUT_S, id="less-than-the-least-time-out-left-once-sealed"),
    ],
)
def test_claim_execution_when_sealing_took_time_should_give_the_claim_what_is_left_once_it_is_sealed(
    monkeypatch, end, left_s, timeout_s
):
    # Arrange: sealing takes 1 s of the time that is left when the claim is built
    clock = FakeClock()
    _make_sealing_take(monkeypatch, clock)
    transport = FakeTransport(_granted(), clock=clock)

    # Act
    outcome = _claim_with_one_end_near(end, left_s, transport, clock)

    # Assert: posted before the end, with the time then left as its time-out, a second at least
    (call,) = transport.calls
    assert isinstance(outcome, Grant)
    assert call.at == clock.start + timedelta(seconds=SEALING_TAKES_S)
    assert call.timeout_s == timeout_s


# --- what the log says of an unreachable ending -------------------------------------------------


def _claim_log(caplog: pytest.LogCaptureFixture) -> str:
    """What the claim module logged, one line per message."""
    return "\n".join(record.getMessage() for record in caplog.records if record.name == claim_module.__name__)


@pytest.mark.parametrize("expired_s", [EXPIRED_ALREADY_S, 0], ids=["ten-seconds-ago", "this-very-instant"])
def test_claim_execution_when_the_bootstrap_has_expired_on_arrival_should_log_that_nothing_was_sent(caplog, expired_s):
    # Arrange
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    dispatch, clock = _dispatch(**{"/claim/expires_at": seconds_from_now(expired_s)}), FakeClock()
    transport = FakeTransport(then=_granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)
    logged = _claim_log(caplog)

    # Assert: the cause is the contract's, and the log tells a late start from a Hub that did not answer.
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert transport.calls == []
    assert NOTHING_WAS_SENT in logged
    assert ENDED_UNREACHABLE not in logged
    assert BOOTSTRAP_HAD_EXPIRED in logged
    assert WALL_TIME_WAS_SPENT not in logged


@pytest.mark.parametrize("left_s", [0.0, -10.0], ids=["this-very-instant", "ten-seconds-ago"])
def test_claim_execution_when_the_wall_time_is_already_spent_should_log_that_nothing_was_sent_and_why(
    caplog, left_s
):
    # Arrange: the bootstrap has not expired; only the start's own time is gone
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=_granted(), clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock, wall_time_left_s=left_s)
    logged = _claim_log(caplog)

    # Assert: one cause in the contract, and a log line that tells this start from a late one
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert transport.calls == []
    assert NOTHING_WAS_SENT in logged
    assert WALL_TIME_WAS_SPENT in logged
    assert BOOTSTRAP_HAD_EXPIRED not in logged
    assert ENDED_UNREACHABLE not in logged


@pytest.mark.parametrize("unanswered", [NO_ANSWER, GATEWAY_FAILURE], ids=["no-answer", "gateway-failures"])
def test_claim_execution_when_none_of_its_claims_was_answered_should_log_unreachable_with_how_many_it_sent(
    caplog, unanswered
):
    # Arrange
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    dispatch, clock = _dispatch(), FakeClock()
    transport = FakeTransport(then=unanswered, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)
    logged = _claim_log(caplog)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) > 1
    assert ENDED_UNREACHABLE_AFTER.findall(logged) == [str(len(transport.calls))]
    assert NOTHING_WAS_SENT not in logged


def test_claim_execution_when_its_one_claim_was_not_answered_should_log_unreachable_after_one_claim(caplog):
    # Arrange: the bootstrap expires before a second claim could be sent.
    caplog.set_level(logging.INFO, logger=claim_module.__name__)
    changes = {"/claim/expires_at": seconds_from_now(EXPIRES_BEFORE_A_SECOND_CLAIM_S)}
    dispatch, clock = _dispatch(**changes), FakeClock()
    transport = FakeTransport(then=NO_ANSWER, clock=clock)

    # Act
    outcome = _claim(dispatch, transport, clock)
    logged = _claim_log(caplog)

    # Assert
    assert outcome == Unclaimed(CAUSE_UNREACHABLE)
    assert len(transport.calls) == 1
    assert ENDED_UNREACHABLE_AFTER.findall(logged) == ["1"]
    assert NOTHING_WAS_SENT not in logged


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
        pytest.param((GATEWAY_FAILURE, _granted()), None, Grant, id="granted-after-a-gateway-failure"),
        pytest.param((), GATEWAY_FAILURE, Unclaimed, id="gateway-failures-until-the-end"),
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
    bodies = [call.body.decode("ascii") for call in transport.calls]
    bodies += [files.read_bytes(GRANT_ENCRYPTED).decode("ascii"), HTML_BODY.decode("ascii")]
    secrets = [dispatch.bootstrap_token, MEDIA_KEY_HEX, RUNTIME_ID, "vh_", "bootstrap_token", "key_hex", *signatures]
    secrets += [ERROR_PAGE_CANARY]
    assert isinstance(outcome, expected)
    assert LOG_PROBE in logged
    assert [secret for secret in secrets if secret in logged] == []
    assert [piece for body in bodies for piece in _slices_shown(logged, body, nameable)] == []
