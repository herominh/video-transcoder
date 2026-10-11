"""One start of the worker for one dispatch: accept, claim, then work or report, and always return the job output.

handle() refuses a dispatch it cannot accept (DispatchRefused, nothing sent, no work). Granted, it hands
the claimed execution to the job runner exactly once; unclaimed, it does no work and reports
`transcode.unclaimed`. Either message is the job's output and is also posted to the callback address,
the same signed bytes both ways, and no failure of the callback costs the output. Every start draws its
own runtime id. load_runtime() checks the contract tree before it reads the configuration.

One start has one wall-time budget, counted from the moment handle() begins: the claiming ends by it,
the job is told when it ends, and the callback is not attempted once it is spent (the output still is
returned).

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import dataclasses
import json
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pytest

from core.failure import Failure
from core.protocol import files, handler, messages, pipeline, signing
from core.protocol.claim import Grant
from core.protocol.dispatch import DispatchRefused
from core.protocol.files import ContractTreeInvalid
from core.protocol.handler import ClaimedExecution, Runtime, handle, load_runtime
from core.protocol.messages import ExecutionIdentity
from core.protocol.outbox import SealedMessage, seal_message
from core.protocol.registry import MessageKind
from core.protocol.settings import SettingsInvalid, WorkerSettings
from core.protocol.transport import HttpAnswer, HttpsTransport, TransportFailed
from tests.contract.v2_support import (
    CALLBACK_URL,
    CLAIM_URL,
    GRANT_ENCRYPTED,
    HUB_KEY,
    HUB_UNCLAIMED_POLL_CONTEXT,
    MEDIA_KEY_HEX,
    REQUEST_ENCRYPTED,
    RUNTIME_ID,
    UNCONFIGURED_KEY,
    WORKER_KEY,
    WORKER_KEY_ID,
    FakeClock,
    FakeTransport,
    Outcome,
    accepted_dispatch,
    catalog_context,
    hub_error_answer,
    modified_fixture,
    signed_answer,
    signed_carrier,
    uuid_source,
    worker_env,
    worker_settings,
)

SETTINGS = worker_settings()
UNCLAIMED_KIND = "transcode.unclaimed"
CAUSE_REFUSED = "refused"
CAUSE_UNREACHABLE = "unreachable"
CAUSE_INVALID_ANSWER = "invalid_answer"
HUB_ERROR_CODE_MEMBER = "hub_error_code"
REFUSAL_CODE = "claim_conflict"
HTTP_OK = 200
HTTP_CONFLICT = 409
# Not a gateway's 502, 503 or 504, which claiming treats as no answer from the Hub and claims again after.
HTTP_INTERNAL_ERROR = 500
HTML_BODY = b"<html><body><h1>500 Internal Server Error</h1></body></html>"
DELIVERED = HttpAnswer(HTTP_OK, b"", None)
NO_ANSWER = TransportFailed("ConnectTimeout")
CLAIM_REFUSED = hub_error_answer(HTTP_CONFLICT, REFUSAL_CODE, retryable=False)
RESULT_MESSAGE_ID = "99999999-9999-7999-8999-0000000000f1"
# The failure the stand-in job reports: it does no work, it only proves what the handler handed it.
STAND_IN_FAILURE = Failure(
    error_class="internal_error", code="stand_in_job", retryable=False, detail="The stand-in job of the test did no work."
)
LOWERCASE_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
REMOVED_VARIABLE = "VH_CLAIM_URL"
HTTP_UNAVAILABLE = 503
RETRYABLE_REFUSAL_CODE = "service_unavailable"
WALL_TIME_LIMIT_POINTER = "/limits/max_wall_time_ms"
# A wall-time limit of one start, shorter than the two minutes a claiming would otherwise last.
WALL_TIME_MS = 90_000
WALL_TIME_S = 90.0
SHORT_WALL_TIME_MS = 30_000
SHORT_WALL_TIME_S = 30.0
# How long the process has been up when a start begins: its monotonic clock does not read zero.
UPTIME_S = 1234.5
CLAIM_TOOK_S = 7
ACCEPTING_TOOK_S = 2.0
# The callback's own time-out, and how long a POST that times out takes on the fake clock.
CALLBACK_TIMEOUT_S = 10.0
SLOW_POST_S = 15.0


class StandInJob:
    """A job runner that records what it was handed and returns a sealed `transcode.result.failed`."""

    def __init__(self) -> None:
        self.calls: list[tuple[ClaimedExecution, Runtime]] = []
        self.returned: list[SealedMessage] = []
        # What the job does besides, once it has been handed its execution (a test makes it take time).
        self.work: Callable[[], None] = lambda: None

    def __call__(self, execution: ClaimedExecution, runtime: Runtime) -> SealedMessage:
        self.calls.append((execution, runtime))
        self.work()
        now = runtime.clock()
        identity: ExecutionIdentity = execution.grant.identity
        message = messages.result_failed(
            protocol_version=execution.dispatch.protocol_version,
            message_id=RESULT_MESSAGE_ID,
            sent_at=now,
            key_id=runtime.settings.worker_to_hub_key_id,
            identity=identity,
            failure=STAND_IN_FAILURE,
            stage=None,
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_RESULT_FAILED,
            identity,
            settings=runtime.settings,
            now=now,
            source_id=execution.dispatch.source.source_id,
        )
        sealed = seal_message(message, context, settings=runtime.settings)
        self.returned.append(sealed)
        return sealed


def _request() -> dict[str, Any]:
    return json.loads(files.read_bytes(REQUEST_ENCRYPTED))


def _job_input(key: bytes = HUB_KEY) -> dict[str, str]:
    return signed_carrier(files.read_bytes(REQUEST_ENCRYPTED), key)


def _job_input_with_wall_time(milliseconds: int) -> dict[str, str]:
    """The fixtures' dispatch with another wall-time limit, signed by the Hub after the change."""
    return signed_carrier(modified_fixture(REQUEST_ENCRYPTED, {WALL_TIME_LIMIT_POINTER: milliseconds}), HUB_KEY)


def _clock_of_a_process_up_for_a_while() -> FakeClock:
    """A clock whose monotonic reading is UPTIME_S while its wall clock still shows the fixtures' now."""
    clock = FakeClock()
    clock.advance(UPTIME_S)
    clock.step_wall(-UPTIME_S)
    return clock


class SlowFirstReading:
    """A wall clock whose first reading takes `seconds` of the monotonic clock and shows the same time:
    handle() reads the wall clock first to accept the dispatch, and accepting takes time."""

    def __init__(self, clock: FakeClock, seconds: float) -> None:
        self._clock = clock
        self._seconds: Optional[float] = seconds

    def __call__(self) -> datetime:
        if self._seconds is not None:
            self._clock.advance(self._seconds)
            self._clock.step_wall(-self._seconds)
            self._seconds = None
        return self._clock()


def _granted() -> HttpAnswer:
    return signed_answer(files.read_bytes(GRANT_ENCRYPTED), HUB_KEY)


def _transport(clock: FakeClock, *claim_answers: Outcome, claim_then: Optional[Outcome] = None) -> FakeTransport:
    """A transport whose claim address answers as given and whose callback address accepts every POST."""
    transport = FakeTransport(clock=clock)
    transport.route(CLAIM_URL, *claim_answers, then=claim_then)
    transport.route(CALLBACK_URL, then=DELIVERED)
    return transport


def _runtime(transport: FakeTransport, clock: FakeClock) -> Runtime:
    """A runtime whose id source gives the fixtures' runtime id first: handle() draws the runtime id before any message id."""
    return Runtime(
        settings=SETTINGS, transport=transport, clock=clock, monotonic=clock.monotonic, sleep=clock.sleep, new_uuid=uuid_source(first=RUNTIME_ID)
    )


def _claimed_runtime_ids(transport: FakeTransport) -> list[str]:
    return [json.loads(call.body)["claim"]["runtime_id"] for call in transport.calls_to(CLAIM_URL)]


# --- granted ------------------------------------------------------------------------------------


def test_handle_when_the_claim_is_granted_should_run_the_job_once_with_the_dispatch_the_grant_and_the_runtime_id():
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, _granted())
    runtime, job = _runtime(transport, clock), StandInJob()
    grant = json.loads(files.read_bytes(GRANT_ENCRYPTED))

    # Act
    handle(_job_input(), runtime=runtime, run_job=job)

    # Assert
    ((execution, handed_runtime),) = job.calls
    assert execution.dispatch == accepted_dispatch(files.read_bytes(REQUEST_ENCRYPTED), settings=SETTINGS)
    assert isinstance(execution.grant, Grant)
    assert execution.grant.identity.dispatch_id == _request()["identity"]["dispatch_id"]
    assert execution.grant.identity.execution_id == grant["identity"]["execution_id"]
    assert execution.grant.generation_id == grant["generation_id"]
    assert execution.grant.media_key == bytes.fromhex(MEDIA_KEY_HEX)
    assert execution.runtime_id == RUNTIME_ID
    assert _claimed_runtime_ids(transport) == [RUNTIME_ID]
    assert handed_runtime is runtime


def test_handle_when_the_claim_is_granted_should_return_the_jobs_message_and_post_the_same_bytes_to_the_callback():
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, _granted())
    job = StandInJob()

    # Act
    output = handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)

    # Assert
    (sealed,) = job.returned
    claim_call, callback_call = transport.calls
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert claim_call.url == CLAIM_URL
    assert (callback_call.url, callback_call.body, callback_call.signature) == (CALLBACK_URL, sealed.body, sealed.signature)


# --- unclaimed ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim_answers", "claim_then", "cause", "hub_error_code"),
    [
        pytest.param((CLAIM_REFUSED,), None, CAUSE_REFUSED, REFUSAL_CODE, id="refused"),
        pytest.param((), NO_ANSWER, CAUSE_UNREACHABLE, None, id="unreachable"),
        pytest.param((HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None),), None, CAUSE_INVALID_ANSWER, None, id="invalid-answer"),
    ],
)
def test_handle_when_the_claim_is_not_granted_should_do_no_work_and_report_unclaimed_both_ways(
    claim_answers, claim_then, cause, hub_error_code
):
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, *claim_answers, claim_then=claim_then)
    job = StandInJob()

    # Act
    output = handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)

    # Assert
    body = output["vh_message"].encode("ascii")
    report = json.loads(body)
    hub_context = catalog_context(HUB_UNCLAIMED_POLL_CONTEXT, now=clock(), known_key_ids=[WORKER_KEY_ID])
    assert job.calls == []
    assert set(output) == {"vh_message", "vh_signature"}
    assert pipeline.validate(body, hub_context).accepted
    assert (report["message_kind"], report["cause"]) == (UNCLAIMED_KIND, cause)
    assert report.get(HUB_ERROR_CODE_MEMBER) == hub_error_code
    assert (HUB_ERROR_CODE_MEMBER in report) == (hub_error_code is not None)
    assert report["identity"] == _request()["identity"]
    assert report["runtime_id"] == RUNTIME_ID
    assert _claimed_runtime_ids(transport)[:1] == [RUNTIME_ID]
    assert signing.verify(body, WORKER_KEY, output["vh_signature"])
    assert [(call.body, call.signature) for call in transport.calls_to(CALLBACK_URL)] == [(body, output["vh_signature"])]
    assert transport.calls[-1].url == CALLBACK_URL


# --- a refused dispatch -------------------------------------------------------------------------


def test_handle_when_the_dispatch_is_refused_should_raise_and_neither_post_nor_work():
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, _granted())
    job = StandInJob()

    # Act / Assert
    with pytest.raises(DispatchRefused):
        handle(_job_input(key=UNCONFIGURED_KEY), runtime=_runtime(transport, clock), run_job=job)
    assert transport.calls == []
    assert job.calls == []


# --- the callback never costs the output --------------------------------------------------------


@pytest.mark.parametrize(
    "callback_failure",
    [NO_ANSWER, RuntimeError("the callback transport broke")],
    ids=["no-answer-every-time", "an-unexpected-error"],
)
def test_handle_when_the_callback_fails_should_still_return_the_job_output(callback_failure):
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, _granted())
    transport.route(CALLBACK_URL, then=callback_failure)
    job = StandInJob()

    # Act
    output = handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)

    # Assert
    (sealed,) = job.returned
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert len(transport.calls_to(CALLBACK_URL)) >= 1


def test_handle_when_the_callback_fails_for_an_unclaimed_report_should_still_return_the_report():
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, CLAIM_REFUSED)
    transport.route(CALLBACK_URL, then=RuntimeError("the callback transport broke"))

    # Act
    output = handle(_job_input(), runtime=_runtime(transport, clock), run_job=StandInJob())

    # Assert
    assert json.loads(output["vh_message"])["cause"] == CAUSE_REFUSED
    assert len(transport.calls_to(CALLBACK_URL)) >= 1


# --- one wall-time budget for the whole start ---------------------------------------------------


def test_handle_when_the_claim_is_granted_at_once_should_tell_the_job_its_wall_time_ends_one_limit_after_the_start():
    # Arrange: the start begins when the process's monotonic clock reads UPTIME_S
    clock = _clock_of_a_process_up_for_a_while()
    transport = _transport(clock, _granted())
    job = StandInJob()

    # Act
    handle(_job_input_with_wall_time(WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert
    ((execution, _),) = job.calls
    assert execution.dispatch.limits.max_wall_time_ms == WALL_TIME_MS
    assert execution.wall_time_ends_at == UPTIME_S + WALL_TIME_S


def test_handle_when_accepting_the_dispatch_took_time_should_count_the_wall_time_from_before_it():
    # Arrange: accepting the dispatch takes 2 s of the start
    clock = _clock_of_a_process_up_for_a_while()
    transport = _transport(clock, _granted())
    runtime = dataclasses.replace(_runtime(transport, clock), clock=SlowFirstReading(clock, ACCEPTING_TOOK_S))
    job = StandInJob()

    # Act
    handle(_job_input_with_wall_time(WALL_TIME_MS), runtime=runtime, run_job=job)

    # Assert: the limit counts from the moment the start began, not from the accepted dispatch
    ((execution, _),) = job.calls
    assert clock.monotonic() == UPTIME_S + ACCEPTING_TOOK_S
    assert execution.wall_time_ends_at == UPTIME_S + WALL_TIME_S


def test_handle_when_the_claim_took_time_should_still_count_the_jobs_wall_time_from_the_start_of_the_start():
    # Arrange: the Hub refuses the first claim for 7 s, then grants: the claim takes 7 s of the start
    clock = _clock_of_a_process_up_for_a_while()
    wait_then_grant = hub_error_answer(
        HTTP_UNAVAILABLE, RETRYABLE_REFUSAL_CODE, retryable=True, retry_after_s=CLAIM_TOOK_S
    )
    transport = _transport(clock, wait_then_grant, _granted())
    read_when_the_job_began: list[float] = []
    job = StandInJob()
    job.work = lambda: read_when_the_job_began.append(clock.monotonic())

    # Act
    handle(_job_input_with_wall_time(WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert: the job begins 7 s into the start and is left the limit less those 7 s
    ((execution, _),) = job.calls
    assert read_when_the_job_began == [UPTIME_S + CLAIM_TOOK_S]
    assert execution.wall_time_ends_at == UPTIME_S + WALL_TIME_S
    assert execution.wall_time_ends_at - read_when_the_job_began[0] == WALL_TIME_S - CLAIM_TOOK_S


def test_handle_when_no_claim_is_answered_should_stop_claiming_when_the_starts_wall_time_ends():
    # Arrange: a start of 30 s; a claiming's own patience would last two minutes
    clock = _clock_of_a_process_up_for_a_while()
    transport = _transport(clock, claim_then=NO_ANSWER)
    job = StandInJob()
    wall_time_ends = clock() + timedelta(seconds=SHORT_WALL_TIME_S)

    # Act
    output = handle(_job_input_with_wall_time(SHORT_WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert
    claims = transport.calls_to(CLAIM_URL)
    assert json.loads(output["vh_message"])["cause"] == CAUSE_UNREACHABLE
    assert job.calls == []
    assert len(claims) > 1
    assert [call.at for call in claims if call.at >= wall_time_ends] == []
    assert clock.elapsed_s - UPTIME_S < SHORT_WALL_TIME_S


@pytest.mark.parametrize(
    "job_takes_s",
    [
        pytest.param(WALL_TIME_S, id="the-whole-wall-time"),
        pytest.param(WALL_TIME_S + 45.0, id="longer-than-the-wall-time"),
    ],
)
def test_handle_when_the_job_spent_the_starts_wall_time_should_return_the_output_without_attempting_the_callback(
    job_takes_s,
):
    # Arrange: a Hub that would take the callback at once
    clock = FakeClock()
    transport = _transport(clock, _granted())
    job = StandInJob()
    job.work = lambda: clock.advance(job_takes_s)

    # Act
    output = handle(_job_input_with_wall_time(WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert: the job output carries the same signed bytes, so nothing is lost but the fast path
    (sealed,) = job.returned
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert transport.calls_to(CALLBACK_URL) == []
    assert clock.waits == []


def test_handle_when_little_of_the_wall_time_is_left_after_the_job_should_give_the_callback_only_what_is_left():
    # Arrange: the job leaves 4 s of the start's 90 s
    clock = FakeClock()
    transport = _transport(clock, _granted())
    job = StandInJob()
    job.work = lambda: clock.advance(WALL_TIME_S - 4.0)

    # Act
    output = handle(_job_input_with_wall_time(WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert: posted, with a time-out shorter than the callback's own
    (sealed,) = job.returned
    (callback_call,) = transport.calls_to(CALLBACK_URL)
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert (callback_call.body, callback_call.signature) == (sealed.body, sealed.signature)
    assert callback_call.timeout_s == 4.0
    assert callback_call.timeout_s < CALLBACK_TIMEOUT_S


def test_handle_when_claiming_spent_the_starts_wall_time_should_return_the_unclaimed_report_without_a_callback():
    # Arrange: a start of 30 s on which every POST times out after 15 s
    clock = FakeClock()
    transport = FakeTransport(clock=clock, seconds_per_post=SLOW_POST_S)
    transport.route(CLAIM_URL, then=NO_ANSWER)
    transport.route(CALLBACK_URL, then=DELIVERED)
    job = StandInJob()

    # Act
    output = handle(_job_input_with_wall_time(SHORT_WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert: the report is the job's output; the callback would have started after the start's end
    body = output["vh_message"].encode("ascii")
    report = json.loads(body)
    assert (report["message_kind"], report["cause"]) == (UNCLAIMED_KIND, CAUSE_UNREACHABLE)
    assert signing.verify(body, WORKER_KEY, output["vh_signature"])
    assert job.calls == []
    assert clock.elapsed_s >= SHORT_WALL_TIME_S
    assert transport.calls_to(CALLBACK_URL) == []


# --- the runtime id -----------------------------------------------------------------------------


def test_handle_when_the_same_dispatch_is_started_twice_should_claim_with_two_different_random_runtime_ids():
    # Arrange: the process's real id source, as load_runtime() wires it.
    clock = FakeClock()
    transport = _transport(clock, claim_then=CLAIM_REFUSED)
    runtime = Runtime(
        settings=SETTINGS, transport=transport, clock=clock, monotonic=clock.monotonic, sleep=clock.sleep, new_uuid=handler._random_uuid
    )
    job_input = _job_input()

    # Act
    first_output = handle(job_input, runtime=runtime, run_job=StandInJob())
    second_output = handle(job_input, runtime=runtime, run_job=StandInJob())

    # Assert
    first_id, second_id = _claimed_runtime_ids(transport)
    assert first_id != second_id
    assert LOWERCASE_UUID.fullmatch(first_id) is not None
    assert LOWERCASE_UUID.fullmatch(second_id) is not None
    assert [json.loads(output["vh_message"])["runtime_id"] for output in (first_output, second_output)] == [first_id, second_id]


def test_claimed_execution_when_shown_with_repr_should_show_neither_runtime_id_token_nor_media_key():
    # Arrange
    clock = FakeClock()
    job = StandInJob()
    handle(_job_input(), runtime=_runtime(_transport(clock, _granted()), clock), run_job=job)
    ((execution, _),) = job.calls
    key = bytes.fromhex(MEDIA_KEY_HEX)
    hidden = [RUNTIME_ID, execution.dispatch.bootstrap_token, MEDIA_KEY_HEX, repr(key), repr(key)[2:-1]]

    # Act
    shown = repr(execution) + str(execution)

    # Assert
    assert execution.runtime_id == RUNTIME_ID
    assert execution.grant.media_key == key
    assert execution.grant.generation_id in shown
    assert [value for value in hidden if value in shown] == []


# --- the process's runtime ----------------------------------------------------------------------


def test_load_runtime_when_the_environment_is_valid_should_return_its_settings_and_the_real_transport():
    # Arrange
    env = worker_env()

    # Act
    runtime = load_runtime(env)

    # Assert
    assert runtime.settings == WorkerSettings.from_env(env)
    assert isinstance(runtime.transport, HttpsTransport)
    assert runtime.clock().utcoffset() == timedelta(0)
    assert LOWERCASE_UUID.fullmatch(runtime.new_uuid()) is not None


def test_load_runtime_when_a_required_variable_is_missing_should_raise_settings_invalid():
    # Arrange
    env = worker_env(**{REMOVED_VARIABLE: None})

    # Act / Assert
    with pytest.raises(SettingsInvalid) as raised:
        load_runtime(env)
    assert raised.value.variable == REMOVED_VARIABLE


def test_load_runtime_when_the_tree_and_the_environment_are_both_broken_should_refuse_the_tree_first(monkeypatch):
    # Arrange
    def refuse_the_tree() -> None:
        raise ContractTreeInvalid("the tree of this test is refused")

    monkeypatch.setattr(files, "verify_contract_tree", refuse_the_tree)

    # Act / Assert
    with pytest.raises(ContractTreeInvalid):
        load_runtime(worker_env(**{REMOVED_VARIABLE: None}))


def test_load_runtime_when_it_loads_should_check_the_contract_tree_exactly_once(monkeypatch):
    # Arrange
    checks: list[tuple[Any, ...]] = []
    monkeypatch.setattr(files, "verify_contract_tree", lambda *arguments: checks.append(arguments))

    # Act
    runtime = load_runtime(worker_env())

    # Assert
    assert isinstance(runtime, Runtime)
    assert checks == [()]
