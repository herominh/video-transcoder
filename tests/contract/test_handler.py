"""One start of the worker for one dispatch: accept, claim, then work or report, and always return the job output.

handle() refuses a dispatch it cannot accept (DispatchRefused, nothing sent, no work). Granted, it hands
the claimed execution to the job runner exactly once; unclaimed, it does no work and reports
`transcode.unclaimed`; a claiming that ends as neither is a wiring bug, and no work is done. Either
message is the job's output and is also posted to the callback address, the same signed bytes both ways,
and no failure of the callback costs the output; a job runner that fails, or returns what is no job
output, posts nothing there. Every start draws its own runtime id. load_runtime() checks the contract
tree before it reads the configuration. Nothing of a message, a signature, the bootstrap token, a key or
a storage credential is logged, and an error of the callback is logged by its class only once it is no
longer being handled, so that a log handler failing on that line reports nothing of the error's own text.

One start has one wall-time budget, counted from the moment handle() begins: the claiming ends by it,
the job is told when it ends, and the callback is not attempted once it is spent (the output still is
returned).

The transport and the clock are the test's own (v2_support): no network, no real waiting.
"""

import contextlib
import dataclasses
import io
import json
import logging
import re
import sys
import time
import types
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Optional

import pytest

from core.failure import Failure
from core.protocol import claim as claim_module
from core.protocol import files, handler, messages
from core.protocol import outbox as outbox_module
from core.protocol import pipeline, signing
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
    LOG_PROBE,
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
    key_forms,
    modified_fixture,
    signed_answer,
    signed_carrier,
    slices_shown,
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
# In a start of 30 s: how long the Hub has the first claim wait, and how long the job then takes. They
# leave 10 s of the start to the second claim and 4 s to the callback, each less than its own time-out.
LATE_CLAIM_WAIT_S = 20
LATE_JOB_TOOK_S = 6.0
# The callback's own time-out, and how long a POST that times out takes on the fake clock.
CALLBACK_TIMEOUT_S = 10.0
SLOW_POST_S = 15.0
RESULT_KIND = "transcode.result.failed"
CALLBACK_REFUSAL_CODE = "conflicting_terminal_result"
CALLBACK_REFUSED = hub_error_answer(HTTP_CONFLICT, CALLBACK_REFUSAL_CODE, retryable=False)
# A value no log line may show, placed in the text of the error a broken callback transport raises.
CALLBACK_ERROR_CANARY = "CANARY-callback-error-text"
# The text of the error a log handler's stream raises when a line cannot be written.
LOG_FAILURE_CANARY = "CANARY-log-failure-text"
# The loggers a start writes on: its own, the claiming's and the delivery's. A line that showed a
# message, the token or a key would come from the last two.
START_LOGGERS = (handler.__name__, claim_module.__name__, outbox_module.__name__)
# The storage credentials the settings hold: no line of a start may show either.
STORAGE_KEY_ID_VARIABLE = "VH_STORAGE_DEFAULT_ACCESS_KEY_ID"
STORAGE_SECRET_VARIABLE = "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY"
STORAGE_CREDENTIALS = (worker_env()[STORAGE_KEY_ID_VARIABLE], worker_env()[STORAGE_SECRET_VARIABLE])
UUID_VERSION_RANDOM = 4


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


# --- a claiming that ends as neither a grant nor unclaimed --------------------------------------


def _a_grant() -> Grant:
    """The Grant a start of the fixtures' dispatch is given."""
    clock = FakeClock()
    job = StandInJob()
    handle(_job_input(), runtime=_runtime(_transport(clock, _granted()), clock), run_job=job)
    ((execution, _),) = job.calls
    return execution.grant


def _all_a_grant_holds(grant: Grant) -> types.SimpleNamespace:
    """Every field of the grant, in a value of a class that is not Grant."""
    return types.SimpleNamespace(**{field.name: getattr(grant, field.name) for field in dataclasses.fields(grant)})


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: "granted", id="text"),
        pytest.param(lambda: dataclasses.asdict(_a_grant()), id="the-fields-of-a-grant-in-a-dict"),
        pytest.param(lambda: _all_a_grant_holds(_a_grant()), id="the-fields-of-a-grant-in-another-class"),
    ],
)
def test_handle_when_the_claiming_ends_as_neither_a_grant_nor_unclaimed_should_raise_type_error_without_work_or_post(
    monkeypatch, ending: Callable[[], object]
):
    # Arrange: a claiming that returns what none may; the Hub would grant and take the callback
    clock = FakeClock()
    transport = _transport(clock, _granted())
    job = StandInJob()
    returned = ending()
    monkeypatch.setattr(handler, "claim_execution", lambda *arguments, **named: returned)
    key = bytes.fromhex(MEDIA_KEY_HEX)

    # Act / Assert: a wiring bug, told without the value, which may hold a media key
    with pytest.raises(TypeError) as raised:
        handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)
    assert job.calls == []
    assert transport.calls == []
    assert [value for value in (MEDIA_KEY_HEX, repr(key), repr(key)[2:-1]) if value in str(raised.value)] == []


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


# --- a job runner that fails, or returns what is no job output ----------------------------------


class JobBroke(Exception):
    """What the job runner of a test raises."""


def _the_executions_claim_sealed_again(execution: ClaimedExecution, runtime: Runtime) -> SealedMessage:
    """A job runner that returns a sealed `transcode.claim`: a message that travels neither way."""
    now = runtime.clock()
    dispatch = execution.dispatch
    message = messages.claim(
        protocol_version=dispatch.protocol_version,
        message_id=RESULT_MESSAGE_ID,
        sent_at=now,
        key_id=runtime.settings.worker_to_hub_key_id,
        identity=dispatch.identity,
        bootstrap_token=dispatch.bootstrap_token,
        runtime_id=execution.runtime_id,
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_CLAIM, dispatch.identity, settings=runtime.settings, now=now
    )
    return seal_message(message, context, settings=runtime.settings)


def _a_sealed_progress_message(execution: ClaimedExecution, runtime: Runtime) -> SealedMessage:
    """A job runner that returns a sealed `transcode.progress`: a message for the callback address only."""
    now = runtime.clock()
    identity = execution.grant.identity
    message = messages.progress(
        protocol_version=execution.dispatch.protocol_version,
        message_id=RESULT_MESSAGE_ID,
        sent_at=now,
        key_id=runtime.settings.worker_to_hub_key_id,
        identity=identity,
        event_seq=1,
        stage="downloading",
        stage_progress_pct=10,
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_PROGRESS, identity, settings=runtime.settings, now=now, last_event_seq=0
    )
    return seal_message(message, context, settings=runtime.settings)


def test_handle_when_the_job_runner_raises_should_let_its_error_out_and_post_nothing_to_the_callback():
    # Arrange
    clock = FakeClock()
    transport = _transport(clock, _granted())
    broke = JobBroke("the job of this test broke")

    def breaking_job(execution: ClaimedExecution, runtime: Runtime) -> SealedMessage:
        raise broke

    # Act / Assert: the claim is all that was posted
    with pytest.raises(JobBroke) as raised:
        handle(_job_input(), runtime=_runtime(transport, clock), run_job=breaking_job)
    assert raised.value is broke
    assert [call.url for call in transport.calls] == [CLAIM_URL]


@pytest.mark.parametrize(
    "job",
    [_the_executions_claim_sealed_again, _a_sealed_progress_message],
    ids=["a-claim", "a-progress-message"],
)
def test_handle_when_the_job_runner_returns_what_is_never_a_job_output_should_raise_value_error_and_post_nothing_more(
    job: Callable[[ClaimedExecution, Runtime], SealedMessage]
):
    # Arrange: a Hub that would take whatever the callback address is sent
    clock = FakeClock()
    transport = _transport(clock, _granted())

    # Act / Assert: the claim is all that was posted, and the refusal shows nothing of the message
    with pytest.raises(ValueError) as raised:
        handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)
    assert [call.url for call in transport.calls] == [CLAIM_URL]
    assert _request()["claim"]["bootstrap_token"] not in str(raised.value)


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


def test_handle_when_its_monotonic_clock_reads_unlike_its_wall_clock_should_hold_the_claim_and_the_callback_to_that_clock():
    # Arrange: a process that has been up for a while, so its monotonic clock reads nothing like its
    # wall clock. A start of 30 s: the Hub has the first claim wait 20 s and grants the second, and the
    # job then takes 6 s
    clock = _clock_of_a_process_up_for_a_while()
    wait_then_grant = hub_error_answer(
        HTTP_UNAVAILABLE, RETRYABLE_REFUSAL_CODE, retryable=True, retry_after_s=LATE_CLAIM_WAIT_S
    )
    transport = _transport(clock, wait_then_grant, _granted())
    job = StandInJob()
    job.work = lambda: clock.advance(LATE_JOB_TOOK_S)

    # Act
    output = handle(_job_input_with_wall_time(SHORT_WALL_TIME_MS), runtime=_runtime(transport, clock), run_job=job)

    # Assert: the start's wall time ends at a reading of the runtime's own monotonic clock, and each
    # request got what that clock then left of it, less than its own time-out: 10 s for the second
    # claim, 4 s for the callback. No other clock's reading gives these two numbers
    (sealed,) = job.returned
    _, second_claim = transport.calls_to(CLAIM_URL)
    (callback_call,) = transport.calls_to(CALLBACK_URL)
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert second_claim.timeout_s == SHORT_WALL_TIME_S - LATE_CLAIM_WAIT_S == 10.0
    assert callback_call.timeout_s == SHORT_WALL_TIME_S - LATE_CLAIM_WAIT_S - LATE_JOB_TOOK_S == 4.0


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


@pytest.mark.parametrize("wall_time_ends_at", [10**400, -(10**400)], ids=["far-ahead", "far-back"])
def test_claimed_execution_when_the_end_of_its_wall_time_is_an_integer_too_large_to_be_a_float_should_raise_value_error(
    wall_time_ends_at,
):
    # Arrange
    clock = FakeClock()
    job = StandInJob()
    handle(_job_input(), runtime=_runtime(_transport(clock, _granted()), clock), run_job=job)
    ((execution, _),) = job.calls

    # Act and Assert: refused as not finite, like an infinity, and not with the conversion's own OverflowError
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(execution, wall_time_ends_at=wall_time_ends_at)


def test_claimed_execution_when_the_end_of_its_wall_time_is_the_largest_whole_number_a_float_holds_should_hold_it():
    # Arrange
    clock = FakeClock()
    job = StandInJob()
    handle(_job_input(), runtime=_runtime(_transport(clock, _granted()), clock), run_job=job)
    ((execution, _),) = job.calls
    largest = int(sys.float_info.max)

    # Act
    held = dataclasses.replace(execution, wall_time_ends_at=largest)

    # Assert: finite, however absurd as a reading of a clock
    assert held.wall_time_ends_at == largest


# --- what a start logs --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim_answer", "callback", "kind", "outcome", "level"),
    [
        pytest.param(_granted(), DELIVERED, RESULT_KIND, "delivered", logging.INFO, id="a-result-delivered"),
        pytest.param(CLAIM_REFUSED, DELIVERED, UNCLAIMED_KIND, "delivered", logging.INFO, id="an-unclaimed-report-delivered"),
        pytest.param(_granted(), NO_ANSWER, RESULT_KIND, "not delivered", logging.INFO, id="a-result-never-answered"),
        pytest.param(_granted(), CALLBACK_REFUSED, RESULT_KIND, "not delivered", logging.INFO, id="a-result-refused"),
        pytest.param(
            _granted(),
            RuntimeError(CALLBACK_ERROR_CANARY),
            RESULT_KIND,
            "failed: RuntimeError",
            logging.ERROR,
            id="an-unexpected-error",
        ),
    ],
)
def test_handle_when_the_callback_has_ended_should_log_one_line_naming_the_kind_and_the_outcome(
    caplog, claim_answer, callback, kind, outcome, level
):
    # Arrange
    caplog.set_level(logging.DEBUG, logger=handler.__name__)
    clock = FakeClock()
    transport = _transport(clock, claim_answer)
    transport.route(CALLBACK_URL, then=callback)

    # Act
    handle(_job_input(), runtime=_runtime(transport, clock), run_job=StandInJob())

    # Assert: an error is named by its class, never by its own text
    (logged,) = [record for record in caplog.records if record.name == handler.__name__]
    line = logged.getMessage()
    assert logged.levelno == level
    assert line.endswith(f"a {kind} {outcome}")
    assert CALLBACK_ERROR_CANARY not in line


class _FailingStream(io.TextIOBase):
    """A standard error that fails every write."""

    def write(self, text: str) -> int:
        raise OSError(LOG_FAILURE_CANARY)


@contextlib.contextmanager
def _start_log_through(log_handler: logging.Handler) -> Iterator[None]:
    """Inside, the logger of a start (core.protocol.handler) hands its records to `log_handler` as well."""
    start_logger = logging.getLogger(handler.__name__)
    start_logger.addHandler(log_handler)
    try:
        yield
    finally:
        start_logger.removeHandler(log_handler)
        log_handler.close()


def test_handle_when_a_log_handler_fails_on_the_line_of_a_broken_callback_should_keep_the_errors_text_out_of_standard_error(
    monkeypatch, capsys
):
    # Arrange: a handler of the kind the process's log has (a stream handler) fails on every line. Such
    # a handler does not raise: Python's logging prints a report of its own to standard error
    # (raiseExceptions, its default), with every error the failure is chained to. The callback's
    # transport breaks on an error whose text is a canary
    monkeypatch.setattr(logging, "raiseExceptions", True)
    clock = FakeClock()
    transport = _transport(clock, _granted())
    transport.route(CALLBACK_URL, then=RuntimeError(CALLBACK_ERROR_CANARY))
    job = StandInJob()

    # Act
    with _start_log_through(logging.StreamHandler(_FailingStream())):
        output = handle(_job_input(), runtime=_runtime(transport, clock), run_job=job)
    written = capsys.readouterr().err

    # Assert: logging did report the handler's failure, its report holds nothing of the callback
    # error's own text, and the output is still returned
    (sealed,) = job.returned
    assert LOG_FAILURE_CANARY in written
    assert CALLBACK_ERROR_CANARY not in written
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}


@pytest.mark.parametrize(
    ("claim_answers", "claim_then", "callback"),
    [
        pytest.param((_granted(),), None, DELIVERED, id="granted-and-delivered"),
        pytest.param((CLAIM_REFUSED,), None, DELIVERED, id="refused"),
        pytest.param((), NO_ANSWER, DELIVERED, id="unreachable"),
        pytest.param((HttpAnswer(HTTP_INTERNAL_ERROR, HTML_BODY, None),), None, DELIVERED, id="invalid-answer"),
        pytest.param((_granted(),), None, NO_ANSWER, id="granted-and-never-delivered"),
        pytest.param((_granted(),), None, CALLBACK_REFUSED, id="granted-and-the-callback-refused"),
        pytest.param((_granted(),), None, RuntimeError(CALLBACK_ERROR_CANARY), id="granted-and-the-callback-broke"),
    ],
)
def test_handle_when_it_ends_any_way_should_log_nothing_of_a_message_a_signature_the_token_or_a_key(
    caplog, claim_answers, claim_then, callback
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, *claim_answers, claim_then=claim_then)
    transport.route(CALLBACK_URL, then=callback)
    job_input, request, grant = _job_input(), _request(), json.loads(files.read_bytes(GRANT_ENCRYPTED))
    answers = [answer for answer in (*claim_answers, claim_then, callback) if isinstance(answer, HttpAnswer)]
    # A line may name the dispatch, the execution, the kind of the message it reports and the code of the
    # Hub's refusal, and nothing else of any of them.
    nameable = (request["identity"]["dispatch_id"], grant["identity"]["execution_id"], RESULT_KIND, UNCLAIMED_KIND)
    nameable += (CALLBACK_REFUSAL_CODE,)

    # Act
    output = handle(job_input, runtime=_runtime(transport, clock), run_job=StandInJob())
    for name in START_LOGGERS:
        logging.getLogger(name).debug(LOG_PROBE)
    logged = caplog.text

    # Assert
    bodies = [job_input["vh_message"], output["vh_message"]]
    bodies += [call.body.decode("ascii") for call in transport.calls] + [answer.body.decode("ascii") for answer in answers]
    signatures = [job_input["vh_signature"], output["vh_signature"]]
    signatures += [call.signature for call in transport.calls] + [answer.signature for answer in answers if answer.signature]
    # A key is held as bytes, so it is looked for as bytes are printed too, not as hex alone.
    keys = [bytes.fromhex(MEDIA_KEY_HEX), WORKER_KEY, HUB_KEY]
    secrets = [request["claim"]["bootstrap_token"], RUNTIME_ID, *(form for key in keys for form in key_forms(key))]
    secrets += ["vh_", "bootstrap_token", "key_hex", CALLBACK_ERROR_CANARY, *STORAGE_CREDENTIALS, *signatures]
    # The capture sees every logger a start writes on, each down to DEBUG.
    assert [record.name for record in caplog.records if record.getMessage() == LOG_PROBE] == list(START_LOGGERS)
    assert LOG_PROBE in logged
    assert [secret for secret in secrets if secret in logged] == []
    assert [piece for body in bodies for piece in slices_shown(logged, body, nameable)] == []


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


def test_load_runtime_when_the_environment_is_valid_should_wire_the_monotonic_clock_and_a_source_of_random_uuids():
    # Act
    runtime = load_runtime(worker_env())
    drawn = [uuid.UUID(runtime.new_uuid()) for _ in range(2)]

    # Assert: a runtime id is random (README 15.1, rule 1), and a wait is measured in seconds that only pass
    assert runtime.monotonic is time.monotonic
    assert [(drawn_id.version, drawn_id.variant) for drawn_id in drawn] == [(UUID_VERSION_RANDOM, uuid.RFC_4122)] * 2
    assert drawn[0] != drawn[1]


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
