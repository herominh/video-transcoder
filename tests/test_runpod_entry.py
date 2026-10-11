"""The RunPod entry point: what the provider gets for one job, and what it never gets.

job_handler() is driven with the test's own runtime (the fake transport and clock of v2_support: no
network, no real waiting) and a stand-in job runner, through the real core.protocol.handler.handle().
A job's output is exactly the carrier handle() returned, the same signed bytes the callback address
got; once a storage request was abandoned in the process, every output also asks the SDK for a fresh
worker. A refused dispatch and a bug leave the handler as the wrapper's own errors: what the provider
records of a failed job (the error's text, and its traceback with every error it is chained to) holds
the refusal's words or the bug's class, and nothing of the job input, of a secret or of an error's own
text.

build_handler() is the real wiring: it refuses a contract tree or a configuration it cannot trust before
it returns a handler, and removes the scratch directories a killed process left in the scratch root.
configure_logging() sets up the process's log when the file is run: the worker's own loggers at INFO,
and nothing from the HTTP library under the storage clients, whose warnings name the storage's host and
a request's whole URL. Importing the module loads neither the SDK nor a v1 module, needs no
configuration and configures no logging.

main() is what the file does when it is run as the program. It is driven here with a starter of the
test's own in the place of the SDK's, and the file itself is run in a process of its own with a stand-in
for the SDK first on that process's module path: neither needs the SDK, and both behave the same where
it is installed.

The count of abandoned storage requests belongs to the process, and a test session abandons some long
before these tests run. So every test here gives the wrapper a count of its own, which starts at zero
and which the test grows (the `abandoned` fixture replaces the name the wrapper reads). Two tests are
the exception (the `the_real_count` fixture): nothing of the wrapper is replaced in them, so it reads
the process's real count through what it imported itself, and that count is zero for the length of the
test and put back afterwards. Each abandons a real request: one that "never returns" is one a test
double keeps until the test lets it go (_Hold, as tests/job/test_storage.py has it), and the deadline
it is abandoned at is a real, short one.

The last tests hand the handler to the SDK's own run_job; they are skipped where the SDK is not
installed (the worker image has it).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

import pytest

from core.bounded_process import Deadline
from core.failure import Failure, WorkerFailure
from core.job import storage
from core.job.runner import JobRunner
from core.job.scratch import SCRATCH_PREFIX
from core.job.storage import DeadlineSource
from core.protocol import files, messages, pipeline, signing
from core.protocol.files import ContractTreeInvalid
from core.protocol.handler import ClaimedExecution, Runtime
from core.protocol.outbox import SealedMessage, seal_message
from core.protocol.registry import MessageKind
from core.protocol.settings import SettingsInvalid
from core.protocol.transport import HttpAnswer, TransportFailed
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
    catalog_context,
    hub_error_answer,
    modified_fixture,
    signed_answer,
    signed_carrier,
    uuid_source,
    worker_env,
    worker_settings,
)
from wrappers import runpod_handler
from wrappers.runpod_handler import JobRefused, WorkerFault, build_handler, job_handler

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = worker_settings()
JOB_ID = "job-1"
MESSAGE_MEMBER = "vh_message"
SIGNATURE_MEMBER = "vh_signature"
REFRESH_WORKER_MEMBER = "refresh_worker"
UNCLAIMED_KIND = "transcode.unclaimed"
CAUSE_REFUSED = "refused"
CAUSE_UNREACHABLE = "unreachable"
REFUSAL_CODE = "claim_conflict"
HTTP_OK = 200
HTTP_CONFLICT = 409
DELIVERED = HttpAnswer(HTTP_OK, b"", None)
NO_ANSWER = TransportFailed("ConnectTimeout")
CLAIM_REFUSED = hub_error_answer(HTTP_CONFLICT, REFUSAL_CODE, retryable=False)
RESULT_MESSAGE_ID = "99999999-9999-7999-8999-0000000000f1"
# The failure the stand-in job reports: it does no work, it only proves what the handler handed it.
STAND_IN_FAILURE = Failure(
    error_class="internal_error",
    code="stand_in_job",
    retryable=False,
    detail="The stand-in job of the test did no work.",
)
# The failure the storage raises for a request a test has it give up at the deadline.
REQUEST_GIVEN_UP = Failure(
    error_class="deadline_exceeded",
    code="request_deadline_exceeded",
    retryable=False,
    detail="The request of the test ran out of time.",
)

# The entry point's own logger, and the line format it configures: a test formats each record with it,
# as the process's handler writes it (a formatter appends the traceback of a record that has `exc_info`).
ENTRY_LOGGER = "wrappers.runpod_handler"
LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
# What Python prints between two errors of a chain.
CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred",
    "The above exception was the direct cause of the following exception",
)

REFUSED_AT_CARRIER = "the dispatch is refused at carrier"
REFUSED_AT_SIGNATURE = "the dispatch is refused at signature"
REFUSED_AT_VALIDATION = "the dispatch is refused at validation: semantic/audience_mismatch"
OTHER_WORKER_AUDIENCE = "worker:another-worker"
# A value no error, traceback or log line may show, placed inside the job input.
INPUT_CANARY = "CANARY-inside-the-job-input"
REQUEST = files.read_bytes(REQUEST_ENCRYPTED)
REQUEST_WITH_CANARY = modified_fixture(REQUEST_ENCRYPTED, {"/source/object_key": f"test/sources/{INPUT_CANARY}.mp4"})
REQUEST_FOR_ANOTHER_WORKER = modified_fixture(
    REQUEST_ENCRYPTED,
    {"/source/object_key": f"test/sources/{INPUT_CANARY}.mp4", "/audience": OTHER_WORKER_AUDIENCE},
)
BOOTSTRAP_TOKEN = json.loads(REQUEST)["claim"]["bootstrap_token"]
GRANT = files.read_bytes(GRANT_ENCRYPTED)
STORAGE_KEY_ID_VARIABLE = "VH_STORAGE_DEFAULT_ACCESS_KEY_ID"
STORAGE_SECRET_VARIABLE = "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY"
# What no log line, error text, traceback or returned value may show: the canary of the job input, the
# bootstrap token (so a claim body too, which holds it), the media key as hex and as a repr shows its
# bytes, the grant body, the signing keys and the storage credentials. A signature is no secret.
SECRETS = (
    INPUT_CANARY,
    BOOTSTRAP_TOKEN,
    MEDIA_KEY_HEX,
    repr(bytes.fromhex(MEDIA_KEY_HEX))[2:-1],
    GRANT.decode("ascii"),
    HUB_KEY.hex(),
    WORKER_KEY.hex(),
    worker_env()[STORAGE_KEY_ID_VARIABLE],
    worker_env()[STORAGE_SECRET_VARIABLE],
)

# The text of a bug, and of the error it is chained to: each quotes what the error choked on.
BUG_CANARY = "CANARY-quoted-by-a-bug"
CHAINED_CANARY = "CANARY-quoted-by-a-chained-error"
BUG_TEXT = f"could not reach https://storage.test.invalid/?X-Amz-Signature={BUG_CANARY}"
CHAINED_TEXT = f"the connection to {CHAINED_CANARY} was reset"
# The text of the error a log line fails on when it cannot be built or written.
LOG_FAILURE_CANARY = "CANARY-quoted-by-a-failing-log"
LOG_FAILURE_TEXT = f"no memory left for {LOG_FAILURE_CANARY}"
# The HTTP library under the storage clients: the parent of every logger it writes on, and a storage
# host, bucket and object key no log line may show.
HTTP_LIBRARY_LOGGER = "urllib3"
STORAGE_HOST_CANARY = "account-canary.r2.example.invalid"
STORAGE_BUCKET_CANARY = "bucket-canary"
STORAGE_KEY_CANARY = "original-videos/uuid-canary/source.mp4"
STORAGE_URL_CANARY = f"https://{STORAGE_HOST_CANARY}/{STORAGE_BUCKET_CANARY}/{STORAGE_KEY_CANARY}"
STORAGE_CANARIES = (STORAGE_HOST_CANARY, STORAGE_BUCKET_CANARY, STORAGE_KEY_CANARY)
# What that library writes with what it talks to, as (logger, level, format, arguments): the four
# warnings of urllib3 2.7.0 (connectionpool.py and connection.py; 1.26 wrote the third on the pool's
# logger), and an error on a logger no version has yet.
HTTP_LIBRARY_LINES = (
    pytest.param(
        "urllib3.connectionpool", logging.WARNING,
        "Connection pool is full, discarding connection: %s. Connection pool size: %s",
        (STORAGE_HOST_CANARY, 10),
        id="the-pool-is-full",
    ),
    pytest.param(
        "urllib3.connectionpool", logging.WARNING,
        "Retrying (%r) after connection broken by '%r': %s",
        ("Retry(total=0)", "ProtocolError()", f"/{STORAGE_BUCKET_CANARY}/{STORAGE_KEY_CANARY}"),
        id="a-broken-connection-is-retried",
    ),
    pytest.param(
        "urllib3.connection", logging.WARNING,
        "Failed to parse headers (url=%s): %s",
        (STORAGE_URL_CANARY, "[MissingHeaderBodySeparatorDefect()], unparsed data: 'x'"),
        id="the-headers-of-an-answer-do-not-parse",
    ),
    pytest.param(
        "urllib3.connection", logging.WARNING,
        "Certificate did not match expected hostname: %s. Certificate: %s",
        (STORAGE_HOST_CANARY, {"subjectAltName": (("DNS", "another.example.invalid"),)}),
        id="a-certificate-is-for-another-host",
    ),
    pytest.param(
        "urllib3.a_logger_of_a_later_version", logging.ERROR,
        "could not finish the request to %s",
        (STORAGE_URL_CANARY,),
        id="an-error-on-a-logger-of-a-later-version",
    ),
)
# How long the loopback server of the real library's test and its one request may take.
LOOPBACK_TIMEOUT_S = 5.0
# Loggers of the worker's own: the entry point's, the storage's and the start's.
WORKER_LOGGERS = ("wrappers.runpod_handler", "core.job.storage", "core.protocol.handler")

IMAGE_DIGEST_VARIABLE = "VH_IMAGE_DIGEST"
SCRATCH_DIR_VARIABLE = "VH_SCRATCH_DIR"
IMAGE_DIGEST = "sha256:" + "5e" * 32
# A value an operator mistyped into the configuration: a refusal names the variable, never this.
CONFIG_CANARY = "CANARY-typed-into-the-configuration"
TAMPERED_FILE = "limits.json"
# How long ago a killed process's scratch directory was last changed: far longer than the least age of a leftover.
LEFT_LONG_AGO_S = 2 * 60 * 60

# A real deadline a test waits out, long enough for a loaded host to reach the request the test keeps.
SHORT_MS = 500
# How long a kept request stays at most when no test lets it go, and how long a request that was let go
# may take to end: a test that would hang fails instead.
KEPT_AT_MOST_S = 20.0
PROMPT_S = 10.0
# The modules the entry point never loads: the SDK, until the process starts serving, and the v1 worker.
NEVER_IMPORTED = (
    "runpod",
    "core.api",
    "core.config",
    "core.transcoder",
    "core.storage",
    "core.callback",
    "core.signing",
)
IMPORT_TIMEOUT_S = 120.0
SDK_RUN_TIMEOUT_S = 60.0

# The SDK's module, and the member of the configuration its serverless loop is started with that holds the handler.
SDK_MODULE = "runpod"
HANDLER_MEMBER = "handler"
# The line of the worker's own that a test writes at each of two moments of a program start.
WHILE_THE_HANDLER_IS_BUILT = "the handler is being built"
WHEN_THE_LOOP_IS_STARTED = "the serverless loop was started"
# A logger the HTTP library writes its warnings about a request on.
HTTP_LIBRARY_REQUEST_LOGGER = "urllib3.connection"
PROGRAM_TIMEOUT_S = 120.0
# A stand-in for the SDK in a process of its own: a package of the SDK's name, first on that process's module
# path. Its import does to the logging what the import of SDK 1.10.1 does: logging.basicConfig hangs an INFO
# handler, in the SDK's format, on a root logger that has none (runpod/serverless/utils/rp_upload.py), and the
# HTTP library's parent logger gets the level WARNING (runpod/__init__.py). Its `serverless.start` starts no
# loop: it writes a warning of the HTTP library that names a storage request and a line of the worker's own,
# and prints what it was started with.
STAND_IN_SDK = f"""\
import json
import logging
import types

logging.basicConfig(level=logging.INFO, format="the SDK's own format: %(message)s", handlers=[logging.StreamHandler()])
logging.getLogger({HTTP_LIBRARY_LOGGER!r}).setLevel(logging.WARNING)


def _start(config):
    http_library = logging.getLogger({HTTP_LIBRARY_REQUEST_LOGGER!r})
    http_library.warning("Failed to parse headers (url=%s)", {STORAGE_URL_CANARY!r})
    logging.getLogger("core.job.storage").info({WHEN_THE_LOOP_IS_STARTED!r})
    print(json.dumps({{"members": sorted(config), "handler_is_callable": callable(config.get({HANDLER_MEMBER!r}))}}))


serverless = types.SimpleNamespace(start=_start)
"""


# --- the test's side of one start -----------------------------------------------------------------


class _StandInJob:
    """A job runner that records what it was handed, does `work`, and returns a sealed `transcode.result.failed`."""

    def __init__(self, work: Callable[[], None] = lambda: None) -> None:
        self.calls: list[tuple[ClaimedExecution, Runtime]] = []
        self.returned: list[SealedMessage] = []
        self._work = work

    def __call__(self, execution: ClaimedExecution, runtime: Runtime) -> SealedMessage:
        self.calls.append((execution, runtime))
        self._work()
        now = runtime.clock()
        message = messages.result_failed(
            protocol_version=execution.dispatch.protocol_version,
            message_id=RESULT_MESSAGE_ID,
            sent_at=now,
            key_id=runtime.settings.worker_to_hub_key_id,
            identity=execution.grant.identity,
            failure=STAND_IN_FAILURE,
            stage=None,
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_RESULT_FAILED,
            execution.grant.identity,
            settings=runtime.settings,
            now=now,
            source_id=execution.dispatch.source.source_id,
        )
        sealed = seal_message(message, context, settings=runtime.settings)
        self.returned.append(sealed)
        return sealed


def _transport(clock: FakeClock, *claim_answers: Outcome, claim_then: Outcome | None = None) -> FakeTransport:
    """A transport whose claim address answers as given and whose callback address accepts every POST."""
    transport = FakeTransport(clock=clock)
    transport.route(CLAIM_URL, *claim_answers, then=claim_then)
    transport.route(CALLBACK_URL, then=DELIVERED)
    return transport


def _runtime(transport: FakeTransport, clock: FakeClock) -> Runtime:
    """A runtime whose id source gives the fixtures' runtime id first: the id the fixtures' grant is for."""
    return Runtime(
        settings=SETTINGS,
        transport=transport,
        clock=clock,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        new_uuid=uuid_source(first=RUNTIME_ID),
    )


def _granted() -> HttpAnswer:
    return signed_answer(GRANT, HUB_KEY)


def _job(job_input: object) -> dict[str, Any]:
    """A job as the SDK hands it to the handler."""
    return {"id": JOB_ID, "input": job_input}


def _dispatch_job() -> dict[str, Any]:
    """A job whose input is the fixtures' dispatch, signed by the Hub."""
    return _job(signed_carrier(REQUEST, HUB_KEY))


def _posted_to_the_callback(transport: FakeTransport) -> list[tuple[bytes, str]]:
    return [(call.body, call.signature) for call in transport.calls_to(CALLBACK_URL)]


def _log_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every captured record as the entry point's handler writes it."""
    return [LOG_FORMATTER.format(record) for record in caplog.records]


def _entry_records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == ENTRY_LOGGER and record.levelno == level]


def _shown(hidden: tuple[str, ...], *texts: str) -> list[str]:
    """The values of `hidden` that one of the texts shows."""
    return [value for value in hidden if any(value in text for text in texts)]


def _failing_log(*arguments: object, **named_arguments: object) -> None:
    """Stands where a log line is built or written and cannot be: its error quotes a canary of its own."""
    raise MemoryError(LOG_FAILURE_TEXT)


class _Hold:
    """Where a test double keeps a request that never returns: until the test lets it go.

    release() lets every kept request go and waits for every thread started since this object was made
    to end, so no thread outlives its test. `entered` is set once a request is kept: a test asserts it,
    or it would pass on a host too slow to reach the request before the deadline.
    """

    def __init__(self) -> None:
        self._let_go = threading.Event()
        self._threads_before = set(threading.enumerate())
        self.entered = threading.Event()

    def keep(self) -> None:
        self.entered.set()
        self._let_go.wait(KEPT_AT_MOST_S)

    def release(self) -> None:
        self._let_go.set()
        started = [thread for thread in threading.enumerate() if thread not in self._threads_before]
        for thread in started:
            thread.join(PROMPT_S)
        assert not any(thread.is_alive() for thread in started), "a request's thread outlived its test"


@pytest.fixture
def hold() -> Iterator[_Hold]:
    kept = _Hold()
    yield kept
    kept.release()


class _NeverAnsweringSource:
    """A source of one byte whose read the storage never answers: the read is kept until the test lets it go."""

    size_bytes = 1

    def __init__(self, hold: _Hold) -> None:
        self._hold = hold

    def read(self, start: int, length: int) -> bytes:
        self._hold.keep()
        return bytes(length)


def _abandoning_a_storage_read(hold: _Hold) -> Callable[[], None]:
    """What a job does when the storage never answers a read: it waits to a real, short deadline, then goes on."""

    def work() -> None:
        source = DeadlineSource(_NeverAnsweringSource(hold), Deadline.after_ms(SHORT_MS))
        with pytest.raises(WorkerFailure):
            source.read(0, 1)

    return work


# The fixture a test asks for to have the wrapper read the process's real count of abandoned requests.
REAL_COUNT_FIXTURE = "the_real_count"


class _AbandonedCount:
    """Stands where the wrapper reads core.job.storage.abandoned_requests: the count of a process of the
    test's own, which starts at zero and grows when the test says a request was abandoned."""

    def __init__(self) -> None:
        self._count = 0

    def __call__(self) -> int:
        return self._count

    def grow(self) -> None:
        self._count += 1


@pytest.fixture(autouse=True)
def abandoned(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _AbandonedCount:
    """Every test begins in a process that never abandoned a storage request, whatever the session did:
    the wrapper reads a count of the test's own. A test that asks for the real count is the exception:
    nothing of the wrapper is replaced there, so that it reads the count through what it imported itself."""
    count = _AbandonedCount()
    if REAL_COUNT_FIXTURE not in request.fixturenames:
        monkeypatch.setattr(runpod_handler, "abandoned_requests", count)
    return count


@pytest.fixture
def the_real_count() -> Iterator[Callable[[], int]]:
    """The process's real count, the one core.job.storage keeps, read by a wrapper nothing was replaced
    in (the `abandoned` fixture leaves the wrapper alone in a test that asks for this one); its reader
    is returned.

    The count is zero when the test begins, whatever the session abandoned before. It never falls by
    itself, so when the test ends it is put back to what it was, whatever the test did: the count
    belongs to the process, and the tests that follow read it too."""
    with storage._abandoned_requests_lock:
        counted_by_the_session = storage._abandoned_requests
        storage._abandoned_requests = 0
    try:
        yield storage.abandoned_requests
    finally:
        with storage._abandoned_requests_lock:
            storage._abandoned_requests = counted_by_the_session


# --- the job's output -----------------------------------------------------------------------------


def test_handler_when_the_claim_is_granted_should_hand_the_claimed_execution_to_the_job_runner(caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    runtime = _runtime(_transport(clock, _granted()), clock)
    stand_in = _StandInJob()

    # Act
    job_handler(runtime, stand_in)(_dispatch_job())

    # Assert
    ((execution, handed_runtime),) = stand_in.calls
    assert isinstance(execution, ClaimedExecution)
    assert execution.dispatch.identity.dispatch_id == json.loads(REQUEST)["identity"]["dispatch_id"]
    assert execution.grant.identity.execution_id == json.loads(GRANT)["identity"]["execution_id"]
    assert execution.grant.media_key == bytes.fromhex(MEDIA_KEY_HEX)
    assert execution.runtime_id == RUNTIME_ID
    assert handed_runtime is runtime
    assert _shown(SECRETS, *_log_lines(caplog)) == []


def test_handler_when_the_job_ran_should_return_exactly_the_carrier_of_its_sealed_result(caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, _granted())
    stand_in = _StandInJob()

    # Act
    output = job_handler(_runtime(transport, clock), stand_in)(_dispatch_job())

    # Assert: the two members of the carrier and nothing else, the same bytes the callback address got
    (sealed,) = stand_in.returned
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}
    assert sorted(output) == ["vh_message", "vh_signature"]
    assert _posted_to_the_callback(transport) == [(sealed.body, sealed.signature)]
    assert _entry_records(caplog, logging.WARNING) == []
    assert _shown(SECRETS, json.dumps(output), *_log_lines(caplog)) == []


def test_handler_when_no_request_was_ever_abandoned_should_return_the_very_dict_the_start_returned(monkeypatch):
    # Arrange: a start whose output carrier the test holds
    carrier = {"vh_message": "{}", "vh_signature": "hmac-sha256=" + "0" * 64}
    monkeypatch.setattr(runpod_handler, "handle", lambda job_input, *, runtime, run_job: carrier)
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock), clock), _StandInJob())

    # Act
    output = handler(_dispatch_job())

    # Assert: the carrier itself, not a copy of it
    assert output is carrier
    assert output == {"vh_message": "{}", "vh_signature": "hmac-sha256=" + "0" * 64}


@pytest.mark.parametrize(
    ("claim_answers", "claim_then", "cause"),
    [
        pytest.param((CLAIM_REFUSED,), None, CAUSE_REFUSED, id="the-hub-refuses-the-claim"),
        pytest.param((), NO_ANSWER, CAUSE_UNREACHABLE, id="the-hub-never-answers"),
    ],
)
def test_handler_when_the_claim_is_not_granted_should_return_exactly_the_carrier_of_a_signed_unclaimed_report(
    caplog, claim_answers, claim_then, cause
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, *claim_answers, claim_then=claim_then)
    stand_in = _StandInJob()

    # Act
    output = job_handler(_runtime(transport, clock), stand_in)(_dispatch_job())

    # Assert: a report the Hub accepts, in the two members of the carrier, and the same bytes posted
    body, signature = output[MESSAGE_MEMBER].encode("ascii"), output[SIGNATURE_MEMBER]
    report = json.loads(body)
    hub_context = catalog_context(HUB_UNCLAIMED_POLL_CONTEXT, now=clock(), known_key_ids=[WORKER_KEY_ID])
    assert sorted(output) == ["vh_message", "vh_signature"]
    assert (report["message_kind"], report["cause"]) == (UNCLAIMED_KIND, cause)
    assert report["identity"] == json.loads(REQUEST)["identity"]
    assert signing.verify(body, WORKER_KEY, signature)
    assert pipeline.validate(body, hub_context).accepted
    assert _posted_to_the_callback(transport) == [(body, signature)]
    assert stand_in.calls == []
    assert _shown(SECRETS, json.dumps(output), *_log_lines(caplog)) == []


# --- a refused dispatch ---------------------------------------------------------------------------

REFUSED_JOBS = [
    pytest.param(
        _job(signed_carrier(REQUEST_WITH_CANARY, UNCONFIGURED_KEY)), REFUSED_AT_SIGNATURE,
        id="a-signature-of-a-key-the-worker-does-not-hold",
    ),
    pytest.param(
        _job({**signed_carrier(REQUEST, HUB_KEY), MESSAGE_MEMBER: REQUEST_WITH_CANARY.decode("ascii")}),
        REFUSED_AT_SIGNATURE,
        id="a-message-changed-after-it-was-signed",
    ),
    pytest.param(
        _job(signed_carrier(REQUEST_FOR_ANOTHER_WORKER, HUB_KEY)), REFUSED_AT_VALIDATION,
        id="a-signed-request-for-another-worker",
    ),
    pytest.param(
        _job({**signed_carrier(REQUEST_WITH_CANARY, HUB_KEY), "note": INPUT_CANARY}), REFUSED_AT_CARRIER,
        id="a-carrier-with-a-third-member",
    ),
    pytest.param(
        _job({MESSAGE_MEMBER: REQUEST_WITH_CANARY.decode("ascii")}), REFUSED_AT_CARRIER,
        id="a-carrier-without-its-signature",
    ),
    pytest.param(_job(INPUT_CANARY), REFUSED_AT_CARRIER, id="an-input-that-is-a-text"),
    pytest.param(_job(None), REFUSED_AT_CARRIER, id="an-input-of-none"),
    pytest.param({"id": JOB_ID, "note": INPUT_CANARY}, REFUSED_AT_CARRIER, id="a-job-without-input"),
    pytest.param([INPUT_CANARY], REFUSED_AT_CARRIER, id="a-job-that-is-a-list"),
    pytest.param(INPUT_CANARY, REFUSED_AT_CARRIER, id="a-job-that-is-a-text"),
    pytest.param(None, REFUSED_AT_CARRIER, id="a-job-that-is-none"),
]


@pytest.mark.parametrize(("job", "refusal_text"), REFUSED_JOBS)
def test_handler_when_the_job_holds_no_dispatch_it_may_act_on_should_raise_its_own_error_with_the_refusals_text(
    caplog, job, refusal_text
):
    # Arrange: a Hub that would grant the claim, so only the refusal keeps the worker from claiming
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, _granted())
    stand_in = _StandInJob()
    handler = job_handler(_runtime(transport, clock), stand_in)

    # Act
    with pytest.raises(JobRefused) as raised:
        handler(job)

    # Assert: what the provider records of it holds the refusal's words, no other error and nothing of the input
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == refusal_text
    assert _shown(CHAIN_MARKERS, recorded) == []
    assert _shown(SECRETS, recorded, *_log_lines(caplog)) == []
    assert (transport.calls, stand_in.calls) == ([], [])


# --- a bug ----------------------------------------------------------------------------------------


def _fail() -> None:
    raise RuntimeError(BUG_TEXT)


def _fail_because_of_another_error() -> None:
    try:
        raise ConnectionError(CHAINED_TEXT)
    except ConnectionError as cause:
        raise LookupError(BUG_TEXT) from cause


def _fail_while_handling_another_error() -> None:
    try:
        raise ConnectionError(CHAINED_TEXT)
    except ConnectionError:
        raise ValueError(BUG_TEXT)


# Each bug with the text of the wrapper's error for it and the classes its one log line names.
BUGS = [
    pytest.param(_fail, "the worker failed on an unexpected RuntimeError", ["RuntimeError"], id="an-error"),
    pytest.param(
        _fail_because_of_another_error,
        "the worker failed on an unexpected LookupError",
        ["LookupError", "ConnectionError"],
        id="an-error-with-a-cause",
    ),
    pytest.param(
        _fail_while_handling_another_error,
        "the worker failed on an unexpected ValueError",
        ["ValueError", "ConnectionError"],
        id="an-error-raised-while-another-was-handled",
    ),
]


@pytest.mark.parametrize(("bug", "fault_text"), [pytest.param(*case.values[:2], id=case.id) for case in BUGS])
def test_handler_when_the_start_fails_on_a_bug_should_raise_its_own_error_naming_the_class_only(bug, fault_text):
    # Arrange
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob(bug))

    # Act
    with pytest.raises(WorkerFault) as raised:
        handler(_dispatch_job())

    # Assert: what the provider records of it holds no error's own text and no other error
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == fault_text
    assert _shown(CHAIN_MARKERS, recorded) == []
    assert _shown((BUG_CANARY, CHAINED_CANARY, *SECRETS), recorded) == []


@pytest.mark.parametrize(
    ("bug", "error_classes"), [pytest.param(case.values[0], case.values[2], id=case.id) for case in BUGS]
)
def test_handler_when_the_start_fails_on_a_bug_should_log_once_its_class_and_stack_without_its_text(
    caplog, bug, error_classes
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob(bug))

    # Act
    with pytest.raises(WorkerFault):
        handler(_dispatch_job())

    # Assert: one error line that says where, by class and stack; no record carries the error itself
    (logged,) = _entry_records(caplog, logging.ERROR)
    line = LOG_FORMATTER.format(logged)
    assert [name for name in error_classes if name not in line] == []
    assert 'File "' in line and ", line " in line
    assert f"in {bug.__name__}" in line
    assert [record for record in caplog.records if record.exc_info] == []
    assert _shown((BUG_CANARY, CHAINED_CANARY, *SECRETS), *_log_lines(caplog)) == []


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        pytest.param(runpod_handler, "_stack_report", id="the-line-cannot-be-built"),
        pytest.param(runpod_handler.logger, "error", id="the-line-cannot-be-written"),
    ],
)
def test_handler_when_the_log_line_of_a_bug_fails_too_should_still_raise_its_own_error_with_nothing_of_either(
    monkeypatch, owner, name
):
    # Arrange: the start fails on a bug, and then the line that would log the bug fails as well
    monkeypatch.setattr(owner, name, _failing_log)
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob(_fail))

    # Act
    with pytest.raises(WorkerFault) as raised:
        handler(_dispatch_job())

    # Assert: the wrapper's error alone, without the bug or the log's failure as its cause or context
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == "the worker failed on an unexpected RuntimeError"
    assert _shown(CHAIN_MARKERS, recorded) == []
    assert _shown((BUG_CANARY, LOG_FAILURE_CANARY, *SECRETS), recorded) == []


class _FailingStream(io.TextIOBase):
    """A standard error that fails every write."""

    def write(self, text: str) -> int:
        raise OSError(LOG_FAILURE_TEXT)


class _FailingFormatter(logging.Formatter):
    """A formatter that cannot build its line."""

    def format(self, record: logging.LogRecord) -> str:
        raise MemoryError(LOG_FAILURE_TEXT)


def _handler_that_cannot_write_its_line() -> logging.Handler:
    return logging.StreamHandler(_FailingStream())


def _handler_that_cannot_build_its_line() -> logging.Handler:
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(_FailingFormatter())
    return handler


@contextlib.contextmanager
def _entry_log_through(handler: logging.Handler) -> Iterator[None]:
    """Inside, the entry point's logger hands its records to `handler` as well."""
    entry_logger = logging.getLogger(ENTRY_LOGGER)
    entry_logger.addHandler(handler)
    try:
        yield
    finally:
        entry_logger.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize(
    "failing_handler",
    [
        pytest.param(_handler_that_cannot_write_its_line, id="the-stream-fails-the-write"),
        pytest.param(_handler_that_cannot_build_its_line, id="the-line-cannot-be-formatted"),
    ],
)
def test_handler_when_a_log_handler_fails_on_the_line_of_a_bug_should_keep_the_bugs_text_out_of_standard_error(
    monkeypatch, capsys, failing_handler
):
    # Arrange: a handler of the kind the process's log has (a stream handler) fails on every line. Such
    # a handler does not raise: Python's logging prints a report of its own to standard error
    # (raiseExceptions, its default), with every error the failure is chained to. The start fails on a
    # bug that has a cause
    monkeypatch.setattr(logging, "raiseExceptions", True)
    clock = FakeClock()
    handler = job_handler(
        _runtime(_transport(clock, _granted()), clock), _StandInJob(_fail_because_of_another_error)
    )

    # Act
    with _entry_log_through(failing_handler()), pytest.raises(WorkerFault) as raised:
        handler(_dispatch_job())
    written = capsys.readouterr().err

    # Assert: logging did report the handler's failure, and the report holds nothing of the bug's own
    # text or of its cause's; the error that leaves is the wrapper's own, alone
    recorded = "".join(traceback.format_exception(raised.value))
    assert LOG_FAILURE_CANARY in written
    assert _shown((BUG_CANARY, CHAINED_CANARY, *SECRETS), written) == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == "the worker failed on an unexpected LookupError"
    assert _shown(CHAIN_MARKERS, recorded) == []
    assert _shown((BUG_CANARY, CHAINED_CANARY, LOG_FAILURE_CANARY, *SECRETS), recorded) == []


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), SystemExit(3)], ids=["an-interrupt", "the-interpreters-exit"])
def test_handler_when_the_process_is_being_stopped_should_let_that_pass_unchanged(stop):
    # Arrange: what stops a process is no Exception, and no bug of the start
    clock = FakeClock()

    def stopped() -> None:
        raise stop

    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob(stopped))

    # Act
    with pytest.raises(type(stop)) as raised:
        handler(_dispatch_job())

    # Assert
    assert raised.value is stop


# --- a fresh worker once the process abandoned a storage request --------------------------------------


def _start(work: Callable[[], None] = lambda: None) -> tuple[dict[str, Any], SealedMessage]:
    """One granted start of this process whose job does `work`: the handler's output and the job's sealed result."""
    clock = FakeClock()
    stand_in = _StandInJob(work)
    output = job_handler(_runtime(_transport(clock, _granted()), clock), stand_in)(_dispatch_job())
    return output, stand_in.returned[0]


def test_handler_when_a_storage_request_is_abandoned_during_the_start_should_add_refresh_worker_to_the_carrier(
    caplog, abandoned
):
    # Arrange: the job gives up a storage request, then reports its result
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, _granted())
    stand_in = _StandInJob(abandoned.grow)

    # Act
    output = job_handler(_runtime(transport, clock), stand_in)(_dispatch_job())

    # Assert: the carrier, unchanged, and the one member the SDK takes out of it; the Hub got the carrier
    (sealed,) = stand_in.returned
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }
    assert _posted_to_the_callback(transport) == [(sealed.body, sealed.signature)]
    assert len(_entry_records(caplog, logging.WARNING)) == 1
    assert _shown(SECRETS, json.dumps(output), *_log_lines(caplog)) == []


def test_handler_when_a_storage_request_was_abandoned_in_an_earlier_start_should_ask_again_for_a_fresh_worker(
    abandoned,
):
    # Arrange: an earlier start of this process abandoned a request, whose thread belongs to the process
    earlier_output, _ = _start(abandoned.grow)

    # Act: a later start that abandons nothing itself
    output, sealed = _start()

    # Assert: the process stays marked, so every output asks until the process ends
    assert earlier_output[REFRESH_WORKER_MEMBER] is True
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }


def test_handler_when_a_request_is_abandoned_while_no_job_runs_should_ask_for_a_fresh_worker_in_the_next_job(
    abandoned,
):
    # Arrange: a start returns; only then does a thread it left behind give up its request
    first_output, first_sealed = _start()
    abandoned.grow()

    # Act
    output, sealed = _start()

    # Assert: the first job could not know; the next one that returns asks
    assert first_output == {"vh_message": first_sealed.body.decode("ascii"), "vh_signature": first_sealed.signature}
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }


def _refused_once_a_request_was_abandoned(abandoned: _AbandonedCount) -> None:
    """A request is given up while no job runs; the next job's dispatch is refused."""
    abandoned.grow()
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob())
    with pytest.raises(JobRefused):
        handler(_job(signed_carrier(REQUEST_WITH_CANARY, UNCONFIGURED_KEY)))


def _failed_on_a_bug_after_abandoning_a_request(abandoned: _AbandonedCount) -> None:
    """A job gives up a request and then fails on a bug."""

    def abandon_then_fail() -> None:
        abandoned.grow()
        _fail()

    with pytest.raises(WorkerFault):
        _start(abandon_then_fail)


@pytest.mark.parametrize(
    "start_that_raised",
    [
        pytest.param(_refused_once_a_request_was_abandoned, id="a-refused-dispatch"),
        pytest.param(_failed_on_a_bug_after_abandoning_a_request, id="a-bug"),
    ],
)
def test_handler_when_a_start_raised_after_a_request_was_abandoned_should_ask_for_a_fresh_worker_in_the_next_job(
    abandoned, start_that_raised
):
    # Arrange: a failed job has no output that could ask
    start_that_raised(abandoned)

    # Act
    output, sealed = _start()

    # Assert
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }


def test_handler_when_the_warning_of_a_marked_process_cannot_be_written_should_still_return_the_output(
    abandoned, monkeypatch
):
    # Arrange: the process is marked, and its log fails on every warning
    abandoned.grow()
    monkeypatch.setattr(runpod_handler.logger, "warning", _failing_log)

    # Act
    output, sealed = _start()

    # Assert: the finished job's result is not lost with the line
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }


def test_handler_when_a_storage_read_is_really_abandoned_during_a_start_should_count_it_and_ask_for_a_fresh_worker(
    caplog, hold, the_real_count
):
    # Arrange: the wrapper reads the process's real count; the job gives up a read the storage never answers
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    transport = _transport(clock, _granted())
    stand_in = _StandInJob(_abandoning_a_storage_read(hold))
    counted_before = the_real_count()

    # Act
    output = job_handler(_runtime(transport, clock), stand_in)(_dispatch_job())

    # Assert: the request was under way, it was counted, and the output of the start asks
    (sealed,) = stand_in.returned
    assert hold.entered.is_set()
    assert the_real_count() - counted_before == 1
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }
    assert _posted_to_the_callback(transport) == [(sealed.body, sealed.signature)]
    assert _shown(SECRETS, json.dumps(output), *_log_lines(caplog)) == []


def test_handler_when_the_storage_itself_abandons_a_request_should_read_its_count_and_ask_for_a_fresh_worker(
    hold, the_real_count
):
    # Arrange: nothing of the wrapper is replaced, and the process has abandoned no request, so a start
    # returns exactly its carrier. Then, while no job runs, the storage gives up a request that never
    # returns, at a real, short deadline
    output_before, sealed_before = _start()
    with pytest.raises(WorkerFailure):
        storage._within(Deadline.after_ms(SHORT_MS), hold.keep, expired=REQUEST_GIVEN_UP)

    # Act: the next start of the process, whose own job abandons nothing
    output, sealed = _start()

    # Assert: the request was under way and the storage counted it, once; the wrapper read that count,
    # so the output of the next start asks
    assert hold.entered.wait(PROMPT_S)
    assert the_real_count() == 1
    assert output_before == {
        "vh_message": sealed_before.body.decode("ascii"),
        "vh_signature": sealed_before.signature,
    }
    assert output == {
        "vh_message": sealed.body.decode("ascii"),
        "vh_signature": sealed.signature,
        "refresh_worker": True,
    }


# --- the process's handler ------------------------------------------------------------------------


def _whole_env(scratch_dir: Path) -> dict[str, str]:
    """A whole, valid configuration of the worker: the protocol's variables and the job's."""
    return {**worker_env(), IMAGE_DIGEST_VARIABLE: IMAGE_DIGEST, SCRATCH_DIR_VARIABLE: str(scratch_dir)}


def _changed(env: dict[str, str], changes: dict[str, str | None]) -> dict[str, str]:
    """The environment with some variables replaced (a value) or removed (None)."""
    kept = {name: value for name, value in env.items() if name not in changes}
    return {**kept, **{name: value for name, value in changes.items() if value is not None}}


def test_build_handler_when_the_tree_and_the_configuration_are_whole_should_return_a_handler(tmp_path):
    # Arrange
    env = _whole_env(tmp_path)

    # Act
    handler = build_handler(env)

    # Assert
    assert callable(handler)


def test_build_handler_when_its_handler_gets_a_dispatch_no_configured_key_signed_should_refuse_it_and_do_no_work(
    tmp_path,
):
    # Arrange: the real runtime and the real job runner; a refused dispatch is never claimed, so no network
    handler = build_handler(_whole_env(tmp_path))

    # Act
    with pytest.raises(JobRefused) as raised:
        handler(_job(signed_carrier(REQUEST_WITH_CANARY, UNCONFIGURED_KEY)))

    # Assert: refused by the configured keys, and the scratch directory of the configuration is untouched
    assert str(raised.value) == "the dispatch is refused at signature"
    assert list(tmp_path.iterdir()) == []


def test_build_handler_when_its_handler_gets_a_granted_dispatch_should_run_it_with_the_job_runner_it_built(
    tmp_path, monkeypatch
):
    # Arrange: the real wiring around the test's runtime (a Hub that grants) and a job runner whose `run`
    # only records what it was handed and returns a sealed result
    clock = FakeClock()
    runtime = _runtime(_transport(clock, _granted()), clock)
    stand_in = _StandInJob()
    runners: list[JobRunner] = []

    def run(runner: JobRunner, claimed: ClaimedExecution, handed_runtime: Runtime) -> SealedMessage:
        runners.append(runner)
        return stand_in(claimed, handed_runtime)

    monkeypatch.setattr(runpod_handler, "load_runtime", lambda env: runtime)
    monkeypatch.setattr(JobRunner, "run", run)

    # Act
    output = build_handler(_whole_env(tmp_path))(_dispatch_job())

    # Assert: the runner the wiring built ran the claimed execution, and its result is the job's output
    ((execution, handed_runtime),) = stand_in.calls
    (sealed,) = stand_in.returned
    assert isinstance(execution, ClaimedExecution)
    assert handed_runtime is runtime
    assert [type(runner) for runner in runners] == [JobRunner]
    assert output == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}


def test_build_handler_when_a_killed_process_left_a_scratch_directory_should_remove_it_before_any_job(tmp_path):
    # Arrange: in the configured scratch root, a directory named as the job runner names an execution's,
    # with the source a killed execution had downloaded, last changed long ago and locked by nobody
    leftover = tmp_path / (SCRATCH_PREFIX + "left-by-a-killed-process")
    leftover.mkdir()
    (leftover / "source").write_bytes(b"the source a killed execution had downloaded")
    changed_at_s = time.time() - LEFT_LONG_AGO_S
    os.utime(leftover, (changed_at_s, changed_at_s))

    # Act
    handler = build_handler(_whole_env(tmp_path))

    # Assert: the process starts with an empty scratch root
    assert callable(handler)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("changes", "variable"),
    [
        pytest.param({"VH_CLAIM_URL": None}, "VH_CLAIM_URL", id="a-missing-protocol-variable"),
        pytest.param(
            {"VH_WORKER_TO_HUB_KEY": CONFIG_CANARY}, "VH_WORKER_TO_HUB_KEY", id="an-invalid-protocol-variable"
        ),
        pytest.param({IMAGE_DIGEST_VARIABLE: None}, "VH_IMAGE_DIGEST", id="a-missing-image-digest"),
        pytest.param(
            {IMAGE_DIGEST_VARIABLE: f"sha256:{CONFIG_CANARY}"}, "VH_IMAGE_DIGEST", id="an-invalid-image-digest"
        ),
    ],
)
def test_build_handler_when_the_configuration_is_not_whole_should_raise_settings_invalid_showing_no_value(
    tmp_path, changes, variable
):
    # Arrange
    env = _changed(_whole_env(tmp_path), changes)

    # Act
    with pytest.raises(SettingsInvalid) as raised:
        build_handler(env)

    # Assert: the variable and the rule; no configured value, in the text or in what a crash would print
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert raised.value.variable == variable
    assert [value for value in env.values() if value in str(raised.value)] == []
    assert _shown((CONFIG_CANARY, *SECRETS), recorded) == []


def test_build_handler_when_the_process_environment_is_whole_should_still_read_only_the_mapping_it_is_given(
    tmp_path, monkeypatch
):
    # Arrange: the process's own environment holds a whole configuration; the mapping given holds none
    for name, value in _whole_env(tmp_path).items():
        monkeypatch.setenv(name, value)

    # Act
    with pytest.raises(SettingsInvalid) as raised:
        build_handler({})

    # Assert: a variable the process environment holds is missing from the mapping, so it is missing
    assert raised.value.rule == "is required"


def test_build_handler_when_the_contract_tree_was_tampered_with_should_raise_contract_tree_invalid(
    tmp_path, monkeypatch
):
    # Arrange: the real start-up check, pointed at a copy of the tree with one more byte in one file
    tree_copy, scratch_dir = tmp_path / "v2", tmp_path / "scratch"
    shutil.copytree(files.CONTRACT_ROOT, tree_copy)
    scratch_dir.mkdir()
    tampered = tree_copy / TAMPERED_FILE
    tampered.write_bytes(tampered.read_bytes() + b" ")
    verify_contract_tree = files.verify_contract_tree
    monkeypatch.setattr(files, "verify_contract_tree", lambda: verify_contract_tree(root=tree_copy))
    env = _whole_env(scratch_dir)

    # Act
    with pytest.raises(ContractTreeInvalid) as raised:
        build_handler(env)

    # Assert: the file and the rule, no configured value
    assert "SHA-256 of limits.json differs" in str(raised.value)
    assert [value for value in env.values() if value in str(raised.value)] == []


def test_build_handler_when_the_tree_and_the_configuration_are_both_broken_should_refuse_the_tree_first(
    tmp_path, monkeypatch
):
    # Arrange
    tree_copy = tmp_path / "v2"
    shutil.copytree(files.CONTRACT_ROOT, tree_copy)
    (tree_copy / TAMPERED_FILE).unlink()
    verify_contract_tree = files.verify_contract_tree
    monkeypatch.setattr(files, "verify_contract_tree", lambda: verify_contract_tree(root=tree_copy))

    # Act / Assert
    with pytest.raises(ContractTreeInvalid):
        build_handler({})


# --- the process's log ----------------------------------------------------------------------------


def _logging_state() -> list[tuple[list[logging.Handler], int, bool]]:
    """The handlers, the level and the handing-on of the root logger and of the HTTP library's."""
    return [
        (list(logger.handlers), logger.level, logger.propagate)
        for logger in (logging.getLogger(), logging.getLogger(HTTP_LIBRARY_LOGGER))
    ]


@contextlib.contextmanager
def _logging_nobody_configured() -> Iterator[None]:
    """Inside, the root logger is as in a process that has just started: no handler, its default level.
    The HTTP library's logger is as nobody touched it either, and barer than the library leaves it: no
    handler (the library hangs one of its own on it), no level, its records handed on to the root.
    On leaving, the handlers, the level and the handing-on of both are what they were before, so that no
    other test meets what was configured inside.

    It is a context of the test's body and not a fixture: pytest hangs handlers of its own on the root
    logger for each phase of a test (set-up, call, tear-down) and takes them off when the phase ends, so
    the handlers a fixture saved at set-up are not the ones to put back at tear-down.
    """
    root, library = logging.getLogger(), logging.getLogger(HTTP_LIBRARY_LOGGER)
    state_before = _logging_state()
    for logger in (root, library):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    root.setLevel(logging.WARNING)
    library.setLevel(logging.NOTSET)
    library.propagate = True
    try:
        yield
    finally:
        for logger, (handlers, level, propagate) in zip((root, library), state_before):
            for configured in list(logger.handlers):
                logger.removeHandler(configured)
                configured.close()
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


class _Records(logging.Handler):
    """Keeps the records it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _server_with_a_malformed_header() -> Iterator[int]:
    """A server on the loopback address that answers one request with an empty 200 whose second header
    line has no colon; gives its port."""
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(LOOPBACK_TIMEOUT_S)

    def answer_one_request() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(LOOPBACK_TIMEOUT_S)
                received = b""
                while b"\r\n\r\n" not in received:
                    chunk = connection.recv(65536)
                    if not chunk:
                        return
                    received += chunk
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\na header line without its colon\r\n"
                    b"Connection: close\r\n\r\n"
                )
        except OSError:
            # The test's own assertions say what did not arrive.
            return

    server = threading.Thread(target=answer_one_request, daemon=True)
    server.start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()
        server.join(LOOPBACK_TIMEOUT_S)


def test_configure_logging_when_called_in_a_fresh_process_should_log_the_worker_at_info_through_one_handler():
    # Arrange
    state_before = _logging_state()

    # Act
    with _logging_nobody_configured():
        runpod_handler.configure_logging()
        worker_levels = [logging.getLogger(name).getEffectiveLevel() for name in WORKER_LOGGERS]
        root_handlers = len(logging.getLogger().handlers)

    # Assert: one handler for the whole process and the worker's loggers at INFO; and the logging state of
    # the test session is what it was
    assert worker_levels == [logging.INFO, logging.INFO, logging.INFO]
    assert root_handlers == 1
    assert _logging_state() == state_before


@pytest.mark.parametrize(("logger_name", "level", "line", "arguments"), HTTP_LIBRARY_LINES)
def test_configure_logging_when_the_http_library_logs_what_it_talks_to_should_keep_the_storage_out_of_the_log(
    logger_name, level, line, arguments, capsys,
):
    # Arrange: the library's logger and one of the worker's own
    library, worker = logging.getLogger(logger_name), logging.getLogger("core.job.storage")

    # Act: the process's log goes to standard error. The SDK is imported after the set-up, and its import
    # gives the parent of the library's loggers a level of its own (runpod/__init__.py: WARNING)
    with _logging_nobody_configured():
        runpod_handler.configure_logging()
        logging.getLogger(HTTP_LIBRARY_LOGGER).setLevel(logging.WARNING)
        library.log(level, line, *arguments)
        worker.info("a line of the worker's own")
        written = capsys.readouterr().err

    # Assert: the worker's line is there, in the process's format; nothing of the library's line is
    assert " INFO core.job.storage: a line of the worker's own\n" in written
    assert [canary for canary in STORAGE_CANARIES if canary in written] == []
    assert logger_name not in written


def test_configure_logging_when_the_real_http_library_reads_a_malformed_header_should_keep_the_request_out_of_the_log(
    capsys,
):
    # Arrange: a storage whose answer has a header line without its colon, which the library reports with
    # the request's whole URL; and a handler on the library's own logger, which still gets what it writes
    urllib3 = pytest.importorskip("urllib3")
    library, worker = logging.getLogger(HTTP_LIBRARY_LOGGER), logging.getLogger("core.job.storage")
    reported = _Records()

    # Act: one request of the real library, after the set-up and the level the SDK's import leaves
    with _server_with_a_malformed_header() as port, _logging_nobody_configured():
        runpod_handler.configure_logging()
        library.setLevel(logging.WARNING)
        library.addHandler(reported)
        with urllib3.PoolManager(retries=False, timeout=LOOPBACK_TIMEOUT_S) as pool:
            answer = pool.request("GET", f"http://127.0.0.1:{port}/{STORAGE_BUCKET_CANARY}/{STORAGE_KEY_CANARY}")
        worker.info("a line of the worker's own")
        written = capsys.readouterr().err

    # Assert: the library did report the answer, with the bucket and the key; the process's log has the
    # worker's line and nothing of the report
    reports = [record.getMessage() for record in reported.records]
    assert answer.status == 200
    assert [report for report in reports if STORAGE_KEY_CANARY in report and STORAGE_BUCKET_CANARY in report] != []
    assert " INFO core.job.storage: a line of the worker's own\n" in written
    assert [canary for canary in STORAGE_CANARIES if canary in written] == []
    assert "urllib3" not in written


# --- importing the entry point --------------------------------------------------------------------


def test_entry_point_when_imported_by_a_fresh_interpreter_should_load_no_sdk_or_v1_module_and_configure_no_logging():
    # Arrange: an interpreter of its own, with no configuration of the worker in its environment
    code = (
        "import logging\n"
        "import sys\n"
        "import wrappers.runpod_handler\n"
        f"print([name for name in {NEVER_IMPORTED!r} if name in sys.modules])\n"
        "print(logging.getLogger().handlers)\n"
        "print(logging.getLogger().level)\n"
        f"print(logging.getLogger({HTTP_LIBRARY_LOGGER!r}).propagate)\n"
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("VH_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Act
    imported = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=IMPORT_TIMEOUT_S, check=False,
    )

    # Assert: it imported, without a configuration to build anything from; it loaded none of the modules,
    # and the logging is as nobody configured it: no handler on the root logger, its default level (30,
    # WARNING), and the HTTP library's logger still hands its records on
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.splitlines() == ["[]", "[]", "30", "True"]


# --- the entry point run as the program -----------------------------------------------------------


class _Starter:
    """Stands where the SDK's serverless loop is started: it keeps the configuration it is started with,
    does `when_started` and returns, taking no job."""

    def __init__(self, when_started: Callable[[], None] = lambda: None) -> None:
        self.configurations: list[dict[str, Any]] = []
        self._when_started = when_started

    def __call__(self, configuration: dict[str, Any]) -> None:
        self.configurations.append(configuration)
        self._when_started()


class _EnvironmentReadOnce(Mapping[str, str]):
    """An environment that does `when_first_read` the first time anything is read from it: the moment the
    handler begins to be built from it."""

    def __init__(self, variables: Mapping[str, str], when_first_read: Callable[[], None]) -> None:
        self._variables = dict(variables)
        self._when_first_read = when_first_read
        self._was_read = False

    def __getitem__(self, name: str) -> str:
        self._note_the_read()
        return self._variables[name]

    def __iter__(self) -> Iterator[str]:
        self._note_the_read()
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def _note_the_read(self) -> None:
        if not self._was_read:
            self._was_read = True
            self._when_first_read()


def _write_a_line_of_each(words: str) -> None:
    """Log a warning of the HTTP library that names a storage request, and `words` as a line of the
    worker's own: a log that is set up shows the second, in the process's format, and nothing of the first."""
    logging.getLogger(HTTP_LIBRARY_REQUEST_LOGGER).warning("Failed to parse headers (url=%s)", STORAGE_URL_CANARY)
    logging.getLogger(ENTRY_LOGGER).info(words)


def test_main_when_given_a_starter_should_set_up_the_log_before_the_handler_is_built_and_before_the_starter_is_called(
    tmp_path, monkeypatch, capsys
):
    # Arrange: the environment writes its two lines when the handler begins to be built from it, and the
    # starter writes its own when it is called. The SDK cannot be imported here, installed or not
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)
    env = _EnvironmentReadOnce(_whole_env(tmp_path), lambda: _write_a_line_of_each(WHILE_THE_HANDLER_IS_BUILT))
    starter = _Starter(lambda: _write_a_line_of_each(WHEN_THE_LOOP_IS_STARTED))

    # Act: in a process whose logging nobody configured, as the image's is when it runs the file
    with _logging_nobody_configured():
        runpod_handler.main(starter, env)
        written = capsys.readouterr().err

    # Assert: each moment met a log that was set up: its line of the worker's own is there, in the
    # process's format, and nothing of the HTTP library's line is
    assert f" INFO {ENTRY_LOGGER}: {WHILE_THE_HANDLER_IS_BUILT}\n" in written
    assert f" INFO {ENTRY_LOGGER}: {WHEN_THE_LOOP_IS_STARTED}\n" in written
    assert [canary for canary in STORAGE_CANARIES if canary in written] == []


def test_main_when_given_a_starter_and_an_environment_should_start_the_loop_once_with_the_handler_of_that_environment(
    tmp_path, monkeypatch
):
    # Arrange: a whole configuration in the mapping given and none in the process's own environment. The
    # SDK cannot be imported here, installed or not: the starter given is the one that is used
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)
    for name in [name for name in os.environ if name.startswith("VH_")]:
        monkeypatch.delenv(name)
    starter = _Starter()

    # Act
    with _logging_nobody_configured():
        runpod_handler.main(starter, _whole_env(tmp_path))

    # Assert: one start, with the one member the SDK reads the handler from; and that handler is the real
    # one of the configuration given: it refuses a dispatch no configured key signed, in the refusal's words
    (configuration,) = starter.configurations
    assert sorted(configuration) == [HANDLER_MEMBER]
    with pytest.raises(JobRefused) as raised:
        configuration[HANDLER_MEMBER](_job(signed_carrier(REQUEST_WITH_CANARY, UNCONFIGURED_KEY)))
    assert str(raised.value) == "the dispatch is refused at signature"


def test_entry_point_when_run_as_the_program_should_set_up_the_log_then_import_the_sdk_and_start_it_with_a_handler(
    tmp_path,
):
    # Arrange: a process of its own, so that nothing of this one's logging or modules is touched. Its
    # environment holds a whole configuration, and a stand-in for the SDK comes first on its module path,
    # so that the SDK is not needed and, where it is installed, not the one that is imported
    sdk_path, scratch_dir = tmp_path / "sdk", tmp_path / "scratch"
    (sdk_path / SDK_MODULE).mkdir(parents=True)
    (sdk_path / SDK_MODULE / "__init__.py").write_text(STAND_IN_SDK, encoding="utf-8")
    scratch_dir.mkdir()
    env = {name: value for name, value in os.environ.items() if not name.startswith("VH_")}
    env.update(_whole_env(scratch_dir))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(sdk_path), os.environ.get("PYTHONPATH"))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Act: the file is run as the image runs it
    started = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "wrappers" / "runpod_handler.py")],
        cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=PROGRAM_TIMEOUT_S, check=False,
    )

    # Assert: the SDK's loop was started, with a handler built from the process's environment. The log
    # was set up before the SDK was imported: the worker's line is in the process's format, not in the
    # one the SDK's import would have given a root logger without a handler, and the log holds nothing
    # of the HTTP library's line
    assert started.returncode == 0, started.stderr
    assert started.stdout.splitlines() == ['{"members": ["handler"], "handler_is_callable": true}']
    assert f" INFO core.job.storage: {WHEN_THE_LOOP_IS_STARTED}\n" in started.stderr
    assert [canary for canary in STORAGE_CANARIES if canary in started.stderr] == []


# --- through the SDK's own run_job ------------------------------------------------------------------


@pytest.fixture
def run_through_the_sdk() -> Callable[[Callable[[Any], dict[str, Any]], Any], dict[str, Any]]:
    """Runs one job through the SDK's run_job, as the worker loop does, and returns what the SDK reports of it.

    run_job calls the handler, takes `refresh_worker` out of a returned dict and turns a raised error into
    the job's error; it opens no connection. The handler does not wait (the transport and the clock are
    the test's), and the wait_for is a belt for whatever the SDK itself awaits.
    """
    pytest.importorskip("runpod")
    from runpod.serverless.modules.rp_job import run_job

    def run(handler: Callable[[Any], dict[str, Any]], job: Any) -> dict[str, Any]:
        return asyncio.run(asyncio.wait_for(run_job(handler, job), SDK_RUN_TIMEOUT_S))

    return run


def test_sdk_run_job_when_the_process_abandoned_no_request_should_record_exactly_the_carrier_and_not_stop_the_pod(
    run_through_the_sdk, capsys
):
    # Arrange
    clock = FakeClock()
    stand_in = _StandInJob()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), stand_in)

    # Act
    result = run_through_the_sdk(handler, _dispatch_job())

    # Assert
    (sealed,) = stand_in.returned
    printed = capsys.readouterr()
    assert result == {"output": {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}}
    assert _shown(SECRETS, json.dumps(result), printed.out, printed.err) == []


def _abandoning_during_the_job(abandoned: _AbandonedCount) -> Callable[[], None]:
    """The work of a job that gives up a storage request."""
    return abandoned.grow


def _abandoned_before_the_job(abandoned: _AbandonedCount) -> Callable[[], None]:
    """A request is given up while no job runs; the work of the job that follows abandons nothing."""
    abandoned.grow()
    return lambda: None


@pytest.mark.parametrize(
    "work_of_a_marked_process",
    [
        pytest.param(_abandoning_during_the_job, id="abandoned-during-this-job"),
        pytest.param(_abandoned_before_the_job, id="abandoned-before-this-job"),
    ],
)
def test_sdk_run_job_when_the_process_abandoned_a_request_should_record_exactly_the_carrier_and_stop_the_pod(
    run_through_the_sdk, abandoned, capsys, work_of_a_marked_process
):
    # Arrange: the process abandons a request, in this job or while none ran: either way it is marked
    clock = FakeClock()
    stand_in = _StandInJob(work_of_a_marked_process(abandoned))
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), stand_in)

    # Act
    result = run_through_the_sdk(handler, _dispatch_job())

    # Assert: the Hub reads the two members of the carrier; the provider is told to stop the pod
    (sealed,) = stand_in.returned
    printed = capsys.readouterr()
    assert result == {
        "output": {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature},
        "stopPod": True,
    }
    assert _shown(SECRETS, json.dumps(result), printed.out, printed.err) == []


def test_sdk_run_job_when_the_dispatch_is_refused_should_record_the_refusals_words_and_nothing_of_the_input(
    run_through_the_sdk, capsys
):
    # Arrange
    clock = FakeClock()
    handler = job_handler(_runtime(_transport(clock, _granted()), clock), _StandInJob())

    # Act
    result = run_through_the_sdk(handler, _job(signed_carrier(REQUEST_WITH_CANARY, UNCONFIGURED_KEY)))

    # Assert: a failed job, no output; the error and its traceback as the provider keeps and prints them
    recorded = json.loads(result["error"])
    printed = capsys.readouterr()
    assert sorted(result) == ["error"]
    assert recorded["error_message"] == "the dispatch is refused at signature"
    assert "JobRefused" in recorded["error_type"]
    assert _shown(CHAIN_MARKERS, recorded["error_traceback"]) == []
    assert _shown(SECRETS, result["error"], printed.out, printed.err) == []


def test_sdk_run_job_when_the_start_fails_on_a_bug_should_record_an_error_naming_the_class_and_nothing_of_its_text(
    run_through_the_sdk, capsys
):
    # Arrange
    clock = FakeClock()
    handler = job_handler(
        _runtime(_transport(clock, _granted()), clock), _StandInJob(_fail_because_of_another_error)
    )

    # Act
    result = run_through_the_sdk(handler, _dispatch_job())

    # Assert
    recorded = json.loads(result["error"])
    printed = capsys.readouterr()
    assert sorted(result) == ["error"]
    assert recorded["error_message"] == "the worker failed on an unexpected LookupError"
    assert "WorkerFault" in recorded["error_type"]
    assert _shown(CHAIN_MARKERS, recorded["error_traceback"]) == []
    assert _shown((BUG_CANARY, CHAINED_CANARY, *SECRETS), result["error"], printed.out, printed.err) == []
