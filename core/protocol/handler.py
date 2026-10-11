"""One start of the worker for one dispatch under contract v2: accept, claim, then work or report.

load_runtime() runs once per process, before any job: it checks the contract tree the image ships
(files.verify_contract_tree) and reads the worker's configuration (settings.WorkerSettings); either
failing stops the process, so a worker never serves a job with a tree or a configuration it cannot trust.

handle() is one runtime (README 15.1): it accepts the dispatch or raises DispatchRefused (nothing of a
refused dispatch is trusted, so nothing is reported under its identity and the provider marks the job
failed), draws a fresh random runtime id that lives in memory only, and claims. Unclaimed, it reports
`transcode.unclaimed` and does no work. Granted, it hands the claimed execution to the job runner, which
returns the execution's result. Either message is posted to the callback address (best effort) and
returned as the job's output, the same signed bytes both ways.

One start has one wall-time budget. handle() reads the runtime's monotonic clock before anything else,
and that reading plus the dispatch's `limits.max_wall_time_ms` is `wall_time_ends_at`, the instant the
start's own waits are held to: no claim is sent at or after it, the job runner gets what is left of it
as its deadline (it travels in the ClaimedExecution), and no callback attempt starts at or after it. A
request that started just before it may take its least time-out past it. A provider bills a
start by the second, so the time the claim took is the job's time too. The job output is returned
whatever the clock says: it carries the same signed bytes as the callback.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping

from . import files, messages
from .claim import Grant, Unclaimed, claim_execution
from .dispatch import AcceptedDispatch, accept_dispatch
from .outbox import SealedMessage, deliver, runpod_output, seal_message
from .registry import MessageKind
from .settings import WorkerSettings
from .transport import HttpsTransport, Transport

logger = logging.getLogger(__name__)

_MS_PER_SECOND = 1000


@dataclass(frozen=True)
class Runtime:
    """What every start of this process shares: the configuration, the carrier, the clocks and the id source.

    `clock` is the aware wall-clock time the contract's timestamps are judged on; `monotonic` gives seconds
    that only move forward, for the worker's own waits.
    """

    settings: WorkerSettings
    transport: Transport
    clock: Callable[[], datetime]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    new_uuid: Callable[[], str]


@dataclass(frozen=True)
class ClaimedExecution:
    """A dispatch with the grant that made this runtime its execution; repr() does not show the runtime id.

    `wall_time_ends_at` is the reading of the runtime's monotonic clock at which the start's wall time
    ends: the reading handle() took first, plus the dispatch's limit. It is only ever compared with
    another reading of that clock. Anything but a finite number is refused: TypeError for another type
    (a bool included), ValueError for a number that is not finite.
    """

    dispatch: AcceptedDispatch
    grant: Grant
    runtime_id: str = field(repr=False)
    wall_time_ends_at: float

    def __post_init__(self) -> None:
        ends_at = self.wall_time_ends_at
        if isinstance(ends_at, bool) or not isinstance(ends_at, (int, float)):
            raise TypeError("wall_time_ends_at must be a number: a reading of the monotonic clock")
        if not math.isfinite(ends_at):
            raise ValueError("wall_time_ends_at must be finite")


# Runs a claimed execution to its end and returns its sealed `transcode.result.completed` or `.failed`.
JobRunner = Callable[[ClaimedExecution, Runtime], SealedMessage]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _random_uuid() -> str:
    return str(uuid.uuid4())


def load_runtime(env: Mapping[str, str]) -> Runtime:
    """The process's runtime, or an error: ContractTreeInvalid for the tree, SettingsInvalid for the configuration."""
    files.verify_contract_tree()
    return Runtime(
        settings=WorkerSettings.from_env(env),
        transport=HttpsTransport(),
        clock=_utc_now,
        monotonic=time.monotonic,
        sleep=time.sleep,
        new_uuid=_random_uuid,
    )


def handle(job_input: object, *, runtime: Runtime, run_job: JobRunner) -> dict[str, str]:
    """Handle one job input and return the job's output carrier; raises DispatchRefused for a refused dispatch."""
    # Read before anything else: the start's wall time counts from here, whatever accepting and claiming take.
    started_at = runtime.monotonic()
    settings = runtime.settings
    dispatch = accept_dispatch(job_input, settings=settings, now=runtime.clock())
    wall_time_ends_at = started_at + dispatch.limits.max_wall_time_ms / _MS_PER_SECOND
    # Fresh and random for every start, never derived from the dispatch or the provider's job (README 15.1, rule 1).
    runtime_id = runtime.new_uuid()
    outcome = claim_execution(
        dispatch,
        settings=settings,
        runtime_id=runtime_id,
        transport=runtime.transport,
        clock=runtime.clock,
        monotonic=runtime.monotonic,
        sleep=runtime.sleep,
        new_message_id=runtime.new_uuid,
        wall_time_ends_at=wall_time_ends_at,
    )
    if isinstance(outcome, Unclaimed):
        report = _unclaimed_report(dispatch, runtime_id, outcome, runtime)
    else:
        claimed = ClaimedExecution(
            dispatch=dispatch, grant=outcome, runtime_id=runtime_id, wall_time_ends_at=wall_time_ends_at
        )
        report = run_job(claimed, runtime)
    output = runpod_output(report)
    _deliver_best_effort(report, runtime, wall_time_ends_at)
    return output


def _unclaimed_report(
    dispatch: AcceptedDispatch, runtime_id: str, outcome: Unclaimed, runtime: Runtime
) -> SealedMessage:
    now = runtime.clock()
    report = messages.unclaimed(
        protocol_version=dispatch.protocol_version,
        message_id=runtime.new_uuid(),
        sent_at=now,
        key_id=runtime.settings.worker_to_hub_key_id,
        identity=dispatch.identity,
        runtime_id=runtime_id,
        cause=outcome.cause,
        hub_error_code=outcome.hub_error_code,
    )
    context = messages.worker_sender_context(
        MessageKind.TRANSCODE_UNCLAIMED, dispatch.identity, settings=runtime.settings, now=now
    )
    return seal_message(report, context, settings=runtime.settings)


def _deliver_best_effort(report: SealedMessage, runtime: Runtime, wall_time_ends_at: float) -> None:
    """The callback is the fast path, the job output the one the Hub can always read: no failure here costs the output.

    No attempt of it starts at or after `wall_time_ends_at`, a reading of the runtime's monotonic clock.
    """
    try:
        delivered = deliver(
            report,
            settings=runtime.settings,
            transport=runtime.transport,
            sleep=runtime.sleep,
            give_up_at=wall_time_ends_at,
            monotonic=runtime.monotonic,
        )
    except Exception as error:  # noqa: BLE001 - whatever fails here, the output is still returned
        logger.error("callback of a %s failed: %s", report.kind.value, type(error).__name__)
        return
    logger.info("callback of a %s %s", report.kind.value, "delivered" if delivered else "not delivered")
