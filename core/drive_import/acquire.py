"""One acquire runtime: signed dispatch, claim, transfer and one terminal on three carriers."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime

from core.protocol import acquire_messages, messages
from core.protocol.acquire_claim import AcquireGrant, claim_execution
from core.protocol.acquire_dispatch import AcceptedAcquire, accept_dispatch
from core.protocol.claim import Unclaimed
from core.protocol.outbox import SealedMessage, deliver, runpod_output, seal_message
from core.protocol.registry import MessageKind
from core.protocol.runtime import Runtime
from core.protocol.settings import AcquireSettings
from core.protocol.transport import TransportFailed

from . import store
from .clock import DeadlineReached, StopLine
from .failure import Code, ImportFailed, Stage
from .transfer import Part, Tally, TransferPlan, TransferResult, transfer
from .transport import TRANSFER_ERRORS, HttpSession

logger = logging.getLogger(__name__)

MS_PER_SECOND = 1_000
RESULT_HEADERS = {"Content-Type": "application/json"}
STAGES: dict[Stage, str | None] = {
    Stage.INPUT: None, Stage.CONFIGURATION: None, Stage.METADATA: "downloading",
    Stage.DOWNLOAD: "downloading", Stage.UPLOAD: "uploading", Stage.VERIFY: "finalizing",
    Stage.RECEIPT: "finalizing",
}


@dataclass(frozen=True)
class ClaimedAcquire:
    dispatch: AcceptedAcquire
    grant: AcquireGrant
    runtime_id: str = field(repr=False)
    wall_time_ends_at: float


class _Clock:
    def __init__(self, runtime: Runtime[AcquireSettings]) -> None:
        self._runtime = runtime

    def now(self) -> datetime:
        return self._runtime.clock()

    def monotonic(self) -> float:
        return self._runtime.monotonic()

    def sleep(self, seconds: float) -> None:
        self._runtime.sleep(seconds)


def handle(
    job_input: object, *, runtime: Runtime[AcquireSettings], http: HttpSession,
    started_at: float | None = None, runtime_id: str | None = None,
) -> dict[str, str]:
    start = runtime.monotonic() if started_at is None else started_at
    identifier = runtime.new_uuid() if runtime_id is None else runtime_id
    dispatch = accept_dispatch(job_input, settings=runtime.settings, now=runtime.clock())
    ends_at = start + dispatch.limits.max_wall_time_ms / MS_PER_SECOND
    outcome = claim_execution(
        dispatch, settings=runtime.settings, runtime_id=identifier, transport=runtime.transport,
        clock=runtime.clock, monotonic=runtime.monotonic, sleep=runtime.sleep,
        new_message_id=runtime.new_uuid, wall_time_ends_at=ends_at,
    )
    if isinstance(outcome, AcquireGrant):
        report = run_acquire(ClaimedAcquire(dispatch, outcome, identifier, ends_at), runtime, http)
    elif isinstance(outcome, Unclaimed):
        now = runtime.clock()
        message = messages.unclaimed(
            protocol_version=dispatch.protocol_version, message_id=runtime.new_uuid(), sent_at=now,
            key_id=runtime.settings.worker_to_hub_key_id, identity=dispatch.identity, runtime_id=identifier,
            cause=outcome.cause, hub_error_code=outcome.hub_error_code,
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_UNCLAIMED, dispatch.identity, now=now,
            settings=replace(runtime.settings, accepted_protocol_versions=(dispatch.protocol_version,)),
        )
        report = seal_message(message, context, settings=runtime.settings)
    else:
        raise TypeError("the acquire claim returned no recognized outcome")
    output = runpod_output(report)
    _callback(report, runtime, ends_at)
    return output


def run_acquire(claimed: ClaimedAcquire, runtime: Runtime[AcquireSettings], http: HttpSession) -> SealedMessage:
    stop = StopLine(_Clock(runtime), claimed.wall_time_ends_at)
    dispatch, grant = claimed.dispatch, claimed.grant
    external, expected = dispatch.external, dispatch.external.expected
    failure: ImportFailed | None = None
    result: TransferResult | None = None
    tally = Tally()
    bug: Exception | None = None
    try:
        plan = TransferPlan(
            file_id=external.file_id, resource_key=external.resource_key, expected_size=expected.size,
            expected_md5=expected.md5, expected_sha1=expected.sha1, expected_sha256=expected.sha256,
            expected_head_revision_id=expected.head_revision_id, part_size=dispatch.plan.part_size,
            parts=tuple(Part(part.number, part.url) for part in grant.parts), stop_line=stop,
        )
        result = transfer(plan, runtime.settings.google_api_key, http, tally)
    except ImportFailed as error:
        failure = error
    except Exception as error:
        bug = error
    if bug is not None:
        # A broken logging sink must never inherit the URL-bearing exception's context.
        _log_bug(bug, with_frames=True)
        bug = None
        code = Code.UPLOAD_FAILED if tally.stage is Stage.UPLOAD else Code.DOWNLOAD_FAILED
        failure = ImportFailed(code, tally.stage)
    if stop.passed():
        failure = ImportFailed(Code.DEADLINE, failure.stage if failure is not None else Stage.VERIFY)
    # A crash building/sealing a completed result has no truthful finalizing code in §2.7.
    # Leave it to the wrapper's sanitized provider failure; never invent a second terminal.
    report = _terminal(claimed, runtime, result, failure, stop)
    _store_terminal(report, grant, http, stop)
    return report


def _terminal(
    claimed: ClaimedAcquire, runtime: Runtime[AcquireSettings], result: TransferResult | None,
    failure: ImportFailed | None, stop: StopLine,
) -> SealedMessage:
    now = runtime.clock()
    envelope: acquire_messages.Envelope = {
        "protocol_version": claimed.dispatch.protocol_version, "message_id": runtime.new_uuid(),
        "sent_at": now, "key_id": runtime.settings.worker_to_hub_key_id, "identity": claimed.grant.identity,
    }
    if failure is None:
        if result is None:
            raise RuntimeError("acquire ended without a transfer result")
        message = acquire_messages.result_completed(
            **envelope, bytes_read=result.bytes, checksum_verified=result.checksum_verified, parts=[part.to_dict() for part in result.parts],
        )
    else:
        message = acquire_messages.result_failed(**envelope, code=failure.code.value, stage=STAGES[failure.stage])
    context = acquire_messages.sender_context(
        MessageKind(message["message_kind"]), claimed.dispatch, claimed.grant.identity,
        settings=runtime.settings, now=now,
    )
    stage = failure.stage if failure is not None else Stage.VERIFY
    check = None if failure is not None and failure.code is Code.DEADLINE else lambda: stop.require(stage)
    try:
        return seal_message(message, context, settings=runtime.settings, require_time=check)
    except ImportFailed:
        pass
    return _terminal(claimed, runtime, None, ImportFailed(Code.DEADLINE, stage), stop)


def _store_terminal(report: SealedMessage, grant: AcquireGrant, http: HttpSession, stop: StopLine) -> None:
    bug: Exception | None = None
    try:
        if stop.passed():
            return
        body = json.dumps(runpod_output(report), ensure_ascii=True, separators=(",", ":")).encode("ascii")
        store.put(http, grant.result_url, body, RESULT_HEADERS, stop, Stage.RECEIPT, lambda attempt: None)
    except (ImportFailed, DeadlineReached, *TRANSFER_ERRORS):
        pass
    except Exception as error:  # Delivery cannot replace an execution's sealed terminal.
        bug = error
    else:
        return
    if bug is not None:
        _log_bug(bug)
    else:
        _delivery_diagnostic("storage")


def _callback(report: SealedMessage, runtime: Runtime[AcquireSettings], ends_at: float) -> None:
    bug: Exception | None = None
    try:
        deliver(
            report, settings=runtime.settings, transport=runtime.transport, sleep=runtime.sleep,
            give_up_at=ends_at, monotonic=runtime.monotonic, strict_deadline=True,
        )
    except (ImportFailed, OSError, TransportFailed):
        pass
    except Exception as error:  # The provider still receives the exact sealed terminal.
        bug = error
    else:
        return
    if bug is not None:
        _log_bug(bug)
    else:
        _delivery_diagnostic("callback")


def _log_bug(error: Exception, *, with_frames: bool = False) -> None:
    try:
        if with_frames:
            frames = "; ".join(frame.f_code.co_name for frame, _ in traceback.walk_tb(error.__traceback__))
            logger.error("acquire failed on %s at %s", type(error).__name__, frames)
        else:
            logger.error("acquire terminal delivery failed on %s", type(error).__name__)
    except Exception:
        pass


def _delivery_diagnostic(channel: str) -> None:
    try:
        logger.info("acquire terminal delivery failed at %s", channel)
    except Exception:
        pass
