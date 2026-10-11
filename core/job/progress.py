"""Best-effort progress of one execution: `transcode.progress` messages posted to the callback address.

The Hub shows progress; it decides nothing by it. So a report never changes an execution's outcome:
whatever goes wrong while one is built, self-checked, signed or posted is logged by its class name and
dropped, and a report is posted once, never again.

Nor does progress hold a billed runtime hostage. Every report is one POST on the execution's own thread,
so its time is the job's time, counted on the dispatch's wall-time limit. Each post therefore has a short
time-out of its own (PROGRESS_TIMEOUT_S), and reporting stops for the rest of the execution, with
nothing more built or posted, as soon as either holds:

    MAX_CONSECUTIVE_UNDELIVERED reports in a row did not reach the Hub, or
    the posts, delivered or not, have taken PROGRESS_ALLOWANCE_S together.

So progress costs one execution at most PROGRESS_ALLOWANCE_S plus one post's PROGRESS_TIMEOUT_S, however
slowly the Hub answers. Each report still takes the next `event_seq` (1, 2, 3, ...), sent or not: the
Hub refuses a sequence number it has already seen, and whether a post arrived is not always known.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, ParamSpec

from core.protocol import messages
from core.protocol.handler import ClaimedExecution, Runtime
from core.protocol.messages import Counters
from core.protocol.outbox import deliver, seal_message
from core.protocol.registry import MessageKind

logger = logging.getLogger(__name__)

PERCENT_DONE = 100
# During a download or an upload, at most one report per this many percent of the stage.
TRANSFER_REPORT_STEP_PERCENT = 10
# A progress message is not worth a second request: the next one says more.
DELIVERY_ATTEMPTS = 1
# After this many reports in a row that were not delivered, no further one is built or posted.
MAX_CONSECUTIVE_UNDELIVERED = 3
# One progress POST's time-out: far below a result's, because nothing is lost with a progress report.
PROGRESS_TIMEOUT_S = 3.0
# How long the posts of one execution may take together before reporting stops.
PROGRESS_ALLOWANCE_S = 30.0

_Arguments = ParamSpec("_Arguments")


def _best_effort(report: Callable[_Arguments, None]) -> Callable[_Arguments, None]:
    """Make a reporting method drop whatever it raises, logged by its class name only."""

    @functools.wraps(report)
    def guarded(*arguments: _Arguments.args, **named_arguments: _Arguments.kwargs) -> None:
        try:
            report(*arguments, **named_arguments)
        except Exception as error:  # noqa: BLE001 - progress never changes an execution's outcome
            logger.warning("a progress report was dropped: %s", type(error).__name__)

    return guarded


def _percent(done: int, total: int) -> int:
    """How much of `total` is done, as a whole percent from 0 to 100; 0 while the total is unknown."""
    if total < 1:
        return 0
    return max(0, min(PERCENT_DONE, done * PERCENT_DONE // total))


class ProgressReporter:
    """Reports one execution's stages, renditions and transferred bytes; no method of it raises."""

    def __init__(self, claimed: ClaimedExecution, runtime: Runtime) -> None:
        if not isinstance(claimed, ClaimedExecution):
            raise TypeError("claimed must be a ClaimedExecution")
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be a Runtime")
        self._protocol_version = claimed.dispatch.protocol_version
        self._identity = claimed.grant.identity
        self._runtime = runtime
        self._event_seq = 0
        self._undelivered_in_a_row = 0
        self._posting_took_s = 0.0
        self._gave_up = False
        self._stage: str | None = None
        self._reported_step = 0
        self._renditions_total = 0
        self._renditions_done = 0
        self._bytes_downloaded = 0
        self._bytes_uploaded = 0

    @_best_effort
    def stage_started(self, stage: str) -> None:
        """The execution entered `stage` (a contract stage name): reported at 0 percent."""
        self._stage = stage
        self._reported_step = 0
        self._send(0)

    @_best_effort
    def renditions_planned(self, total: int) -> None:
        """How many renditions the execution will encode; later reports carry it."""
        self._renditions_total = total

    @_best_effort
    def rendition_done(self) -> None:
        """One more rendition is encoded and verified."""
        self._renditions_done += 1
        self._send(_percent(self._renditions_done, self._renditions_total))

    @_best_effort
    def downloaded(self, done_bytes: int, total_bytes: int) -> None:
        """`done_bytes` of the source's `total_bytes` are downloaded."""
        self._bytes_downloaded = done_bytes
        self._send_transfer_step(_percent(done_bytes, total_bytes))

    @_best_effort
    def uploaded(self, done_bytes: int, total_bytes: int) -> None:
        """`done_bytes` of the generation's `total_bytes` are uploaded."""
        self._bytes_uploaded = done_bytes
        self._send_transfer_step(_percent(done_bytes, total_bytes))

    def _send_transfer_step(self, percent: int) -> None:
        step = percent // TRANSFER_REPORT_STEP_PERCENT
        if step > self._reported_step:
            self._reported_step = step
            self._send(percent)

    def _send(self, percent: int) -> None:
        if self._stage is None:
            raise ValueError("progress is reported within a stage")
        previous_event_seq = self._event_seq
        self._event_seq = previous_event_seq + 1
        if self._gave_up:
            return
        delivered = False
        try:
            delivered = self._post(self._stage, percent, previous_event_seq)
        finally:
            self._note_delivery(delivered)

    def _note_delivery(self, delivered: bool) -> None:
        """Count the reports in a row that did not arrive, and give up for good once that count or the
        time the posts took reaches its limit; the one log line says which."""
        self._undelivered_in_a_row = 0 if delivered else self._undelivered_in_a_row + 1
        if self._undelivered_in_a_row >= MAX_CONSECUTIVE_UNDELIVERED:
            self._gave_up = True
            logger.warning(
                "progress reporting stopped for this execution: %d reports in a row were not delivered",
                self._undelivered_in_a_row,
            )
        elif self._posting_took_s >= PROGRESS_ALLOWANCE_S:
            self._gave_up = True
            logger.warning(
                "progress reporting stopped for this execution: its posts took %.1f s, the allowance is %.1f s",
                self._posting_took_s, PROGRESS_ALLOWANCE_S,
            )

    def _post(self, stage: str, percent: int, previous_event_seq: int) -> bool:
        """Build, self-check, sign and post the report numbered `self._event_seq`; True when the Hub took it."""
        settings = self._runtime.settings
        now = self._runtime.clock()
        message = messages.progress(
            protocol_version=self._protocol_version,
            message_id=self._runtime.new_uuid(),
            sent_at=now,
            key_id=settings.worker_to_hub_key_id,
            identity=self._identity,
            event_seq=self._event_seq,
            stage=stage,
            stage_progress_pct=percent,
            counters=Counters(
                renditions_total=self._renditions_total,
                renditions_done=self._renditions_done,
                bytes_downloaded=self._bytes_downloaded,
                bytes_uploaded=self._bytes_uploaded,
            ),
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_PROGRESS,
            self._identity,
            settings=settings,
            now=now,
            last_event_seq=previous_event_seq,
        )
        sealed = seal_message(message, context, settings=settings)
        started = self._runtime.monotonic()
        try:
            return deliver(
                sealed,
                settings=settings,
                transport=self._runtime.transport,
                sleep=self._runtime.sleep,
                attempts=DELIVERY_ATTEMPTS,
                timeout_s=PROGRESS_TIMEOUT_S,
            )
        finally:
            # Delivered or not: a post that waits out its time-out costs the job the same.
            self._posting_took_s += self._runtime.monotonic() - started
