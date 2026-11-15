"""The heartbeat of an execution at protocol 2.2.0 (README 15.4, runtime rule 6): a request at least every 180 s.

ffmpeg holds the job's thread for hours, so the lease is renewed from a daemon thread of its own: whenever
HEARTBEAT_INTERVAL_S (120 s, inside the contract's 180) have passed on the capability client's clock since the
request of its newest accepted answer was sent (any request renews the lease, a stage's or an upload's as well),
it sends a heartbeat (`put` null, `source` false) at the stage the execution is in. A failure the client raises
ends the heartbeat: the client keeps it, lost() names it, and the job's thread meets it where it looks (a stage's
entry, between the objects of a stage, and the stop check of every bounded process, core/bounded_process.py
stopped_when()). The heartbeat's own exchange is held to the execution's deadline by the client. Once the client's
work is ended (CapabilityClient.end()) it sends nothing more and its waits between tries stop at once, so stop()
waits for one exchange in flight at most. An unexpected error of the thread ends the work as a retryable
internal_error/heartbeat_failed (the client's fail()), logged by its class alone; nothing of a request or an answer
is logged here.
"""

from __future__ import annotations

import logging
import threading

from core.failure import Failure, WorkerFailure, make_detail
from core.protocol.capabilities_client import (
    CAPABILITY_REQUEST_TIMEOUT_S,
    WAIT_SLICE_S,
    CapabilityClient,
    ExchangeEnded,
)

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 120.0
# How often, at most, the thread looks again at the time of the newest accepted request.
POLL_S = 1.0
# How long stop() waits for the thread once the client's work is ended: one exchange in flight, and a margin for a
# slice of a wait and the thread's own steps.
JOIN_TIMEOUT_S = CAPABILITY_REQUEST_TIMEOUT_S + 10 * WAIT_SLICE_S
THREAD_NAME = "vh-capability-heartbeat"


class Heartbeat:
    """The heartbeat of one execution: start() it once its client is made, stop() it before the execution ends."""

    def __init__(
        self, client: CapabilityClient, *, interval_s: float = HEARTBEAT_INTERVAL_S, poll_s: float = POLL_S
    ) -> None:
        if not isinstance(client, CapabilityClient):
            raise TypeError("client must be a CapabilityClient")
        if not interval_s > 0 or not poll_s > 0:
            raise ValueError("interval_s and poll_s must be positive")
        self._client = client
        self._interval_s = interval_s
        self._poll_s = poll_s
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = JOIN_TIMEOUT_S) -> bool:
        """End the client's work (CapabilityClient.end(): nothing more is sent, a wait between tries stops), stop the
        thread and wait for it, at most `timeout_s`; whether it has ended. A thread never started counts as ended."""
        self._client.end()
        self._stopped.set()
        if not self._thread.is_alive():
            return True
        self._thread.join(timeout_s)
        ended = not self._thread.is_alive()
        if not ended:
            logger.warning("the capability heartbeat did not end within %.1f s", timeout_s)
        return ended

    def _run(self) -> None:
        failed_on: str | None = None
        try:
            self._beat_until_stopped()
        except (WorkerFailure, ExchangeEnded):
            # The client keeps a failure (lost() names it to the job); an ended exchange is no failure.
            return
        except Exception as error:  # noqa: BLE001 - the heartbeat never ends the process; the client's lost() does
            failed_on = type(error).__name__
        if failed_on is not None:
            # Logged and recorded here, after the `except` block has ended: the error's own text reaches no handler.
            logger.error("the capability heartbeat ended on an unexpected %s", failed_on)
            self._client.fail(Failure(
                "internal_error", "heartbeat_failed", True, make_detail(f"the capability heartbeat failed: {failed_on}")
            ))

    def _beat_until_stopped(self) -> None:
        client = self._client
        while not self._stopped.is_set():
            if client.ended or client.lost() is not None:
                return
            last_sent_at = client.last_accepted_sent_at()
            if last_sent_at is None or client.stage is None:
                self._stopped.wait(self._poll_s)
                continue
            due_in_s = last_sent_at + self._interval_s - client.monotonic()
            if due_in_s > 0:
                self._stopped.wait(min(due_in_s, self._poll_s))
                continue
            client.heartbeat()
