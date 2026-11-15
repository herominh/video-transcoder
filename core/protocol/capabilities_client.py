"""The capability exchange of protocol 2.2.0, the runtime's side (README section 15.4, the runtime's ten rules).

A runtime of a 2.2.0 dispatch holds no storage credential: it asks the Hub's capability address (from its trusted
configuration, https only, the claim's transport: certificate verified, no redirect, one deadline per exchange) for a
read URL of its source and for put URLs of the objects it writes, and every request it sends renews its lease and
reports its stage. One CapabilityClient serves one execution, made once its claim was granted:

- ask() sends one `execution.capabilities.request` at a time (a lock), `seq` strictly increasing from 1, each try a
  new message (a new id, send time, `seq` and signature). An answer is used only when its signature verifies under a
  Hub-to-worker key and it is accepted at `worker_receiver` (K3 to K5 included); any other answer is discarded unused.
- A retryable `hub.error`, a temporary refusal without one (429, 502, 503, 504) or no answer at all is asked again
  with the claim's backoff and its reading of `Retry-After`, never at or after the end of the lease by this runtime's
  count nor of the execution's deadline. Every other ending is final and ends the work: authority_refused with the
  Hub's code, `payload_too_large` for a 413 without a `hub.error`, `capability_answer_invalid` for an answer it does
  not accept or any other status, and `lease_unrenewed` when no accepted answer came before the lease's end
  (`capability_deadline_exceeded`, a deadline_exceeded, when the execution's deadline came first).
- The lease and every URL's usable time are counted on the monotonic clock from the instant the request was SENT:
  `expires_at` minus the answer's `sent_at` (rules 5 and 6). Before its first accepted answer the runtime has been
  told no lease; it counts the least one an answer promises (FIRST_LEASE_S, Hub rule 6) from the client's making.
- Once the work is ended (a final refusal, an answer not accepted, the lease run out by its own count), lost() names
  that failure for good and no further request is sent (rule 6: a stopped runtime asks for no further URL).

Nothing of a request, an answer or a URL is logged, kept or put in an error (rule 10): a log line names the execution,
the `seq`, the stage and a code. No repr of what this module returns shows a URL.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Mapping

from core.failure import Failure, WorkerFailure

from . import messages
from .claim import CLAIM_BACKOFF_S, Grant
from .dispatch import AcceptedDispatch
from .inbound import NotAccepted, accept_hub_error, accept_signed
from .messages import CapabilityPut
from .outbox import SealedMessage, seal_message
from .registry import MessageKind, load_registry
from .semantic import parse_timestamp_ms
from .settings import CAPABILITY_PROTOCOL_VERSION, WorkerSettings
from .transport import (
    HTTP_ERROR_STATUSES,
    HTTP_PAYLOAD_TOO_LARGE,
    TEMPORARY_REFUSAL_STATUSES,
    HttpAnswer,
    Transport,
    TransportFailed,
)

logger = logging.getLogger(__name__)

HTTP_OK = 200
MS_PER_SECOND = 1_000
# The longest one exchange may take, as a claim's.
CAPABILITY_REQUEST_TIMEOUT_S = 15.0
# The least lease an answer of the Hub promises (README 15.4, Hub rule 6): counted from the client's making until
# the first answer is accepted.
FIRST_LEASE_S = 600.0
# The contract's stages, in the only order a runtime moves through them (README 15.4, runtime rule 8).
STAGES = ("probing", "downloading", "transcoding", "finalizing", "uploading")
AUTHORITY_REFUSED = "authority_refused"
ANSWER_INVALID = "capability_answer_invalid"
LEASE_UNRENEWED = "lease_unrenewed"
PAYLOAD_TOO_LARGE = "payload_too_large"
CAPABILITY_DEADLINE_EXCEEDED = Failure(
    "deadline_exceeded", "capability_deadline_exceeded", False,
    "the execution ran out of time before the Hub answered its capability request",
)


@dataclass(frozen=True)
class GrantedUrl:
    """One URL of an accepted answer and the reading of the client's monotonic clock it may be used before."""

    url: str = field(repr=False)
    use_before: float


@dataclass(frozen=True)
class PutUrl:
    """A put URL, the one path (relative to the granted prefix) it was granted for, and the reading of the client's
    monotonic clock it may be used before."""

    path: str
    url: str = field(repr=False)
    use_before: float


@dataclass(frozen=True)
class CapabilityAnswer:
    """An accepted `execution.capabilities.granted`: what it answers, the lease by this runtime's count, its URLs."""

    seq: int
    lease_ends_at: float
    source: GrantedUrl | None
    puts: tuple[PutUrl, ...]


@dataclass(frozen=True)
class _Retry:
    retry_after_s: float | None


def _refused(code: str, detail: str) -> Failure:
    return Failure(AUTHORITY_REFUSED, code, False, detail)


class CapabilityClient:
    """The capability exchange of one granted execution at protocol 2.2.0.

    `clock` gives the aware wall-clock time messages are stamped and judged with, `monotonic` the seconds the lease
    and the URLs are counted on, `sleep` the waits between tries and `time_left` the seconds left before the
    execution's deadline. Thread-safe: the heartbeat asks from a thread of its own.
    """

    def __init__(
        self,
        *,
        dispatch: AcceptedDispatch,
        grant: Grant,
        runtime_id: str,
        settings: WorkerSettings,
        transport: Transport,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
        new_message_id: Callable[[], str],
        time_left: Callable[[], float],
    ) -> None:
        if not isinstance(dispatch, AcceptedDispatch) or not isinstance(grant, Grant):
            raise TypeError("dispatch and grant must be an accepted dispatch and its grant")
        if dispatch.protocol_version != CAPABILITY_PROTOCOL_VERSION:
            raise ValueError("the capability exchange belongs to a dispatch of protocol 2.2.0")
        if not isinstance(settings, WorkerSettings) or settings.capabilities_url is None or settings.storage_host is None:
            raise ValueError("settings must hold the capability address and the storage host")
        self._dispatch = dispatch
        self._grant = grant
        self._runtime_id = runtime_id
        self._settings = settings
        self._transport = transport
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._new_message_id = new_message_id
        self._time_left = time_left
        # One request at a time; the state is read from other threads while a request is out.
        self._exchange_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._seq = 0
        self._stage: str | None = None
        self._media_duration_ms: int | None = None
        self._lease_ends_at = monotonic() + FIRST_LEASE_S
        self._last_accepted_sent_at: float | None = None
        self._lost: Failure | None = None

    # --- what the next requests report --------------------------------------------------------------

    @property
    def stage(self) -> str | None:
        with self._state_lock:
            return self._stage

    def set_stage(self, stage: str) -> None:
        """The stage the next requests report; stages only move forward (ValueError otherwise)."""
        if stage not in STAGES:
            raise ValueError("stage must be a contract stage")
        with self._state_lock:
            if self._stage is not None and STAGES.index(stage) < STAGES.index(self._stage):
                raise ValueError("stages only move forward")
            self._stage = stage

    def set_media_duration(self, duration_ms: int) -> None:
        """The admitted duration the next requests report; set once (ValueError for another value later)."""
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 1:
            raise ValueError("duration_ms must be a positive integer")
        with self._state_lock:
            if self._media_duration_ms is not None and self._media_duration_ms != duration_ms:
                raise ValueError("the media duration is set once")
            self._media_duration_ms = duration_ms

    def monotonic(self) -> float:
        """A reading of the clock the lease and every `use_before` are counted on."""
        return self._monotonic()

    def last_accepted_sent_at(self) -> float | None:
        """When the request of the newest accepted answer was sent (a reading of monotonic()); None before the first."""
        with self._state_lock:
            return self._last_accepted_sent_at

    # --- the end of the work ------------------------------------------------------------------------

    def lost(self) -> WorkerFailure | None:
        """The failure that ended the lease or the authority, once one did; None while the work may go on."""
        with self._state_lock:
            if self._lost is None and self._monotonic() >= self._lease_ends_at:
                self._lost = _refused(LEASE_UNRENEWED, "the lease ended before the Hub renewed it")
                logger.warning("execution %s: the lease ended by this runtime's count", self._execution_id)
            return None if self._lost is None else WorkerFailure(self._lost)

    def raise_if_lost(self) -> None:
        lost = self.lost()
        if lost is not None:
            raise lost

    def _lose(self, failure: Failure) -> WorkerFailure:
        with self._state_lock:
            if self._lost is None:
                self._lost = failure
            return WorkerFailure(self._lost)

    @property
    def _execution_id(self) -> str:
        return self._grant.identity.execution_id

    # --- one ask ------------------------------------------------------------------------------------

    def ask(self, stage: str, *, source: bool = False, put: CapabilityPut | None = None) -> CapabilityAnswer:
        """Ask at `stage` (which becomes the stage the requests report) for the source URL, put URLs, or, with
        neither, nothing but the lease: the accepted answer, or WorkerFailure once the work is ended (lost())."""
        if not isinstance(source, bool):
            raise TypeError("source must be a bool")
        if put is not None and not isinstance(put, CapabilityPut):
            raise TypeError("put must be a CapabilityPut or None")
        self.set_stage(stage)
        return self._ask(source, put)

    def heartbeat(self) -> CapabilityAnswer:
        """Renew the lease (`put` null, `source` false) at the stage the execution is in by then; ValueError before the
        first stage was set, else as ask()."""
        if self.stage is None:
            raise ValueError("a heartbeat reports a stage: none was set")
        return self._ask(False, None)

    def _ask(self, source: bool, put: CapabilityPut | None) -> CapabilityAnswer:
        attempt = 0
        while True:
            retry_after_s: float | None = None
            with self._exchange_lock:
                self.raise_if_lost()
                if self._bound_left_s() <= 0:
                    break
                request, sealed = self._sealed(source, put)
                left_s = self._bound_left_s()
                if left_s <= 0:
                    break
                sent_at = self._monotonic()
                answer = self._post(sealed, min(CAPABILITY_REQUEST_TIMEOUT_S, left_s), request)
                outcome = _Retry(None) if answer is None else self._judge(answer, request, sent_at)
                if isinstance(outcome, CapabilityAnswer):
                    return outcome
                if isinstance(outcome, Failure):
                    raise self._lose(outcome)
                retry_after_s = outcome.retry_after_s
            wait_s = max(float(CLAIM_BACKOFF_S[min(attempt, len(CLAIM_BACKOFF_S) - 1)]), float(retry_after_s or 0))
            attempt += 1
            if wait_s >= self._bound_left_s():
                break
            self._sleep(wait_s)
        raise self._lose(self._unanswered())

    def _bound_left_s(self) -> float:
        """Seconds until the nearer of the lease's end by this runtime's count and the execution's deadline."""
        with self._state_lock:
            lease_left_s = self._lease_ends_at - self._monotonic()
        return min(lease_left_s, self._time_left())

    def _unanswered(self) -> Failure:
        with self._state_lock:
            lease_left_s = self._lease_ends_at - self._monotonic()
        if self._time_left() < lease_left_s:
            return CAPABILITY_DEADLINE_EXCEEDED
        return _refused(LEASE_UNRENEWED, "no accepted capability answer came before the lease ended")

    def _sealed(self, source: bool, put: CapabilityPut | None) -> tuple[dict[str, Any], SealedMessage]:
        """The next request at the stage the execution is in, built, self-checked and signed; its `seq` is spent
        whatever becomes of it."""
        settings, now = self._settings, self._clock()
        with self._state_lock:
            last_seq, self._seq = self._seq, self._seq + 1
            stage, duration_ms = self._stage, self._media_duration_ms
        request = messages.capabilities_request(
            protocol_version=self._dispatch.protocol_version,
            message_id=self._new_message_id(),
            sent_at=now,
            key_id=settings.worker_to_hub_key_id,
            identity=self._grant.identity,
            runtime_id=self._runtime_id,
            seq=last_seq + 1,
            stage=stage,
            media_duration_ms=duration_ms,
            source=source,
            put=put,
        )
        context = messages.worker_sender_context(
            MessageKind.EXECUTION_CAPABILITIES_REQUEST, self._grant.identity,
            settings=replace(settings, accepted_protocol_versions=(self._dispatch.protocol_version,)), now=now,
            runtime_id=self._runtime_id, last_capability_seq=last_seq,
        )
        return request, seal_message(request, context, settings=settings)

    def _post(self, sealed: SealedMessage, timeout_s: float, request: Mapping[str, Any]) -> HttpAnswer | None:
        registry = load_registry()
        largest_answer = max(
            registry.max_bytes(MessageKind.EXECUTION_CAPABILITIES_GRANTED), registry.max_bytes(MessageKind.HUB_ERROR)
        )
        try:
            return self._transport.post(
                self._settings.capabilities_url, sealed.body, sealed.signature,  # type: ignore[arg-type]
                timeout_s=timeout_s, max_body_bytes=largest_answer,
            )
        except TransportFailed:
            pass
        self._log("got no answer", request)
        return None

    def _judge(self, answer: HttpAnswer, request: Mapping[str, Any], sent_at: float) -> CapabilityAnswer | Failure | _Retry:
        if answer.status == HTTP_OK:
            return self._accepted(answer, request, sent_at)
        error = None
        if answer.status in HTTP_ERROR_STATUSES:
            error = accept_hub_error(
                answer.body, context=messages.receiver_context(MessageKind.HUB_ERROR, settings=self._settings)
            )
        if error is None and answer.status in TEMPORARY_REFUSAL_STATUSES:
            self._log(f"refused for now: status {answer.status}", request)
            return _Retry(answer.retry_after_s)
        if error is None and answer.status == HTTP_PAYLOAD_TOO_LARGE:
            self._log(f"refused: {PAYLOAD_TOO_LARGE}", request)
            return _refused(PAYLOAD_TOO_LARGE, "the Hub refused the capability request as too large")
        if error is None:
            self._log(f"got an answer that is no hub.error: status {answer.status}", request)
            return _refused(ANSWER_INVALID, f"the Hub answered a capability request with status {answer.status}")
        code, retryable = error["code"], error["retryable"]
        self._log(f"refused: {code} (retryable: {retryable})", request)
        if retryable:
            return _Retry(error.get("retry_after_s"))
        return _refused(code, f"the Hub refused the capability request: {code}")

    def _accepted(self, answer: HttpAnswer, request: Mapping[str, Any], sent_at: float) -> CapabilityAnswer | Failure:
        dispatch, grant, settings = self._dispatch, self._grant, self._settings
        context = messages.receiver_context(
            MessageKind.EXECUTION_CAPABILITIES_GRANTED,
            settings=replace(settings, accepted_protocol_versions=(dispatch.protocol_version,)),
            now=self._clock(),
            execution=grant.identity,
            runtime_id=self._runtime_id,
            capability_request=request,
            capability_storage_host=settings.storage_host,
            capability_output=(dispatch.output.bucket, grant.output_prefix),
            capability_source=(dispatch.source.bucket, dispatch.source.object_key),
        )
        try:
            granted = accept_signed(answer.body, answer.signature, context=context, keys=settings.hub_to_worker_keys)
        except NotAccepted as refusal:
            # The refusal names the step, the layer and the reason only: never a pointer, a URL or the body.
            refused_at = str(refusal)
        else:
            return self._answer_of(granted, sent_at, request)
        self._log(f"its answer is not accepted at {refused_at}", request)
        return _refused(ANSWER_INVALID, "the Hub's capability answer is not accepted")

    def _answer_of(self, granted: Mapping[str, Any], sent_at: float, request: Mapping[str, Any]) -> CapabilityAnswer | Failure:
        sent_ms = parse_timestamp_ms(granted["sent_at"])

        def use_before(expires_at: str) -> float:
            # The validator accepted every instant as real (S1), so each parses.
            return sent_at + (parse_timestamp_ms(expires_at) - sent_ms) / MS_PER_SECOND  # type: ignore[operator]

        source, put = granted["source"], granted["put"]
        answer = CapabilityAnswer(
            seq=granted["seq"],
            lease_ends_at=use_before(granted["lease"]["expires_at"]),
            source=None if source is None else GrantedUrl(url=source["url"], use_before=use_before(source["expires_at"])),
            puts=() if put is None else tuple(
                PutUrl(path=item["path"], url=item["url"], use_before=use_before(put["expires_at"]))
                for item in put["objects"]
            ),
        )
        with self._state_lock:
            # An answer that came after the lease had already ended by this runtime's count renews nothing (rule 6).
            if self._lost is None and self._monotonic() < self._lease_ends_at:
                self._lease_ends_at = answer.lease_ends_at
                self._last_accepted_sent_at = sent_at
                renewed = True
            else:
                renewed = False
        if not renewed:
            self.raise_if_lost()
        self._log("granted", request)
        return answer

    def _log(self, what: str, request: Mapping[str, Any]) -> None:
        logger.info(
            "capability request %d of execution %s at %s %s", request["seq"], self._execution_id, request["stage"], what
        )
