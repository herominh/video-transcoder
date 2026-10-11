"""What the worker sends, ready to travel: self-checked, encoded, signed, then carried (README sections 12, 13).

seal_message() is the one way a built message becomes bytes with a signature: it encodes the message
within its kind's byte limit, validates those exact bytes with the worker's own validator under the
sender context (`worker_sender` runs what the Hub's receiver runs), and signs them. A message that fails
its self-check is a wiring bug and is never sent.

A result or a `transcode.unclaimed` report travels two ways with the same bytes: deliver() posts it to
the callback address (best effort: the provider's job output is the path the Hub can always read), and
runpod_output() is the value the handler returns. A claim travels neither way (README 15.1, rule 5).

A delivery can be held to an instant of the caller's monotonic clock (`give_up_at`): the handler gives
the one at which its start's wall time ends, so no callback attempt starts at or after it (one that
started just before it may take its least time-out past it). A delivery that stops there costs nothing
but time: the job output carries the same signed bytes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from . import messages, pipeline, schema
from .context import TrustedContext
from .inbound import accept_hub_error
from .registry import MessageKind, load_registry
from .settings import WorkerSettings
from .transport import Transport, TransportFailed

logger = logging.getLogger(__name__)

RUNPOD_OUTPUT_SCHEMA = "carriers/runpod-output.schema.json"
RUNPOD_OUTPUT_CARRIER = "runpod.output"
MESSAGE_MEMBER = "vh_message"
SIGNATURE_MEMBER = "vh_signature"
CALLBACK_ATTEMPTS = 3
CALLBACK_TIMEOUT_S = 10.0
# A POST that starts with less time left than this before the delivery gives up still gets this long: it
# started before the end.
CALLBACK_MIN_TIMEOUT_S = 1.0
CALLBACK_BACKOFF_S = (1, 2)
_HTTP_SUCCESS = range(200, 300)
# What the handler returns as the job's output, and what goes to the callback address.
_OUTPUT_KINDS = frozenset(
    {MessageKind.TRANSCODE_RESULT_COMPLETED, MessageKind.TRANSCODE_RESULT_FAILED, MessageKind.TRANSCODE_UNCLAIMED}
)
_CALLBACK_KINDS = _OUTPUT_KINDS | {MessageKind.TRANSCODE_PROGRESS}


class SelfCheckFailed(RuntimeError):
    """The worker built a message its own validator refuses: a wiring bug, raised before anything is sent."""


@dataclass(frozen=True)
class SealedMessage:
    """The exact bytes of one message and their signature."""

    kind: MessageKind
    body: bytes = field(repr=False)
    signature: str = field(repr=False)


def seal_message(message: Mapping[str, Any], context: TrustedContext, *, settings: WorkerSettings) -> SealedMessage:
    """Encode the message, validate its bytes under the sender context and sign them."""
    body = messages.encode(message)
    verdict = pipeline.validate(body, context)
    kind = MessageKind(message["message_kind"])
    if not verdict.accepted:
        raise SelfCheckFailed(
            f"a {kind.value} fails its self-check at {verdict.layer.value}/{verdict.reason.value}"  # type: ignore[union-attr]
        )
    return SealedMessage(kind=kind, body=body, signature=messages.seal(body, settings.worker_to_hub_key))


def runpod_output(sealed: SealedMessage) -> dict[str, str]:
    """The RunPod job output of a result or an unclaimed report: `{vh_message, vh_signature}`."""
    if sealed.kind not in _OUTPUT_KINDS:
        raise ValueError(f"a {sealed.kind.value} is never a job output")
    value = {MESSAGE_MEMBER: sealed.body.decode("ascii"), SIGNATURE_MEMBER: sealed.signature}
    serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    within_limit = len(serialized) <= load_registry().carrier_max_bytes(RUNPOD_OUTPUT_CARRIER)
    if not within_limit or schema.first_error_pointer(RUNPOD_OUTPUT_SCHEMA, value) is not None:
        raise ValueError(f"a {sealed.kind.value} does not fit the RunPod output carrier")
    return value


def deliver(
    sealed: SealedMessage,
    *,
    settings: WorkerSettings,
    transport: Transport,
    sleep: Callable[[float], None],
    attempts: int = CALLBACK_ATTEMPTS,
    timeout_s: float = CALLBACK_TIMEOUT_S,
    give_up_at: float | None = None,
    monotonic: Callable[[], float] | None = None,
) -> bool:
    """POST the message to the callback address; True when the Hub answered 2xx. It never raises for a failed delivery.

    Sent again only after a transport failure or a `hub.error` the Hub called retryable. Unlike a claim, a
    delivery is not sent again after a gateway's 502, 503 or 504 without a `hub.error`: the job output
    carries the same bytes, so the Hub loses nothing but time. `timeout_s` is each POST's time-out, a
    positive finite number of seconds (ValueError otherwise, before any POST).

    `give_up_at` and `monotonic` come together or not at all (ValueError otherwise, before any POST):
    `monotonic` gives seconds that only ever move forward, and `give_up_at` is the reading of it, a finite
    number, at which the delivery ends. With them no attempt starts at or after that instant (the first
    included: then nothing is sent), no wait between attempts is started that would end at or after it,
    and each POST's time-out is the smaller of `timeout_s` and the time left, where less time left than
    CALLBACK_MIN_TIMEOUT_S counts as that much: an attempt that started before the end gets that long,
    never more than `timeout_s`. Without them nothing bounds the delivery but its attempts.
    """
    if sealed.kind not in _CALLBACK_KINDS:
        raise ValueError(f"a {sealed.kind.value} never goes to the callback address")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < math.inf:
        raise ValueError("timeout_s must be a positive finite number of seconds")
    time_left_s = _time_left(give_up_at, monotonic)
    hub_error_context = messages.receiver_context(MessageKind.HUB_ERROR, settings=settings)
    for attempt in range(attempts):
        if attempt > 0:
            wait_s = float(CALLBACK_BACKOFF_S[min(attempt, len(CALLBACK_BACKOFF_S)) - 1])
            if time_left_s is not None and wait_s >= time_left_s():
                return _stopped_for_want_of_time(sealed)
            sleep(wait_s)
        post_timeout_s = timeout_s
        if time_left_s is not None:
            left_s = time_left_s()
            if left_s <= 0:
                return _stopped_for_want_of_time(sealed)
            post_timeout_s = min(timeout_s, max(CALLBACK_MIN_TIMEOUT_S, left_s))
        try:
            answer = transport.post(
                settings.callback_url,
                sealed.body,
                sealed.signature,
                timeout_s=post_timeout_s,
                max_body_bytes=load_registry().max_bytes(MessageKind.HUB_ERROR),
            )
        except TransportFailed as failure:
            logger.info("callback of a %s got no answer: %s", sealed.kind.value, failure)
            continue
        if answer.status in _HTTP_SUCCESS:
            return True
        error = accept_hub_error(answer.body, context=hub_error_context)
        if error is None or not error["retryable"]:
            code = "none" if error is None else error["code"]
            logger.info("callback of a %s refused: status %d, code %s", sealed.kind.value, answer.status, code)
            return False
        logger.info("callback of a %s refused for now: %s", sealed.kind.value, error["code"])
    return False


def _time_left(
    give_up_at: float | None, monotonic: Callable[[], float] | None
) -> Callable[[], float] | None:
    """How many seconds a delivery has until `give_up_at`, to be asked whenever it matters; None when
    nothing bounds the delivery. ValueError unless both are given or neither, and unless `give_up_at` is
    a finite number."""
    if give_up_at is None and monotonic is None:
        return None
    if give_up_at is None or monotonic is None:
        raise ValueError("give_up_at and monotonic are given together or not at all")
    if isinstance(give_up_at, bool) or not isinstance(give_up_at, (int, float)) or not math.isfinite(give_up_at):
        raise ValueError("give_up_at must be a finite reading of the monotonic clock")
    ends_at, read_clock = give_up_at, monotonic
    return lambda: ends_at - read_clock()


def _stopped_for_want_of_time(sealed: SealedMessage) -> bool:
    """A delivery that ends because its time is spent: logged once by the message's kind, and not delivered."""
    logger.info("callback of a %s stopped: no time is left for it", sealed.kind.value)
    return False
