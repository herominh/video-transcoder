"""The claim exchange of contract v2, the worker's side (README section 15.1).

A dispatch names work, not a running process: before it reads the source or writes any object, a runtime
sends `transcode.claim` to the Hub's claim address (from its trusted configuration, https only) and works
only on a `200` whose body verifies under a Hub-to-worker key and is an accepted `transcode.claim.granted`
for exactly this dispatch, this runtime and this encryption. Every other ending is an Unclaimed outcome,
and the runtime does no work:

    refused         a `hub.error` that is not retryable, or still retryable when the claiming ends
    unreachable     no answer from the Hub until the claiming ends
    invalid_answer  any other answer: a grant that does not verify or is not accepted, a `200` that is
                    not a grant, a redirect, a body that is no `hub.error` of this version

A `hub.error` is read only under a 4xx or 5xx status: under any other status its body is "any other
answer". A 502, 503 or 504 that carries no `hub.error` is no answer from the Hub: it is what a gateway
in front of the Hub says while it cannot reach the claim core (a restart), so the request failed in
transport, and claiming again is safe, because the Hub gives one runtime the same grant again (README
15.2, row 3). It claims again only after such a failure in transport or a retryable `hub.error`, with the
same runtime id and token and a new message id and send time, never at or after the dispatch's
`claim.expires_at`, for no longer than CLAIM_PATIENCE_S (a billed runtime does not wait out a bootstrap
that can last an hour), and never at or after the instant its start's wall time ends
(`wall_time_ends_at`: the handler counts the dispatch's wall-time limit from the moment the start
began). The claiming ends at the earliest of the three, so a dispatch whose wall-time limit is shorter
than the patience stops claiming at its limit. Building and sealing a claim takes time, so the time
left is read twice for each: before the claim is built, and again once it is sealed, immediately before
the POST. A claim whose time ran out in between is not sent, and it counts neither as a claim nor as a
failure in transport: the claiming ends as when no time was left to build it.
The contract allows the retries, it does not require them; giving up earlier reports the same outcome as
the expiry would. The expiry is judged on the wall clock, as the contract states it; the patience and
the wall time are measured on a monotonic reading, so a wall clock that is stepped while the Hub does not
answer cannot stretch either.

The body of a claim and of a claim response is never logged, stored or put in an error, and the media key
of a grant exists only inside the returned Grant, whose repr() does not show it (README 15.1, rules 4, 5).
The Grant holds the key until whoever runs the execution drops it (Grant.drop_media_key): rule 4 gives a
runtime the key until its encode and the encode's own checks are done, never past its execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import messages
from .dispatch import AcceptedDispatch
from .inbound import NotAccepted, accept_hub_error, accept_signed
from .messages import ExecutionIdentity
from .outbox import SealedMessage, seal_message
from .registry import MessageKind, load_registry
from .settings import WorkerSettings
from .transport import GATEWAY_FAILURE_STATUSES, HTTP_ERROR_STATUSES, HttpAnswer, Transport, TransportFailed

logger = logging.getLogger(__name__)

HTTP_OK = 200
CAUSE_REFUSED = "refused"
CAUSE_UNREACHABLE = "unreachable"
CAUSE_INVALID_ANSWER = "invalid_answer"
# How long one runtime keeps claiming, the request's own time-out, and the waits between attempts (the last repeats).
CLAIM_PATIENCE_S = 120
CLAIM_REQUEST_TIMEOUT_S = 15.0
# A claim that starts with less time left than this still gets this long: it was sent before the end.
CLAIM_MIN_REQUEST_TIMEOUT_S = 1.0
CLAIM_BACKOFF_S = (1, 2, 4, 8, 15)


@dataclass(frozen=True)
class Grant:
    """An accepted `transcode.claim.granted`: the execution, its generation, where it writes and its media key.

    Nothing of it changes but the key, which drop_media_key() lets go of.
    """

    identity: ExecutionIdentity
    generation_id: str
    output_location_id: str
    output_prefix: str
    media_key_id: str | None
    media_key: bytes | None = field(repr=False)

    def drop_media_key(self) -> None:
        """Let go of the media key: `media_key` is None from here on, however often this is called.

        The id of the key stays. Python cannot erase the key's bytes: no longer holding them is what a
        runtime can do (README 15.1, rule 4).
        """
        object.__setattr__(self, "media_key", None)


@dataclass(frozen=True)
class Unclaimed:
    """This start does no work; `hub_error_code` is set exactly when the cause is `refused`."""

    cause: str
    hub_error_code: str | None = None


@dataclass(frozen=True)
class _RetryableRefusal:
    code: str
    retry_after_s: int | None


def claim_execution(
    dispatch: AcceptedDispatch,
    *,
    settings: WorkerSettings,
    runtime_id: str,
    transport: Transport,
    clock: Callable[[], datetime],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    new_message_id: Callable[[], str],
    wall_time_ends_at: float,
) -> Grant | Unclaimed:
    """Claim the dispatch for this runtime.

    `clock` gives the aware current time, `monotonic` seconds that only ever move forward, and `sleep` waits
    that many seconds. `wall_time_ends_at` is the reading of `monotonic`, a finite number, at which the
    start's wall time ends (ValueError otherwise, an integer too large to be a float included, before
    any claim): no claim is sent at or after it.
    """
    try:
        finite = (
            not isinstance(wall_time_ends_at, bool)
            and isinstance(wall_time_ends_at, (int, float))
            and math.isfinite(wall_time_ends_at)
        )
    except OverflowError:
        # An integer too large to be a float: no reading of a clock, and nothing can be counted to it.
        finite = False
    if not finite:
        raise ValueError("wall_time_ends_at must be a finite reading of the monotonic clock")
    claiming_ends_at = min(monotonic() + CLAIM_PATIENCE_S, wall_time_ends_at)

    def time_left_s(now: datetime) -> float:
        return min((dispatch.claim_expires_at - now).total_seconds(), claiming_ends_at - monotonic())

    last_refusal_code: str | None = None
    attempt = 0
    while True:
        now = clock()
        if time_left_s(now) <= 0:
            break
        wait_s = float(CLAIM_BACKOFF_S[min(attempt, len(CLAIM_BACKOFF_S) - 1)])
        sealed = _sealed_claim(dispatch, settings, runtime_id, now, new_message_id())
        # Building and sealing took time (milliseconds, more on a starved host), so both ends are read
        # again, immediately before the POST. A claim whose time ran out meanwhile is not sent: the
        # claiming ends here as it does above, and what was sealed was neither a claim nor a failure.
        now = clock()
        left_s = time_left_s(now)
        if left_s <= 0:
            break
        answer = _send(dispatch, settings, transport, sealed, left_s)
        outcome = None if answer is None else _judge(answer, dispatch, settings, runtime_id, clock())
        if isinstance(outcome, (Grant, Unclaimed)):
            return outcome
        if isinstance(outcome, _RetryableRefusal):
            last_refusal_code = outcome.code
            wait_s = max(wait_s, float(outcome.retry_after_s or 0))
        attempt += 1
        if wait_s >= time_left_s(clock()):
            break
        sleep(wait_s)
    dispatch_id = dispatch.identity.dispatch_id
    if last_refusal_code is not None:
        logger.info("claim of dispatch %s ended refused: %s", dispatch_id, last_refusal_code)
        return Unclaimed(CAUSE_REFUSED, last_refusal_code)
    # The contract has one cause for all three; the log tells a late start, and a start whose own time
    # was spent before it could claim, from a network fault.
    if attempt > 0:
        logger.info("claim of dispatch %s ended unreachable after %d claims", dispatch_id, attempt)
    elif dispatch.claim_expires_at <= now:
        logger.info("claim of dispatch %s: its bootstrap had expired before the first claim, nothing was sent", dispatch_id)
    else:
        logger.info(
            "claim of dispatch %s: the start's wall time was spent before the first claim, nothing was sent",
            dispatch_id,
        )
    return Unclaimed(CAUSE_UNREACHABLE)


def _sealed_claim(
    dispatch: AcceptedDispatch, settings: WorkerSettings, runtime_id: str, now: datetime, message_id: str
) -> SealedMessage:
    """One claim of this runtime, sent at `now`: built, self-checked and signed, not yet posted."""
    claim = messages.claim(
        protocol_version=dispatch.protocol_version,
        message_id=message_id,
        sent_at=now,
        key_id=settings.worker_to_hub_key_id,
        identity=dispatch.identity,
        bootstrap_token=dispatch.bootstrap_token,
        runtime_id=runtime_id,
    )
    context = messages.worker_sender_context(MessageKind.TRANSCODE_CLAIM, dispatch.identity, settings=settings, now=now)
    return seal_message(claim, context, settings=settings)


def _send(
    dispatch: AcceptedDispatch, settings: WorkerSettings, transport: Transport, sealed: SealedMessage, left_s: float
) -> HttpAnswer | None:
    """Post one sealed claim, `left_s` seconds (more than none) before the claiming ends, as read
    immediately before this call; None when it got no answer."""
    timeout_s = max(CLAIM_MIN_REQUEST_TIMEOUT_S, min(CLAIM_REQUEST_TIMEOUT_S, left_s))
    largest_answer = max(
        load_registry().max_bytes(MessageKind.TRANSCODE_CLAIM_GRANTED), load_registry().max_bytes(MessageKind.HUB_ERROR)
    )
    try:
        return transport.post(
            settings.claim_url,
            sealed.body,
            sealed.signature,
            timeout_s=timeout_s,
            max_body_bytes=largest_answer,
        )
    except TransportFailed as failure:
        logger.info("claim of dispatch %s got no answer: %s", dispatch.identity.dispatch_id, failure)
        return None


def _judge(
    answer: HttpAnswer, dispatch: AcceptedDispatch, settings: WorkerSettings, runtime_id: str, now: datetime
) -> Grant | Unclaimed | _RetryableRefusal | None:
    """What an answer means: the grant, the end of the claiming, a refusal for now, or None for no answer from the Hub."""
    dispatch_id = dispatch.identity.dispatch_id
    if answer.status == HTTP_OK:
        grant = _accepted_grant(answer, dispatch, settings, runtime_id, now)
        if grant is None:
            return Unclaimed(CAUSE_INVALID_ANSWER)
        logger.info("claim of dispatch %s granted: execution %s", dispatch_id, grant.identity.execution_id)
        return grant
    error = None
    if answer.status in HTTP_ERROR_STATUSES:
        error = accept_hub_error(
            answer.body, context=messages.receiver_context(MessageKind.HUB_ERROR, settings=settings)
        )
    if error is None and answer.status in GATEWAY_FAILURE_STATUSES:
        logger.info("claim of dispatch %s did not reach the Hub: status %d", dispatch_id, answer.status)
        return None
    if error is None:
        logger.info("claim of dispatch %s got an answer that is no hub.error: status %d", dispatch_id, answer.status)
        return Unclaimed(CAUSE_INVALID_ANSWER)
    code, retryable = error["code"], error["retryable"]
    logger.info("claim of dispatch %s refused: %s (retryable: %s)", dispatch_id, code, retryable)
    if retryable:
        return _RetryableRefusal(code, error.get("retry_after_s"))
    return Unclaimed(CAUSE_REFUSED, code)


def _accepted_grant(
    answer: HttpAnswer, dispatch: AcceptedDispatch, settings: WorkerSettings, runtime_id: str, now: datetime
) -> Grant | None:
    context = messages.receiver_context(
        MessageKind.TRANSCODE_CLAIM_GRANTED,
        settings=settings,
        now=now,
        dispatch=dispatch.identity,
        output_location_id=dispatch.output.location_id,
        runtime_id=runtime_id,
        encryption=dispatch.encryption,
    )
    try:
        granted = accept_signed(answer.body, answer.signature, context=context, keys=settings.hub_to_worker_keys)
    except NotAccepted as refusal:
        logger.info("claim of dispatch %s: the grant is not accepted at %s", dispatch.identity.dispatch_id, refusal)
        return None
    media_key = granted.get("media_key")
    # The dispatch fields of the identity are the worker's own record; the grant adds the execution.
    return Grant(
        identity=ExecutionIdentity.of(
            dispatch.identity,
            execution_id=granted["identity"]["execution_id"],
            execution_fence=granted["identity"]["execution_fence"],
        ),
        generation_id=granted["generation_id"],
        output_location_id=granted["output"]["location_id"],
        output_prefix=granted["output"]["prefix"],
        media_key_id=None if media_key is None else media_key["media_key_id"],
        media_key=None if media_key is None else bytes.fromhex(media_key["key_hex"]),
    )
