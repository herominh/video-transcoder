"""The claim exchange of contract v2, the worker's side (README section 15.1).

A dispatch names work, not a running process: before it reads the source or writes any object, a runtime
sends `transcode.claim` to the Hub's claim address (from its trusted configuration, https only) and works
only on a `200` whose body verifies under a Hub-to-worker key and is an accepted `transcode.claim.granted`
for exactly this dispatch, this runtime and this encryption. Every other ending is an Unclaimed outcome,
and the runtime does no work:

    refused         a `hub.error` that is not retryable, or still retryable when the claiming ends
    unreachable     no answer until the claiming ends
    invalid_answer  any other answer: a grant that does not verify or is not accepted, a `200` that is
                    not a grant, a redirect, a body that is no `hub.error` of this version

A `hub.error` is read only under a 4xx or 5xx status: under any other status its body is "any other
answer". It claims again only after a transport failure or a retryable `hub.error`, with the same runtime
id and token and a new message id and send time, never at or after the dispatch's `claim.expires_at`, and
for no longer than CLAIM_PATIENCE_S: a billed runtime does not wait out a bootstrap that can last an hour.
The contract allows the retries, it does not require them; giving up earlier reports the same outcome as
the expiry would. The expiry is judged on the wall clock, as the contract states it; the patience is
measured on a monotonic reading, so a wall clock that is stepped while the Hub does not answer cannot
stretch it.

The body of a claim and of a claim response is never logged, stored or put in an error, and the media key
of a grant exists only inside the returned Grant, whose repr() does not show it (README 15.1, rules 4, 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import messages
from .dispatch import AcceptedDispatch
from .inbound import NotAccepted, accept_hub_error, accept_signed
from .messages import ExecutionIdentity
from .outbox import seal_message
from .registry import MessageKind, load_registry
from .settings import WorkerSettings
from .transport import HttpAnswer, Transport, TransportFailed

logger = logging.getLogger(__name__)

HTTP_OK = 200
_HTTP_ERROR_STATUSES = range(400, 600)
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
    """An accepted `transcode.claim.granted`: the execution, its generation, where it writes and its media key."""

    identity: ExecutionIdentity
    generation_id: str
    output_location_id: str
    output_prefix: str
    media_key_id: str | None
    media_key: bytes | None = field(repr=False)


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
) -> Grant | Unclaimed:
    """Claim the dispatch for this runtime.

    `clock` gives the aware current time, `monotonic` seconds that only ever move forward, and `sleep` waits
    that many seconds.
    """
    patience_ends_at = monotonic() + CLAIM_PATIENCE_S

    def time_left_s(now: datetime) -> float:
        return min((dispatch.claim_expires_at - now).total_seconds(), patience_ends_at - monotonic())

    last_refusal_code: str | None = None
    attempt = 0
    while True:
        now = clock()
        left_s = time_left_s(now)
        if left_s <= 0:
            break
        wait_s = float(CLAIM_BACKOFF_S[min(attempt, len(CLAIM_BACKOFF_S) - 1)])
        answer = _send(dispatch, settings, runtime_id, transport, now, new_message_id(), left_s)
        if answer is not None:
            outcome = _judge(answer, dispatch, settings, runtime_id, clock())
            if not isinstance(outcome, _RetryableRefusal):
                return outcome
            last_refusal_code = outcome.code
            wait_s = max(wait_s, float(outcome.retry_after_s or 0))
        attempt += 1
        if wait_s >= time_left_s(clock()):
            break
        sleep(wait_s)
    if last_refusal_code is not None:
        logger.info("claim of dispatch %s ended refused: %s", dispatch.identity.dispatch_id, last_refusal_code)
        return Unclaimed(CAUSE_REFUSED, last_refusal_code)
    logger.info("claim of dispatch %s ended unreachable", dispatch.identity.dispatch_id)
    return Unclaimed(CAUSE_UNREACHABLE)


def _send(
    dispatch: AcceptedDispatch,
    settings: WorkerSettings,
    runtime_id: str,
    transport: Transport,
    now: datetime,
    message_id: str,
    left_s: float,
) -> HttpAnswer | None:
    """One claim request, `left_s` seconds before the claiming ends; None when it got no answer."""
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
    sealed = seal_message(claim, context, settings=settings)
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
) -> Grant | Unclaimed | _RetryableRefusal:
    dispatch_id = dispatch.identity.dispatch_id
    if answer.status == HTTP_OK:
        grant = _accepted_grant(answer, dispatch, settings, runtime_id, now)
        if grant is None:
            return Unclaimed(CAUSE_INVALID_ANSWER)
        logger.info("claim of dispatch %s granted: execution %s", dispatch_id, grant.identity.execution_id)
        return grant
    error = None
    if answer.status in _HTTP_ERROR_STATUSES:
        error = accept_hub_error(
            answer.body, context=messages.receiver_context(MessageKind.HUB_ERROR, settings=settings)
        )
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
