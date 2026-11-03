"""The acquire claim uses the common exchange and accepts only acquire PUT authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from .acquire_dispatch import AcceptedAcquire, receiver_context
from .claim import (
    CAUSE_INVALID_ANSWER, HTTP_OK, Unclaimed, _RetryableRefusal, _sealed_claim, _send, exchange, judge_refusal,
)
from .inbound import NotAccepted, accept_signed
from .messages import ExecutionIdentity
from .registry import MessageKind
from .settings import AcquireSettings
from .transport import HttpAnswer, Transport


@dataclass(frozen=True)
class PartUrl:
    number: int
    url: str = field(repr=False)


@dataclass(frozen=True)
class AcquireGrant:
    identity: ExecutionIdentity
    output_location_id: str
    output_prefix: str
    parts: tuple[PartUrl, ...] = field(repr=False)
    result_url: str = field(repr=False)


def claim_execution(
    dispatch: AcceptedAcquire, *, settings: AcquireSettings, runtime_id: str, transport: Transport,
    clock: Callable[[], datetime], monotonic: Callable[[], float], sleep: Callable[[float], None],
    new_message_id: Callable[[], str], wall_time_ends_at: float,
) -> AcquireGrant | Unclaimed:
    return exchange(
        dispatch, clock=clock, monotonic=monotonic, sleep=sleep, wall_time_ends_at=wall_time_ends_at,
        build=lambda now: _sealed_claim(dispatch, settings, runtime_id, now, new_message_id()),
        send=lambda sealed, left: _send(
            dispatch, settings, transport, sealed, left,
            grant_kind=MessageKind.SOURCE_ACQUIRE_GRANTED, strict_timeout=True,
        ),
        judge=lambda answer, now: _judge(answer, dispatch, settings, runtime_id, now),
    )


def _judge(
    answer: HttpAnswer, dispatch: AcceptedAcquire, settings: AcquireSettings, runtime_id: str, now: datetime,
) -> AcquireGrant | Unclaimed | _RetryableRefusal:
    if answer.status != HTTP_OK:
        return judge_refusal(answer, dispatch.identity.dispatch_id, settings)
    try:
        granted = accept_signed(
            answer.body, answer.signature, keys=settings.hub_to_worker_keys,
            context=receiver_context(settings=settings, now=now, dispatch=dispatch, runtime_id=runtime_id),
        )
    except NotAccepted:
        return Unclaimed(CAUSE_INVALID_ANSWER)
    return AcquireGrant(
        identity=ExecutionIdentity.of(
            dispatch.identity, execution_id=granted["identity"]["execution_id"],
            execution_fence=granted["identity"]["execution_fence"],
        ),
        output_location_id=granted["output"]["location_id"], output_prefix=granted["output"]["prefix"],
        parts=tuple(PartUrl(part["n"], part["url"]) for part in granted["upload"]["parts"]),
        result_url=granted["upload"]["result_url"],
    )
