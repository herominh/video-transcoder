"""Shared helpers of the worker-path tests of contract v2: inbound, dispatch, transport, claim, outbox, handler.

Not a test module. What several of those files need lives here once: the synthetic values the fixtures
use (README section 14), an environment for WorkerSettings.from_env() that matches them, this suite's own
test keys (the fixtures carry no signature, so a test signs the exact bytes it sends), a clock and a
transport the test owns, and the builders of what the Hub answers.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Final, Mapping, NamedTuple, Union

from core.protocol import files, signing
from core.protocol.context import TrustedContext
from core.protocol.dispatch import AcceptedDispatch, accept_dispatch
from core.protocol.settings import WorkerSettings
from core.protocol.transport import HttpAnswer

# --- the fixtures' synthetic values (README section 14) -----------------------------------------

FIXTURE_NOW: Final = datetime(2030, 1, 1, tzinfo=timezone.utc)
PROTOCOL_VERSION: Final = "2.0.0-draft"
WORKER_AUDIENCE: Final = "worker:runpod-test"
HUB_KEY_ID: Final = "test-hub2wk-k1"
WORKER_KEY_ID: Final = "test-wk2hub-k1"
CELL_ID: Final = "cell-test-a"
RUNTIME_ID: Final = "88888888-8888-7888-8888-888888888888"
MEDIA_KEY_HEX: Final = "00112233445566778899aabbccddeeff"

REQUEST_ENCRYPTED: Final = "fixtures/request/P01-encrypted-three-renditions.json"
REQUEST_UNENCRYPTED: Final = "fixtures/request/P02-unencrypted.json"
GRANT_ENCRYPTED: Final = "fixtures/claim-granted/P24-grant-to-the-claiming-runtime.json"
GRANT_UNENCRYPTED: Final = "fixtures/claim-granted/P27-unencrypted-grant-without-media-key.json"
UNCLAIMED_REFUSED: Final = "fixtures/unclaimed/P25-refused-with-claim-conflict.json"
HUB_ERROR_FIXTURES: Final = (
    "fixtures/hub-error/P12-schema-violation-with-pointer.json",
    "fixtures/hub-error/P13-unsupported-version-lists-supported.json",
)
WORKER_DISPATCH_CONTEXT: Final = "fixtures/contexts/ctx-worker-dispatch.json"
WORKER_GRANT_CONTEXT: Final = "fixtures/contexts/ctx-worker-claim-grant.json"
WORKER_GRANT_CONTEXT_UNENCRYPTED: Final = "fixtures/contexts/ctx-worker-claim-grant-unencrypted.json"
WORKER_ERROR_CONTEXT: Final = "fixtures/contexts/ctx-worker-error.json"
HUB_CLAIM_CONTEXT: Final = "fixtures/contexts/ctx-hub-claim.json"
HUB_UNCLAIMED_POLL_CONTEXT: Final = "fixtures/contexts/ctx-hub-unclaimed-poll.json"

# --- what this suite configures -----------------------------------------------------------------

# Test-only keys of 32 bytes, one per role. Each is a canary too: no error, repr or log line may show its hex.
HUB_KEY: Final = bytes.fromhex("a17e5700" * 8)
SECOND_HUB_KEY: Final = bytes.fromhex("b27e5711" * 8)
WORKER_KEY: Final = bytes.fromhex("c37e5722" * 8)
UNCONFIGURED_KEY: Final = bytes.fromhex("d47e5733" * 8)
SECOND_HUB_KEY_ID: Final = "test-hub2wk-k2"
CLAIM_URL: Final = "https://hub.test.invalid/transcode/claim"
CALLBACK_URL: Final = "https://hub.test.invalid/transcode/callback"

HTTP_OK: Final = 200
HUB_ERROR_KIND: Final = "hub.error"
HUB_ERROR_MESSAGE_ID: Final = "99999999-9999-7999-8999-0000000000e0"
HUB_ERROR_TEXT: Final = "The Hub refuses this request."
JSON_SEPARATORS: Final = (",", ":")
MICROSECONDS_PER_MILLISECOND: Final = 1_000
# One distinct well-formed lowercase UUID per number (a version 4, variant 8 shape).
GENERATED_UUID_FORMAT: Final = "00000000-0000-4000-8000-{:012x}"
# In `changes` of modified_fixture(): remove the member instead of setting it.
REMOVED: Final = object()

_VALID_ENV: Final[Mapping[str, str]] = {
    "VH_ACCEPTED_PROTOCOL_VERSIONS": PROTOCOL_VERSION,
    "VH_WORKER_AUDIENCE": WORKER_AUDIENCE,
    "VH_HUB_TO_WORKER_KEYS": f"{HUB_KEY_ID}={HUB_KEY.hex()}",
    "VH_WORKER_TO_HUB_KEY_ID": WORKER_KEY_ID,
    "VH_WORKER_TO_HUB_KEY": WORKER_KEY.hex(),
    "VH_CLAIM_URL": CLAIM_URL,
    "VH_CALLBACK_URL": CALLBACK_URL,
    "VH_STORAGE_PROFILES": "default",
    "VH_STORAGE_DEFAULT_ENDPOINT": "https://storage.test.invalid",
    "VH_STORAGE_DEFAULT_ACCESS_KEY_ID": "test-access-key-id",
    "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY": "test-secret-access-key",
}


def worker_env(**changes: str | None) -> dict[str, str]:
    """The environment that matches the fixtures, with some variables replaced (a value) or removed (None)."""
    env = dict(_VALID_ENV)
    for variable, value in changes.items():
        if value is None:
            env.pop(variable, None)
        else:
            env[variable] = value
    return env


def worker_settings(**changes: str | None) -> WorkerSettings:
    """The worker's settings read from worker_env(**changes)."""
    return WorkerSettings.from_env(worker_env(**changes))


def two_hub_keys() -> str:
    """A VH_HUB_TO_WORKER_KEYS value holding both Hub-to-worker test keys, each under its own id."""
    return f"{HUB_KEY_ID}={HUB_KEY.hex()},{SECOND_HUB_KEY_ID}={SECOND_HUB_KEY.hex()}"


# --- bytes, moments and fixtures ----------------------------------------------------------------


def compact(document: Mapping[str, Any]) -> bytes:
    """The document as compact ASCII JSON, members in their own order (the writer convention, README 13)."""
    return json.dumps(document, ensure_ascii=True, separators=JSON_SEPARATORS).encode("ascii")


def moment(text: str) -> datetime:
    """The aware UTC datetime of a contract timestamp, read by the standard library's own parser."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def timestamp_text(instant: datetime) -> str:
    """The contract timestamp of an aware UTC datetime (`YYYY-MM-DDTHH:MM:SS.mmmZ`)."""
    if instant.utcoffset() != timedelta(0):
        raise ValueError("timestamp_text() takes an aware UTC datetime")
    milliseconds = instant.microsecond // MICROSECONDS_PER_MILLISECOND
    return f"{instant.strftime('%Y-%m-%dT%H:%M:%S')}.{milliseconds:03d}Z"


def seconds_from_now(seconds: float) -> str:
    """The contract timestamp of the instant `seconds` after the fixtures' now (negative: before it)."""
    return timestamp_text(FIXTURE_NOW + timedelta(seconds=seconds))


def modified_fixture(path: str, changes: Mapping[str, Any]) -> bytes:
    """A fixture as compact JSON after each `/a/b` pointer of `changes` was set to its value, or removed (REMOVED)."""
    document = files.read_json(path)
    for pointer, value in changes.items():
        if not pointer.startswith("/"):
            raise ValueError("a change is addressed by a JSON pointer such as /claim/expires_at")
        *parents, member = pointer[1:].split("/")
        target = document
        for parent in parents:
            target = target[parent]
        if value is REMOVED:
            del target[member]
        else:
            target[member] = value
    return compact(document)


def padded_to(body: bytes, size: int) -> bytes:
    """The JSON text followed by spaces up to exactly `size` bytes (insignificant whitespace counts, README 3)."""
    if len(body) > size:
        raise ValueError("the body is already longer than the wanted size")
    return body + b" " * (size - len(body))


# --- signing what the Hub sends -----------------------------------------------------------------


class SignedMessage(NamedTuple):
    """Exact message bytes and the `X-VH-Signature` value made over them."""

    body: bytes
    signature: str


def sign_message(body: bytes, key: bytes) -> SignedMessage:
    return SignedMessage(body, signing.sign(body, key))


def resigned_fixture(path: str, changes: Mapping[str, Any], key: bytes) -> SignedMessage:
    """A modified copy of a fixture (modified_fixture()), signed with `key` after the change."""
    return sign_message(modified_fixture(path, changes), key)


def signed_carrier(message_bytes: bytes, key: bytes) -> dict[str, str]:
    """The value of a RunPod job's `input`: the message text and the signature of its exact bytes."""
    return {"vh_message": message_bytes.decode("ascii"), "vh_signature": signing.sign(message_bytes, key)}


def accepted_dispatch(
    request: bytes, *, settings: WorkerSettings, now: datetime = FIXTURE_NOW
) -> AcceptedDispatch:
    """The dispatch the worker accepts from `request` when the Hub's configured key signed it."""
    return accept_dispatch(signed_carrier(request, HUB_KEY), settings=settings, now=now)


def catalog_context(path: str, *, now: datetime | None = None, **changes: Any) -> TrustedContext:
    """A context of the fixture catalog, at another `now` and with other top-level fields when given."""
    data = files.read_json(path)
    if now is not None:
        data["now"] = timestamp_text(now)
    data.update(changes)
    return TrustedContext.from_dict(data)


# --- what the Hub answers -----------------------------------------------------------------------


def signed_answer(body: bytes, key: bytes, status: int = HTTP_OK) -> HttpAnswer:
    """An HTTP answer whose body is signed with `key`, as the Hub signs a claim grant."""
    return HttpAnswer(status=status, body=body, signature=signing.sign(body, key))


def hub_error_body(code: str, *, retryable: bool, retry_after_s: int | None = None) -> bytes:
    """A `hub.error` as the Hub writes it (compact JSON), sent at the fixtures' now."""
    error: dict[str, Any] = {"code": code, "message": HUB_ERROR_TEXT, "retryable": retryable}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    return compact(
        {
            "protocol_version": PROTOCOL_VERSION,
            "message_kind": HUB_ERROR_KIND,
            "message_id": HUB_ERROR_MESSAGE_ID,
            "sent_at": timestamp_text(FIXTURE_NOW),
            "error": error,
        }
    )


def hub_error_answer(status: int, code: str, *, retryable: bool, retry_after_s: int | None = None) -> HttpAnswer:
    """The unsigned HTTP answer that carries hub_error_body()."""
    return HttpAnswer(
        status=status, body=hub_error_body(code, retryable=retryable, retry_after_s=retry_after_s), signature=None
    )


# --- a clock, an id source and a transport the test owns ----------------------------------------


class FakeClock:
    """Calling it gives the aware UTC time, monotonic() the seconds that passed.

    sleep() advances both and records the wait; advance() advances both; step_wall() moves the wall clock
    alone, as an operator or a time service would. Nothing else moves either.
    """

    def __init__(self, start: datetime = FIXTURE_NOW) -> None:
        self.start = start
        self.waits: list[float] = []
        self._now = start
        self._passed_s = 0.0

    def __call__(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._passed_s

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._passed_s += seconds

    def step_wall(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    @property
    def elapsed_s(self) -> float:
        return self._passed_s


def uuid_source(first: str | None = None) -> Callable[[], str]:
    """A source of distinct well-formed lowercase UUIDs: `first` when given, then one per call."""
    drawn = 0

    def draw() -> str:
        nonlocal drawn
        drawn += 1
        if first is not None and drawn == 1:
            return first
        return GENERATED_UUID_FORMAT.format(drawn)

    return draw


@dataclass(frozen=True)
class PostCall:
    """One POST the code under test made; `at` is the fake clock's time when it was made."""

    url: str
    body: bytes
    signature: str
    timeout_s: float
    max_body_bytes: int
    at: datetime | None


class UnexpectedPost(BaseException):
    """A POST no outcome was queued for. Not an Exception, so nothing under test can swallow it."""


Outcome = Union[HttpAnswer, Exception]


class _Outcomes:
    """The answers of one address: the queued ones in order, then `then` for every later POST."""

    def __init__(self, queued: tuple[Outcome, ...], then: Outcome | None) -> None:
        self.queued: Deque[Outcome] = deque(queued)
        self.then = then

    def next(self) -> Outcome | None:
        return self.queued.popleft() if self.queued else self.then


class FakeTransport:
    """A core.protocol.transport.Transport that never touches a network.

    Each POST is recorded, then answered with the next queued outcome: an HttpAnswer is returned, an
    exception (TransportFailed, or anything else a transport could raise) is raised. `then` answers every
    POST after the queue is empty. route() gives one address its own outcomes. `seconds_per_post` is how
    long a POST takes on the fake clock.
    """

    def __init__(
        self,
        *outcomes: Outcome,
        then: Outcome | None = None,
        clock: FakeClock | None = None,
        seconds_per_post: float = 0.0,
    ) -> None:
        self.calls: list[PostCall] = []
        self._default = _Outcomes(outcomes, then)
        self._routes: dict[str, _Outcomes] = {}
        self._clock = clock
        self._seconds_per_post = seconds_per_post

    def route(self, url: str, *outcomes: Outcome, then: Outcome | None = None) -> None:
        """Answer the POSTs to `url` with these outcomes instead of the common ones."""
        self._routes[url] = _Outcomes(outcomes, then)

    def calls_to(self, url: str) -> list[PostCall]:
        return [call for call in self.calls if call.url == url]

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        at = None if self._clock is None else self._clock()
        self.calls.append(PostCall(url, body, signature, timeout_s, max_body_bytes, at))
        if self._clock is not None:
            self._clock.advance(self._seconds_per_post)
        outcome = self._routes.get(url, self._default).next()
        if outcome is None:
            raise UnexpectedPost(f"POST number {len(self.calls)} has no queued outcome")
        if isinstance(outcome, Exception):
            # A fresh instance each time: a raised exception keeps the traceback of its last raise.
            raise type(outcome)(*outcome.args)
        return outcome
