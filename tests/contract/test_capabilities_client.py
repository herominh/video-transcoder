"""The capability client of protocol 2.2.0 (README 15.4, the runtime's rules), against a fake Hub that judges every
request with the contract's own validator and answers signed messages built in the test. No network."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import pytest

from core.failure import Failure, WorkerFailure
from core.protocol import capabilities_client
from core.protocol.capabilities_client import FIRST_LEASE_S, CapabilityClient, ExchangeEnded
from core.protocol.claim import CLAIM_PATIENCE_S
from core.protocol.messages import CapabilityObject, CapabilityPut
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.contract.v2_support import SECOND_HUB_KEY, FakeClock, uuid_source
from tests.job.capability_fakes import (
    LEASE_S,
    SIGNATURE_CANARY,
    STORE_HOST,
    FakeHub,
    capability_claimed,
    capability_settings,
    hub_error,
)

DEADLINE_S = 3600.0
SEGMENT_MD5 = "AAAAAAAAAAAAAAAAAAAAAA=="
THUMBNAIL_MD5 = "BBBBBBBBBBBBBBBBBBBBBB=="
FINAL_CODES = (
    ("lease_expired", 403), ("stage_budget_exceeded", 403), ("capability_refused", 403),
    ("capability_conflict", 409), ("capability_limit_exceeded", 422), ("stale_capability_sequence", 409),
)


class _Setup:
    def __init__(
        self, *, time_left_s: float = DEADLINE_S, lease_s: int = LEASE_S, sleep: Callable[[float], None] | None = None
    ) -> None:
        self.claimed = capability_claimed()
        self.clock = FakeClock()
        self.hub = FakeHub(self.claimed, self.clock, lease_s=lease_s)
        self.time_left_s = time_left_s
        self.client = CapabilityClient(
            dispatch=self.claimed.dispatch,
            grant=self.claimed.grant,
            runtime_id=self.claimed.runtime_id,
            settings=capability_settings(),
            transport=self.hub,
            clock=self.clock,
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep if sleep is None else sleep,
            new_message_id=uuid_source(),
            time_left=lambda: self.time_left_s - self.clock.monotonic(),
        )


def _put(*paths: str) -> CapabilityPut:
    md5s = (SEGMENT_MD5, THUMBNAIL_MD5)
    return CapabilityPut(
        lifetime_seconds=180,
        objects=tuple(CapabilityObject(path=path, size_bytes=100 + i, md5=md5s[i % 2]) for i, path in enumerate(paths)),
    )


def _changed(change: Callable[[dict[str, Any]], None], **grant: Any) -> Callable[[dict[str, Any], FakeHub], HttpAnswer]:
    def answer(request: dict[str, Any], hub: FakeHub) -> HttpAnswer:
        document = hub.granted_document(request)
        change(document)
        return hub.grant(request, document=document, **grant)

    return answer


def _refusal(setup: _Setup, call: Callable[[], Any]) -> WorkerFailure:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value


# --- the sequence and the retries --------------------------------------------------------------------------


def test_ask_when_answers_come_and_one_is_lost_should_raise_seq_by_one_per_message():
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([TransportFailed("NoAnswer"), "grant"])
    # Act
    setup.client.ask("probing", source=True)
    setup.client.ask("downloading")
    # Assert
    assert [request["seq"] for request in setup.hub.requests] == [1, 2, 3]
    assert len({request["message_id"] for request in setup.hub.requests}) == 3


def test_ask_when_the_hub_refuses_for_now_should_ask_again_after_its_retry_after_and_be_granted():
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([hub_error(503, "service_unavailable", retryable=True, retry_after_s=5), "grant"])
    # Act
    answer = setup.client.ask("probing", source=True)
    # Assert
    assert answer.source is not None
    assert sum(setup.clock.waits) == pytest.approx(5.0)
    assert setup.client.lost() is None


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_ask_when_a_temporary_status_carries_no_hub_error_should_ask_again(status):
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([HttpAnswer(status, b"busy", None), "grant"])
    # Act
    setup.client.ask("probing")
    # Assert
    assert len(setup.hub.requests) == 2


def test_ask_when_the_hub_keeps_refusing_for_now_should_stop_before_the_lease_ends_as_lease_unrenewed():
    # Arrange: an accepted answer gives a lease of 60 s, then the Hub refuses for now every time.
    setup = _Setup(lease_s=60)
    setup.client.ask("probing")
    setup.hub.script.extend([HttpAnswer(503, b"", None)] * 50)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("downloading"))
    # Assert
    failure = refusal.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("authority_refused", "lease_unrenewed", False)
    assert all(sent < 60.0 for sent in setup.hub.sent_at_monotonic)
    assert setup.client.lost() is not None


def test_ask_when_the_deadline_comes_before_the_lease_should_stop_there_as_a_deadline():
    # Arrange
    setup = _Setup(time_left_s=20.0)
    setup.hub.script.extend([HttpAnswer(503, b"", None)] * 50)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("probing"))
    # Assert
    assert (refusal.failure.error_class, refusal.failure.code) == ("deadline_exceeded", "capability_deadline_exceeded")
    assert all(sent < 20.0 for sent in setup.hub.sent_at_monotonic)


def test_ask_when_no_answer_was_ever_accepted_should_stop_within_the_claims_patience_as_lease_unrenewed():
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([TransportFailed("NoAnswer")] * 200)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("probing"))
    # Assert: the waits are 1, 2, 4, 8, then 15 s each, so the last send lies within 15 s of the patience's end.
    assert refusal.failure.code == "lease_unrenewed"
    last_sent = setup.hub.sent_at_monotonic[-1]
    assert CLAIM_PATIENCE_S - 15 <= last_sent < CLAIM_PATIENCE_S
    assert setup.clock.monotonic() < FIRST_LEASE_S


# --- final refusals -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("code,status", FINAL_CODES, ids=[code for code, _ in FINAL_CODES])
def test_ask_when_the_hub_refuses_for_good_should_end_the_work_as_authority_refused_with_its_code(code, status):
    # Arrange
    setup = _Setup()
    setup.hub.script.append(hub_error(status, code, retryable=False))
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("probing", source=True))
    # Assert
    failure = refusal.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("authority_refused", code, False)
    lost = setup.client.lost()
    assert lost is not None and lost.failure == failure


def test_ask_when_the_work_has_ended_should_send_no_further_request():
    # Arrange
    setup = _Setup()
    setup.hub.script.append(hub_error(403, "capability_refused", retryable=False))
    _refusal(setup, lambda: setup.client.ask("probing"))
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("downloading", put=_put("thumbnail.jpg")))
    # Assert
    assert refusal.failure.code == "capability_refused"
    assert len(setup.hub.requests) == 1


@pytest.mark.parametrize("answer,code", [
    (HttpAnswer(413, b"too large", None), "payload_too_large"),
    (HttpAnswer(500, b"<html>", None), "capability_answer_invalid"),
    (HttpAnswer(302, b"", None), "capability_answer_invalid"),
    (HttpAnswer(200, b"{}", None), "capability_answer_invalid"),
], ids=["non-envelope-413", "non-envelope-500", "redirect", "unsigned-200"])
def test_ask_when_the_answer_is_no_hub_error_and_no_temporary_refusal_should_end_the_work(answer, code):
    # Arrange
    setup = _Setup()
    setup.hub.script.append(answer)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("probing"))
    # Assert
    assert (refusal.failure.error_class, refusal.failure.code) == ("authority_refused", code)
    assert len(setup.hub.requests) == 1


# --- answers that are not accepted ----------------------------------------------------------------------------


def _set(pointer: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        *parents, member = pointer.strip("/").split("/")
        target: Any = document
        for parent in parents:
            target = target[int(parent)] if parent.isdigit() else target[parent]
        target[int(member) if member.isdigit() else member] = value

    return change


def _replace_in_url(index: int, old: str, new: str) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        item = document["put"]["objects"][index]
        item["url"] = item["url"].replace(old, new, 1)

    return change


def _drop_last(document: dict[str, Any]) -> None:
    document["put"]["objects"].pop()


def _add_one(document: dict[str, Any]) -> None:
    extra = dict(document["put"]["objects"][0])
    extra["path"] = "extra.ts"
    document["put"]["objects"].append(extra)


def _expires_later(document: dict[str, Any]) -> None:
    document["put"]["expires_at"] = document["put"]["expires_at"].replace("00:03:00", "00:04:00")


INVALID_ANSWERS = {
    "wrong-signature": _changed(lambda document: None, key=SECOND_HUB_KEY),
    "wrong-seq": _changed(_set("/seq", 9)),
    "another-stage": _changed(_set("/stage/name", "transcoding")),
    "missing-path": _changed(_drop_last),
    "extra-path": _changed(_add_one),
    "another-host": _changed(_replace_in_url(0, STORE_HOST, "elsewhere.test.invalid")),
    "another-bucket": _changed(_replace_in_url(0, "/test-output-bucket/", "/other-output-bucket/")),
    "another-key": _changed(_replace_in_url(0, "segment_0000.ts?", "segment_0009.ts?")),
    "other-signed-headers": _changed(_replace_in_url(0, "content-md5%3Bhost", "host")),
    "expiry-later-than-asked": _changed(_expires_later),
}


@pytest.mark.parametrize("name", list(INVALID_ANSWERS))
def test_ask_when_the_answer_is_not_accepted_should_discard_it_and_end_the_work_as_answer_invalid(name):
    # Arrange
    setup = _Setup()
    setup.hub.script.append(INVALID_ANSWERS[name])
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("uploading", put=_put("v360/segment_0000.ts", "thumbnail.jpg")))
    # Assert
    assert (refusal.failure.error_class, refusal.failure.code, refusal.failure.retryable) == (
        "authority_refused", "capability_answer_invalid", False)
    assert setup.client.lost() is not None


def test_ask_when_every_member_is_as_asked_should_accept_the_same_answer_the_negatives_change():
    # Arrange: the control of the negatives above.
    setup = _Setup()
    setup.hub.script.append(_changed(lambda document: None))
    # Act
    answer = setup.client.ask("uploading", put=_put("v360/segment_0000.ts", "thumbnail.jpg"))
    # Assert
    assert [url.path for url in answer.puts] == ["v360/segment_0000.ts", "thumbnail.jpg"]


# --- the lease and the URLs, counted from the send -----------------------------------------------------------


def test_answer_when_it_arrives_later_should_count_the_lease_and_the_urls_from_the_instant_the_request_was_sent():
    # Arrange: the Hub answers 7 s after the request was sent (at 0), with a lease of 900 s and URLs of 180 s.
    setup = _Setup()

    def late(request: dict[str, Any], hub: FakeHub) -> HttpAnswer:
        setup.clock.advance(7.0)
        return hub.grant(request)

    setup.hub.script.append(late)
    # Act
    answer = setup.client.ask("uploading", put=_put("thumbnail.jpg"))
    # Assert
    assert answer.lease_ends_at == pytest.approx(900.0)
    assert answer.puts[0].use_before == pytest.approx(180.0)


def test_lost_when_the_lease_has_run_out_by_the_runtimes_count_should_end_the_work_as_lease_unrenewed():
    # Arrange
    setup = _Setup()
    setup.client.ask("probing")
    setup.clock.advance(LEASE_S)
    # Act
    lost = setup.client.lost()
    # Assert
    assert lost is not None and lost.failure.code == "lease_unrenewed"
    assert _refusal(setup, lambda: setup.client.ask("downloading")).failure.code == "lease_unrenewed"
    assert len(setup.hub.requests) == 1


def test_stage_when_set_back_should_be_refused_and_the_duration_should_be_set_once():
    # Arrange
    setup = _Setup()
    setup.client.set_stage("transcoding")
    setup.client.set_media_duration(2000)
    # Act / Assert
    with pytest.raises(ValueError):
        setup.client.set_stage("downloading")
    with pytest.raises(ValueError):
        setup.client.set_media_duration(3000)
    setup.client.heartbeat()
    request = setup.hub.requests[-1]
    assert (request["stage"], request["media_duration_ms"], request["put"], request["source"]) == (
        "transcoding", 2000, None, False)


# --- what is never logged or shown ---------------------------------------------------------------------------


def test_exchange_when_logged_and_shown_should_hold_no_url_md5_path_list_or_runtime_id(caplog):
    # Arrange
    setup = _Setup()
    setup.hub.script.extend(["grant", INVALID_ANSWERS["another-host"]])
    caplog.set_level(logging.DEBUG)
    # Act
    answer = setup.client.ask("uploading", put=_put("v360/segment_0000.ts", "thumbnail.jpg"))
    refusal = _refusal(setup, lambda: setup.client.ask("uploading", put=_put("v360/segment_0001.ts")))
    shown = repr(answer) + str(refusal) + repr(refusal.failure)
    # Assert: a log line names no path either; a repr may name the path a URL was granted for, never the URL.
    for secret in (SIGNATURE_CANARY, STORE_HOST, SEGMENT_MD5, THUMBNAIL_MD5, setup.claimed.runtime_id, "X-Amz"):
        assert secret not in caplog.text
        assert secret not in shown
    for path in ("segment_0000", "segment_0001", "thumbnail.jpg"):
        assert path not in caplog.text


# --- the claim's patience, late answers, the ending of the work --------------------------------------------------


def test_ask_when_a_temporary_status_names_a_retry_after_should_wait_that_long():
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([HttpAnswer(429, b"", None, retry_after_s=7.0), "grant"])
    # Act
    setup.client.ask("probing")
    # Assert
    assert sum(setup.clock.waits) == pytest.approx(7.0)


def test_answer_when_it_comes_after_the_lease_ended_by_the_runtimes_count_should_renew_nothing_and_end_the_work():
    # Arrange: a first answer gives a lease of 900 s; the next one is accepted only 900 s after its request was sent.
    setup = _Setup()
    setup.client.ask("probing")
    accepted_at = setup.client.last_accepted_sent_at()

    def late(request: dict[str, Any], hub: FakeHub) -> HttpAnswer:
        setup.clock.advance(LEASE_S)
        return hub.grant(request)

    setup.hub.script.append(late)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("downloading"))
    # Assert
    assert refusal.failure.code == "lease_unrenewed"
    assert setup.client.last_accepted_sent_at() == accepted_at


def test_ask_once_when_the_route_refuses_or_is_silent_should_return_none_after_one_request_and_end_nothing():
    # Arrange
    for answer in (HttpAnswer(503, b"", None), TransportFailed("NoAnswer"),
                   hub_error(403, "capability_refused", retryable=False)):
        setup = _Setup()
        setup.client.ask("finalizing")
        setup.hub.script.append(answer)
        # Act
        granted = setup.client.ask_once(put=_put("result.json"))
        # Assert
        assert granted is None
        assert len(setup.hub.requests) == 2
        assert setup.clock.waits == []
        assert setup.client.lost() is None


def test_ask_once_when_the_work_has_ended_should_still_ask_once_and_renew_no_lease():
    # Arrange
    setup = _Setup()
    setup.client.ask("finalizing")
    accepted_at = setup.client.last_accepted_sent_at()
    setup.clock.advance(10)
    setup.client.end()
    # Act
    granted = setup.client.ask_once(put=_put("result.json"))
    # Assert
    assert granted is not None and [url.path for url in granted.puts] == ["result.json"]
    assert setup.client.last_accepted_sent_at() == accepted_at


def test_ask_when_the_work_has_ended_should_send_nothing_and_raise_no_failure():
    # Arrange
    setup = _Setup()
    setup.client.ask("transcoding")
    setup.client.end()
    # Act / Assert
    with pytest.raises(ExchangeEnded):
        setup.client.heartbeat()
    assert len(setup.hub.requests) == 1
    assert setup.client.lost() is None


def test_ask_when_the_work_ends_during_its_retries_should_stop_waiting_and_send_no_further_request():
    # Arrange: the Hub refuses for now; end() comes while the client waits before its next try.
    waited: list[float] = []

    def sleep_until_ended(seconds: float) -> None:
        waited.append(seconds)
        setup.clock.advance(seconds)
        setup.client.end()

    setup = _Setup(sleep=sleep_until_ended)
    setup.client.ask("transcoding")
    setup.hub.script.extend([HttpAnswer(503, b"", None)] * 20)
    # Act / Assert
    with pytest.raises(ExchangeEnded):
        setup.client.heartbeat()
    assert len(setup.hub.requests) == 2
    assert len(waited) == 1


def test_fail_when_the_runtime_faults_should_end_the_work_with_that_failure():
    # Arrange
    setup = _Setup()
    failure = Failure("internal_error", "heartbeat_failed", True, "the heartbeat failed: RuntimeError")
    # Act
    setup.client.fail(failure)
    # Assert
    lost = setup.client.lost()
    assert lost is not None and lost.failure == failure


# --- ask_once behind an exchange in flight (round 2) -----------------------------------------------------------


def _beat_held_until(setup: _Setup, release: threading.Event, answer: Callable[[dict[str, Any], FakeHub], HttpAnswer]):
    """Start a heartbeat on a thread of its own whose answer the Hub holds until `release`; return the thread and the
    event that says the beat is in flight."""
    in_flight = threading.Event()

    def held(request: dict[str, Any], hub: FakeHub) -> HttpAnswer:
        in_flight.set()
        release.wait(5.0)
        return answer(request, hub)

    setup.hub.script.append(held)

    def beat() -> None:
        try:
            setup.client.heartbeat()
        except WorkerFailure:
            pass

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    assert in_flight.wait(5.0)
    return thread


def test_ask_once_when_an_exchange_holds_the_lock_should_give_up_within_one_exchange(monkeypatch):
    # Arrange: one exchange lasts at most 0.3 s here; the heartbeat's is held for longer.
    monkeypatch.setattr(capabilities_client, "CAPABILITY_REQUEST_TIMEOUT_S", 0.3)
    setup = _Setup()
    setup.client.ask("finalizing")
    release = threading.Event()
    thread = _beat_held_until(setup, release, lambda request, hub: hub.grant(request))
    started = time.monotonic()
    try:
        # Act
        granted = setup.client.ask_once(put=_put("result.json"))
        elapsed = time.monotonic() - started
    finally:
        release.set()
        thread.join(5.0)
    # Assert
    assert granted is None
    assert elapsed < 1.5
    assert len(setup.hub.requests) == 2


class _ReportingLock:
    """The client's exchange lock, saying when someone asks for it: a test then knows ask_once is waiting."""

    def __init__(self, inner: Any, asked: threading.Event) -> None:
        self._inner = inner
        self._asked = asked

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self._asked.set()
        return self._inner.acquire(blocking, timeout)

    def release(self) -> None:
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exception: object) -> None:
        self.release()


def test_ask_once_when_the_exchange_it_waited_for_ends_the_work_should_send_nothing():
    # Arrange: the heartbeat in flight is refused for good, its answer held until ask_once waits for the lock.
    setup = _Setup()
    setup.client.ask("finalizing")
    release, asked = threading.Event(), threading.Event()
    thread = _beat_held_until(
        setup, release, lambda request, hub: hub_error(403, "lease_expired", retryable=False))
    setup.client._exchange_lock = _ReportingLock(setup.client._exchange_lock, asked)
    threading.Thread(target=lambda: asked.wait(5.0) and release.set(), daemon=True).start()
    # Act
    granted = setup.client.ask_once(put=_put("result.json"))
    thread.join(5.0)
    # Assert
    assert granted is None
    assert len(setup.hub.requests) == 2
    assert setup.client.lost().failure.code == "lease_expired"
