"""The capability client of protocol 2.2.0 (README 15.4, the runtime's rules), against a fake Hub that judges every
request with the contract's own validator and answers signed messages built in the test. No network."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pytest

from core.failure import WorkerFailure
from core.protocol.capabilities_client import FIRST_LEASE_S, CapabilityClient
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
    def __init__(self, *, time_left_s: float = DEADLINE_S, lease_s: int = LEASE_S) -> None:
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
            sleep=self.clock.sleep,
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
    assert setup.clock.waits == [5.0]
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


def test_ask_when_no_answer_was_ever_accepted_should_count_the_least_lease_from_the_client_made():
    # Arrange
    setup = _Setup()
    setup.hub.script.extend([TransportFailed("NoAnswer")] * 200)
    # Act
    refusal = _refusal(setup, lambda: setup.client.ask("probing"))
    # Assert
    assert refusal.failure.code == "lease_unrenewed"
    assert all(sent < FIRST_LEASE_S for sent in setup.hub.sent_at_monotonic)


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
