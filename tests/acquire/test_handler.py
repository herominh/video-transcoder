"""Only an authenticated acquire grant can authorize traffic, and one terminal survives delivery faults."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from core.protocol import files, signing
from core.protocol.acquire_claim import claim_execution
from core.protocol.acquire_dispatch import accept_dispatch
from core.protocol.dispatch import DispatchRefused
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import MessageKind, load_registry
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.acquire.support import EXECUTION, PART_SIZE, Scenario, grant_for, wire
from tests.contract.v2_support import (
    ACQUIRE_PROTOCOL_VERSION, GRANT_UNENCRYPTED, HUB_KEY, PROTOCOL_VERSION, REQUEST_ENCRYPTED, WORKER_KEY,
    hub_error_answer, modified_fixture, signed_answer, signed_carrier,
)
from tests.drive_import.fakes import Fail, Put
from wrappers.runpod_acquire_handler import JobRefused


def test_acquire_when_granted_should_deliver_identical_signed_bytes_on_all_three_channels() -> None:
    # Arrange
    scenario = Scenario()
    # Act
    output = scenario.run()
    message = json.loads(output["vh_message"])
    # Assert
    assert message["message_kind"] == "source.acquire.result.completed"
    assert message["protocol_version"] == "2.1.0"
    assert message["identity"]["execution_id"] == EXECUTION
    assert message["bytes"] == scenario.file.size
    assert message["checksum_verified"] == "sha256"
    assert signing.verify(output["vh_message"].encode("ascii"), WORKER_KEY, output["vh_signature"])
    assert json.loads(scenario.http.result_puts[0].body) == output
    assert scenario.hub.callbacks[0].body.decode("ascii") == output["vh_message"]
    assert scenario.hub.callbacks[0].signature == output["vh_signature"]
    assert scenario.http.result_puts[0].headers["Content-Type"] == "application/json"
    assert int(scenario.http.result_puts[0].headers["Content-Length"]) == len(scenario.http.result_puts[0].body)
    assert scenario.http.part_puts[0].headers["Content-MD5"]
    assert all(request.method in ("GET", "PUT") for request in scenario.http.requests)
    assert scenario.http.closed


@pytest.mark.parametrize("carrier", [{}, {"external": {}}, None, {"vh_message": "bad", "vh_signature": "bad"}])
def test_acquire_when_the_dispatch_is_invalid_should_send_no_claim_or_transfer(carrier: Any) -> None:
    # Arrange
    scenario = Scenario()
    # Act / Assert
    with pytest.raises(JobRefused):
        scenario.run(carrier if carrier is not None else "not-a-carrier")
    assert scenario.hub.posts == []
    assert scenario.http.requests == []
    assert scenario.http.closed


@pytest.mark.parametrize("change", ["signature", "family", "version", "integer-float", "integer-exponent", "zero"])
def test_acquire_dispatch_when_not_acceptable_should_fail_before_any_network_work(change: str) -> None:
    # Arrange
    scenario = Scenario()
    raw = wire(scenario.request)
    if change == "family":
        scenario.request["message_kind"] = "transcode.request"
        raw = wire(scenario.request)
    elif change == "version":
        scenario.request["protocol_version"] = "2.0.0"
        raw = wire(scenario.request)
    elif change == "integer-float":
        raw = raw.replace(b'"size":37', b'"size":37.0')
    elif change == "integer-exponent":
        raw = raw.replace(b'"size":37', b'"size":37e0')
    elif change == "zero":
        raw = raw.replace(b'"size":37', b'"size":0')
    carrier = signed_carrier(raw, HUB_KEY)
    if change == "signature":
        carrier["vh_signature"] = "hmac-sha256=" + "0" * 64
    # Act / Assert
    with pytest.raises(JobRefused):
        scenario.run(carrier)
    assert not scenario.hub.posts and not scenario.http.requests


@pytest.mark.parametrize("version", [PROTOCOL_VERSION, ACQUIRE_PROTOCOL_VERSION])
def test_acquire_endpoint_when_sent_a_signed_transcode_request_should_refuse_the_kind_and_do_no_work(version: str) -> None:
    # Arrange: the version is listed, the signature is the Hub's; only the kind is not this endpoint's.
    scenario = Scenario()
    request = modified_fixture(REQUEST_ENCRYPTED, {"/protocol_version": version})
    carrier = signed_carrier(request, HUB_KEY)
    # Act
    with pytest.raises(DispatchRefused) as refused:
        accept_dispatch(carrier, settings=scenario.runtime.settings, now=scenario.clock.now())
    with pytest.raises(JobRefused) as job_refused:
        scenario.run(carrier)
    # Assert
    assert refused.value.verdict is not None
    assert (refused.value.verdict.layer, refused.value.verdict.reason) == (Layer.VERSION, ReasonCode.UNEXPECTED_MESSAGE_KIND)
    assert str(job_refused.value) == "the dispatch is refused at validation: version/unexpected_message_kind"
    assert not scenario.hub.posts and not scenario.http.requests


@pytest.mark.parametrize("answer,cause,code", [
    # An encode grant at the dispatch's version: only the kind is wrong, so only the kind check refuses it.
    (signed_answer(modified_fixture(GRANT_UNENCRYPTED, {"/protocol_version": ACQUIRE_PROTOCOL_VERSION}), HUB_KEY),
     "invalid_answer", None),
    (HttpAnswer(200, b"{}", None), "invalid_answer", None),
    (HttpAnswer(302, b"", None), "invalid_answer", None),
    (HttpAnswer(413, b"not an envelope", None), "refused", "payload_too_large"),
    (hub_error_answer(409, "claim_conflict", retryable=False), "refused", "claim_conflict"),
])
def test_acquire_claim_when_not_granted_should_report_unclaimed_without_reading_or_writing(
    answer: HttpAnswer, cause: str, code: str | None,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.hub.claim_actions = [answer]
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["message_kind"] == "transcode.unclaimed"
    assert output["cause"] == cause
    assert output.get("hub_error_code") == code
    assert scenario.http.requests == []


def test_acquire_when_a_provider_duplicates_the_start_should_draw_a_new_runtime_and_obey_conflict() -> None:
    # Arrange
    scenario = Scenario()
    first = scenario.run()
    transfers = len(scenario.http.requests)
    scenario.hub.claim_actions = [hub_error_answer(409, "claim_conflict", retryable=False)]
    # Act
    duplicate = scenario.run()
    claims = [json.loads(post.body) for post in scenario.hub.claims]
    # Assert
    assert json.loads(first["vh_message"])["message_kind"] == "source.acquire.result.completed"
    assert json.loads(duplicate["vh_message"])["cause"] == "refused"
    assert claims[0]["claim"]["runtime_id"] != claims[1]["claim"]["runtime_id"]
    assert len(scenario.http.requests) == transfers


def test_acquire_claim_when_a_grant_answer_is_lost_should_retry_with_fresh_envelope_and_accept_resigned_authority() -> None:
    # Arrange
    scenario = Scenario()
    scenario.hub.claim_actions = [TransportFailed("lost response")]
    # Act
    result = json.loads(scenario.run()["vh_message"])
    first, second = [json.loads(post.body) for post in scenario.hub.claims]
    old, fresh = scenario.hub.grants
    # Assert
    assert result["message_kind"] == "source.acquire.result.completed"
    assert first["claim"] == second["claim"]
    assert first["identity"] == second["identity"]
    assert first["message_id"] != second["message_id"] and first["sent_at"] != second["sent_at"]
    assert scenario.hub.claims[0].signature != scenario.hub.claims[1].signature
    assert old["identity"] == fresh["identity"] and old["output"] == fresh["output"]
    old_url, fresh_url = old["upload"]["parts"][0]["url"], fresh["upload"]["parts"][0]["url"]
    assert old_url != fresh_url
    before, after = parse_qs(urlsplit(old_url).query), parse_qs(urlsplit(fresh_url).query)
    assert int(before["X-Amz-Expires"][0]) == int(after["X-Amz-Expires"][0]) + 1
    assert before["uploadId"] == after["uploadId"] and before["partNumber"] == after["partNumber"]
    assert scenario.http.part_puts[0].url == fresh_url
    assert all(post.cap == load_registry().max_bytes(MessageKind.SOURCE_ACQUIRE_GRANTED) for post in scenario.hub.claims)


@pytest.mark.parametrize("answer,wait", [
    (HttpAnswer(503, b"edge", None, retry_after_s=4), 4),
    (hub_error_answer(503, "service_unavailable", retryable=True, retry_after_s=3), 3),
])
def test_acquire_claim_when_transiently_refused_should_honor_retry_after(answer: HttpAnswer, wait: int) -> None:
    # Arrange
    scenario = Scenario()
    scenario.hub.claim_actions = [answer]
    # Act
    scenario.run()
    # Assert
    assert scenario.clock.slept == [wait]
    assert len(scenario.hub.claims) == 2


@pytest.mark.parametrize("end,expected_claims,cause", [("expiry", 0, "expired"), ("wall", 1, "unreachable"), ("patience", 11, "unreachable")])
def test_acquire_claim_when_its_bound_is_reached_should_stop_without_transfer(end: str, expected_claims: int, cause: str) -> None:
    # Arrange
    scenario = Scenario()
    if end == "expiry":
        scenario.clock.advance(60)
    elif end == "wall":
        scenario.request["limits"]["max_wall_time_ms"] = 500
    else:
        scenario.request["limits"]["max_wall_time_ms"] = 1_000_000
        scenario.request["claim"]["expires_at"] = "2030-01-01T00:50:00.000Z"
    scenario.hub.claim_actions = [TransportFailed("timeout")] * 100
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["cause"] == cause
    assert not scenario.http.requests
    assert len(scenario.hub.claims) == expected_claims
    assert sum(scenario.clock.slept) < 120
    if end == "wall":
        assert scenario.hub.claims[0].timeout <= 0.5


@pytest.mark.parametrize("channel", ["result-put", "callback", "both"])
def test_acquire_when_terminal_delivery_fails_should_preserve_completed_provider_bytes(channel: str) -> None:
    # Arrange
    scenario = Scenario()
    if channel in ("result-put", "both"):
        scenario.http.puts[scenario.urls()[1]] = [Fail(OSError("secret response body"))] * 4
    if channel in ("callback", "both"):
        scenario.hub.callback_actions = [TransportFailed("secret URL and key")] * 3
    # Act
    output = scenario.run()
    # Assert
    assert json.loads(output["vh_message"])["message_kind"] == "source.acquire.result.completed"
    assert all(post.body.decode("ascii") == output["vh_message"] and post.signature == output["vh_signature"]
               for post in scenario.hub.callbacks)
    if channel == "callback":
        assert json.loads(scenario.http.result_puts[0].body) == output


def test_acquire_when_result_storage_reaches_the_deadline_should_return_the_sealed_completed_terminal() -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[scenario.urls()[1]] = [Put(seconds_after=60)]
    # Act
    output = scenario.run()
    # Assert
    assert json.loads(output["vh_message"])["message_kind"] == "source.acquire.result.completed"
    assert json.loads(scenario.http.result_puts[0].body) == output
    assert scenario.hub.callbacks == []


@pytest.mark.parametrize("fault", ["wrong-key", "wrong-key-id", "byte-change", "wrong-status", "oversize"])
def test_acquire_claim_when_grant_authentication_or_framing_is_invalid_should_use_no_capability(fault: str) -> None:
    # Arrange
    scenario = Scenario()

    def answer(grant: dict[str, Any]) -> HttpAnswer:
        if fault == "wrong-key-id":
            grant["key_id"] = "test-unknown-k1"
        body = wire(grant)
        if fault == "oversize":
            body += b" " * (2_097_153 - len(body))
        signature = signing.sign(body, WORKER_KEY if fault == "wrong-key" else HUB_KEY)
        if fault == "byte-change":
            body += b" "
        return HttpAnswer(201 if fault == "wrong-status" else 200, body, signature)

    scenario.hub.claim_actions = [answer]
    # Act
    report = json.loads(scenario.run()["vh_message"])
    # Assert
    assert report["cause"] == "invalid_answer"
    assert not scenario.http.requests
