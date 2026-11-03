"""The acquired facts, byte stream and last-part integrity govern the terminal outcome."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from core.drive_import import acquire
from core.drive_import.clock import StopLine
from core.drive_import.failure import ImportFailed
from core.drive_import.transfer import Part, TransferPlan, transfer
from core.protocol import acquire_messages, messages
from core.protocol.acquire_dispatch import accept_dispatch
from core.protocol.messages import ExecutionIdentity
from core.protocol.outbox import seal_message
from core.protocol.registry import MessageKind
from tests.acquire.support import EXECUTION, PART_SIZE, Scenario, wire
from tests.contract.v2_support import HUB_KEY, signed_carrier
from tests.drive_import.fakes import DriveFile, Fail, FakeResponse, Late, Put, Serve, google_error, json_answer

ERROR_RELATION = [
    ("file_removed", "source_unavailable", False, "downloading"),
    ("link_not_shared", "source_unavailable", False, "downloading"),
    ("provider_quota", "source_unavailable", True, "downloading"),
    ("provider_refused", "source_unavailable", False, "downloading"),
    ("not_configured", "configuration_error", False, None),
    ("size_exceeded", "input_limits_exceeded", False, "downloading"),
    ("size_short", "source_integrity_mismatch", False, "downloading"),
    ("integrity_failed", "source_integrity_mismatch", False, "finalizing"),
    ("source_changed_at_provider", "source_integrity_mismatch", False, "downloading"),
    ("deadline", "deadline_exceeded", False, None),
    ("deadline", "deadline_exceeded", False, "downloading"),
    ("deadline", "deadline_exceeded", False, "uploading"),
    ("deadline", "deadline_exceeded", False, "finalizing"),
    ("download_failed", "source_unavailable", True, "downloading"),
    ("upload_failed", "output_write_failed", True, "uploading"),
    ("input_refused", "input_invalid", False, None),
]


@pytest.mark.parametrize("code,category,retryable,stage", ERROR_RELATION)
def test_acquire_failure_when_mapped_should_use_the_exact_contract_relation(
    code: str, category: str, retryable: bool, stage: str | None,
) -> None:
    # Arrange
    scenario = Scenario()
    dispatch = accept_dispatch(signed_carrier(wire(scenario.request), HUB_KEY), settings=scenario.runtime.settings, now=scenario.clock.now())
    identity = ExecutionIdentity.of(dispatch.identity, execution_id=EXECUTION, execution_fence=2)
    # Act
    message = acquire_messages.result_failed(
        protocol_version=dispatch.protocol_version, message_id=scenario.runtime.new_uuid(), sent_at=scenario.clock.now(),
        key_id=scenario.runtime.settings.worker_to_hub_key_id, identity=identity, code=code, stage=stage,
    )
    sealed = seal_message(message, acquire_messages.sender_context(
        MessageKind.SOURCE_ACQUIRE_RESULT_FAILED, dispatch, identity, settings=scenario.runtime.settings, now=scenario.clock.now(),
    ), settings=scenario.runtime.settings)
    # Assert
    error = json.loads(sealed.body)["error"]
    assert (error["class"], error["code"], error["retryable"], error["stage"]) == (category, code, retryable, stage)
    assert set(error) == {"class", "code", "retryable", "stage", "detail"}


@pytest.mark.parametrize("status,reason,code", [
    (404, None, "file_removed"), (403, "insufficientFilePermissions", "link_not_shared"),
    (429, None, "provider_quota"), (403, "cannotDownloadAbusiveFile", "provider_refused"),
    (400, "keyInvalid", "not_configured"), (302, None, "download_failed"),
])
def test_acquire_when_google_refuses_should_map_the_failure_without_uploading_a_part(
    status: int, reason: str | None, code: str,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [google_error(status, reason)]
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == code
    assert scenario.http.part_puts == []


@pytest.mark.parametrize("field", ["size", "md5Checksum", "sha1Checksum", "sha256Checksum", "headRevisionId"])
@pytest.mark.parametrize("change", ["missing", "different"])
def test_acquire_when_recorded_metadata_has_changed_should_fail_before_uploading_any_part(field: str, change: str) -> None:
    # Arrange
    scenario = Scenario()
    if change == "missing":
        scenario.http.metadata_document.pop(field)
    else:
        scenario.http.metadata_document[field] = "other-fact"
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == "source_changed_at_provider"
    assert scenario.http.part_puts == []
    assert len(scenario.http.media_gets) == 0


@pytest.mark.parametrize("actual,code", [(0, "size_short"), (36, "size_short"), (38, "size_exceeded")])
def test_acquire_when_stream_length_differs_should_withhold_the_last_part(actual: int, code: str) -> None:
    # Arrange: metadata still describes the accepted snapshot.
    scenario = Scenario()
    scenario.http.file = DriveFile(actual)
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == code
    assert scenario.http.part_puts == []


def test_acquire_when_stream_integrity_fails_should_withhold_the_last_part_after_earlier_parts() -> None:
    # Arrange
    scenario = Scenario(PART_SIZE + 37)
    scenario.http.file = DriveFile(scenario.file.size, seed=42)
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == "integrity_failed"
    assert output["error"]["stage"] == "finalizing"
    assert len(scenario.http.part_puts) == 1
    assert scenario.http.part_puts[0].length == PART_SIZE
    assert "partNumber=1" in scenario.http.part_puts[0].url


@pytest.mark.parametrize("mode", ["sha256", "md5", "none"])
def test_acquire_when_hashes_are_optional_should_report_the_strongest_recorded_checksum(mode: str) -> None:
    # Arrange
    scenario = Scenario()
    expected = scenario.request["external"]["expected"]
    if mode != "sha256":
        expected["sha256"] = None
    if mode == "none":
        expected["md5"] = None
    # Act
    result = json.loads(scenario.run()["vh_message"])
    # Assert
    assert result["checksum_verified"] == mode


@pytest.mark.parametrize("etag", ["MiXeD-Case+Opaque/==", "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "A" * 128])
def test_acquire_when_storage_returns_an_opaque_etag_should_preserve_its_exact_case_and_characters(etag: str) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[scenario.urls()[0]] = [Put(headers={"ETag": '"' + etag + '"'})]
    # Act
    result = json.loads(scenario.run()["vh_message"])
    # Assert
    assert result["parts"] == [{"n": 1, "etag": etag}]


@pytest.mark.parametrize("etag", ["https://secret.invalid/key", "has space", "", "A" * 129, 'a"b'])
def test_acquire_when_an_etag_cannot_be_carried_should_fail_without_echoing_it(etag: str) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[scenario.urls()[0]] = [Put(headers={"ETag": etag})]
    # Act
    output = scenario.run()
    # Assert
    assert json.loads(output["vh_message"])["error"]["code"] == "upload_failed"
    if etag:
        assert etag not in output["vh_message"]


@pytest.mark.parametrize("phase,stage", [("metadata", "downloading"), ("read", "downloading"), ("put", "uploading"), ("retry", "uploading")])
def test_acquire_when_the_wall_budget_ends_should_stop_reads_writes_and_retries(phase: str, stage: str) -> None:
    # Arrange
    scenario = Scenario()
    if phase == "metadata":
        scenario.http.metadata = [Late(json_answer(200, scenario.file.metadata()), 60)]
    elif phase == "read":
        scenario.http.media = [Serve(break_after=1, seconds_before_break=60)]
    elif phase == "put":
        scenario.http.puts[scenario.urls()[0]] = [Put(seconds_after=60)]
    else:
        scenario.http.puts[scenario.urls()[0]] = [Put(status=503, seconds_after=59.8)]
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert (output["error"]["code"], output["error"]["stage"]) == ("deadline", stage)
    assert len(scenario.http.part_puts) <= 1
    assert not scenario.clock.slept
    if phase != "retry":
        assert not scenario.http.result_puts and not scenario.hub.callbacks


def test_acquire_when_claim_time_spends_the_budget_should_report_deadline_without_source_traffic() -> None:
    # Arrange
    scenario = Scenario()

    def late(grant: dict[str, Any]) -> Any:
        scenario.clock.advance(60)
        return scenario.hub.signed_grant(grant)

    scenario.hub.claim_actions = [late]
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == "deadline" and output["error"]["stage"] is None
    assert scenario.http.requests == [] and not scenario.hub.callbacks


def test_acquire_when_terminal_construction_runs_out_of_time_should_seal_deadline_instead_of_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    scenario = Scenario()
    seal = messages.seal

    def slow(body: bytes, key: bytes) -> str:
        if json.loads(body)["message_kind"] == "source.acquire.result.completed":
            scenario.clock.advance(60)
        return seal(body, key)

    monkeypatch.setattr(messages, "seal", slow)
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["error"]["code"] == "deadline"
    assert output["error"]["stage"] == "finalizing"
    assert not scenario.http.result_puts and not scenario.hub.callbacks


def test_acquire_when_a_download_breaks_should_resume_the_unwritten_part_only() -> None:
    # Arrange
    scenario = Scenario(PART_SIZE + 37)
    scenario.http.media = [Serve(break_after=PART_SIZE + 10), Serve()]
    # Act
    output = json.loads(scenario.run()["vh_message"])
    # Assert
    assert output["message_kind"] == "source.acquire.result.completed"
    assert len(scenario.http.part_puts) == 2
    assert scenario.http.media_gets[1].headers["Range"] == f"bytes={PART_SIZE}-"
    assert scenario.clock.slept == [0.5]


@pytest.mark.parametrize("member,value", [("expected_size", 37.0), ("part_size", float(PART_SIZE)), ("parts", (Part(1.0, "secret"),))])
def test_transfer_when_an_internal_plan_has_integral_floats_should_refuse_without_coercion(member: str, value: Any) -> None:
    # Arrange
    scenario = Scenario()
    plan = TransferPlan("TestFile_01", None, 37, None, None, None, None, PART_SIZE,
                        (Part(1, scenario.urls()[0]),), StopLine(scenario.clock, scenario.clock.monotonic() + 60))
    plan = replace(plan, **{member: value})
    # Act / Assert
    with pytest.raises(ImportFailed) as caught:
        transfer(plan, scenario.runtime.settings.google_api_key, scenario.http)
    assert caught.value.code.value == "input_refused"
    assert scenario.http.requests == []
