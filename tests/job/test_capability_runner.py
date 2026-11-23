"""JobRunner at protocol 2.2.0 (README 15.4): the capability path, with the stages that need ffprobe or ffmpeg replaced
by tests/job/test_runner.py's stand-ins, a fake Hub that judges every capability request and a fake store behind the
shared session. No storage credential, no boto3 client, no network."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from core import profile as profile_module
from core.bounded_process import Deadline, run_bounded
from core.failure import Failure, WorkerFailure
from core.job import runner as runner_module
from core.job import storage as storage_module
from core.job.heartbeat import HEARTBEAT_INTERVAL_S, THREAD_NAME
from core.job.runner import JobRunner
from core.profile import GIB, PILOT_PROFILE, profile_digest
from core.protocol import pipeline
from core.protocol.outbox import SealedMessage
from core.protocol.transport import HttpAnswer, TransportFailed
from tests.contract.v2_support import FakeClock, uuid_source
from tests.job.capability_fakes import (
    FakeHub,
    FakeStore,
    capability_claimed,
    capability_settings,
    hub_error,
    wait_until,
)
from tests.job.conftest import (
    COMPLETED_KIND,
    FAILED_KIND,
    MANIFEST_NAME,
    accepted_by_the_hub,
    hub_storage_context,
    job_settings,
    runtime_of,
)
from tests.job.test_runner import FFMPEG_VERSION, SOURCE_OBJECT, StandInStages, _facts

RESULT_NAME = "result.json"
STAGES = ["probing", "downloading", "transcoding", "finalizing", "uploading"]
# A run whose stored result is not granted still reports within this many real seconds.
STOPPED_WITHIN_S = 5.0
# A scratch disk with room for every rendition's reservation, so that they are encoded in parallel.
ROOMY_DISK_BYTES = 100 * GIB
ENCODER_STOPPED = Failure("encoder_failed", "stand_in_encode_stopped", True, "the stand-in encode was stopped")


class _Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.scratch_root = tmp_path / "scratch"
        self.scratch_root.mkdir()
        monkeypatch.setattr(profile_module, "SUPPORTED_PROFILES", (PILOT_PROFILE,))
        named = {"id": PILOT_PROFILE.profile_id, "version": PILOT_PROFILE.version, "sha256": profile_digest(PILOT_PROFILE)}
        self.claimed = capability_claimed(request_changes={"/source/size_bytes": len(SOURCE_OBJECT), "/profile": named})
        self.clock = FakeClock()
        self.hub = FakeHub(self.claimed, self.clock)
        self.store = FakeStore(self.hub, SOURCE_OBJECT)
        self.runtime = dataclasses.replace(runtime_of(self.hub, self.clock, capability_settings()), new_uuid=uuid_source())
        self.stages = StandInStages(self.scratch_root, _facts())
        self.stages.install(monkeypatch)
        self.client_factory_calls: list[str] = []
        self.runner = JobRunner(
            job_settings(self.scratch_root),
            source_client_factory=lambda profile: self._refused_client("source"),
            output_client_factory=lambda profile: self._refused_client("output"),
            ffmpeg_version=lambda deadline: FFMPEG_VERSION,
            url_session_factory=lambda: self.store,
            upload_session_factory=lambda: self.store,
        )

    def _refused_client(self, side: str) -> Any:
        self.client_factory_calls.append(side)
        raise AssertionError("a 2.2.0 execution makes no storage client")

    def run(self) -> SealedMessage:
        return self.runner.run(self.claimed, self.runtime)

    def result(self) -> dict[str, Any]:
        return accepted_by_the_hub(self.run(), self.claimed, now=self.clock())


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def _heartbeat_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == THREAD_NAME]


def test_run_when_the_dispatch_is_two_two_zero_should_complete_through_urls_with_no_storage_client(harness):
    # Act
    result = harness.result()
    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.client_factory_calls == []
    assert harness.store.stored_order[-2:] == [MANIFEST_NAME, RESULT_NAME]
    manifest = harness.store.objects[MANIFEST_NAME]
    assert pipeline.validate(manifest, hub_storage_context(harness.claimed)).accepted
    stored = json.loads(harness.store.objects[RESULT_NAME])
    assert json.loads(stored["vh_message"])["message_kind"] == COMPLETED_KIND


def test_run_when_the_dispatch_is_two_two_zero_should_report_the_stages_in_order_and_the_duration_after_probing(harness):
    # Act
    harness.run()
    # Assert
    requests = harness.hub.requests
    first_of_each = []
    for request in requests:
        if not first_of_each or first_of_each[-1] != request["stage"]:
            first_of_each.append(request["stage"])
    assert first_of_each == STAGES
    probing = [request for request in requests if request["stage"] == "probing"]
    assert probing[0]["source"] is True and probing[0]["seq"] == 1
    assert all(request["media_duration_ms"] is None for request in probing)
    later = [request for request in requests if request["stage"] != "probing"]
    assert {request["media_duration_ms"] for request in later} == {_facts().duration_ms}
    assert [request["seq"] for request in requests] == list(range(1, len(requests) + 1))


def test_run_when_the_dispatch_is_two_two_zero_should_read_the_source_through_its_url_conditionally(harness):
    # Act
    harness.run()
    # Assert
    assert harness.store.gets
    for get in harness.store.gets:
        assert get["url"] in harness.hub.issued
        assert get["headers"]["If-Match"] == '"test-etag-0001"'


def test_run_when_the_hub_refuses_a_stage_should_fail_authority_refused_at_that_stage_and_store_no_result(harness):
    # Arrange: probing's request is granted, downloading's refused.
    harness.hub.script.extend(["grant", hub_error(403, "stage_budget_exceeded", retryable=False)])
    # Act
    result = harness.result()
    # Assert
    error = result["error"]
    assert result["message_kind"] == FAILED_KIND
    assert (error["class"], error["code"], error["retryable"], error["stage"]) == (
        "authority_refused", "stage_budget_exceeded", False, "downloading")
    assert RESULT_NAME not in harness.store.objects
    assert harness.store.closed
    assert not _heartbeat_threads()
    assert list(harness.scratch_root.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="bounded processes run under POSIX process groups")
def test_run_when_the_lease_is_lost_during_an_encode_should_stop_it_and_fail_with_the_hubs_code_at_its_stage(
    harness, monkeypatch, caplog
):
    # Arrange: the stand-in encode runs a long process; while it runs, 120 s pass on the Hub's clock and the
    # heartbeat that follows is refused, which stops the process through the stop check. The renditions are
    # encoded in parallel, whatever this host's disk holds: a lost authority's failure must win there too.
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: ROOMY_DISK_BYTES)
    stand_in = harness.stages.encode_rendition
    keys_seen: list[bytes | None] = []

    def encode_that_runs(*args: Any, **named: Any) -> Any:
        keys_seen.append(named["media_key"])
        harness.hub.script.append(hub_error(403, "lease_expired", retryable=False))
        threading.Timer(0.1, harness.clock.advance, args=(HEARTBEAT_INTERVAL_S,)).start()
        ended = run_bounded(["sleep", "30"], deadline=Deadline.after_ms(60_000), max_stdout_bytes=64,
                            max_stderr_bytes=64, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, cwd="/")
        if ended.stopped_by_monitor:
            raise WorkerFailure(ENCODER_STOPPED)
        return stand_in(*args, **named)

    monkeypatch.setattr(runner_module, "encode_rendition", encode_that_runs)
    caplog.set_level(logging.INFO, logger=runner_module.__name__)
    # Act
    result = harness.result()
    # Assert
    assert any("renditions parallel:" in record.getMessage() for record in caplog.records)
    error = result["error"]
    assert (error["class"], error["code"], error["retryable"], error["stage"]) == (
        "authority_refused", "lease_expired", False, "transcoding")
    assert harness.claimed.grant.media_key is None
    assert not _heartbeat_threads()
    assert RESULT_NAME not in harness.store.objects


def test_run_when_the_preflight_refuses_the_source_should_store_the_failed_result_through_a_url(harness):
    # Arrange: a source longer than the dispatch's limit (four hours) is not admitted.
    harness.stages.facts = _facts(duration_ms=20_000_000)
    # Act
    result = harness.result()
    # Assert
    assert result["message_kind"] == FAILED_KIND and result["error"]["stage"] == "probing"
    stored = json.loads(harness.store.objects[RESULT_NAME])
    assert json.loads(stored["vh_message"])["message_id"] == result["message_id"]
    assert harness.client_factory_calls == []


# --- the stored result never holds back the report (contract 15.1 rule 9) -------------------------------------


def _result_requests(harness: _Harness) -> list[dict[str, Any]]:
    return [request for request in harness.hub.requests
            if [o["path"] for o in (request["put"] or {}).get("objects", ())] == [RESULT_NAME]]


@pytest.mark.parametrize("answer", [HttpAnswer(503, b"", None), TransportFailed("NoAnswer")],
                         ids=["refused-for-now", "no-answer"])
def test_run_when_the_results_url_is_not_granted_should_return_the_result_after_one_request_and_no_wait(
    harness, answer
):
    # Arrange
    harness.hub.result_answer = answer
    started = time.monotonic()
    # Act
    sealed = harness.run()
    # Assert
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert result["message_kind"] == COMPLETED_KIND
    assert len(_result_requests(harness)) == 1
    assert harness.clock.waits == []
    assert RESULT_NAME not in harness.store.objects
    assert time.monotonic() - started < STOPPED_WITHIN_S


@pytest.mark.parametrize("answer", [HttpAnswer(503, b"", None), TransportFailed("NoAnswer")],
                         ids=["refused-for-now", "no-answer"])
def test_run_when_it_fails_at_transcoding_and_the_results_url_is_not_granted_should_report_at_once(
    harness, monkeypatch, answer
):
    # Arrange
    harness.hub.result_answer = answer

    def failing_encode(*args: Any, **named: Any) -> Any:
        raise WorkerFailure(ENCODER_STOPPED)

    monkeypatch.setattr(runner_module, "encode_rendition", failing_encode)
    # Act
    sealed = harness.run()
    # Assert
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert (result["error"]["code"], result["error"]["stage"]) == (ENCODER_STOPPED.code, "transcoding")
    assert len(_result_requests(harness)) == 1
    assert harness.clock.waits == []
    assert not _heartbeat_threads()


def test_run_when_the_result_is_stored_should_store_the_bytes_it_returns(harness):
    # Act
    sealed = harness.run()
    # Assert
    stored = json.loads(harness.store.objects[RESULT_NAME])
    assert stored == {"vh_message": sealed.body.decode("ascii"), "vh_signature": sealed.signature}


# --- a lost authority stops the download ---------------------------------------------------------------------


def test_run_when_the_authority_is_lost_during_the_download_should_read_no_further_chunk(harness, monkeypatch):
    # Arrange: the source is downloaded in chunks of 1 KiB; at the second chunk 120 s pass and the heartbeat that
    # follows is refused for good.
    monkeypatch.setattr(storage_module, "DOWNLOAD_CHUNK_BYTES", 1024)
    gets_at_loss: list[int] = []

    def lose_the_authority() -> None:
        harness.hub.script.append(hub_error(403, "capability_refused", retryable=False))
        asked = len(harness.hub.requests)
        harness.clock.advance(HEARTBEAT_INTERVAL_S)
        assert wait_until(lambda: len(harness.hub.requests) > asked)
        gets_at_loss.append(len(harness.store.gets))

    harness.store.get_answers.extend([lambda: None, lambda: None, lose_the_authority])
    # Act
    result = harness.result()
    # Assert
    error = result["error"]
    assert (error["class"], error["code"], error["retryable"], error["stage"]) == (
        "authority_refused", "capability_refused", False, "downloading")
    assert len(harness.store.gets) - gets_at_loss[0] <= 1
    assert len(harness.store.gets) < len(SOURCE_OBJECT) // 1024


def test_run_when_the_result_is_stored_should_have_ended_the_clients_work_first(harness, monkeypatch):
    # Arrange: what the client says of its work at the moment the result is stored.
    ended_at_store: list[bool] = []
    store = runner_module.upload_result_by_url

    def recording(capabilities, session, **named):
        ended_at_store.append(capabilities.ended)
        return store(capabilities, session, **named)

    monkeypatch.setattr(runner_module, "upload_result_by_url", recording)
    # Act
    harness.run()
    # Assert: no heartbeat could follow the result's request.
    assert ended_at_store == [True]
    assert harness.hub.requests[-1] is _result_requests(harness)[-1]
