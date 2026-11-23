"""JobRunner's transcoding stage in parallel: each rendition on a thread of its own, within its own reservation.

The harness and the stand-in stages are tests/job/test_runner.py's; each test here wraps the stand-in encode to hold
the renditions at a point where they must all be running, or to fail one of them. A stand-in that must be stopped
runs a real bounded process (`sleep`), so that what stops it is core/bounded_process.py's own stop check, as for a
real ffmpeg. The scratch disk reads as roomy unless a test says otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from core.bounded_process import Deadline, run_bounded, stopped_when
from core.encode import OutputAllowance, RenditionOutput
from core.encode_command import EncodeSource
from core.failure import Failure, WorkerFailure
from core.job import reservation as reservation_module
from core.job import runner as runner_module
from core.job.progress import ProgressReporter
from core.job.reservation import rendition_reservation
from core.profile import GIB, PILOT_PROFILE
from core.package import reserve_package
from core.renditions import RenditionPlan, plan_renditions
from tests.job.conftest import COMPLETED_KIND, MANIFEST_NAME
from tests.job.test_runner import CANARY, MEDIA_KEY, Harness, _error_of, _facts, _video

PLANNED = ["720p", "540p", "360p"]
# Five renditions of a 1080p source, more than are encoded at once.
FIVE_PLANNED = ["1080p", "720p", "540p", "360p", "240p"]
# What the harness's source admits: 2 seconds.
ADMITTED_MS = 2000
ROOMY_DISK_BYTES = 100 * GIB
# Long enough for every thread of a test to reach where it waits, short enough to fail a test that never gets there.
WAIT_S = 10.0
FIRST_FAILURE = Failure("encoder_failed", "stand_in_first_failure", True, "the stand-in rendition failed first")
STOPPED = Failure("encoder_failed", "stand_in_encode_stopped", True, "the stand-in encode was stopped")
FINAL_FAILURE = Failure("input_limits_exceeded", "decoded_frame_too_large", False, "the stand-in frame was too large")
SCRATCH_FULL = Failure("resource_exhausted", "scratch_full", True, "the scratch disk is full")
# A dispatch's most artifacts where a test sets it: plenty for three stand-in renditions and the package.
ARTIFACT_LIMIT = 100
posix_only = pytest.mark.skipif(os.name != "posix", reason="bounded processes run under POSIX process groups")


def _run_until_stopped() -> bool:
    """A real bounded process as long-lived as an encode: whether a stop check stopped it."""
    ended = run_bounded(
        ["sleep", "30"],
        deadline=Deadline.after_ms(60_000),
        max_stdout_bytes=64,
        max_stderr_bytes=64,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        cwd="/",
    )
    return ended.stopped_by_monitor


class Encodes:
    """Wraps the harness's stand-in encode: `act[name]` runs in place of it for a rendition that has one, and
    each call notes on which thread it ran and whether the grant still held the media key when it ended."""

    def __init__(self, harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        self.harness = harness
        self.stand_in = harness.stages.encode_rendition
        self.act: dict[str, Callable[[], None]] = {}
        self.lock = threading.Lock()
        self.threads: dict[str, str] = {}
        self.ended: list[str] = []
        self.key_held_at_end: dict[str, bool] = {}
        self.has_ended: defaultdict[str, threading.Event] = defaultdict(threading.Event)
        # How many encodes ran at once, at most.
        self.running = 0
        self.most_running = 0
        monkeypatch.setattr(runner_module, "encode_rendition", self)

    def __call__(self, source: EncodeSource, plan: RenditionPlan, **named: Any) -> RenditionOutput:
        with self.lock:
            self.threads[plan.name] = threading.current_thread().name
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        try:
            self.act.get(plan.name, lambda: None)()
            return self.stand_in(source, plan, **named)
        finally:
            with self.lock:
                self.running -= 1
                self.ended.append(plan.name)
                self.key_held_at_end[plan.name] = self.harness.claimed.grant.media_key is not None
            self.has_ended[plan.name].set()


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    made = Harness(tmp_path, monkeypatch, parallel=True)
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: ROOMY_DISK_BYTES)
    return made


@pytest.fixture
def encodes(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> Encodes:
    return Encodes(harness, monkeypatch)


def _all_running(encodes: Encodes, barrier: threading.Barrier) -> None:
    for name in PLANNED:
        encodes.act[name] = barrier.wait


def _stopped_sibling(stops: dict[str, bool]) -> Callable[[str], Callable[[], None]]:
    def sibling(name: str) -> Callable[[], None]:
        def act() -> None:
            stops[name] = _run_until_stopped()
            raise WorkerFailure(STOPPED)

        return act

    return sibling


# --- every rendition, all at once ------------------------------------------------------------------------


def test_run_when_the_reservations_fit_should_encode_the_renditions_at_the_same_time_and_complete(harness, encodes):
    # Arrange: each encode waits until all three have started, which only encodes that overlap ever do
    _all_running(encodes, threading.Barrier(len(PLANNED), timeout=WAIT_S))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert sorted(encodes.ended) == sorted(PLANNED)
    assert len(set(encodes.threads.values())) == len(PLANNED)
    assert threading.current_thread().name not in encodes.threads.values()


def test_run_when_the_renditions_end_in_another_order_should_hand_on_their_outputs_in_planned_order(harness, encodes):
    # Arrange: the smallest ends first and the largest last, each waiting for the next smaller one to have ended
    def after(earlier: str) -> Callable[[], None]:
        return lambda: None if encodes.has_ended[earlier].wait(WAIT_S) else pytest.fail(f"{earlier} never ended")

    encodes.act = {"540p": after("360p"), "720p": after("540p")}

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert encodes.ended == ["360p", "540p", "720p"]
    manifest = json.loads(harness.uploaded[MANIFEST_NAME])
    assert [rendition["name"] for rendition in manifest["renditions"]] == PLANNED
    assert harness.stages.thumbnail_references[0].name == "720p"


def test_run_when_the_renditions_run_in_parallel_should_give_each_its_own_reservation(harness, encodes):
    # Act
    harness.run()

    # Assert: each its own room, not what the earlier ones left of the job's, and each the media key
    given = {call.name: call.allowance for call in harness.stages.encodes}
    assert given == {
        name: rendition_reservation(_plan_of(harness, name), PILOT_PROFILE, ADMITTED_MS) for name in PLANNED
    }
    assert all(call.media_key == MEDIA_KEY for call in harness.stages.encodes)


def _plan_of(harness: Harness, name: str) -> RenditionPlan:
    facts = harness.stages.facts
    plans = plan_renditions(facts.video_streams[0], has_audio=True, requested=PLANNED, profile=PILOT_PROFILE)
    return next(plan for plan in plans if plan.name == name)


def test_run_when_the_renditions_run_in_parallel_should_report_each_one_done_once_from_the_jobs_thread(
    harness, encodes, monkeypatch
):
    # Arrange
    reported_on: list[str] = []
    rendition_done = ProgressReporter.rendition_done

    def noted(reporter: ProgressReporter) -> None:
        reported_on.append(threading.current_thread().name)
        rendition_done(reporter)

    monkeypatch.setattr(ProgressReporter, "rendition_done", noted)

    # Act
    harness.run()

    # Assert
    assert reported_on == [threading.current_thread().name] * len(PLANNED)


def test_run_when_the_renditions_run_in_parallel_should_drop_the_key_only_after_every_one_ended(harness, encodes):
    # Arrange: the largest ends well after the others
    encodes.act["720p"] = lambda: time.sleep(0.3)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert encodes.key_held_at_end == {name: True for name in PLANNED}
    assert harness.claimed.grant.media_key is None


def test_run_when_it_chooses_parallel_should_say_so_in_one_line_with_numbers_only(harness, encodes, caplog):
    # Arrange
    caplog.set_level(logging.INFO, logger=runner_module.__name__)

    # Act
    harness.run()

    # Assert
    lines = [record.getMessage() for record in caplog.records if "renditions" in record.getMessage()]
    assert len(lines) == 1 and lines[0].endswith("renditions parallel: 3 renditions")
    assert str(harness.scratch_root) not in caplog.text


# --- one at a time when parallel is not safe -------------------------------------------------------------


def _assert_one_at_a_time(harness: Harness) -> None:
    first, second, third = (call.allowance for call in harness.stages.encodes)
    assert [call.name for call in harness.stages.encodes] == PLANNED
    assert second.max_bytes == first.max_bytes - harness.stages.outputs[0].total_bytes
    assert third.max_bytes == second.max_bytes - harness.stages.outputs[1].total_bytes


def test_run_when_the_scratch_disks_free_space_cannot_be_read_should_encode_one_at_a_time(
    tmp_path, monkeypatch, caplog
):
    # Arrange: the operating system refuses to tell the free space
    harness = Harness(tmp_path, monkeypatch, parallel=True)
    caplog.set_level(logging.INFO, logger=runner_module.__name__)

    def refused(path: Any) -> Any:
        raise PermissionError(13, "refused", str(path))

    monkeypatch.setattr(reservation_module.os, "statvfs", refused)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    _assert_one_at_a_time(harness)
    assert "renditions sequential: the scratch disk's free space is unknown" in caplog.text


def test_run_when_the_scratch_disk_cannot_hold_every_reservation_should_encode_one_at_a_time(harness, monkeypatch):
    # Arrange: a disk with a few bytes free beside the source
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: 4096)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    _assert_one_at_a_time(harness)


def test_run_when_the_job_has_one_rendition_should_encode_it_within_the_whole_allowance(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, parallel=True, request_changes={"/renditions": ["720p"]})
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: ROOMY_DISK_BYTES)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    (only,) = harness.stages.encodes
    assert only.allowance != rendition_reservation(_plan_of(harness, "720p"), PILOT_PROFILE, ADMITTED_MS)
    assert only.allowance.max_artifacts == harness.claimed.dispatch.limits.max_artifact_count - 2


def test_run_when_the_reservations_exceed_the_jobs_output_limit_should_encode_one_at_a_time(tmp_path, monkeypatch):
    # Arrange: room for the package and every stand-in rendition, but not for three reservations of 1 MiB and more
    generous = OutputAllowance(max_bytes=10**9, max_artifacts=ARTIFACT_LIMIT)
    package_bytes = generous.max_bytes - reserve_package(generous, PILOT_PROFILE).max_bytes
    harness = Harness(
        tmp_path,
        monkeypatch,
        parallel=True,
        request_changes={
            "/limits/max_artifact_count": generous.max_artifacts,
            "/limits/max_output_bytes": package_bytes + 2 * 1024 * 1024,
        },
    )
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: ROOMY_DISK_BYTES)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    _assert_one_at_a_time(harness)


# --- the first failure wins ------------------------------------------------------------------------------


@posix_only
def test_run_when_a_rendition_fails_should_stop_its_siblings_processes(harness, encodes):
    # Arrange: the two larger renditions run a long process; the smallest fails once all three have started
    started = threading.Barrier(len(PLANNED), timeout=WAIT_S)
    stops: dict[str, bool] = {}
    sibling = _stopped_sibling(stops)

    def fails() -> None:
        started.wait()
        raise WorkerFailure(FIRST_FAILURE)

    encodes.act = {
        "720p": lambda: (started.wait(), sibling("720p")()),
        "540p": lambda: (started.wait(), sibling("540p")()),
        "360p": fails,
    }

    # Act
    started_at = time.monotonic()
    harness.result()

    # Assert: both stopped by the stop check, long before their 30 seconds
    assert stops == {"720p": True, "540p": True}
    assert time.monotonic() - started_at < WAIT_S


@posix_only
def test_run_when_a_rendition_fails_should_end_with_its_failure_never_a_siblings_stop(harness, encodes):
    # Arrange: the failing rendition is the last of the plan; the stopped ones come before it in the plan
    started = threading.Barrier(len(PLANNED), timeout=WAIT_S)
    sibling = _stopped_sibling({})

    def fails() -> None:
        started.wait()
        raise WorkerFailure(FIRST_FAILURE)

    encodes.act = {
        "720p": lambda: (started.wait(), sibling("720p")()),
        "540p": lambda: (started.wait(), sibling("540p")()),
        "360p": fails,
    }

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == (
        FIRST_FAILURE.error_class, FIRST_FAILURE.code, FIRST_FAILURE.retryable, "transcoding"
    )
    assert sorted(encodes.ended) == sorted(PLANNED)
    assert encodes.key_held_at_end == {name: True for name in PLANNED}
    assert harness.claimed.grant.media_key is None
    assert harness.left_in_scratch == []


@posix_only
def test_run_when_a_rendition_meets_a_bug_should_end_on_it_once_its_siblings_ended(harness, encodes, caplog):
    # Arrange: the middle rendition raises a bug; the other two run until they are stopped
    caplog.set_level(logging.DEBUG)
    started = threading.Barrier(len(PLANNED), timeout=WAIT_S)
    stops: dict[str, bool] = {}
    sibling = _stopped_sibling(stops)

    def bug() -> None:
        started.wait()
        raise RuntimeError(CANARY)

    encodes.act = {
        "720p": lambda: (started.wait(), sibling("720p")()),
        "540p": bug,
        "360p": lambda: (started.wait(), sibling("360p")()),
    }

    # Act
    result = harness.result()

    # Assert: the bug, as a bug, and the key held until both siblings had ended
    assert _error_of(result) == ("internal_error", "unexpected_error", True, "transcoding")
    assert result["error"]["detail"] == "RuntimeError"
    assert stops == {"720p": True, "360p": True}
    assert encodes.key_held_at_end == {name: True for name in PLANNED}
    assert CANARY not in caplog.text


@posix_only
def test_run_when_a_sibling_meets_a_bug_after_a_final_failure_should_end_on_the_final_failure_and_log_the_bug(
    harness, encodes, caplog
):
    # Arrange: a final failure first; a sibling, stopped by it, then meets a bug
    caplog.set_level(logging.DEBUG)
    started = threading.Barrier(len(PLANNED), timeout=WAIT_S)

    def fails() -> None:
        started.wait()
        raise WorkerFailure(FINAL_FAILURE)

    def bug_once_stopped() -> None:
        started.wait()
        _run_until_stopped()
        raise RuntimeError(CANARY)

    encodes.act = {"720p": fails, "540p": bug_once_stopped, "360p": started.wait}

    # Act
    result = harness.result()

    # Assert: not turned into a retryable bug; the bug logged once, by its class and stack, never its text
    assert _error_of(result) == (
        FINAL_FAILURE.error_class, FINAL_FAILURE.code, FINAL_FAILURE.retryable, "transcoding"
    )
    bug_lines = [record for record in caplog.records if "a rendition ended on an unexpected" in record.getMessage()]
    assert len(bug_lines) == 1
    assert "RuntimeError" in bug_lines[0].getMessage()
    assert bug_lines[0].exc_info is None
    assert CANARY not in caplog.text


@posix_only
def test_run_when_a_sibling_finds_the_scratch_disk_full_after_another_failure_should_end_on_the_full_disk(
    harness, encodes
):
    # Arrange: a final failure first; a sibling then finds the disk full
    started = threading.Barrier(len(PLANNED), timeout=WAIT_S)

    def fails() -> None:
        started.wait()
        raise WorkerFailure(FINAL_FAILURE)

    def full_once_stopped() -> None:
        started.wait()
        _run_until_stopped()
        raise WorkerFailure(SCRATCH_FULL)

    encodes.act = {"720p": fails, "540p": full_once_stopped, "360p": started.wait}

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("resource_exhausted", "scratch_full", True, "transcoding")


# --- at most three at once -------------------------------------------------------------------------------


def _five_renditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    full_hd = _video(width=1920, height=1080, display_width=1920, display_height=1080)
    made = Harness(
        tmp_path, monkeypatch, parallel=True, facts=_facts(videos=(full_hd,)),
        request_changes={"/renditions": FIVE_PLANNED},
    )
    monkeypatch.setattr(runner_module, "scratch_free_bytes", lambda directory: ROOMY_DISK_BYTES)
    return made


def test_run_when_five_renditions_are_planned_should_never_encode_more_than_three_at_once(tmp_path, monkeypatch):
    # Arrange: each encode lasts long enough for the others to start if they could
    harness = _five_renditions(tmp_path, monkeypatch)
    encodes = Encodes(harness, monkeypatch)
    for name in FIVE_PLANNED:
        encodes.act[name] = lambda: time.sleep(0.2)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert sorted(encodes.ended) == sorted(FIVE_PLANNED)
    assert encodes.most_running == runner_module.MAX_PARALLEL_RENDITIONS == 3
    manifest = json.loads(harness.uploaded[MANIFEST_NAME])
    assert [rendition["name"] for rendition in manifest["renditions"]] == FIVE_PLANNED


@posix_only
def test_run_when_a_sibling_fails_while_a_rendition_waits_for_a_thread_should_never_start_it(tmp_path, monkeypatch):
    # Arrange: the first three take the threads; one fails while the other two run until stopped
    harness = _five_renditions(tmp_path, monkeypatch)
    encodes = Encodes(harness, monkeypatch)
    started = threading.Barrier(3, timeout=WAIT_S)
    sibling = _stopped_sibling({})

    def fails() -> None:
        started.wait()
        raise WorkerFailure(FIRST_FAILURE)

    encodes.act = {
        "1080p": lambda: (started.wait(), sibling("1080p")()),
        "720p": lambda: (started.wait(), sibling("720p")()),
        "540p": fails,
    }

    # Act
    result = harness.result()

    # Assert: the two that waited never ran
    assert _error_of(result) == (
        FIRST_FAILURE.error_class, FIRST_FAILURE.code, FIRST_FAILURE.retryable, "transcoding"
    )
    assert set(encodes.threads) == {"1080p", "720p", "540p"}
    assert harness.claimed.grant.media_key is None


@posix_only
def test_run_when_the_jobs_thread_runs_under_a_stop_check_should_stop_every_renditions_process_with_it(
    harness, encodes
):
    # Arrange: a stop check entered on the job's thread, as a 2.2.0 execution enters its lost authority's,
    # which turns true once every rendition runs its process
    authority_lost = threading.Event()
    started = threading.Barrier(len(PLANNED) + 1, timeout=WAIT_S)
    stops: dict[str, bool] = {}
    sibling = _stopped_sibling(stops)
    for name in PLANNED:
        encodes.act[name] = (lambda name: lambda: (started.wait(), sibling(name)()))(name)

    def lose_the_authority() -> None:
        started.wait()
        time.sleep(0.1)
        authority_lost.set()

    threading.Thread(target=lose_the_authority, daemon=True).start()

    # Act
    with stopped_when(authority_lost.is_set):
        result = harness.result()

    # Assert
    assert stops == {name: True for name in PLANNED}
    assert _error_of(result) == (STOPPED.error_class, STOPPED.code, STOPPED.retryable, "transcoding")
