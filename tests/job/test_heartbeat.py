"""The heartbeat of protocol 2.2.0 (core/job/heartbeat.py) and the stop of a bounded process once the authority is
lost (core/bounded_process.py stopped_when()). The Hub is a fake on the test's clock; the waits of the heartbeat's
thread are real but short."""

from __future__ import annotations

import os
import threading
import time

import pytest

from core.bounded_process import Deadline, run_bounded, stopped_when
from core.job.heartbeat import HEARTBEAT_INTERVAL_S, THREAD_NAME, Heartbeat
from core.protocol.capabilities_client import CapabilityClient
from core.protocol.transport import HttpAnswer
from tests.contract.v2_support import FakeClock, uuid_source
from tests.job.capability_fakes import FakeHub, capability_claimed, capability_settings, hub_error, wait_until

POLL_S = 0.005
# Long enough for the heartbeat's thread to have looked many times, were it to send too early.
QUIET_S = 0.2
SLEEP_ARGV = ["sleep", "30"]
# A lost lease ends a running process within this many real seconds.
STOPPED_WITHIN_S = 5.0


class _Setup:
    def __init__(self) -> None:
        self.claimed = capability_claimed()
        self.clock = FakeClock()
        self.hub = FakeHub(self.claimed, self.clock)
        self.client = CapabilityClient(
            dispatch=self.claimed.dispatch, grant=self.claimed.grant, runtime_id=self.claimed.runtime_id,
            settings=capability_settings(), transport=self.hub, clock=self.clock, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, new_message_id=uuid_source(), time_left=lambda: 3600.0,
        )
        self.heartbeat = Heartbeat(self.client, poll_s=POLL_S)


def _heartbeat_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == THREAD_NAME]


def test_heartbeat_when_120_seconds_pass_without_a_request_should_send_one_and_none_earlier():
    # Arrange
    setup = _Setup()
    setup.client.ask("transcoding")
    setup.heartbeat.start()
    try:
        # Act
        setup.clock.advance(HEARTBEAT_INTERVAL_S - 1)
        time.sleep(QUIET_S)
        before = len(setup.hub.requests)
        setup.clock.advance(1)
        sent = wait_until(lambda: len(setup.hub.requests) == 2)
    finally:
        stopped = setup.heartbeat.stop()
    # Assert
    assert before == 1 and sent
    beat = setup.hub.requests[1]
    assert (beat["seq"], beat["stage"], beat["put"], beat["source"]) == (2, "transcoding", None, False)
    assert stopped


def test_heartbeat_when_a_request_of_the_job_renewed_the_lease_should_count_from_that_request():
    # Arrange
    setup = _Setup()
    setup.client.ask("probing")
    setup.heartbeat.start()
    try:
        # Act
        setup.clock.advance(100)
        setup.client.ask("downloading")
        setup.clock.advance(100)
        time.sleep(QUIET_S)
        quiet = len(setup.hub.requests)
        setup.clock.advance(20)
        sent = wait_until(lambda: len(setup.hub.requests) == 3)
    finally:
        setup.heartbeat.stop()
    # Assert
    assert quiet == 2 and sent
    assert setup.hub.requests[2]["stage"] == "downloading"


def test_heartbeat_when_the_hub_ends_the_lease_should_end_and_leave_the_failure_to_lost():
    # Arrange
    setup = _Setup()
    setup.client.ask("transcoding")
    setup.hub.script.append(hub_error(403, "lease_expired", retryable=False))
    setup.heartbeat.start()
    # Act
    setup.clock.advance(HEARTBEAT_INTERVAL_S)
    lost = wait_until(lambda: setup.client.lost() is not None)
    ended = wait_until(lambda: not _heartbeat_threads())
    # Assert
    assert lost and ended
    assert setup.client.lost().failure.code == "lease_expired"
    assert setup.heartbeat.stop()


def test_heartbeat_when_stopped_should_end_its_thread():
    # Arrange
    setup = _Setup()
    setup.heartbeat.start()
    # Act
    stopped = setup.heartbeat.stop()
    # Assert
    assert stopped
    assert not _heartbeat_threads()


@pytest.mark.skipif(os.name != "posix", reason="bounded processes run under POSIX process groups")
def test_lost_lease_when_a_bounded_process_runs_should_stop_it_through_the_stop_check():
    # Arrange: the Hub ends the lease at the heartbeat 120 s after the job's last request.
    setup = _Setup()
    setup.client.ask("transcoding")
    setup.hub.script.append(hub_error(403, "lease_expired", retryable=False))
    setup.heartbeat.start()
    threading.Timer(0.2, setup.clock.advance, args=(HEARTBEAT_INTERVAL_S,)).start()
    started = time.monotonic()
    try:
        # Act
        with stopped_when(lambda: setup.client.lost() is not None):
            result = run_bounded(
                SLEEP_ARGV, deadline=Deadline.after_ms(60_000), max_stdout_bytes=1024, max_stderr_bytes=1024,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, cwd="/",
            )
    finally:
        setup.heartbeat.stop()
    # Assert
    assert result.stopped_by_monitor and result.returncode is None
    assert time.monotonic() - started < STOPPED_WITHIN_S


@pytest.mark.skipif(os.name != "posix", reason="bounded processes run under POSIX process groups")
def test_stop_check_when_it_holds_already_should_start_no_process_and_outside_its_block_should_stop_none():
    # Arrange
    argv = ["true"]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    # Act
    with stopped_when(lambda: True):
        refused = run_bounded(argv, deadline=Deadline.after_ms(10_000), max_stdout_bytes=64, max_stderr_bytes=64,
                              env=env, cwd="/")
    ran = run_bounded(argv, deadline=Deadline.after_ms(10_000), max_stdout_bytes=64, max_stderr_bytes=64,
                      env=env, cwd="/")
    # Assert
    assert refused.stopped_by_monitor and refused.returncode is None
    assert ran.returncode == 0 and not ran.stopped_by_monitor


def test_heartbeat_when_stopped_with_a_beat_in_flight_and_refusals_for_now_should_send_nothing_after_it():
    # Arrange: the beat's answer is held, then refused for now; more refusals for now would follow.
    setup = _Setup()
    setup.client.ask("transcoding")
    in_flight, release = threading.Event(), threading.Event()

    def held(request, hub):
        in_flight.set()
        release.wait(5.0)
        return HttpAnswer(503, b"", None)

    setup.hub.script.extend([held] + [HttpAnswer(503, b"", None)] * 50)
    setup.heartbeat.start()
    setup.clock.advance(HEARTBEAT_INTERVAL_S)
    assert wait_until(in_flight.is_set)
    threading.Timer(0.1, release.set).start()
    # Act
    stopped = setup.heartbeat.stop()
    sent = len(setup.hub.requests)
    time.sleep(QUIET_S)
    # Assert
    assert stopped and not _heartbeat_threads()
    assert sent == 2 and len(setup.hub.requests) == 2
    assert setup.client.lost() is None


def test_heartbeat_when_its_thread_meets_an_unexpected_error_should_end_the_work_as_internal_error():
    # Arrange
    setup = _Setup()
    setup.client.ask("transcoding")
    setup.hub.script.append(RuntimeError("an unexpected error of the transport"))
    setup.heartbeat.start()
    # Act
    setup.clock.advance(HEARTBEAT_INTERVAL_S)
    lost = wait_until(lambda: setup.client.lost() is not None)
    setup.heartbeat.stop()
    # Assert
    assert lost
    failure = setup.client.lost().failure
    assert (failure.error_class, failure.code) == ("internal_error", "heartbeat_failed")
