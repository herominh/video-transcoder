"""Acquire deadline enforcement through the real requests transport, with synthetic peers only."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import replace
from typing import Any, Mapping

import pytest

from core.drive_import import google
from core.drive_import.acquire import ClaimedAcquire, run_acquire
from core.drive_import.clock import StopLine
from core.drive_import.transport import HttpResponse, RequestsSession
from core.protocol.acquire_claim import AcquireGrant, claim_execution
from core.protocol.acquire_dispatch import accept_dispatch
from tests.acquire.support import Scenario, wire
from tests.contract.v2_support import HUB_KEY, RUNTIME_ID, signed_carrier
from tests.drive_import.test_transport import (
    ENDED_WITHIN_S, ONE, STOP_LINE_S, _Tls, _loopback_server, tls,
)

DNS_STOP_S = 0.1
DNS_HELD_S = 0.25


def claimed(scenario: Scenario, seconds: float) -> ClaimedAcquire:
    dispatch = accept_dispatch(signed_carrier(wire(scenario.request), HUB_KEY), settings=scenario.runtime.settings, now=scenario.clock.now())
    grant = claim_execution(
        dispatch, settings=scenario.runtime.settings, runtime_id=RUNTIME_ID, transport=scenario.hub,
        clock=scenario.clock.now, monotonic=scenario.clock.monotonic, sleep=scenario.clock.sleep,
        new_message_id=scenario.runtime.new_uuid, wall_time_ends_at=scenario.clock.monotonic() + 60,
    )
    assert isinstance(grant, AcquireGrant)
    return ClaimedAcquire(dispatch, grant, RUNTIME_ID, time.monotonic() + seconds)


@pytest.mark.parametrize("phase", ["dns", "connect"])
def test_acquire_when_connection_setup_crosses_the_deadline_should_start_no_later_connection(
    phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: all resolution and connect operations are intercepted; no external connection is possible.
    scenario = Scenario()
    authority = claimed(scenario, DNS_STOP_S)
    runtime = replace(scenario.runtime, monotonic=time.monotonic, sleep=time.sleep)
    connects: list[float] = []

    def resolve(*args: Any, **kwargs: Any) -> Any:
        if phase == "dns":
            time.sleep(DNS_HELD_S)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))] * 3

    def connect(sock: socket.socket, address: Any) -> None:
        connects.append(time.monotonic())
        time.sleep(sock.gettimeout() or DNS_HELD_S)
        raise TimeoutError("synthetic connect timeout")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", connect)
    session = RequestsSession()
    # Act
    try:
        report = run_acquire(authority, runtime, session)
    finally:
        session.close()
    # Assert
    error = json.loads(report.body)["error"]
    assert error["code"] == "deadline" and error["stage"] == "downloading"
    assert all(start < authority.wall_time_ends_at for start in connects)
    assert len(connects) == (0 if phase == "dns" else 1)


class LocalSession(RequestsSession):
    def __init__(self, base: str, certificate: str) -> None:
        super().__init__(ca_bundle=certificate)
        self.base = base

    def put(self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *,
            stop_line: StopLine) -> HttpResponse:
        return super().put(self.base + "/hold", data, headers, timeout, stop_line=stop_line)


@pytest.mark.parametrize("phase", ["metadata-trickles", "media-trickles", "put", "handshake"])
def test_acquire_when_a_peer_holds_io_should_close_at_the_wall_deadline(
    phase: str, monkeypatch: pytest.MonkeyPatch, tls: _Tls,
) -> None:
    # Arrange
    scenario = Scenario(ONE.size)
    runtime = replace(scenario.runtime, monotonic=time.monotonic, sleep=time.sleep)
    with _loopback_server(tls=tls, drive_behaviour=phase, handshake_delay=10 if phase == "handshake" else 0) as (base, seen):
        monkeypatch.setattr(google, "FILES_URL", base + "/drive/v3/files/")
        session = LocalSession(base, str(tls.certificate))
        authority = claimed(scenario, STOP_LINE_S)
        started = time.monotonic()
        # Act
        try:
            report = run_acquire(authority, runtime, session)
        finally:
            session.close()
        elapsed = time.monotonic() - started
    # Assert
    error = json.loads(report.body)["error"]
    assert error["code"] == "deadline"
    assert error["stage"] == ("uploading" if phase == "put" else "downloading")
    assert elapsed < ENDED_WITHIN_S
    assert not any("acquire-result" in path for _, path, _ in seen.requests)
