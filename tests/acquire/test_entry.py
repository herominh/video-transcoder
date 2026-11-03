"""Entry boot, lifecycle and redaction, including failures while the logger itself is broken."""

from __future__ import annotations

import io
import json
import logging
import traceback
from dataclasses import replace
from typing import Any

import pytest

from core.protocol import files
from core.protocol.acquire_claim import claim_execution
from core.protocol.acquire_dispatch import accept_dispatch, receiver_context
from core.protocol.claim import Unclaimed
from core.protocol.settings import SettingsInvalid
from tests.acquire.support import SIGNATURE_CANARY, Scenario, environment, wire
from tests.contract.v2_support import HUB_KEY, RUNTIME_ID, WORKER_KEY, key_forms, signed_carrier
from tests.drive_import.fakes import API_KEY, FILE_ID, RESOURCE_KEY, Fail
from wrappers import runpod_acquire_handler as entry


class BrokenLog(io.TextIOBase):
    def write(self, text: str) -> int:
        raise OSError("test logging sink failed")


def test_acquire_entry_when_booting_should_check_the_tree_before_accepting_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    checks: list[str] = []

    def refuse() -> None:
        checks.append("tree")
        raise RuntimeError("test tree invalid")

    monkeypatch.setattr(files, "verify_contract_tree", refuse)
    # Act / Assert
    with pytest.raises(RuntimeError, match="test tree invalid"):
        entry.build_handler({})
    assert checks == ["tree"]


def test_acquire_entry_when_configured_with_a_draft_should_refuse_before_sdk_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    started: list[Any] = []
    monkeypatch.setattr(entry, "configure_logging", lambda: None)
    # Act / Assert
    with pytest.raises(SettingsInvalid):
        entry.main(start=started.append, env={**environment(), "VH_ACCEPTED_PROTOCOL_VERSIONS": "2.1.0"})
    assert not started


def test_acquire_entry_when_started_should_configure_logging_before_boot_and_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    order: list[str] = []
    monkeypatch.setattr(entry, "configure_logging", lambda: order.append("logging"))

    def build(env: Any) -> Any:
        order.append("boot")
        return lambda job: {}

    monkeypatch.setattr(entry, "build_handler", build)
    # Act
    entry.main(start=lambda config: order.append("sdk"), env=environment())
    # Assert
    assert order == ["logging", "boot", "sdk"]


def test_acquire_entry_when_reused_should_make_and_close_a_session_and_transport_for_each_start() -> None:
    # Arrange
    scenario = Scenario()
    sessions: list[Any] = []
    transports: list[object] = []

    def session() -> Any:
        fresh = type(scenario.http)(scenario.file, scenario.clock)
        sessions.append(fresh)
        return fresh

    def transport() -> Any:
        fresh = type(scenario.hub)(scenario.request, scenario.clock)
        transports.append(fresh)
        return fresh

    handler = entry.job_handler(scenario.runtime, session, transport)
    carrier = signed_carrier(wire(scenario.request), HUB_KEY)
    # Act
    handler({"input": carrier})
    handler({"input": carrier})
    # Assert
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(session.closed for session in sessions)
    assert len(transports) == 2 and transports[0] is not transports[1]


def test_acquire_authority_when_rendered_should_hide_tokens_keys_urls_and_external_identifiers() -> None:
    # Arrange
    scenario = Scenario()
    dispatch = accept_dispatch(signed_carrier(wire(scenario.request), HUB_KEY), settings=scenario.runtime.settings, now=scenario.clock.now())
    grant = claim_execution(
        dispatch, settings=scenario.runtime.settings, runtime_id=RUNTIME_ID, transport=scenario.hub,
        clock=scenario.clock.now, monotonic=scenario.clock.monotonic, sleep=scenario.clock.sleep,
        new_message_id=scenario.runtime.new_uuid, wall_time_ends_at=scenario.clock.monotonic() + 60,
    )
    assert not isinstance(grant, Unclaimed)
    context = receiver_context(settings=scenario.runtime.settings, now=scenario.clock.now(), dispatch=dispatch, runtime_id=RUNTIME_ID)
    # Act
    shown = repr((dispatch, grant, grant.parts, context, scenario.runtime))
    # Assert
    for value in (dispatch.bootstrap_token, API_KEY, FILE_ID, RESOURCE_KEY, SIGNATURE_CANARY, "https://", RUNTIME_ID,
                  *key_forms(HUB_KEY), *key_forms(WORKER_KEY)):
        assert value not in shown


@pytest.mark.parametrize("source", ["google", "part", "callback", "unexpected"])
@pytest.mark.parametrize("broken_logger", [False, True])
def test_acquire_when_errors_contain_secrets_should_redact_output_exception_chains_and_logs(
    source: str, broken_logger: bool, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    scenario = Scenario()
    secret = " | ".join((API_KEY, FILE_ID, RESOURCE_KEY, scenario.request["claim"]["bootstrap_token"],
                         SIGNATURE_CANARY, scenario.urls()[0], HUB_KEY.hex(), WORKER_KEY.hex()))
    if source == "google":
        scenario.http.metadata = [Fail(OSError(secret))] * 3
    elif source == "part":
        scenario.http.puts[scenario.urls()[0]] = [Fail(OSError(secret))] * 4
    elif source == "callback":
        scenario.hub.callback_actions = [ValueError(secret)]
    else:
        scenario.http.metadata = [Fail(ValueError(secret))]
    caplog.set_level(logging.DEBUG)
    loggers = [logging.getLogger(name) for name in ("core.protocol.claim", "core.protocol.outbox", "core.drive_import.transfer", entry.__name__)]
    sink = logging.StreamHandler(BrokenLog())
    if broken_logger:
        monkeypatch.setattr(logging, "raiseExceptions", True)
        for logger in loggers:
            logger.addHandler(sink)
    output = ""
    error_text = ""
    # Act
    try:
        if source == "unexpected":
            with pytest.raises(entry.WorkerFault) as caught:
                scenario.run()
            assert caught.value.__cause__ is None and caught.value.__context__ is None
            error_text = "".join(traceback.format_exception(caught.value)) + repr(caught.value)
        else:
            output = repr(scenario.run())
    finally:
        for logger in loggers:
            if broken_logger:
                logger.removeHandler(sink)
    shown = output + error_text + caplog.text + capsys.readouterr().err
    # Assert
    for value in (API_KEY, FILE_ID, RESOURCE_KEY, scenario.request["claim"]["bootstrap_token"], SIGNATURE_CANARY,
                  scenario.urls()[0], HUB_KEY.hex(), WORKER_KEY.hex()):
        assert value not in shown
    assert scenario.http.closed
    if broken_logger and source != "callback":
        assert "test logging sink failed" in shown


def test_acquire_entry_when_http_libraries_log_should_suppress_their_url_bearing_records() -> None:
    # Arrange
    old = {name: (list(logging.getLogger(name).handlers), logging.getLogger(name).propagate) for name in entry.HTTP_LIBRARY_LOGGERS}
    # Act
    try:
        entry.configure_logging()
        # Assert
        for name in entry.HTTP_LIBRARY_LOGGERS:
            logger = logging.getLogger(name)
            assert not logger.propagate
            assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)
    finally:
        for name, (handlers, propagate) in old.items():
            logging.getLogger(name).handlers = handlers
            logging.getLogger(name).propagate = propagate


def test_acquire_entry_when_imported_should_disable_tls_key_logging_before_loading_requests() -> None:
    # Arrange
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "SSLKEYLOGFILE": "must-not-write-tls-keys"}
    script = "import wrappers.runpod_acquire_handler; import os; assert 'SSLKEYLOGFILE' not in os.environ"
    # Act
    process = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    # Assert
    assert process.returncode == 0, process.stderr
