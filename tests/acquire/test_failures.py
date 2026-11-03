"""Unexpected transfer faults retain signed terminal evidence without leaking their exception chains."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import Any

import pytest

from core.drive_import import acquire, transfer
from core.drive_import.clock import DeadlineReached
from core.drive_import.failure import Code, ImportFailed, Stage
from core.protocol import outbox, signing
from core.protocol.transport import TransportFailed
from tests.acquire.support import PART_SIZE, Scenario
from tests.acquire.test_entry import BrokenLog
from tests.contract.v2_support import WORKER_KEY
from wrappers import runpod_acquire_handler as entry

SECRET = "https://secret.invalid/presigned?token=exception-text-must-not-escape"


def _raise_chained(kind: type[Exception]) -> None:
    try:
        raise ValueError(SECRET)
    except ValueError as cause:
        raise kind(SECRET) from cause


@pytest.mark.parametrize("kind", [MemoryError, transfer.SelfCheckFailed])
@pytest.mark.parametrize("stage", ["downloading", "uploading"])
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("broken_logger", [False, True])
def test_transfer_crash_when_granted_should_seal_the_stage_failure_and_preserve_its_carriers(
    kind: type[Exception], stage: str, expired: bool, broken_logger: bool,
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = Scenario()
    put = scenario.http.put
    carriers: list[dict[str, str]] = []
    store_terminal, callback = acquire._store_terminal, acquire._callback

    def fail() -> None:
        if expired:
            scenario.clock.advance(60)
        _raise_chained(kind)

    def buffer(self: Any, chunk: bytes) -> None:
        fail()

    def upload(url: str, *args: Any, **kwargs: Any) -> Any:
        if "partNumber=" in url:
            fail()
        return put(url, *args, **kwargs)

    def storage_carrier(report: outbox.SealedMessage, *args: Any) -> None:
        carriers.append(outbox.runpod_output(report))
        store_terminal(report, *args)

    def callback_carrier(report: outbox.SealedMessage, *args: Any) -> None:
        carriers.append(outbox.runpod_output(report))
        callback(report, *args)

    if stage == "downloading":
        monkeypatch.setattr(transfer._Transfer, "_take", buffer)
    else:
        monkeypatch.setattr(scenario.http, "put", upload)
    monkeypatch.setattr(acquire, "_store_terminal", storage_carrier)
    monkeypatch.setattr(acquire, "_callback", callback_carrier)
    log_bug = acquire._log_bug

    def checked_log(error: Exception, **kwargs: Any) -> None:
        assert sys.exc_info() == (None, None, None)
        log_bug(error, **kwargs)

    monkeypatch.setattr(acquire, "_log_bug", checked_log)
    caplog.set_level(logging.DEBUG)
    sink = logging.StreamHandler(BrokenLog())
    monkeypatch.setattr(logging, "raiseExceptions", True)
    if broken_logger:
        acquire.logger.addHandler(sink)
    try:
        output = scenario.run()
    finally:
        acquire.logger.removeHandler(sink)

    message = json.loads(output["vh_message"])
    assert message["message_kind"] == "source.acquire.result.failed"
    code = "deadline" if expired else "download_failed" if stage == "downloading" else "upload_failed"
    assert (message["error"]["code"], message["error"]["stage"], message["error"]["retryable"]) == (code, stage, not expired)
    assert signing.verify(output["vh_message"].encode("ascii"), WORKER_KEY, output["vh_signature"])
    assert carriers == [output, output]
    if expired:
        # §2.6 forbids starting either network delivery after the wall deadline.
        assert not scenario.http.result_puts and not scenario.hub.callbacks
    else:
        assert json.loads(scenario.http.result_puts[0].body) == output
        assert scenario.hub.callbacks[0].body == output["vh_message"].encode("ascii")
        assert scenario.hub.callbacks[0].signature == output["vh_signature"]
    shown = repr(output) + caplog.text + capsys.readouterr().err
    assert SECRET not in shown and "exception-text-must-not-escape" not in shown
    assert kind.__name__ in caplog.text and "_raise_chained" in caplog.text
    if broken_logger:
        assert "test logging sink failed" in shown
    assert scenario.http.closed


def test_download_crash_after_a_successful_part_should_use_downloading_again(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = Scenario(PART_SIZE + 37)
    take = transfer._Transfer._take

    def buffer(self: Any, chunk: bytes) -> None:
        if self._part > 1:
            _raise_chained(MemoryError)
        take(self, chunk)

    monkeypatch.setattr(transfer._Transfer, "_take", buffer)
    output = scenario.run()
    assert len(scenario.http.part_puts) == 1
    assert json.loads(output["vh_message"])["error"]["code"] == "download_failed"
    assert json.loads(scenario.http.result_puts[0].body) == output
    assert scenario.hub.callbacks[0].body == output["vh_message"].encode("ascii")


@pytest.mark.parametrize("expired", [False, True])
def test_completed_sealing_crash_should_return_only_a_sanitized_provider_failure(
    expired: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = Scenario()
    attempts: list[str] = []

    def fail(message: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        attempts.append(message["message_kind"])
        if expired:
            scenario.clock.advance(60)
        _raise_chained(outbox.SelfCheckFailed)

    monkeypatch.setattr(acquire, "seal_message", fail)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    sink = logging.StreamHandler(BrokenLog())
    entry.logger.addHandler(sink)
    try:
        with pytest.raises(entry.WorkerFault) as caught:
            scenario.run()
    finally:
        entry.logger.removeHandler(sink)
    assert attempts == ["source.acquire.result.completed"]
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    shown = "".join(traceback.format_exception(caught.value)) + caplog.text + capsys.readouterr().err
    assert SECRET not in shown and "SelfCheckFailed" in shown
    assert not scenario.http.result_puts and not scenario.hub.callbacks
    assert scenario.http.closed


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_transfer_when_interrupted_should_not_convert_base_exceptions_to_terminals(
    kind: type[BaseException], monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario()

    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise kind()

    monkeypatch.setattr(acquire, "transfer", interrupt)
    with pytest.raises(kind):
        scenario.run()
    assert not scenario.http.result_puts and not scenario.hub.callbacks
    assert scenario.http.closed


@pytest.mark.parametrize("channel,kind", [
    ("storage", OSError), ("storage", ImportFailed), ("storage", DeadlineReached), ("storage", ValueError),
    ("callback", OSError), ("callback", ImportFailed), ("callback", TransportFailed), ("callback", ValueError),
])
@pytest.mark.parametrize("broken_logger", [False, True])
def test_delivery_crash_should_distinguish_transport_failure_from_bug_without_changing_the_terminal(
    channel: str, kind: type[Exception], broken_logger: bool, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = Scenario()
    put = acquire.store.put
    carriers: list[dict[str, str]] = []

    def fail() -> None:
        if kind is ImportFailed:
            raise ImportFailed(Code.UPLOAD_FAILED, Stage.RECEIPT)
        _raise_chained(kind)

    def storage(http: Any, url: str, body: Any, headers: Any, stop: Any, stage: Stage, retry: Any) -> Any:
        if stage is Stage.RECEIPT:
            carriers.append(json.loads(body))
            fail()
        return put(http, url, body, headers, stop, stage, retry)

    def callback(report: outbox.SealedMessage, **kwargs: Any) -> None:
        carriers.append(outbox.runpod_output(report))
        fail()

    if channel == "storage":
        monkeypatch.setattr(acquire.store, "put", storage)
    else:
        monkeypatch.setattr(acquire, "deliver", callback)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    sink = logging.StreamHandler(BrokenLog())
    if broken_logger:
        acquire.logger.addHandler(sink)
    try:
        output = scenario.run()
    finally:
        acquire.logger.removeHandler(sink)
    assert carriers == [output]
    assert json.loads(output["vh_message"])["message_kind"] == "source.acquire.result.completed"
    if channel == "storage":
        assert scenario.hub.callbacks[0].body == output["vh_message"].encode("ascii")
        assert scenario.hub.callbacks[0].signature == output["vh_signature"]
    else:
        assert json.loads(scenario.http.result_puts[0].body) == output
    shown = repr(output) + caplog.text + capsys.readouterr().err
    assert SECRET not in shown
    if kind is ValueError:
        assert "acquire terminal delivery failed on ValueError" in caplog.text
        assert f"delivery failed at {channel}" not in caplog.text
        assert "_raise_chained" not in caplog.text
    else:
        assert f"delivery failed at {channel}" in caplog.text
        assert "delivery failed on" not in caplog.text
    if broken_logger:
        assert "test logging sink failed" in shown
