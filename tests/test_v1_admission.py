"""Host-only admission checks; storage, encoding and provider execution are mocked."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import BackgroundTasks, Request
from fastapi.testclient import TestClient

from core import api
from core.config import Settings
from core.signing import sign_request
from core.v1_admission import (
    DEVELOPMENT_ADAPTER_ENV,
    MODAL_STAGED_DETAIL,
    V1_DISABLED_DETAIL,
    ExecutionDisabled,
    refuse_modal_execution,
)

DISABLED_VALUES = [None, "", "0", "true", " 1", "1 "]
VALID_REQUEST = {
    "uuid": "927f2e68-3342-4fcb-9812-66ce1c7ff3e8",
    "qualities": ["360p"],
    "callback_url": "https://hub.example.com/api/transcode/callback",
    "s3_bucket": "videohub-test",
    "s3_path_prefix": "videos/test",
    "s3_original_path": "original-videos/test/original.mp4",
}


def _set_admission(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(DEVELOPMENT_ADAPTER_ENV, raising=False)
    else:
        monkeypatch.setenv(DEVELOPMENT_ADAPTER_ENV, value)


@pytest.fixture
def settings(monkeypatch):
    settings = Settings(ffmpeg_encoder="libx264", ffmpeg_preset="fast", webhook_secret="test-secret")
    monkeypatch.setattr(api.Settings, "from_env", Mock(return_value=settings))
    monkeypatch.setattr(api, "resolve_encoder", Mock(return_value=("libx264", "fast")))
    return settings


@pytest.mark.parametrize("value", DISABLED_VALUES)
def test_transcode_when_development_not_admitted_should_refuse_before_reading_body(monkeypatch, value):
    # Arrange
    _set_admission(monkeypatch, value)
    body = AsyncMock(side_effect=AssertionError("body must not be read"))
    monkeypatch.setattr(Request, "body", body)
    verify = Mock(side_effect=AssertionError("signature must not be checked"))
    monkeypatch.setattr(api, "verify_request", verify)
    validate = Mock(side_effect=AssertionError("body must not be validated"))
    monkeypatch.setattr(api.TranscodeRequest, "model_validate_json", validate)
    schedule = Mock()
    monkeypatch.setattr(BackgroundTasks, "add_task", schedule)

    # Act
    with TestClient(api.app) as client:
        response = client.post("/transcode", content=b"{invalid json", headers={"Content-Type": "application/json"})

    # Assert
    assert response.status_code == 503
    assert response.json() == {"detail": V1_DISABLED_DETAIL}
    body.assert_not_called()
    verify.assert_not_called()
    validate.assert_not_called()
    schedule.assert_not_called()


def test_transcode_when_admitted_and_signed_should_schedule_valid_request(monkeypatch, settings):
    # Arrange
    _set_admission(monkeypatch, "1")
    schedule = Mock()
    monkeypatch.setattr(BackgroundTasks, "add_task", schedule)
    body = json.dumps(VALID_REQUEST).encode()

    # Act
    with TestClient(api.app) as client:
        response = client.post("/transcode", content=body, headers=sign_request(body, settings.webhook_secret))

    # Assert
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "uuid": VALID_REQUEST["uuid"]}
    schedule.assert_called_once_with(api._process_transcode, api.TranscodeRequest(**VALID_REQUEST), settings)


def test_transcode_when_admitted_with_bad_signature_should_refuse_without_scheduling(monkeypatch, settings):
    # Arrange
    _set_admission(monkeypatch, "1")
    schedule = Mock()
    monkeypatch.setattr(BackgroundTasks, "add_task", schedule)
    body = json.dumps(VALID_REQUEST).encode()

    # Act
    with TestClient(api.app) as client:
        response = client.post("/transcode", content=body, headers=sign_request(body, "wrong-secret"))

    # Assert
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid signature"}
    schedule.assert_not_called()


@pytest.mark.parametrize("value", DISABLED_VALUES)
def test_process_when_development_not_admitted_should_raise_without_side_effects(monkeypatch, settings, value):
    # Arrange: admission is read again when the task starts, even if previously enabled.
    _set_admission(monkeypatch, "1")
    request = api.TranscodeRequest(**VALID_REQUEST)
    _set_admission(monkeypatch, value)
    side_effects = []
    for target, name in [
        (api.tempfile, "mkdtemp"),
        (api, "download_from_s3"),
        (api, "upload_results"),
        (api, "transcode_to_hls"),
        (api, "send_progress"),
        (api, "send_result"),
        (api, "cleanup"),
    ]:
        mock = Mock(side_effect=AssertionError(f"{name} must not be called"))
        monkeypatch.setattr(target, name, mock)
        side_effects.append(mock)

    # Act
    with pytest.raises(ExecutionDisabled) as error:
        api._process_transcode(request, settings)

    # Assert
    assert str(error.value) == V1_DISABLED_DETAIL
    for mock in side_effects:
        mock.assert_not_called()


@pytest.mark.parametrize("value", [*DISABLED_VALUES, "1"])
def test_health_when_admission_changes_should_remain_available(monkeypatch, value):
    # Arrange
    _set_admission(monkeypatch, value)

    # Act
    with TestClient(api.app) as client:
        response = client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["v1_execution_admitted"] is (value == "1")


@pytest.mark.parametrize("fields", [("s3_bucket",), ("s3_original_path",), ("s3_bucket", "s3_original_path")])
@pytest.mark.parametrize("value", [None, ""])
def test_transcode_when_url_source_lacks_s3_fields_should_return_422_without_fetching(
    monkeypatch, settings, fields, value
):
    # Arrange
    _set_admission(monkeypatch, "1")
    request = {**VALID_REQUEST, "source_url": "https://example.com/arbitrary.mp4"}
    for field in fields:
        if value is None:
            request.pop(field)
        else:
            request[field] = value
    body = json.dumps(request).encode()
    schedule, download, fetch = Mock(), Mock(), Mock()
    monkeypatch.setattr(BackgroundTasks, "add_task", schedule)
    monkeypatch.setattr(api, "download_from_s3", download)
    monkeypatch.setattr("requests.sessions.Session.request", fetch)

    # Act
    with TestClient(api.app) as client:
        response = client.post("/transcode", content=body, headers=sign_request(body, settings.webhook_secret))

    # Assert
    assert response.status_code == 422
    assert {tuple(error["loc"]) for error in response.json()["detail"]} == {(field,) for field in fields}
    schedule.assert_not_called()
    download.assert_not_called()
    fetch.assert_not_called()


@pytest.fixture
def modal_wrapper(monkeypatch):
    # The host has no Modal SDK. Inert decorators expose the actual local function bodies.
    modal_stub = Mock()
    modal_stub.App.return_value.function.side_effect = lambda **kwargs: lambda fn: fn
    modal_stub.asgi_app.side_effect = lambda: lambda fn: fn
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location(
        "modal_app_under_test", Path(__file__).parents[1] / "wrappers" / "modal_app.py"
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


@pytest.mark.parametrize("value", [*DISABLED_VALUES, "1"])
def test_modal_admission_when_any_development_setting_should_refuse(monkeypatch, value):
    # Arrange
    _set_admission(monkeypatch, value)

    # Act
    with pytest.raises(ExecutionDisabled) as error:
        refuse_modal_execution()

    # Assert
    assert str(error.value) == MODAL_STAGED_DETAIL


@pytest.mark.parametrize("value", [None, "1"])
def test_modal_transcode_when_staged_should_refuse_without_reading_or_spawning(monkeypatch, modal_wrapper, value):
    # Arrange
    _set_admission(monkeypatch, value)
    process = Mock()
    monkeypatch.setattr(modal_wrapper, "process_transcode", process)
    body = AsyncMock(side_effect=AssertionError("body must not be read"))
    monkeypatch.setattr(Request, "body", body)

    # Act
    with TestClient(modal_wrapper.web()) as client:
        response = client.post("/transcode", content=b"{invalid json", headers={"Content-Type": "application/json"})

    # Assert
    assert response.status_code == 503
    assert response.json() == {"detail": MODAL_STAGED_DETAIL}
    body.assert_not_called()
    process.assert_not_called()
    process.spawn.assert_not_called()


@pytest.mark.parametrize("value", [None, "1"])
def test_modal_process_when_called_directly_should_refuse_before_validation(monkeypatch, modal_wrapper, value):
    # Arrange
    _set_admission(monkeypatch, value)
    validate, process = Mock(), Mock()
    monkeypatch.setattr(api.TranscodeRequest, "model_validate", validate)
    monkeypatch.setattr(api, "_process_transcode", process)

    # Act
    with pytest.raises(ExecutionDisabled) as error:
        modal_wrapper.process_transcode({})

    # Assert
    assert str(error.value) == MODAL_STAGED_DETAIL
    validate.assert_not_called()
    process.assert_not_called()
