"""Tests for _process_transcode result semantics and the RunPod handler.

These pin the reliability contract:
- the pipeline RETURNS the result payload (job output for serverless pollers)
- a callback delivery failure never flips a successful transcode to "failed"
- the RunPod handler raises on failure so RunPod's ledger tells the truth
- uuid is validated before it touches filesystem paths / S3 keys
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.api import TranscodeRequest, _process_transcode, app
from core.config import Settings
from core.signing import sign_request

VALID_UUID = "927f2e68-3342-4fcb-9812-66ce1c7ff3e8"

VALID_REQUEST = {
    "uuid": VALID_UUID,
    "source_url": "https://example.com/video.mp4",
    "qualities": ["360p"],
    "encryption_key_hex": None,
    "segment_duration": 6,
    "callback_url": "https://hub.example.com/api/transcode/callback",
    "s3_bucket": "videohub-test",
    "s3_path_prefix": f"videos/{VALID_UUID}",
    "s3_original_path": f"original-videos/{VALID_UUID}/original.mp4",
}

TRANSCODE_RESULT = {
    "duration": 42,
    "qualities": [
        {
            "name": "360p",
            "width": 640,
            "height": 360,
            "bitrate": 600000,
            "playlist": "360p/playlist.m3u8",
        }
    ],
    "master_playlist": "master.m3u8",
    "thumbnail": "thumbnail.jpg",
}


def _settings() -> Settings:
    return Settings(
        ffmpeg_encoder="libx264",
        ffmpeg_preset="fast",
        webhook_secret="test-secret",
    )


def _request() -> TranscodeRequest:
    return TranscodeRequest.model_validate(VALID_REQUEST)


class TestProcessTranscodeResult:
    @pytest.mark.parametrize(
        ("pipeline_error", "expected_status"),
        [(None, "ready"), (RuntimeError("download exploded"), "failed")],
    )
    def test_process_when_cleanup_fails_should_preserve_pipeline_result(
        self, monkeypatch, tmp_path, pipeline_error, expected_status
    ):
        # Arrange
        monkeypatch.setattr("core.api.tempfile.mkdtemp", lambda **kwargs: str(tmp_path))
        monkeypatch.setattr("core.api.download_source", Mock(side_effect=pipeline_error))
        monkeypatch.setattr("core.api.upload_original", Mock())
        monkeypatch.setattr("core.api.os.path.getsize", Mock(return_value=1234))
        monkeypatch.setattr("core.api.transcode_to_hls", Mock(return_value=TRANSCODE_RESULT))
        monkeypatch.setattr("core.api.upload_results", Mock())
        monkeypatch.setattr("core.api.send_progress", Mock(return_value=True))
        monkeypatch.setattr("core.api.cleanup", Mock(side_effect=PermissionError("cleanup denied")))
        mock_send = Mock()
        monkeypatch.setattr("core.api.send_result", mock_send)
        settings = _settings()

        # Act
        payload = _process_transcode(_request(), settings)

        # Assert
        assert payload["status"] == expected_status
        assert payload["uuid"] == VALID_UUID
        if pipeline_error is None:
            assert payload["duration"] == TRANSCODE_RESULT["duration"]
            assert payload["qualities"] == TRANSCODE_RESULT["qualities"]
            assert payload["master_playlist"] == TRANSCODE_RESULT["master_playlist"]
            assert payload["source_filesize"] == 1234
        else:
            assert payload["error_message"] == "download exploded"
        mock_send.assert_called_once_with(
            VALID_REQUEST["callback_url"], settings.webhook_secret, payload
        )

    @patch("core.api.send_result")
    @patch("core.api.send_progress", return_value=True)
    @patch("core.api.upload_results")
    @patch("core.api.transcode_to_hls", return_value=TRANSCODE_RESULT)
    @patch("core.api.os.path.getsize", return_value=1234)
    @patch("core.api.upload_original")
    @patch("core.api.download_source")
    def test_process_when_success_should_return_ready_payload(
        self, _dl, _up_orig, _size, _hls, _up_res, _prog, mock_send
    ):
        # Act
        payload = _process_transcode(_request(), _settings())

        # Assert: payload is the full ready result…
        assert payload["status"] == "ready"
        assert payload["uuid"] == VALID_UUID
        assert payload["duration"] == 42
        assert payload["master_playlist"] == "master.m3u8"
        assert payload["source_filesize"] == 1234
        # …and the same payload was offered to the callback fast-path.
        mock_send.assert_called_once()
        assert mock_send.call_args.args[2] == payload

    @patch("core.api.send_result")
    @patch("core.api.send_progress", return_value=True)
    @patch("core.api.download_source", side_effect=RuntimeError("download exploded"))
    def test_process_when_pipeline_fails_should_return_failed_payload(
        self, _dl, _prog, mock_send
    ):
        # Act
        payload = _process_transcode(_request(), _settings())

        # Assert
        assert payload["status"] == "failed"
        assert "download exploded" in payload["error_message"]
        # The failure payload (not a bogus ready) is what gets delivered.
        assert mock_send.call_args.args[2]["status"] == "failed"

    @patch("core.api.send_result", side_effect=ConnectionError("hub unreachable"))
    @patch("core.api.send_progress", return_value=True)
    @patch("core.api.upload_results")
    @patch("core.api.transcode_to_hls", return_value=TRANSCODE_RESULT)
    @patch("core.api.os.path.getsize", return_value=1234)
    @patch("core.api.upload_original")
    @patch("core.api.download_source")
    def test_process_when_callback_delivery_fails_should_still_return_ready(
        self, _dl, _up_orig, _size, _hls, _up_res, _prog, _send
    ):
        """Regression: a delivery hiccup used to trigger a 'failed' callback
        for an already-successful transcode."""
        # Act — must not raise despite send_result failing.
        payload = _process_transcode(_request(), _settings())

        # Assert
        assert payload["status"] == "ready"

    @patch("core.api.send_result")
    @patch("core.api.send_progress", return_value=False)
    @patch("core.api.upload_results")
    @patch("core.api.transcode_to_hls", return_value=TRANSCODE_RESULT)
    @patch("core.api.os.path.getsize", return_value=1234)
    @patch("core.api.upload_original")
    @patch("core.api.download_source")
    def test_process_when_hub_unreachable_should_back_off_progress(
        self, _dl, _up_orig, _size, _hls, _up_res, mock_prog, _send
    ):
        # Arrange
        with patch("core.api.monotonic", return_value=100, create=True):
            # Act
            _process_transcode(_request(), _settings())

        # Assert: immediate events during backoff do not each waste a timeout.
        assert mock_prog.call_count == 1

    @pytest.mark.parametrize("hub_recovers", [False, True])
    @patch("core.api.send_result")
    @patch("core.api.upload_results")
    @patch("core.api.os.path.getsize", return_value=1234)
    @patch("core.api.upload_original")
    @patch("core.api.download_source")
    def test_process_when_progress_delivery_fails_should_retry_after_bounded_backoff(
        self, _dl, _up_orig, _size, _up_res, _send, monkeypatch, hub_recovers
    ):
        # Arrange
        now = 100
        attempts = []

        def post_progress(*args, **kwargs):
            attempts.append(now)
            if len(attempts) == 1 or not hub_recovers:
                raise requests.exceptions.ConnectTimeout("hub temporarily unreachable")
            return Mock(status_code=200)

        def transcode(**kwargs):
            nonlocal now
            for now in (129, 130, 131, 159, 160, 189, 190):
                kwargs["progress_callback"](50, "Rendition completed")
            return TRANSCODE_RESULT

        monkeypatch.setattr("core.api.monotonic", lambda: now, raising=False)
        monkeypatch.setattr("core.callback.requests.post", post_progress)
        monkeypatch.setattr("core.api.transcode_to_hls", transcode)

        # Act
        payload = _process_transcode(_request(), _settings())

        # Assert
        assert payload["status"] == "ready"
        if hub_recovers:
            assert attempts == [100, 130, 131, 159, 160, 189, 190, 190]
        else:
            assert attempts == [100, 130, 160, 190]
        _send.assert_called_once()


class TestTranscodeRequestUuidValidation:
    @pytest.mark.parametrize("bad_uuid", ["../../etc/passwd", VALID_UUID.replace("-", "")])
    def test_transcode_when_uuid_invalid_should_return_422_json(self, monkeypatch, bad_uuid):
        # Arrange
        settings = _settings()
        monkeypatch.setattr("core.api.Settings.from_env", lambda: settings)
        mock_process = Mock()
        monkeypatch.setattr("core.api._process_transcode", mock_process)
        body = json.dumps({**VALID_REQUEST, "uuid": bad_uuid}).encode()
        headers = {"Content-Type": "application/json", **sign_request(body, settings.webhook_secret)}

        # Act
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/transcode", content=body, headers=headers)

        # Assert
        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["uuid"]
        assert error["type"] == "value_error"
        assert "uuid must be" in error["msg"]
        mock_process.assert_not_called()

    @pytest.mark.parametrize("bad_uuid", ["../../etc/passwd", VALID_UUID.replace("-", "")])
    def test_modal_transcode_when_uuid_invalid_should_return_422_json(
        self, monkeypatch, bad_uuid
    ):
        # Arrange: import the actual endpoint with an inert Modal SDK.
        modal_stub = Mock()
        modal_stub.App.return_value.function.side_effect = lambda **kwargs: lambda fn: fn
        modal_stub.asgi_app.side_effect = lambda: lambda fn: fn
        monkeypatch.setitem(sys.modules, "modal", modal_stub)
        spec = importlib.util.spec_from_file_location(
            "modal_app_under_test", Path(__file__).parents[1] / "wrappers" / "modal_app.py"
        )
        modal_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modal_app)
        mock_process = Mock()
        monkeypatch.setattr(modal_app, "process_transcode", mock_process)
        monkeypatch.setattr(sys, "path", sys.path.copy())
        settings = _settings()
        monkeypatch.setattr("core.config.Settings.from_env", lambda: settings)
        body = json.dumps({**VALID_REQUEST, "uuid": bad_uuid}).encode()
        headers = {"Content-Type": "application/json", **sign_request(body, settings.webhook_secret)}

        # Act
        with TestClient(modal_app.web(), raise_server_exceptions=False) as client:
            response = client.post("/transcode", content=body, headers=headers)

        # Assert
        assert response.status_code == 422
        error = response.json()["detail"][0]
        assert error["loc"] == ["uuid"]
        assert error["type"] == "value_error"
        assert "uuid must be" in error["msg"]
        mock_process.spawn.assert_not_called()

    def test_request_when_uuid_valid_should_pass(self):
        request = TranscodeRequest.model_validate(VALID_REQUEST)
        assert request.uuid == VALID_UUID

    @pytest.mark.parametrize(
        "bad_uuid",
        [
            "../../etc/passwd",
            "not-a-uuid",
            f"{VALID_UUID}/../escape",
            "",
            "urn:uuid:927f2e68-3342-4fcb-9812-66ce1c7ff3e8",  # non-canonical form
        ],
    )
    def test_request_when_uuid_invalid_should_reject(self, bad_uuid):
        with pytest.raises(ValidationError):
            TranscodeRequest.model_validate({**VALID_REQUEST, "uuid": bad_uuid})


class TestRunPodHandler:
    @patch("core.api._process_transcode")
    def test_handler_when_transcode_fails_should_raise(self, mock_process):
        """RunPod must mark the job FAILED — returning normally would record a
        failed transcode as COMPLETED and mislead the Video Hub poller."""
        from wrappers.runpod_handler import handler

        mock_process.return_value = {
            "uuid": VALID_UUID,
            "status": "failed",
            "error_message": "FFmpeg failed for 360p",
        }

        with pytest.raises(RuntimeError, match="FFmpeg failed for 360p"):
            handler({"input": VALID_REQUEST})

    @patch("core.api._process_transcode")
    def test_handler_when_transcode_succeeds_should_return_full_payload(
        self, mock_process
    ):
        from wrappers.runpod_handler import handler

        ready_payload = {
            "uuid": VALID_UUID,
            "status": "ready",
            "duration": 42,
            "qualities": TRANSCODE_RESULT["qualities"],
            "master_playlist": "master.m3u8",
            "thumbnail": "thumbnail.jpg",
        }
        mock_process.return_value = ready_payload

        result = handler({"input": VALID_REQUEST})

        assert result == ready_payload

    def test_handler_when_input_uuid_invalid_should_reject(self):
        from wrappers.runpod_handler import handler

        with pytest.raises(ValidationError):
            handler({"input": {**VALID_REQUEST, "uuid": "../../escape"}})
