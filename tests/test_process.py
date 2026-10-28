"""Tests for _process_transcode result semantics.

These pin the reliability contract:
- the development pipeline returns the result payload
- a callback delivery failure never flips a successful transcode to "failed"
- uuid is validated before it touches filesystem paths / S3 keys
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.api import TranscodeRequest, _process_transcode, app
from core.config import Settings
from core.signing import sign_request
from core.v1_admission import DEVELOPMENT_ADAPTER_ENV

VALID_UUID = "927f2e68-3342-4fcb-9812-66ce1c7ff3e8"

VALID_REQUEST = {
    "uuid": VALID_UUID,
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


@pytest.fixture(autouse=True)
def development_adapter(monkeypatch):
    monkeypatch.setenv(DEVELOPMENT_ADAPTER_ENV, "1")


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
        monkeypatch.setattr("core.api.download_from_s3", Mock(side_effect=pipeline_error))
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
    @patch("core.api.download_from_s3")
    def test_process_when_success_should_return_ready_payload(
        self, _dl, _size, _hls, _up_res, _prog, mock_send
    ):
        # Act
        payload = _process_transcode(_request(), _settings())

        # Assert: payload is the full ready result…
        assert payload["status"] == "ready"
        assert payload["uuid"] == VALID_UUID
        assert payload["duration"] == 42
        assert payload["master_playlist"] == "master.m3u8"
        assert payload["source_filesize"] == 1234
        _dl.assert_called_once()
        assert _dl.call_args.args[:3] == (_settings(), VALID_REQUEST["s3_bucket"], VALID_REQUEST["s3_original_path"])
        assert _dl.call_args.args[3].endswith(f"source-{VALID_UUID}.mp4")
        # …and the same payload was offered to the callback fast-path.
        mock_send.assert_called_once()
        assert mock_send.call_args.args[2] == payload

    @patch("core.api.send_result")
    @patch("core.api.send_progress", return_value=True)
    @patch("core.api.download_from_s3", side_effect=RuntimeError("download exploded"))
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
    @patch("core.api.download_from_s3")
    def test_process_when_callback_delivery_fails_should_still_return_ready(
        self, _dl, _size, _hls, _up_res, _prog, _send
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
    @patch("core.api.download_from_s3")
    def test_process_when_hub_unreachable_should_back_off_progress(
        self, _dl, _size, _hls, _up_res, mock_prog, _send
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
    @patch("core.api.download_from_s3")
    def test_process_when_progress_delivery_fails_should_retry_after_bounded_backoff(
        self, _dl, _size, _up_res, _send, monkeypatch, hub_recovers
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
