import os

import pytest

from core.config import (
    QUALITY_PRESETS,
    Settings,
    _detected_encoder,
    _detected_preset,
    parse_bitrate,
    select_qualities,
)


class TestParseBitrate:
    def test_kilobits(self):
        assert parse_bitrate("2500k") == 2500000

    def test_megabits(self):
        assert parse_bitrate("5m") == 5000000

    def test_raw_number(self):
        assert parse_bitrate("128000") == 128000

    def test_uppercase(self):
        assert parse_bitrate("1200K") == 1200000


class TestSelectQualities:
    def test_filters_by_source_height(self):
        result = select_qualities(["1080p", "720p", "480p"], source_height=720)
        assert "1080p" not in result
        assert "720p" in result
        assert "480p" in result

    def test_all_qualities_when_source_is_large(self):
        result = select_qualities(["1080p", "720p", "480p"], source_height=1080)
        assert len(result) == 3

    def test_fallback_caps_at_source_resolution(self):
        # Source is 480p, requested only 1080p/720p — should pick 480p (not upscale).
        result = select_qualities(["1080p", "720p"], source_height=480)
        assert len(result) == 1
        assert "480p" in result

    def test_fallback_to_240p_when_source_smaller_than_all_presets(self):
        result = select_qualities(["1080p", "720p"], source_height=100)
        assert len(result) == 1
        assert "240p" in result

    def test_fallback_unknown_quality_caps_at_source(self):
        result = select_qualities(["nonexistent"], source_height=400)
        assert "360p" in result

    def test_empty_requested_picks_highest_fitting(self):
        result = select_qualities([], source_height=1080)
        assert "1080p" in result

    def test_single_quality_at_exact_height(self):
        result = select_qualities(["720p"], source_height=720)
        assert "720p" in result
        assert result["720p"]["height"] == 720


class TestQualityPresets:
    def test_all_presets_have_required_keys(self):
        required = {"width", "height", "bitrate", "audio_bitrate", "maxrate", "bufsize"}
        for name, preset in QUALITY_PRESETS.items():
            assert required.issubset(preset.keys()), f"{name} missing keys"

    def test_presets_sorted_descending_by_height(self):
        heights = [p["height"] for p in QUALITY_PRESETS.values()]
        assert heights == sorted(heights, reverse=True)

    def test_seven_presets(self):
        assert len(QUALITY_PRESETS) == 7


class TestSettings:
    def test_from_env_defaults_uses_detected_encoder(self):
        # Clear relevant env vars to test auto-detection fallback.
        env_keys = [
            "FFMPEG_ENCODER", "FFMPEG_PRESET",
            "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
            "S3_ENDPOINT", "S3_REGION", "WEBHOOK_SECRET",
        ]
        saved = {k: os.environ.pop(k, None) for k in env_keys}

        try:
            s = Settings.from_env()
            assert s.ffmpeg_encoder == _detected_encoder
            assert s.ffmpeg_preset == _detected_preset
            assert s.s3_region == "auto"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_reads_values(self):
        os.environ["FFMPEG_ENCODER"] = "libx264"
        os.environ["FFMPEG_PRESET"] = "medium"
        os.environ["WEBHOOK_SECRET"] = "test-secret"

        try:
            s = Settings.from_env()
            assert s.ffmpeg_encoder == "libx264"
            assert s.ffmpeg_preset == "medium"
            assert s.webhook_secret == "test-secret"
        finally:
            del os.environ["FFMPEG_ENCODER"]
            del os.environ["FFMPEG_PRESET"]
            del os.environ["WEBHOOK_SECRET"]

    def test_frozen(self):
        s = Settings()
        with pytest.raises(AttributeError):
            s.ffmpeg_encoder = "libx264"  # type: ignore[misc]
