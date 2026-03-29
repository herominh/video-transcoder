import os
import tempfile

import pytest

from core.config import Settings
from core.transcoder import (
    _parse_fps,
    create_key_info_file,
    generate_master_playlist,
)


class TestParseFps:
    def test_fraction(self):
        assert _parse_fps("30/1") == 30.0

    def test_ntsc(self):
        assert abs(_parse_fps("30000/1001") - 29.97) < 0.01

    def test_decimal(self):
        assert _parse_fps("25.0") == 25.0

    def test_zero_denominator_falls_back(self):
        assert _parse_fps("30/0") == 30.0

    def test_empty_string(self):
        assert _parse_fps("") == 30.0


class TestGenerateMasterPlaylist:
    def test_sorted_by_bitrate_descending(self):
        qualities = {
            "480p": {
                "name": "480p",
                "width": 854,
                "height": 480,
                "bitrate": 1200000,
                "playlist": "480p/playlist.m3u8",
            },
            "720p": {
                "name": "720p",
                "width": 1280,
                "height": 720,
                "bitrate": 2500000,
                "playlist": "720p/playlist.m3u8",
            },
        }

        result = generate_master_playlist(qualities)

        assert result.startswith("#EXTM3U\n")
        lines = result.strip().split("\n")

        # 720p (higher bitrate) should come before 480p.
        stream_inf_lines = [l for l in lines if l.startswith("#EXT-X-STREAM-INF")]
        assert "BANDWIDTH=2500000" in stream_inf_lines[0]
        assert "BANDWIDTH=1200000" in stream_inf_lines[1]

    def test_contains_resolution_and_name(self):
        qualities = {
            "360p": {
                "name": "360p",
                "width": 640,
                "height": 360,
                "bitrate": 600000,
                "playlist": "360p/playlist.m3u8",
            },
        }

        result = generate_master_playlist(qualities)
        assert 'RESOLUTION=640x360' in result
        assert 'NAME="360p"' in result
        assert "360p/playlist.m3u8" in result


class TestCreateKeyInfoFile:
    def test_creates_key_and_keyinfo(self):
        key_hex = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_key_info_file(tmpdir, key_hex)

            # Check keyinfo file exists and has correct content.
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "enc.key" in content

            # Check enc.key has correct raw bytes.
            key_path = os.path.join(tmpdir, "enc.key")
            assert os.path.exists(key_path)
            with open(key_path, "rb") as f:
                key_bytes = f.read()
            assert len(key_bytes) == 16
            assert key_bytes == bytes.fromhex(key_hex)

    def test_invalid_hex_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                create_key_info_file(tmpdir, "not-valid-hex")
