from __future__ import annotations

import os
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import pytest

from core.hls_playlist import (
    MediaPlaylist,
    PlaylistInvalid,
    average_bandwidth_bps,
    parse_media_playlist,
    peak_bandwidth_bps,
    read_media_playlist,
)

KEY_URI = "enc.key"
MAX_SEGMENTS = 3600
# Written by ffmpeg 8 with the command of core/encode_command.py (1 s segments, encrypted).
REAL_ENCRYPTED = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:6\n"
    "#EXT-X-TARGETDURATION:1\n"
    "#EXT-X-MEDIA-SEQUENCE:0\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    "#EXT-X-INDEPENDENT-SEGMENTS\n"
    '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",IV=0x00000000000000000000000000000000\n'
    "#EXTINF:1.000000,\n"
    "segment_0000.ts\n"
    '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",IV=0x00000000000000000000000000000001\n'
    "#EXTINF:1.000000,\n"
    "segment_0001.ts\n"
    '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",IV=0x00000000000000000000000000000002\n'
    "#EXTINF:1.000000,\n"
    "segment_0002.ts\n"
    "#EXT-X-ENDLIST\n"
)
# Written by ffmpeg 8 for a plain rendition of 2.6 s in 1 s segments.
REAL_PLAIN = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:6\n"
    "#EXT-X-TARGETDURATION:1\n"
    "#EXT-X-MEDIA-SEQUENCE:0\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    "#EXT-X-INDEPENDENT-SEGMENTS\n"
    "#EXTINF:1.000000,\n"
    "segment_0000.ts\n"
    "#EXTINF:1.000000,\n"
    "segment_0001.ts\n"
    "#EXTINF:0.600000,\n"
    "segment_0002.ts\n"
    "#EXT-X-ENDLIST\n"
)
# Written by ffmpeg 8 for a plain rendition of 0.4 s: the nearest integer to 0.4 is 0.
REAL_BELOW_HALF_SECOND = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:6\n"
    "#EXT-X-TARGETDURATION:0\n"
    "#EXT-X-MEDIA-SEQUENCE:0\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    "#EXT-X-INDEPENDENT-SEGMENTS\n"
    "#EXTINF:0.400000,\n"
    "segment_0000.ts\n"
    "#EXT-X-ENDLIST\n"
)
SECOND_IV = "IV=0x00000000000000000000000000000001"
FIRST_IV = "IV=0x00000000000000000000000000000000"
SECOND_KEY_LINE = '#EXT-X-KEY:METHOD=AES-128,URI="enc.key",IV=0x00000000000000000000000000000001\n'
PLANTED_URL_HOST = "attacker.example"


def _parse(
    text: str | bytes, *, encrypted: bool, key_uri: str = KEY_URI, max_segments: int = MAX_SEGMENTS
) -> MediaPlaylist:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return parse_media_playlist(data, encrypted=encrypted, key_uri=key_uri, max_segments=max_segments)


def test_parse_media_playlist_when_given_an_encrypted_playlist_as_ffmpeg_writes_it_should_list_its_segments() -> None:
    # Arrange
    text = REAL_ENCRYPTED

    # Act
    playlist = _parse(text, encrypted=True)

    # Assert
    assert playlist.target_duration_s == 1
    assert [segment.name for segment in playlist.segments] == ["segment_0000.ts", "segment_0001.ts", "segment_0002.ts"]
    assert [segment.duration for segment in playlist.segments] == [Fraction(1)] * 3


def test_parse_media_playlist_when_given_a_plain_playlist_as_ffmpeg_writes_it_should_keep_exact_durations() -> None:
    # Arrange
    text = REAL_PLAIN

    # Act
    playlist = _parse(text, encrypted=False)

    # Assert
    assert [segment.duration for segment in playlist.segments] == [Fraction(1), Fraction(1), Fraction(3, 5)]
    assert playlist.segments[-1].name == "segment_0002.ts"


def test_parse_media_playlist_when_a_rendition_shorter_than_half_a_second_declares_target_0_should_accept_it() -> None:
    # Arrange
    text = REAL_BELOW_HALF_SECOND

    # Act
    playlist = _parse(text, encrypted=False)

    # Assert
    assert playlist.target_duration_s == 0
    assert [segment.duration for segment in playlist.segments] == [Fraction(2, 5)]


def test_parse_media_playlist_when_target_0_meets_a_segment_rounding_to_1_should_refuse_it() -> None:
    # Arrange
    text = REAL_BELOW_HALF_SECOND.replace("#EXTINF:0.400000,", "#EXTINF:0.600000,")

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=False)

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_parse_media_playlist_when_the_target_duration_is_negative_should_refuse_it() -> None:
    # Arrange
    text = REAL_BELOW_HALF_SECOND.replace("#EXT-X-TARGETDURATION:0", "#EXT-X-TARGETDURATION:-1")

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=False)

    # Assert
    assert raised.value.code == "playlist_invalid"


# Written by ffmpeg 8 for 1.5 s at 60 fps: its target is the nearest integer to a duration just
# under 1.5, while the six-decimal EXTINF reads exactly 1.5.
REAL_TIE = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:6\n"
    "#EXT-X-TARGETDURATION:1\n"
    "#EXT-X-MEDIA-SEQUENCE:0\n"
    "#EXT-X-PLAYLIST-TYPE:VOD\n"
    "#EXT-X-INDEPENDENT-SEGMENTS\n"
    "#EXTINF:1.500000,\n"
    "segment_0000.ts\n"
    "#EXT-X-ENDLIST\n"
)


def test_parse_media_playlist_when_a_segment_lasts_its_target_plus_exactly_half_a_second_should_accept_it() -> None:
    # Arrange
    text = REAL_TIE

    # Act
    playlist = _parse(text, encrypted=False)

    # Assert
    assert [segment.duration for segment in playlist.segments] == [Fraction(3, 2)]


def test_parse_media_playlist_when_a_segment_lasts_beyond_its_target_plus_half_a_second_should_refuse_it() -> None:
    # Arrange
    text = REAL_TIE.replace("#EXTINF:1.500000,", "#EXTINF:1.500001,")

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=False)

    # Assert
    assert raised.value.code == "playlist_invalid"


def _swap_segment_names(text: str) -> str:
    return text.replace("segment_0000.ts", "#SWAP#").replace("segment_0001.ts", "segment_0000.ts").replace(
        "#SWAP#", "segment_0001.ts"
    )


ENCRYPTED_REFUSALS: dict[str, Callable[[str], str]] = {
    "missing ENDLIST": lambda text: text.replace("#EXT-X-ENDLIST\n", ""),
    "wrong IV": lambda text: text.replace(SECOND_IV, FIRST_IV),
    "missing key line": lambda text: text.replace(SECOND_KEY_LINE, ""),
    "different key URI": lambda text: text.replace('URI="enc.key"', 'URI="https://attacker.example/k"'),
    "out-of-order segments": _swap_segment_names,
    "renamed segment": lambda text: text.replace("segment_0001.ts", "../segment_0001.ts"),
    "segment named by URL": lambda text: text.replace("segment_0001.ts", f"http://{PLANTED_URL_HOST}/s.ts"),
    "unknown tag between segments": lambda text: text.replace(
        SECOND_KEY_LINE, "#EXT-X-DISCONTINUITY\n" + SECOND_KEY_LINE
    ),
    "unknown tag in the header": lambda text: text.replace(
        "#EXT-X-VERSION:6\n", f"#EXT-X-VERSION:6\n#EXT-X-SESSION-DATA:URI=\"http://{PLANTED_URL_HOST}/\"\n"
    ),
    "repeated header tag": lambda text: text.replace("#EXT-X-VERSION:6\n", "#EXT-X-VERSION:6\n#EXT-X-VERSION:6\n"),
    "missing header tag": lambda text: text.replace("#EXT-X-PLAYLIST-TYPE:VOD\n", ""),
    "not starting with EXTM3U": lambda text: text.replace("#EXTM3U\n", ""),
    "blank line": lambda text: text.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts\n", "#EXTINF:1.000000,\n\nsegment_0001.ts\n"
    ),
    "CRLF line breaks": lambda text: text.replace("\n", "\r\n"),
    "non-ASCII byte": lambda text: text.replace("segment_0002.ts", "segment_0002.té"),
    "no final line break": lambda text: text[:-1],
    "two final line breaks": lambda text: text + "\n",
    "zero duration": lambda text: text.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts", "#EXTINF:0.000000,\nsegment_0001.ts"
    ),
    "duration above the target": lambda text: text.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts", "#EXTINF:1.600000,\nsegment_0001.ts"
    ),
    "duration without decimals": lambda text: text.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts", "#EXTINF:1,\nsegment_0001.ts"
    ),
    "duration longer than 20 characters": lambda text: text.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts", "#EXTINF:1.0000000000000000000,\nsegment_0001.ts"
    ),
    "media sequence other than 0": lambda text: text.replace("#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-MEDIA-SEQUENCE:1"),
    "target duration of zero with one-second segments": lambda text: text.replace(
        "#EXT-X-TARGETDURATION:1", "#EXT-X-TARGETDURATION:0"
    ),
    "event playlist": lambda text: text.replace("#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-PLAYLIST-TYPE:EVENT"),
    "line after ENDLIST": lambda text: text + "segment_0003.ts\n",
    "no segment": lambda text: text.split('#EXT-X-KEY')[0] + "#EXT-X-ENDLIST\n",
}


@pytest.mark.parametrize("mutate", ENCRYPTED_REFUSALS.values(), ids=ENCRYPTED_REFUSALS.keys())
def test_parse_media_playlist_when_an_encrypted_playlist_departs_from_ffmpegs_grammar_should_refuse_it(
    mutate: Callable[[str], str],
) -> None:
    # Arrange
    data = mutate(REAL_ENCRYPTED).encode("utf-8")

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(data, encrypted=True)

    # Assert
    assert raised.value.code == "playlist_invalid"
    assert PLANTED_URL_HOST not in raised.value.detail


def test_parse_media_playlist_when_a_plain_playlist_holds_a_key_line_should_refuse_it() -> None:
    # Arrange
    text = REAL_PLAIN.replace(
        "#EXTINF:1.000000,\nsegment_0001.ts", SECOND_KEY_LINE + "#EXTINF:1.000000,\nsegment_0001.ts"
    )

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=False)

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_parse_media_playlist_when_an_encrypted_playlist_is_read_as_plain_should_refuse_it() -> None:
    # Arrange
    text = REAL_ENCRYPTED

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=False)

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_parse_media_playlist_when_a_plain_playlist_is_read_as_encrypted_should_refuse_it() -> None:
    # Arrange
    text = REAL_PLAIN

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=True)

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_parse_media_playlist_when_the_key_uri_differs_from_the_expected_one_should_refuse_it() -> None:
    # Arrange
    text = REAL_ENCRYPTED

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=True, key_uri="other.key")

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_parse_media_playlist_when_it_lists_more_segments_than_allowed_should_refuse_with_too_many_segments() -> None:
    # Arrange
    text = REAL_ENCRYPTED

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        _parse(text, encrypted=True, max_segments=2)

    # Assert
    assert raised.value.code == "too_many_segments"


def test_parse_media_playlist_when_it_lists_exactly_the_allowed_segments_should_accept_it() -> None:
    # Arrange
    text = REAL_ENCRYPTED

    # Act
    playlist = _parse(text, encrypted=True, max_segments=3)

    # Assert
    assert len(playlist.segments) == 3


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="ascii")
    return path


def test_read_media_playlist_when_the_file_exceeds_max_bytes_should_refuse_with_playlist_too_large(
    tmp_path: Path,
) -> None:
    # Arrange
    path = _write(tmp_path / "playlist.m3u8", REAL_PLAIN)
    max_bytes = len(REAL_PLAIN) - 1

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        read_media_playlist(path, max_bytes=max_bytes, encrypted=False, key_uri=KEY_URI, max_segments=MAX_SEGMENTS)

    # Assert
    assert raised.value.code == "playlist_too_large"


def test_read_media_playlist_when_the_file_fits_max_bytes_exactly_should_parse_it(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path / "playlist.m3u8", REAL_PLAIN)

    # Act
    playlist = read_media_playlist(
        path, max_bytes=len(REAL_PLAIN), encrypted=False, key_uri=KEY_URI, max_segments=MAX_SEGMENTS
    )

    # Assert
    assert len(playlist.segments) == 3


@pytest.mark.skipif(os.name != "posix", reason="symbolic links are created the POSIX way")
def test_read_media_playlist_when_the_playlist_is_a_symlink_should_refuse_it(tmp_path: Path) -> None:
    # Arrange
    target = _write(tmp_path / "elsewhere.m3u8", REAL_PLAIN)
    link = tmp_path / "playlist.m3u8"
    link.symlink_to(target)

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        read_media_playlist(
            link, max_bytes=len(REAL_PLAIN), encrypted=False, key_uri=KEY_URI, max_segments=MAX_SEGMENTS
        )

    # Assert
    assert raised.value.code == "playlist_invalid"


@pytest.mark.parametrize("make", ["missing", "directory", "fifo"])
def test_read_media_playlist_when_the_playlist_is_not_a_regular_file_should_refuse_it(
    make: str, tmp_path: Path
) -> None:
    # Arrange
    path = tmp_path / "playlist.m3u8"
    if make == "directory":
        path.mkdir()
    elif make == "fifo":
        os.mkfifo(path)

    # Act
    with pytest.raises(PlaylistInvalid) as raised:
        read_media_playlist(
            path, max_bytes=len(REAL_PLAIN), encrypted=False, key_uri=KEY_URI, max_segments=MAX_SEGMENTS
        )

    # Assert
    assert raised.value.code == "playlist_invalid"


def test_average_bandwidth_bps_when_measured_should_be_every_bit_over_the_whole_duration() -> None:
    # Arrange: 1.5 MB over 12 s
    segments = [(Fraction(6), 750_000), (Fraction(6), 750_000)]

    # Act
    average = average_bandwidth_bps(segments)

    # Assert
    assert average == 1_000_000


def test_average_bandwidth_bps_when_the_rate_is_fractional_should_round_up() -> None:
    # Arrange: 8 bits over 3 s is 2.67 bps
    segments = [(Fraction(3), 1)]

    # Act
    average = average_bandwidth_bps(segments)

    # Assert
    assert average == 3


def test_average_bandwidth_bps_when_the_duration_is_a_third_of_a_second_should_compute_exactly() -> None:
    # Arrange: 8 bits over 1/3 s is exactly 24 bps (binary floating point would round it up to 25)
    segments = [(Fraction(1, 3), 1)]

    # Act
    average = average_bandwidth_bps(segments)

    # Assert
    assert average == 24


def test_peak_bandwidth_bps_when_a_segment_lasts_exactly_half_the_target_should_count_it() -> None:
    # Arrange: target 2 s, so windows of 1 to 3 s; the 1 s segment is a window, the 4 s one is not
    segments = [(Fraction(1), 2_000), (Fraction(4), 100)]

    # Act
    peak = peak_bandwidth_bps(segments, 2)

    # Assert
    assert peak == 16_000


def test_peak_bandwidth_bps_when_a_segment_lasts_exactly_one_and_a_half_targets_should_count_it() -> None:
    # Arrange: target 1 s, so windows of 0.5 to 1.5 s; only the 1.5 s segment is one
    segments = [(Fraction(3, 2), 3_000), (Fraction(8, 5), 100)]

    # Act
    peak = peak_bandwidth_bps(segments, 1)

    # Assert
    assert peak == 16_000


def test_peak_bandwidth_bps_when_the_last_segment_is_short_and_dense_should_not_let_it_dominate() -> None:
    # Arrange: target 6 s, windows of 3 to 9 s; the 0.5 s tail alone is no window, only with the
    # segment before it: 8 * 700,000 bytes / 6.5 s = 861,538.46 bps
    segments = [(Fraction(6), 600_000), (Fraction(6), 600_000), (Fraction(1, 2), 100_000)]

    # Act
    peak = peak_bandwidth_bps(segments, 6)

    # Assert
    assert peak == 861_539


def test_peak_bandwidth_bps_when_the_rendition_is_shorter_than_half_the_target_should_be_the_average() -> None:
    # Arrange: target 6 s needs windows of at least 3 s; the whole rendition lasts 2 s
    segments = [(Fraction(1), 1_000), (Fraction(1), 3_000)]

    # Act
    peak = peak_bandwidth_bps(segments, 6)

    # Assert
    assert peak == 16_000
    assert peak == average_bandwidth_bps(segments)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(Fraction(0), 1)],
        [(Fraction(-1), 1)],
        [(1.5, 1)],
        [(Fraction(1), -1)],
        [(Fraction(1), True)],
        [(Fraction(1),)],
    ],
)
def test_bandwidth_when_a_measurement_is_invalid_should_raise(segments: list[tuple[object, ...]]) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        average_bandwidth_bps(segments)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        peak_bandwidth_bps(segments, 6)  # type: ignore[arg-type]


def test_peak_bandwidth_bps_when_the_target_is_0_should_be_the_average() -> None:
    # Arrange: no run of segments can last between 0 and 0 seconds
    segments = [(Fraction(2, 5), 1_000)]

    # Act
    peak = peak_bandwidth_bps(segments, 0)

    # Assert
    assert peak == 20_000
    assert peak == average_bandwidth_bps(segments)


@pytest.mark.parametrize("target", [-1, True, 1.5])
def test_peak_bandwidth_bps_when_the_target_is_not_an_int_of_at_least_0_should_raise(target: int) -> None:
    # Arrange
    segments = [(Fraction(1), 1)]

    # Act / Assert
    with pytest.raises(ValueError):
        peak_bandwidth_bps(segments, target)
