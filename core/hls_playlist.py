"""Read back a media playlist the encoder wrote, and measure a rendition's bandwidth.

The parser accepts exactly the grammar ffmpeg 8 writes with the command of `core/encode_command.py`
and refuses everything else: whatever the encoder's output turns out to hold, nothing reaches the
generation manifest that the worker did not expect to the byte.
"""

from __future__ import annotations

import errno
import math
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from core.encode_command import SEGMENT_NAME_PATTERN
from core.failure import CODE_PATTERN, make_detail

PLAYLIST_TOO_LARGE = "playlist_too_large"
TOO_MANY_SEGMENTS = "too_many_segments"
PLAYLIST_INVALID = "playlist_invalid"

LINE_SEPARATOR = "\n"
FORBIDDEN_CHARACTERS = ("\r",)
EXTM3U = "#EXTM3U"
ENDLIST = "#EXT-X-ENDLIST"
KEY_TAG_PREFIX = "#EXT-X-KEY:"
EXTINF_PREFIX = "#EXTINF:"
EXTINF_SUFFIX = ","
# The decimal ffmpeg prints with "%f"; longer text is no duration it writes.
EXTINF_MAX_CHARS = 20
EXTINF_PATTERN = re.compile(r"#EXTINF:([0-9]+\.[0-9]+),")
# ffmpeg prints these integers with "%d": no sign, no leading zero, and never more digits than this.
INT_PATTERN_TEXT = r"(0|[1-9][0-9]{0,8})"
HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "version": re.compile(rf"#EXT-X-VERSION:{INT_PATTERN_TEXT}"),
    "target_duration": re.compile(rf"#EXT-X-TARGETDURATION:{INT_PATTERN_TEXT}"),
    "media_sequence": re.compile(r"#EXT-X-MEDIA-SEQUENCE:0"),
    "playlist_type": re.compile(r"#EXT-X-PLAYLIST-TYPE:VOD"),
    "independent_segments": re.compile(r"#EXT-X-INDEPENDENT-SEGMENTS"),
}
# ffmpeg declares a target of 0 for a rendition shorter than half a second (the nearest integer to
# its one segment); every segment must then also last at most half a second, which the EXTINF
# check enforces.
MIN_TARGET_DURATION_S = 0
EXTINF_TIE = Fraction(1, 2)
IV_HEX_DIGITS = 32
# The URI is a quoted string on the key line.
KEY_URI_FORBIDDEN = '"'
# RFC 8216 section 4.3.4.2: the peak is measured over windows of 0.5 to 1.5 times the target duration.
PEAK_WINDOW_MIN = Fraction(1, 2)
PEAK_WINDOW_MAX = Fraction(3, 2)
BITS_PER_BYTE = 8
READ_CHUNK_BYTES = 64 * 1024
# An open that refuses a missing file, a symbolic link (ELOOP on Linux and macOS, EMLINK on FreeBSD),
# a socket (ENXIO) or a path through a non-directory: the playlist is not there as a regular file.
NOT_A_PLAYLIST_ERRNOS = frozenset({errno.ENOENT, errno.ELOOP, errno.EMLINK, errno.ENXIO, errno.ENOTDIR})
# O_NONBLOCK keeps a FIFO planted in place of the playlist from blocking the open.
OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


class PlaylistInvalid(Exception):
    """A media playlist the worker refuses. The detail is our own words, never the playlist's text."""

    def __init__(self, code: str, detail: str) -> None:
        if not isinstance(code, str) or CODE_PATTERN.fullmatch(code) is None:
            raise ValueError(f"invalid diagnostic code: {code!r}")
        safe_detail = make_detail(detail)
        super().__init__(f"{code}: {safe_detail}")
        self.code = code
        self.detail = safe_detail


@dataclass(frozen=True, slots=True)
class PlaylistSegment:
    name: str  # "segment_0000.ts"
    duration: Fraction  # EXTINF, seconds, > 0


@dataclass(frozen=True, slots=True)
class MediaPlaylist:
    target_duration_s: int
    segments: tuple[PlaylistSegment, ...]


def _invalid(detail: str) -> PlaylistInvalid:
    return PlaylistInvalid(PLAYLIST_INVALID, detail)


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an int of at least 1, got {value!r}")


def _require_key_uri(key_uri: object) -> None:
    if not isinstance(key_uri, str) or not key_uri or not key_uri.isascii() or not key_uri.isprintable():
        raise ValueError("key_uri must be a non-empty printable ASCII string")
    if KEY_URI_FORBIDDEN in key_uri:
        raise ValueError("key_uri holds a character the key line cannot carry")


def _lines(data: bytes) -> list[str]:
    if not data.isascii():
        raise _invalid("the playlist holds a byte outside ASCII")
    text = data.decode("ascii")
    if any(character in text for character in FORBIDDEN_CHARACTERS):
        raise _invalid("the playlist holds a carriage return")
    if not text.endswith(LINE_SEPARATOR):
        raise _invalid("the playlist does not end with a line break")
    lines = text[: -len(LINE_SEPARATOR)].split(LINE_SEPARATOR)
    if any(line == "" for line in lines):
        raise _invalid("the playlist holds a blank line")
    return lines


def _header_tag(line: str) -> tuple[str, re.Match[str]]:
    for name, pattern in HEADER_PATTERNS.items():
        match = pattern.fullmatch(line)
        if match is not None:
            return name, match
    raise _invalid("the playlist header holds an unexpected line")


def _header(lines: Sequence[str], first_segment_line: int) -> int:
    """The target duration, after checking every header tag appears exactly once."""
    seen: dict[str, re.Match[str]] = {}
    for line in lines[1:first_segment_line]:
        name, match = _header_tag(line)
        if name in seen:
            raise _invalid(f"the playlist header repeats its {name} tag")
        seen[name] = match
    missing = [name for name in HEADER_PATTERNS if name not in seen]
    if missing:
        raise _invalid(f"the playlist header lacks its {missing[0]} tag")
    # The pattern takes no sign: a negative target is an unexpected line.
    return int(seen["target_duration"].group(1))


def _key_line(key_uri: str, sequence_number: int) -> str:
    return f'{KEY_TAG_PREFIX}METHOD=AES-128,URI="{key_uri}",IV=0x{sequence_number:0{IV_HEX_DIGITS}x}'


def _segment_name(sequence_number: int) -> str:
    return SEGMENT_NAME_PATTERN % sequence_number


def _segment_duration(line: str, target_duration_s: int) -> Fraction:
    match = EXTINF_PATTERN.fullmatch(line)
    if match is None or len(match.group(1)) > EXTINF_MAX_CHARS:
        raise _invalid("a segment's duration line is malformed")
    duration = Fraction(match.group(1))
    if duration <= 0:
        raise _invalid("a segment's duration is not positive")
    # RFC 8216: an EXTINF rounded to the nearest integer is at most the target. ffmpeg derives the
    # target with lrint() from the unprinted duration, which can fall just under x.5 while the
    # six-decimal EXTINF reads x.500000, so a tie is accepted: only beyond target + 1/2 is refused.
    if duration > target_duration_s + EXTINF_TIE:
        raise _invalid("a segment's duration exceeds the target duration")
    return duration


def _first_segment_line(lines: Sequence[str], encrypted: bool) -> int:
    marker = KEY_TAG_PREFIX if encrypted else EXTINF_PREFIX
    for position, line in enumerate(lines):
        if line.startswith(marker):
            return position
    raise _invalid("the playlist lists no segment")


def parse_media_playlist(data: bytes, *, encrypted: bool, key_uri: str, max_segments: int) -> MediaPlaylist:
    """Parse a media playlist in exactly the shape ffmpeg 8 writes it; raise PlaylistInvalid otherwise.

    With `encrypted`, every segment carries its own AES-128 key line naming `key_uri` and its media
    sequence number as IV; without it, no key line may appear anywhere.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(encrypted, bool):
        raise TypeError("encrypted must be a bool")
    _require_key_uri(key_uri)
    _require_positive_int("max_segments", max_segments)

    lines = _lines(data)
    if lines[0] != EXTM3U:
        raise _invalid("the playlist does not start with its EXTM3U line")
    if lines[-1] != ENDLIST:
        raise _invalid("the playlist does not end with its ENDLIST line")
    if not encrypted and any(line.startswith(KEY_TAG_PREFIX) for line in lines):
        raise _invalid("a plain playlist holds a key line")
    first_segment_line = _first_segment_line(lines, encrypted)
    target_duration_s = _header(lines, first_segment_line)

    body = lines[first_segment_line:-1]
    lines_per_segment = 3 if encrypted else 2
    if len(body) % lines_per_segment != 0:
        raise _invalid("the playlist's segment lines are incomplete")
    segments: list[PlaylistSegment] = []
    for start in range(0, len(body), lines_per_segment):
        sequence_number = len(segments)
        entry = body[start: start + lines_per_segment]
        if encrypted and entry[0] != _key_line(key_uri, sequence_number):
            raise _invalid(f"the key line of segment {sequence_number} is not the expected one")
        duration = _segment_duration(entry[-2], target_duration_s)
        if entry[-1] != _segment_name(sequence_number):
            raise _invalid(f"segment {sequence_number} is not named in sequence")
        if sequence_number == max_segments:
            raise PlaylistInvalid(TOO_MANY_SEGMENTS, f"the playlist lists more than {max_segments} segments")
        segments.append(PlaylistSegment(name=entry[-1], duration=duration))
    return MediaPlaylist(target_duration_s=target_duration_s, segments=tuple(segments))


def _read_capped(descriptor: int, max_bytes: int) -> bytes:
    data = bytearray()
    while len(data) <= max_bytes:
        chunk = os.read(descriptor, min(READ_CHUNK_BYTES, max_bytes + 1 - len(data)))
        if not chunk:
            break
        data += chunk
    return bytes(data)


def read_media_playlist(
    path: Path, *, max_bytes: int, encrypted: bool, key_uri: str, max_segments: int
) -> MediaPlaylist:
    """Read at most `max_bytes` of the regular file at `path` (never through a symbolic link) and
    parse it. Raises PlaylistInvalid for a missing, non-regular, oversized or malformed playlist,
    and OSError when the file cannot be read."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("path must be an absolute Path")
    _require_positive_int("max_bytes", max_bytes)
    try:
        descriptor = os.open(path, OPEN_FLAGS)
    except OSError as error:
        if error.errno in NOT_A_PLAYLIST_ERRNOS:
            raise _invalid("the playlist is missing or is a symbolic link") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _invalid("the playlist is not a regular file")
        data = _read_capped(descriptor, max_bytes)
    finally:
        os.close(descriptor)
    if len(data) > max_bytes:
        raise PlaylistInvalid(PLAYLIST_TOO_LARGE, f"the playlist exceeds {max_bytes} bytes")
    return parse_media_playlist(data, encrypted=encrypted, key_uri=key_uri, max_segments=max_segments)


def _validated_measurements(segments: object) -> list[tuple[Fraction, int]]:
    if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence) or not segments:
        raise ValueError("segments must be a non-empty sequence of (duration, size) pairs")
    measurements: list[tuple[Fraction, int]] = []
    for entry in segments:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("each segment must be a (duration, size) pair")
        duration, size = entry
        if not isinstance(duration, Fraction) or duration <= 0:
            raise ValueError("a segment duration must be a positive Fraction")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("a segment size must be an int of at least 0")
        measurements.append((duration, size))
    return measurements


def _bits_per_second(total_bytes: int, total_seconds: Fraction) -> int:
    return math.ceil(BITS_PER_BYTE * total_bytes / total_seconds)


def average_bandwidth_bps(segments: Sequence[tuple[Fraction, int]]) -> int:
    """Every byte of the rendition over its whole duration, rounded up (RFC 8216 AVERAGE-BANDWIDTH)."""
    measurements = _validated_measurements(segments)
    total_seconds = sum((duration for duration, _ in measurements), Fraction(0))
    return _bits_per_second(sum(size for _, size in measurements), total_seconds)


def peak_bandwidth_bps(segments: Sequence[tuple[Fraction, int]], target_duration_s: int) -> int:
    """The largest rate over any run of contiguous segments lasting 0.5 to 1.5 times the target
    duration, rounded up (RFC 8216 BANDWIDTH); the average when no run lasts that long, which is
    always so for a target of 0."""
    measurements = _validated_measurements(segments)
    if (
        not isinstance(target_duration_s, int)
        or isinstance(target_duration_s, bool)
        or target_duration_s < MIN_TARGET_DURATION_S
    ):
        raise ValueError(f"target_duration_s must be an int of at least 0, got {target_duration_s!r}")
    shortest = PEAK_WINDOW_MIN * target_duration_s
    longest = PEAK_WINDOW_MAX * target_duration_s
    peak: Fraction | None = None
    for start in range(len(measurements)):
        window_seconds = Fraction(0)
        window_bytes = 0
        for duration, size in measurements[start:]:
            window_seconds += duration
            window_bytes += size
            if window_seconds > longest:
                break
            if window_seconds >= shortest:
                rate = BITS_PER_BYTE * window_bytes / window_seconds
                peak = rate if peak is None or rate > peak else peak
    if peak is None:
        return average_bandwidth_bps(measurements)
    return math.ceil(peak)
