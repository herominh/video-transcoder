"""Small media for the encode stage, generated once per session with the local ffmpeg, and the
encode test profiles (the pilot profile at x264's fastest preset)."""

from __future__ import annotations

import dataclasses
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.profile import PILOT_PROFILE, MediaProfile
from core.source_facts import SourceFacts, parse_ffprobe_output
from tests.preflight.conftest import (
    FFMPEG_TIMEOUT_S,
    FULL_DURATION_S,
    LANDSCAPE_SIZE,
    PORTRAIT_SIZE,
    SMALL_SIZE,
    _ffmpeg,
    _h264_aac,
    _h264_only,
    _has_encoders,
    _video_input,
)

FAST_PRESET = "ultrafast"
# The pilot's preset: x264 "medium" codes B-frames, so a segment's last frames in decode order are
# not its last in presentation order.
PRODUCTION_PRESET = "medium"
ONE_SECOND_SEGMENTS = 1
MS_PER_SECOND = 1000
# Shorter than a second, and long enough that ffmpeg's playlist declares a target duration of one
# second (it rounds the longest segment to the nearest integer).
SUBSECOND_DURATION_S = 0.6
# Shorter than half a second: ffmpeg's playlist declares a target duration of 0.
BELOW_HALF_SECOND_DURATION_S = 0.4
QUADRANT_SIZE = "160x120"
QUADRANT_DURATION_S = 1
QUADRANT_FPS = 25
# Top left, top right, bottom left, bottom right: four colours far apart in RGB.
QUADRANT_COLOURS = ("red", "lime", "blue", "white")
QUADRANT_LAYOUT = "0_0|w0_0|0_h0|w0_h0"
ROTATIONS = (90, 180, 270)
# A stream whose frames grow mid-stream: the probe sees the small size, the decoder meets the large.
RESIZE_SMALL_SIZE = "64x64"
RESIZE_LARGE_SIZE = "192x192"
# The large frames start well before the admitted duration's cut.
RESIZE_SMALL_DURATION_S = 0.4
RESIZE_PART_DURATION_S = 1
SVT_AV1_FASTEST_PRESET = "12"
# 200 is not a multiple of the decoder's 64-byte stride alignment.
AT_PIXEL_CAP_SIZE = "200x100"
# A frame whose top half is red and bottom half blue, with a container crop of its bottom half.
CROP_HALF_SIZE = "320x120"
CROP_BOTTOM_ROWS = 120
CROP_COLOURS = ("red", "blue")
CROP_DURATION_S = 1
# A slide show: one picture every ten seconds, with sound throughout.
SLIDESHOW_RATE = "1/10"
SLIDESHOW_DURATION_S = 20
# 4 s, for a rendition of four 1 s segments.
SEGMENTED_DURATION_S = 4
# 1.5 s at 60 fps: ffmpeg writes "#EXT-X-TARGETDURATION:1" with "#EXTINF:1.500000,".
TIE_DURATION_S = 1.5
TIE_FPS = 60
# NTSC's 30000/1001 frame rate.
NTSC_RATE = "30000/1001"
# A dense AV1 source: 30 Matroska blocks at a declared 30 fps (1 s), each holding a temporal unit
# that carries 1,000 frames: 30,000 1080p frames for the decoder, several seconds of work for a
# second the container declares.
DENSE_SIZE = "1920x1080"
DENSE_FRAMES_PER_BLOCK = 1000
DENSE_BLOCKS = 30
IVF_HEADER_BYTES = 32
IVF_FRAME_HEADER = struct.Struct("<IQ")  # frame size, timestamp
IVF_FRAME_COUNT_OFFSET = 24
AV1_OBU_EXTENSION_FLAG = 0x04
AV1_OBU_TYPE_SHIFT = 3
AV1_OBU_TYPE_MASK = 0x0F
AV1_SEQUENCE_HEADER_AND_TEMPORAL_DELIMITER = frozenset({1, 2})
LEB128_MORE = 0x80
LEB128_VALUE = 0x7F
# A VP9 source whose second keyframe declares a frame too large to be valid at all.
OVERSIZED_PART_DURATION_S = 0.4
OVERSIZED_EDGE = 32768
VP9_SYNC_CODE = b"\x49\x83\x42"
# A profile 0 keyframe's frame_width_minus_1 and frame_height_minus_1, in bits from the frame's start.
VP9_WIDTH_BIT = 36
VP9_HEIGHT_BIT = 52
VP9_SIZE_BITS = 16
VP9_HEADER_BYTES = 9
# Matroska element IDs (EBML) on the way to a video track's pixel crop.
EBML_SEGMENT = 0x18538067
EBML_TRACKS = 0x1654AE6B
EBML_TRACK_ENTRY = 0xAE
EBML_VIDEO = 0xE0
EBML_PIXEL_CROP_BOTTOM = b"\x54\xaa"
EBML_VOID = 0xEC
EBML_MIN_VOID_BYTES = 2  # its one-byte ID and one-byte size
EBML_ONE_BYTE_SIZE = 0x80
EBML_SIZE_BITS_PER_BYTE = 7
BITS_PER_BYTE = 8
TITLE_MARKER = "PrivateTitleMarker"
LOCATION_MARKER = "+48.8577+002.2950/"
# An HLS playlist saved under a video file name: a demuxer that followed it would fetch this URL.
DISGUISED_PLAYLIST = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:3\n"
    "#EXT-X-TARGETDURATION:6\n"
    "#EXTINF:6.0,\n"
    "http://169.254.169.254/latest/meta-data/\n"
    "#EXT-X-ENDLIST\n"
)


@dataclass(frozen=True, slots=True)
class EncodeMedia:
    landscape: Path  # 320x240, 2 s, 25 fps, H.264 + AAC
    silent: Path  # 320x240, 2 s, no audio
    portrait: Path  # 240x320, with audio
    anamorphic: Path  # stored 320x240, SAR 4:3
    subsecond: Path  # 160x120, 0.6 s
    below_half_second: Path  # 160x120, 0.4 s
    quadrants: Path  # 320x240, four solid colour quadrants, no rotation
    rotated: dict[int, Path]  # the quadrants with a display rotation of 90, 180 and 270
    tagged: Path  # carries a title and a location tag
    disguised_playlist: Path  # an HLS text file named .mp4
    resized_h264: Path  # MPEG-TS, H.264 frames of 64x64, then of 192x192
    resized_av1: Path | None  # Matroska, AV1 frames of 64x64, then of 192x192; None without SVT-AV1 and dav1d
    at_pixel_cap: Path  # 200x100
    cropped: Path  # Matroska, 320x240 red over blue, its container crop removing the blue half
    slideshow: Path  # 160x120, a picture every 10 s for 20 s, with audio
    tie: Path  # 160x120, 1.5 s at 60 fps
    ntsc: Path  # 160x120, 2 s at 30000/1001 fps, with audio
    dense_av1: Path | None  # AV1 in Matroska declaring 1 s at 30 fps, carrying 30,000 frames; None without SVT/dav1d
    oversized_vp9: Path | None  # VP9 in Matroska, 64x64, then a keyframe declaring 32768x32768; None without libvpx
    # Real plain renditions ffmpeg wrote with the pilot's preset, independently of the module under
    # test, for the stand-in encoders to copy: `landscape` as one 2 s segment; 4 s of 25 fps in
    # four 1 s segments; the slideshow at 1 fps in 6 s segments.
    plain_rendition: Path
    segmented_rendition: Path
    slideshow_rendition: Path


def _quadrants(root: Path) -> Path:
    target = root / "quadrants.mp4"
    inputs = [
        arg
        for colour in QUADRANT_COLOURS
        for arg in (
            "-f", "lavfi", "-i",
            f"color=c={colour}:size={QUADRANT_SIZE}:duration={QUADRANT_DURATION_S}:rate={QUADRANT_FPS}",
        )
    ]
    _ffmpeg(
        *inputs,
        "-filter_complex", f"[0][1][2][3]xstack=inputs=4:layout={QUADRANT_LAYOUT}",
        "-c:v", "libx264", "-preset", FAST_PRESET, "-pix_fmt", "yuv420p",
        str(target),
    )
    return target


def _rotated(source: Path, root: Path, degrees: int) -> Path:
    target = root / f"quadrants_rotated_{degrees}.mp4"
    _ffmpeg("-display_rotation", str(degrees), "-i", str(source), "-c", "copy", str(target))
    return target


def _has_decoders(*names: str) -> bool:
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S
    ).stdout.decode("utf-8", "replace")
    available = {line.split()[1] for line in listing.splitlines() if len(line.split()) > 1}
    return all(name in available for name in names)


def _resized_h264(root: Path) -> Path:
    small = _h264_only(root / "resize_small.ts", RESIZE_SMALL_SIZE, RESIZE_SMALL_DURATION_S)
    large = _h264_only(root / "resize_large.ts", RESIZE_LARGE_SIZE, RESIZE_PART_DURATION_S)
    target = root / "resized_h264.ts"
    # Transport streams concatenate byte for byte.
    target.write_bytes(small.read_bytes() + large.read_bytes())
    return target


def _av1_part(root: Path, size: str) -> Path:
    target = root / f"resize_av1_{size}.mkv"
    _ffmpeg(
        *_video_input(size, RESIZE_PART_DURATION_S),
        "-c:v", "libsvtav1", "-preset", SVT_AV1_FASTEST_PRESET, "-pix_fmt", "yuv420p",
        str(target),
    )
    return target


def _resized_av1(root: Path) -> Path | None:
    if not _has_encoders("libsvtav1") or not _has_decoders("libdav1d"):
        return None
    parts = root / "resize_av1_parts.txt"
    parts.write_text(
        "".join(f"file '{_av1_part(root, size)}'\n" for size in (RESIZE_SMALL_SIZE, RESIZE_LARGE_SIZE)),
        encoding="utf-8",
    )
    target = root / "resized_av1.mkv"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(parts), "-c", "copy", str(target))
    return target


def _ebml_id(data: bytes, position: int) -> tuple[int, int]:
    """(id, length) of the EBML element ID at `position`; its length is its first byte's leading zeros + 1."""
    length = BITS_PER_BYTE + 1 - data[position].bit_length()
    return int.from_bytes(data[position: position + length], "big"), length


def _ebml_size(data: bytes, position: int) -> tuple[int, int]:
    """(size, length) of the EBML size at `position`."""
    length = BITS_PER_BYTE + 1 - data[position].bit_length()
    marker = 1 << (EBML_SIZE_BITS_PER_BYTE * length)
    return int.from_bytes(data[position: position + length], "big") & (marker - 1), length


@dataclass(frozen=True, slots=True)
class _EbmlElement:
    element_id: int
    start: int  # the element's first byte
    size_position: int
    size_length: int
    data_start: int
    end: int  # one past its last byte


def _ebml_children(data: bytes, start: int, end: int) -> list[_EbmlElement]:
    children: list[_EbmlElement] = []
    position = start
    while position < end:
        element_id, id_length = _ebml_id(data, position)
        size, size_length = _ebml_size(data, position + id_length)
        data_start = position + id_length + size_length
        children.append(
            _EbmlElement(element_id, position, position + id_length, size_length, data_start, data_start + size)
        )
        position = data_start + size
    return children


def _ebml_child(data: bytes, parent: _EbmlElement | None, wanted: int) -> _EbmlElement:
    start, end = (0, len(data)) if parent is None else (parent.data_start, parent.end)
    for child in _ebml_children(data, start, end):
        if child.element_id == wanted:
            return child
    raise ValueError("the wanted Matroska element is missing")


def _ebml_void(length: int) -> bytes:
    if not EBML_MIN_VOID_BYTES <= length < EBML_MIN_VOID_BYTES + EBML_ONE_BYTE_SIZE - 1:
        raise ValueError("a Void element of that length cannot be written with a one-byte size")
    return bytes([EBML_VOID, EBML_ONE_BYTE_SIZE | (length - EBML_MIN_VOID_BYTES)]) + bytes(
        length - EBML_MIN_VOID_BYTES
    )


def _with_pixel_crop_bottom(data: bytes, rows: int) -> bytes:
    """`data`, a Matroska file ffmpeg wrote, with PixelCropBottom = `rows` in its first video track.

    The crop takes the room of the Void element ffmpeg reserves inside the track entry, so nothing
    outside the track entry moves and every offset the file records (seek head, cues) still holds.
    """
    crop = EBML_PIXEL_CROP_BOTTOM + bytes([EBML_ONE_BYTE_SIZE | 1, rows])
    segment = _ebml_child(data, None, EBML_SEGMENT)
    tracks = _ebml_child(data, segment, EBML_TRACKS)
    entry = _ebml_child(data, tracks, EBML_TRACK_ENTRY)
    children = _ebml_children(data, entry.data_start, entry.end)
    video = next(child for child in children if child.element_id == EBML_VIDEO)
    void = next(child for child in children if child.element_id == EBML_VOID and child.end - child.start >= len(crop))
    remaining_void = void.end - void.start - len(crop)
    replacement = _ebml_void(remaining_void) if remaining_void else b""
    video_size = video.end - video.data_start + len(crop)
    marker = 1 << (EBML_SIZE_BITS_PER_BYTE * video.size_length)
    if video_size >= marker - 1:
        raise ValueError("the video element's size field cannot hold the crop")
    size_field = (marker | video_size).to_bytes(video.size_length, "big")
    buffer = bytearray(data)
    # The later edit first, so the earlier one's positions still hold.
    edits = [
        (void.start, void.end, replacement),
        (video.end, video.end, crop),
        (video.size_position, video.size_position + video.size_length, size_field),
    ]
    for start, end, content in sorted(edits, key=lambda edit: edit[0], reverse=True):
        buffer[start:end] = content
    return bytes(buffer)


def _cropped(root: Path) -> Path:
    uncropped = root / "crop_source.mkv"
    inputs = [
        arg
        for colour in CROP_COLOURS
        for arg in (
            "-f", "lavfi", "-i", f"color=c={colour}:size={CROP_HALF_SIZE}:duration={CROP_DURATION_S}:rate=25",
        )
    ]
    _ffmpeg(
        *inputs,
        "-filter_complex", "[0][1]vstack=inputs=2",
        "-c:v", "libx264", "-preset", FAST_PRESET, "-pix_fmt", "yuv420p",
        "-write_crc32", "0",
        str(uncropped),
    )
    target = root / "cropped.mkv"
    target.write_bytes(_with_pixel_crop_bottom(uncropped.read_bytes(), CROP_BOTTOM_ROWS))
    return target


def _slideshow(root: Path) -> Path:
    target = root / "slideshow.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size={SMALL_SIZE}:rate={SLIDESHOW_RATE}:duration={SLIDESHOW_DURATION_S}",
        "-f", "lavfi", "-i", f"sine=duration={SLIDESHOW_DURATION_S}",
        "-c:v", "libx264", "-preset", FAST_PRESET, "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(target),
    )
    return target


def _ntsc(root: Path) -> Path:
    target = root / "ntsc.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size={SMALL_SIZE}:rate={NTSC_RATE}:duration={FULL_DURATION_S}",
        "-f", "lavfi", "-i", f"sine=duration={FULL_DURATION_S}",
        "-c:v", "libx264", "-preset", FAST_PRESET, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(target),
    )
    return target


def _ivf_frames(data: bytes) -> list[bytes]:
    frames: list[bytes] = []
    position = IVF_HEADER_BYTES
    while position < len(data):
        size, _ = IVF_FRAME_HEADER.unpack_from(data, position)
        position += IVF_FRAME_HEADER.size
        frames.append(data[position: position + size])
        position += size
    return frames


def _write_ivf(target: Path, header: bytes, frames: list[bytes]) -> Path:
    patched = bytearray(header[:IVF_HEADER_BYTES])
    struct.pack_into("<I", patched, IVF_FRAME_COUNT_OFFSET, len(frames))
    with target.open("wb") as handle:
        handle.write(patched)
        for timestamp, frame in enumerate(frames):
            handle.write(IVF_FRAME_HEADER.pack(len(frame), timestamp))
            handle.write(frame)
    return target


def _leb128(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & LEB128_VALUE) << shift
        shift += 7
        if not byte & LEB128_MORE:
            return value, position


def _av1_frame_obus(temporal_unit: bytes) -> bytes:
    """The OBUs of a temporal unit but its temporal delimiter and sequence header."""
    kept = b""
    position = 0
    while position < len(temporal_unit):
        start = position
        obu_header = temporal_unit[position]
        position += 2 if obu_header & AV1_OBU_EXTENSION_FLAG else 1
        size, position = _leb128(temporal_unit, position)
        position += size
        if (obu_header >> AV1_OBU_TYPE_SHIFT) & AV1_OBU_TYPE_MASK not in AV1_SEQUENCE_HEADER_AND_TEMPORAL_DELIMITER:
            kept += temporal_unit[start:position]
    return kept


def _dense_av1(root: Path) -> Path | None:
    """The reviewer's dense source: every block is one temporal unit carrying many shown frames."""
    if not _has_encoders("libsvtav1") or not _has_decoders("libdav1d"):
        return None
    single = root / "dense_single.ivf"
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=gray:size={DENSE_SIZE}:rate=30",
        "-frames:v", "1", "-c:v", "libsvtav1", "-preset", SVT_AV1_FASTEST_PRESET, "-pix_fmt", "yuv420p",
        "-f", "ivf", str(single),
    )
    data = single.read_bytes()
    temporal_unit = _ivf_frames(data)[0]
    block = temporal_unit + _av1_frame_obus(temporal_unit) * (DENSE_FRAMES_PER_BLOCK - 1)
    dense = _write_ivf(root / "dense.ivf", data, [block] * DENSE_BLOCKS)
    target = root / "dense_av1.mkv"
    _ffmpeg("-i", str(dense), "-c", "copy", str(target))
    return target


def _vp9_with_frame_size(keyframe: bytes, width: int, height: int) -> bytes:
    """A profile 0 VP9 keyframe that declares another frame size."""
    if keyframe[1:4] != VP9_SYNC_CODE:
        raise ValueError("not a profile 0 VP9 keyframe")
    total_bits = VP9_HEADER_BYTES * 8
    value = int.from_bytes(keyframe[:VP9_HEADER_BYTES], "big")
    for bit, field in ((VP9_WIDTH_BIT, width - 1), (VP9_HEIGHT_BIT, height - 1)):
        shift = total_bits - bit - VP9_SIZE_BITS
        value = (value & ~(((1 << VP9_SIZE_BITS) - 1) << shift)) | (field << shift)
    return value.to_bytes(VP9_HEADER_BYTES, "big") + keyframe[VP9_HEADER_BYTES:]


def _oversized_vp9(root: Path) -> Path | None:
    """Codex's source: the probe sees 64x64 frames; a later keyframe declares 32768x32768."""
    if not _has_encoders("libvpx-vp9"):
        return None
    small = root / "oversized_part.ivf"
    _ffmpeg(
        *_video_input(RESIZE_SMALL_SIZE, OVERSIZED_PART_DURATION_S),
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
        "-auto-alt-ref", "0", "-lag-in-frames", "0", "-pix_fmt", "yuv420p",
        "-f", "ivf", str(small),
    )
    data = small.read_bytes()
    frames = _ivf_frames(data)
    oversized = [_vp9_with_frame_size(frames[0], OVERSIZED_EDGE, OVERSIZED_EDGE), *frames[1:]]
    merged = _write_ivf(root / "oversized.ivf", data, frames + oversized)
    target = root / "oversized_vp9.mkv"
    _ffmpeg("-i", str(merged), "-c", "copy", str(target))
    return target


def _segmented_source(root: Path) -> Path:
    target = root / "segmented_source.mp4"
    _ffmpeg(
        *_video_input(LANDSCAPE_SIZE, SEGMENTED_DURATION_S),
        "-f", "lavfi", "-i", f"sine=duration={SEGMENTED_DURATION_S}",
        "-c:v", "libx264", "-preset", FAST_PRESET, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(target),
    )
    return target


def _rendition_of(source: Path, target: Path, *video_filter: str, segment_s: int = 6) -> Path:
    """A plain HLS rendition in the pilot's shape: x264 medium, a keyframe opening every segment."""
    target.mkdir()
    _ffmpeg(
        "-i", str(source),
        *video_filter,
        "-c:v", "libx264", "-preset", PRODUCTION_PRESET, "-pix_fmt", "yuv420p",
        "-sc_threshold", "0", "-force_key_frames", f"expr:gte(t,n_forced*{segment_s})",
        "-c:a", "aac",
        "-f", "hls", "-hls_time", str(segment_s), "-hls_playlist_type", "vod", "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
        "-hls_segment_filename", str(target / "segment_%04d.ts"),
        str(target / "playlist.m3u8"),
    )
    return target


def _build_media(root: Path) -> EncodeMedia:
    quadrants = _quadrants(root)
    disguised = root / "disguised_playlist.mp4"
    disguised.write_text(DISGUISED_PLAYLIST, encoding="ascii")
    landscape = _h264_aac(root / "landscape.mp4", LANDSCAPE_SIZE)
    slideshow = _slideshow(root)
    return EncodeMedia(
        landscape=landscape,
        silent=_h264_only(root / "silent.mp4", LANDSCAPE_SIZE, FULL_DURATION_S),
        portrait=_h264_aac(root / "portrait.mp4", PORTRAIT_SIZE),
        anamorphic=_h264_only(root / "anamorphic.mp4", LANDSCAPE_SIZE, FULL_DURATION_S, "-vf", "setsar=4/3"),
        subsecond=_h264_only(root / "subsecond.mp4", SMALL_SIZE, SUBSECOND_DURATION_S),
        below_half_second=_h264_only(root / "below_half_second.mp4", SMALL_SIZE, BELOW_HALF_SECOND_DURATION_S),
        quadrants=quadrants,
        rotated={degrees: _rotated(quadrants, root, degrees) for degrees in ROTATIONS},
        tagged=_h264_aac(
            root / "tagged.mp4",
            SMALL_SIZE,
            "-metadata", f"title={TITLE_MARKER}",
            "-metadata", f"location={LOCATION_MARKER}",
        ),
        disguised_playlist=disguised,
        resized_h264=_resized_h264(root),
        resized_av1=_resized_av1(root),
        at_pixel_cap=_h264_only(root / "at_pixel_cap.mp4", AT_PIXEL_CAP_SIZE, RESIZE_PART_DURATION_S),
        cropped=_cropped(root),
        slideshow=slideshow,
        tie=_h264_only(root / "tie.mp4", SMALL_SIZE, TIE_DURATION_S, fps=TIE_FPS),
        ntsc=_ntsc(root),
        dense_av1=_dense_av1(root),
        oversized_vp9=_oversized_vp9(root),
        plain_rendition=_rendition_of(landscape, root / "plain_rendition"),
        segmented_rendition=_rendition_of(_segmented_source(root), root / "segmented_rendition", segment_s=1),
        slideshow_rendition=_rendition_of(slideshow, root / "slideshow_rendition", "-vf", "fps=1"),
    )


def require_media_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required to generate and inspect the media fixtures")


@pytest.fixture(scope="session")
def encode_media(tmp_path_factory: pytest.TempPathFactory) -> EncodeMedia:
    require_media_tools()
    return _build_media(tmp_path_factory.mktemp("encode-media"))


def probe_facts(path: Path) -> SourceFacts:
    """The facts the preflight would establish for a local file."""
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    )
    return parse_ffprobe_output(completed.stdout, duration_estimated=False)


def fast_profile(*, segment_duration_s: int | None = None, **encode_overrides: object) -> MediaProfile:
    """The pilot profile at x264's fastest preset; a shorter segment duration also lowers the
    longest admitted source, which must fit the segment limit."""
    encode = dataclasses.replace(PILOT_PROFILE.encode, encoder_preset=FAST_PRESET, **encode_overrides)
    source = PILOT_PROFILE.source
    if segment_duration_s is not None:
        encode = dataclasses.replace(encode, segment_duration_s=segment_duration_s)
    max_encoded_ms = encode.segment_duration_s * MS_PER_SECOND * encode.max_segments_per_rendition
    if source.max_duration_ms > max_encoded_ms:
        source = dataclasses.replace(source, max_duration_ms=max_encoded_ms)
    return dataclasses.replace(PILOT_PROFILE, source=source, encode=encode)
