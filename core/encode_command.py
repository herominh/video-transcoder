"""The exact ffmpeg command line of one rendition encode: isolated, and bounded on the decoded stream.

The preflight's facts are what the container and the probe window declare. This command makes the
encode hold to them whatever the decoded stream turns out to be:

- isolation: the input may only be read as a local file, through the one demuxer the preflight
  saw; the output may only be written to local files (through `crypto` when encrypted); the child
  sees a cleared environment and no stdin;
- duration: the output is cut at the admitted duration (`-t`), with a frame count belt;
- frame rate: the `fps` filter makes the output constant-rate at the admitted rate, held between
  the profile's floor and ceiling, so a stream denser than it declares cannot multiply the
  encoder's work. The decoder still decodes every frame it is given, and no count of packets
  bounds that (an AV1 temporal unit or a VP9 superframe carries many frames): the rendition's
  time budget (`core/encode.py`) is what bounds decoding;
- decoded pixels: the decoder refuses any frame above the profile's pixel cap, plus the slack its
  buffer alignment needs (`-max_pixels`);
- rotation: autorotation is off and exactly the admitted quarter turn is applied, so an
  orientation carried only in the coded stream is ignored, as the preflight's contract says;
- only the admitted streams are mapped; metadata and chapters are dropped (a phone's location
  tag never reaches the output).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from core.profile import ALLOW_LIST_ENTRY_PATTERN, MediaProfile
from core.renditions import RenditionPlan
from core.source_facts import SourceFacts

# The child's whole environment, as for the preflight's ffprobe.
FFMPEG_ENV = {"LC_ALL": "C", "PATH": os.defpath}
INPUT_PROTOCOLS = "file"
OUTPUT_PROTOCOLS_PLAIN = "file"
OUTPUT_PROTOCOLS_ENCRYPTED = "crypto,file"
FILE_URL_PREFIX = "file:"
PLAYLIST_NAME = "playlist.m3u8"
SEGMENT_NAME_PATTERN = "segment_%04d.ts"
HLS_SEGMENT_TYPE = "mpegts"
HLS_FLAGS_PLAIN = "independent_segments"
HLS_FLAGS_ENCRYPTED = "independent_segments+periodic_rekey"
FIRST_SEGMENT_NUMBER = 0
LOG_LEVEL = "warning"
DEMUXER_NAME_SEPARATOR = ","
MS_PER_SECOND = 1000
# The largest buffer stride alignment libavcodec uses (AVX-512 builds); arm64 uses 16.
DECODER_ALIGNMENT = 64
# The rotation filters ffmpeg's own autorotation inserts for an exact quarter turn of the display
# matrix, keyed by the preflight's rotation_degrees.
ROTATION_FILTERS: dict[int, tuple[str, ...]] = {
    0: (),
    90: ("transpose=cclock",),
    180: ("hflip", "vflip"),
    270: ("transpose=clock",),
}
# Characters that would turn a path into something else on this command line: a protocol prefix,
# an image-sequence pattern, or a line of the key-info file.
UNSAFE_PATH_CHARACTERS = (":", "%", "\n", "\r")


@dataclass(frozen=True, slots=True)
class EncodeSource:
    """The admitted source as the encode reads it: a local file and the preflight's decisions."""

    path: Path  # absolute; the downloaded bytes the preflight's facts describe
    demuxer: str  # the one allowed demuxer the preflight saw
    video_index: int  # ffprobe's stream index of the admitted video stream
    audio_index: int | None  # the first audio stream, or None for a silent source
    rotation_degrees: int
    duration_ms: int  # the admitted duration

    def __post_init__(self) -> None:
        require_safe_path("path", self.path)
        if self.rotation_degrees not in ROTATION_FILTERS:
            raise ValueError(f"rotation_degrees must be one of {sorted(ROTATION_FILTERS)}")
        for name, minimum in (("video_index", 0), ("duration_ms", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} is out of range: {value!r}")
        if self.audio_index is not None and (
            not isinstance(self.audio_index, int) or isinstance(self.audio_index, bool) or self.audio_index < 0
        ):
            raise ValueError(f"audio_index is out of range: {self.audio_index!r}")
        if not isinstance(self.demuxer, str) or ALLOW_LIST_ENTRY_PATTERN.fullmatch(self.demuxer) is None:
            raise ValueError(f"demuxer must be a single demuxer name, got {self.demuxer!r}")

    @classmethod
    def from_facts(cls, path: Path, facts: SourceFacts, profile: MediaProfile) -> EncodeSource:
        """The encode's view of an admitted source. Raises ValueError for facts admission refuses."""
        names = {name.strip() for name in facts.demuxer.split(DEMUXER_NAME_SEPARATOR)}
        allowed = sorted(names.intersection(profile.source.allowed_demuxers))
        if len(allowed) != 1:
            raise ValueError("the source's container is not exactly one allowed demuxer")
        if len(facts.video_streams) != 1:
            raise ValueError("an admitted source has exactly one video stream")
        video = facts.video_streams[0]
        return cls(
            path=path,
            demuxer=allowed[0],
            video_index=video.index,
            audio_index=facts.audio_streams[0].index if facts.audio_streams else None,
            rotation_degrees=video.rotation_degrees,
            duration_ms=facts.duration_ms,
        )


def require_safe_path(name: str, path: object) -> None:
    """Refuse a path that is relative or that the command line would read as something else."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute Path")
    if any(character in str(path) for character in UNSAFE_PATH_CHARACTERS):
        raise ValueError(f"{name} holds a character the command line would interpret")


def decoder_pixel_limit(profile: MediaProfile) -> int:
    """The decoder's -max_pixels: the profile's pixel cap, plus what buffer alignment adds to it.

    libavcodec checks a frame against max_pixels with its width rounded up to the buffer stride
    alignment (up to 64 on x86-64), and some paths check the coded size too, so an admitted frame
    of exactly the cap with an unaligned side (2160x3840 portrait 4K) would otherwise be refused.
    """
    caps = profile.source
    slack = DECODER_ALIGNMENT - 1
    return caps.max_display_pixels + slack * (caps.max_display_width + caps.max_display_height) + slack * slack


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds // MS_PER_SECOND}.{milliseconds % MS_PER_SECOND:03d}"


def _max_frames(source: EncodeSource, plan: RenditionPlan) -> int:
    frames = source.duration_ms * plan.frame_rate_num / (MS_PER_SECOND * plan.frame_rate_den)
    return math.ceil(frames) + 1


def _video_filter(source: EncodeSource, plan: RenditionPlan, profile: MediaProfile) -> str:
    # The frame rate first, so nothing after it sees more frames than the output keeps.
    steps = [f"fps={plan.frame_rate_num}/{plan.frame_rate_den}"]
    steps += ROTATION_FILTERS[source.rotation_degrees]
    steps += [f"scale={plan.width}:{plan.height}", "setsar=1", f"format={profile.encode.pixel_format}"]
    return ",".join(steps)


def _audio_arguments(source: EncodeSource, plan: RenditionPlan, profile: MediaProfile) -> list[str]:
    if source.audio_index is None or plan.audio_bitrate is None:
        return ["-an"]
    return [
        "-c:a", profile.encode.audio_codec,
        "-b:a", str(plan.audio_bitrate),
        "-ar", str(profile.encode.audio_sample_rate),
        "-ac", str(profile.encode.audio_channels),
    ]


def rendition_argv(
    executable: str,
    source: EncodeSource,
    plan: RenditionPlan,
    profile: MediaProfile,
    *,
    output_dir: Path,
    key_info_path: Path | None,
) -> list[str]:
    """The argv that encodes `source` into one HLS rendition in the empty `output_dir`.

    With `key_info_path` the segments are AES-128 encrypted, each with its own IV (its media
    sequence number); without it they are plain.
    """
    require_safe_path("output_dir", output_dir)
    if key_info_path is not None:
        require_safe_path("key_info_path", key_info_path)
    if (source.audio_index is None) != (plan.audio_bitrate is None):
        raise ValueError("the plan's audio does not match the source's")
    encode = profile.encode
    segment_s = str(encode.segment_duration_s)
    gop = str(plan.gop_frames)
    audio_map = [] if source.audio_index is None else ["-map", f"0:{source.audio_index}"]
    encryption = [] if key_info_path is None else ["-hls_key_info_file", str(key_info_path)]
    return [
        executable,
        "-hide_banner", "-nostdin", "-nostats", "-loglevel", LOG_LEVEL,
        # Input: a local file, read only through the admitted demuxer, decoded within the pixel cap.
        "-protocol_whitelist", INPUT_PROTOCOLS,
        "-format_whitelist", source.demuxer,
        "-f", source.demuxer,
        "-noautorotate",
        # The container's own crop (MKV PixelCrop, MOV clap) is not in the preflight's geometry;
        # only the codec's crop is, so only that one is applied.
        "-apply_cropping", "codec",
        "-max_pixels", str(decoder_pixel_limit(profile)),
        "-i", f"{FILE_URL_PREFIX}{source.path}",
        # Output: local files only; the admitted streams only; no metadata carried over.
        "-protocol_whitelist", OUTPUT_PROTOCOLS_PLAIN if key_info_path is None else OUTPUT_PROTOCOLS_ENCRYPTED,
        "-map", f"0:{source.video_index}",
        *audio_map,
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-sn", "-dn",
        "-t", _seconds(source.duration_ms),
        "-frames:v", str(_max_frames(source, plan)),
        "-vf", _video_filter(source, plan, profile),
        "-c:v", encode.video_encoder,
        "-preset", encode.encoder_preset,
        "-profile:v", encode.h264_profile,
        "-level:v", plan.level.name,
        "-pix_fmt", encode.pixel_format,
        "-b:v", str(plan.video_bitrate),
        "-maxrate", str(plan.video_maxrate),
        "-bufsize", str(plan.video_bufsize),
        "-g", gop,
        "-keyint_min", gop,
        "-sc_threshold", "0",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_s})",
        *_audio_arguments(source, plan, profile),
        "-f", "hls",
        "-hls_time", segment_s,
        "-hls_playlist_type", "vod",
        "-hls_list_size", "0",
        "-hls_segment_type", HLS_SEGMENT_TYPE,
        "-start_number", str(FIRST_SEGMENT_NUMBER),
        "-hls_flags", HLS_FLAGS_PLAIN if key_info_path is None else HLS_FLAGS_ENCRYPTED,
        *encryption,
        "-hls_segment_filename", str(output_dir / SEGMENT_NAME_PATTERN),
        "-n",
        str(output_dir / PLAYLIST_NAME),
    ]
