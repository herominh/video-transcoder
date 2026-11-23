"""What each planned rendition reserves of the job's output allowance and of the scratch disk, and whether the
renditions of a job may be encoded in parallel.

The renditions share one scratch disk. Encoded one at a time, each takes what the earlier ones left, and the
free-disk check that types a failed write as the retryable scratch_full (core/encode.py) sees the disk as that one
encode left it. Encoded together, a failed sibling's removal could free the space another rendition's check relies
on, and one rendition could take room another needs. So the runner encodes them in parallel only when every
rendition can be given a room of its own up front: its reservation (rendition_reservation()), which its output
monitor holds it to, and which the job's allowance and the disk's free space both hold together (encode_mode()).
Otherwise it encodes them one at a time, as it always did.

A reservation is an upper bound taken from the plan alone, never from what an encode turned out to write: the
video's rate ceiling and the audio's bitrate over the admitted duration, times a wide factor; the video's whole VBV
buffer and one frame more at both rates (the longest a frame is, at the profile's frame-rate floor), which that
factor does not cover on a short source; and the most a media
playlist may take; and one segment per segment duration, two more, and the playlist.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.encode import MIN_FREE_BYTES, OutputAllowance
from core.profile import MediaProfile
from core.renditions import RenditionPlan

# A rendition's bytes are reserved at this many times what its rate ceilings let it take over the admitted
# duration. It covers MPEG-TS overhead and VBV bursts with a wide margin: disk is cheap, and a rendition refused
# for a reservation too small is a paid encode lost.
RESERVATION_FACTOR = 2
BITS_PER_BYTE = 8
MS_PER_SECOND = 1000
# Segments beyond the whole segment durations the admitted duration holds: the last, shorter one, and the frame
# the cut lets a rendition run past the admitted duration.
EXTRA_SEGMENTS = 2
PLAYLIST_FILES = 1


@dataclass(frozen=True, slots=True)
class EncodeMode:
    """Whether a job's renditions are encoded in parallel, and why, in words and numbers only (a log line's)."""

    parallel: bool
    reason: str


def _require_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int of at least {minimum}, got {value!r}")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def rendition_reservation(plan: RenditionPlan, profile: MediaProfile, duration_ms: int) -> OutputAllowance:
    """The room one planned rendition of a source of `duration_ms` (the admitted duration) is given up front.

    Bytes, each term rounded up:
    - the video's rate ceiling plus the audio's bitrate (none without audio) over the duration, times
      RESERVATION_FACTOR. The ceiling is the rung's `video_maxrate`: the pilot's fixed bitrate holds the video to
      it (-maxrate), and a quality level holds it to its `video_bitrate` (core/encode_command.py), which a rung
      never sets above its `video_maxrate`;
    - the rung's `video_bufsize` (-bufsize, in both modes), in bytes. VBV holds any stretch of T seconds to the
      buffer plus the ceiling times T, so a source shorter than about a second and a half could take more than
      the term above: a rendition that passes its reservation is a final output_too_large, which encoding one
      at a time would not have met;
    - one frame at the profile's frame-rate floor (the longest a frame is: a second at the pilot's 1 fps, rounded
      up to a millisecond), at the video's ceiling and the audio's bitrate: the frame the cut lets a rendition run
      past the admitted duration, which the buffer's bound on the stretch does not count, and the audio beside it;
    - the profile's `max_playlist_bytes`.
    Files: a segment per segment duration, rounded up, EXTRA_SEGMENTS more, never more segments than the profile
    allows a rendition; and the playlist.
    """
    if not isinstance(plan, RenditionPlan):
        raise TypeError("plan must be a RenditionPlan")
    if not isinstance(profile, MediaProfile):
        raise TypeError("profile must be a MediaProfile")
    _require_int("duration_ms", duration_ms, 0)
    encode = profile.encode
    rate_bps = plan.video_maxrate + (plan.audio_bitrate or 0)
    media_bytes = _ceil_div(rate_bps * duration_ms * RESERVATION_FACTOR, BITS_PER_BYTE * MS_PER_SECOND)
    buffer_bytes = _ceil_div(plan.video_bufsize, BITS_PER_BYTE)
    longest_frame_ms = _ceil_div(MS_PER_SECOND * encode.min_frame_rate_den, encode.min_frame_rate_num)
    frame_past_the_cut_bytes = _ceil_div(rate_bps * longest_frame_ms, BITS_PER_BYTE * MS_PER_SECOND)
    segments = _ceil_div(duration_ms, encode.segment_duration_s * MS_PER_SECOND) + EXTRA_SEGMENTS
    return OutputAllowance(
        max_bytes=media_bytes + buffer_bytes + frame_past_the_cut_bytes + encode.max_playlist_bytes,
        max_artifacts=min(segments, encode.max_segments_per_rendition) + PLAYLIST_FILES,
    )


def scratch_free_bytes(directory: Path) -> int | None:
    """The bytes an unprivileged process may still write on the file system of `directory`; None when the
    operating system does not tell (the error is not kept: its text names the path)."""
    try:
        usage = os.statvfs(directory)
    except OSError:
        return None
    return usage.f_bavail * usage.f_frsize


def encode_mode(
    reservations: Sequence[OutputAllowance], allowance: OutputAllowance, free_bytes: int | None
) -> EncodeMode:
    """Parallel only when there is more than one rendition, their reservations together fit the renditions'
    `allowance` (bytes and files), and `free_bytes` of the scratch disk hold their bytes together and
    MIN_FREE_BYTES more, the margin below which core/encode.py takes a failed write for a full disk. A free space
    that could not be read (None) means one at a time."""
    if not isinstance(allowance, OutputAllowance):
        raise TypeError("allowance must be an OutputAllowance")
    if not all(isinstance(reservation, OutputAllowance) for reservation in reservations):
        raise TypeError("reservations must be OutputAllowance values")
    if free_bytes is not None:
        _require_int("free_bytes", free_bytes, 0)
    if len(reservations) < 2:
        return EncodeMode(parallel=False, reason=f"sequential: {len(reservations)} rendition")
    reserved_bytes = sum(reservation.max_bytes for reservation in reservations)
    reserved_files = sum(reservation.max_artifacts for reservation in reservations)
    if reserved_bytes > allowance.max_bytes:
        return EncodeMode(
            parallel=False,
            reason=f"sequential: {reserved_bytes} bytes reserved exceed the allowance's {allowance.max_bytes}",
        )
    if reserved_files > allowance.max_artifacts:
        return EncodeMode(
            parallel=False,
            reason=f"sequential: {reserved_files} files reserved exceed the allowance's {allowance.max_artifacts}",
        )
    if free_bytes is None:
        return EncodeMode(parallel=False, reason="sequential: the scratch disk's free space is unknown")
    needed_bytes = reserved_bytes + MIN_FREE_BYTES
    if free_bytes < needed_bytes:
        return EncodeMode(
            parallel=False, reason=f"sequential: {free_bytes} bytes free on the scratch disk, {needed_bytes} needed"
        )
    return EncodeMode(parallel=True, reason=f"parallel: {len(reservations)} renditions")
