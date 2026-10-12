"""JobRunner's own decisions, fast and without ffmpeg.

The three stage functions that need ffprobe or ffmpeg are replaced in core.job.runner's namespace by
stand-ins (StandInStages): `run_preflight` reads a little of the source through the reader it is given
and decides admission with the real decide_admission; `encode_rendition` and `make_thumbnail` write
small files and return real values of the stages' own types. Everything else is the real code: the
download through S3RangeSource, the rendition planning, the allowances, the master playlist, the
manifest and its self-check, the upload, the progress messages and both results. So these tests check
what the runner decides (what it resolves first, the order, the stage it reports, what it never
uploads, what it never tells), not what ffmpeg produces: tests/job/test_runner_end_to_end.py does that.

The storage is conftest's fake S3 client (one for both sides unless a test gives the output its own),
the Hub a fake transport and a fake clock: no network.
"""

from __future__ import annotations

import base64
import dataclasses
import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from core.admission import RequestLimits, decide_admission
from core.bounded_process import Deadline
from core.encode import ArtifactFile, OutputAllowance, RenditionOutput
from core.encode_command import EncodeSource
from core.failure import Diagnostic, Failure, WorkerFailure
from core import profile as profile_module
from core.job import progress as progress_module
from core.job import runner as runner_module
from core.job import storage as storage_module
from core.job.ffmpeg_version import forget_installed_ffmpeg_version, installed_ffmpeg_version
from core.job.progress import ProgressReporter
from core.job.runner import JobRunner
from core.job.scratch import remove_leftover_scratch, scratch_prefix
from core.package import CONTRACT_MAX_DURATION_MS, manifest_max_bytes, reserve_package
from core.probe import PROBE_SOURCE_SLOW, PreflightResult
from core.profile import GIB, PILOT_PROFILE, MediaProfile, profile_digest
from core.protocol import files, pipeline
from core.protocol.handler import Runtime, handle
from core.protocol.messages import ExecutionIdentity
from core.protocol.outbox import SealedMessage, SelfCheckFailed
from core.protocol.settings import StorageProfile, WorkerSettings
from core.protocol.transport import HttpAnswer, TransportFailed
from core.renditions import RenditionPlan
from core.source_facts import AudioStreamFacts, SourceFacts, VideoStreamFacts
from core.source_reader import RangeSource, ReadStats
from core.thumbnail import ThumbnailResult
from tests.contract.v2_support import (
    CALLBACK_URL,
    CLAIM_URL,
    GRANT_ENCRYPTED,
    HUB_KEY,
    MEDIA_KEY_HEX,
    REQUEST_ENCRYPTED,
    RUNTIME_ID,
    FakeClock,
    FakeTransport,
    hub_error_answer,
    modified_fixture,
    signed_answer,
    signed_carrier,
    timestamp_text,
    uuid_source,
    worker_settings,
)
from tests.job.conftest import (
    COMPLETED_KIND,
    ClockDeadline,
    DELIVERED,
    FAILED_KIND,
    HANG_RELEASE_S,
    IMAGE_DIGEST,
    LOCK_RELEASE_S,
    MANIFEST_NAME,
    SETTINGS,
    FakeS3,
    accepted_bytes_by_the_hub,
    accepted_by_the_hub,
    claimed_execution,
    client_error,
    held_at_most_a_while,
    hub_storage_context,
    job_settings,
    reported_log_failures,
    runtime_of,
    settle_requests_first,
)

SOURCE_BUCKET = "test-source-bucket"
OUTPUT_BUCKET = "test-output-bucket"
SOURCE_KEY = "test/sources/source-0001.mp4"
SOURCE_ETAG = "test-etag-0001"
OTHER_ETAG = "test-etag-0002"
SOURCE_OBJECT = bytes(range(256)) * 80  # 20,480 bytes that are no media: no stand-in decodes them
PROBED_BYTES = 1024
DOWNLOAD_CHUNK_BYTES = 4096
FFMPEG_VERSION = "7.1.1-test"
PILOT_TRIPLE = {
    "id": PILOT_PROFILE.profile_id,
    "version": PILOT_PROFILE.version,
    "sha256": profile_digest(PILOT_PROFILE),
}
MEDIA_KEY = bytes.fromhex(MEDIA_KEY_HEX)
OTHER_UUID = "cccccccc-cccc-7ccc-8ccc-cccccccccccc"
UNKNOWN_CREDENTIAL_REF = "archive"
CANARY = "CANARYtextOfAnUnexpectedError0123"
CAUSE_CANARY = "CAUSEcanaryOfTheErrorBehindIt4567"
PATH_CANARY = "/CANARY/path/of/a/leftover"
HTTP_FORBIDDEN = 403
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVER_ERROR = 500
STAGES = ("probing", "downloading", "transcoding", "finalizing", "uploading")
# The function of core.job.runner whose failure ends each stage.
STAGE_FUNCTIONS = {
    "probing": "run_preflight",
    "downloading": "download_source",
    "transcoding": "encode_rendition",
    "finalizing": "make_thumbnail",
    "uploading": "upload_generation",
}
STAGE_FAILURE = Failure("encoder_failed", "stand_in_stage_failed", True, "The stand-in stage of the test failed.")
# A stand-in stage that ran out of time, and a start of ten minutes of which the claim took 12.5 s.
STAGE_DEADLINE = Failure(
    "deadline_exceeded", "stand_in_deadline_exceeded", False, "The stand-in stage ran out of time."
)
TEN_MINUTES_MS = 600_000
CLAIM_TOOK_S = 12.5
SEGMENT_BYTES = 400
NO_THUMBNAIL_DETAIL = "the stand-in thumbnail encoder wrote no image"
PERCENT_STEPS_PER_STAGE = 10
NO_ANSWER = TransportFailed("ConnectTimeout")
UNDELIVERED_BEFORE_GIVING_UP = 3
# What a progress post may take, and what all of an execution's posts may take together.
PROGRESS_TIMEOUT_S = 3.0
PROGRESS_ALLOWANCE_S = 30.0
# A Hub that takes every report, each just inside the progress time-out.
SLOW_POST_S = 2.9
POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW = 11  # ten take 29 s, the eleventh reaches the 30 s
# How many reports a Hub takes before it takes no more, where a test has the third one lost in a row
# be the one that reaches the allowance too: eight take 23.2 s, and the eleventh post ends at 31.9 s.
DELIVERED_BEFORE_THE_HUB_IS_LOST = 8
# A wall-time limit all the reports of one job, each that slow, would outlast.
WALL_TIME_A_SLOW_HUB_WOULD_OUTLAST_S = 35
CLIENT_CANARY = "CANARYendpointInAClientError.example.invalid/bucket?X-Amz-Signature=0123"
WALL_TIME_POINTER = "/limits/max_wall_time_ms"
HTTP_UNAVAILABLE = 503
PROGRESS_KIND = "transcode.progress"
# The factor every wait between two attempts is multiplied by where a test states the waits, in place of
# the jitter's draw: the nominal 0.5, 2 and 8 s times it are 0.375, 1.5 and 6 s, each exact in binary.
FIXED_FACTOR = 0.75
# A storage request's ceiling a test waits out, where the worker's own is half a minute and more.
SHORT_CEILING_S = 0.05
SCRATCH_FULL_DETAIL = "the scratch disk is full"
SCRATCH_NOT_SET_UP_DETAIL = "the execution's scratch directory could not be set up: {errno_name}"
# How long a scratch directory has gone unchanged where a test makes it look old: during a long encode
# nothing is made or removed in the directory itself.
TWO_HOURS_S = 2 * 60 * 60
# The two places where the making of an execution's scratch directory can fail.
THE_SCRATCH_ROOT = "the-directory-itself"
INSIDE_THE_SCRATCH_ROOT = "a-directory-inside-it"


def _video(index: int = 0, **changes: Any) -> VideoStreamFacts:
    facts: dict[str, Any] = {
        "index": index,
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "sample_aspect_num": 1,
        "sample_aspect_den": 1,
        "rotation_degrees": 0,
        "display_width": 1280,
        "display_height": 720,
        "frame_rate_num": 25,
        "frame_rate_den": 1,
        "field_order": "progressive",
        "color_transfer": None,
        "pix_fmt": "yuv420p",
        "dolby_vision_profile": None,
    }
    facts.update(changes)
    return VideoStreamFacts(**facts)


def _facts(
    *, videos: tuple[VideoStreamFacts, ...] | None = None, audio_streams: int = 1, **changes: Any
) -> SourceFacts:
    video_streams = (_video(),) if videos is None else videos
    audio = tuple(
        AudioStreamFacts(index=len(video_streams) + number, codec_name="aac", channels=2)
        for number in range(audio_streams)
    )
    facts: dict[str, Any] = {
        "demuxer": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration_ms": 2000,
        "duration_estimated": False,
        "video_streams": video_streams,
        "audio_streams": audio,
        "attached_picture_count": 0,
        "total_stream_count": len(video_streams) + audio_streams,
    }
    facts.update(changes)
    return SourceFacts(**facts)


def _artifact(output_root: Path, path: str, kind: str, data: bytes) -> ArtifactFile:
    (output_root / path).write_bytes(data)
    return ArtifactFile(path=path, kind=kind, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


@dataclasses.dataclass(frozen=True)
class EncodeCall:
    """What one encode_rendition call was handed, and what the scratch directory held at that moment."""

    name: str
    media_key: bytes | None
    allowance: OutputAllowance
    deadline: Deadline
    source_path: Path
    source_bytes: bytes
    output_root: Path
    scratch_files: dict[str, bytes]


class StandInStages:
    """Stand-ins of the three stage functions that need ffprobe or ffmpeg; they record what they were handed."""

    def __init__(self, scratch_root: Path, facts: SourceFacts) -> None:
        self.scratch_root = scratch_root
        self.facts = facts
        self.probed_bytes = PROBED_BYTES
        self.thumbnail_available = True
        self.after_preflight: Callable[[], None] = lambda: None
        self.preflight_deadlines: list[Deadline] = []
        self.encodes: list[EncodeCall] = []
        self.outputs: list[RenditionOutput] = []
        self.thumbnail_deadlines: list[Deadline] = []
        self.thumbnail_references: list[RenditionPlan] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner_module, "run_preflight", self.run_preflight)
        monkeypatch.setattr(runner_module, "encode_rendition", self.encode_rendition)
        monkeypatch.setattr(runner_module, "make_thumbnail", self.make_thumbnail)

    def run_preflight(
        self, source: RangeSource, *, profile: MediaProfile, limits: RequestLimits, deadline: Deadline
    ) -> PreflightResult:
        self.preflight_deadlines.append(deadline)
        source.read(0, self.probed_bytes)
        decision = decide_admission(self.facts, size_bytes=source.size_bytes, profile=profile, limits=limits)
        self.after_preflight()
        read = ReadStats(bytes_read=self.probed_bytes, requests=1)
        return PreflightResult(facts=self.facts, decision=decision, read=read)

    def encode_rendition(
        self,
        source: EncodeSource,
        plan: RenditionPlan,
        *,
        profile: MediaProfile,
        output_root: Path,
        work_dir: Path,
        media_key: bytes | None,
        allowance: OutputAllowance,
        deadline: Deadline,
    ) -> RenditionOutput:
        scratch_files = {
            path.relative_to(self.scratch_root).as_posix(): path.read_bytes()
            for path in self.scratch_root.rglob("*")
            if path.is_file()
        }
        self.encodes.append(
            EncodeCall(
                name=plan.name,
                media_key=media_key,
                allowance=allowance,
                deadline=deadline,
                source_path=source.path,
                source_bytes=source.path.read_bytes(),
                output_root=output_root,
                scratch_files=scratch_files,
            )
        )
        (output_root / plan.name).mkdir()
        name, filler = plan.name, plan.name.encode("ascii") * SEGMENT_BYTES
        artifacts = (
            _artifact(output_root, f"{name}/playlist.m3u8", "hls_media_playlist", b"#EXTM3U\n" + filler[:40]),
            _artifact(output_root, f"{name}/segment_0000.ts", "hls_segment", b"\x47" + filler[:SEGMENT_BYTES]),
            _artifact(output_root, f"{name}/segment_0001.ts", "hls_segment", b"\x47" + filler[: SEGMENT_BYTES // 2]),
        )
        total_bytes = sum(artifact.size_bytes for artifact in artifacts)
        if total_bytes > allowance.max_bytes or len(artifacts) > allowance.max_artifacts:
            shutil.rmtree(output_root / plan.name)
            raise WorkerFailure(
                Failure("input_limits_exceeded", "output_too_large", False, "the stand-in rendition does not fit")
            )
        output = RenditionOutput(
            name=plan.name,
            width=plan.width,
            height=plan.height,
            frame_rate_num=plan.frame_rate_num,
            frame_rate_den=plan.frame_rate_den,
            codecs=plan.codecs,
            has_audio=plan.audio_bitrate is not None,
            encrypted=media_key is not None,
            playlist_path=f"{plan.name}/playlist.m3u8",
            segment_count=2,
            # The middle rendition runs a frame longer than the others: the generation's duration is the longest.
            duration_ms=self.facts.duration_ms + (40 if len(self.encodes) == 2 else 0),
            bandwidth_bps=plan.video_maxrate,
            average_bandwidth_bps=plan.video_bitrate,
            total_bytes=total_bytes,
            artifacts=artifacts,
        )
        self.outputs.append(output)
        return output

    def make_thumbnail(
        self,
        source: EncodeSource,
        reference: RenditionPlan,
        *,
        profile: MediaProfile,
        output_root: Path,
        work_dir: Path,
        deadline: Deadline,
    ) -> ThumbnailResult:
        self.thumbnail_deadlines.append(deadline)
        self.thumbnail_references.append(reference)
        if not self.thumbnail_available:
            return ThumbnailResult(
                artifact=None, diagnostic=Diagnostic(code="thumbnail_unavailable", detail=NO_THUMBNAIL_DETAIL)
            )
        picture = b"\xff\xd8 the stand-in picture \xff\xd9"
        return ThumbnailResult(artifact=_artifact(output_root, "thumbnail.jpg", "thumbnail", picture), diagnostic=None)


class FfmpegVersionAsked:
    """A stand-in of the ffmpeg version check: it answers FFMPEG_VERSION and records the deadline of each ask."""

    def __init__(self) -> None:
        self.deadlines: list[Deadline] = []

    def __call__(self, deadline: Deadline) -> str:
        self.deadlines.append(deadline)
        return FFMPEG_VERSION


class Harness:
    """One execution ready to run: the fixtures' dispatch and grant, a source object in the fake storage,
    the stand-in stages, and the Hub's side to judge what comes back.

    The execution's start began when the harness's clock read zero and its claim took no time, so the
    dispatch's whole wall time is left until a test advances that clock before it runs."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        encrypted: bool = True,
        request_changes: Mapping[str, Any] | None = None,
        facts: SourceFacts | None = None,
        ffmpeg_version: Callable[[Deadline], str] = lambda deadline: FFMPEG_VERSION,
        profile: MediaProfile = PILOT_PROFILE,
        settings: WorkerSettings = SETTINGS,
        seconds_per_post: float = 0.0,
    ) -> None:
        self.scratch_root = tmp_path / "scratch"
        self.scratch_root.mkdir()
        # One storage for the source and the output, unless a test gives the output its own.
        self.s3 = FakeS3()
        self.output_s3 = self.s3
        self.s3.hold(SOURCE_BUCKET, SOURCE_KEY, SOURCE_OBJECT, SOURCE_ETAG)
        # `profile` is the one revision the worker holds, and the one the dispatch names unless a test changes it.
        monkeypatch.setattr(profile_module, "SUPPORTED_PROFILES", (profile,))
        named = {"id": profile.profile_id, "version": profile.version, "sha256": profile_digest(profile)}
        changes = {"/source/size_bytes": len(SOURCE_OBJECT), "/profile": named, **(request_changes or {})}
        self.claimed = claimed_execution(encrypted=encrypted, request_changes=changes, settings=settings)
        self.clock = FakeClock()
        self.transport = FakeTransport(then=DELIVERED, clock=self.clock, seconds_per_post=seconds_per_post)
        self.ids_drawn = 0
        self.drawn_ids: list[str] = []
        self._next_id = uuid_source()
        self.runtime = dataclasses.replace(runtime_of(self.transport, self.clock, settings), new_uuid=self._draw_id)
        self.stages = StandInStages(self.scratch_root, _facts() if facts is None else facts)
        self.stages.install(monkeypatch)
        self.source_access: list[StorageProfile] = []
        self.output_access: list[StorageProfile] = []
        # What each client factory raises instead of making a client, when a test sets one.
        self.source_client_error: Exception | None = None
        self.output_client_error: Exception | None = None
        self.runner = JobRunner(
            job_settings(self.scratch_root),
            source_client_factory=self._source_client_of,
            output_client_factory=self._output_client_of,
            ffmpeg_version=ffmpeg_version,
        )
        monkeypatch.setattr(storage_module, "DOWNLOAD_CHUNK_BYTES", DOWNLOAD_CHUNK_BYTES)

    def _draw_id(self) -> str:
        self.ids_drawn += 1
        self.drawn_ids.append(self._next_id())
        return self.drawn_ids[-1]

    def _source_client_of(self, profile: StorageProfile) -> FakeS3:
        self.source_access.append(profile)
        if self.source_client_error is not None:
            raise self.source_client_error
        return self.s3

    def _output_client_of(self, profile: StorageProfile) -> FakeS3:
        self.output_access.append(profile)
        if self.output_client_error is not None:
            raise self.output_client_error
        return self.output_s3

    def count_wall_time_on_the_fake_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the execution's wall-time deadline pass on the test's clock, not on the real one."""
        clock = self.clock

        class DeadlineOnTheFakeClock(Deadline):
            """A Deadline where the runner names one; the one it counts from a duration passes on the test's clock."""

            @classmethod
            def after_ms(cls, milliseconds: int) -> ClockDeadline:
                return ClockDeadline(clock, milliseconds / 1000)

        monkeypatch.setattr(runner_module, "Deadline", DeadlineOnTheFakeClock)
        # What is left on the test's clock also bounds, in real seconds, how long the job waits for a
        # storage request's thread. So each request has ended before the job begins to wait for it.
        settle_requests_first(monkeypatch)

    @property
    def access_profiles(self) -> list[StorageProfile]:
        """Every access profile a client was made of, the source's first."""
        return [*self.source_access, *self.output_access]

    @property
    def bytes_served(self) -> int:
        """How many bytes of the source the storage sent, re-reads counted each time."""
        served = 0
        for call in self.s3.get_calls:
            first, last = call["Range"].removeprefix("bytes=").split("-")
            served += int(last) - int(first) + 1
        return served

    def run(self) -> SealedMessage:
        return self.runner.run(self.claimed, self.runtime)

    def result(self) -> dict[str, Any]:
        """Run, and return the result once the Hub's validator accepted its signed bytes."""
        return accepted_by_the_hub(self.run(), self.claimed, now=self.clock())

    def regrant(self, **changes: Any) -> None:
        """Hand the runner the fixtures' grant with some fields changed, as only a bug before it could."""
        grant = dataclasses.replace(self.claimed.grant, **changes)
        self.claimed = dataclasses.replace(self.claimed, grant=grant)

    @property
    def progress(self) -> list[dict[str, Any]]:
        return [json.loads(call.body) for call in self.transport.calls_to(CALLBACK_URL)]

    @property
    def uploaded(self) -> dict[str, bytes]:
        """What the storage holds of the generation, by path below the granted prefix."""
        prefix = self.claimed.grant.output_prefix + "/"
        return {call.key.removeprefix(prefix): call.body for call in self.output_s3.stored}

    @property
    def left_in_scratch(self) -> list[str]:
        return sorted(path.name for path in self.scratch_root.iterdir())


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


class ClosableS3(FakeS3):
    """A FakeS3 that can be closed, as the real client can. It counts its closes and notes how many
    requests it had taken at the last of them; `close_error`, when a test sets one, is what a close raises."""

    def __init__(self) -> None:
        super().__init__()
        self.closes = 0
        self.requests_at_its_close: int | None = None
        self.close_error: Exception | None = None

    def close(self) -> None:
        self.closes += 1
        self.requests_at_its_close = self.request_count
        if self.close_error is not None:
            raise self.close_error


def _give_closable_clients(harness: Harness) -> tuple[ClosableS3, ClosableS3]:
    """Give each side of the harness a storage client of its own that can be closed: the source's, which
    holds the source object, and the output's."""
    source, output = ClosableS3(), ClosableS3()
    source.hold(SOURCE_BUCKET, SOURCE_KEY, SOURCE_OBJECT, SOURCE_ETAG)
    harness.s3, harness.output_s3 = source, output
    return source, output


@pytest.fixture
def no_request_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test begins in a process that never abandoned a storage request, whatever the session did."""
    monkeypatch.setattr(storage_module, "_abandoned_requests", 0)


@pytest.fixture
def fixed_factor(monkeypatch: pytest.MonkeyPatch) -> float:
    """Every wait between two attempts is its nominal seconds times FIXED_FACTOR: the jitter's draw is replaced."""
    monkeypatch.setattr(storage_module, "_jitter_factor", lambda: FIXED_FACTOR)
    return FIXED_FACTOR


@pytest.fixture
def never_answered() -> Iterator[threading.Event]:
    """What a storage stand-in that never answers waits on. It is set when the test ends, and the request
    threads the test started are then waited for, so that none outlives its test."""
    threads_before = set(threading.enumerate())
    let_go = threading.Event()
    yield let_go
    let_go.set()
    for thread in set(threading.enumerate()) - threads_before:
        thread.join(HANG_RELEASE_S)


def _lock_as_another_would(directory: Path) -> int | None:
    """Try the exclusive lock on `directory` without waiting, on a descriptor of the test's own, as
    another process would: the descriptor that now holds it, or None when someone else holds the lock.
    (flock belongs to the open file description, so a second open contends with the first.)"""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _is_locked(directory: Path) -> bool:
    """Whether someone holds the exclusive lock on `directory`; the test's own try at it is let go at once."""
    descriptor = _lock_as_another_would(directory)
    if descriptor is None:
        return True
    os.close(descriptor)
    return False


def _is_still_open_on(descriptor: int, opened_on: tuple[int, int]) -> bool:
    """Whether `descriptor` is still open on the file or directory it was opened on, which is told by
    its device and inode (`opened_on`): a number that was closed and given to another file is not."""
    try:
        status = os.fstat(descriptor)
    except OSError:
        return False
    return (status.st_dev, status.st_ino) == opened_on


def _error_of(result: Mapping[str, Any]) -> tuple[str, str, bool, str | None]:
    error = result["error"]
    return error["class"], error["code"], error["retryable"], error["stage"]


def _raising(error: BaseException) -> Callable[..., Any]:
    def stand_in(*arguments: Any, **named_arguments: Any) -> Any:
        raise error

    return stand_in


def a_stage_with_a_bug(*arguments: Any, **named_arguments: Any) -> Any:
    """Fails as a bug would: an error whose text is a canary, raised while handling another whose text is one too."""
    try:
        raise KeyError(CAUSE_CANARY)
    except KeyError as cause:
        raise RuntimeError(CANARY) from cause


# --- a completed execution ------------------------------------------------------------------------


def test_run_when_every_stage_succeeds_should_return_a_completed_result_the_hub_accepts(harness):
    # Act
    result = harness.result()

    # Assert
    dispatch, grant = harness.claimed.dispatch, harness.claimed.grant
    assert result["message_kind"] == COMPLETED_KIND
    assert result["identity"] == grant.identity.to_dict()
    assert result["generation_id"] == grant.generation_id
    assert result["profile"] == PILOT_TRIPLE
    assert result["source"] == {
        "source_id": dispatch.source.source_id,
        "bytes_read": len(SOURCE_OBJECT),
        "etag_observed": SOURCE_ETAG,
    }
    assert result["encoder"] == {
        "image_digest": IMAGE_DIGEST,
        "ffmpeg_version": FFMPEG_VERSION,
        "video_encoder": "libx264",
        "preset": "medium",
    }
    assert "diagnostics" not in result


def test_run_when_every_stage_succeeds_should_locate_the_manifest_it_stored(harness):
    # Act
    result = harness.result()

    # Assert
    stored_manifest = harness.uploaded[MANIFEST_NAME]
    listed = json.loads(stored_manifest)["artifacts"]
    assert result["manifest"] == {
        "location_id": harness.claimed.dispatch.output.location_id,
        "path": MANIFEST_NAME,
        "sha256": hashlib.sha256(stored_manifest).hexdigest(),
        "size_bytes": len(stored_manifest),
        "artifact_count": len(listed),
        "total_bytes": sum(artifact["size_bytes"] for artifact in listed),
    }


def test_run_when_every_stage_succeeds_should_store_a_manifest_the_hub_accepts(harness):
    # Act
    harness.run()

    # Assert
    verdict = pipeline.validate(harness.uploaded[MANIFEST_NAME], hub_storage_context(harness.claimed))
    assert verdict.accepted, verdict


def test_run_when_every_stage_succeeds_should_state_the_media_as_the_stored_manifest_does(harness):
    # Act
    result = harness.result()

    # Assert
    assert result["media"] == json.loads(harness.uploaded[MANIFEST_NAME])["media"]
    assert result["media"] == {
        "duration_ms": 2040,
        "max_width": 1280,
        "max_height": 720,
        "has_audio": True,
        "rendition_count": 3,
    }


def test_run_when_every_stage_succeeds_should_store_exactly_the_manifests_files_under_the_granted_prefix(harness):
    # Act
    harness.run()

    # Assert
    prefix = harness.claimed.grant.output_prefix
    manifest = json.loads(harness.uploaded[MANIFEST_NAME])
    listed = {artifact["path"]: artifact for artifact in manifest["artifacts"]}
    assert [call.key for call in harness.s3.put_calls][-1] == f"{prefix}/{MANIFEST_NAME}"
    assert all(call.key.startswith(f"{prefix}/") and call.bucket == OUTPUT_BUCKET for call in harness.s3.put_calls)
    assert set(harness.uploaded) == set(listed) | {MANIFEST_NAME}
    assert len(harness.s3.put_calls) == len(listed) + 1
    for path, artifact in listed.items():
        assert artifact["sha256"] == hashlib.sha256(harness.uploaded[path]).hexdigest()


def test_run_when_the_source_is_downloaded_should_hand_the_encode_its_exact_bytes_under_a_fixed_name(harness):
    # Act
    harness.run()

    # Assert
    first = harness.stages.encodes[0]
    assert first.source_bytes == SOURCE_OBJECT
    assert harness.scratch_root in first.source_path.parents
    assert first.source_path.relative_to(harness.scratch_root).parts[1:] == ("source", "source")
    assert all(call["IfMatch"] == f'"{SOURCE_ETAG}"' for call in harness.s3.get_calls)


def test_run_when_the_preflight_already_read_the_whole_object_should_not_report_more_than_the_object_holds(
    harness,
):
    # Arrange: a small object, which the preflight reads entirely before the download reads it again
    harness.stages.probed_bytes = len(SOURCE_OBJECT)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.bytes_served == 2 * len(SOURCE_OBJECT)
    assert result["source"]["bytes_read"] == len(SOURCE_OBJECT)


def test_run_when_the_source_and_the_output_name_different_access_profiles_should_use_each_for_its_own_side(
    tmp_path, monkeypatch
):
    # Arrange: the output lies behind a second access profile, in a storage of its own
    settings = worker_settings(
        VH_STORAGE_PROFILES="default,archive",
        VH_STORAGE_ARCHIVE_ENDPOINT="https://archive.test.invalid",
        VH_STORAGE_ARCHIVE_ACCESS_KEY_ID="archive-access-key-id",
        VH_STORAGE_ARCHIVE_SECRET_ACCESS_KEY="archive-secret-access-key",
    )
    harness = Harness(tmp_path, monkeypatch, settings=settings, request_changes={"/output/credential_ref": "archive"})
    harness.output_s3 = FakeS3()

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert [access.credential_ref for access in harness.source_access] == ["default"]
    assert [access.credential_ref for access in harness.output_access] == ["archive"]
    assert harness.s3.put_calls == [] and len(harness.s3.get_calls) > 1
    assert harness.output_s3.get_calls == [] and MANIFEST_NAME in harness.uploaded


def test_run_when_the_thumbnail_is_unavailable_should_complete_and_carry_its_diagnostic(harness):
    # Arrange
    harness.stages.thumbnail_available = False

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert result["diagnostics"] == [{"code": "thumbnail_unavailable", "detail": NO_THUMBNAIL_DETAIL}]
    assert "thumbnail.jpg" not in harness.uploaded
    assert json.loads(harness.uploaded[MANIFEST_NAME])["thumbnail_path"] is None


def test_run_when_the_dispatch_is_unencrypted_should_encode_without_a_key_and_complete(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, encrypted=False)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert [call.media_key for call in harness.stages.encodes] == [None, None, None]
    assert json.loads(harness.uploaded[MANIFEST_NAME])["encryption"] == {"mode": "none"}


def test_run_when_it_ends_should_leave_no_scratch_directory_behind(harness):
    # Act
    harness.run()

    # Assert
    used = harness.stages.encodes[0].output_root
    assert harness.scratch_root in used.parents
    assert harness.left_in_scratch == []


def test_run_when_it_makes_its_scratch_directory_should_name_it_with_this_hosts_scratch_prefix(harness):
    # Act
    harness.run()

    # Assert: a directory of its own directly in the scratch root, whose name begins as the removal of
    # leftovers expects of this host's: killed in it, the execution leaves what a later start removes,
    # and no worker of another host takes it for its own
    used = harness.stages.encodes[0].output_root.parent
    assert used.parent == harness.scratch_root
    assert used.name.startswith(scratch_prefix()) and used.name != scratch_prefix()


def test_handle_when_given_the_job_runner_should_return_its_result_as_the_job_output_and_post_the_same_bytes(
    harness,
):
    # Arrange: one whole start of the worker: the signed dispatch, the claim, then the runner
    request = modified_fixture(
        REQUEST_ENCRYPTED, {"/source/size_bytes": len(SOURCE_OBJECT), "/profile": dict(PILOT_TRIPLE)}
    )
    transport = FakeTransport(clock=harness.clock)
    transport.route(CLAIM_URL, signed_answer(files.read_bytes(GRANT_ENCRYPTED), HUB_KEY))
    transport.route(CALLBACK_URL, then=DELIVERED)
    runtime = Runtime(
        settings=SETTINGS,
        transport=transport,
        clock=harness.clock,
        monotonic=harness.clock.monotonic,
        sleep=harness.clock.sleep,
        new_uuid=uuid_source(first=RUNTIME_ID),
    )

    # Act
    output = handle(signed_carrier(request, HUB_KEY), runtime=runtime, run_job=harness.runner.run)

    # Assert
    result = accepted_bytes_by_the_hub(
        output["vh_message"].encode("ascii"), output["vh_signature"], harness.claimed, now=harness.clock()
    )
    last_post = transport.calls_to(CALLBACK_URL)[-1]
    assert result["message_kind"] == COMPLETED_KIND
    assert (last_post.body.decode("ascii"), last_post.signature) == (output["vh_message"], output["vh_signature"])
    assert MANIFEST_NAME in harness.uploaded


# --- the renditions -------------------------------------------------------------------------------


def test_run_when_it_encodes_should_take_the_renditions_one_by_one_in_plan_order(harness):
    # Act
    harness.run()

    # Assert: largest first, and each already on disk when the next starts
    assert [call.name for call in harness.stages.encodes] == ["720p", "480p", "360p"]
    assert "720p/segment_0001.ts" in {
        Path(*Path(path).parts[2:]).as_posix() for path in harness.stages.encodes[1].scratch_files
    }


def test_run_when_it_encodes_should_give_each_rendition_only_what_the_earlier_ones_left(harness):
    # Act
    harness.run()

    # Assert
    first, second, third = (call.allowance for call in harness.stages.encodes)
    taken_by_first, taken_by_second = harness.stages.outputs[0], harness.stages.outputs[1]
    assert second.max_bytes == first.max_bytes - taken_by_first.total_bytes
    assert second.max_artifacts == first.max_artifacts - len(taken_by_first.artifacts)
    assert third.max_bytes == second.max_bytes - taken_by_second.total_bytes
    assert third.max_artifacts == second.max_artifacts - len(taken_by_second.artifacts)


def test_run_when_it_encodes_should_keep_the_packages_room_out_of_the_renditions_allowance(harness):
    # Arrange: what the dispatch allows the whole job
    limits = harness.claimed.dispatch.limits

    # Act
    harness.run()

    # Assert
    first = harness.stages.encodes[0].allowance
    assert first.max_bytes < limits.max_output_bytes - PILOT_PROFILE.encode.thumbnail_max_bytes
    assert first.max_artifacts == limits.max_artifact_count - 2


def test_run_when_the_scratch_ceiling_is_nearer_than_the_output_limit_should_leave_the_renditions_what_the_source_does(
    tmp_path, monkeypatch
):
    # Arrange: the worker holds a profile whose scratch ceiling, 8 GiB for the source and every output,
    # is below what the dispatch allows the output alone
    ceiling_bytes = 8 * GIB
    held = dataclasses.replace(
        PILOT_PROFILE,
        source=dataclasses.replace(PILOT_PROFILE.source, max_source_bytes=GIB),
        encode=dataclasses.replace(PILOT_PROFILE.encode, max_scratch_bytes=ceiling_bytes),
    )
    harness = Harness(tmp_path, monkeypatch, profile=held)
    limits = harness.claimed.dispatch.limits
    whole = OutputAllowance(max_bytes=ceiling_bytes, max_artifacts=limits.max_artifact_count)
    package_bytes = whole.max_bytes - reserve_package(whole, held).max_bytes

    # Act
    result = harness.result()

    # Assert: the ceiling less the source's own size, less the package's room
    assert result["message_kind"] == COMPLETED_KIND
    assert limits.max_output_bytes > ceiling_bytes
    assert harness.stages.encodes[0].allowance.max_bytes == ceiling_bytes - len(SOURCE_OBJECT) - package_bytes


def test_run_when_the_source_has_no_audio_should_plan_and_report_renditions_without_audio(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, facts=_facts(audio_streams=0))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert [output.has_audio for output in harness.stages.outputs] == [False, False, False]
    assert result["media"]["has_audio"] is False
    assert json.loads(harness.uploaded[MANIFEST_NAME])["media"]["has_audio"] is False


def test_run_when_it_makes_the_thumbnail_should_take_the_largest_rendition_as_its_reference(harness):
    # Act
    harness.run()

    # Assert: the first of the plan, which the dispatch's 720p, 480p and 360p make the largest
    (reference,) = harness.stages.thumbnail_references
    assert (reference.name, reference.width, reference.height) == ("720p", 1280, 720)
    assert reference.height == max(output.height for output in harness.stages.outputs)


def test_run_when_it_writes_the_manifest_should_hold_it_to_the_room_the_plan_kept_for_it(harness, monkeypatch):
    # Arrange: note the most bytes the manifest is allowed
    allowed: list[int] = []
    write_manifest = runner_module.write_manifest

    def write_and_note(**arguments: Any) -> Any:
        allowed.append(arguments["max_bytes"])
        return write_manifest(**arguments)

    monkeypatch.setattr(runner_module, "write_manifest", write_and_note)

    # Act
    harness.run()

    # Assert: what a manifest of as many artifacts as the dispatch allows can take, which the package's
    # reservation set aside, and no more
    assert allowed == [manifest_max_bytes(harness.claimed.dispatch.limits.max_artifact_count)]
    assert len(harness.uploaded[MANIFEST_NAME]) <= allowed[0]


def test_run_when_a_rendition_runs_a_frame_past_the_longest_admitted_source_should_report_the_contracts_longest(
    tmp_path, monkeypatch
):
    # Arrange: a source of the 6 hours the contract admits at most; the stand-in's middle rendition runs
    # a frame longer, as an encode's cut may
    harness = Harness(
        tmp_path,
        monkeypatch,
        facts=_facts(duration_ms=CONTRACT_MAX_DURATION_MS),
        request_changes={"/limits/max_source_duration_ms": CONTRACT_MAX_DURATION_MS},
    )

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert max(output.duration_ms for output in harness.stages.outputs) > CONTRACT_MAX_DURATION_MS
    assert result["media"]["duration_ms"] == CONTRACT_MAX_DURATION_MS
    assert result["media"] == json.loads(harness.uploaded[MANIFEST_NAME])["media"]


def test_run_when_the_output_limit_holds_one_rendition_only_should_fail_at_the_second(tmp_path, monkeypatch):
    # Arrange: room for the package and for one and a half stand-in renditions
    artifact_limit = 100
    rendition_bytes = len(b"#EXTM3U\n") + 40 + (1 + SEGMENT_BYTES) + (1 + SEGMENT_BYTES // 2)
    generous = OutputAllowance(max_bytes=10**9, max_artifacts=artifact_limit)
    package_bytes = generous.max_bytes - reserve_package(generous, PILOT_PROFILE).max_bytes
    harness = Harness(
        tmp_path,
        monkeypatch,
        request_changes={
            "/limits/max_artifact_count": artifact_limit,
            "/limits/max_output_bytes": package_bytes + rendition_bytes + rendition_bytes // 2,
        },
    )

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("input_limits_exceeded", "output_too_large", False, "transcoding")
    assert [call.name for call in harness.stages.encodes] == ["720p", "480p"]
    assert harness.stages.outputs[0].total_bytes == rendition_bytes
    assert harness.s3.put_calls == []


def test_run_when_it_runs_should_hold_every_stage_to_the_dispatchs_wall_time_from_its_start(tmp_path, monkeypatch):
    # Arrange
    wall_time_ms = 600_000
    harness = Harness(tmp_path, monkeypatch, request_changes={"/limits/max_wall_time_ms": wall_time_ms})
    started = time.monotonic()

    # Act
    harness.run()
    ended = time.monotonic()

    # Assert
    stages = harness.stages
    deadlines = [*stages.preflight_deadlines, *(call.deadline for call in stages.encodes), *stages.thumbnail_deadlines]
    assert len(deadlines) == 5
    assert len({deadline.expires_at for deadline in deadlines}) == 1
    assert started + wall_time_ms / 1000 <= deadlines[0].expires_at <= ended + wall_time_ms / 1000


def test_run_when_the_wall_time_has_passed_should_fail_with_deadline_exceeded(tmp_path, monkeypatch):
    # Arrange: a wall time of 5 s on the test's clock, and a preflight that takes 6 s of it
    harness = Harness(tmp_path, monkeypatch, request_changes={"/limits/max_wall_time_ms": 5_000})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.stages.after_preflight = lambda: harness.clock.advance(6)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("deadline_exceeded", "download_deadline_exceeded", False, "downloading")
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


def test_run_when_the_wall_time_ends_while_the_generation_is_finished_should_fail_the_upload_and_store_nothing(
    tmp_path, monkeypatch
):
    # Arrange: a wall time of 5 s on the test's clock, and a manifest that takes 6 s of it to write
    harness = Harness(tmp_path, monkeypatch, request_changes={"/limits/max_wall_time_ms": 5_000})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    write_manifest = runner_module.write_manifest

    def write_a_manifest_slowly(**arguments: Any) -> Any:
        harness.clock.advance(6)
        return write_manifest(**arguments)

    monkeypatch.setattr(runner_module, "write_manifest", write_a_manifest_slowly)

    # Act
    result = harness.result()

    # Assert: the upload is held to the deadline every other stage is held to
    assert _error_of(result) == ("deadline_exceeded", "upload_deadline_exceeded", False, "uploading")
    assert len(harness.stages.encodes) == 3
    assert harness.s3.put_calls == []


# --- the start's wall time, of which the claim has already taken some -------------------------------


def test_run_when_the_claim_took_part_of_the_wall_time_should_fail_a_stage_that_outlasts_the_rest(
    tmp_path, monkeypatch
):
    # Arrange: a start of 100 s on the test's clock whose claim took 90 s of them; the preflight takes 11 s
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 100_000})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.clock.advance(90)
    harness.stages.after_preflight = lambda: harness.clock.advance(11)

    # Act
    result = harness.result()

    # Assert: the stages had the 10 s the claim left, not the dispatch's whole 100 s
    assert _error_of(result) == ("deadline_exceeded", "download_deadline_exceeded", False, "downloading")
    assert harness.stages.encodes == []


def test_run_when_the_claim_took_part_of_the_wall_time_should_complete_a_job_that_fits_the_rest(
    tmp_path, monkeypatch
):
    # Arrange: the same start; the preflight takes 9 s of the 10 s the claim left
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 100_000})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.clock.advance(90)
    harness.stages.after_preflight = lambda: harness.clock.advance(9)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert MANIFEST_NAME in harness.uploaded


def test_run_when_the_claim_took_part_of_the_wall_time_should_give_every_stage_a_deadline_that_much_sooner(
    tmp_path, monkeypatch
):
    # Arrange: 600 s of wall time, 90 s of them gone on the runtime's clock before the run
    wall_time_ms, spent_s = 600_000, 90
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: wall_time_ms})
    harness.clock.advance(spent_s)
    started = time.monotonic()

    # Act
    harness.run()
    ended = time.monotonic()

    # Assert
    left_s = wall_time_ms / 1000 - spent_s
    stages = harness.stages
    deadlines = [*stages.preflight_deadlines, *(call.deadline for call in stages.encodes), *stages.thumbnail_deadlines]
    assert len(deadlines) == 5
    assert len({deadline.expires_at for deadline in deadlines}) == 1
    assert started + left_s <= deadlines[0].expires_at <= ended + left_s


@pytest.mark.parametrize(
    "spent_before_the_run_s",
    [
        pytest.param(5.0, id="exactly-the-wall-time"),
        pytest.param(65.0, id="a-minute-more-than-the-wall-time"),
        pytest.param(4.9996, id="all-but-less-than-a-millisecond"),
    ],
)
def test_run_when_the_starts_wall_time_is_spent_should_fail_before_any_client_read_directory_or_report(
    tmp_path, monkeypatch, spent_before_the_run_s
):
    # Arrange: a start of 5 s whose claim took them all
    ffmpeg_version = FfmpegVersionAsked()
    harness = Harness(
        tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000}, ffmpeg_version=ffmpeg_version
    )
    harness.clock.advance(spent_before_the_run_s)

    # Act: the runner seals the result only if the worker's own validator accepts it, and result() returns
    # it once the Hub's validator accepted the signed bytes too
    result = harness.result()

    # Assert: typed, at no stage, before a client, a read, a put, a directory, a tool or a report; and
    # worth another dispatch, because nothing was done and only the claim took the time
    assert result["message_kind"] == FAILED_KIND
    assert _error_of(result) == ("deadline_exceeded", "wall_time_spent_before_start", True, None)
    assert harness.access_profiles == []
    assert harness.s3.request_count == 0
    assert ffmpeg_version.deadlines == []
    assert harness.stages.preflight_deadlines == [] and harness.stages.encodes == []
    assert harness.left_in_scratch == []
    assert harness.progress == [] and harness.transport.calls == []
    assert "encoder" not in result and "source" not in result


def test_run_when_the_starts_wall_time_is_spent_should_make_no_scratch_directory(tmp_path, monkeypatch):
    # Arrange: note every directory the runner asks for
    made: list[object] = []
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000})
    harness.clock.advance(5.0)
    monkeypatch.setattr(runner_module.tempfile, "mkdtemp", lambda **named: made.append(named))

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result)[:2] == ("deadline_exceeded", "wall_time_spent_before_start")
    assert made == []


def test_run_when_a_whole_millisecond_of_the_wall_time_is_left_should_still_run(tmp_path, monkeypatch):
    # Arrange: a start of 5 s on the test's clock with about 2 ms left; nothing takes time on that clock
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.clock.advance(4.998)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND


@pytest.mark.parametrize(
    "wall_time_ends_at",
    [
        pytest.param(1e12, id="some-thirty-thousand-years-ahead"),
        pytest.param(1e306, id="too-far-ahead-to-count-in-milliseconds"),
        pytest.param(10**306, id="as-far-ahead-as-a-whole-number"),
        pytest.param(sys.float_info.max, id="the-largest-float"),
    ],
)
def test_run_when_the_end_of_the_wall_time_reads_absurdly_far_ahead_should_run_within_the_dispatchs_limit_at_most(
    tmp_path, monkeypatch, wall_time_ends_at
):
    # Arrange: a start of ten minutes whose wall time, by the reading it was given, ends absurdly late
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: TEN_MINUTES_MS})
    harness.claimed = dataclasses.replace(harness.claimed, wall_time_ends_at=wall_time_ends_at)
    started = time.monotonic()

    # Act
    result = harness.result()
    ended = time.monotonic()

    # Assert: what is left of a start's wall time is never more than its limit, so the stages get that
    stages = harness.stages
    deadlines = [*stages.preflight_deadlines, *(call.deadline for call in stages.encodes), *stages.thumbnail_deadlines]
    limit_s = TEN_MINUTES_MS / 1000
    assert result["message_kind"] == COMPLETED_KIND
    assert len({deadline.expires_at for deadline in deadlines}) == 1
    assert started + limit_s <= deadlines[0].expires_at <= ended + limit_s


@pytest.mark.parametrize(
    "wall_time_ends_at",
    [
        pytest.param(-1e306, id="too-far-back-to-count-in-milliseconds"),
        pytest.param(-(10**306), id="as-far-back-as-a-whole-number"),
        pytest.param(-sys.float_info.max, id="the-least-float"),
    ],
)
def test_run_when_the_end_of_the_wall_time_reads_absurdly_far_back_should_fail_as_spent_before_the_start(
    tmp_path, monkeypatch, wall_time_ends_at
):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000})
    harness.claimed = dataclasses.replace(harness.claimed, wall_time_ends_at=wall_time_ends_at)

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert: spent, as with any reading in the past, and nothing was done. The detail is the failure's
    # own and nothing more: a count of the seconds the start had spent would tell of the bound such a
    # reading is held to (a number of some 300 digits), not of the start
    assert _error_of(result) == ("deadline_exceeded", "wall_time_spent_before_start", True, None)
    assert result["error"]["detail"] == "the start's wall time was spent before the execution could begin"
    assert harness.access_profiles == [] and harness.s3.request_count == 0
    assert harness.left_in_scratch == []


def test_handle_when_the_claim_took_most_of_the_wall_time_should_leave_the_runner_only_the_rest(
    tmp_path, monkeypatch
):
    # Arrange: one whole start of 100 s on the test's clock: the Hub refuses the first claim for 90 s, then
    # grants; the preflight takes 11 s
    harness = Harness(tmp_path, monkeypatch)
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.stages.after_preflight = lambda: harness.clock.advance(11)
    request = modified_fixture(
        REQUEST_ENCRYPTED,
        {"/source/size_bytes": len(SOURCE_OBJECT), "/profile": dict(PILOT_TRIPLE), WALL_TIME_POINTER: 100_000},
    )
    transport = FakeTransport(clock=harness.clock)
    transport.route(
        CLAIM_URL,
        hub_error_answer(HTTP_UNAVAILABLE, "service_unavailable", retryable=True, retry_after_s=90),
        signed_answer(files.read_bytes(GRANT_ENCRYPTED), HUB_KEY),
    )
    transport.route(CALLBACK_URL, then=DELIVERED)
    runtime = Runtime(
        settings=SETTINGS,
        transport=transport,
        clock=harness.clock,
        monotonic=harness.clock.monotonic,
        sleep=harness.clock.sleep,
        new_uuid=uuid_source(first=RUNTIME_ID),
    )

    # Act
    output = handle(signed_carrier(request, HUB_KEY), runtime=runtime, run_job=harness.runner.run)

    # Assert: the stages had 10 s, and the result, sealed after the start's end, is returned and not posted
    result = accepted_bytes_by_the_hub(
        output["vh_message"].encode("ascii"), output["vh_signature"], harness.claimed, now=harness.clock()
    )
    posted_kinds = {json.loads(call.body)["message_kind"] for call in transport.calls_to(CALLBACK_URL)}
    assert harness.clock.waits == [90.0]
    assert _error_of(result) == ("deadline_exceeded", "download_deadline_exceeded", False, "downloading")
    assert posted_kinds == {PROGRESS_KIND}


def test_run_when_it_asks_for_the_ffmpeg_version_should_hand_it_the_deadline_every_stage_is_held_to(
    tmp_path, monkeypatch
):
    # Arrange
    ffmpeg_version = FfmpegVersionAsked()
    harness = Harness(tmp_path, monkeypatch, ffmpeg_version=ffmpeg_version)

    # Act
    result = harness.result()

    # Assert: asked once, under the execution's own deadline
    (asked_under,) = ffmpeg_version.deadlines
    assert result["message_kind"] == COMPLETED_KIND
    assert isinstance(asked_under, Deadline)
    assert asked_under is harness.stages.preflight_deadlines[0]


def test_run_when_the_wall_time_ends_before_the_ffmpeg_version_is_known_should_fail_retryable_before_any_request(
    tmp_path, monkeypatch
):
    # Arrange: the real version check with no version remembered, in a start of 5 s on the test's clock
    # whose time runs out just as the check is asked
    forget_installed_ffmpeg_version()

    def asked_too_late(deadline: Deadline) -> str:
        harness.clock.advance(6)
        return installed_ffmpeg_version(deadline)

    harness = Harness(
        tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000}, ffmpeg_version=asked_too_late
    )
    harness.count_wall_time_on_the_fake_clock(monkeypatch)

    # Act: result() returns once the Hub's validator accepted the signed bytes
    result = harness.result()

    # Assert
    assert _error_of(result) == ("deadline_exceeded", "ffmpeg_version_deadline_exceeded", True, None)
    assert harness.s3.request_count == 0
    assert "encoder" not in result


# --- what a deadline failure tells of the start's time ----------------------------------------------


def test_run_when_a_transfer_runs_out_of_time_should_say_how_much_of_the_start_was_gone_before_the_execution_began(
    tmp_path, monkeypatch
):
    # Arrange: a start of 100.9 s on the test's clock whose claim took 90.4 s of them; the preflight takes
    # 11 s, more than the 10.5 s that were left
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 100_900})
    harness.count_wall_time_on_the_fake_clock(monkeypatch)
    harness.clock.advance(90.4)
    harness.stages.after_preflight = lambda: harness.clock.advance(11)

    # Act: the runner seals the result only if the worker's own validator accepts it, and result() returns
    # it once the Hub's validator accepted the signed bytes too
    result = harness.result()

    # Assert: the failure's own detail, whole, then both numbers in whole seconds rounded down
    assert _error_of(result) == ("deadline_exceeded", "download_deadline_exceeded", False, "downloading")
    assert result["error"]["detail"] == (
        "the job ran out of time while it downloaded the source"
        "; the start had spent 90 s of its 100 s before the execution began"
    )


@pytest.mark.parametrize("stage", [None, *STAGES], ids=["before-any-stage", *STAGES])
def test_run_when_it_ends_on_a_deadline_failure_should_tell_the_starts_time_whatever_ran_out_of_it(
    tmp_path, monkeypatch, stage
):
    # Arrange: a start of ten minutes whose claim took 12.5 s; the ffmpeg check, or one stage, runs out of time
    ran_out_of_time = _raising(WorkerFailure(STAGE_DEADLINE))
    harness = Harness(
        tmp_path,
        monkeypatch,
        request_changes={WALL_TIME_POINTER: TEN_MINUTES_MS},
        **({"ffmpeg_version": ran_out_of_time} if stage is None else {}),
    )
    if stage is not None:
        monkeypatch.setattr(runner_module, STAGE_FUNCTIONS[stage], ran_out_of_time)
    harness.clock.advance(CLAIM_TOOK_S)

    # Act
    result = harness.result()

    # Assert
    assert result["error"] == {
        "class": "deadline_exceeded",
        "code": "stand_in_deadline_exceeded",
        "retryable": False,
        "stage": stage,
        "detail": (
            "The stand-in stage ran out of time."
            "; the start had spent 12 s of its 600 s before the execution began"
        ),
    }


@pytest.mark.parametrize(
    ("spent_before_the_run_s", "expected_detail"),
    [
        pytest.param(
            5.0,
            "the start's wall time was spent before the execution could begin"
            "; the start had spent 5 s of its 5 s before the execution began",
            id="exactly-the-wall-time",
        ),
        pytest.param(
            65.0,
            "the start's wall time was spent before the execution could begin"
            "; the start had spent 65 s of its 5 s before the execution began",
            id="a-minute-more-than-the-wall-time",
        ),
        pytest.param(
            4.9996,
            "the start's wall time was spent before the execution could begin"
            "; the start had spent 5 s of its 5 s before the execution began",
            id="all-but-less-than-a-millisecond",
        ),
    ],
)
def test_run_when_the_starts_wall_time_is_spent_should_say_how_much_of_it_the_start_had_spent(
    tmp_path, monkeypatch, spent_before_the_run_s, expected_detail
):
    # Arrange: a start of 5 s whose claim took them all, or more than them
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: 5_000})
    harness.clock.advance(spent_before_the_run_s)

    # Act
    result = harness.result()

    # Assert: the true numbers, also where the claim overran the limit; less than a millisecond left
    # counts as spent, as the execution itself counts it
    assert _error_of(result) == ("deadline_exceeded", "wall_time_spent_before_start", True, None)
    assert result["error"]["detail"] == expected_detail


@pytest.mark.parametrize(
    ("own_chars", "expected_chars"),
    [
        pytest.param(935, 1000, id="just-room-for-the-numbers"),
        pytest.param(936, 936, id="one-character-too-many"),
    ],
)
def test_run_when_a_deadline_failures_detail_leaves_no_room_for_the_numbers_should_keep_it_whole_and_without_them(
    tmp_path, monkeypatch, own_chars, expected_chars
):
    # Arrange: a detail may hold 1000 characters and the numbers of this start take 65; a stage runs out
    # of time with a detail of its own of `own_chars` characters
    own_detail = "x" * own_chars
    harness = Harness(tmp_path, monkeypatch, request_changes={WALL_TIME_POINTER: TEN_MINUTES_MS})
    out_of_room = Failure("deadline_exceeded", "stand_in_deadline_exceeded", False, own_detail)
    monkeypatch.setattr(runner_module, "encode_rendition", _raising(WorkerFailure(out_of_room)))

    # Act: result() returns once both validators accepted the result
    detail = harness.result()["error"]["detail"]

    # Assert: never cut, and with the numbers only where they fit
    assert detail.startswith(own_detail)
    assert len(detail) == expected_chars
    assert detail.endswith("; the start had spent 0 s of its 600 s before the execution began") == (own_chars == 935)


# --- settled before anything is read ----------------------------------------------------------------


@pytest.mark.parametrize("location", ["source", "output"])
def test_run_when_a_credential_ref_is_unknown_should_fail_before_any_storage_request(tmp_path, monkeypatch, location):
    # Arrange: the worker holds the profile `default` only
    harness = Harness(tmp_path, monkeypatch, request_changes={f"/{location}/credential_ref": UNKNOWN_CREDENTIAL_REF})

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == FAILED_KIND
    assert _error_of(result) == ("configuration_error", "unknown_credential_ref", False, None)
    assert harness.s3.request_count == 0
    assert harness.access_profiles == []
    assert harness.stages.preflight_deadlines == []
    assert "source" not in result
    assert harness.progress == []


@pytest.mark.parametrize(
    "named",
    [
        pytest.param({**PILOT_TRIPLE, "sha256": "ab" * 32}, id="another-digest"),
        pytest.param({**PILOT_TRIPLE, "version": 2}, id="another-version"),
        pytest.param({**PILOT_TRIPLE, "id": "test-h264-sdr"}, id="another-id"),
    ],
)
def test_run_when_the_dispatch_names_a_profile_revision_the_worker_does_not_hold_should_fail_before_any_request(
    tmp_path, monkeypatch, named
):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, request_changes={"/profile": named})

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("configuration_error", "unsupported_profile", False, None)
    assert harness.s3.request_count == 0
    assert harness.stages.preflight_deadlines == []


@pytest.mark.parametrize(
    "grant_changes",
    [
        pytest.param({"media_key": None}, id="no-key"),
        pytest.param({"media_key": MEDIA_KEY[:-1]}, id="a-key-of-15-bytes"),
        pytest.param({"media_key": MEDIA_KEY + b"\x00"}, id="a-key-of-17-bytes"),
        pytest.param({"media_key": MEDIA_KEY_HEX}, id="the-key-as-text"),
        pytest.param({"media_key_id": OTHER_UUID}, id="another-key-id"),
    ],
)
def test_run_when_the_grants_key_does_not_fit_an_encrypted_dispatch_should_fail_before_any_request(
    harness, grant_changes
):
    # Arrange
    harness.regrant(**grant_changes)

    # Act
    sealed = harness.run()

    # Assert
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert _error_of(result) == ("internal_error", "grant_key_mismatch", False, None)
    assert harness.s3.request_count == 0
    assert MEDIA_KEY_HEX.encode("ascii") not in sealed.body


def test_run_when_the_grant_carries_a_key_for_an_unencrypted_dispatch_should_fail_before_any_request(
    tmp_path, monkeypatch
):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, encrypted=False)
    harness.regrant(media_key=MEDIA_KEY)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("internal_error", "grant_key_mismatch", False, None)
    assert harness.s3.request_count == 0
    assert harness.stages.encodes == []


@pytest.mark.parametrize("side", ["source", "output"])
def test_run_when_a_storage_client_cannot_be_made_should_fail_before_any_storage_request(harness, caplog, side):
    # Arrange: the factory of one side refuses its access profile, in words that quote the endpoint
    caplog.set_level(logging.DEBUG)
    setattr(harness, f"{side}_client_error", ValueError(CLIENT_CANARY))

    # Act
    sealed = harness.run()

    # Assert: typed and final, at no stage, before a read, a put, a scratch directory or an encode
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert _error_of(result) == ("configuration_error", "storage_client_unavailable", False, None)
    assert result["error"]["detail"] == f"the {side} storage client could not be made: ValueError"
    assert harness.s3.request_count == 0
    assert harness.stages.preflight_deadlines == [] and harness.stages.encodes == []
    assert harness.left_in_scratch == []
    assert harness.progress == []


@pytest.mark.parametrize("side", ["source", "output"])
def test_run_when_a_storage_client_cannot_be_made_should_tell_the_errors_class_and_never_its_text(
    harness, caplog, side
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    setattr(harness, f"{side}_client_error", ValueError(CLIENT_CANARY))
    as_a_handler_writes_it = logging.Formatter()

    # Act
    sealed = harness.run()

    # Assert
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    assert CLIENT_CANARY.encode("ascii") not in sealed.body
    assert all(CLIENT_CANARY not in line for line in written)
    assert all(record.exc_info is None for record in caplog.records)
    assert any(f"the {side} storage client could not be made: ValueError" in line for line in written)


@pytest.mark.parametrize("side", ["source", "output"])
@pytest.mark.parametrize(
    ("shortage", "named"),
    [
        pytest.param(
            OSError(errno.EMFILE, "Too many open files", CLIENT_CANARY), "OSError (EMFILE)", id="no-descriptor-left"
        ),
        pytest.param(OSError(errno.ENOMEM, "Cannot allocate memory"), "OSError (ENOMEM)", id="no-system-memory-left"),
        pytest.param(MemoryError(CLIENT_CANARY), "MemoryError", id="no-python-memory-left"),
        pytest.param(OSError(), "OSError (unknown)", id="an-os-error-without-an-errno"),
        pytest.param(
            OSError(errno.EAGAIN, "Resource temporarily unavailable"), "BlockingIOError (EAGAIN)",
            id="a-subclass-of-os-error",
        ),
        pytest.param(
            OSError(errno.ETIMEDOUT, "Operation timed out", CLIENT_CANARY), "TimeoutError (ETIMEDOUT)",
            id="a-time-out",
        ),
    ],
)
def test_run_when_a_storage_client_start_meets_a_passing_shortage_should_fail_retryable_before_any_request(
    harness, side, shortage, named
):
    # Arrange: the factory of one side runs out of something of the worker's own, as boto3 does when no
    # descriptor is left to load its data files with
    setattr(harness, f"{side}_client_error", shortage)

    # Act
    sealed = harness.run()

    # Assert: worth another start, at no stage, before a read, a put, a scratch directory or an encode
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert _error_of(result) == ("resource_exhausted", "storage_client_start_failed", True, None)
    assert result["error"]["detail"] == f"the {side} storage client could not be made: {named}"
    assert harness.s3.request_count == 0
    assert harness.stages.preflight_deadlines == [] and harness.stages.encodes == []
    assert harness.left_in_scratch == []
    assert harness.progress == []


@pytest.mark.parametrize("side", ["source", "output"])
@pytest.mark.parametrize(
    ("refusal", "named"),
    [
        pytest.param(
            FileNotFoundError(errno.ENOENT, "No such file", CLIENT_CANARY), "FileNotFoundError", id="a-missing-file"
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", CLIENT_CANARY), "PermissionError", id="a-forbidden-file"
        ),
    ],
)
def test_run_when_a_storage_client_start_meets_a_missing_or_unreadable_file_should_stay_a_configuration_error(
    harness, side, refusal, named
):
    # Arrange: an OSError as a shortage is, but one the next start would meet again
    setattr(harness, f"{side}_client_error", refusal)

    # Act
    result = harness.result()

    # Assert: final, and told by its class alone
    assert _error_of(result) == ("configuration_error", "storage_client_unavailable", False, None)
    assert result["error"]["detail"] == f"the {side} storage client could not be made: {named}"


@pytest.mark.parametrize("side", ["source", "output"])
def test_run_when_a_storage_client_start_meets_a_passing_shortage_should_tell_its_errno_name_and_never_its_text(
    harness, caplog, side
):
    # Arrange: no descriptor left, in an error whose file name quotes the endpoint
    caplog.set_level(logging.DEBUG)
    setattr(harness, f"{side}_client_error", OSError(errno.EMFILE, "Too many open files", CLIENT_CANARY))
    as_a_handler_writes_it = logging.Formatter()

    # Act
    sealed = harness.run()

    # Assert
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    assert CLIENT_CANARY.encode("ascii") not in sealed.body
    assert all(CLIENT_CANARY not in line for line in written)
    assert all(record.exc_info is None for record in caplog.records)
    assert any(f"the {side} storage client could not be made: OSError (EMFILE)" in line for line in written)


def test_run_when_both_storage_clients_can_be_made_should_make_each_once_before_the_first_read(harness):
    # Arrange: note how many output clients exist when the preflight has just read the source
    output_clients_at_the_preflight: list[int] = []
    harness.stages.after_preflight = lambda: output_clients_at_the_preflight.append(len(harness.output_access))

    # Act
    result = harness.result()

    # Assert: the output's client was made before the first read, and neither client was made again
    assert result["message_kind"] == COMPLETED_KIND
    assert output_clients_at_the_preflight == [1]
    assert (len(harness.source_access), len(harness.output_access)) == (1, 1)


def test_run_when_the_ffmpeg_version_cannot_be_determined_should_fail_before_any_request(tmp_path, monkeypatch):
    # Arrange
    unavailable = WorkerFailure(Failure("configuration_error", "ffmpeg_unavailable", False, "ffmpeg is not installed"))
    harness = Harness(tmp_path, monkeypatch, ffmpeg_version=_raising(unavailable))

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("configuration_error", "ffmpeg_unavailable", False, None)
    assert harness.s3.request_count == 0
    assert "encoder" not in result


def test_run_when_the_ffmpeg_version_cannot_be_determined_should_make_no_storage_client(tmp_path, monkeypatch):
    # Arrange
    unavailable = WorkerFailure(Failure("configuration_error", "ffmpeg_unavailable", False, "ffmpeg is not installed"))
    harness = Harness(tmp_path, monkeypatch, ffmpeg_version=_raising(unavailable))

    # Act
    result = harness.result()

    # Assert: the version is asked for before either client is made
    assert _error_of(result) == ("configuration_error", "ffmpeg_unavailable", False, None)
    assert harness.access_profiles == []


# --- the source -----------------------------------------------------------------------------------


def test_run_when_the_source_is_inadmissible_should_fail_with_the_first_refusal_and_send_the_others(
    tmp_path, monkeypatch
):
    # Arrange: too long for the dispatch, too many audio streams, a codec and a scan the profile refuses
    facts = _facts(videos=(_video(codec_name="wmv3", field_order="tt"),), audio_streams=5)
    harness = Harness(tmp_path, monkeypatch, facts=facts, request_changes={"/limits/max_source_duration_ms": 1000})

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("input_limits_exceeded", "duration_exceeded", False, "probing")
    assert result["error"]["detail"] == "duration 2000 ms exceeds the limit of 1000 ms"
    assert [diagnostic["code"] for diagnostic in result["diagnostics"]] == [
        "too_many_audio_streams", "unsupported_video_codec", "interlaced_unsupported"
    ]
    assert "wmv3" in result["diagnostics"][1]["detail"]


def test_run_when_the_source_is_inadmissible_should_download_and_store_nothing(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, facts=_facts(videos=(_video(codec_name="wmv3"),)))

    # Act
    result = harness.result()

    # Assert: the one read is the preflight's
    assert _error_of(result) == ("input_invalid", "unsupported_video_codec", False, "probing")
    assert len(harness.s3.get_calls) == 1
    assert harness.s3.put_calls == []
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


def test_run_when_the_source_is_inadmissible_should_still_say_what_it_read_and_what_would_have_encoded(
    tmp_path, monkeypatch
):
    # Arrange
    harness = Harness(tmp_path, monkeypatch, facts=_facts(videos=(_video(codec_name="wmv3"),)))

    # Act
    result = harness.result()

    # Assert
    assert result["source"] == {
        "source_id": harness.claimed.dispatch.source.source_id,
        "bytes_read": PROBED_BYTES,
        "etag_observed": SOURCE_ETAG,
    }
    assert result["encoder"]["ffmpeg_version"] == FFMPEG_VERSION


def test_run_when_the_source_is_refused_for_more_reasons_than_a_result_holds_should_send_the_first_eight_others(
    tmp_path, monkeypatch
):
    # Arrange: two video streams, each refused for its codec, size, rate, scan and transfer, and more
    refused = {
        "codec_name": "wmv3",
        "width": 7680,
        "height": 4320,
        "display_width": 7680,
        "display_height": 4320,
        "frame_rate_num": 240,
        "field_order": "tt",
        "color_transfer": "smpte2084",
    }
    facts = _facts(videos=(_video(0, **refused), _video(1, **refused)), audio_streams=5, duration_estimated=True)
    harness = Harness(tmp_path, monkeypatch, facts=facts)
    limits = harness.claimed.dispatch.limits
    every_refusal = decide_admission(
        facts,
        size_bytes=len(SOURCE_OBJECT),
        profile=PILOT_PROFILE,
        limits=RequestLimits(limits.max_source_bytes, limits.max_source_duration_ms),
    ).refusals

    # Act
    result = harness.result()

    # Assert
    assert len(every_refusal) > 9
    assert result["error"]["code"] == every_refusal[0].code
    assert [diagnostic["code"] for diagnostic in result["diagnostics"]] == [
        refusal.code for refusal in every_refusal[1:9]
    ]


def test_run_when_a_requested_rendition_is_not_in_the_profile_should_fail_before_the_download(tmp_path, monkeypatch):
    # Arrange: the worker holds a profile whose ladder ends at 1080p; the dispatch asks for 720p, 480p and 360p
    encode = PILOT_PROFILE.encode
    held = dataclasses.replace(PILOT_PROFILE, encode=dataclasses.replace(encode, ladder=encode.ladder[:3]))
    harness = Harness(tmp_path, monkeypatch, profile=held)

    # Act
    result = harness.result()

    # Assert: the one read is the preflight's
    assert _error_of(result) == ("configuration_error", "rendition_not_in_profile", False, "probing")
    assert len(harness.s3.get_calls) == 1
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []
    assert [message["stage"] for message in harness.progress] == ["probing"]


@pytest.mark.parametrize(
    ("limit", "value", "code"),
    [
        pytest.param("max_output_bytes", 1000, "output_too_large", id="too-few-bytes"),
        pytest.param("max_artifact_count", 1, "too_many_artifacts", id="too-few-files"),
    ],
)
def test_run_when_the_dispatchs_output_limits_cannot_hold_the_package_should_fail_before_the_download(
    tmp_path, monkeypatch, limit, value, code
):
    # Arrange: less room than the master playlist, the thumbnail and the manifest alone are kept
    harness = Harness(tmp_path, monkeypatch, request_changes={f"/limits/{limit}": value})

    # Act
    result = harness.result()

    # Assert: the one read is the preflight's
    assert _error_of(result) == ("input_limits_exceeded", code, False, "probing")
    assert len(harness.s3.get_calls) == 1
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


def test_run_when_the_stored_object_has_another_etag_than_the_dispatch_should_fail_and_store_nothing(harness):
    # Arrange: the object was replaced after the Hub recorded it
    harness.s3.hold(SOURCE_BUCKET, SOURCE_KEY, SOURCE_OBJECT, OTHER_ETAG)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("source_integrity_mismatch", "source_changed", False, "probing")
    assert "source" not in result
    assert harness.s3.put_calls == []
    assert harness.stages.encodes == []


def test_run_when_the_object_changes_after_the_preflight_should_fail_the_download_and_store_nothing(harness):
    # Arrange
    harness.stages.after_preflight = lambda: harness.s3.hold(SOURCE_BUCKET, SOURCE_KEY, SOURCE_OBJECT, OTHER_ETAG)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("source_integrity_mismatch", "source_changed", False, "downloading")
    assert harness.s3.put_calls == []
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


def test_run_when_the_object_changes_after_the_preflight_should_claim_no_observation_of_the_source(harness):
    # Arrange: the preflight reads the recorded object, which is replaced before the download reads it
    harness.stages.after_preflight = lambda: harness.s3.hold(SOURCE_BUCKET, SOURCE_KEY, SOURCE_OBJECT, OTHER_ETAG)

    # Act
    result = harness.result()

    # Assert: the object is no longer the recorded one, so the result says nothing of what was read of
    # it, and it does not give the recorded ETag as the one observed
    assert _error_of(result)[:2] == ("source_integrity_mismatch", "source_changed")
    assert "source" not in result
    assert result["encoder"]["ffmpeg_version"] == FFMPEG_VERSION


def test_run_when_the_object_turns_out_shorter_after_the_preflight_should_still_say_what_it_read(harness):
    # Arrange: the preflight reads the recorded object; the download then meets one of the recorded ETag
    # that holds fewer bytes than recorded, which is another failure of the same class
    shorter = SOURCE_OBJECT[: len(SOURCE_OBJECT) // 2]
    harness.stages.after_preflight = lambda: harness.s3.hold(SOURCE_BUCKET, SOURCE_KEY, shorter, SOURCE_ETAG)

    # Act
    result = harness.result()

    # Assert: only a changed object goes without the block
    assert _error_of(result) == ("source_integrity_mismatch", "source_size_mismatch", False, "downloading")
    assert result["source"] == {
        "source_id": harness.claimed.dispatch.source.source_id,
        "bytes_read": PROBED_BYTES,
        "etag_observed": SOURCE_ETAG,
    }


def test_run_when_it_fails_after_the_preflight_read_the_whole_object_should_not_report_more_than_it_holds(harness):
    # Arrange: the preflight reads the object entirely, the download one chunk more, then it is refused for good
    harness.stages.probed_bytes = len(SOURCE_OBJECT)
    harness.s3.fail_get(3, client_error("AccessDenied", HTTP_FORBIDDEN, "GetObject"))

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("source_unavailable", "source_access_denied", False, "downloading")
    assert harness.bytes_served > len(SOURCE_OBJECT)
    assert result["source"]["bytes_read"] == len(SOURCE_OBJECT)


def test_run_when_the_download_fails_midway_should_say_how_much_of_the_source_it_read(harness):
    # Arrange: the preflight's read and the first chunk arrive, the second chunk is refused for good
    harness.s3.fail_get(3, client_error("AccessDenied", HTTP_FORBIDDEN, "GetObject"))

    # Act
    result = harness.result()

    # Assert: the first chunk holds the bytes the preflight read, and they are counted once
    assert _error_of(result) == ("source_unavailable", "source_access_denied", False, "downloading")
    assert result["source"]["bytes_read"] == DOWNLOAD_CHUNK_BYTES
    assert harness.left_in_scratch == []


def test_run_when_the_download_fails_before_it_wrote_what_the_preflight_read_should_report_the_preflights_bytes(
    harness,
):
    # Arrange: the preflight reads two chunks' worth of the object; the download writes its first chunk,
    # and its second is refused for good
    harness.stages.probed_bytes = 2 * DOWNLOAD_CHUNK_BYTES
    harness.s3.fail_get(3, client_error("AccessDenied", HTTP_FORBIDDEN, "GetObject"))

    # Act
    result = harness.result()

    # Assert: the larger of the two, never their sum
    assert _error_of(result) == ("source_unavailable", "source_access_denied", False, "downloading")
    assert result["source"]["bytes_read"] == 2 * DOWNLOAD_CHUNK_BYTES


# --- a scratch directory that cannot be made ------------------------------------------------------


def _fail_the_making_of_the_scratch_directory(monkeypatch: pytest.MonkeyPatch, place: str, error: OSError) -> None:
    """Make the execution's own directory fail to be made with `error`, or the second of the directories
    inside it, once the directory itself and the first of them exist."""
    if place == THE_SCRATCH_ROOT:
        monkeypatch.setattr(runner_module.tempfile, "mkdtemp", _raising(error))
        return
    make = Path.mkdir

    def make_all_but_the_output_directory(directory: Path, *arguments: Any, **named_arguments: Any) -> None:
        if directory.name == runner_module.OUTPUT_DIR_NAME:
            raise error
        make(directory, *arguments, **named_arguments)

    monkeypatch.setattr(Path, "mkdir", make_all_but_the_output_directory)


@pytest.mark.parametrize("place", [THE_SCRATCH_ROOT, INSIDE_THE_SCRATCH_ROOT])
@pytest.mark.parametrize(
    ("error", "code", "detail"),
    [
        pytest.param(
            OSError(errno.ENOSPC, "No space left on device", PATH_CANARY), "scratch_full", SCRATCH_FULL_DETAIL,
            id="a-full-disk",
        ),
        pytest.param(
            OSError(errno.EDQUOT, "Disk quota exceeded", PATH_CANARY), "scratch_full", SCRATCH_FULL_DETAIL,
            id="a-spent-quota",
        ),
        pytest.param(
            FileNotFoundError(errno.ENOENT, "No such file or directory", PATH_CANARY),
            "scratch_create_failed",
            SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="ENOENT"),
            id="a-missing-directory",
        ),
        pytest.param(
            PermissionError(errno.EACCES, "Permission denied", PATH_CANARY),
            "scratch_create_failed",
            SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="EACCES"),
            id="a-forbidden-directory",
        ),
        pytest.param(
            PermissionError(errno.EPERM, "Operation not permitted", PATH_CANARY),
            "scratch_create_failed",
            SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="EPERM"),
            id="an-operation-that-is-not-permitted",
        ),
        pytest.param(
            OSError(errno.EMFILE, "Too many open files", PATH_CANARY),
            "scratch_create_failed",
            SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="EMFILE"),
            id="no-descriptor-left",
        ),
        pytest.param(
            OSError(errno.EIO, "Input/output error", PATH_CANARY),
            "scratch_create_failed",
            SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="EIO"),
            id="an-error-of-the-disk",
        ),
        pytest.param(
            OSError(), "scratch_create_failed", SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="unknown"),
            id="an-os-error-without-an-errno",
        ),
    ],
)
def test_run_when_its_scratch_directory_cannot_be_made_should_fail_retryable_at_downloading_and_leave_nothing(
    harness, monkeypatch, place, error, code, detail
):
    # Arrange
    _fail_the_making_of_the_scratch_directory(monkeypatch, place, error)

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert: never a bug and never final, a directory that is missing or may not be written included:
    # the process start found the scratch root usable, so this is the one worker's trouble after its
    # start, and another start is worth it; and whatever was made of the directory is gone
    assert _error_of(result) == ("resource_exhausted", code, True, "downloading")
    assert result["error"]["detail"] == detail
    assert harness.left_in_scratch == []
    assert harness.stages.encodes == [] and harness.s3.put_calls == []


@pytest.mark.parametrize("place", [THE_SCRATCH_ROOT, INSIDE_THE_SCRATCH_ROOT])
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError(errno.ENOSPC, "No space left on device", PATH_CANARY), id="a-full-disk"),
        pytest.param(PermissionError(errno.EACCES, "Permission denied", PATH_CANARY), id="a-forbidden-directory"),
        pytest.param(OSError(errno.EMFILE, "Too many open files", PATH_CANARY), id="no-descriptor-left"),
    ],
)
def test_run_when_its_scratch_directory_cannot_be_made_should_tell_the_errno_name_and_never_the_errors_text(
    harness, monkeypatch, caplog, place, error
):
    # Arrange: an error whose file name is a path
    caplog.set_level(logging.DEBUG)
    _fail_the_making_of_the_scratch_directory(monkeypatch, place, error)
    as_a_handler_writes_it = logging.Formatter()

    # Act
    sealed = harness.run()

    # Assert
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    assert PATH_CANARY.encode("ascii") not in sealed.body
    assert all(PATH_CANARY not in line for line in written)
    assert all(record.exc_info is None for record in caplog.records)
    assert any(
        "scratch directory could not be set up" in line and errno.errorcode[error.errno] in line for line in written
    )


def test_run_when_a_directory_inside_its_scratch_directory_cannot_be_made_should_remove_what_was_made_by_then(
    harness, monkeypatch
):
    # Arrange: the execution's directory and the first directory inside it are made, the second cannot
    # be; note what the execution's directory holds at that moment
    held_at_the_failure: list[list[str]] = []
    make = Path.mkdir

    def make_all_but_the_output_directory(directory: Path, *arguments: Any, **named_arguments: Any) -> None:
        if directory.name == runner_module.OUTPUT_DIR_NAME:
            held_at_the_failure.append(sorted(path.name for path in directory.parent.iterdir()))
            raise OSError(errno.EIO, "Input/output error")
        make(directory, *arguments, **named_arguments)

    monkeypatch.setattr(Path, "mkdir", make_all_but_the_output_directory)

    # Act
    result = harness.result()

    # Assert
    assert held_at_the_failure == [[runner_module.SOURCE_DIR_NAME]]
    assert result["message_kind"] == FAILED_KIND
    assert harness.left_in_scratch == []


def test_run_when_it_raises_after_its_scratch_directory_was_made_should_still_remove_it(harness):
    # Arrange: an execution id that aliases the video's id fails every self-check: one of them ends the
    # job once it has encoded in its scratch directory, and the failed result's then makes run() raise
    dispatch_identity = harness.claimed.dispatch.identity
    harness.regrant(
        identity=ExecutionIdentity.of(dispatch_identity, execution_id=dispatch_identity.video_uuid, execution_fence=2)
    )

    # Act
    with pytest.raises(SelfCheckFailed):
        harness.run()

    # Assert
    used = harness.stages.encodes[0].output_root
    assert harness.scratch_root in used.parents
    assert harness.left_in_scratch == []


# --- the lock on the scratch directory --------------------------------------------------------------


def test_run_when_it_works_in_its_scratch_directory_should_hold_its_lock_from_its_making_to_its_removal(
    harness, monkeypatch
):
    # Arrange: note whether the execution's directory is locked when the first directory is made inside
    # it, when a rendition is encoded in it, and when it is removed
    locked: dict[str, bool] = {}
    make, encode, remove_tree = Path.mkdir, harness.stages.encode_rendition, shutil.rmtree

    def make_and_note(directory: Path, *arguments: Any, **named_arguments: Any) -> None:
        if directory.parent.parent == harness.scratch_root:
            locked.setdefault("before anything is made inside it", _is_locked(directory.parent))
        make(directory, *arguments, **named_arguments)

    def encode_and_note(*arguments: Any, **named_arguments: Any) -> RenditionOutput:
        locked["during an encode"] = _is_locked(named_arguments["output_root"].parent)
        return encode(*arguments, **named_arguments)

    def remove_and_note(path: Any, *arguments: Any, **named_arguments: Any) -> None:
        if Path(path).parent == harness.scratch_root:
            locked["while it is removed"] = _is_locked(Path(path))
        remove_tree(path, *arguments, **named_arguments)

    monkeypatch.setattr(Path, "mkdir", make_and_note)
    monkeypatch.setattr(runner_module, "encode_rendition", encode_and_note)
    monkeypatch.setattr(runner_module.shutil, "rmtree", remove_and_note)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert locked == {
        "before anything is made inside it": True,
        "during an encode": True,
        "while it is removed": True,
    }
    assert harness.left_in_scratch == []


def test_run_when_another_process_removes_leftovers_during_an_encode_should_keep_its_own_scratch_directory(
    harness, monkeypatch
):
    # Arrange: a killed process left a directory beside the execution's. At the first encode the
    # execution's own has gone unchanged for two hours, as during a long encode, and another process
    # that starts on the same scratch root removes the leftovers. That process is a thread here, which
    # the encode waits for only a while: a removal that waited for the execution's lock would otherwise
    # wait for an execution that waits for it
    long_ago_s = time.time() - TWO_HOURS_S
    leftover = harness.scratch_root / (scratch_prefix() + "killed")
    (leftover / "source").mkdir(parents=True)
    os.utime(leftover, (long_ago_s, long_ago_s))
    removed: list[int] = []
    removed_by_the_time_the_encode_went_on: list[list[int]] = []
    another_process = threading.Thread(
        target=lambda: removed.append(remove_leftover_scratch(harness.scratch_root)), daemon=True
    )
    encode = harness.stages.encode_rendition

    def encode_while_another_process_starts(*arguments: Any, **named_arguments: Any) -> RenditionOutput:
        if not removed_by_the_time_the_encode_went_on:
            os.utime(named_arguments["output_root"].parent, (long_ago_s, long_ago_s))
            another_process.start()
            another_process.join(LOCK_RELEASE_S)
            removed_by_the_time_the_encode_went_on.append(list(removed))
        return encode(*arguments, **named_arguments)

    monkeypatch.setattr(runner_module, "encode_rendition", encode_while_another_process_starts)

    # Act
    try:
        result = harness.result()
    finally:
        if another_process.is_alive():
            another_process.join(LOCK_RELEASE_S)

    # Assert: the removal was done while the execution ran, the leftover is gone, the live execution's
    # directory was passed over, and the job went on in it
    assert removed_by_the_time_the_encode_went_on == [[1]]
    assert not leftover.exists()
    assert result["message_kind"] == COMPLETED_KIND
    assert MANIFEST_NAME in harness.uploaded


def test_run_when_its_scratch_directory_was_removed_should_have_closed_the_descriptor_of_its_lock(
    harness, monkeypatch
):
    # Arrange: note the descriptor the execution's lock lives on, and the directory it was opened on
    held: list[tuple[int, tuple[int, int]]] = []
    hold = runner_module.hold_scratch_lock

    def hold_and_note(directory: Path) -> int:
        descriptor = hold(directory)
        opened_on = os.fstat(descriptor)
        held.append((descriptor, (opened_on.st_dev, opened_on.st_ino)))
        return descriptor

    monkeypatch.setattr(runner_module, "hold_scratch_lock", hold_and_note)

    # Act
    result = harness.result()

    # Assert: a warm worker runs job after job, so a descriptor left open with each would use them up
    ((descriptor, directory),) = held
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.left_in_scratch == []
    assert not _is_still_open_on(descriptor, directory)


def test_run_when_its_scratch_directory_cannot_be_removed_should_let_go_of_its_lock_all_the_same(
    harness, monkeypatch
):
    # Arrange: the removal fails for the length of the run
    with monkeypatch.context() as during_the_run:
        during_the_run.setattr(runner_module.shutil, "rmtree", _raising(PermissionError(13, "denied", PATH_CANARY)))

        # Act
        result = harness.result()

    # Assert: what is left is locked by nobody, so the next start's removal of leftovers takes it once it is old
    (left,) = harness.scratch_root.iterdir()
    long_ago_s = time.time() - TWO_HOURS_S
    os.utime(left, (long_ago_s, long_ago_s))
    assert result["message_kind"] == COMPLETED_KIND
    assert not _is_locked(left)
    assert remove_leftover_scratch(harness.scratch_root) == 1
    assert harness.left_in_scratch == []


def test_run_when_another_holds_the_lock_of_its_new_scratch_directory_should_fail_retryable_and_remove_it(
    harness, monkeypatch
):
    # Arrange: the instant the execution's directory is made, another process holds its lock (for a
    # while at most: an execution that waited for the lock would otherwise never return)
    letting_go: list[Callable[[], None]] = []
    make = tempfile.mkdtemp

    def make_a_directory_another_locks(*arguments: Any, **named_arguments: Any) -> str:
        made = make(*arguments, **named_arguments)
        descriptor = _lock_as_another_would(Path(made)) if Path(made).parent == harness.scratch_root else None
        if descriptor is not None:
            letting_go.append(held_at_most_a_while(descriptor))
        return made

    monkeypatch.setattr(runner_module.tempfile, "mkdtemp", make_a_directory_another_locks)

    # Act
    try:
        result = harness.result()
    finally:
        for let_go in letting_go:
            let_go()

    # Assert: the execution does not work in a directory it could not lock; a passing fault, typed
    assert len(letting_go) == 1
    assert _error_of(result) == ("resource_exhausted", "scratch_create_failed", True, "downloading")
    assert result["error"]["detail"] == SCRATCH_NOT_SET_UP_DETAIL.format(
        errno_name=errno.errorcode[errno.EWOULDBLOCK]
    )
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


def test_run_when_its_new_scratch_directory_is_gone_before_its_lock_is_taken_should_fail_retryable_and_leave_nothing(
    harness, monkeypatch
):
    # Arrange: the execution's directory is removed the instant it was made, before the execution locks it
    vanished: list[str] = []
    make = tempfile.mkdtemp

    def make_a_directory_that_vanishes(*arguments: Any, **named_arguments: Any) -> str:
        made = make(*arguments, **named_arguments)
        if Path(made).parent == harness.scratch_root:
            os.rmdir(made)
            vanished.append(made)
        return made

    monkeypatch.setattr(runner_module.tempfile, "mkdtemp", make_a_directory_that_vanishes)

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert: the process start found the scratch root usable, so this is the one worker's trouble
    # after its start: worth another start, never final, and the execution worked in no directory
    assert len(vanished) == 1
    assert _error_of(result) == ("resource_exhausted", "scratch_create_failed", True, "downloading")
    assert result["error"]["detail"] == SCRATCH_NOT_SET_UP_DETAIL.format(errno_name="ENOENT")
    assert harness.stages.encodes == []
    assert harness.left_in_scratch == []


# --- failures and bugs in every stage ---------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_run_when_a_stage_fails_should_return_a_failed_result_with_that_failure_and_that_stage(
    harness, monkeypatch, stage
):
    # Arrange
    monkeypatch.setattr(runner_module, STAGE_FUNCTIONS[stage], _raising(WorkerFailure(STAGE_FAILURE)))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == FAILED_KIND
    assert result["error"] == {
        "class": STAGE_FAILURE.error_class,
        "code": STAGE_FAILURE.code,
        "retryable": True,
        "stage": stage,
        "detail": STAGE_FAILURE.detail,
    }
    assert f"{harness.claimed.grant.output_prefix}/{MANIFEST_NAME}" not in [call.key for call in harness.s3.put_calls]
    assert harness.left_in_scratch == []


@pytest.mark.parametrize("stage", STAGES)
def test_run_when_a_stage_raises_an_unexpected_error_should_fail_with_its_class_name_only(
    harness, monkeypatch, caplog, stage
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(runner_module, STAGE_FUNCTIONS[stage], a_stage_with_a_bug)

    # Act
    sealed = harness.run()

    # Assert
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert _error_of(result) == ("internal_error", "unexpected_error", True, stage)
    assert result["error"]["detail"] == "RuntimeError"
    for canary in (CANARY, CAUSE_CANARY):
        assert canary.encode("ascii") not in sealed.body
        assert all(canary.encode("ascii") not in call.body for call in harness.transport.calls)
    assert harness.left_in_scratch == []


@pytest.mark.parametrize("stage", STAGES)
def test_run_when_a_stage_raises_an_unexpected_error_should_log_its_class_and_stack_and_never_its_text(
    harness, monkeypatch, caplog, stage
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(runner_module, STAGE_FUNCTIONS[stage], a_stage_with_a_bug)
    as_a_handler_writes_it = logging.Formatter()

    # Act
    harness.run()

    # Assert: neither the error's text nor its cause's, in a message or in anything a handler would append to it
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    for canary in (CANARY, CAUSE_CANARY):
        assert all(canary not in record.getMessage() for record in caplog.records)
        assert all(canary not in line for line in written)
        assert canary not in caplog.text
    assert all(record.exc_info is None and record.exc_text is None for record in caplog.records)
    (bug,) = [record for record in caplog.records if "RuntimeError" in record.getMessage()]
    assert bug.levelno == logging.ERROR
    assert "in a_stage_with_a_bug" in bug.getMessage()
    assert "chained to KeyError" in bug.getMessage()


def test_run_when_an_unexpected_error_comes_before_any_stage_should_still_return_a_failed_result(
    tmp_path, monkeypatch
):
    # Arrange: the configured ffmpeg names itself with text no result can carry
    harness = Harness(
        tmp_path, monkeypatch, ffmpeg_version=lambda deadline: "see https://ffmpeg.example.invalid/version"
    )

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("internal_error", "unexpected_error", True, None)
    assert result["error"]["detail"] == "ValueError"
    assert harness.s3.request_count == 0
    assert "encoder" not in result


def test_run_when_the_storage_refuses_the_generation_should_fail_at_uploading_without_the_manifest(harness):
    # Arrange: the storage refuses every write
    harness.s3.fail_put(1, client_error("AccessDenied", HTTP_FORBIDDEN, "PutObject"), and_every_later_one=True)

    # Act
    result = harness.result()

    # Assert: final, so that the Hub does not dispatch an encode again for a write that is never allowed,
    # and the refused object was sent once
    assert _error_of(result) == ("output_write_failed", "upload_refused", False, "uploading")
    assert len(harness.s3.put_calls) == 1
    assert harness.s3.stored == []
    assert result["source"]["bytes_read"] == len(SOURCE_OBJECT)


def test_run_when_the_storage_keeps_failing_a_write_for_now_should_fail_retryable_at_uploading(harness):
    # Arrange: the storage answers every write with an error of its own
    failing_for_now = client_error("InternalError", HTTP_INTERNAL_SERVER_ERROR, "PutObject")
    harness.s3.fail_put(1, failing_for_now, and_every_later_one=True)

    # Act
    result = harness.result()

    # Assert: worth another dispatch, and the manifest was never stored
    assert _error_of(result) == ("output_write_failed", "upload_failed", True, "uploading")
    assert harness.s3.stored == []


def test_run_when_a_put_fails_once_should_wait_on_the_runtimes_clock_and_complete(harness, fixed_factor):
    # Arrange: the storage answers the first write with an error of its own, and takes every later one
    harness.s3.fail_put(1, client_error("InternalError", HTTP_INTERNAL_SERVER_ERROR, "PutObject"))

    # Act
    result = harness.result()

    # Assert: the wait before the second request (0.5 s times the factor 0.75) was made through the
    # runtime's `sleep`, so it passed on the runtime's clock and not on the real one
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.clock.waits == [0.375]
    assert MANIFEST_NAME in harness.uploaded


def test_run_when_a_read_of_the_download_fails_once_should_wait_on_the_runtimes_clock_and_complete(
    harness, fixed_factor
):
    # Arrange: the preflight's read arrives; the first read of the download is answered with an error of
    # the storage's own, and every later one arrives
    harness.s3.fail_get(2, client_error("InternalError", HTTP_INTERNAL_SERVER_ERROR, "GetObject"))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.clock.waits == [0.375]
    assert result["source"]["bytes_read"] == len(SOURCE_OBJECT)


def test_run_when_every_put_stalls_should_fail_retryable_as_stalled_at_uploading_without_the_manifest(
    harness, monkeypatch, fixed_factor, never_answered
):
    # Arrange: a storage that takes every write and never answers it. From the upload on, a request may
    # take some 50 ms where the worker allows half a minute and more: only requests that are kept run
    # under it
    real_upload = runner_module.upload_generation
    answer = harness.s3.put_object
    put_started = threading.Event()

    def upload_under_a_short_ceiling(*arguments: Any, **named_arguments: Any) -> int:
        monkeypatch.setattr(storage_module, "REQUEST_BASE_S", SHORT_CEILING_S)
        return real_upload(*arguments, **named_arguments)

    def put_object_that_hangs(**request: Any) -> Any:
        put_started.set()
        never_answered.wait(HANG_RELEASE_S)
        return answer(**request)

    monkeypatch.setattr(runner_module, "upload_generation", upload_under_a_short_ceiling)
    harness.s3.put_object = put_object_that_hangs
    started = time.monotonic()

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert: a passing fault of the storage, worth another dispatch, not a deadline that would be final;
    # the three waits between the four requests passed on the runtime's clock
    assert _error_of(result) == ("output_write_failed", "upload_stalled", True, "uploading")
    assert result["error"]["detail"] == "an upload did not end within the time an object of its size may take"
    assert put_started.wait(HANG_RELEASE_S)
    assert harness.clock.waits == [0.375, 1.5, 6.0]
    assert harness.s3.stored == []
    assert time.monotonic() - started < HANG_RELEASE_S
    assert harness.left_in_scratch == []


def test_run_when_the_preflights_read_stalls_should_fail_retryable_as_stalled_at_probing(
    harness, monkeypatch, never_answered
):
    # Arrange: a storage that takes the preflight's read, the first request of the execution, and never
    # answers it; a request may take some 50 ms, and the start has its whole wall time left
    answer = harness.s3.get_object
    read_started = threading.Event()

    def get_object_that_hangs(**request: Any) -> Any:
        read_started.set()
        never_answered.wait(HANG_RELEASE_S)
        return answer(**request)

    harness.s3.get_object = get_object_that_hangs
    monkeypatch.setattr(storage_module, "REQUEST_BASE_S", SHORT_CEILING_S)
    started = time.monotonic()

    # Act
    result = harness.result()

    # Assert: the read was given up at its own ceiling, long before the deadline, and passes
    assert _error_of(result) == ("source_unavailable", "source_read_stalled", True, "probing")
    assert result["error"]["detail"] == "a read of the source did not end within the time a read of its size may take"
    assert read_started.wait(HANG_RELEASE_S)
    assert time.monotonic() - started < HANG_RELEASE_S
    assert harness.stages.encodes == [] and harness.s3.stored == []


def test_run_when_the_preflight_runs_out_of_its_time_on_a_slow_storage_should_fail_retryable_at_probing(
    harness, monkeypatch
):
    # Arrange: the preflight ends as the real one does when its own time budget ran out while the
    # reads of the source took most of it, and the start has its whole wall time left
    monkeypatch.setattr(runner_module, "run_preflight", _raising(WorkerFailure(PROBE_SOURCE_SLOW)))

    # Act: result() returns once the worker's own validator sealed the result and the Hub's accepted it
    result = harness.result()

    # Assert: a passing fault of the storage, worth another dispatch, not a deadline that would be final;
    # the detail is the preflight's own sentence, with nothing about the start's wall time added to it
    assert result["message_kind"] == FAILED_KIND
    assert result["error"] == {
        "class": "source_unavailable",
        "code": "probe_source_slow",
        "retryable": True,
        "stage": "probing",
        "detail": "the preflight ran out of time while the storage was slow",
    }
    assert harness.stages.encodes == [] and harness.s3.stored == []
    assert harness.left_in_scratch == []


def test_run_when_a_read_of_the_preflight_never_returns_should_fail_at_probing_once_the_wall_time_ends(harness):
    # Arrange: a start with a fifth of a second left, and a storage that takes the preflight's read and
    # never answers it
    never_answered = threading.Event()
    answer = harness.s3.get_object

    def get_object_that_hangs(**request):
        never_answered.wait(HANG_RELEASE_S)
        return answer(**request)

    harness.s3.get_object = get_object_that_hangs
    claimed = dataclasses.replace(harness.claimed, wall_time_ends_at=harness.clock.monotonic() + 0.2)
    started = time.monotonic()

    # Act
    try:
        sealed = harness.runner.run(claimed, harness.runtime)
    finally:
        never_answered.set()

    # Assert: the read in flight was given up at the deadline, not waited out
    result = accepted_by_the_hub(sealed, claimed, now=harness.clock())
    assert _error_of(result) == ("deadline_exceeded", "source_read_deadline_exceeded", False, "probing")
    assert time.monotonic() - started < HANG_RELEASE_S
    assert harness.stages.encodes == [] and harness.s3.stored == []


def test_run_when_the_manifest_it_wrote_fails_its_self_check_should_fail_and_upload_nothing(harness, monkeypatch):
    # Arrange: the manifest on disk names another source than the dispatch's
    real_write_manifest = runner_module.write_manifest

    def write_a_manifest_of_another_source(**arguments: Any) -> Any:
        written = real_write_manifest(**arguments)
        path = arguments["output_root"] / written.path
        path.write_bytes(path.read_bytes().replace(b"66666666-6666-7666-8666-666666666666", OTHER_UUID.encode("ascii")))
        return written

    monkeypatch.setattr(runner_module, "write_manifest", write_a_manifest_of_another_source)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("internal_error", "manifest_self_check_failed", False, "finalizing")
    assert "semantic/source_mismatch" in result["error"]["detail"]
    assert OTHER_UUID not in result["error"]["detail"]
    assert harness.s3.put_calls == []


def test_run_when_the_completed_result_would_fail_its_self_check_should_fail_at_finalizing_and_upload_nothing(
    harness,
):
    # Arrange: a grant whose generation is not its execution, which no accepted grant is
    harness.regrant(generation_id=OTHER_UUID)

    # Act
    result = harness.result()

    # Assert: found once the generation was finished and before the uploading stage began
    assert result["message_kind"] == FAILED_KIND
    assert _error_of(result) == ("internal_error", "result_self_check_failed", False, "finalizing")
    assert "generation_execution_mismatch" in result["error"]["detail"]
    assert harness.s3.put_calls == []
    assert [message["stage"] for message in harness.progress][-1] == "finalizing"


def test_run_when_the_result_built_after_the_upload_fails_its_self_check_should_return_a_failed_result_at_uploading(
    harness, monkeypatch
):
    # Arrange: the completed result passes its check before the upload, and the one built anew once the
    # generation is stored does not. Nothing a dispatch or a grant can hold brings that about, so the
    # runner's seal is stood in for: the second completed result it is handed is sealed as one of
    # another generation than the execution's, which the worker's own validator refuses
    seal = runner_module.seal_message
    completed_results: list[str] = []

    def seal_the_second_completed_result_as_another_generations(
        message: Mapping[str, Any], *arguments: Any, **named_arguments: Any
    ) -> SealedMessage:
        if message["message_kind"] == COMPLETED_KIND:
            completed_results.append(message["message_id"])
            if len(completed_results) == 2:
                message = {**message, "generation_id": OTHER_UUID}
        return seal(message, *arguments, **named_arguments)

    monkeypatch.setattr(runner_module, "seal_message", seal_the_second_completed_result_as_another_generations)

    # Act: result() returns once the Hub's validator accepted the signed bytes
    result = harness.result()

    # Assert: the Hub still gets a signed result, a failed one that is final, of the stage the execution
    # was in; the generation it must not take for completed is stored
    assert len(completed_results) == 2
    assert result["message_kind"] == FAILED_KIND
    assert _error_of(result) == ("internal_error", "result_self_check_failed", False, "uploading")
    assert "generation_execution_mismatch" in result["error"]["detail"]
    assert MANIFEST_NAME in harness.uploaded


def _completed_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The log lines that say an execution completed."""
    return [record.getMessage() for record in caplog.records if " completed: " in record.getMessage()]


def test_run_when_it_completes_should_log_that_once(harness, caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)

    # Act
    result = harness.result()

    # Assert: the result that was only built to be checked before the upload is no completion
    assert result["message_kind"] == COMPLETED_KIND
    assert len(_completed_lines(caplog)) == 1


def test_run_when_the_upload_fails_should_never_log_that_the_execution_completed(harness, caplog):
    # Arrange: the completed result passes its check before the upload, then the storage refuses every write
    caplog.set_level(logging.DEBUG)
    harness.s3.fail_put(1, client_error("AccessDenied", HTTP_FORBIDDEN, "PutObject"), and_every_later_one=True)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result) == ("output_write_failed", "upload_refused", False, "uploading")
    assert _completed_lines(caplog) == []


def test_run_when_the_upload_took_time_should_date_the_result_after_it_and_give_it_an_id_drawn_after_it(harness):
    # Arrange: the storage takes a minute to answer the first write
    answer = harness.s3.put_object

    def put_object_that_takes_a_minute_once(**request: Any) -> Any:
        if not harness.s3.put_calls:
            harness.clock.advance(60)
        return answer(**request)

    harness.s3.put_object = put_object_that_takes_a_minute_once

    # Act
    result = harness.result()

    # Assert: the result is built anew once the generation is stored, not the one checked before the upload
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.clock.elapsed_s == 60
    assert result["sent_at"] == timestamp_text(harness.clock())
    assert result["message_id"] == harness.drawn_ids[-1]


def test_run_when_the_failed_result_itself_cannot_be_sealed_should_raise(tmp_path, monkeypatch):
    # Arrange: an execution id that aliases the video's id fails every self-check; the unknown ref fails the job
    harness = Harness(tmp_path, monkeypatch, request_changes={"/source/credential_ref": UNKNOWN_CREDENTIAL_REF})
    dispatch_identity = harness.claimed.dispatch.identity
    harness.regrant(
        identity=ExecutionIdentity.of(dispatch_identity, execution_id=dispatch_identity.video_uuid, execution_fence=2)
    )

    # Act and Assert
    with pytest.raises(SelfCheckFailed):
        harness.run()


def test_run_when_the_scratch_directory_cannot_be_removed_should_still_return_the_result(harness, monkeypatch, caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(runner_module.shutil, "rmtree", _raising(PermissionError(13, "denied", PATH_CANARY)))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert any("PermissionError" in record.getMessage() for record in caplog.records)
    assert PATH_CANARY not in caplog.text


# --- the storage clients are closed -----------------------------------------------------------------


def test_run_when_it_completes_should_close_both_storage_clients_once_after_their_last_request(
    harness, no_request_abandoned
):
    # Arrange
    source, output = _give_closable_clients(harness)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert (source.closes, output.closes) == (1, 1)
    assert source.requests_at_its_close == source.request_count > 0
    assert output.requests_at_its_close == output.request_count > 0


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(_raising(WorkerFailure(STAGE_FAILURE)), id="a-failed-stage"),
        pytest.param(a_stage_with_a_bug, id="a-bug"),
    ],
)
def test_run_when_it_fails_should_still_close_both_storage_clients(harness, monkeypatch, no_request_abandoned, ending):
    # Arrange: the encode fails, after both clients were made and the source was read
    source, output = _give_closable_clients(harness)
    monkeypatch.setattr(runner_module, "encode_rendition", ending)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == FAILED_KIND
    assert (source.closes, output.closes) == (1, 1)


def test_run_when_it_raises_should_still_close_both_storage_clients(harness, no_request_abandoned):
    # Arrange: an execution id that aliases the video's id fails every self-check: one of them ends the
    # job once both clients were made and the source was read, and the failed result's then makes
    # run() raise
    source, output = _give_closable_clients(harness)
    dispatch_identity = harness.claimed.dispatch.identity
    harness.regrant(
        identity=ExecutionIdentity.of(dispatch_identity, execution_id=dispatch_identity.video_uuid, execution_fence=2)
    )

    # Act
    with pytest.raises(SelfCheckFailed):
        harness.run()

    # Assert: closed once each, though no result came back
    assert source.request_count > 0
    assert (source.closes, output.closes) == (1, 1)


def test_run_when_the_outputs_client_cannot_be_made_should_close_the_sources(harness, no_request_abandoned):
    # Arrange: the source's client is made first, then the output's factory refuses
    source, _ = _give_closable_clients(harness)
    harness.output_client_error = ValueError(CLIENT_CANARY)

    # Act
    result = harness.result()

    # Assert
    assert _error_of(result)[:2] == ("configuration_error", "storage_client_unavailable")
    assert source.closes == 1


def test_run_when_a_storage_client_has_no_close_should_leave_it_alone_and_close_the_other(
    harness, caplog, no_request_abandoned
):
    # Arrange: the source's client is one that cannot be closed
    caplog.set_level(logging.DEBUG)
    output = ClosableS3()
    harness.output_s3 = output

    # Act
    result = harness.result()

    # Assert: left alone is not a close that was tried and failed: no line says a client could not be closed
    assert result["message_kind"] == COMPLETED_KIND
    assert not hasattr(harness.s3, "close")
    assert output.closes == 1
    assert [record.getMessage() for record in caplog.records if "could not be closed" in record.getMessage()] == []


def test_run_when_closing_a_storage_client_fails_should_return_the_result_and_log_the_errors_class_only(
    harness, caplog, no_request_abandoned
):
    # Arrange: the close of the source's client fails, in words that quote the endpoint
    caplog.set_level(logging.DEBUG)
    source, output = _give_closable_clients(harness)
    source.close_error = RuntimeError(CLIENT_CANARY)
    as_a_handler_writes_it = logging.Formatter()

    # Act
    sealed = harness.run()

    # Assert: the result stands, the other client is closed all the same, and the error's text is told nowhere
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    assert result["message_kind"] == COMPLETED_KIND
    assert (source.closes, output.closes) == (1, 1)
    assert any("storage client could not be closed: RuntimeError" in line for line in written)
    assert all(CLIENT_CANARY not in line for line in written)
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("abandoned", ["before-the-job", "while-the-job-ran"])
def test_run_when_the_process_has_abandoned_a_storage_request_should_leave_both_storage_clients_open(
    harness, monkeypatch, no_request_abandoned, abandoned
):
    # Arrange: the process's count of abandoned requests is above zero when the job ends: such a request
    # may still be using a client
    source, output = _give_closable_clients(harness)

    def count_one_abandoned_request() -> None:
        monkeypatch.setattr(storage_module, "_abandoned_requests", 1)

    if abandoned == "before-the-job":
        count_one_abandoned_request()
    else:
        harness.stages.after_preflight = count_one_abandoned_request

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert storage_module.abandoned_requests() == 1
    assert (source.closes, output.closes) == (0, 0)


# --- a log line whose handler fails -------------------------------------------------------------------
# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while the error it tells of
# is still being handled would put that error's own text into the report. `failing_log` makes every
# line's handler fail; the result is what it is without it.


def test_run_when_a_log_handler_fails_on_the_line_of_a_bug_should_keep_the_bugs_text_out_of_standard_error(
    harness, monkeypatch, capsys, failing_log
):
    # Arrange: a stage fails on a bug that has a cause, each with a text of its own
    monkeypatch.setattr(runner_module, "encode_rendition", a_stage_with_a_bug)

    # Act
    sealed = harness.run()

    # Assert: logging did report its failure on the bug's line, with nothing of the bug's or its cause's text
    written = reported_log_failures(capsys, "ended on an unexpected")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert [canary for canary in (CANARY, CAUSE_CANARY) if canary in written] == []
    assert _error_of(result) == ("internal_error", "unexpected_error", True, "transcoding")
    assert result["error"]["detail"] == "RuntimeError"


@pytest.mark.parametrize("side", ["source", "output"])
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            ValueError(CLIENT_CANARY),
            ("configuration_error", "storage_client_unavailable", False, None),
            id="a-profile-no-client-can-be-made-of",
        ),
        pytest.param(
            OSError(errno.EMFILE, "Too many open files", CLIENT_CANARY),
            ("resource_exhausted", "storage_client_start_failed", True, None),
            id="a-passing-shortage",
        ),
    ],
)
def test_run_when_a_log_handler_fails_on_the_line_of_a_client_that_cannot_be_made_should_keep_the_errors_text_out(
    harness, capsys, failing_log, side, error, expected
):
    # Arrange: the factory of one side fails, in words that quote the endpoint
    setattr(harness, f"{side}_client_error", error)

    # Act
    sealed = harness.run()

    # Assert
    written = reported_log_failures(capsys, f"the {side} storage client could not be made")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert CLIENT_CANARY not in written
    assert _error_of(result) == expected


def test_run_when_a_log_handler_fails_on_the_line_of_a_client_that_cannot_be_closed_should_keep_the_errors_text_out(
    harness, capsys, failing_log, no_request_abandoned
):
    # Arrange: the close of the source's client fails, in words that quote the endpoint
    source, output = _give_closable_clients(harness)
    source.close_error = RuntimeError(CLIENT_CANARY)

    # Act
    sealed = harness.run()

    # Assert
    written = reported_log_failures(capsys, "a storage client could not be closed")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert CLIENT_CANARY not in written
    assert result["message_kind"] == COMPLETED_KIND
    assert (source.closes, output.closes) == (1, 1)


def test_run_when_a_log_handler_fails_on_the_line_of_a_scratch_directory_left_behind_should_keep_its_path_out(
    harness, monkeypatch, capsys, failing_log
):
    # Arrange: the removal of the execution's directory fails, in words that name a path
    monkeypatch.setattr(runner_module.shutil, "rmtree", _raising(PermissionError(13, "denied", PATH_CANARY)))

    # Act
    sealed = harness.run()

    # Assert
    written = reported_log_failures(capsys, "scratch directory could not be removed")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert PATH_CANARY not in written
    assert result["message_kind"] == COMPLETED_KIND


@pytest.mark.parametrize("place", [THE_SCRATCH_ROOT, INSIDE_THE_SCRATCH_ROOT])
def test_run_when_a_log_handler_fails_on_the_line_of_a_scratch_directory_not_set_up_should_keep_its_path_out(
    harness, monkeypatch, capsys, failing_log, place
):
    # Arrange: the execution's directory, or one inside it, cannot be made, in words that name a path
    full_disk = OSError(errno.ENOSPC, "No space left on device", PATH_CANARY)
    _fail_the_making_of_the_scratch_directory(monkeypatch, place, full_disk)

    # Act
    sealed = harness.run()

    # Assert
    written = reported_log_failures(capsys, "scratch directory could not be set up")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert PATH_CANARY not in written
    assert _error_of(result) == ("resource_exhausted", "scratch_full", True, "downloading")
    assert harness.left_in_scratch == []


@pytest.mark.parametrize(
    ("claimed", "runtime"),
    [pytest.param("dispatch", None, id="no-claimed-execution"), pytest.param(None, "runtime", id="no-runtime")],
)
def test_run_when_an_argument_has_the_wrong_type_should_raise_type_error(harness, claimed, runtime):
    # Arrange
    arguments = (harness.claimed if claimed is None else claimed, harness.runtime if runtime is None else runtime)

    # Act and Assert
    with pytest.raises(TypeError):
        harness.runner.run(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "wall_time_ends_at",
    [pytest.param(True, id="a-bool"), pytest.param("3600.0", id="text"), pytest.param(None, id="none")],
)
def test_claimed_execution_when_the_end_of_its_wall_time_is_no_number_should_raise_type_error(
    harness, wall_time_ends_at
):
    # Arrange
    claimed = harness.claimed

    # Act and Assert
    with pytest.raises(TypeError):
        dataclasses.replace(claimed, wall_time_ends_at=wall_time_ends_at)


@pytest.mark.parametrize(
    "wall_time_ends_at",
    [
        pytest.param(float("nan"), id="not-a-number"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("-inf"), id="minus-infinite"),
    ],
)
def test_claimed_execution_when_the_end_of_its_wall_time_is_not_finite_should_raise_value_error(
    harness, wall_time_ends_at
):
    # Arrange
    claimed = harness.claimed

    # Act and Assert
    with pytest.raises(ValueError):
        dataclasses.replace(claimed, wall_time_ends_at=wall_time_ends_at)


@pytest.mark.parametrize("wall_time_ends_at", [3600, 3600.5, -2.5], ids=["a-whole-number", "a-fraction", "negative"])
def test_claimed_execution_when_the_end_of_its_wall_time_is_a_finite_number_should_hold_it(
    harness, wall_time_ends_at
):
    # Act
    claimed = dataclasses.replace(harness.claimed, wall_time_ends_at=wall_time_ends_at)

    # Assert
    assert claimed.wall_time_ends_at == wall_time_ends_at


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"job_settings": {"image_digest": IMAGE_DIGEST}}, id="settings-as-a-mapping"),
        pytest.param({"source_client_factory": "storage_client"}, id="source-client-factory-not-callable"),
        pytest.param({"output_client_factory": "upload_client"}, id="output-client-factory-not-callable"),
        pytest.param({"ffmpeg_version": FFMPEG_VERSION}, id="ffmpeg-version-as-text"),
    ],
)
def test_runner_when_built_with_an_argument_of_the_wrong_type_should_raise_type_error(tmp_path, arguments):
    # Arrange
    valid: dict[str, Any] = {
        "job_settings": job_settings(tmp_path),
        "source_client_factory": lambda access: FakeS3(),
        "output_client_factory": lambda access: FakeS3(),
        "ffmpeg_version": lambda deadline: "7",
    }
    valid.update(arguments)

    # Act and Assert
    with pytest.raises(TypeError):
        JobRunner(valid.pop("job_settings"), **valid)


# --- secrets --------------------------------------------------------------------------------------


def _encodings_of(secret: bytes) -> list[bytes]:
    return [
        secret,
        secret.hex().encode("ascii"),
        secret.hex().upper().encode("ascii"),
        base64.b64encode(secret),
        repr(secret).encode("ascii"),
    ]


def test_run_when_the_dispatch_is_encrypted_should_hand_the_media_key_to_every_encode(harness):
    # Act
    harness.run()

    # Assert
    assert [call.media_key for call in harness.stages.encodes] == [MEDIA_KEY, MEDIA_KEY, MEDIA_KEY]


def test_run_when_the_transcoding_stage_has_ended_should_hold_no_media_key_in_the_stages_that_follow(
    harness, monkeypatch
):
    # Arrange: note what the grant holds when the thumbnail is made, when the manifest is written and at
    # every write to the storage
    held: dict[str, list[bytes | None]] = {"the thumbnail": [], "the manifest": [], "a write to the storage": []}

    def noting(what: str, call: Callable[..., Any]) -> Callable[..., Any]:
        def note_and_call(*arguments: Any, **named_arguments: Any) -> Any:
            held[what].append(harness.claimed.grant.media_key)
            return call(*arguments, **named_arguments)

        return note_and_call

    monkeypatch.setattr(runner_module, "make_thumbnail", noting("the thumbnail", harness.stages.make_thumbnail))
    monkeypatch.setattr(runner_module, "write_manifest", noting("the manifest", runner_module.write_manifest))
    harness.s3.put_object = noting("a write to the storage", harness.s3.put_object)

    # Act
    result = harness.result()

    # Assert: every encode had the key, and nothing after the last of them could read it
    assert result["message_kind"] == COMPLETED_KIND
    assert [call.media_key for call in harness.stages.encodes] == [MEDIA_KEY, MEDIA_KEY, MEDIA_KEY]
    assert held["the thumbnail"] == [None] and held["the manifest"] == [None]
    assert len(held["a write to the storage"]) > 1 and set(held["a write to the storage"]) == {None}
    assert harness.claimed.grant.media_key is None
    assert harness.claimed.grant.media_key_id == harness.claimed.dispatch.encryption.media_key_id


@pytest.mark.parametrize(
    "ending",
    [
        pytest.param(WorkerFailure(STAGE_FAILURE), id="a-failed-encode"),
        pytest.param(RuntimeError(CANARY), id="a-bug"),
    ],
)
def test_run_when_the_transcoding_stage_ends_on_a_failure_should_hold_no_media_key_when_the_result_is_built(
    harness, monkeypatch, ending
):
    # Arrange: the second rendition's encode fails; note what the grant holds each time an id is drawn
    # (for every progress message, and at last for the result)
    held_at_each_draw: list[bytes | None] = []
    encode, draw = harness.stages.encode_rendition, harness.runtime.new_uuid

    def encode_the_first_only(*arguments: Any, **named_arguments: Any) -> RenditionOutput:
        if harness.stages.encodes:
            raise ending
        return encode(*arguments, **named_arguments)

    def draw_and_note() -> str:
        held_at_each_draw.append(harness.claimed.grant.media_key)
        return draw()

    monkeypatch.setattr(runner_module, "encode_rendition", encode_the_first_only)
    harness.runtime = dataclasses.replace(harness.runtime, new_uuid=draw_and_note)

    # Act
    result = harness.result()

    # Assert: held while the stage ran, and let go of as soon as it ended, before the failed result
    assert _error_of(result)[3] == "transcoding"
    assert [call.media_key for call in harness.stages.encodes] == [MEDIA_KEY]
    assert MEDIA_KEY in held_at_each_draw
    assert held_at_each_draw[-1] is None
    assert harness.claimed.grant.media_key is None


class _EncodeSourceThatCannotBeMade:
    """Stands where the runner names EncodeSource: making one of a source's facts fails as a bug would."""

    from_facts = staticmethod(a_stage_with_a_bug)


def test_run_when_the_transcoding_stage_ends_before_its_first_encode_should_hold_no_media_key_when_the_result_is_built(
    harness, monkeypatch
):
    # Arrange: the stage has begun and fails before any encode, where it makes of the source's facts
    # what the encodes read; note what the grant holds each time an id is drawn (for every progress
    # message, and at last for the result)
    held_at_each_draw: list[bytes | None] = []
    draw = harness.runtime.new_uuid

    def draw_and_note() -> str:
        held_at_each_draw.append(harness.claimed.grant.media_key)
        return draw()

    monkeypatch.setattr(runner_module, "EncodeSource", _EncodeSourceThatCannotBeMade)
    harness.runtime = dataclasses.replace(harness.runtime, new_uuid=draw_and_note)

    # Act
    result = harness.result()

    # Assert: however the stage ends, the key is let go of as it ends, before the failed result is built
    assert _error_of(result) == ("internal_error", "unexpected_error", True, "transcoding")
    assert harness.stages.encodes == []
    assert MEDIA_KEY in held_at_each_draw
    assert held_at_each_draw[-1] is None
    assert harness.claimed.grant.media_key is None


def _end_with_the_wall_time_spent(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    harness.clock.advance(harness.claimed.dispatch.limits.max_wall_time_ms / 1000)


def _end_on_a_failed_preflight(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "run_preflight", _raising(WorkerFailure(STAGE_FAILURE)))


def _end_on_a_bug_in_the_download(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "download_source", a_stage_with_a_bug)


@pytest.mark.parametrize(
    "arrange_the_ending",
    [
        pytest.param(_end_with_the_wall_time_spent, id="the-wall-time-was-spent"),
        pytest.param(_end_on_a_failed_preflight, id="a-failed-preflight"),
        pytest.param(_end_on_a_bug_in_the_download, id="a-bug-in-the-download"),
    ],
)
def test_run_when_it_ends_before_the_transcoding_stage_should_have_dropped_the_media_key_when_it_returns(
    harness, monkeypatch, arrange_the_ending
):
    # Arrange
    arrange_the_ending(harness, monkeypatch)

    # Act
    result = harness.result()

    # Assert: no encode ever had the key, and the grant does not keep it past the execution
    assert result["message_kind"] == FAILED_KIND
    assert harness.stages.encodes == []
    assert harness.claimed.grant.media_key is None
    assert harness.claimed.grant.media_key_id == harness.claimed.dispatch.encryption.media_key_id


def test_run_when_it_raises_should_have_dropped_the_media_key_all_the_same(tmp_path, monkeypatch):
    # Arrange: an execution id that aliases the video's id fails every self-check, the failed result's
    # included; the unknown ref fails the job before any stage
    harness = Harness(tmp_path, monkeypatch, request_changes={"/source/credential_ref": UNKNOWN_CREDENTIAL_REF})
    dispatch_identity = harness.claimed.dispatch.identity
    harness.regrant(
        identity=ExecutionIdentity.of(dispatch_identity, execution_id=dispatch_identity.video_uuid, execution_fence=2)
    )
    held_before = harness.claimed.grant.media_key

    # Act
    with pytest.raises(SelfCheckFailed):
        harness.run()

    # Assert
    assert held_before == MEDIA_KEY
    assert harness.claimed.grant.media_key is None


def test_run_when_the_dispatch_is_encrypted_should_keep_the_media_key_out_of_every_message_object_and_log(
    harness, caplog
):
    # Arrange
    caplog.set_level(logging.DEBUG)

    # Act
    sealed = harness.run()

    # Assert
    logged = caplog.text.encode("utf-8") + repr([record.args for record in caplog.records]).encode("utf-8")
    told = [sealed.body, logged, *(call.body for call in harness.transport.calls), *harness.uploaded.values()]
    assert len(harness.transport.calls) > 0 and len(harness.uploaded) > 0 and caplog.records
    for encoding in _encodings_of(MEDIA_KEY):
        assert all(encoding not in text for text in told)


def test_run_when_the_dispatch_is_encrypted_should_never_write_the_media_key_into_its_scratch_directory(harness):
    # Act
    harness.run()

    # Assert: what the scratch directory held when the last rendition's encode started
    held = harness.stages.encodes[-1].scratch_files
    assert len(held) > 1
    for encoding in _encodings_of(MEDIA_KEY):
        assert all(encoding not in content for content in held.values())


def test_run_when_it_logs_should_name_no_storage_credential_and_no_endpoint(harness, caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)

    # Act
    harness.run()

    # Assert
    (access,) = set(harness.access_profiles)
    assert caplog.records
    for secret in (access.secret_access_key, access.access_key_id, access.endpoint):
        assert secret not in caplog.text


# --- progress -------------------------------------------------------------------------------------


def test_progress_when_the_execution_completes_should_number_its_messages_from_one_without_a_repeat(harness):
    # Act
    harness.run()

    # Assert
    sequence = [message["event_seq"] for message in harness.progress]
    assert sequence == list(range(1, len(sequence) + 1))
    assert len(sequence) >= len(STAGES)


def test_progress_when_the_execution_completes_should_send_messages_the_hub_accepts_one_after_another(harness):
    # Act
    harness.run()

    # Assert: each message is accepted by a Hub that has accepted the one before it
    calls = harness.transport.calls_to(CALLBACK_URL)
    assert len(calls) == len(harness.transport.calls) > 0
    for accepted_so_far, call in enumerate(calls):
        message = accepted_bytes_by_the_hub(
            call.body, call.signature, harness.claimed, now=harness.clock(), last_event_seq=accepted_so_far
        )
        assert message["message_kind"] == "transcode.progress"
        assert message["identity"] == harness.claimed.grant.identity.to_dict()


def test_progress_when_the_execution_completes_should_open_every_stage_at_zero_in_the_order_it_ran(harness):
    # Act
    harness.run()

    # Assert
    opened = [message["stage"] for message in harness.progress if message["stage_progress_pct"] == 0]
    assert opened == ["probing", "downloading", "transcoding", "finalizing", "uploading"]
    stages_in_order = [message["stage"] for message in harness.progress]
    assert stages_in_order == sorted(stages_in_order, key=opened.index)


def test_progress_when_the_renditions_are_encoded_should_count_them_one_by_one(harness):
    # Act
    harness.run()

    # Assert: the plan is known when the stage opens, then one message follows each rendition
    transcoding = [message for message in harness.progress if message["stage"] == "transcoding"]
    assert [
        (message["counters"]["renditions_done"], message["counters"]["renditions_total"]) for message in transcoding
    ] == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert [message["stage_progress_pct"] for message in transcoding] == [0, 33, 66, 100]


def test_progress_when_the_source_is_downloaded_in_many_chunks_should_report_at_most_once_per_ten_percent(
    harness, monkeypatch
):
    # Arrange: 80 chunks
    monkeypatch.setattr(storage_module, "DOWNLOAD_CHUNK_BYTES", len(SOURCE_OBJECT) // 80)

    # Act
    harness.run()

    # Assert
    downloading = [message for message in harness.progress if message["stage"] == "downloading"]
    percents = [message["stage_progress_pct"] for message in downloading]
    assert percents == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert downloading[-1]["counters"]["bytes_downloaded"] == len(SOURCE_OBJECT)


def test_progress_when_the_generation_is_uploaded_should_report_its_bytes_up_to_the_whole(harness):
    # Act
    harness.run()

    # Assert
    uploading = [message for message in harness.progress if message["stage"] == "uploading"]
    percents = [message["stage_progress_pct"] for message in uploading]
    assert percents == sorted(percents) and percents[0] == 0 and percents[-1] == 100
    assert len(percents) <= 1 + PERCENT_STEPS_PER_STAGE
    assert uploading[-1]["counters"]["bytes_uploaded"] == sum(len(body) for body in harness.uploaded.values())


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(NO_ANSWER, id="no-answer"),
        pytest.param(RuntimeError("the transport broke"), id="a-transport-bug"),
        pytest.param(HttpAnswer(HTTP_SERVER_ERROR, b"<html>502</html>", None), id="an-error-answer"),
    ],
)
def test_progress_when_the_hub_never_takes_a_message_should_post_three_then_stop_and_not_change_the_result(
    tmp_path, monkeypatch, caplog, outcome
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    harness = Harness(tmp_path, monkeypatch)
    harness.transport.route(CALLBACK_URL, then=outcome)

    # Act
    result = harness.result()

    # Assert: three posts for the whole job, each tried once, and one line saying that reporting stopped
    assert result["message_kind"] == COMPLETED_KIND
    assert MANIFEST_NAME in harness.uploaded
    assert [message["event_seq"] for message in harness.progress] == [1, 2, 3]
    assert harness.clock.waits == []
    assert len([record for record in caplog.records if "progress reporting stopped" in record.getMessage()]) == 1


def test_progress_when_the_hub_takes_every_report_slowly_should_stop_once_its_allowance_is_spent(
    tmp_path, monkeypatch, caplog
):
    # Arrange: every report is accepted, each only after 2.9 s
    caplog.set_level(logging.DEBUG)
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=SLOW_POST_S)

    # Act
    result = harness.result()

    # Assert: the posts the allowance holds and no more, within the allowance plus one post's time-out
    stopped = [record.getMessage() for record in caplog.records if "progress reporting stopped" in record.getMessage()]
    assert result["message_kind"] == COMPLETED_KIND
    assert [message["event_seq"] for message in harness.progress] == list(
        range(1, POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW + 1)
    )
    assert harness.clock.elapsed_s == pytest.approx(POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW * SLOW_POST_S)
    assert harness.clock.elapsed_s <= PROGRESS_ALLOWANCE_S + PROGRESS_TIMEOUT_S
    assert len(stopped) == 1 and "allowance" in stopped[0] and "in a row" not in stopped[0]


def test_progress_when_slow_reports_are_lost_as_well_should_count_their_time_too(tmp_path, monkeypatch, caplog):
    # Arrange: two lost, one delivered, over and over (never three lost in a row), each taking 2.9 s
    caplog.set_level(logging.DEBUG)
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=SLOW_POST_S)
    harness.transport.route(CALLBACK_URL, *(NO_ANSWER, NO_ANSWER, DELIVERED) * 10, then=DELIVERED)

    # Act
    result = harness.result()

    # Assert
    stopped = [record.getMessage() for record in caplog.records if "progress reporting stopped" in record.getMessage()]
    assert result["message_kind"] == COMPLETED_KIND
    assert len(harness.progress) == POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW
    assert len(stopped) == 1 and "allowance" in stopped[0]


def test_progress_when_slow_posts_end_on_an_error_of_the_transport_should_count_their_time_too(
    tmp_path, monkeypatch, caplog
):
    # Arrange: every other post ends on an error the delivery does not catch, the others are taken
    # (never three lost in a row), and each takes 2.9 s either way
    caplog.set_level(logging.DEBUG)
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=SLOW_POST_S)
    harness.transport.route(CALLBACK_URL, *(RuntimeError("the transport broke"), DELIVERED) * 10, then=DELIVERED)

    # Act
    result = harness.result()

    # Assert: the posts the allowance holds and no more
    stopped = [record.getMessage() for record in caplog.records if "progress reporting stopped" in record.getMessage()]
    assert result["message_kind"] == COMPLETED_KIND
    assert len(harness.progress) == POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW
    assert len(stopped) == 1 and "allowance" in stopped[0]


def test_progress_when_the_third_report_lost_in_a_row_also_spends_the_allowance_should_say_once_that_it_stopped(
    tmp_path, monkeypatch, caplog
):
    # Arrange: eight reports are taken and every later one is lost, each after 2.9 s: the eleventh is
    # the third lost in a row, and with it the posts have taken more than the allowance
    caplog.set_level(logging.DEBUG)
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=SLOW_POST_S)
    harness.transport.route(CALLBACK_URL, *(DELIVERED,) * DELIVERED_BEFORE_THE_HUB_IS_LOST, then=NO_ANSWER)

    # Act
    result = harness.result()

    # Assert: both reasons to stop hold at once, and one line says so
    stopped = [record.getMessage() for record in caplog.records if "progress reporting stopped" in record.getMessage()]
    assert result["message_kind"] == COMPLETED_KIND
    assert len(harness.progress) == DELIVERED_BEFORE_THE_HUB_IS_LOST + UNDELIVERED_BEFORE_GIVING_UP
    assert harness.clock.elapsed_s > PROGRESS_ALLOWANCE_S
    assert len(stopped) == 1


def test_progress_when_the_hub_is_slow_should_not_cost_the_job_its_wall_time(tmp_path, monkeypatch):
    # Arrange: the wall-time limit and the Hub's slowness on one clock; all of this job's reports, each
    # taken after 2.9 s, would outlast the 35 s the dispatch allows (the next test counts them)
    harness = Harness(
        tmp_path,
        monkeypatch,
        seconds_per_post=SLOW_POST_S,
        request_changes={"/limits/max_wall_time_ms": WALL_TIME_A_SLOW_HUB_WOULD_OUTLAST_S * 1000},
    )
    harness.count_wall_time_on_the_fake_clock(monkeypatch)

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert MANIFEST_NAME in harness.uploaded
    assert harness.clock.elapsed_s < WALL_TIME_A_SLOW_HUB_WOULD_OUTLAST_S


def test_progress_when_the_hub_answers_at_once_should_receive_every_report(harness):
    # Act
    result = harness.result()

    # Assert: up to the last report of the upload; so many that, each taken slowly, they would outlast the wall time
    posted = harness.progress
    assert result["message_kind"] == COMPLETED_KIND
    assert [message["event_seq"] for message in posted] == list(range(1, len(posted) + 1))
    assert (posted[-1]["stage"], posted[-1]["stage_progress_pct"]) == ("uploading", 100)
    assert len(posted) * SLOW_POST_S > WALL_TIME_A_SLOW_HUB_WOULD_OUTLAST_S
    assert harness.clock.elapsed_s == 0


def test_progress_when_the_hub_is_slow_but_within_the_allowance_should_receive_every_report(tmp_path, monkeypatch):
    # Arrange: one second for each report: all of them together stay below the allowance
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=1.0)

    # Act
    harness.run()

    # Assert
    posted = harness.progress
    assert harness.clock.elapsed_s < PROGRESS_ALLOWANCE_S
    assert [message["event_seq"] for message in posted] == list(range(1, len(posted) + 1))
    assert (posted[-1]["stage"], posted[-1]["stage_progress_pct"]) == ("uploading", 100)


def test_progress_when_it_posts_should_give_every_post_its_own_short_time_out(harness):
    # Act
    harness.run()

    # Assert
    assert len(harness.transport.calls) > 0
    assert {call.timeout_s for call in harness.transport.calls} == {PROGRESS_TIMEOUT_S}


def test_progress_when_it_has_given_up_should_build_no_further_message(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch)
    harness.transport.route(CALLBACK_URL, then=NO_ANSWER)

    # Act
    harness.run()

    # Assert: an id was drawn for each of the three posted messages, for the completed result that is
    # checked before the upload and for the one that is returned, for nothing else
    assert harness.ids_drawn == UNDELIVERED_BEFORE_GIVING_UP + 2


def test_progress_when_two_messages_are_lost_and_the_third_arrives_should_keep_reporting(tmp_path, monkeypatch):
    # Arrange
    harness = Harness(tmp_path, monkeypatch)
    harness.transport.route(CALLBACK_URL, NO_ANSWER, NO_ANSWER, then=DELIVERED)

    # Act
    result = harness.result()

    # Assert: every message of the job was posted, up to the last of the upload
    posted = harness.progress
    assert result["message_kind"] == COMPLETED_KIND
    assert [message["event_seq"] for message in posted] == list(range(1, len(posted) + 1))
    assert (posted[-1]["stage"], posted[-1]["stage_progress_pct"]) == ("uploading", 100)


def test_progress_when_lost_messages_never_come_three_in_a_row_should_keep_reporting(tmp_path, monkeypatch):
    # Arrange: two lost, one delivered, three times over: six lost in all, never three in a row
    harness = Harness(tmp_path, monkeypatch)
    lost_twice_then_delivered = (NO_ANSWER, NO_ANSWER, DELIVERED)
    harness.transport.route(CALLBACK_URL, *lost_twice_then_delivered * 3, then=DELIVERED)

    # Act
    result = harness.result()

    # Assert
    posted = harness.progress
    assert result["message_kind"] == COMPLETED_KIND
    assert len(posted) > len(lost_twice_then_delivered) * 3
    assert [message["event_seq"] for message in posted] == list(range(1, len(posted) + 1))
    assert (posted[-1]["stage"], posted[-1]["stage_progress_pct"]) == ("uploading", 100)


def test_progress_when_a_message_cannot_even_be_built_should_not_change_the_result(harness, monkeypatch):
    # Arrange: the progress reporter's own message builder is broken
    monkeypatch.setattr("core.job.progress.messages.progress", _raising(ValueError(CANARY)))

    # Act
    result = harness.result()

    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert harness.transport.calls == []


def test_progress_when_a_report_is_dropped_should_log_the_errors_class_and_never_its_text(
    harness, monkeypatch, caplog
):
    # Arrange: the message builder fails, in words no log line may show
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("core.job.progress.messages.progress", _raising(ValueError(CANARY)))
    as_a_handler_writes_it = logging.Formatter()

    # Act
    harness.run()

    # Assert: one line for each report that was dropped before reporting stopped, by the class alone,
    # in the message and in anything a handler would append to it
    written = [as_a_handler_writes_it.format(record) for record in caplog.records]
    dropped = [line for line in written if "a progress report was dropped" in line]
    assert dropped == ["a progress report was dropped: ValueError"] * UNDELIVERED_BEFORE_GIVING_UP
    assert all(CANARY not in line for line in written)
    assert all(record.exc_info is None and record.exc_text is None for record in caplog.records)


def test_progress_when_the_workers_own_validator_refuses_a_message_should_never_post_it(harness, caplog):
    # Arrange: an execution id that aliases the video's id, which the validator refuses in every message
    caplog.set_level(logging.DEBUG)
    dispatch_identity = harness.claimed.dispatch.identity
    harness.regrant(
        identity=ExecutionIdentity.of(dispatch_identity, execution_id=dispatch_identity.video_uuid, execution_fence=2)
    )
    reporter = ProgressReporter(harness.claimed, harness.runtime)

    # Act
    reporter.stage_started("probing")
    reporter.stage_started("downloading")

    # Assert: each was built (an id was drawn for it), refused by the self-check and dropped, not posted
    dropped = [record.getMessage() for record in caplog.records if "dropped" in record.getMessage()]
    assert harness.ids_drawn == 2
    assert harness.transport.calls == []
    assert dropped == ["a progress report was dropped: SelfCheckFailed"] * 2


@pytest.mark.parametrize(
    ("done", "total", "percent"),
    [
        pytest.param(1, 3, 33, id="a-third-rounded-down"),
        pytest.param(200, 200, 100, id="all-of-it"),
        pytest.param(300, 200, 100, id="more-than-the-total"),
        pytest.param(-5, 200, 0, id="less-than-nothing"),
        pytest.param(5, 0, 0, id="a-total-of-zero"),
        pytest.param(5, -1, 0, id="a-total-below-zero"),
    ],
)
def test_percent_when_given_any_numbers_should_be_a_whole_percent_from_zero_to_a_hundred(done, total, percent):
    # Act and Assert: 0 while the total is unknown, and never outside what a progress message may carry
    assert progress_module._percent(done, total) == percent


def test_progress_when_more_is_transferred_than_the_total_says_should_report_a_hundred_percent(harness):
    # Arrange
    reporter = ProgressReporter(harness.claimed, harness.runtime)
    reporter.stage_started("downloading")

    # Act: more bytes than the object was said to hold
    reporter.downloaded(300, 200)

    # Assert: the Hub accepts the message, which a percentage above a hundred would not let it
    opened, reported = harness.transport.calls_to(CALLBACK_URL)
    message = accepted_bytes_by_the_hub(
        reported.body, reported.signature, harness.claimed, now=harness.clock(), last_event_seq=1
    )
    assert json.loads(opened.body)["stage_progress_pct"] == 0
    assert message["stage_progress_pct"] == 100
    assert message["counters"]["bytes_downloaded"] == 300


def test_progress_when_the_total_is_unknown_should_report_no_step_and_drop_nothing(harness, caplog):
    # Arrange
    caplog.set_level(logging.DEBUG)
    reporter = ProgressReporter(harness.claimed, harness.runtime)
    reporter.stage_started("uploading")

    # Act: bytes of a total of zero
    reporter.uploaded(4096, 0)

    # Assert: only the stage's opening was posted, and no report failed on the way
    assert [json.loads(call.body)["stage_progress_pct"] for call in harness.transport.calls] == [0]
    assert [record.getMessage() for record in caplog.records if "dropped" in record.getMessage()] == []


# The reporter's own lines under a log handler that fails (see "a log line whose handler fails" above):
# the error a post ended on must not be what that handler's failure is chained to, neither on the line of
# the report it cost nor on the line that says reporting stopped, which is settled as that post ends.


def test_progress_when_a_log_handler_fails_on_the_line_of_a_dropped_report_should_keep_the_errors_text_out(
    harness, capsys, failing_log
):
    # Arrange: the first post ends on an error of the transport, in words no output may show; the Hub
    # takes every later report, so reporting never stops and the reporter writes that one line only
    harness.transport.route(CALLBACK_URL, RuntimeError(CANARY), then=DELIVERED)

    # Act
    sealed = harness.run()

    # Assert: logging did report its failure on that line, with nothing of the error's text
    written = reported_log_failures(capsys, "a progress report was dropped")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    posted = harness.progress
    assert CANARY not in written
    assert "progress reporting stopped" not in written
    assert result["message_kind"] == COMPLETED_KIND
    assert (posted[-1]["stage"], posted[-1]["stage_progress_pct"]) == ("uploading", 100)


def test_progress_when_a_log_handler_fails_on_the_line_of_three_reports_lost_in_a_row_should_keep_the_errors_text_out(
    harness, capsys, failing_log
):
    # Arrange: every post ends on an error of the transport, in words no output may show: the third
    # lost in a row stops the reporting as it ends
    harness.transport.route(CALLBACK_URL, then=RuntimeError(CANARY))

    # Act
    sealed = harness.run()

    # Assert: logging did report its failure on the line that says so, with nothing of the error's text
    written = reported_log_failures(capsys, "reports in a row were not delivered")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert CANARY not in written
    assert result["message_kind"] == COMPLETED_KIND
    assert [message["event_seq"] for message in harness.progress] == [1, 2, 3]


def test_progress_when_a_log_handler_fails_on_the_line_of_a_spent_allowance_should_keep_the_errors_text_out(
    tmp_path, monkeypatch, capsys, failing_log
):
    # Arrange: every other post ends on an error of the transport, in words no output may show, the
    # others are taken (never three lost in a row), and each takes 2.9 s either way: the eleventh, one
    # that ends on that error, spends the allowance
    harness = Harness(tmp_path, monkeypatch, seconds_per_post=SLOW_POST_S)
    harness.transport.route(CALLBACK_URL, *(RuntimeError(CANARY), DELIVERED) * 10, then=DELIVERED)

    # Act
    sealed = harness.run()

    # Assert: logging did report its failure on the line that says so, with nothing of the error's text
    written = reported_log_failures(capsys, "its posts took")
    result = accepted_by_the_hub(sealed, harness.claimed, now=harness.clock())
    assert CANARY not in written
    assert "reports in a row were not delivered" not in written
    assert result["message_kind"] == COMPLETED_KIND
    assert len(harness.progress) == POSTS_THE_ALLOWANCE_HOLDS_WHEN_SLOW


def test_progress_when_a_log_handler_fails_on_the_line_of_a_stop_at_an_interrupted_post_should_keep_its_text_out(
    harness, monkeypatch, capsys, failing_log
):
    # Arrange: two reports got no answer; the third post is interrupted, in words no output may show
    harness.transport.route(CALLBACK_URL, then=NO_ANSWER)
    reporter = ProgressReporter(harness.claimed, harness.runtime)
    reporter.stage_started("probing")
    reporter.stage_started("downloading")
    interrupt = KeyboardInterrupt(CANARY)

    # Act: the interrupted post, then one more report, on a transport that takes posts again
    with monkeypatch.context() as interrupted:
        interrupted.setattr(harness.transport, "post", _raising(interrupt))
        with pytest.raises(KeyboardInterrupt) as raised:
            reporter.stage_started("transcoding")
    ids_drawn_by_then = harness.ids_drawn
    reporter.stage_started("finalizing")

    # Assert: the very same interrupt went on, its report counted as the third lost in a row (reporting
    # stopped for good: nothing more is built, and the transport took the two posts before it and none
    # after), and the line that says so quotes nothing of it
    written = reported_log_failures(capsys, "reports in a row were not delivered")
    assert raised.value is interrupt
    assert CANARY not in written
    assert (ids_drawn_by_then, harness.ids_drawn) == (UNDELIVERED_BEFORE_GIVING_UP, UNDELIVERED_BEFORE_GIVING_UP)
    assert len(harness.transport.calls) == UNDELIVERED_BEFORE_GIVING_UP - 1
