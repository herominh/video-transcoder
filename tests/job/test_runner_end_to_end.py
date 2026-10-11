"""JobRunner end to end with the real ffprobe and ffmpeg (skipped without them; POSIX only).

A small generated MP4 is the source object in conftest's fake S3 client. The dispatch is the fixtures'
(P01 encrypted, P02 unencrypted) adapted to that object, the grant the fixtures' (P24, P27). The run is
the real one from the preflight's loopback probe to the last PutObject: nothing of the job is replaced.
(One test, of a read that never returns, states the ffmpeg version instead of asking the installed
ffmpeg for it, so that the little wall time it leaves the job is the preflight's alone.)

The pilot profile's smallest rung is as large as the 320x240 source, so the worker is given a test
profile instead: the pilot at x264's fastest preset with two small rungs. It is registered by replacing
core.profile.SUPPORTED_PROFILES for the length of a run, and the dispatch names that profile's own digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from core import profile as profile_module
from core.job.runner import JobRunner
from core.profile import LadderRung, profile_digest
from core.protocol import pipeline
from core.protocol.handler import ClaimedExecution
from core.protocol.outbox import SealedMessage
from tests.contract.v2_support import CALLBACK_URL, MEDIA_KEY_HEX, FakeClock, FakeTransport
from tests.encode.conftest import fast_profile
from tests.job.conftest import (
    COMPLETED_KIND,
    DELIVERED,
    FAILED_KIND,
    HANG_RELEASE_S,
    IMAGE_DIGEST,
    MANIFEST_NAME,
    FakeS3,
    accepted_bytes_by_the_hub,
    accepted_by_the_hub,
    claimed_execution,
    hub_storage_context,
    job_settings,
    runtime_of,
)
from tests.preflight.conftest import FFMPEG_TIMEOUT_S

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the stages run under POSIX process groups")

# Two rungs the 320x240 source fits: contract names, with smaller short edges than the names say.
TEST_LADDER = (
    LadderRung("360p", 240, 600_000, 640_000, 900_000, 64_000),
    LadderRung("240p", 120, 300_000, 320_000, 450_000, 48_000),
)
PROFILE = fast_profile(ladder=TEST_LADDER)
REQUESTED = ["360p", "240p"]
SOURCE_BUCKET = "test-source-bucket"
OUTPUT_BUCKET = "test-output-bucket"
SOURCE_KEY = "test/sources/source-0001.mp4"
SOURCE_ETAG = "test-etag-0001"
OTHER_ETAG = "test-etag-0002"
MEDIA_KEY = bytes.fromhex(MEDIA_KEY_HEX)
MEDIA_KEY_ID = "77777777-7777-7777-8777-777777777777"
TS_SYNC_BYTE = 0x47
AES_BLOCK_BYTES = 16
JPEG_START = b"\xff\xd8"
# What is left of a start's wall time where a test has the job wait it out, and the version a test
# states for the ffmpeg where the installed one's must not take any of that time.
WALL_TIME_LEFT_S = 0.5
STATED_FFMPEG_VERSION = "7.1-stated"


@dataclass(frozen=True)
class _Run:
    """One finished run: what the runner was handed, what it returned, and everything it told anyone."""

    claimed: ClaimedExecution
    sealed: SealedMessage
    s3: FakeS3
    clock: FakeClock
    transport: FakeTransport
    scratch_root: Path
    logged: str

    @property
    def result(self) -> dict[str, Any]:
        """The result, once the Hub's validator accepted its signed bytes."""
        return accepted_by_the_hub(self.sealed, self.claimed, now=self.clock())

    @property
    def uploaded(self) -> dict[str, bytes]:
        """What the storage holds, by path below the granted prefix."""
        prefix = self.claimed.grant.output_prefix + "/"
        return {call.key.removeprefix(prefix): call.body for call in self.s3.stored}

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.uploaded[MANIFEST_NAME])

    def segments(self) -> dict[str, bytes]:
        return {path: body for path, body in self.uploaded.items() if path.endswith(".ts")}


@contextmanager
def _everything_logged() -> Iterator[list[str]]:
    """Every log line of every logger while the block runs, formatted with its arguments."""
    lines: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    collector, root = Collector(level=logging.DEBUG), logging.getLogger()
    earlier_level = root.level
    root.addHandler(collector)
    root.setLevel(logging.DEBUG)
    try:
        yield lines
    finally:
        root.setLevel(earlier_level)
        root.removeHandler(collector)


def _claimed(media: Path, *, encrypted: bool) -> ClaimedExecution:
    """The fixtures' dispatch adapted to the source object and the test profile, claimed with the fixtures' grant."""
    return claimed_execution(
        encrypted=encrypted,
        request_changes={
            "/source/size_bytes": media.stat().st_size,
            "/source/etag": SOURCE_ETAG,
            "/renditions": REQUESTED,
            "/profile": {"id": PROFILE.profile_id, "version": PROFILE.version, "sha256": profile_digest(PROFILE)},
        },
    )


def _run_job(root: Path, media: Path, *, encrypted: bool, stored_etag: str = SOURCE_ETAG) -> _Run:
    scratch_root = root / "scratch"
    scratch_root.mkdir()
    s3 = FakeS3()
    s3.hold(SOURCE_BUCKET, SOURCE_KEY, media, stored_etag)
    claimed = _claimed(media, encrypted=encrypted)
    clock = FakeClock()
    transport = FakeTransport(then=DELIVERED, clock=clock)
    runner = JobRunner(
        job_settings(scratch_root), source_client_factory=lambda access: s3, output_client_factory=lambda access: s3
    )
    with pytest.MonkeyPatch.context() as patched, _everything_logged() as lines:
        patched.setattr(profile_module, "SUPPORTED_PROFILES", (PROFILE,))
        sealed = runner.run(claimed, runtime_of(transport, clock))
    return _Run(
        claimed=claimed, sealed=sealed, s3=s3, clock=clock, transport=transport, scratch_root=scratch_root,
        logged="\n".join(lines),
    )


@pytest.fixture(scope="module")
def encrypted_run(source_media: Path, tmp_path_factory: pytest.TempPathFactory) -> _Run:
    return _run_job(tmp_path_factory.mktemp("job-encrypted"), source_media, encrypted=True)


@pytest.fixture(scope="module")
def plain_run(source_media: Path, tmp_path_factory: pytest.TempPathFactory) -> _Run:
    return _run_job(tmp_path_factory.mktemp("job-plain"), source_media, encrypted=False)


@pytest.fixture(params=["encrypted_run", "plain_run"], ids=["encrypted", "plain"])
def run(request: pytest.FixtureRequest) -> _Run:
    return request.getfixturevalue(request.param)


def test_job_when_run_end_to_end_should_return_a_completed_result_the_hub_accepts(run: _Run) -> None:
    # Act
    result = run.result

    # Assert
    dispatch, grant = run.claimed.dispatch, run.claimed.grant
    assert result["message_kind"] == COMPLETED_KIND
    assert result["identity"] == grant.identity.to_dict()
    assert result["generation_id"] == grant.generation_id
    assert result["profile"] == dispatch.profile.to_dict()
    assert result["encoder"]["image_digest"] == IMAGE_DIGEST
    assert result["encoder"]["preset"] == "ultrafast"
    assert "diagnostics" not in result


def test_job_when_run_end_to_end_should_put_every_object_under_the_granted_prefix_and_the_manifest_last(
    run: _Run,
) -> None:
    # Arrange
    prefix = run.claimed.grant.output_prefix

    # Act
    keys = [call.key for call in run.s3.put_calls]

    # Assert
    assert len(keys) > 1
    assert all(key.startswith(f"{prefix}/") for key in keys)
    assert {call.bucket for call in run.s3.put_calls} == {OUTPUT_BUCKET}
    assert keys[-1] == f"{prefix}/{MANIFEST_NAME}"
    assert all(not call.failed for call in run.s3.put_calls)


def test_job_when_run_end_to_end_should_store_a_manifest_the_hub_accepts_and_the_result_locates(run: _Run) -> None:
    # Arrange
    stored_manifest = run.uploaded[MANIFEST_NAME]

    # Act
    verdict = pipeline.validate(stored_manifest, hub_storage_context(run.claimed))

    # Assert
    assert verdict.accepted, verdict
    located = run.result["manifest"]
    assert located["path"] == MANIFEST_NAME
    assert located["sha256"] == hashlib.sha256(stored_manifest).hexdigest()
    assert located["size_bytes"] == len(stored_manifest)
    assert located["location_id"] == run.claimed.dispatch.output.location_id
    assert located["artifact_count"] == len(run.manifest["artifacts"])


def test_job_when_run_end_to_end_should_store_exactly_the_manifests_artifacts_and_the_manifest(run: _Run) -> None:
    # Arrange
    listed = {artifact["path"]: artifact for artifact in run.manifest["artifacts"]}

    # Act
    uploaded = run.uploaded

    # Assert
    assert set(uploaded) == set(listed) | {MANIFEST_NAME}
    assert len(run.s3.put_calls) == len(listed) + 1
    for path, artifact in listed.items():
        assert artifact["size_bytes"] == len(uploaded[path])
        assert artifact["sha256"] == hashlib.sha256(uploaded[path]).hexdigest()
    assert [rendition["name"] for rendition in run.manifest["renditions"]] == REQUESTED
    assert uploaded["thumbnail.jpg"].startswith(JPEG_START)


def test_job_when_run_end_to_end_should_state_the_media_and_the_read_as_they_were(run: _Run) -> None:
    # Arrange: what the storage served, read by read: the preflight read this small object whole, then the download
    served = 0
    for call in run.s3.get_calls:
        first, last = call["Range"].removeprefix("bytes=").split("-")
        served += int(last) - int(first) + 1
    source_bytes = run.claimed.dispatch.source.size_bytes

    # Act
    result = run.result

    # Assert
    assert result["media"] == run.manifest["media"]
    assert result["media"]["max_width"] == 320 and result["media"]["max_height"] == 240
    assert result["media"]["rendition_count"] == len(REQUESTED) and result["media"]["has_audio"] is True
    assert served > source_bytes
    assert result["source"]["bytes_read"] == source_bytes
    assert result["source"]["etag_observed"] == SOURCE_ETAG
    assert all(call["IfMatch"] == f'"{SOURCE_ETAG}"' for call in run.s3.get_calls)


def test_job_when_run_end_to_end_should_name_the_ffmpeg_that_encoded(run: _Run) -> None:
    # Arrange
    first_line = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S
    ).stdout.decode("ascii", "replace").splitlines()[0]

    # Act
    version = run.result["encoder"]["ffmpeg_version"]

    # Assert
    assert version != ""
    assert first_line.startswith(f"ffmpeg version {version}")


def test_job_when_run_end_to_end_should_send_progress_the_hub_accepts_and_leave_no_scratch(run: _Run) -> None:
    # Arrange
    calls = run.transport.calls_to(CALLBACK_URL)

    # Act
    stages = []
    for accepted_so_far, call in enumerate(calls):
        message = accepted_bytes_by_the_hub(
            call.body, call.signature, run.claimed, now=run.clock(), last_event_seq=accepted_so_far
        )
        stages.append(message["stage"])

    # Assert
    assert list(dict.fromkeys(stages)) == ["probing", "downloading", "transcoding", "finalizing", "uploading"]
    assert list(run.scratch_root.iterdir()) == []


def test_job_when_the_dispatch_is_encrypted_should_store_segments_only_the_granted_key_opens(
    encrypted_run: _Run,
) -> None:
    # Arrange
    segments = encrypted_run.segments()
    first_segment = encrypted_run.uploaded["360p/segment_0000.ts"]

    # Act: AES-128-CBC with the segment's media sequence number (0) as its IV
    decryptor = Cipher(algorithms.AES(MEDIA_KEY), modes.CBC(bytes(AES_BLOCK_BYTES))).decryptor()
    opened = decryptor.update(first_segment[:AES_BLOCK_BYTES])

    # Assert
    assert len(segments) >= len(REQUESTED)
    assert any(segment[0] != TS_SYNC_BYTE for segment in segments.values())
    assert all(len(segment) % AES_BLOCK_BYTES == 0 for segment in segments.values())
    assert opened[0] == TS_SYNC_BYTE
    assert encrypted_run.manifest["encryption"] == {"mode": "aes-128", "media_key_id": MEDIA_KEY_ID}


def test_job_when_the_dispatch_is_encrypted_should_keep_the_key_out_of_every_object_message_and_log(
    encrypted_run: _Run,
) -> None:
    # Arrange
    key_forms = [MEDIA_KEY, MEDIA_KEY_HEX.encode("ascii"), MEDIA_KEY_HEX.upper().encode("ascii")]
    messages = [encrypted_run.sealed.body, *(call.body for call in encrypted_run.transport.calls)]

    # Act
    told = [*encrypted_run.uploaded.values(), *messages, encrypted_run.logged.encode("utf-8")]

    # Assert
    assert encrypted_run.logged != ""
    for key_form in key_forms:
        assert all(key_form not in text for text in told)
    assert not any(path.endswith((".key", ".info", ".keyinfo")) for path in encrypted_run.uploaded)


def test_job_when_the_dispatch_is_unencrypted_should_store_plain_segments_that_play(
    plain_run: _Run, tmp_path: Path
) -> None:
    # Arrange: the stored objects, laid out as a player would fetch them
    for path, body in plain_run.uploaded.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(body)

    # Act
    played = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
            "-protocol_whitelist", "file", "-f", "hls", "-i", str(tmp_path / "master.m3u8"),
            "-map", "0:v:0", "-f", "null", "-",
        ],
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    )

    # Assert
    assert played.returncode == 0, played.stderr.decode("utf-8", "replace")
    assert all(segment[0] == TS_SYNC_BYTE for segment in plain_run.segments().values())
    assert plain_run.manifest["encryption"] == {"mode": "none"}
    assert b"#EXT-X-KEY" not in plain_run.uploaded["360p/playlist.m3u8"]


def test_job_when_the_stored_source_has_another_etag_than_the_dispatch_should_fail_and_upload_nothing(
    source_media: Path, tmp_path: Path
) -> None:
    # Arrange: the object was replaced after the Hub recorded its ETag
    # Act
    run = _run_job(tmp_path, source_media, encrypted=True, stored_etag=OTHER_ETAG)

    # Assert
    error = run.result["error"]
    assert run.result["message_kind"] == FAILED_KIND
    assert (error["class"], error["code"], error["retryable"], error["stage"]) == (
        "source_integrity_mismatch", "source_changed", False, "probing"
    )
    assert run.s3.put_calls == []
    assert "source" not in run.result
    assert list(run.scratch_root.iterdir()) == []


def test_job_when_a_read_of_the_real_preflight_never_returns_should_fail_for_good_at_probing_once_the_wall_time_ends(
    source_media: Path, tmp_path: Path
) -> None:
    # Arrange: a storage that takes every read and never answers it, and a start with half a second of
    # its wall time left. The real preflight reads on the threads of its loopback server while the
    # job's own thread waits for ffprobe: the read and ffprobe are given up at the same deadline, and
    # which of the two is noticed first names the failure. The ffmpeg version is a stated one, so
    # that asking the installed ffmpeg takes none of the half second.
    never_answered = threading.Event()
    threads_before = set(threading.enumerate())
    s3 = FakeS3()
    s3.hold(SOURCE_BUCKET, SOURCE_KEY, source_media, SOURCE_ETAG)
    answer = s3.get_object

    def get_object_that_hangs(**request: Any) -> Any:
        never_answered.wait(HANG_RELEASE_S)
        return answer(**request)

    s3.get_object = get_object_that_hangs
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    clock = FakeClock()
    claimed = replace(_claimed(source_media, encrypted=True), wall_time_ends_at=clock.monotonic() + WALL_TIME_LEFT_S)
    runner = JobRunner(
        job_settings(scratch_root),
        source_client_factory=lambda access: s3,
        output_client_factory=lambda access: s3,
        ffmpeg_version=lambda deadline: STATED_FFMPEG_VERSION,
    )
    started = time.monotonic()

    # Act
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(profile_module, "SUPPORTED_PROFILES", (PROFILE,))
            sealed = runner.run(claimed, runtime_of(FakeTransport(then=DELIVERED, clock=clock), clock))
        took_s = time.monotonic() - started
    finally:
        never_answered.set()
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(HANG_RELEASE_S)

    # Assert: given up at the deadline, not waited out; final, as a stage that runs out of wall time is
    error = accepted_by_the_hub(sealed, claimed, now=clock())["error"]
    assert (error["class"], error["retryable"], error["stage"]) == ("deadline_exceeded", False, "probing")
    assert error["code"] in {"source_read_deadline_exceeded", "probe_deadline_exceeded"}
    assert took_s < HANG_RELEASE_S
    assert s3.put_calls == []
    assert list(scratch_root.iterdir()) == []
