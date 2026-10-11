"""The two ways the job touches a bucket.

download_source() copies the whole source through the real S3RangeSource (so through its conditional,
checked range reads) against the fake S3 client of conftest. upload_generation() stores exactly the
files it is told, each under the granted prefix, the manifest last, and refuses anything that would put
an object elsewhere before it sends a single request.

The last section sends an upload through the real boto3 client that upload_client() makes: a hook on
the client answers every prepared request before it leaves the process, so no connection is opened.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import EndpointConnectionError

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import Failure, WorkerFailure
from core.job import storage
from core.job.storage import download_source, storage_client, upload_client, upload_generation
from core.package import ManifestOutput
from core.protocol.settings import StorageProfile
from core.source_reader import S3RangeSource
from tests.job.conftest import MANIFEST_NAME, FakeS3, ScriptedDeadline, client_error

BUCKET = "test-source-bucket"
KEY = "test/sources/source-0001.mp4"
ETAG = "test-etag-0001"
OTHER_ETAG = "test-etag-0002"
CHUNK_BYTES = 1000
# Three whole chunks and a shorter fourth.
OBJECT = bytes(range(256)) * 13 + b"tail"
GENEROUS_MS = 60_000
HTTP_FORBIDDEN = 403
OUTPUT_BUCKET = "test-output-bucket"
PREFIX = "test/media/22222222-2222-7222-8222-222222222222/55555555-5555-7555-8555-555555555555"
ENDPOINT_CANARY = "https://account-canary.r2.example.invalid"
SECRET_CANARY = "SECRETcanary0123456789"
UNREACHABLE = EndpointConnectionError(endpoint_url=f"{ENDPOINT_CANARY}/{OUTPUT_BUCKET}/{PREFIX}/master.m3u8")
ACCESS_DENIED = client_error("AccessDenied", HTTP_FORBIDDEN, "PutObject", f"{SECRET_CANARY} at {ENDPOINT_CANARY}")
OWNER_ONLY = 0o600
ARTIFACT_KIND_BY_SUFFIX = {
    ".m3u8": "hls_media_playlist",
    ".ts": "hls_segment",
    ".jpg": "thumbnail",
}


def _failure_of(call: Callable[[], Any]) -> Failure:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value.failure


def _deadline() -> Deadline:
    return Deadline.after_ms(GENEROUS_MS)


def _passed_deadline() -> Deadline:
    return Deadline(time.monotonic() - 1)


# --- the source -----------------------------------------------------------------------------------


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(storage, "DOWNLOAD_CHUNK_BYTES", CHUNK_BYTES)
    return CHUNK_BYTES


@pytest.fixture
def fake_s3() -> FakeS3:
    fake = FakeS3()
    fake.hold(BUCKET, KEY, OBJECT, ETAG)
    return fake


def _source(fake: FakeS3, *, size_bytes: int = len(OBJECT)) -> S3RangeSource:
    return S3RangeSource(fake, bucket=BUCKET, key=KEY, size_bytes=size_bytes, etag=ETAG)


class _StubSource:
    """A RangeSource that answers every read with what its `answer` function gives."""

    def __init__(self, size_bytes: int, answer: Callable[[int, int], bytes]) -> None:
        self.size_bytes = size_bytes
        self.reads: list[tuple[int, int]] = []
        self._answer = answer

    def read(self, start: int, length: int) -> bytes:
        self.reads.append((start, length))
        return self._answer(start, length)


def test_download_when_the_source_spans_several_chunks_should_write_exactly_its_bytes_in_order(
    tmp_path, fake_s3, small_chunks
):
    # Arrange
    destination = tmp_path / "source"

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)
    assert [call["Range"] for call in fake_s3.get_calls] == [
        "bytes=0-999", "bytes=1000-1999", "bytes=2000-2999", f"bytes=3000-{len(OBJECT) - 1}"
    ]


def test_download_when_it_reads_the_source_should_make_every_read_conditional_on_the_recorded_etag(
    tmp_path, fake_s3, small_chunks
):
    # Act
    download_source(_source(fake_s3), destination=tmp_path / "source", deadline=_deadline())

    # Assert
    assert len(fake_s3.get_calls) > 1
    assert {call["IfMatch"] for call in fake_s3.get_calls} == {f'"{ETAG}"'}


def test_download_when_it_creates_the_file_should_make_it_readable_by_its_owner_only(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"

    # Act
    download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert stat.S_IMODE(destination.stat().st_mode) & ~OWNER_ONLY == 0


def test_download_when_a_read_fails_retryably_should_read_that_chunk_again_and_finish(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: the second read never reaches the storage
    fake_s3.fail_get(2, UNREACHABLE)
    destination = tmp_path / "source"

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert destination.read_bytes() == OBJECT
    assert written == len(OBJECT)
    assert [call["Range"] for call in fake_s3.get_calls][1:3] == ["bytes=1000-1999", "bytes=1000-1999"]


def test_download_when_a_chunk_keeps_failing_retryably_should_give_up_after_two_more_reads(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: every read from the second on fails
    fake_s3.fail_get(2, UNREACHABLE, and_every_later_one=True)
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "storage_unreachable", True)
    assert [call["Range"] for call in fake_s3.get_calls] == ["bytes=0-999"] + ["bytes=1000-1999"] * 3
    assert not destination.exists()


def test_download_when_a_read_fails_for_good_should_not_read_again(tmp_path, small_chunks):
    # Arrange: the storage holds no such object, which no retry changes
    empty = FakeS3()
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(_source(empty), destination=destination, deadline=_deadline()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "source_missing", False)
    assert len(empty.get_calls) == 1
    assert not destination.exists()


def test_download_when_the_source_changes_between_two_chunks_should_fail_as_an_integrity_mismatch(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: after the first chunk the object is replaced by one of the same size and another ETag
    destination = tmp_path / "source"

    def replace_the_object(total_bytes: int) -> None:
        fake_s3.hold(BUCKET, KEY, bytes(len(OBJECT)), OTHER_ETAG)

    # Act
    failure = _failure_of(
        lambda: download_source(
            _source(fake_s3), destination=destination, deadline=_deadline(), on_bytes=replace_the_object
        )
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_integrity_mismatch", "source_changed", False
    )
    assert len(fake_s3.get_calls) == 2
    assert not destination.exists()


def test_download_when_the_source_is_shorter_than_recorded_should_fail_as_an_integrity_mismatch(
    tmp_path, fake_s3, small_chunks
):
    # Arrange: the dispatch records one chunk more than the object holds
    destination = tmp_path / "source"
    source = _source(fake_s3, size_bytes=len(OBJECT) + CHUNK_BYTES)

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=_deadline()))

    # Assert
    assert (failure.error_class, failure.retryable) == ("source_integrity_mismatch", False)
    assert failure.code == "source_size_mismatch"
    assert not destination.exists()


@pytest.mark.parametrize("answered_bytes", [CHUNK_BYTES - 1, CHUNK_BYTES + 1], ids=["one-short", "one-long"])
def test_download_when_a_read_gives_another_length_than_asked_should_fail_as_a_size_mismatch(
    tmp_path, small_chunks, answered_bytes
):
    # Arrange: a source that does not keep its promise of exactly `length` bytes
    source = _StubSource(3 * CHUNK_BYTES, lambda start, length: bytes(answered_bytes))
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=_deadline()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "source_integrity_mismatch", "source_size_mismatch", False
    )
    assert source.reads == [(0, CHUNK_BYTES)]
    assert not destination.exists()


def test_download_when_the_deadline_has_passed_should_fail_before_it_reads(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(
        lambda: download_source(_source(fake_s3), destination=destination, deadline=_passed_deadline())
    )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "download_deadline_exceeded", False
    )
    assert fake_s3.get_calls == []
    assert not destination.exists()


def test_download_when_the_deadline_passes_between_two_chunks_should_stop_reading(tmp_path, small_chunks):
    # Arrange: the deadline has not passed before the first chunk and has before the second
    deadline = ScriptedDeadline(False, True)
    source = _StubSource(3 * CHUNK_BYTES, lambda start, length: bytes(length))
    destination = tmp_path / "source"

    # Act
    failure = _failure_of(lambda: download_source(source, destination=destination, deadline=deadline))

    # Assert
    assert failure.code == "download_deadline_exceeded"
    assert source.reads == [(0, CHUNK_BYTES)]
    assert deadline.asked == 2
    assert not destination.exists()


def test_download_when_the_destination_exists_should_refuse_it_and_leave_it_alone(tmp_path, fake_s3):
    # Arrange
    destination = tmp_path / "source"
    destination.write_bytes(b"an earlier file")

    # Act
    with pytest.raises(ValueError):
        download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert destination.read_bytes() == b"an earlier file"
    assert fake_s3.get_calls == []


def test_download_when_the_destination_is_a_link_should_refuse_it_and_write_nothing_through_it(tmp_path, fake_s3):
    # Arrange: a link to a file that does not exist yet
    target = tmp_path / "elsewhere"
    destination = tmp_path / "source"
    destination.symlink_to(target)

    # Act
    with pytest.raises(ValueError):
        download_source(_source(fake_s3), destination=destination, deadline=_deadline())

    # Assert
    assert not target.exists()
    assert fake_s3.get_calls == []


@pytest.mark.parametrize(
    ("error_number", "expected"),
    [
        pytest.param(errno.ENOSPC, ("resource_exhausted", "scratch_full", True), id="disk-full"),
        pytest.param(errno.EDQUOT, ("resource_exhausted", "scratch_full", True), id="quota-exceeded"),
        pytest.param(errno.EIO, ("output_write_failed", "source_write_failed", True), id="another-write-error"),
    ],
)
def test_download_when_a_write_fails_should_fail_typed_and_remove_the_partial_file(
    tmp_path, fake_s3, small_chunks, monkeypatch, error_number, expected
):
    # Arrange: the first chunk is written, the second write fails
    destination = tmp_path / "source"
    real_write = os.write
    writes: list[int] = []

    def failing_second_write(descriptor: int, data: bytes) -> int:
        writes.append(len(data))
        if len(writes) == 1:
            return real_write(descriptor, data)
        raise OSError(error_number, f"cannot write {destination}")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(os, "write", failing_second_write)
        failure = _failure_of(
            lambda: download_source(_source(fake_s3), destination=destination, deadline=_deadline())
        )

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == expected
    assert str(tmp_path) not in failure.detail
    assert len(writes) == 2
    assert not destination.exists()


def test_download_when_progress_is_asked_for_should_report_the_running_total_after_each_chunk(
    tmp_path, fake_s3, small_chunks
):
    # Arrange
    totals: list[int] = []

    # Act
    download_source(_source(fake_s3), destination=tmp_path / "source", deadline=_deadline(), on_bytes=totals.append)

    # Assert
    assert totals == [1000, 2000, 3000, len(OBJECT)]


def test_download_when_the_progress_callback_raises_should_still_finish(tmp_path, fake_s3, small_chunks):
    # Arrange
    destination = tmp_path / "source"

    def broken_callback(total_bytes: int) -> None:
        raise RuntimeError("the progress display broke")

    # Act
    written = download_source(_source(fake_s3), destination=destination, deadline=_deadline(), on_bytes=broken_callback)

    # Assert
    assert written == len(OBJECT)
    assert destination.read_bytes() == OBJECT


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        pytest.param("destination", Path("relative/source"), ValueError, id="relative-destination"),
        pytest.param("destination", "/tmp/source-as-text", ValueError, id="destination-as-text"),
        pytest.param("deadline", 60_000, TypeError, id="deadline-as-a-number"),
        pytest.param("on_bytes", "not-callable", TypeError, id="progress-not-callable"),
    ],
)
def test_download_when_an_argument_is_invalid_should_refuse_it_before_it_reads(
    tmp_path, fake_s3, argument, value, error
):
    # Arrange
    arguments: dict[str, Any] = {"destination": tmp_path / "source", "deadline": _deadline(), "on_bytes": None}
    arguments[argument] = value

    # Act
    with pytest.raises(error):
        download_source(_source(fake_s3), **arguments)

    # Assert
    assert fake_s3.get_calls == []


# --- the generation --------------------------------------------------------------------------------


class _Generation:
    """A generation on disk: a few small files under an output root, and what a manifest records of them."""

    FILES = {
        "master.m3u8": b"#EXTM3U\n#EXT-X-VERSION:3\n",
        "thumbnail.jpg": b"\xff\xd8 a small picture \xff\xd9",
        "360p/playlist.m3u8": b"#EXTM3U\n#EXT-X-TARGETDURATION:6\n",
        "360p/segment_0000.ts": b"\x47" + bytes(187) + b"first segment",
        "360p/segment_0001.ts": b"\x47" + bytes(187) + b"second segment, a little longer",
    }
    MANIFEST = b'{"manifest_version":"1.0.0-draft","document_kind":"generation.manifest"}'

    def __init__(self, root: Path) -> None:
        self.output_root = root / "output"
        (self.output_root / "360p").mkdir(parents=True)
        for path, data in self.FILES.items():
            (self.output_root / path).write_bytes(data)
        (self.output_root / MANIFEST_NAME).write_bytes(self.MANIFEST)
        self.artifacts = [artifact_of(path, data) for path, data in self.FILES.items()]
        self.manifest = ManifestOutput(
            path=MANIFEST_NAME,
            sha256=hashlib.sha256(self.MANIFEST).hexdigest(),
            size_bytes=len(self.MANIFEST),
            artifact_count=len(self.FILES),
            total_bytes=sum(len(data) for data in self.FILES.values()),
        )

    def upload(self, client: Any, **changes: Any) -> int:
        arguments: dict[str, Any] = {
            "bucket": OUTPUT_BUCKET,
            "prefix": PREFIX,
            "output_root": self.output_root,
            "artifacts": self.artifacts,
            "manifest": self.manifest,
            "deadline": _deadline(),
        }
        arguments.update(changes)
        return upload_generation(client, **arguments)


def artifact_of(path: str, data: bytes) -> ArtifactFile:
    return ArtifactFile(
        path=path,
        kind=ARTIFACT_KIND_BY_SUFFIX[os.path.splitext(path)[1]],
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def unchecked_artifact(path: object, size_bytes: int = 1) -> ArtifactFile:
    """An ArtifactFile holding a path its own constructor refuses: what a bug elsewhere could hand the upload."""
    artifact = object.__new__(ArtifactFile)
    for name, value in (("path", path), ("kind", "hls_segment"), ("size_bytes", size_bytes), ("sha256", "0" * 64)):
        object.__setattr__(artifact, name, value)
    return artifact


@pytest.fixture
def generation(tmp_path: Path) -> _Generation:
    return _Generation(tmp_path)


def test_upload_when_the_generation_is_whole_should_put_every_file_under_the_prefix_in_the_bucket(generation):
    # Arrange
    client = FakeS3()

    # Act
    uploaded = generation.upload(client)

    # Assert
    expected = {f"{PREFIX}/{path}": data for path, data in _Generation.FILES.items()}
    expected[f"{PREFIX}/{MANIFEST_NAME}"] = _Generation.MANIFEST
    assert {call.key: call.body for call in client.stored} == expected
    assert {call.bucket for call in client.stored} == {OUTPUT_BUCKET}
    assert all(call.content_length == len(call.body) for call in client.stored)
    assert uploaded == sum(len(data) for data in expected.values())


def test_upload_when_the_generation_is_whole_should_put_the_artifacts_in_order_and_the_manifest_last(generation):
    # Arrange
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    assert [call.key for call in client.put_calls] == [
        f"{PREFIX}/master.m3u8",
        f"{PREFIX}/thumbnail.jpg",
        f"{PREFIX}/360p/playlist.m3u8",
        f"{PREFIX}/360p/segment_0000.ts",
        f"{PREFIX}/360p/segment_0001.ts",
        f"{PREFIX}/{MANIFEST_NAME}",
    ]


def test_upload_when_it_puts_a_file_should_name_the_content_type_of_its_suffix(generation):
    # Arrange
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    assert {call.key.removeprefix(f"{PREFIX}/"): call.content_type for call in client.stored} == {
        "master.m3u8": "application/vnd.apple.mpegurl",
        "thumbnail.jpg": "image/jpeg",
        "360p/playlist.m3u8": "application/vnd.apple.mpegurl",
        "360p/segment_0000.ts": "video/mp2t",
        "360p/segment_0001.ts": "video/mp2t",
        MANIFEST_NAME: "application/json",
    }


def test_upload_when_other_files_lie_in_the_output_root_should_put_only_what_it_was_told(generation):
    # Arrange: a key file, a stray segment and a leftover directory beside the generation's files
    root = generation.output_root
    (root / "enc.key").write_bytes(b"0123456789abcdef")
    (root / "360p" / "segment_0002.ts").write_bytes(b"\x47 a segment no manifest lists")
    (root / "720p").mkdir()
    (root / "720p" / "playlist.m3u8").write_bytes(b"#EXTM3U\n")
    client = FakeS3()

    # Act
    generation.upload(client)

    # Assert
    listed = {*_Generation.FILES, MANIFEST_NAME}
    assert {call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls} == listed


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("../outside.ts", id="parent-segment-first"),
        pytest.param("360p/../../outside.ts", id="parent-segment-inside"),
        pytest.param("/etc/outside.ts", id="absolute"),
        pytest.param("360p/Segment_0000.ts", id="upper-case"),
        pytest.param("a/b/c/d/segment.ts", id="five-segments"),
        pytest.param("360p/.segment.ts", id="hidden-file"),
        pytest.param("./360p/segment_0000.ts", id="current-directory-segment"),
        pytest.param("360p//segment_0000.ts", id="empty-segment"),
        pytest.param("360p/segment_0000.ts\n", id="trailing-line-break"),
        pytest.param("360p/segment 0000.ts", id="a-space"),
        pytest.param("", id="empty"),
        pytest.param(None, id="not-text"),
    ],
)
def test_upload_when_an_artifacts_path_is_no_contract_rel_path_should_refuse_it_before_any_request(
    tmp_path, generation, path
):
    # Arrange: where such a path points, a regular file exists, so only the path's form refuses it
    (tmp_path / "outside.ts").write_bytes(b"x")
    (generation.output_root / "360p" / "Segment_0000.ts").write_bytes(b"x")
    (generation.output_root / "360p" / ".segment.ts").write_bytes(b"x")
    (generation.output_root / "360p" / "segment 0000.ts").write_bytes(b"x")
    (generation.output_root / "a" / "b" / "c" / "d").mkdir(parents=True)
    (generation.output_root / "a" / "b" / "c" / "d" / "segment.ts").write_bytes(b"x")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, unchecked_artifact(path)])

    # Assert
    assert client.put_calls == []
    assert repr(path) in str(raised.value)
    assert str(generation.output_root) not in str(raised.value)
    assert PREFIX not in str(raised.value)


def test_upload_when_an_artifact_is_a_link_to_a_file_should_refuse_it_before_any_request(tmp_path, generation):
    # Arrange: the listed segment is replaced by a link to a file outside the output root
    outside = tmp_path / "outside.ts"
    outside.write_bytes(_Generation.FILES["360p/segment_0001.ts"])
    listed = generation.output_root / "360p" / "segment_0001.ts"
    listed.unlink()
    listed.symlink_to(outside)
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'360p/segment_0001.ts'" in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_upload_when_an_artifacts_directory_is_a_link_should_refuse_it_before_any_request(tmp_path, generation):
    # Arrange: the rendition directory is replaced by a link to a directory holding the same regular files
    real = generation.output_root / "360p"
    moved = tmp_path / "moved-360p"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'360p/playlist.m3u8'" in str(raised.value)


@pytest.mark.parametrize(
    "make_it",
    [
        pytest.param(lambda path: path.unlink(), id="missing"),
        pytest.param(lambda path: (path.unlink(), path.mkdir()), id="a-directory"),
    ],
)
def test_upload_when_an_artifact_is_not_a_regular_file_should_refuse_it_before_any_request(generation, make_it):
    # Arrange
    make_it(generation.output_root / "thumbnail.jpg")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client)

    # Assert
    assert client.put_calls == []
    assert "'thumbnail.jpg'" in str(raised.value)


def test_upload_when_two_artifacts_share_a_path_should_refuse_them_before_any_request(generation):
    # Arrange
    twice = [*generation.artifacts, generation.artifacts[-1]]
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=twice)

    # Assert
    assert client.put_calls == []
    assert "'360p/segment_0001.ts'" in str(raised.value)


def test_upload_when_an_artifact_has_the_manifests_path_should_refuse_it_before_any_request(generation):
    # Arrange: an artifact that would be stored where the completion marker goes
    impostor = unchecked_artifact(MANIFEST_NAME, size_bytes=len(_Generation.MANIFEST))
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, impostor])

    # Assert
    assert client.put_calls == []
    assert repr(MANIFEST_NAME) in str(raised.value)


def test_upload_when_an_artifact_has_a_suffix_without_a_content_type_should_refuse_it_before_any_request(generation):
    # Arrange: a well-formed path of a regular file, of a kind the worker never writes
    (generation.output_root / "360p" / "enc.key").write_bytes(b"0123456789abcdef")
    client = FakeS3()

    # Act
    with pytest.raises(ValueError) as raised:
        generation.upload(client, artifacts=[*generation.artifacts, unchecked_artifact("360p/enc.key", 16)])

    # Assert
    assert client.put_calls == []
    assert "'360p/enc.key'" in str(raised.value)


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param(f"/{PREFIX}", id="leading-slash"),
        pytest.param(f"{PREFIX}/", id="trailing-slash"),
        pytest.param("test/media/../55555555-5555-7555-8555-555555555555", id="parent-segment"),
        pytest.param("../55555555-5555-7555-8555-555555555555", id="parent-segment-first"),
        pytest.param("test/./55555555-5555-7555-8555-555555555555", id="current-directory-segment"),
        pytest.param("test//55555555-5555-7555-8555-555555555555", id="empty-segment"),
        pytest.param("", id="empty"),
        pytest.param(None, id="not-text"),
    ],
)
def test_upload_when_the_prefix_is_no_contract_object_key_should_refuse_it_before_any_request(generation, prefix):
    # Arrange
    client = FakeS3()

    # Act
    with pytest.raises(ValueError):
        generation.upload(client, prefix=prefix)

    # Assert
    assert client.put_calls == []


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        pytest.param("bucket", "", ValueError, id="empty-bucket"),
        pytest.param("output_root", Path("relative/output"), ValueError, id="relative-output-root"),
        pytest.param("output_root", Path("/nonexistent/output/root"), ValueError, id="missing-output-root"),
        pytest.param("artifacts", "master.m3u8", TypeError, id="artifacts-as-text"),
        pytest.param("artifacts", [{"path": "master.m3u8"}], TypeError, id="artifacts-as-mappings"),
        pytest.param("manifest", MANIFEST_NAME, TypeError, id="manifest-as-text"),
        pytest.param("deadline", 60_000, TypeError, id="deadline-as-a-number"),
    ],
)
def test_upload_when_an_argument_is_invalid_should_refuse_it_before_any_request(generation, argument, value, error):
    # Arrange
    client = FakeS3()

    # Act
    with pytest.raises(error):
        generation.upload(client, **{argument: value})

    # Assert
    assert client.put_calls == []


def test_upload_when_the_client_is_no_s3_client_should_raise_type_error(generation):
    # Act and Assert
    with pytest.raises(TypeError):
        generation.upload(object())


def test_upload_when_an_artifact_changed_size_on_disk_should_fail_without_putting_it_or_the_manifest(generation):
    # Arrange: the second segment grew after the manifest recorded it
    with (generation.output_root / "360p" / "segment_0001.ts").open("ab") as segment:
        segment.write(b"more")
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "internal_error", "artifact_changed_before_upload", True
    )
    uploaded = [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls]
    assert uploaded == ["master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/segment_0000.ts"]


def test_upload_when_the_manifest_changed_size_on_disk_should_fail_without_putting_it(generation):
    # Arrange
    (generation.output_root / MANIFEST_NAME).write_bytes(_Generation.MANIFEST[:-1])
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client))

    # Assert
    assert (failure.error_class, failure.code) == ("internal_error", "artifact_changed_before_upload")
    assert f"{PREFIX}/{MANIFEST_NAME}" not in [call.key for call in client.put_calls]
    assert len(client.stored) == len(_Generation.FILES)


def test_upload_when_a_put_fails_once_should_send_the_whole_file_again_and_finish(generation):
    # Arrange: the fourth request breaks after half of the segment was read
    client = FakeS3()
    client.fail_put(4, UNREACHABLE)
    segment = _Generation.FILES["360p/segment_0000.ts"]

    # Act
    uploaded = generation.upload(client)

    # Assert
    broken, again = client.put_calls[3], client.put_calls[4]
    assert (broken.key, broken.failed) == (f"{PREFIX}/360p/segment_0000.ts", True)
    assert 0 < len(broken.body) < len(segment)
    assert (again.key, again.failed, again.body) == (f"{PREFIX}/360p/segment_0000.ts", False, segment)
    assert [call.key for call in client.stored][-1] == f"{PREFIX}/{MANIFEST_NAME}"
    assert len(client.stored) == len(_Generation.FILES) + 1
    assert uploaded == sum(len(call.body) for call in client.stored)


@pytest.mark.parametrize(
    ("error", "named"),
    [
        pytest.param(UNREACHABLE, "EndpointConnectionError", id="no-answer"),
        pytest.param(ACCESS_DENIED, "ClientError AccessDenied", id="an-error-answer"),
    ],
)
def test_upload_when_a_put_keeps_failing_should_give_up_after_two_more_requests_and_never_put_the_manifest(
    generation, caplog, error, named
):
    # Arrange: every request from the third on fails
    caplog.set_level(logging.DEBUG)
    client = FakeS3()
    client.fail_put(3, error, and_every_later_one=True)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        generation.upload(client)

    # Assert
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_failed", True)
    assert [call.key.removeprefix(f"{PREFIX}/") for call in client.put_calls] == [
        "master.m3u8", "thumbnail.jpg", "360p/playlist.m3u8", "360p/playlist.m3u8", "360p/playlist.m3u8"
    ]
    assert named in failure.detail
    for told in (failure.detail, str(raised.value), caplog.text):
        assert "example.invalid" not in told
        assert SECRET_CANARY not in told
        assert "://" not in told
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


def test_upload_when_the_deadline_has_passed_should_fail_before_any_request(generation):
    # Arrange
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=_passed_deadline()))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "deadline_exceeded", "upload_deadline_exceeded", False
    )
    assert client.put_calls == []


def test_upload_when_the_deadline_passes_between_two_objects_should_stop_before_the_manifest(generation):
    # Arrange: the deadline has not passed before the first object and has before the second
    deadline = ScriptedDeadline(False, True)
    client = FakeS3()

    # Act
    failure = _failure_of(lambda: generation.upload(client, deadline=deadline))

    # Assert
    assert failure.code == "upload_deadline_exceeded"
    assert [call.key for call in client.put_calls] == [f"{PREFIX}/master.m3u8"]
    assert deadline.asked == 2


def test_upload_when_progress_is_asked_for_should_report_the_running_total_after_each_object(generation):
    # Arrange
    client = FakeS3()
    totals: list[int] = []

    # Act
    uploaded = generation.upload(client, on_bytes=totals.append)

    # Assert
    sizes = [len(call.body) for call in client.stored]
    assert totals == [sum(sizes[: count + 1]) for count in range(len(sizes))]
    assert totals[-1] == uploaded


def test_upload_when_the_progress_callback_raises_should_still_finish(generation):
    # Arrange
    client = FakeS3()

    def broken_callback(total_bytes: int) -> None:
        raise RuntimeError("the progress display broke")

    # Act
    generation.upload(client, on_bytes=broken_callback)

    # Assert
    assert len(client.stored) == len(_Generation.FILES) + 1


# --- the clients ----------------------------------------------------------------------------------

ACCESS_PROFILE = StorageProfile(
    credential_ref="default",
    endpoint="https://storage.test.invalid",
    region="auto",
    access_key_id="test-access-key-id",
    secret_access_key="test-secret-access-key",
)
HTTP_OK = 200
STREAMING_PAYLOAD_PREFIX = "STREAMING-"
CHUNKED_CONTENT_ENCODING = "aws-chunked"
CHECKSUM_WHEN_REQUIRED = "when_required"


@dataclass(frozen=True)
class _PreparedRequest:
    """One request as the real client was about to send it; header names in lower case."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes


class _EmptyRawResponse:
    """The body of an answer that has none, as botocore reads one."""

    def stream(self, *arguments: Any, **named_arguments: Any) -> Any:
        return iter(())


def _text(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def _answered_before_any_network(client: Any) -> list[_PreparedRequest]:
    """Answer every PutObject of the real client with a 200 before it leaves the process, and return the
    list each prepared request is recorded in."""
    prepared: list[_PreparedRequest] = []

    def answer(request: Any, **named_arguments: Any) -> AWSResponse:
        body = request.body.read() if hasattr(request.body, "read") else request.body
        headers = {name.lower(): _text(value) for name, value in request.headers.items()}
        prepared.append(_PreparedRequest(request.method, request.url, headers, body))
        return AWSResponse(request.url, HTTP_OK, {}, _EmptyRawResponse())

    client.meta.events.register("before-send.s3.PutObject", answer)
    return prepared


def test_upload_when_sent_through_the_upload_client_should_address_each_object_by_bucket_prefix_and_path(generation):
    # Arrange
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    files = {**_Generation.FILES, MANIFEST_NAME: _Generation.MANIFEST}

    # Act
    uploaded = generation.upload(client)

    # Assert: path-style addresses under the granted prefix, in order, each carrying its file
    assert [(request.method, request.url) for request in prepared] == [
        ("PUT", f"https://storage.test.invalid/{OUTPUT_BUCKET}/{PREFIX}/{path}") for path in files
    ]
    assert [request.body for request in prepared] == list(files.values())
    assert prepared[-1].headers["content-type"] == "application/json"
    assert uploaded == sum(len(data) for data in files.values())


def test_upload_when_sent_through_the_upload_client_should_be_a_plain_put_with_its_content_length(generation):
    # Arrange
    client = upload_client(ACCESS_PROFILE)
    prepared = _answered_before_any_network(client)
    sizes = [len(data) for data in (*_Generation.FILES.values(), _Generation.MANIFEST)]

    # Act
    generation.upload(client)

    # Assert: no `aws-chunked` body with a checksum trailer, which not every S3-compatible store accepts
    assert [request.headers["content-length"] for request in prepared] == [str(size) for size in sizes]
    for request in prepared:
        assert CHUNKED_CONTENT_ENCODING not in request.headers.get("content-encoding", "")
        assert "x-amz-trailer" not in request.headers
        assert "transfer-encoding" not in request.headers
        assert not request.headers.get("x-amz-content-sha256", "").startswith(STREAMING_PAYLOAD_PREFIX)


def test_upload_client_when_given_an_access_profile_should_reach_its_endpoint_as_the_source_reader_does():
    # Act: making a client opens no connection
    writer, reader = upload_client(ACCESS_PROFILE), storage_client(ACCESS_PROFILE)

    # Assert: the reader's signing, addressing, time-outs and single attempt; a checksum only where required
    assert writer.meta.endpoint_url == "https://storage.test.invalid"
    assert writer.meta.region_name == "auto"
    assert writer.meta.service_model.service_name == "s3"
    for setting in ("signature_version", "s3", "connect_timeout", "read_timeout", "retries"):
        assert getattr(writer.meta.config, setting) == getattr(reader.meta.config, setting), setting
    assert writer.meta.config.request_checksum_calculation == CHECKSUM_WHEN_REQUIRED
    assert writer.meta.config.response_checksum_validation == CHECKSUM_WHEN_REQUIRED


def test_storage_client_when_given_an_access_profile_should_make_a_client_of_its_endpoint_and_region():
    # Act: making a client opens no connection
    client = storage_client(ACCESS_PROFILE)

    # Assert
    assert client.meta.endpoint_url == "https://storage.test.invalid"
    assert client.meta.region_name == "auto"
    assert client.meta.service_model.service_name == "s3"


@pytest.mark.parametrize("make_client", [storage_client, upload_client])
def test_client_when_given_something_else_than_a_profile_should_raise_type_error(make_client):
    # Act and Assert
    with pytest.raises(TypeError):
        make_client({"endpoint": "https://storage.test.invalid"})
