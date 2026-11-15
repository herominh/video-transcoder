"""The upload of protocol 2.2.0 through put URLs (core/job/url_upload.py): what it stores and in which order, how many
PUTs run at once, what each PUT carries, the batches, the retries, the refusals, the stop at the first failure and at
the deadline, and the stored result. The Hub and the store are fakes; no network."""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from pathlib import Path

import pytest

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import WorkerFailure
from core.job import storage as storage_module
from core.job import url_upload
from core.job.url_upload import upload_generation_by_urls, upload_result_by_url
from tests.drive_import.fakes import FakeResponse
from core.package import ManifestOutput
from core.protocol.capabilities_client import CapabilityClient
from tests.contract.v2_support import FakeClock, uuid_source
from tests.job.capability_fakes import (
    FakeHub,
    FakeStore,
    capability_claimed,
    capability_settings,
    failed_result_of,
    hub_error,
    wait_until,
)
from tests.job.conftest import ScriptedDeadline

MANIFEST_NAME = "generation-manifest.json"
MANIFEST_BYTES = b'{"the":"manifest"}'
RESULT_NAME = "result.json"
KIND_BY_SUFFIX = {".m3u8": "hls_media_playlist", ".ts": "hls_segment", ".jpg": "thumbnail"}


class _Waits:
    """The `sleep` of an upload: it records each wait and waits for none."""

    def __init__(self) -> None:
        self.seconds: list[float] = []
        self._lock = threading.Lock()

    def __call__(self, seconds: float) -> None:
        with self._lock:
            self.seconds.append(seconds)


class _Generation:
    def __init__(self, root: Path, segments: int) -> None:
        self.output_root = root / "output"
        (self.output_root / "v360").mkdir(parents=True)
        files = {"thumbnail.jpg": b"\xff\xd8 picture \xff\xd9", "v360/playlist.m3u8": b"#EXTM3U\n"}
        files.update({f"v360/segment_{n:04d}.ts": b"\x47" + n.to_bytes(4, "big") * 50 for n in range(segments)})
        self.artifacts = []
        for path, data in files.items():
            (self.output_root / path).write_bytes(data)
            suffix = path[path.rindex("."):]
            self.artifacts.append(ArtifactFile(path=path, kind=KIND_BY_SUFFIX[suffix], size_bytes=len(data),
                                               sha256=hashlib.sha256(data).hexdigest()))
        (self.output_root / MANIFEST_NAME).write_bytes(MANIFEST_BYTES)
        self.manifest = ManifestOutput(
            path=MANIFEST_NAME, sha256=hashlib.sha256(MANIFEST_BYTES).hexdigest(), size_bytes=len(MANIFEST_BYTES),
            artifact_count=len(self.artifacts), total_bytes=sum(a.size_bytes for a in self.artifacts),
        )

    def bytes_of(self, path: str) -> bytes:
        return (self.output_root / path).read_bytes()


class _Setup:
    def __init__(self, tmp_path: Path, *, segments: int = 4) -> None:
        self.claimed = capability_claimed()
        self.clock = FakeClock()
        self.hub = FakeHub(self.claimed, self.clock)
        self.store = FakeStore(self.hub)
        self.client = CapabilityClient(
            dispatch=self.claimed.dispatch, grant=self.claimed.grant, runtime_id=self.claimed.runtime_id,
            settings=capability_settings(), transport=self.hub, clock=self.clock, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, new_message_id=uuid_source(), time_left=lambda: 3600.0,
        )
        self.client.set_stage("uploading")
        self.generation = _Generation(tmp_path, segments)
        self.waits = _Waits()
        self.totals: list[int] = []

    def upload(self, *, deadline: Deadline | None = None) -> int:
        return upload_generation_by_urls(
            self.client, self.store, prefix=self.claimed.grant.output_prefix,
            output_root=self.generation.output_root, artifacts=self.generation.artifacts,
            manifest=self.generation.manifest, deadline=deadline or Deadline.after_ms(60_000),
            sleep=self.waits, on_bytes=self.totals.append,
        )

    def put_requests(self) -> list[dict]:
        return [request for request in self.hub.requests if request["put"] is not None]


def _failure(call) -> WorkerFailure:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value


def test_upload_when_every_put_succeeds_should_store_every_artifact_then_the_manifest_last(tmp_path):
    # Arrange
    setup = _Setup(tmp_path, segments=6)
    expected = {artifact.path for artifact in setup.generation.artifacts}
    # Act
    uploaded = setup.upload()
    # Assert
    assert setup.store.stored_order[-1] == MANIFEST_NAME
    assert set(setup.store.stored_order[:-1]) == expected
    assert all(setup.store.objects[path] == setup.generation.bytes_of(path) for path in setup.store.objects)
    assert uploaded == setup.generation.manifest.total_bytes + len(MANIFEST_BYTES)
    assert setup.totals[-1] == uploaded and setup.totals == sorted(setup.totals)
    lifetimes = [request["put"]["lifetime_seconds"] for request in setup.put_requests()]
    assert lifetimes == [180, 60]
    assert [o["path"] for o in setup.put_requests()[-1]["put"]["objects"]] == [MANIFEST_NAME]


def test_upload_when_each_put_is_sent_should_carry_the_declared_content_md5_length_and_type(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    # Act
    setup.upload()
    # Assert
    declared = {o["path"]: o for request in setup.put_requests() for o in request["put"]["objects"]}
    for put in setup.store.puts:
        data = setup.generation.bytes_of(put.path)
        md5 = base64.b64encode(hashlib.md5(data).digest()).decode("ascii")
        assert put.headers["Content-MD5"] == md5 == declared[put.path]["md5"]
        assert put.headers["Content-Length"] == str(len(data)) == str(declared[put.path]["size_bytes"])
        assert put.headers["Content-Type"] == storage_module.CONTENT_TYPES[put.path[put.path.rindex("."):]]
        assert put.timeout[0] == url_upload.CONNECT_TIMEOUT_S


def test_upload_when_many_objects_go_should_run_at_most_twelve_puts_at_once(tmp_path):
    # Arrange: each PUT is held until twelve are under way (or a moment passed).
    setup = _Setup(tmp_path, segments=40)
    setup.store.hold_until_active = url_upload.UPLOAD_PARALLELISM
    # Act
    setup.upload()
    # Assert
    assert setup.store.peak == url_upload.UPLOAD_PARALLELISM


def test_upload_when_more_than_a_batch_of_objects_go_should_ask_in_batches_of_at_most_128_in_order(tmp_path):
    # Arrange
    setup = _Setup(tmp_path, segments=300)
    # Act
    setup.upload()
    # Assert
    batches = [[o["path"] for o in request["put"]["objects"]] for request in setup.put_requests()]
    assert [len(batch) for batch in batches] == [128, 128, 46, 1]
    assert [path for batch in batches[:-1] for path in batch] == [a.path for a in setup.generation.artifacts]


def test_upload_when_a_file_changed_after_the_manifest_should_refuse_it_before_its_batch_is_asked(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    (setup.generation.output_root / "v360/segment_0001.ts").write_bytes(b"\x47" + b"x" * 200)
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert failure.code == "artifact_changed_before_upload"
    assert setup.put_requests() == []
    assert setup.store.puts == []


def test_upload_when_the_store_fails_for_now_should_try_the_object_again_and_store_it(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["v360/segment_0000.ts"] = [503]
    # Act
    setup.upload()
    # Assert
    assert "v360/segment_0000.ts" in setup.store.objects
    assert 0.25 <= sum(setup.waits.seconds) <= 0.5


def test_upload_when_the_store_refuses_a_url_should_ask_for_it_again_and_store_the_object(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [403]
    # Act
    setup.upload()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects
    asked = [[o["path"] for o in request["put"]["objects"]] for request in setup.put_requests()]
    assert asked[1] == ["thumbnail.jpg"] and asked[-1] == [MANIFEST_NAME]


def test_upload_when_the_store_refuses_an_objects_url_four_times_should_end_as_upload_url_refused(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [403, 403, 403, 403]
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("output_write_failed", "upload_url_refused", True)
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_the_store_refuses_an_object_otherwise_should_end_for_good_as_upload_refused(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["v360/playlist.m3u8"] = [400]
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert (failure.code, failure.retryable) == ("upload_refused", False)
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_one_object_fails_should_start_no_further_put_and_never_store_the_manifest(tmp_path):
    # Arrange: a run of one PUT at a time, so that the order after the failing object is the manifest's.
    setup = _Setup(tmp_path, segments=10)
    setup.store.put_answers["v360/playlist.m3u8"] = [400]
    # Act
    _failure(lambda: upload_generation_by_urls(
        setup.client, setup.store, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
        artifacts=setup.generation.artifacts, manifest=setup.generation.manifest,
        deadline=Deadline.after_ms(60_000), sleep=setup.waits, parallelism=1,
    ))
    # Assert
    sent = [put.path for put in setup.store.puts]
    assert sent == ["thumbnail.jpg", "v360/playlist.m3u8"]
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_the_deadline_has_passed_should_start_no_put(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    # Act
    failure = _failure(lambda: setup.upload(deadline=ScriptedDeadline(True))).failure
    # Assert
    assert failure.code == "upload_deadline_exceeded"
    assert setup.store.puts == []
    assert setup.put_requests() == []


def test_upload_when_the_authority_is_lost_during_it_should_stop_and_end_with_that_failure(tmp_path):
    # Arrange: the lease runs out by the runtime's count while the first object is being stored.
    setup = _Setup(tmp_path, segments=10)
    setup.store.put_answers["thumbnail.jpg"] = [lambda: setup.clock.advance(3600)]
    # Act
    raised = _failure(lambda: upload_generation_by_urls(
        setup.client, setup.store, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
        artifacts=setup.generation.artifacts, manifest=setup.generation.manifest,
        deadline=Deadline.after_ms(60_000), sleep=setup.waits, parallelism=1,
    ))
    # Assert
    assert (raised.failure.error_class, raised.failure.code) == ("authority_refused", "lease_unrenewed")
    lost = setup.client.lost()
    assert lost is not None and lost.failure == raised.failure
    assert len(setup.store.puts) == 1
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_the_hub_refuses_the_batch_should_end_as_authority_refused_and_put_nothing(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.hub.script.append(hub_error(403, "stage_budget_exceeded", retryable=False))
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert (failure.error_class, failure.code) == ("authority_refused", "stage_budget_exceeded")
    assert setup.store.puts == []


def test_upload_when_an_object_is_larger_than_one_put_url_carries_should_refuse_before_any_request(tmp_path, monkeypatch):
    # Arrange
    setup = _Setup(tmp_path)
    monkeypatch.setattr(url_upload, "MAX_OBJECT_BYTES", 100)
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert failure.code == "object_too_large"
    assert setup.hub.requests == []


# --- the stored result -------------------------------------------------------------------------------------


def _result(setup: _Setup):
    return failed_result_of(setup.claimed)


def test_result_when_stored_should_go_through_a_url_of_its_own_with_a_short_lifetime(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    result = _result(setup)
    # Act
    stored = upload_result_by_url(setup.client, setup.store, result=result, deadline=Deadline.after_ms(60_000))
    # Assert
    assert stored is True
    (request,) = setup.put_requests()
    assert request["put"]["lifetime_seconds"] == 60
    assert [o["path"] for o in request["put"]["objects"]] == [RESULT_NAME]
    assert RESULT_NAME in setup.store.objects


def test_result_when_its_put_fails_should_raise_once_and_try_nothing_else(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers[RESULT_NAME] = [503]
    # Act
    failure = _failure(lambda: upload_result_by_url(
        setup.client, setup.store, result=_result(setup), deadline=Deadline.after_ms(60_000))).failure
    # Assert
    assert failure.code == "upload_failed"
    assert len(setup.store.puts) == 1


def test_result_when_the_authority_was_lost_should_ask_for_no_url_and_store_nothing(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.hub.script.append(hub_error(403, "lease_expired", retryable=False))
    _failure(lambda: setup.client.heartbeat())
    asked = len(setup.hub.requests)
    # Act
    stored = upload_result_by_url(setup.client, setup.store, result=_result(setup), deadline=Deadline.after_ms(60_000))
    # Assert
    assert stored is False
    assert len(setup.hub.requests) == asked
    assert setup.store.puts == []


# --- the round-1 review's cases ------------------------------------------------------------------------------


def _s3_error(status: int, code: str) -> FakeResponse:
    return FakeResponse(status, {"Content-Type": "application/xml"},
                        f"<?xml version='1.0'?><Error><Code>{code}</Code><Message>no</Message></Error>".encode())


def _upload_at(setup: _Setup, parallelism: int, deadline: Deadline | None = None) -> int:
    return upload_generation_by_urls(
        setup.client, setup.store, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
        artifacts=setup.generation.artifacts, manifest=setup.generation.manifest,
        deadline=deadline or Deadline.after_ms(60_000), sleep=setup.waits, parallelism=parallelism,
    )


@pytest.mark.parametrize("answer", [
    _s3_error(400, "RequestTimeout"), _s3_error(400, "IncompleteBody"), _s3_error(400, "BadDigest"),
    FakeResponse(409), FakeResponse(408), FakeResponse(429, {"Retry-After": "3"}), FakeResponse(503),
], ids=["400-RequestTimeout", "400-IncompleteBody", "400-BadDigest", "409", "408", "429", "503"])
def test_upload_when_the_store_refuses_in_a_way_that_passes_should_try_again_and_store_the_object(tmp_path, answer):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [answer]
    # Act
    setup.upload()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects
    assert [put.path for put in setup.store.puts].count("thumbnail.jpg") == 2
    if answer.status_code == 429:
        assert sum(setup.waits.seconds) == pytest.approx(3.0)


def test_upload_when_the_store_refuses_with_a_code_that_does_not_pass_should_end_for_good(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [_s3_error(400, "InvalidArgument")]
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert (failure.code, failure.retryable) == ("upload_refused", False)
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_a_put_times_out_should_try_it_again(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [TimeoutError("timed out")]
    # Act
    setup.upload()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects


def test_upload_when_its_urls_run_out_of_time_before_their_puts_start_should_ask_again_uncounted_and_complete(tmp_path):
    # Arrange: one PUT at a time, each taking 60 s of the 180 s its batch's URLs live: three start per batch.
    setup = _Setup(tmp_path, segments=20)
    for artifact in setup.generation.artifacts:
        setup.store.put_answers[artifact.path] = [lambda: setup.clock.advance(60)] * 10
    # Act
    _upload_at(setup, parallelism=1)
    # Assert
    assert set(setup.store.objects) == {a.path for a in setup.generation.artifacts} | {MANIFEST_NAME}
    assert len(setup.put_requests()) > 5
    assert all(put.status == 200 for put in setup.store.puts)


def test_upload_when_a_url_is_within_five_seconds_of_its_end_should_ask_again_without_a_put(tmp_path):
    # Arrange: the first PUT takes the clock to 4 s before the batch's URLs end.
    setup = _Setup(tmp_path, segments=1)
    setup.store.put_answers["thumbnail.jpg"] = [lambda: setup.clock.advance(176)]
    # Act
    _upload_at(setup, parallelism=1)
    # Assert: the playlist's URL of the first batch was never used; its one PUT went through a later URL.
    first_expiry = min(issued.expires_at for issued in setup.hub.issued.values())
    (playlist_put,) = [put for put in setup.store.puts if put.path == "v360/playlist.m3u8"]
    assert setup.hub.issued[playlist_put.url].expires_at > first_expiry
    assert [len(request["put"]["objects"]) for request in setup.put_requests()] == [3, 2, 1]


def test_upload_when_a_file_changes_after_its_batch_was_asked_should_refuse_it_before_its_put(tmp_path):
    # Arrange: the PUT before it changes the file's bytes, not its size, after its batch's URLs were asked.
    setup = _Setup(tmp_path, segments=1)
    changed = setup.generation.output_root / "v360/playlist.m3u8"
    setup.store.put_answers["thumbnail.jpg"] = [lambda: changed.write_bytes(b"#EXTM3X\n")]
    # Act
    failure = _failure(lambda: _upload_at(setup, parallelism=1)).failure
    # Assert
    assert failure.code == "artifact_changed_before_upload"
    assert [put.path for put in setup.store.puts] == ["thumbnail.jpg"]


def test_upload_when_one_object_fails_at_full_parallelism_should_start_no_put_after_it(tmp_path):
    # Arrange: forty objects; twelve PUTs start together, the thumbnail's is refused for good at once while the
    # others take a fifth of a second.
    setup = _Setup(tmp_path, segments=40)
    setup.store.hold_until_active = url_upload.UPLOAD_PARALLELISM
    for artifact in setup.generation.artifacts[1:]:
        setup.store.put_answers[artifact.path] = [lambda: time.sleep(0.2)]
    setup.store.put_answers["thumbnail.jpg"] = [400]
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert failure.code == "upload_refused"
    assert len(setup.store.puts_started) == url_upload.UPLOAD_PARALLELISM
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_the_deadline_passes_with_puts_in_flight_should_give_them_up_and_start_no_other(tmp_path):
    # Arrange: every PUT is held far longer than the 0.5 s the deadline leaves.
    setup = _Setup(tmp_path, segments=30)
    release = threading.Event()
    for artifact in setup.generation.artifacts:
        setup.store.put_answers[artifact.path] = [lambda: release.wait(5.0)]
    before = storage_module.abandoned_requests()
    try:
        # Act
        failure = _failure(lambda: setup.upload(deadline=Deadline.after_ms(500))).failure
        started = len(setup.store.puts_started)
    finally:
        release.set()
    # Assert
    assert failure.code == "upload_deadline_exceeded"
    assert started == url_upload.UPLOAD_PARALLELISM
    assert storage_module.abandoned_requests() - before == url_upload.UPLOAD_PARALLELISM


def test_upload_when_an_answer_outlasts_its_ceiling_should_give_it_up_and_store_the_object_again(tmp_path, monkeypatch):
    # Arrange: the first PUT of the thumbnail holds its answer past a ceiling of a fifth of a second.
    setup = _Setup(tmp_path, segments=1)
    monkeypatch.setattr(url_upload, "_request_ceiling_s", lambda size: 0.2)
    release = threading.Event()
    setup.store.put_answers["thumbnail.jpg"] = [lambda: release.wait(5.0)]
    before = storage_module.abandoned_requests()
    try:
        # Act
        _upload_at(setup, parallelism=1)
    finally:
        release.set()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects
    assert storage_module.abandoned_requests() - before == 1
    assert setup.store.puts_started.count("thumbnail.jpg") == 2


# --- the round-2 review's cases ------------------------------------------------------------------------------


def _stalling(setup: _Setup, path: str, release: threading.Event, seconds_each: float) -> None:
    """Every PUT of `path` moves the client's clock `seconds_each`, then holds its answer past the ceiling."""

    def stall() -> None:
        setup.clock.advance(seconds_each)
        release.wait(5.0)

    setup.store.put_answers[path] = [stall] * 20


def _asks_of(setup: _Setup, path: str) -> int:
    return sum(1 for request in setup.put_requests() if path in [o["path"] for o in request["put"]["objects"]])


def test_upload_when_every_manifest_put_stalls_should_end_upload_stalled_after_four_puts(tmp_path, monkeypatch):
    # Arrange: each attempt takes 31 s of the manifest URL's 60, so only two attempts fit in one URL.
    setup = _Setup(tmp_path, segments=1)
    monkeypatch.setattr(url_upload, "_request_ceiling_s", lambda size: 0.05)
    release = threading.Event()
    _stalling(setup, MANIFEST_NAME, release, 31)
    try:
        # Act
        failure = _failure(lambda: _upload_at(setup, parallelism=1)).failure
    finally:
        release.set()
    # Assert
    assert (failure.code, failure.retryable) == ("upload_stalled", True)
    assert setup.store.puts_started.count(MANIFEST_NAME) == 4
    assert _asks_of(setup, MANIFEST_NAME) == 2


def test_upload_when_an_artifacts_attempts_outlast_its_url_should_end_upload_stalled_after_four_puts(
    tmp_path, monkeypatch
):
    # Arrange: each attempt of the thumbnail takes 100 s of its batch URL's 180.
    setup = _Setup(tmp_path, segments=1)
    monkeypatch.setattr(url_upload, "_request_ceiling_s", lambda size: 0.05)
    release = threading.Event()
    _stalling(setup, "thumbnail.jpg", release, 100)
    try:
        # Act
        failure = _failure(lambda: _upload_at(setup, parallelism=1)).failure
    finally:
        release.set()
    # Assert
    assert failure.code == "upload_stalled"
    assert setup.store.puts_started.count("thumbnail.jpg") == 4
    assert _asks_of(setup, "thumbnail.jpg") == 2
    assert MANIFEST_NAME not in setup.store.objects


def test_upload_when_a_refusals_error_body_cannot_be_read_should_try_again_and_store_the_object(tmp_path):
    # Arrange: a 400 whose error body breaks off.
    setup = _Setup(tmp_path)

    def broken(size: int):
        yield b"<Error><Co"
        raise ConnectionError("https://store.test.invalid/the-whole-url?X-Amz-Signature=0")

    setup.store.put_answers["thumbnail.jpg"] = [FakeResponse(400, {}, chunks=broken)]
    # Act
    setup.upload()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects
    assert setup.store.puts_started.count("thumbnail.jpg") == 2


def test_upload_when_a_refusals_error_body_is_long_should_read_only_its_first_four_kib(tmp_path):
    # Arrange: the code that would pass lies beyond the first 4 KiB of the body.
    setup = _Setup(tmp_path)
    read: list[int] = []
    body = b"<Error>" + b" " * 8192 + b"<Code>RequestTimeout</Code></Error>"

    def chunks(size: int):
        for start in range(0, len(body), size):
            read.append(size)
            yield body[start:start + size]

    setup.store.put_answers["thumbnail.jpg"] = [FakeResponse(400, {}, chunks=chunks)]
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert failure.code == "upload_refused"
    assert sum(read) <= url_upload.ERROR_BODY_MAX_BYTES


def test_upload_when_a_pool_thread_cannot_be_started_should_fail_retryable_and_end_the_started_ones(
    tmp_path, monkeypatch
):
    # Arrange: the third thread of the pool cannot be started.
    setup = _Setup(tmp_path)
    start = threading.Thread.start
    pool_starts: list[str] = []

    def failing_third(thread: threading.Thread) -> None:
        if thread.name.startswith(url_upload.POOL_THREAD_PREFIX):
            pool_starts.append(thread.name)
            if len(pool_starts) == 3:
                raise RuntimeError("can't start new thread")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_third)
    # Act
    failure = _failure(setup.upload).failure
    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True)
    assert wait_until(lambda: not [t for t in threading.enumerate() if t.name.startswith(url_upload.POOL_THREAD_PREFIX)])
    assert setup.store.puts == []
