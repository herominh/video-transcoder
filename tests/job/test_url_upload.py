"""The upload of protocol 2.2.0 through put URLs (core/job/url_upload.py): what it stores and in which order, how many
PUTs run at once, what each PUT carries, the batches, the retries, the refusals, the stop at the first failure and at
the deadline, and the stored result. The Hub and the store are fakes; no network but for the last section's store on
the loopback address, which each lane's real session (core/net/transport.py RequestsSession) PUTs to, to show that a
PUT given up is closed with its lane's session (the Hub's register row TD-196)."""

from __future__ import annotations

import base64
import gc
import hashlib
import logging
import select
import socketserver
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import pytest

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import WorkerFailure
from core.job import storage as storage_module
from core.job import url_upload
from core.job.url_upload import upload_generation_by_urls, upload_result_by_url
from core.net.clock import StopLine
from core.net.transport import HttpResponse, RequestsSession
from tests.drive_import.fakes import FakeResponse
from core.package import ManifestOutput
from core.protocol.capabilities_client import CapabilityClient
from tests.contract.v2_support import FakeClock, uuid_source
from tests.job.capability_fakes import (
    OUTPUT_BUCKET,
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
        # Every lane's session is the fake store.
        self.lanes = lambda: self.store
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
            self.client, self.lanes, prefix=self.claimed.grant.output_prefix,
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
        setup.client, setup.lanes, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
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
        setup.client, setup.lanes, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
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
    stored = upload_result_by_url(setup.client, setup.lanes, result=result, deadline=Deadline.after_ms(60_000))
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
        setup.client, setup.lanes, result=_result(setup), deadline=Deadline.after_ms(60_000))).failure
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
    stored = upload_result_by_url(setup.client, setup.lanes, result=_result(setup), deadline=Deadline.after_ms(60_000))
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
        setup.client, setup.lanes, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
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


def _stalling(
    setup: _Setup, monkeypatch: pytest.MonkeyPatch, path: str, release: threading.Event, seconds_each: float
) -> None:
    """Every PUT of `path` moves the client's clock `seconds_each`, then holds its answer past the ceiling (0.05 s).
    The pool's slot waits for that move before it judges the stall, so the attempt has taken its time on the clock
    whenever the store's thread came to it."""
    advanced = threading.Semaphore(0)

    def stall() -> None:
        setup.clock.advance(seconds_each)
        advanced.release()
        release.wait(5.0)

    setup.store.put_answers[path] = [stall] * 20
    monkeypatch.setattr(url_upload, "_request_ceiling_s", lambda size: 0.05)
    within = url_upload._within

    def within_after_the_move(deadline: Deadline, call, **named):
        try:
            return within(deadline, call, **named)
        finally:
            if call.args[0].upload.relative_path == path:
                assert advanced.acquire(timeout=5.0), "the stalling PUT never reached the store"

    monkeypatch.setattr(url_upload, "_within", within_after_the_move)


def _asks_of(setup: _Setup, path: str) -> int:
    return sum(1 for request in setup.put_requests() if path in [o["path"] for o in request["put"]["objects"]])


def test_upload_when_every_manifest_put_stalls_should_end_upload_stalled_after_four_puts(tmp_path, monkeypatch):
    # Arrange: each attempt takes 31 s of the manifest URL's 60, so only two attempts fit in one URL.
    setup = _Setup(tmp_path, segments=1)
    release = threading.Event()
    _stalling(setup, monkeypatch, MANIFEST_NAME, release, 31)
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
    release = threading.Event()
    _stalling(setup, monkeypatch, "thumbnail.jpg", release, 100)
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
    raised = _failure(setup.upload)
    # Assert: the interpreter's error is neither the failure's cause nor its context.
    failure = raised.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True)
    assert raised.__cause__ is None and raised.__context__ is None
    assert wait_until(lambda: not [t for t in threading.enumerate() if t.name.startswith(url_upload.POOL_THREAD_PREFIX)])
    assert setup.store.puts == []



def _broken_body(size: int):
    yield b"<Error><Co"
    raise ConnectionError("https://store.test.invalid/the-whole-url?X-Amz-Signature=0")


def test_upload_when_a_403s_error_body_breaks_off_should_ask_for_a_fresh_url_and_count_the_refusal(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [FakeResponse(403, {}, chunks=_broken_body)] * 4
    # Act
    failure = _failure(setup.upload).failure
    # Assert: every refusal was answered with a fresh URL, and four of them end the upload.
    assert failure.code == "upload_url_refused"
    assert _asks_of(setup, "thumbnail.jpg") == 4
    thumbnail_urls = [put.url for put in setup.store.puts if put.path == "thumbnail.jpg"]
    assert len(set(thumbnail_urls)) == 4


def test_upload_when_a_429s_error_body_breaks_off_should_wait_its_retry_after(tmp_path):
    # Arrange
    setup = _Setup(tmp_path)
    setup.store.put_answers["thumbnail.jpg"] = [FakeResponse(429, {"Retry-After": "20"}, chunks=_broken_body)]
    # Act
    setup.upload()
    # Assert
    assert "thumbnail.jpg" in setup.store.objects
    assert sum(setup.waits.seconds) == pytest.approx(20.0)


def test_upload_when_its_session_factory_raises_should_fail_retryable_as_a_request_that_could_not_start(
    tmp_path, caplog
):
    # Arrange: a factory whose error quotes a URL.
    setup = _Setup(tmp_path)

    def failing_factory():
        raise MemoryError("https://store.test.invalid/the-whole-url?X-Amz-Signature=0")

    # Act
    with caplog.at_level(logging.DEBUG):
        raised = _failure(lambda: upload_generation_by_urls(
            setup.client, failing_factory, prefix=setup.claimed.grant.output_prefix,
            output_root=setup.generation.output_root, artifacts=setup.generation.artifacts,
            manifest=setup.generation.manifest, deadline=Deadline.after_ms(60_000), sleep=setup.waits,
        ))
    # Assert: the failure says no session could be made, the error is neither its cause nor its context, and no line
    # quotes it.
    failure = raised.failure
    assert (failure.error_class, failure.code, failure.retryable) == (
        "resource_exhausted", "storage_request_start_failed", True)
    assert failure.detail == "a storage request could not be started: no session could be made for it"
    assert raised.__cause__ is None and raised.__context__ is None
    assert "X-Amz-Signature" not in caplog.text and "store.test.invalid" not in caplog.text
    assert setup.store.puts == []


# --- the lanes' sessions on the loopback address (TD-196) -----------------------------------------------------------

# The ceiling a test gives a PUT it means to be given up; how often a trickling answer sends its next byte (well inside
# that ceiling, which is also the request's read time-out, so that only the give-up ends the request); how long the
# server holds such an answer at most, so that no test can hang the suite; how soon after the give-up the server must
# see the connection closed and the request's thread must have ended.
CEILING_S = 0.3
LONG_CEILING_S = 10.0
TRICKLE_EVERY_S = 0.05
HOLDS_AT_MOST_S = 15.0
CUT_SEEN_WITHIN_S = 3.0
DEADLINE_MS = 800
# How long a refusal's error body says it is, and how much of it comes before the connection closes.
ERROR_BODY_DECLARED_BYTES = 100
ERROR_BODY_SENT_BYTES = 10


@dataclass
class _Connection:
    """One connection the loopback store accepted, and when it saw the client close it."""

    number: int
    closed: threading.Event = field(default_factory=threading.Event)
    closed_at: float | None = None

    def close_seen(self) -> None:
        self.closed_at = time.monotonic()
        self.closed.set()


@dataclass(frozen=True)
class _Arrival:
    """One PUT the loopback store read whole: its object's path, its connection and when its body had been read."""

    path: str
    connection: _Connection
    at: float


@dataclass(frozen=True)
class _Trickle:
    """An answer whose head never ends: once `after` (if any) has returned, a status line and then one byte of a header
    every TRICKLE_EVERY_S, until the client closes the connection."""

    after: Callable[[], object] | None = None


@dataclass(frozen=True)
class _Silent:
    """No answer at all: nothing is sent back until the client closes the connection, so that a request given up and
    cut ends with a transfer error, not with the answer cut short that a _Trickle leaves. The request's read time-out
    is the ceiling, so against a silent store it could end the request itself, at the same instant as the give-up:
    a test that means the give-up to end it sets the store's `read_timeout_s` longer."""


@dataclass(frozen=True)
class _RefusalBrokenOff:
    """A 400 whose error body breaks off: its head declares ERROR_BODY_DECLARED_BYTES, then ERROR_BODY_SENT_BYTES of
    them come and the connection closes, so that reading the body for its error code ends with a transfer error."""


class _Lane(RequestsSession):
    """A lane's real session (one connection, as the runner's upload_session() makes it) whose PUTs go to the loopback
    store, the granted URL's path kept; it notes the thread each PUT runs on and when it was cut."""

    def __init__(self, store: "_LoopbackStore") -> None:
        super().__init__(connections=1)
        self._store = store
        self.cut_at: float | None = None
        self.puts: list[tuple[str, threading.Thread, float]] = []

    def put(self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *,
            stop_line: StopLine) -> HttpResponse:
        path = urlsplit(url).path
        if self._store.read_timeout_s is not None:
            timeout = (timeout[0], self._store.read_timeout_s)
        self.puts.append((path.removeprefix(self._store.prefix_path), threading.current_thread(), time.monotonic()))
        return super().put(self._store.base + path, data, headers, timeout, stop_line=stop_line)

    def abandon(self) -> None:
        self.cut_at = time.monotonic()
        super().abandon()


class _LoopbackStore:
    """A store on the loopback address. `plan` maps an object's path to what its next PUTs get: a _Trickle, a _Silent,
    a _RefusalBrokenOff, or a callable run before the 200; every other PUT is answered 200 at once, with an empty
    body."""

    def __init__(self, prefix_path: str) -> None:
        self.prefix_path = prefix_path
        self.base = ""
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.connections: list[_Connection] = []
        self.arrivals: list[_Arrival] = []
        self.answered: list[str] = []
        self.plan: dict[str, list[Any]] = {}
        self.lanes: list[_Lane] = []
        # A read time-out each lane's PUTs are sent with instead of the upload's (None: the upload's own).
        self.read_timeout_s: float | None = None

    def lane(self) -> _Lane:
        """The upload's session factory."""
        made = _Lane(self)
        with self.lock:
            self.lanes.append(made)
        return made

    def arrivals_of(self, path: str) -> list[_Arrival]:
        with self.lock:
            return [arrival for arrival in self.arrivals if arrival.path == path]

    def lane_of(self, path: str) -> _Lane:
        """The lane whose session made the first PUT of `path`."""
        return next(lane for lane in self.lanes if any(put[0] == path for put in lane.puts))

    def put_thread_of(self, path: str) -> threading.Thread:
        """The thread the first PUT of `path` ran on."""
        return next(put[1] for put in self.lane_of(path).puts if put[0] == path)


def _store_handler(store: _LoopbackStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the standard library's name
            return

        def setup(self) -> None:
            super().setup()
            with store.lock:
                self.accepted = _Connection(len(store.connections) + 1)
                store.connections.append(self.accepted)

        def do_PUT(self) -> None:  # noqa: N802 - the standard library's name
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            path = urlsplit(self.path).path.removeprefix(store.prefix_path)
            with store.lock:
                store.arrivals.append(_Arrival(path, self.accepted, time.monotonic()))
                planned = store.plan.get(path, [])
                behaviour = planned.pop(0) if planned else None
            if len(body) < length:
                return
            if isinstance(behaviour, _Trickle):
                self._trickle(behaviour)
                return
            if isinstance(behaviour, _Silent):
                self._stay_silent()
                return
            if isinstance(behaviour, _RefusalBrokenOff):
                self._refuse_and_break_off()
                return
            if callable(behaviour):
                behaviour()
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            with store.lock:
                store.answered.append(path)

        def _trickle(self, trickle: _Trickle) -> None:
            self.close_connection = True
            if trickle.after is not None:
                trickle.after()
            sock = self.connection
            started = time.monotonic()
            try:
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
                while not store.stop.is_set() and time.monotonic() - started < HOLDS_AT_MOST_S:
                    readable, _, _ = select.select([sock], [], [], TRICKLE_EVERY_S)
                    if readable and sock.recv(1) == b"":
                        self.accepted.close_seen()
                        return
                    self.wfile.write(b"a")
            except OSError:
                # The client shut the connection: what the test waits for.
                self.accepted.close_seen()

        def _refuse_and_break_off(self) -> None:
            self.close_connection = True
            self.wfile.write(
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/xml\r\n"
                + f"Content-Length: {ERROR_BODY_DECLARED_BYTES}\r\nConnection: close\r\n\r\n".encode("ascii")
                + b"<Error><Co"[:ERROR_BODY_SENT_BYTES]
            )

        def _stay_silent(self) -> None:
            self.close_connection = True
            sock = self.connection
            started = time.monotonic()
            try:
                while not store.stop.is_set() and time.monotonic() - started < HOLDS_AT_MOST_S:
                    readable, _, _ = select.select([sock], [], [], TRICKLE_EVERY_S)
                    if readable and sock.recv(1) == b"":
                        self.accepted.close_seen()
                        return
            except OSError:
                # The client shut the connection: what the test waits for.
                self.accepted.close_seen()

    return Handler


class _StoreServer(ThreadingHTTPServer):
    """The standard library's server without its look-up of the host's name (server_bind asks socket.getfqdn())."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A connection the client shut is what these tests make: nothing to report.
        return


@contextmanager
def _loopback_store(setup: _Setup) -> Iterator[_LoopbackStore]:
    store = _LoopbackStore(f"/{OUTPUT_BUCKET}/{setup.claimed.grant.output_prefix}/")
    server = _StoreServer(("127.0.0.1", 0), _store_handler(store))
    store.base = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield store
    finally:
        store.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def _upload_to(store: _LoopbackStore, setup: _Setup, *, parallelism: int, deadline: Deadline | None = None) -> int:
    return upload_generation_by_urls(
        setup.client, store.lane, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
        artifacts=setup.generation.artifacts, manifest=setup.generation.manifest,
        deadline=deadline or Deadline.after_ms(60_000), sleep=setup.waits, parallelism=parallelism,
    )


def _short_ceiling_for(setup: _Setup, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """Every PUT of `path` (told by its size, which no other object of the setup shares) may take CEILING_S, every
    other one LONG_CEILING_S, so that only `path`'s PUTs are given up whatever the machine's load."""
    sizes = {artifact.path: artifact.size_bytes for artifact in setup.generation.artifacts}
    others = [size for other, size in sizes.items() if other != path] + [len(MANIFEST_BYTES)]
    assert sizes[path] not in others
    monkeypatch.setattr(
        url_upload, "_request_ceiling_s", lambda size: CEILING_S if size == sizes[path] else LONG_CEILING_S
    )


def _ended(thread: threading.Thread) -> bool:
    """Whether `thread` ended within CUT_SEEN_WITHIN_S and is no longer among the process's threads."""
    return wait_until(lambda: not thread.is_alive(), within_s=CUT_SEEN_WITHIN_S) and all(
        running is not thread for running in threading.enumerate())


@pytest.mark.parametrize("withheld", [_Trickle(), _Silent()], ids=["answer-trickles", "no-answer"])
def test_upload_when_a_put_is_given_up_at_its_ceiling_should_close_its_connection_end_its_thread_and_retry_anew(
    tmp_path, monkeypatch, withheld
):
    # Arrange: the thumbnail's first answer never completes (its head trickles: the cut request ends with what was read
    # of it; or nothing comes: it ends with a transfer error); a PUT of the thumbnail may take 0.3 s. The bytes each PUT
    # of the thumbnail is read into are noted (the list's reference is the only one the test keeps), and the first one's
    # reference count is taken while the retry reads the file again, once the given-up PUT's thread has ended. The
    # cyclic collector is off, so that a reference cycle keeping the body would show.
    setup = _Setup(tmp_path, segments=0)
    _short_ceiling_for(setup, monkeypatch, "thumbnail.jpg")
    before = storage_module.abandoned_requests()
    read_bodies: list[bytes] = []
    counted_at_retry: list[int] = []
    alone = [bytes(bytearray(len(b"the thumbnail")))]
    held_by_a_list_alone = sys.getrefcount(alone[0])
    bytes_of = url_upload._bytes_of
    collecting = gc.isenabled()
    gc.disable()
    try:
        with _loopback_store(setup) as store:
            def noting_bytes_of(item):
                if item.upload.relative_path == "thumbnail.jpg" and read_bodies:
                    wait_until(lambda: not store.put_thread_of("thumbnail.jpg").is_alive(), within_s=CUT_SEEN_WITHIN_S)
                    counted_at_retry.append(sys.getrefcount(read_bodies[0]))
                data = bytes_of(item)
                if item.upload.relative_path == "thumbnail.jpg":
                    read_bodies.append(data)
                return data

            monkeypatch.setattr(url_upload, "_bytes_of", noting_bytes_of)
            store.plan["thumbnail.jpg"] = [withheld]
            # Only the give-up at the ceiling may end the withheld answer, never the read time-out.
            store.read_timeout_s = LONG_CEILING_S
            # Act
            uploaded = _upload_to(store, setup, parallelism=1)
            first, _ = store.arrivals_of("thumbnail.jpg")
            closed = first.connection.closed.wait(CUT_SEEN_WITHIN_S)
            given_up_lane = store.lane_of("thumbnail.jpg")
            thread_ended = _ended(store.put_thread_of("thumbnail.jpg"))
            lanes = list(store.lanes)
            counted_after = sys.getrefcount(read_bodies[0])
            holders = [
                type(holder).__name__ for holder in gc.get_referrers(read_bodies[0]) if holder is not read_bodies
            ]
    finally:
        if collecting:
            gc.enable()
    # Assert: the given-up PUT's connection was closed by the client and its thread ended, and nothing but the test's
    # list holds its body once the upload returned, nor while the retry read the file; the lane was cut before the
    # object's next attempt, which went through a fresh session and was stored.
    assert uploaded == setup.generation.manifest.total_bytes + len(MANIFEST_BYTES)
    assert closed and thread_ended
    assert len(read_bodies) == 2 and read_bodies[0] is not read_bodies[1]
    assert counted_after == held_by_a_list_alone, holders
    assert counted_at_retry == [held_by_a_list_alone]
    assert len(lanes) == 2 and lanes[0] is given_up_lane
    assert [path for path, _, _ in given_up_lane.puts] == ["thumbnail.jpg"]
    retried_at = next(at for path, _, at in lanes[1].puts if path == "thumbnail.jpg")
    assert given_up_lane.cut_at is not None and given_up_lane.cut_at < retried_at
    assert store.answered.count("thumbnail.jpg") == 1
    assert storage_module.abandoned_requests() - before == 1


def test_upload_when_one_lanes_put_is_cut_should_let_another_lanes_put_in_flight_end_with_its_200(
    tmp_path, monkeypatch
):
    # Arrange: two lanes. The thumbnail's first answer never completes and is given up at 0.3 s; the playlist's PUT,
    # under a ceiling of 10 s, is answered only once the thumbnail's connection was seen closed, so it is in flight
    # through the cut.
    setup = _Setup(tmp_path, segments=0)
    _short_ceiling_for(setup, monkeypatch, "thumbnail.jpg")
    playlist_arrived, thumbnail_arrived = threading.Event(), threading.Event()
    with _loopback_store(setup) as store:
        def both_under_way() -> None:
            thumbnail_arrived.set()
            playlist_arrived.wait(CUT_SEEN_WITHIN_S)

        def after_the_cut() -> None:
            playlist_arrived.set()
            thumbnail_arrived.wait(CUT_SEEN_WITHIN_S)
            store.arrivals_of("thumbnail.jpg")[0].connection.closed.wait(CUT_SEEN_WITHIN_S)

        store.plan["thumbnail.jpg"] = [_Trickle(after=both_under_way)]
        store.plan["v360/playlist.m3u8"] = [after_the_cut]
        # Act
        _upload_to(store, setup, parallelism=2)
        thumbnail_first = store.arrivals_of("thumbnail.jpg")[0]
        (playlist,) = store.arrivals_of("v360/playlist.m3u8")
        playlist_lane = store.lane_of("v360/playlist.m3u8")
        thumbnail_lane = store.lane_of("thumbnail.jpg")
    # Assert: the playlist's one PUT was under way when the thumbnail's lane was cut, kept its own connection and
    # session, and was answered 200.
    assert thumbnail_first.connection.closed_at is not None and playlist.at < thumbnail_first.connection.closed_at
    assert playlist.connection is not thumbnail_first.connection
    assert playlist_lane is not thumbnail_lane
    assert thumbnail_lane.cut_at is not None and playlist_lane.cut_at is None
    playlist_began_at = next(at for path, _, at in playlist_lane.puts if path == "v360/playlist.m3u8")
    assert playlist_began_at < thumbnail_lane.cut_at
    assert store.answered.count("v360/playlist.m3u8") == 1
    assert len(store.arrivals_of("thumbnail.jpg")) == 2


def test_upload_when_a_put_is_given_up_at_the_deadline_should_close_its_connection_and_end_final(tmp_path):
    # Arrange: the thumbnail's answer never completes, under the request's own long ceiling and a deadline of 0.8 s.
    setup = _Setup(tmp_path, segments=0)
    with _loopback_store(setup) as store:
        store.plan["thumbnail.jpg"] = [_Trickle()]
        # Act
        failure = _failure(lambda: _upload_to(store, setup, parallelism=1, deadline=Deadline.after_ms(DEADLINE_MS)))
        (first,) = store.arrivals_of("thumbnail.jpg")
        closed = first.connection.closed.wait(CUT_SEEN_WITHIN_S)
        thread_ended = _ended(store.put_thread_of("thumbnail.jpg"))
        lane = store.lane_of("thumbnail.jpg")
    # Assert: the deadline's failure, final; the request's connection closed and its thread ended.
    assert (failure.failure.code, failure.failure.retryable) == ("upload_deadline_exceeded", False)
    assert closed and thread_ended
    assert lane.cut_at is not None
    assert store.answered == []


def test_result_when_its_put_is_given_up_should_close_its_connection_and_raise_as_before(tmp_path, monkeypatch):
    # Arrange: the stored result's answer never completes; its ceiling is 0.3 s.
    setup = _Setup(tmp_path, segments=0)
    monkeypatch.setattr(url_upload, "_request_ceiling_s", lambda size: CEILING_S)
    with _loopback_store(setup) as store:
        store.plan[RESULT_NAME] = [_Trickle()]
        # Act
        failure = _failure(lambda: upload_result_by_url(
            setup.client, store.lane, result=_result(setup), deadline=Deadline.after_ms(60_000)))
        (first,) = store.arrivals_of(RESULT_NAME)
        closed = first.connection.closed.wait(CUT_SEEN_WITHIN_S)
        thread_ended = _ended(store.put_thread_of(RESULT_NAME))
        lane = store.lane_of(RESULT_NAME)
    # Assert: what a failed stored result raised before, its connection closed and its thread ended.
    assert failure.failure.code == "upload_failed"
    assert closed and thread_ended
    assert lane.cut_at is not None


def test_upload_when_one_lane_puts_two_objects_in_turn_should_open_a_connection_for_each_put_and_keep_the_lanes_session(
    tmp_path
):
    # Arrange: one lane; every PUT is answered 200 at once (its answer is closed unread, which closes its connection).
    setup = _Setup(tmp_path, segments=0)
    with _loopback_store(setup) as store:
        # Act
        _upload_to(store, setup, parallelism=1)
        (thumbnail,) = store.arrivals_of("thumbnail.jpg")
        (playlist,) = store.arrivals_of("v360/playlist.m3u8")
        lanes = list(store.lanes)
    # Assert: the two objects went one after the other through the lane's one session, each on a connection of its own.
    assert thumbnail.at < playlist.at
    assert store.answered[:2] == ["thumbnail.jpg", "v360/playlist.m3u8"]
    assert len(lanes) == 1 and lanes[0].cut_at is None
    assert [put[0] for put in lanes[0].puts] == ["thumbnail.jpg", "v360/playlist.m3u8", MANIFEST_NAME]
    assert playlist.connection is not thumbnail.connection


def test_upload_when_a_refusals_error_body_breaks_off_should_keep_no_reference_to_that_puts_body(tmp_path, monkeypatch):
    # Arrange: the thumbnail's first PUT is refused with a 400 whose error body breaks off (read for its error code,
    # it ends with a transfer error: no answer, tried again); its second is stored. The bytes each PUT of the thumbnail
    # is read into are noted (the list's reference is the only one the test keeps). The cyclic collector is off, so
    # that a reference cycle keeping the first body would show.
    setup = _Setup(tmp_path, segments=0)
    read_bodies: list[bytes] = []
    alone = [bytes(bytearray(len(b"the thumbnail")))]
    held_by_a_list_alone = sys.getrefcount(alone[0])
    bytes_of = url_upload._bytes_of

    def noting_bytes_of(item):
        data = bytes_of(item)
        if item.upload.relative_path == "thumbnail.jpg":
            read_bodies.append(data)
        return data

    monkeypatch.setattr(url_upload, "_bytes_of", noting_bytes_of)
    collecting = gc.isenabled()
    gc.disable()
    try:
        with _loopback_store(setup) as store:
            store.plan["thumbnail.jpg"] = [_RefusalBrokenOff()]
            # Act
            _upload_to(store, setup, parallelism=1)
            first_put_thread = store.put_thread_of("thumbnail.jpg")
            thread_ended = _ended(first_put_thread)
            counted_after = sys.getrefcount(read_bodies[0])
            holders = [
                type(holder).__name__ for holder in gc.get_referrers(read_bodies[0]) if holder is not read_bodies
            ]
    finally:
        if collecting:
            gc.enable()
    # Assert: the refusal was tried again and stored, and nothing but the test's list holds the refused PUT's body.
    assert store.answered.count("thumbnail.jpg") == 1
    assert len(store.arrivals_of("thumbnail.jpg")) == 2
    assert thread_ended
    assert len(read_bodies) == 2 and read_bodies[0] is not read_bodies[1]
    assert counted_after == held_by_a_list_alone, holders
