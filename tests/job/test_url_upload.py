"""The upload of protocol 2.2.0 through put URLs (core/job/url_upload.py): what it stores and in which order, how many
PUTs run at once, what each PUT carries, the batches, the retries, the refusals, the stop at the first failure and at
the deadline, and the stored result. The Hub and the store are fakes; no network."""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path

import pytest

from core.bounded_process import Deadline
from core.encode import ArtifactFile
from core.failure import WorkerFailure
from core.job import storage as storage_module
from core.job import url_upload
from core.job.url_upload import upload_generation_by_urls, upload_result_by_url
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
    assert len(setup.waits.seconds) == 1


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
    _failure(lambda: upload_generation_by_urls(
        setup.client, setup.store, prefix=setup.claimed.grant.output_prefix, output_root=setup.generation.output_root,
        artifacts=setup.generation.artifacts, manifest=setup.generation.manifest,
        deadline=Deadline.after_ms(60_000), sleep=setup.waits, parallelism=1,
    ))
    # Assert
    lost = setup.client.lost()
    assert lost is not None and lost.failure.code == "lease_unrenewed"
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
