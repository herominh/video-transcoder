"""JobRunner end to end at protocol 2.2.0 with the real ffprobe and ffmpeg (skipped without them; POSIX only): the
source read through its granted URL, the generation stored through put URLs, the stages reported in order; and the
same media through a 2.0.0 dispatch still taking the credential path. The Hub and the store are fakes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from core import profile as profile_module
from core.job.runner import JobRunner
from core.profile import profile_digest
from core.protocol import pipeline
from tests.contract.v2_support import FakeClock, FakeTransport, uuid_source
from tests.job.capability_fakes import (
    CAPABILITIES_URL,
    FakeHub,
    FakeStore,
    capability_claimed,
    capability_settings,
)
from tests.job.conftest import (
    COMPLETED_KIND,
    DELIVERED,
    MANIFEST_NAME,
    FakeS3,
    accepted_by_the_hub,
    claimed_execution,
    hub_storage_context,
    job_settings,
    runtime_of,
)
from tests.job.test_runner_end_to_end import OUTPUT_BUCKET, PROFILE, REQUESTED, SOURCE_BUCKET, SOURCE_ETAG, SOURCE_KEY

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the stages run under POSIX process groups")

STAGES = ["probing", "downloading", "transcoding", "finalizing", "uploading"]


def _changes(media: Path) -> dict[str, Any]:
    return {
        "/source/size_bytes": media.stat().st_size,
        "/source/etag": SOURCE_ETAG,
        "/renditions": REQUESTED,
        "/profile": {"id": PROFILE.profile_id, "version": PROFILE.version, "sha256": profile_digest(PROFILE)},
    }


def test_job_when_the_dispatch_is_two_two_zero_should_complete_through_urls_alone(source_media, tmp_path):
    # Arrange
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    claimed = capability_claimed(request_changes=_changes(source_media))
    clock = FakeClock()
    hub = FakeHub(claimed, clock)
    store = FakeStore(hub, source_media.read_bytes())
    factories: list[str] = []
    runner = JobRunner(
        job_settings(scratch_root),
        source_client_factory=lambda access: factories.append("source"),
        output_client_factory=lambda access: factories.append("output"),
        url_session_factory=lambda: store,
    )
    runtime = dataclasses.replace(runtime_of(hub, clock, capability_settings()), new_uuid=uuid_source())
    # Act
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(profile_module, "SUPPORTED_PROFILES", (PROFILE,))
        sealed = runner.run(claimed, runtime)
    result = accepted_by_the_hub(sealed, claimed, now=clock())
    # Assert
    assert result["message_kind"] == COMPLETED_KIND, result
    assert factories == []
    manifest_bytes = store.objects[MANIFEST_NAME]
    assert pipeline.validate(manifest_bytes, hub_storage_context(claimed)).accepted
    assert result["manifest"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    listed = {artifact["path"]: artifact for artifact in json.loads(manifest_bytes)["artifacts"]}
    assert set(store.objects) == set(listed) | {MANIFEST_NAME, "result.json"}
    assert store.stored_order[-2:] == [MANIFEST_NAME, "result.json"]
    for path, artifact in listed.items():
        assert hashlib.sha256(store.objects[path]).hexdigest() == artifact["sha256"]
    reported: list[str] = []
    for request in hub.requests:
        if not reported or reported[-1] != request["stage"]:
            reported.append(request["stage"])
    assert reported == STAGES
    assert hub.requests[0]["media_duration_ms"] is None
    assert hub.requests[-1]["media_duration_ms"] is not None


def test_job_when_the_same_media_comes_at_two_zero_zero_should_take_the_credential_path(source_media, tmp_path):
    # Arrange: the image accepts both versions; this dispatch is signed at 2.0.0.
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    s3 = FakeS3()
    s3.hold(SOURCE_BUCKET, SOURCE_KEY, source_media, SOURCE_ETAG)
    claimed = claimed_execution(encrypted=False, request_changes=_changes(source_media), settings=capability_settings())
    clock = FakeClock()
    transport = FakeTransport(then=DELIVERED, clock=clock)
    sessions: list[str] = []
    runner = JobRunner(
        job_settings(scratch_root), source_client_factory=lambda access: s3, output_client_factory=lambda access: s3,
        url_session_factory=lambda: sessions.append("made"),
    )
    # Act
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(profile_module, "SUPPORTED_PROFILES", (PROFILE,))
        sealed = runner.run(claimed, runtime_of(transport, clock, capability_settings()))
    result = accepted_by_the_hub(sealed, claimed, now=clock())
    # Assert
    assert result["message_kind"] == COMPLETED_KIND
    assert sessions == []
    assert transport.calls_to(CAPABILITIES_URL) == []
    assert {call.bucket for call in s3.stored} == {OUTPUT_BUCKET}
    assert s3.stored[-2].key.endswith(MANIFEST_NAME)
