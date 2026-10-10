"""The last outputs of a job, end to end with the real ffmpeg: two renditions encoded within the room
`reserve_package` leaves them, the thumbnail, the master playlist and the manifest, which the
contract validator accepts and which describes exactly the files on disk."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from core.bounded_process import Deadline
from core.encode import OutputAllowance, RenditionOutput, encode_rendition
from core.encode_command import EncodeSource
from core.package import (
    EncryptionRef,
    ManifestIdentity,
    ManifestOutput,
    ProfileRef,
    manifest_max_bytes,
    reserve_package,
    write_manifest,
    write_master_playlist,
)
from core.profile import GIB, LadderRung
from core.protocol.context import TrustedContext
from core.protocol.pipeline import validate
from core.renditions import plan_renditions
from core.thumbnail import ThumbnailResult, make_thumbnail
from tests.encode.conftest import EncodeMedia, fast_profile, probe_facts
from tests.preflight.conftest import FFMPEG_TIMEOUT_S

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the encode runs under POSIX process groups")

# Two rungs the 320x240 fixture fits (the pilot ladder has only one at or below 240 lines): contract
# names, with smaller short edges than the names say.
TEST_LADDER = (
    LadderRung("360p", 240, 600_000, 640_000, 900_000, 64_000),
    LadderRung("240p", 120, 300_000, 320_000, 450_000, 48_000),
)
PROFILE = fast_profile(ladder=TEST_LADDER)
REQUESTED = ["360p", "240p"]
JOB_ARTIFACTS = 500
GENEROUS_DEADLINE_MS = 120_000
PROFILE_SHA256 = "cd" * 32
MANIFEST_NAME = "generation-manifest.json"


@dataclass(frozen=True, slots=True)
class _Package:
    output_root: Path
    work_dir: Path
    arguments: dict[str, Any]  # what the manifest was written from
    manifest: ManifestOutput

    @property
    def raw(self) -> bytes:
        return (self.output_root / MANIFEST_NAME).read_bytes()


def _encode_job(root: Path, media_path: Path, media_key: bytes | None) -> _Package:
    """One job's outputs, stage by stage, as D1 will run them."""
    facts = probe_facts(media_path)
    output_root, work_dir = root / "output", root / "work"
    output_root.mkdir()
    work_dir.mkdir()
    source = EncodeSource.from_facts(media_path, facts, PROFILE)
    plans = plan_renditions(
        facts.video_streams[0], has_audio=bool(facts.audio_streams), requested=REQUESTED, profile=PROFILE
    )
    job = OutputAllowance.for_job(
        max_output_bytes=GIB,
        max_artifact_count=JOB_ARTIFACTS,
        source_size_bytes=media_path.stat().st_size,
        profile=PROFILE,
    )
    allowance = reserve_package(job, PROFILE)
    deadline = Deadline.after_ms(GENEROUS_DEADLINE_MS)
    renditions: list[RenditionOutput] = []
    for plan in plans:
        output = encode_rendition(
            source,
            plan,
            profile=PROFILE,
            output_root=output_root,
            work_dir=work_dir,
            media_key=media_key,
            allowance=allowance,
            deadline=deadline,
        )
        allowance = allowance.after(output)
        renditions.append(output)
    thumbnail = make_thumbnail(
        source, plans[0], profile=PROFILE, output_root=output_root, work_dir=work_dir, deadline=deadline
    )
    master = write_master_playlist(renditions, output_root=output_root)
    arguments: dict[str, Any] = {
        "identity": ManifestIdentity(
            org_uuid=str(uuid.uuid4()),
            video_uuid=str(uuid.uuid4()),
            source_id=str(uuid.uuid4()),
            attempt_id=str(uuid.uuid4()),
            dispatch_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
        ),
        "profile_ref": ProfileRef(id=PROFILE.profile_id, version=PROFILE.version, sha256=PROFILE_SHA256),
        "profile": PROFILE,
        "encryption": (
            EncryptionRef(mode="none", media_key_id=None)
            if media_key is None
            else EncryptionRef(mode="aes-128", media_key_id=str(uuid.uuid4()))
        ),
        "requested_renditions": REQUESTED,
        "renditions": renditions,
        "master": master,
        "thumbnail": thumbnail,
        "created_at": datetime.now(timezone.utc),
    }
    manifest = write_manifest(**arguments, output_root=output_root, max_bytes=manifest_max_bytes(job.max_artifacts))
    return _Package(output_root=output_root, work_dir=work_dir, arguments=arguments, manifest=manifest)


@pytest.fixture(scope="module")
def plain_package(encode_media: EncodeMedia, tmp_path_factory: pytest.TempPathFactory) -> _Package:
    return _encode_job(tmp_path_factory.mktemp("package-plain"), encode_media.landscape, None)


@pytest.fixture(scope="module")
def encrypted_package(encode_media: EncodeMedia, tmp_path_factory: pytest.TempPathFactory) -> _Package:
    return _encode_job(tmp_path_factory.mktemp("package-encrypted"), encode_media.landscape, os.urandom(16))


@pytest.fixture(params=["plain_package", "encrypted_package"], ids=["plain", "encrypted"])
def package(request: pytest.FixtureRequest) -> _Package:
    return request.getfixturevalue(request.param)


def _worker_sender_context(arguments: dict[str, Any]) -> TrustedContext:
    identity: ManifestIdentity = arguments["identity"]
    profile_ref: ProfileRef = arguments["profile_ref"]
    encryption: EncryptionRef = arguments["encryption"]
    expected_encryption = {"mode": encryption.mode}
    if encryption.media_key_id is not None:
        expected_encryption["media_key_id"] = encryption.media_key_id
    return TrustedContext.from_dict(
        {
            "context_version": 1,
            "role": "worker_sender",
            "channel": "storage",
            "accepted_manifest_versions": ["1.0.0-draft"],
            "accepted_message_kinds": ["generation.manifest"],
            "expect": {
                "org_uuid": identity.org_uuid,
                "video_uuid": identity.video_uuid,
                "attempt_id": identity.attempt_id,
                "dispatch_id": identity.dispatch_id,
                "execution_id": identity.execution_id,
                "source_id": identity.source_id,
                "profile": {"id": profile_ref.id, "version": profile_ref.version, "sha256": profile_ref.sha256},
                "encryption": expected_encryption,
                "renditions": list(arguments["requested_renditions"]),
                "max_artifact_count": JOB_ARTIFACTS,
            },
        }
    )


def _listed(package: _Package) -> dict[str, Any]:
    return {artifact["path"]: artifact for artifact in json.loads(package.raw)["artifacts"]}


def test_package_when_a_job_is_encoded_end_to_end_should_write_a_manifest_the_contract_validator_accepts(
    package: _Package,
) -> None:
    # Arrange
    context = _worker_sender_context(package.arguments)

    # Act
    verdict = validate(package.raw, context)

    # Assert
    assert verdict.accepted, verdict
    thumbnail: ThumbnailResult = package.arguments["thumbnail"]
    assert thumbnail.artifact is not None
    assert [output.name for output in package.arguments["renditions"]] == REQUESTED


def test_package_when_a_job_is_encoded_end_to_end_should_leave_exactly_the_manifest_and_its_artifacts(
    package: _Package,
) -> None:
    # Arrange
    listed = set(_listed(package))

    # Act
    on_disk = {
        path.relative_to(package.output_root).as_posix()
        for path in package.output_root.rglob("*")
        if not path.is_dir()
    }
    directories = {path.name for path in package.output_root.rglob("*") if path.is_dir()}

    # Assert
    assert on_disk == listed | {MANIFEST_NAME}
    assert directories == set(REQUESTED)
    assert list(package.work_dir.iterdir()) == []


def test_package_when_a_job_is_encoded_end_to_end_should_list_every_file_with_its_true_size_and_digest(
    package: _Package,
) -> None:
    # Arrange
    listed = _listed(package)
    raw = package.raw

    # Act
    on_disk = {path: (package.output_root / path).read_bytes() for path in listed}

    # Assert
    for path, artifact in listed.items():
        assert artifact["size_bytes"] == len(on_disk[path])
        assert artifact["sha256"] == hashlib.sha256(on_disk[path]).hexdigest()
    assert package.manifest.size_bytes == len(raw)
    assert package.manifest.sha256 == hashlib.sha256(raw).hexdigest()
    assert package.manifest.artifact_count == len(listed)
    assert package.manifest.total_bytes == sum(len(data) for data in on_disk.values())


def test_package_when_a_plain_job_is_encoded_end_to_end_should_decode_through_its_master_playlist(
    plain_package: _Package,
) -> None:
    # Arrange: only the plain job (encrypted renditions name a key URI only the Hub serves)
    master = plain_package.output_root / "master.m3u8"

    # Act
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
            "-protocol_whitelist", "file", "-f", "hls", "-i", str(master),
            "-map", "0:v:0", "-f", "null", "-",
        ],
        capture_output=True,
        timeout=FFMPEG_TIMEOUT_S,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
