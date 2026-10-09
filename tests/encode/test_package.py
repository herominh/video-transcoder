from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import core.package
from core.encode import ArtifactFile, OutputAllowance, RenditionOutput
from core.failure import Diagnostic, WorkerFailure
from core.package import (
    MANIFEST_ARTIFACT_MAX_BYTES,
    MANIFEST_HEADER_MAX_BYTES,
    MASTER_PLAYLIST_MAX_BYTES,
    EncryptionRef,
    ManifestIdentity,
    ManifestOutput,
    ProfileRef,
    manifest_bytes,
    manifest_max_bytes,
    master_playlist_bytes,
    reserve_package,
    write_manifest,
    write_master_playlist,
)
from core.profile import GIB, MIB, PILOT_PROFILE, RENDITION_NAMES, MediaProfile
from core.thumbnail import ThumbnailResult
from tests.contract.context import TrustedContext
from tests.contract.pipeline import validate
from tests.encode.conftest import fast_profile

ORG_UUID = "11111111-1111-7111-8111-111111111111"
VIDEO_UUID = "22222222-2222-7222-8222-222222222222"
ATTEMPT_ID = "33333333-3333-7333-8333-333333333333"
DISPATCH_ID = "44444444-4444-7444-8444-444444444444"
EXECUTION_ID = "55555555-5555-7555-8555-555555555555"
SOURCE_ID = "66666666-6666-7666-8666-666666666666"
MEDIA_KEY_ID = "77777777-7777-7777-8777-777777777777"
# Holds letters, so its uppercase form differs.
LETTERED_UUID = "abcdef01-2345-7789-8abc-def012345678"
PROFILE_SHA256 = "ab" * 32
IDENTITY = ManifestIdentity(
    org_uuid=ORG_UUID,
    video_uuid=VIDEO_UUID,
    source_id=SOURCE_ID,
    attempt_id=ATTEMPT_ID,
    dispatch_id=DISPATCH_ID,
    execution_id=EXECUTION_ID,
)
PROFILE_REF = ProfileRef(id=PILOT_PROFILE.profile_id, version=PILOT_PROFILE.version, sha256=PROFILE_SHA256)
ENCRYPTED = EncryptionRef(mode="aes-128", media_key_id=MEDIA_KEY_ID)
PLAIN = EncryptionRef(mode="none", media_key_id=None)
CREATED_AT = datetime(2029, 12, 31, 23, 58, tzinfo=timezone.utc)
AUDIO_CODECS = "avc1.64001f,mp4a.40.2"
VIDEO_CODECS = "avc1.64001f"
PAL_RATE = (25, 1)
NTSC_RATE = (30000, 1001)
THREE_RUNGS = (
    ("720p", 1280, 720, 2_800_000, 2_400_000),
    ("480p", 854, 480, 1_400_000, 1_200_000),
    ("360p", 640, 360, 800_000, 700_000),
)
PORTRAIT_RUNGS = (("720p", 720, 1280, 2_800_000, 2_400_000), ("360p", 360, 640, 800_000, 700_000))
TWELVE_SECONDS_MS = 12_000
PLAYLIST_BYTES = 384
SEGMENT_BYTES = 500_000
MASTER = ArtifactFile(path="master.m3u8", kind="hls_master_playlist", size_bytes=512, sha256="0" * 64)
THUMBNAIL = ThumbnailResult(
    artifact=ArtifactFile(path="thumbnail.jpg", kind="thumbnail", size_bytes=40_000, sha256="a" * 64),
    diagnostic=None,
)
NO_THUMBNAIL = ThumbnailResult(
    artifact=None, diagnostic=Diagnostic(code="thumbnail_unavailable", detail="the thumbnail encoder wrote no image")
)
# Contract v2's worst case: 7 renditions x (a playlist and 3,600 segments), the master, the thumbnail.
MAX_SEGMENTS = 3600
MAX_ARTIFACTS = 25_209
MAX_DURATION_MS = 21_600_000
MAX_EDGE = 3840
MAX_BANDWIDTH = 999_999_999
WIDEST_CODECS = "c" * 32 + "," + "c" * 31  # the contract's 64 characters
FOUR_GIB = 4 * GIB
# 7 x 4 GiB + 25,193 x 42,000,000 bytes stays below the contract's 1 TiB in all.
WORST_SEGMENT_BYTES = 42_000_000
WORST_DIAGNOSTIC = Diagnostic(code="d" * 48, detail='"' * 200)  # every character escaped in JSON
CONTRACT_MAX_DIAGNOSTICS = 8
THUMBNAIL_UNAVAILABLE = "thumbnail_unavailable"
JOB_BYTES = 10 * GIB
JOB_ARTIFACTS = 500
FILE_MODE_BITS = 0o777


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _rendition(
    name: str,
    width: int,
    height: int,
    bandwidth: tuple[int, int] = (2_800_000, 2_400_000),
    *,
    segments: int = 2,
    segment_bytes: Callable[[int], int] = lambda number: SEGMENT_BYTES,
    has_audio: bool = True,
    encrypted: bool = False,
    frame_rate: tuple[int, int] = PAL_RATE,
    codecs: str | None = None,
    duration_ms: int = TWELVE_SECONDS_MS,
) -> RenditionOutput:
    playlist = ArtifactFile(
        path=f"{name}/playlist.m3u8", kind="hls_media_playlist", size_bytes=PLAYLIST_BYTES, sha256=_digest(name)
    )
    segment_files = tuple(
        ArtifactFile(
            path=f"{name}/segment_{number:04d}.ts",
            kind="hls_segment",
            size_bytes=segment_bytes(number),
            sha256=_digest(f"{name}/{number}"),
        )
        for number in range(segments)
    )
    artifacts = (playlist, *segment_files)
    return RenditionOutput(
        name=name,
        width=width,
        height=height,
        frame_rate_num=frame_rate[0],
        frame_rate_den=frame_rate[1],
        codecs=codecs or (AUDIO_CODECS if has_audio else VIDEO_CODECS),
        has_audio=has_audio,
        encrypted=encrypted,
        playlist_path=f"{name}/playlist.m3u8",
        segment_count=segments,
        duration_ms=duration_ms,
        bandwidth_bps=bandwidth[0],
        average_bandwidth_bps=bandwidth[1],
        total_bytes=sum(artifact.size_bytes for artifact in artifacts),
        artifacts=artifacts,
    )


def _renditions(
    rungs: tuple[tuple[str, int, int, int, int], ...] = THREE_RUNGS, **options: Any
) -> list[RenditionOutput]:
    return [_rendition(name, width, height, (peak, average), **options) for name, width, height, peak, average in rungs]


def _with_artifacts(output: RenditionOutput, artifacts: tuple[ArtifactFile, ...]) -> RenditionOutput:
    return dataclasses.replace(output, artifacts=artifacts)


def _manifest_arguments(**overrides: Any) -> dict[str, Any]:
    """An encrypted job of three renditions with a thumbnail, as the stages would hand it over."""
    arguments: dict[str, Any] = {
        "identity": IDENTITY,
        "profile_ref": PROFILE_REF,
        "profile": PILOT_PROFILE,
        "encryption": ENCRYPTED,
        "requested_renditions": [name for name, *_ in THREE_RUNGS],
        "renditions": _renditions(encrypted=True),
        "master": MASTER,
        "thumbnail": THUMBNAIL,
        "extra_diagnostics": (),
        "created_at": CREATED_AT,
    }
    arguments.update(overrides)
    return arguments


def _context(arguments: dict[str, Any], *, max_artifact_count: int = MAX_ARTIFACTS) -> TrustedContext:
    """The worker's self-check context, built from the same inputs the manifest was."""
    identity: ManifestIdentity = arguments["identity"]
    profile_ref: ProfileRef = arguments["profile_ref"]
    encryption: EncryptionRef = arguments["encryption"]
    expected_encryption: dict[str, str] = {"mode": encryption.mode}
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
                "max_artifact_count": max_artifact_count,
            },
        }
    )


def _assert_failure(raised: pytest.ExceptionInfo[WorkerFailure], expected: tuple[str, str, bool]) -> None:
    failure = raised.value.failure
    assert (failure.error_class, failure.code, failure.retryable) == expected


def _output_root(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    return root


def _package_os(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Replace what core.package sees of `os`, leaving the real module alone for everything else."""
    view = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if not name.startswith("__")})
    for name, replacement in overrides.items():
        setattr(view, name, replacement)
    monkeypatch.setattr(core.package, "os", view)


def _failing(error_number: int) -> Callable[..., Any]:
    def fail(*args: object, **kwargs: object) -> Any:
        raise OSError(error_number, os.strerror(error_number))

    return fail


def _closing_then_failing(error_number: int) -> Callable[[int], None]:
    """A close that releases the descriptor, then reports an error (as NFS may on a full disk)."""

    def close(descriptor: int) -> None:
        os.close(descriptor)
        raise OSError(error_number, os.strerror(error_number))

    return close


# --- the master playlist ---------------------------------------------------------------------------


def test_master_playlist_bytes_when_three_renditions_carry_audio_should_list_them_in_order_with_measured_rates(
) -> None:
    # Arrange
    renditions = _renditions()

    # Act
    data = master_playlist_bytes(renditions)

    # Assert
    assert data == (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        "#EXT-X-INDEPENDENT-SEGMENTS\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=2800000,AVERAGE-BANDWIDTH=2400000,RESOLUTION=1280x720,FRAME-RATE=25.000,"
        'CODECS="avc1.64001f,mp4a.40.2"\n'
        "720p/playlist.m3u8\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=1400000,AVERAGE-BANDWIDTH=1200000,RESOLUTION=854x480,FRAME-RATE=25.000,"
        'CODECS="avc1.64001f,mp4a.40.2"\n'
        "480p/playlist.m3u8\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=800000,AVERAGE-BANDWIDTH=700000,RESOLUTION=640x360,FRAME-RATE=25.000,"
        'CODECS="avc1.64001f,mp4a.40.2"\n'
        "360p/playlist.m3u8\n"
    ).encode("ascii")


def test_master_playlist_bytes_when_the_source_is_silent_should_name_only_the_video_codec() -> None:
    # Arrange
    renditions = [_rendition("240p", 426, 240, (300_000, 250_000), has_audio=False)]

    # Act
    data = master_playlist_bytes(renditions)

    # Assert
    assert data == (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        "#EXT-X-INDEPENDENT-SEGMENTS\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=300000,AVERAGE-BANDWIDTH=250000,RESOLUTION=426x240,FRAME-RATE=25.000,"
        'CODECS="avc1.64001f"\n'
        "240p/playlist.m3u8\n"
    ).encode("ascii")


@pytest.mark.parametrize(
    ("frame_rate", "expected"),
    [
        (NTSC_RATE, "29.970"),
        ((24000, 1001), "23.976"),
        ((60, 1), "60.000"),
        ((1, 1), "1.000"),
        ((2001, 2000), "1.001"),
        ((1999, 2000), "1.000"),
    ],
    ids=["NTSC", "film NTSC", "60 fps", "the 1 fps floor", "a half rounds up", "a half rounds up into the units"],
)
def test_master_playlist_bytes_when_the_frame_rate_is_fractional_should_round_it_half_up_to_three_decimals(
    frame_rate: tuple[int, int], expected: str
) -> None:
    # Arrange
    renditions = [_rendition("240p", 426, 240, frame_rate=frame_rate)]

    # Act
    text = master_playlist_bytes(renditions).decode("ascii")

    # Assert
    assert f",FRAME-RATE={expected}," in text


def _replace_first(field: str, value: Any) -> Callable[[], list[RenditionOutput]]:
    def build() -> list[RenditionOutput]:
        renditions = _renditions()
        renditions[0] = dataclasses.replace(renditions[0], **{field: value})
        return renditions

    return build


@pytest.mark.parametrize(
    "build",
    [
        lambda: [],
        lambda: _renditions() * 3,
        lambda: [_renditions()[0], _renditions()[0]],
        lambda: [_rendition("4320p", 7680, 4320)],
        _replace_first("playlist_path", "720p/index.m3u8"),
        _replace_first("codecs", "avc1.64001f,mp4a.40.2,opus"),
        _replace_first("codecs", "avc1 64001f"),
        _replace_first("codecs", ""),
        _replace_first("codecs", "c" * 32 + "," + "c" * 32),
        _replace_first("has_audio", False),
        _replace_first("encrypted", True),
        _replace_first("bandwidth_bps", 0),
        lambda: "720p",
    ],
    ids=[
        "no rendition",
        "nine renditions",
        "a name twice",
        "a name the contract does not know",
        "a playlist path other than its name's",
        "three codecs",
        "codecs with a space",
        "no codecs",
        "codecs of 65 characters",
        "renditions disagreeing on audio",
        "renditions disagreeing on encryption",
        "a bandwidth of zero",
        "a string instead of renditions",
    ],
)
def test_master_playlist_bytes_when_a_precondition_is_broken_should_raise(
    build: Callable[[], Any],
) -> None:
    # Arrange
    renditions = build()

    # Act / Assert
    with pytest.raises(ValueError):
        master_playlist_bytes(renditions)


def test_master_playlist_bytes_when_a_rendition_is_not_a_rendition_output_should_raise_type_error() -> None:
    # Arrange
    renditions: list[Any] = [*_renditions()[:2], {"name": "360p"}]

    # Act / Assert
    with pytest.raises(TypeError):
        master_playlist_bytes(renditions)


def _widest_renditions() -> list[RenditionOutput]:
    return [
        _rendition(
            name, MAX_EDGE, MAX_EDGE, (MAX_BANDWIDTH, MAX_BANDWIDTH), frame_rate=(60000, 1001), codecs=WIDEST_CODECS
        )
        for name in RENDITION_NAMES
    ]


def test_master_playlist_bytes_when_seven_renditions_take_every_field_at_its_widest_should_fit_its_bound() -> None:
    # Arrange
    renditions = _widest_renditions()

    # Act
    data = master_playlist_bytes(renditions)

    # Assert
    assert data.count(b"#EXT-X-STREAM-INF:") == len(RENDITION_NAMES)
    assert len(data) <= MASTER_PLAYLIST_MAX_BYTES


def test_master_playlist_bytes_when_the_text_outgrows_its_bound_should_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: no valid input reaches 4 KiB, so the bound is lowered to prove the check
    monkeypatch.setattr(core.package, "MASTER_PLAYLIST_MAX_BYTES", len(master_playlist_bytes(_renditions())) - 1)

    # Act / Assert
    with pytest.raises(ValueError):
        master_playlist_bytes(_renditions())


def test_write_master_playlist_when_written_should_create_the_file_its_artifact_describes(tmp_path: Path) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    renditions = _renditions()

    # Act
    artifact = write_master_playlist(renditions, output_root=output_root)

    # Assert
    on_disk = (output_root / "master.m3u8").read_bytes()
    assert on_disk == master_playlist_bytes(renditions)
    assert (artifact.path, artifact.kind) == ("master.m3u8", "hls_master_playlist")
    assert (artifact.size_bytes, artifact.sha256) == (len(on_disk), hashlib.sha256(on_disk).hexdigest())
    assert os.listdir(output_root) == ["master.m3u8"]


@pytest.mark.parametrize("occupant", ["file", "dangling link"])
def test_write_master_playlist_when_the_name_is_taken_should_raise_and_leave_it_alone(
    occupant: str, tmp_path: Path
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    taken = output_root / "master.m3u8"
    link_target = tmp_path / "elsewhere.m3u8"
    if occupant == "file":
        taken.write_bytes(b"kept")
    else:
        taken.symlink_to(link_target)

    # Act
    with pytest.raises(ValueError):
        write_master_playlist(_renditions(), output_root=output_root)

    # Assert
    if occupant == "file":
        assert taken.read_bytes() == b"kept"
    else:
        assert taken.is_symlink()
        assert not link_target.exists()


@pytest.mark.parametrize("output_root", [Path("relative/output"), Path("/no/such/directory"), Path("/tmp/a:b")])
def test_write_master_playlist_when_the_output_root_is_unusable_should_raise(output_root: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        write_master_playlist(_renditions(), output_root=output_root)


@pytest.mark.parametrize(
    ("failing_call", "error_number", "expected"),
    [
        ("write", errno.ENOSPC, ("resource_exhausted", "scratch_full", True)),
        ("write", errno.EDQUOT, ("resource_exhausted", "scratch_full", True)),
        ("open", errno.ENOSPC, ("resource_exhausted", "scratch_full", True)),
        ("write", errno.EIO, ("output_write_failed", "package_write_failed", True)),
        ("close", errno.EIO, ("output_write_failed", "package_write_failed", True)),
        ("open", errno.EACCES, ("output_write_failed", "package_write_failed", True)),
    ],
    ids=[
        "the disk fills while writing",
        "the quota runs out while writing",
        "the disk is full when the file is created",
        "a write fails",
        "the close fails",
        "the file cannot be created",
    ],
)
def test_write_master_playlist_when_the_write_fails_should_report_it_typed_and_leave_no_file(
    failing_call: str,
    error_number: int,
    expected: tuple[str, str, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    failing = _closing_then_failing(error_number) if failing_call == "close" else _failing(error_number)
    _package_os(monkeypatch, **{failing_call: failing})

    # Act
    with pytest.raises(WorkerFailure) as raised:
        write_master_playlist(_renditions(), output_root=output_root)

    # Assert
    _assert_failure(raised, expected)
    if expected[1] == "package_write_failed":
        assert errno.errorcode[error_number] in raised.value.failure.detail
    assert str(output_root) not in raised.value.failure.detail
    assert os.listdir(output_root) == []


def test_write_master_playlist_when_the_system_writes_a_few_bytes_at_a_time_should_still_write_them_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    _package_os(monkeypatch, write=lambda descriptor, data: os.write(descriptor, bytes(data[:7])))

    # Act
    artifact = write_master_playlist(_renditions(), output_root=output_root)

    # Assert
    on_disk = (output_root / "master.m3u8").read_bytes()
    assert on_disk == master_playlist_bytes(_renditions())
    assert artifact.size_bytes == len(on_disk)


def test_write_master_playlist_when_a_write_makes_no_progress_should_fail_instead_of_spinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    _package_os(monkeypatch, write=lambda descriptor, data: 0)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        write_master_playlist(_renditions(), output_root=output_root)

    # Assert
    _assert_failure(raised, ("output_write_failed", "package_write_failed", True))
    assert os.listdir(output_root) == []


# --- the generation manifest -----------------------------------------------------------------------


def _accepted(raw: bytes, arguments: dict[str, Any], **context: Any) -> None:
    verdict = validate(raw, _context(arguments, **context))
    assert verdict.accepted, verdict


def test_manifest_bytes_when_an_encrypted_job_has_a_thumbnail_should_pass_the_contract_validator() -> None:
    # Arrange
    arguments = _manifest_arguments()

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    document = json.loads(raw)
    assert document["encryption"] == {"mode": "aes-128", "media_key_id": MEDIA_KEY_ID}
    assert document["thumbnail_path"] == "thumbnail.jpg"
    assert "diagnostics" not in document
    assert [artifact["path"] for artifact in document["artifacts"]][:3] == [
        "master.m3u8", "thumbnail.jpg", "720p/playlist.m3u8",
    ]


def test_manifest_bytes_when_a_plain_job_has_no_thumbnail_should_carry_its_diagnostic_and_pass_the_validator() -> None:
    # Arrange
    arguments = _manifest_arguments(
        encryption=PLAIN,
        renditions=_renditions(has_audio=False),
        thumbnail=NO_THUMBNAIL,
        extra_diagnostics=(Diagnostic(code="audio_dropped", detail="a later audio stream was not kept"),),
    )

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    document = json.loads(raw)
    assert document["encryption"] == {"mode": "none"}
    assert document["thumbnail_path"] is None
    assert document["media"]["has_audio"] is False
    assert [diagnostic["code"] for diagnostic in document["diagnostics"]] == [THUMBNAIL_UNAVAILABLE, "audio_dropped"]
    assert all(artifact["kind"] != "thumbnail" for artifact in document["artifacts"])


def test_manifest_bytes_when_the_job_has_one_rendition_should_pass_the_contract_validator() -> None:
    # Arrange
    rendition = _rendition("240p", 426, 240, (300_000, 250_000), segments=1, encrypted=True)
    arguments = _manifest_arguments(requested_renditions=["720p", "240p"], renditions=[rendition])

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    assert json.loads(raw)["media"] == {
        "duration_ms": TWELVE_SECONDS_MS, "max_width": 426, "max_height": 240, "has_audio": True, "rendition_count": 1,
    }


def test_manifest_bytes_when_the_source_is_portrait_should_report_its_geometry_and_pass_the_validator() -> None:
    # Arrange
    arguments = _manifest_arguments(
        requested_renditions=["720p", "360p"], renditions=_renditions(PORTRAIT_RUNGS, encrypted=True)
    )

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    document = json.loads(raw)
    assert (document["media"]["max_width"], document["media"]["max_height"]) == (720, 1280)
    assert [(entry["width"], entry["height"]) for entry in document["renditions"]] == [(720, 1280), (360, 640)]


def _worst_case_arguments() -> dict[str, Any]:
    profile = dataclasses.replace(PILOT_PROFILE, profile_id="p" * 32, version=65_535)
    renditions = [
        _rendition(
            name,
            MAX_EDGE,
            MAX_EDGE,
            (MAX_BANDWIDTH, MAX_BANDWIDTH),
            segments=MAX_SEGMENTS,
            segment_bytes=lambda number: FOUR_GIB if number == 0 else WORST_SEGMENT_BYTES,
            encrypted=True,
            frame_rate=(60000, 1001),
            codecs=WIDEST_CODECS,
            duration_ms=MAX_DURATION_MS,
        )
        for name in RENDITION_NAMES
    ]
    return _manifest_arguments(
        profile=profile,
        profile_ref=ProfileRef(id=profile.profile_id, version=profile.version, sha256="f" * 64),
        requested_renditions=list(RENDITION_NAMES),
        renditions=renditions,
        master=dataclasses.replace(MASTER, size_bytes=MASTER_PLAYLIST_MAX_BYTES),
        thumbnail=ThumbnailResult(
            artifact=dataclasses.replace(THUMBNAIL.artifact, size_bytes=PILOT_PROFILE.encode.thumbnail_max_bytes),
            diagnostic=None,
        ),
        extra_diagnostics=(WORST_DIAGNOSTIC,) * CONTRACT_MAX_DIAGNOSTICS,
    )


def _compact(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("ascii")


def test_manifest_bytes_when_the_job_reaches_every_contract_maximum_should_pass_the_validator_within_its_reserve(
) -> None:
    # Arrange
    arguments = _worst_case_arguments()

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    document = json.loads(raw)
    entry_sizes = [len(_compact(entry)) + len(b",") for entry in document["artifacts"]]
    assert document["artifact_count"] == len(document["artifacts"]) == MAX_ARTIFACTS
    assert document["total_bytes"] <= core.package.CONTRACT_MAX_TOTAL_BYTES
    assert len(document["diagnostics"]) == CONTRACT_MAX_DIAGNOSTICS
    assert max(entry_sizes) <= MANIFEST_ARTIFACT_MAX_BYTES
    assert len(raw) - sum(entry_sizes) <= MANIFEST_HEADER_MAX_BYTES
    assert len(raw) <= manifest_max_bytes(MAX_ARTIFACTS)


def test_manifest_bytes_when_a_rendition_runs_one_frame_past_six_hours_should_cap_the_duration_at_the_contracts(
) -> None:
    # Arrange
    renditions = _renditions(encrypted=True, duration_ms=MAX_DURATION_MS + 40)
    arguments = _manifest_arguments(renditions=renditions)

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    _accepted(raw, arguments)
    assert json.loads(raw)["media"]["duration_ms"] == MAX_DURATION_MS


def test_manifest_bytes_when_the_renditions_last_differently_should_report_the_longest() -> None:
    # Arrange
    renditions = _renditions(encrypted=True)
    renditions[2] = dataclasses.replace(renditions[2], duration_ms=TWELVE_SECONDS_MS + 40)

    # Act
    raw = manifest_bytes(**_manifest_arguments(renditions=renditions))

    # Assert
    assert json.loads(raw)["media"]["duration_ms"] == TWELVE_SECONDS_MS + 40


def test_manifest_bytes_when_serialized_should_be_compact_in_the_contracts_key_order_without_a_final_newline() -> None:
    # Arrange
    arguments = _manifest_arguments(thumbnail=NO_THUMBNAIL)

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    document = json.loads(raw, object_pairs_hook=lambda pairs: pairs)
    keys = [key for key, _ in document]
    values = dict(document)
    assert keys == [
        "manifest_version", "document_kind", "generation_id", "created_at", "identity", "profile", "encryption",
        "media", "renditions", "master_playlist_path", "thumbnail_path", "artifact_count", "total_bytes",
        "artifacts", "diagnostics",
    ]
    assert [key for key, _ in values["identity"]] == [
        "org_uuid", "video_uuid", "source_id", "attempt_id", "dispatch_id", "execution_id",
    ]
    assert [key for key, _ in values["artifacts"][1]] == ["path", "kind", "size_bytes", "sha256", "rendition"]
    assert raw == _compact(json.loads(raw))
    assert not raw.endswith(b"\n")
    assert values["generation_id"] == EXECUTION_ID


def test_manifest_bytes_when_created_at_is_in_another_zone_should_write_utc_with_truncated_milliseconds() -> None:
    # Arrange
    created_at = datetime(2030, 1, 1, 6, 59, 59, 999_999, tzinfo=timezone(timedelta(hours=7)))

    # Act
    raw = manifest_bytes(**_manifest_arguments(created_at=created_at))

    # Assert
    assert json.loads(raw)["created_at"] == "2029-12-31T23:59:59.999Z"


def _profile_with_two_segments() -> dict[str, Any]:
    profile = fast_profile(max_segments_per_rendition=2)
    renditions = _renditions(encrypted=True)
    renditions[1] = _rendition("480p", 854, 480, segments=3, encrypted=True)
    return {"profile": profile, "renditions": renditions}


def _first_rendition_artifacts(build: Callable[[RenditionOutput], tuple[ArtifactFile, ...]]) -> dict[str, Any]:
    renditions = _renditions(encrypted=True)
    renditions[0] = _with_artifacts(renditions[0], build(renditions[0]))
    return {"renditions": renditions}


def _foreign_segment(output: RenditionOutput) -> tuple[ArtifactFile, ...]:
    stray = dataclasses.replace(output.artifacts[1], path="elsewhere/segment_0000.ts")
    return (output.artifacts[0], stray)


def _beyond_one_tib() -> dict[str, Any]:
    big = _rendition("720p", 1280, 720, segments=257, segment_bytes=lambda number: FOUR_GIB, encrypted=True)
    return {"requested_renditions": ["720p"], "renditions": [big]}


@pytest.mark.parametrize(
    "overrides",
    [
        lambda: {"profile_ref": dataclasses.replace(PROFILE_REF, id="another-profile")},
        lambda: {"profile_ref": dataclasses.replace(PROFILE_REF, version=PILOT_PROFILE.version + 1)},
        lambda: {"renditions": []},
        lambda: {"renditions": _renditions(encrypted=True) * 3},
        lambda: {"renditions": [_renditions(encrypted=True)[0]] * 2},
        lambda: {"requested_renditions": ["720p", "480p"]},
        lambda: {"requested_renditions": "720p"},
        lambda: {"renditions": _renditions(encrypted=False)},
        lambda: {"encryption": PLAIN},
        lambda: {"master": dataclasses.replace(MASTER, kind="hls_media_playlist")},
        lambda: {"master": dataclasses.replace(MASTER, path="index.m3u8")},
        lambda: {"extra_diagnostics": (Diagnostic(code=THUMBNAIL_UNAVAILABLE, detail="said twice"),)},
        lambda: {"thumbnail": NO_THUMBNAIL, "extra_diagnostics": (WORST_DIAGNOSTIC,) * CONTRACT_MAX_DIAGNOSTICS},
        lambda: {"extra_diagnostics": (WORST_DIAGNOSTIC,) * (CONTRACT_MAX_DIAGNOSTICS + 1)},
        lambda: _first_rendition_artifacts(lambda output: (*output.artifacts, output.artifacts[1])),
        _profile_with_two_segments,
        lambda: _first_rendition_artifacts(lambda output: output.artifacts[:1]),
        lambda: _first_rendition_artifacts(lambda output: (output.artifacts[1], output.artifacts[0])),
        lambda: _first_rendition_artifacts(lambda output: (output.artifacts[0], output.artifacts[0])),
        lambda: _first_rendition_artifacts(_foreign_segment),
        _beyond_one_tib,
        lambda: {"created_at": datetime(2029, 12, 31, 23, 58)},
    ],
    ids=[
        "a profile reference naming another profile",
        "a profile reference naming another version",
        "no rendition",
        "nine renditions",
        "a rendition twice",
        "a rendition that was not requested",
        "a string instead of the requested names",
        "plain renditions in an encrypted job",
        "encrypted renditions in a plain job",
        "a master of another kind",
        "a master at another path",
        "a thumbnail_unavailable diagnostic not from the thumbnail",
        "nine diagnostics with the thumbnail's",
        "nine diagnostics",
        "two artifacts at one path",
        "more segments than the profile allows",
        "a rendition without a segment",
        "a segment before the playlist",
        "a second playlist",
        "a file outside the rendition's directory",
        "more than 1 TiB in all",
        "a naive created_at",
    ],
)
def test_manifest_bytes_when_a_precondition_is_broken_should_raise(overrides: Callable[[], dict[str, Any]]) -> None:
    # Arrange
    arguments = _manifest_arguments(**overrides())

    # Act / Assert
    with pytest.raises(ValueError):
        manifest_bytes(**arguments)


@pytest.mark.parametrize(
    "field", ["identity", "profile_ref", "profile", "encryption", "master", "thumbnail", "created_at"]
)
def test_manifest_bytes_when_an_argument_has_the_wrong_type_should_raise_type_error(field: str) -> None:
    # Arrange
    arguments = _manifest_arguments(**{field: "not the right type"})

    # Act / Assert
    with pytest.raises(TypeError):
        manifest_bytes(**arguments)


@pytest.mark.parametrize(
    ("constant", "value"),
    [("CONTRACT_MAX_ARTIFACTS", 10), ("CONTRACT_MANIFEST_MAX_BYTES", 2048)],
    ids=["more artifacts than the contract lists", "a document longer than the contract's limit"],
)
def test_manifest_bytes_when_a_contract_limit_is_passed_should_raise(
    constant: str, value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: no valid input reaches these limits (the segment limit and the reserve keep below
    # them), so each is lowered to prove its check
    monkeypatch.setattr(core.package, constant, value)

    # Act / Assert
    with pytest.raises(ValueError):
        manifest_bytes(**_manifest_arguments())


@pytest.mark.parametrize("alias", ["attempt_id", "dispatch_id", "execution_id", "source_id", "org_uuid"])
def test_manifest_identity_when_an_id_equals_the_video_uuid_should_raise(alias: str) -> None:
    # Arrange
    fields = dataclasses.asdict(IDENTITY)
    fields[alias] = VIDEO_UUID

    # Act / Assert
    with pytest.raises(ValueError):
        ManifestIdentity(**fields)


@pytest.mark.parametrize("value", [LETTERED_UUID.upper(), ORG_UUID[:-1], ORG_UUID.replace("-", "_"), None])
def test_manifest_identity_when_an_id_is_not_a_lowercase_uuid_should_raise(value: Any) -> None:
    # Arrange
    fields = dataclasses.asdict(IDENTITY)
    fields["org_uuid"] = value

    # Act / Assert
    with pytest.raises(ValueError):
        ManifestIdentity(**fields)


@pytest.mark.parametrize(
    ("mode", "media_key_id"),
    [("aes-128", None), ("none", MEDIA_KEY_ID), ("aes-256", MEDIA_KEY_ID), ("aes-128", LETTERED_UUID.upper())],
    ids=["aes-128 without a key id", "none with a key id", "an unknown mode", "an uppercase key id"],
)
def test_encryption_ref_when_mode_and_key_id_do_not_agree_should_raise(mode: str, media_key_id: str | None) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        EncryptionRef(mode=mode, media_key_id=media_key_id)


@pytest.mark.parametrize(
    "overrides",
    [{"id": "P"}, {"id": "x" * 33}, {"version": 0}, {"version": 65_536}, {"version": True}, {"sha256": "AB" * 32}],
)
def test_profile_ref_when_a_field_breaks_the_contract_should_raise(overrides: dict[str, Any]) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        dataclasses.replace(PROFILE_REF, **overrides)


def test_write_manifest_when_written_should_create_the_file_its_output_describes(tmp_path: Path) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    arguments = _manifest_arguments()

    # Act
    output = write_manifest(**arguments, output_root=output_root, max_bytes=manifest_max_bytes(JOB_ARTIFACTS))

    # Assert
    on_disk = (output_root / "generation-manifest.json").read_bytes()
    document = json.loads(on_disk)
    assert on_disk == manifest_bytes(**arguments)
    assert output == ManifestOutput(
        path="generation-manifest.json",
        sha256=hashlib.sha256(on_disk).hexdigest(),
        size_bytes=len(on_disk),
        artifact_count=document["artifact_count"],
        total_bytes=document["total_bytes"],
    )
    assert document["artifact_count"] == 2 + sum(len(rendition.artifacts) for rendition in arguments["renditions"])
    assert os.listdir(output_root) == ["generation-manifest.json"]


def test_write_manifest_when_the_manifest_exists_should_raise_and_leave_it_alone(tmp_path: Path) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    (output_root / "generation-manifest.json").write_bytes(b"kept")

    # Act
    with pytest.raises(ValueError):
        write_manifest(**_manifest_arguments(), output_root=output_root, max_bytes=manifest_max_bytes(JOB_ARTIFACTS))

    # Assert
    assert (output_root / "generation-manifest.json").read_bytes() == b"kept"


def test_write_manifest_when_the_document_outgrows_its_reserved_room_should_fail_and_write_nothing(
    tmp_path: Path,
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    arguments = _manifest_arguments()
    reserved = len(manifest_bytes(**arguments)) - 1

    # Act
    with pytest.raises(WorkerFailure) as raised:
        write_manifest(**arguments, output_root=output_root, max_bytes=reserved)

    # Assert
    _assert_failure(raised, ("internal_error", "manifest_too_large", False))
    assert os.listdir(output_root) == []


def test_write_manifest_when_the_document_fills_its_reserved_room_exactly_should_write_it(tmp_path: Path) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    arguments = _manifest_arguments()
    reserved = len(manifest_bytes(**arguments))

    # Act
    output = write_manifest(**arguments, output_root=output_root, max_bytes=reserved)

    # Assert
    assert output.size_bytes == reserved


def test_write_manifest_when_the_disk_fills_should_report_scratch_full_and_leave_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    _package_os(monkeypatch, write=_failing(errno.ENOSPC))

    # Act
    with pytest.raises(WorkerFailure) as raised:
        write_manifest(**_manifest_arguments(), output_root=output_root, max_bytes=manifest_max_bytes(JOB_ARTIFACTS))

    # Assert
    _assert_failure(raised, ("resource_exhausted", "scratch_full", True))
    assert os.listdir(output_root) == []


def test_write_manifest_and_master_when_written_should_create_files_readable_by_others(tmp_path: Path) -> None:
    # Arrange
    output_root = _output_root(tmp_path)
    previous_umask = os.umask(0o022)

    # Act
    try:
        write_master_playlist(_renditions(encrypted=True), output_root=output_root)
        write_manifest(**_manifest_arguments(), output_root=output_root, max_bytes=manifest_max_bytes(JOB_ARTIFACTS))
    finally:
        os.umask(previous_umask)

    # Assert
    for name in ("master.m3u8", "generation-manifest.json"):
        assert (output_root / name).stat().st_mode & FILE_MODE_BITS == 0o644


# --- the package's room in the job's allowance -----------------------------------------------------


def test_reserve_package_when_the_job_has_room_should_give_the_renditions_the_rest() -> None:
    # Arrange
    job = OutputAllowance(max_bytes=JOB_BYTES, max_artifacts=JOB_ARTIFACTS)
    manifest_room = 16 * 1024 + 256 * JOB_ARTIFACTS
    package_room = 4096 + PILOT_PROFILE.encode.thumbnail_max_bytes + manifest_room

    # Act
    renditions = reserve_package(job, PILOT_PROFILE)

    # Assert
    assert renditions == OutputAllowance(max_bytes=JOB_BYTES - package_room, max_artifacts=JOB_ARTIFACTS - 2)


@pytest.mark.parametrize(
    ("max_bytes_short", "max_artifacts", "expected_code"),
    [(1, JOB_ARTIFACTS, "output_too_large"), (0, 1, "too_many_artifacts")],
    ids=["one byte short of the package", "room for one file"],
)
def test_reserve_package_when_the_job_cannot_hold_the_package_should_fail_typed(
    max_bytes_short: int, max_artifacts: int, expected_code: str
) -> None:
    # Arrange
    package_room = 4096 + PILOT_PROFILE.encode.thumbnail_max_bytes + manifest_max_bytes(max_artifacts)
    job = OutputAllowance(max_bytes=package_room - max_bytes_short, max_artifacts=max_artifacts)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        reserve_package(job, PILOT_PROFILE)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", expected_code, False))


def test_reserve_package_when_the_job_holds_exactly_the_package_should_leave_the_renditions_nothing() -> None:
    # Arrange
    package_room = 4096 + PILOT_PROFILE.encode.thumbnail_max_bytes + manifest_max_bytes(2)
    job = OutputAllowance(max_bytes=package_room, max_artifacts=2)

    # Act
    renditions = reserve_package(job, PILOT_PROFILE)

    # Assert
    assert renditions == OutputAllowance(max_bytes=0, max_artifacts=0)


@pytest.mark.parametrize(
    ("max_artifacts", "expected"),
    [(0, 16 * 1024), (MAX_ARTIFACTS, 16 * 1024 + 256 * MAX_ARTIFACTS), (1_000_000, 8 * MIB)],
    ids=["no artifact", "the contract's most", "capped at the contract's 8 MiB"],
)
def test_manifest_max_bytes_when_given_an_artifact_allowance_should_bound_the_manifest(
    max_artifacts: int, expected: int
) -> None:
    # Arrange / Act
    bound = manifest_max_bytes(max_artifacts)

    # Assert
    assert bound == expected


def test_output_allowance_reserve_when_the_room_fits_exactly_should_leave_zero() -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=1000, max_artifacts=3)

    # Act
    remaining = allowance.reserve(1000, 3)

    # Assert
    assert remaining == OutputAllowance(max_bytes=0, max_artifacts=0)


@pytest.mark.parametrize(
    ("max_bytes", "max_artifacts", "expected_code"),
    [(1001, 3, "output_too_large"), (1000, 4, "too_many_artifacts")],
    ids=["one byte over", "one file over"],
)
def test_output_allowance_reserve_when_the_room_does_not_fit_should_fail_typed(
    max_bytes: int, max_artifacts: int, expected_code: str
) -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=1000, max_artifacts=3)

    # Act
    with pytest.raises(WorkerFailure) as raised:
        allowance.reserve(max_bytes, max_artifacts)

    # Assert
    _assert_failure(raised, ("input_limits_exceeded", expected_code, False))


@pytest.mark.parametrize(("max_bytes", "max_artifacts"), [(-1, 0), (0, -1), (True, 0), (0, 1.0)])
def test_output_allowance_reserve_when_an_argument_is_not_an_int_of_at_least_zero_should_raise(
    max_bytes: Any, max_artifacts: Any
) -> None:
    # Arrange
    allowance = OutputAllowance(max_bytes=1000, max_artifacts=3)

    # Act / Assert
    with pytest.raises(ValueError):
        allowance.reserve(max_bytes, max_artifacts)


def test_reserve_package_when_the_profile_is_missing_should_raise_type_error() -> None:
    # Arrange
    job = OutputAllowance(max_bytes=JOB_BYTES, max_artifacts=JOB_ARTIFACTS)

    # Act / Assert
    with pytest.raises(TypeError):
        reserve_package(job, None)  # type: ignore[arg-type]


def test_reserve_package_when_the_profile_allows_a_larger_thumbnail_should_set_aside_that_much_more() -> None:
    # Arrange
    profiles: list[MediaProfile] = [PILOT_PROFILE, fast_profile(thumbnail_max_bytes=2 * MIB)]

    # Act
    rooms = [reserve_package(OutputAllowance(max_bytes=JOB_BYTES, max_artifacts=10), p).max_bytes for p in profiles]

    # Assert: only the thumbnail's room follows the profile
    assert rooms[0] - rooms[1] == MIB


# --- the manifest's values, pinned by a literal document --------------------------------------------

LITERAL_IDENTITY = ManifestIdentity(
    org_uuid="a1a1a1a1-0000-7000-8000-000000000001",
    video_uuid="b2b2b2b2-0000-7000-8000-000000000002",
    source_id="c3c3c3c3-0000-7000-8000-000000000003",
    attempt_id="d4d4d4d4-0000-7000-8000-000000000004",
    dispatch_id="e5e5e5e5-0000-7000-8000-000000000005",
    execution_id="f6f6f6f6-0000-7000-8000-000000000006",
)
LITERAL_MEDIA_KEY_ID = "0a0a0a0a-0000-7000-8000-00000000000a"


def _literal_rendition(
    name: str,
    size: tuple[int, int],
    codecs: str,
    duration_ms: int,
    bandwidth: tuple[int, int],
    files: tuple[tuple[int, str], tuple[int, str], tuple[int, str]],
) -> RenditionOutput:
    """A rendition of a playlist and two segments, each file given as (size_bytes, sha256)."""
    names = ["playlist.m3u8", "segment_0000.ts", "segment_0001.ts"]
    kinds = ["hls_media_playlist", "hls_segment", "hls_segment"]
    artifacts = tuple(
        ArtifactFile(path=f"{name}/{file_name}", kind=kind, size_bytes=size_bytes, sha256=sha256)
        for file_name, kind, (size_bytes, sha256) in zip(names, kinds, files, strict=True)
    )
    return RenditionOutput(
        name=name, width=size[0], height=size[1], frame_rate_num=30000, frame_rate_den=1001, codecs=codecs,
        has_audio=True, encrypted=True, playlist_path=f"{name}/playlist.m3u8", segment_count=2,
        duration_ms=duration_ms, bandwidth_bps=bandwidth[0], average_bandwidth_bps=bandwidth[1],
        total_bytes=sum(size_bytes for size_bytes, _ in files), artifacts=artifacts,
    )


def _literal_arguments() -> dict[str, Any]:
    return {
        "identity": LITERAL_IDENTITY,
        "profile_ref": ProfileRef(id="pilot-h264-sdr", version=1, sha256="9" * 64),
        "profile": PILOT_PROFILE,
        "encryption": EncryptionRef(mode="aes-128", media_key_id=LITERAL_MEDIA_KEY_ID),
        "requested_renditions": ["720p", "480p", "360p"],
        "renditions": [
            _literal_rendition(
                "720p", (1280, 720), "avc1.64001f,mp4a.40.2", 12_345, (2_811_111, 2_422_222),
                ((401, "1" * 64), (1_500_001, "2" * 64), (1_200_002, "3" * 64)),
            ),
            _literal_rendition(
                "360p", (640, 360), "avc1.64001e,mp4a.40.2", 12_301, (833_333, 744_444),
                ((402, "4" * 64), (400_003, "5" * 64), (300_004, "6" * 64)),
            ),
        ],
        "master": ArtifactFile(path="master.m3u8", kind="hls_master_playlist", size_bytes=321, sha256="7" * 64),
        "thumbnail": ThumbnailResult(
            artifact=ArtifactFile(path="thumbnail.jpg", kind="thumbnail", size_bytes=40_005, sha256="8" * 64),
            diagnostic=None,
        ),
        "extra_diagnostics": (Diagnostic(code="audio_dropped", detail="a second audio stream was not kept"),),
        "created_at": datetime(2030, 1, 2, 3, 4, 5, 678_901, tzinfo=timezone.utc),
    }


LITERAL_DOCUMENT = {
    "manifest_version": "1.0.0-draft",
    "document_kind": "generation.manifest",
    "generation_id": "f6f6f6f6-0000-7000-8000-000000000006",
    "created_at": "2030-01-02T03:04:05.678Z",
    "identity": {
        "org_uuid": "a1a1a1a1-0000-7000-8000-000000000001",
        "video_uuid": "b2b2b2b2-0000-7000-8000-000000000002",
        "source_id": "c3c3c3c3-0000-7000-8000-000000000003",
        "attempt_id": "d4d4d4d4-0000-7000-8000-000000000004",
        "dispatch_id": "e5e5e5e5-0000-7000-8000-000000000005",
        "execution_id": "f6f6f6f6-0000-7000-8000-000000000006",
    },
    "profile": {"id": "pilot-h264-sdr", "version": 1, "sha256": "9" * 64},
    "encryption": {"mode": "aes-128", "media_key_id": "0a0a0a0a-0000-7000-8000-00000000000a"},
    "media": {"duration_ms": 12_345, "max_width": 1280, "max_height": 720, "has_audio": True, "rendition_count": 2},
    "renditions": [
        {
            "name": "720p",
            "width": 1280,
            "height": 720,
            "bandwidth_bps": 2_811_111,
            "average_bandwidth_bps": 2_422_222,
            "codecs": "avc1.64001f,mp4a.40.2",
            "playlist_path": "720p/playlist.m3u8",
        },
        {
            "name": "360p",
            "width": 640,
            "height": 360,
            "bandwidth_bps": 833_333,
            "average_bandwidth_bps": 744_444,
            "codecs": "avc1.64001e,mp4a.40.2",
            "playlist_path": "360p/playlist.m3u8",
        },
    ],
    "master_playlist_path": "master.m3u8",
    "thumbnail_path": "thumbnail.jpg",
    "artifact_count": 8,
    "total_bytes": 3_441_139,
    "artifacts": [
        {"path": "master.m3u8", "kind": "hls_master_playlist", "size_bytes": 321, "sha256": "7" * 64},
        {"path": "thumbnail.jpg", "kind": "thumbnail", "size_bytes": 40_005, "sha256": "8" * 64},
        {
            "path": "720p/playlist.m3u8", "kind": "hls_media_playlist", "size_bytes": 401, "sha256": "1" * 64,
            "rendition": "720p",
        },
        {
            "path": "720p/segment_0000.ts", "kind": "hls_segment", "size_bytes": 1_500_001, "sha256": "2" * 64,
            "rendition": "720p",
        },
        {
            "path": "720p/segment_0001.ts", "kind": "hls_segment", "size_bytes": 1_200_002, "sha256": "3" * 64,
            "rendition": "720p",
        },
        {
            "path": "360p/playlist.m3u8", "kind": "hls_media_playlist", "size_bytes": 402, "sha256": "4" * 64,
            "rendition": "360p",
        },
        {
            "path": "360p/segment_0000.ts", "kind": "hls_segment", "size_bytes": 400_003, "sha256": "5" * 64,
            "rendition": "360p",
        },
        {
            "path": "360p/segment_0001.ts", "kind": "hls_segment", "size_bytes": 300_004, "sha256": "6" * 64,
            "rendition": "360p",
        },
    ],
    "diagnostics": [{"code": "audio_dropped", "detail": "a second audio stream was not kept"}],
}


def test_manifest_bytes_when_given_a_small_job_should_write_exactly_the_document_its_inputs_describe() -> None:
    # Arrange: two renditions whose every value differs, a thumbnail, one diagnostic
    arguments = _literal_arguments()

    # Act
    raw = manifest_bytes(**arguments)

    # Assert
    assert json.loads(raw) == LITERAL_DOCUMENT
    assert raw == json.dumps(LITERAL_DOCUMENT, separators=(",", ":")).encode("ascii")
    _accepted(raw, arguments)
