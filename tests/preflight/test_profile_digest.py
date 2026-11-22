"""Which media profile a dispatch names: the profile's digest and the lookup by id, version and digest."""

from __future__ import annotations

import dataclasses
import re

import pytest

from core import profile as profile_module
from core.profile import (
    PILOT_PROFILE,
    QUALITY_PROFILES,
    SUPPORTED_PROFILES,
    MediaProfile,
    profile_digest,
    supported_profile,
)

# The digest of the pilot profile as it is today. A change of this value means the profile changed, and
# a changed profile is a new revision: bump PILOT_PROFILE's `version`, then pin the new digest here, and
# tell the Hub the new `{id, version, sha256}`: it dispatches the old one until then, which this worker
# refuses (`unsupported_profile`). Never update this value alone.
PILOT_PROFILE_SHA256 = "7886f49534ee439c96008ac1455447310c0599a03b3c8713a3497e236e063e2f"
# The quality levels' digests, under the same rule: a change of one is a new revision of that level.
QUALITY_PROFILE_SHA256 = {
    "pilot-h264-sdr-crf21": "2052c7d13a3af719ca977ea6c5e0047012e90eacc0d4467cf9bc3b8e76e37a9e",
    "pilot-h264-sdr-crf22": "c9e07ef2f7469078f86bbbf72c5b9b5d9c5e0c0c6e1259fdc3f75c714d09bbf8",
    "pilot-h264-sdr-crf23": "41d55d560d9a628a4167ec562d0267b6ea083629acf4b610951e3d94b378c272",
    "pilot-h264-sdr-crf25": "9fe2d74edfa948c44f730dd61aa64f721bb606336491b6a5b7625ce40a6eee16",
}
QUALITY_PROFILE_VERSION = 1
LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")


def _with_another_rung_bitrate(profile: MediaProfile) -> MediaProfile:
    """The profile with one bit per second more in its smallest rung: a value three levels down."""
    ladder = profile.encode.ladder
    smallest = dataclasses.replace(ladder[-1], audio_bitrate=ladder[-1].audio_bitrate + 1)
    encode = dataclasses.replace(profile.encode, ladder=(*ladder[:-1], smallest))
    return dataclasses.replace(profile, encode=encode)


def _flattened(value: object, path: str = "") -> dict[str, object]:
    """Every leaf of a profile's fields by its dotted path, e.g. "encode.crf"."""
    if isinstance(value, dict):
        leaves: dict[str, object] = {}
        for key, inner in value.items():
            leaves.update(_flattened(inner, f"{path}.{key}" if path else key))
        return leaves
    return {path: value}


def test_profile_digest_when_the_profile_is_the_pilot_should_be_its_pinned_value():
    # Act
    digest = profile_digest(PILOT_PROFILE)

    # Assert
    assert digest == PILOT_PROFILE_SHA256
    assert LOWERCASE_SHA256.fullmatch(digest) is not None


def test_profile_digest_when_called_twice_should_give_the_same_value():
    # Act
    first, second = profile_digest(PILOT_PROFILE), profile_digest(PILOT_PROFILE)

    # Assert
    assert first == second


def test_profile_digest_when_an_equal_profile_is_built_anew_should_give_the_same_value():
    # Arrange: equal in every field, another object throughout
    rebuilt = dataclasses.replace(
        PILOT_PROFILE,
        source=dataclasses.replace(PILOT_PROFILE.source),
        probe=dataclasses.replace(PILOT_PROFILE.probe),
        encode=dataclasses.replace(PILOT_PROFILE.encode, ladder=tuple(PILOT_PROFILE.encode.ladder)),
    )

    # Act
    digest = profile_digest(rebuilt)

    # Assert
    assert digest == PILOT_PROFILE_SHA256


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(dataclasses.replace(PILOT_PROFILE, version=3), id="version"),
        pytest.param(dataclasses.replace(PILOT_PROFILE, profile_id="pilot-h264-sdr-b"), id="profile-id"),
        pytest.param(
            dataclasses.replace(
                PILOT_PROFILE, source=dataclasses.replace(PILOT_PROFILE.source, max_audio_streams=3)
            ),
            id="a-source-cap",
        ),
        pytest.param(
            dataclasses.replace(
                PILOT_PROFILE, source=dataclasses.replace(PILOT_PROFILE.source, reject_hdr=False)
            ),
            id="a-source-flag",
        ),
        pytest.param(
            dataclasses.replace(
                PILOT_PROFILE,
                source=dataclasses.replace(PILOT_PROFILE.source, allowed_demuxers=("avi", "matroska", "mov")),
            ),
            id="an-allow-list",
        ),
        pytest.param(
            dataclasses.replace(
                PILOT_PROFILE, probe=dataclasses.replace(PILOT_PROFILE.probe, max_requests=65)
            ),
            id="a-probe-budget",
        ),
        pytest.param(
            dataclasses.replace(
                PILOT_PROFILE, encode=dataclasses.replace(PILOT_PROFILE.encode, encoder_preset="slow")
            ),
            id="the-encoder-preset",
        ),
        pytest.param(_with_another_rung_bitrate(PILOT_PROFILE), id="a-ladder-rung"),
    ],
)
def test_profile_digest_when_any_field_changes_should_change(changed):
    # Act
    digest = profile_digest(changed)

    # Assert
    assert digest != PILOT_PROFILE_SHA256
    assert LOWERCASE_SHA256.fullmatch(digest) is not None


def test_profile_digest_when_given_something_else_than_a_profile_should_raise_type_error():
    # Act and Assert
    with pytest.raises(TypeError):
        profile_digest(dataclasses.asdict(PILOT_PROFILE))  # type: ignore[arg-type]


def test_supported_profile_when_the_id_version_and_digest_are_the_pilots_should_return_the_pilot_profile():
    # Act
    found = supported_profile("pilot-h264-sdr", 2, PILOT_PROFILE_SHA256)

    # Assert
    assert found is PILOT_PROFILE
    assert SUPPORTED_PROFILES[0] is PILOT_PROFILE


@pytest.mark.parametrize(
    ("profile_id", "version", "sha256"),
    [
        pytest.param("pilot-h264-hdr", 2, PILOT_PROFILE_SHA256, id="another-id"),
        pytest.param("pilot-h264-sdr", 3, PILOT_PROFILE_SHA256, id="another-version"),
        pytest.param("pilot-h264-sdr", 2, "ab" * 32, id="another-digest"),
        pytest.param("pilot-h264-sdr", 2, PILOT_PROFILE_SHA256.upper(), id="the-digest-in-upper-case"),
        pytest.param("pilot-h264-sdr", "2", PILOT_PROFILE_SHA256, id="the-version-as-text"),
        # Version 1 (its ladder held 480p, which left the contract) under its own pinned digest
        pytest.param(
            "pilot-h264-sdr",
            1,
            "0c6b34361abe8b124e90aec0d6ee74d2ae6c9048baf6c2d5db68e0fa1be50d53",
            id="the-retired-version-1",
        ),
    ],
)
def test_supported_profile_when_one_of_the_three_differs_should_return_none(profile_id, version, sha256):
    # Act
    found = supported_profile(profile_id, version, sha256)

    # Assert
    assert found is None


def test_supported_profile_when_a_held_profile_was_changed_without_a_new_digest_should_not_match_the_old_one(
    monkeypatch,
):
    # Arrange: the worker now holds a changed profile under the pilot's id and version
    changed = dataclasses.replace(
        PILOT_PROFILE, encode=dataclasses.replace(PILOT_PROFILE.encode, encoder_preset="slow")
    )
    monkeypatch.setattr(profile_module, "SUPPORTED_PROFILES", (changed,))

    # Act
    by_the_old_digest = supported_profile("pilot-h264-sdr", 2, PILOT_PROFILE_SHA256)
    by_its_own_digest = supported_profile("pilot-h264-sdr", 2, profile_digest(changed))

    # Assert
    assert by_the_old_digest is None
    assert by_its_own_digest is changed


def test_supported_profile_when_the_version_is_a_bool_equal_to_a_held_version_should_return_none(monkeypatch):
    # Arrange: the worker holds a profile of version 1, which True equals in Python
    held = dataclasses.replace(PILOT_PROFILE, version=1)
    monkeypatch.setattr(profile_module, "SUPPORTED_PROFILES", (held,))

    # Act
    found = supported_profile("pilot-h264-sdr", True, profile_digest(held))

    # Assert
    assert found is None


@pytest.mark.parametrize("level", QUALITY_PROFILES, ids=lambda level: level.profile_id)
def test_profile_digest_when_the_profile_is_a_quality_level_should_be_its_pinned_value(level):
    # Act
    digest = profile_digest(level)

    # Assert
    assert digest == QUALITY_PROFILE_SHA256[level.profile_id]


def test_supported_profiles_when_listed_should_be_the_pilot_then_the_four_quality_levels():
    # Act
    held = [(profile.profile_id, profile.version) for profile in SUPPORTED_PROFILES]

    # Assert
    assert held == [
        ("pilot-h264-sdr", 2),
        ("pilot-h264-sdr-crf21", QUALITY_PROFILE_VERSION),
        ("pilot-h264-sdr-crf22", QUALITY_PROFILE_VERSION),
        ("pilot-h264-sdr-crf23", QUALITY_PROFILE_VERSION),
        ("pilot-h264-sdr-crf25", QUALITY_PROFILE_VERSION),
    ]


@pytest.mark.parametrize("profile_id", sorted(QUALITY_PROFILE_SHA256))
def test_supported_profile_when_the_triple_is_a_quality_levels_should_return_that_level(profile_id):
    # Act
    found = supported_profile(profile_id, QUALITY_PROFILE_VERSION, QUALITY_PROFILE_SHA256[profile_id])

    # Assert
    assert found is not None
    assert found.profile_id == profile_id
    assert found in QUALITY_PROFILES


@pytest.mark.parametrize("profile_id", sorted(QUALITY_PROFILE_SHA256))
def test_supported_profile_when_a_quality_level_is_named_with_version_2_should_return_none(profile_id):
    # Act
    found = supported_profile(profile_id, 2, QUALITY_PROFILE_SHA256[profile_id])

    # Assert
    assert found is None


@pytest.mark.parametrize("profile_id", sorted(QUALITY_PROFILE_SHA256))
def test_supported_profile_when_a_quality_level_is_named_with_another_levels_digest_should_return_none(profile_id):
    # Arrange
    other_digests = [digest for other, digest in QUALITY_PROFILE_SHA256.items() if other != profile_id]

    # Act
    found = [supported_profile(profile_id, QUALITY_PROFILE_VERSION, digest) for digest in other_digests]

    # Assert
    assert found == [None] * len(other_digests)


@pytest.mark.parametrize("level", QUALITY_PROFILES, ids=lambda level: level.profile_id)
def test_quality_level_when_compared_with_the_pilot_should_differ_only_in_id_version_preset_and_crf(level):
    # Arrange
    pilot = _flattened(dataclasses.asdict(PILOT_PROFILE))
    quality = _flattened(dataclasses.asdict(level))

    # Act
    differing = {path for path in pilot.keys() | quality.keys() if pilot.get(path) != quality.get(path)}

    # Assert
    assert differing == {"profile_id", "version", "encode.encoder_preset", "encode.crf"}
    assert (quality["version"], quality["encode.encoder_preset"]) == (QUALITY_PROFILE_VERSION, "fast")
    assert quality["profile_id"] == f"pilot-h264-sdr-crf{quality['encode.crf']}"


def test_profile_digest_when_a_quality_levels_crf_changes_should_change():
    # Arrange
    level = QUALITY_PROFILES[0]
    changed = dataclasses.replace(level, encode=dataclasses.replace(level.encode, crf=level.encode.crf + 1))

    # Act
    digest = profile_digest(changed)

    # Assert
    assert digest != profile_digest(level)
    assert digest not in QUALITY_PROFILE_SHA256.values()


def test_profile_digest_when_the_pilot_gains_a_crf_should_change():
    # Arrange: a crf that is set is part of the identity, unlike the absent one
    with_crf = dataclasses.replace(PILOT_PROFILE, encode=dataclasses.replace(PILOT_PROFILE.encode, crf=23))

    # Act
    digest = profile_digest(with_crf)

    # Assert
    assert digest != PILOT_PROFILE_SHA256
