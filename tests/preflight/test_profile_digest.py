"""Which media profile a dispatch names: the profile's digest and the lookup by id, version and digest."""

from __future__ import annotations

import dataclasses
import re

import pytest

from core import profile as profile_module
from core.profile import PILOT_PROFILE, SUPPORTED_PROFILES, MediaProfile, profile_digest, supported_profile

# The digest of the pilot profile as it is today. A change of this value means the profile changed, and
# a changed profile is a new revision: bump PILOT_PROFILE's `version`, then pin the new digest here, and
# tell the Hub the new `{id, version, sha256}`: it dispatches the old one until then, which this worker
# refuses (`unsupported_profile`). Never update this value alone.
PILOT_PROFILE_SHA256 = "0c6b34361abe8b124e90aec0d6ee74d2ae6c9048baf6c2d5db68e0fa1be50d53"
LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")


def _with_another_rung_bitrate(profile: MediaProfile) -> MediaProfile:
    """The profile with one bit per second more in its smallest rung: a value three levels down."""
    ladder = profile.encode.ladder
    smallest = dataclasses.replace(ladder[-1], audio_bitrate=ladder[-1].audio_bitrate + 1)
    encode = dataclasses.replace(profile.encode, ladder=(*ladder[:-1], smallest))
    return dataclasses.replace(profile, encode=encode)


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
        pytest.param(dataclasses.replace(PILOT_PROFILE, version=2), id="version"),
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
    found = supported_profile("pilot-h264-sdr", 1, PILOT_PROFILE_SHA256)

    # Assert
    assert found is PILOT_PROFILE
    assert SUPPORTED_PROFILES == (PILOT_PROFILE,)


@pytest.mark.parametrize(
    ("profile_id", "version", "sha256"),
    [
        pytest.param("pilot-h264-hdr", 1, PILOT_PROFILE_SHA256, id="another-id"),
        pytest.param("pilot-h264-sdr", 2, PILOT_PROFILE_SHA256, id="another-version"),
        pytest.param("pilot-h264-sdr", 1, "ab" * 32, id="another-digest"),
        pytest.param("pilot-h264-sdr", 1, PILOT_PROFILE_SHA256.upper(), id="the-digest-in-upper-case"),
        pytest.param("pilot-h264-sdr", True, PILOT_PROFILE_SHA256, id="a-bool-for-version-1"),
        pytest.param("pilot-h264-sdr", "1", PILOT_PROFILE_SHA256, id="the-version-as-text"),
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
    by_the_old_digest = supported_profile("pilot-h264-sdr", 1, PILOT_PROFILE_SHA256)
    by_its_own_digest = supported_profile("pilot-h264-sdr", 1, profile_digest(changed))

    # Assert
    assert by_the_old_digest is None
    assert by_its_own_digest is changed
