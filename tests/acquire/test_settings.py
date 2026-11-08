"""Acquisition configuration is credential-free, lists the live versions and refuses any other."""

from dataclasses import replace

import pytest

from core.protocol import settings as settings_module
from core.protocol.registry import ContractRegistry
from core.protocol.settings import AcquireSettings, SettingsInvalid, WorkerSettings
from tests.acquire.support import environment
from tests.contract.v2_support import worker_env


def test_acquire_settings_when_storage_credentials_are_absent_should_accept_the_common_configuration() -> None:
    # Arrange
    env = environment()
    # Act
    settings = AcquireSettings.from_env(env)
    # Assert
    assert not any(name.startswith("VH_STORAGE_") for name in env)
    assert not hasattr(settings, "storage_profiles")
    assert settings.clock_skew_s == 300
    assert settings.google_api_key == env["VH_GOOGLE_DRIVE_API_KEY"]


def test_encode_settings_when_storage_profiles_are_absent_should_still_refuse_startup() -> None:
    # Arrange / Act / Assert
    with pytest.raises(SettingsInvalid, match="VH_STORAGE_PROFILES"):
        WorkerSettings.from_env(environment())
    assert WorkerSettings.from_env(worker_env()).storage_profiles


@pytest.mark.parametrize("variable", [
    "VH_ACCEPTED_PROTOCOL_VERSIONS", "VH_WORKER_AUDIENCE", "VH_HUB_TO_WORKER_KEYS",
    "VH_WORKER_TO_HUB_KEY_ID", "VH_WORKER_TO_HUB_KEY", "VH_CLAIM_URL", "VH_CALLBACK_URL",
    "VH_GOOGLE_DRIVE_API_KEY", "VH_ACQUIRE_STORAGE_HOST",
])
def test_acquire_settings_when_a_required_variable_is_missing_should_name_only_the_variable(variable: str) -> None:
    # Arrange
    env = environment()
    env.pop(variable)
    # Act / Assert
    with pytest.raises(SettingsInvalid, match=variable):
        AcquireSettings.from_env(env)


@pytest.mark.parametrize("variable,value", [
    ("VH_ACCEPTED_PROTOCOL_VERSIONS", "2.2.0"), ("VH_ACCEPTED_PROTOCOL_VERSIONS", "2.0.0,2.2.0"),
    ("VH_CLOCK_SKEW_S", "299"), ("VH_CLOCK_SKEW_S", "301"), ("VH_CLOCK_SKEW_S", "300.0"),
    ("VH_ACQUIRE_STORAGE_HOST", "STORE.example.invalid"), ("VH_ACQUIRE_STORAGE_HOST", "store.invalid."),
    ("VH_ACQUIRE_STORAGE_HOST", "https://store.invalid"), ("VH_ACQUIRE_STORAGE_HOST", "store.invalid:443"),
    ("VH_ACQUIRE_STORAGE_HOST", "user@store.invalid"), ("VH_ACQUIRE_STORAGE_HOST", "store.invalid/path"),
    ("VH_ACQUIRE_STORAGE_HOST", "-store.invalid"), ("VH_GOOGLE_DRIVE_API_KEY", "secret key"),
    ("VH_GOOGLE_DRIVE_API_KEY", "secret\nkey"), ("VH_CLAIM_URL", "http://hub.invalid/secret"),
])
def test_acquire_settings_when_a_value_violates_its_rule_should_refuse_without_echoing_it(variable: str, value: str) -> None:
    # Arrange
    env = {**environment(), variable: value}
    # Act / Assert
    with pytest.raises(SettingsInvalid) as caught:
        AcquireSettings.from_env(env)
    assert caught.value.variable == variable
    assert value not in str(caught.value)


def test_acquire_settings_when_every_live_version_is_listed_should_accept_them() -> None:
    # Arrange
    env = {**environment(), "VH_ACCEPTED_PROTOCOL_VERSIONS": "2.0.0,2.1.0"}
    # Act
    settings = AcquireSettings.from_env(env)
    # Assert
    assert settings.accepted_protocol_versions == ("2.0.0", "2.1.0")


@pytest.mark.parametrize("value", ["2.0.0", "2.1.0"])
def test_acquire_settings_when_the_list_misses_a_live_version_should_refuse_the_start(value: str) -> None:
    # Arrange: 2.0.0 alone refuses every acquire request; 2.1.0 alone cannot read the Hub's 2.0.0 hub.error.
    env = {**environment(), "VH_ACCEPTED_PROTOCOL_VERSIONS": value}
    # Act / Assert
    with pytest.raises(SettingsInvalid, match="every live protocol version") as caught:
        AcquireSettings.from_env(env)
    assert caught.value.variable == "VH_ACCEPTED_PROTOCOL_VERSIONS"
    assert value not in str(caught.value)


def test_acquire_settings_when_the_live_versions_are_listed_in_another_order_should_accept_them() -> None:
    # Arrange
    env = {**environment(), "VH_ACCEPTED_PROTOCOL_VERSIONS": "2.1.0,2.0.0"}
    # Act
    settings = AcquireSettings.from_env(env)
    # Assert
    assert settings.accepted_protocol_versions == ("2.1.0", "2.0.0")


@pytest.mark.parametrize("value", ["2.2.0", "2.0.0,2.2.0"])
def test_acquire_settings_when_an_entry_is_a_draft_version_should_refuse_the_start(
    value: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the shipped tree holds no draft, so this test's registry defines one.
    shipped = ContractRegistry.load()
    versions = {**shipped.versions, "protocol": {**shipped.versions["protocol"], "draft": ["2.2.0"]}}
    monkeypatch.setattr(settings_module, "load_registry", lambda: replace(shipped, versions=versions))
    # Act / Assert
    with pytest.raises(SettingsInvalid, match="live protocol version") as caught:
        AcquireSettings.from_env({**environment(), "VH_ACCEPTED_PROTOCOL_VERSIONS": value})
    assert caught.value.variable == "VH_ACCEPTED_PROTOCOL_VERSIONS"


def test_acquire_settings_when_rendered_should_hide_keys_and_endpoints() -> None:
    # Arrange
    env = environment()
    settings = AcquireSettings.from_env(env)
    # Act
    shown = repr(settings)
    # Assert
    for name in ("VH_GOOGLE_DRIVE_API_KEY", "VH_WORKER_TO_HUB_KEY", "VH_CLAIM_URL", "VH_CALLBACK_URL"):
        assert env[name] not in shown
    assert "https://" not in shown
    assert repr(replace(settings, google_api_key="CANARY-key")) == shown
