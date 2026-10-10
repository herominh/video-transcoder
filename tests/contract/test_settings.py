"""The worker's v2 configuration loads only when every variable holds, and never shows a key or a secret."""

import re
import traceback
from typing import Any, Iterator, Mapping

import pytest

from core.protocol import files
from core.protocol.settings import (
    MAX_HUB_TO_WORKER_KEYS,
    SettingsInvalid,
    StorageProfile,
    UnknownCredentialRef,
    WorkerSettings,
)

# Canaries: none of these may ever appear in an error, a traceback or a repr.
HUB_KEY_HEX = "c0ffee11" * 8
SECOND_HUB_KEY_HEX = "c0ffee22" * 8
WORKER_KEY_HEX = "deadbeef" * 8
ACCESS_KEY_ID = "AKIACANARYACCESSKEYID"
SECRET_ACCESS_KEY = "CanarySecretAccessKeyValue0123456789"
CANARIES = (HUB_KEY_HEX, SECOND_HUB_KEY_HEX, WORKER_KEY_HEX, ACCESS_KEY_ID, SECRET_ACCESS_KEY)
# Showing this many leading characters of a canary is showing it: a shortened key, or one in another case, counts.
CANARY_PREFIX_LENGTH = 16
# A value shorter than this is not looked for in a refusal: see _refusal().
MIN_LOOKED_FOR_VALUE_LENGTH = 8

VALID_ENV = {
    "VH_ACCEPTED_PROTOCOL_VERSIONS": "2.0.0-draft",
    "VH_WORKER_AUDIENCE": "worker:runpod-test",
    "VH_HUB_TO_WORKER_KEYS": f"test-hub2wk-k1={HUB_KEY_HEX},test-hub2wk-k2={SECOND_HUB_KEY_HEX}",
    "VH_WORKER_TO_HUB_KEY_ID": "test-wk2hub-k1",
    "VH_WORKER_TO_HUB_KEY": WORKER_KEY_HEX,
    "VH_CLAIM_URL": "https://hub.test.invalid/transcode/claim",
    "VH_CALLBACK_URL": "https://hub.test.invalid/transcode/callback",
    "VH_STORAGE_PROFILES": "default",
    "VH_STORAGE_DEFAULT_ENDPOINT": "https://storage.test.invalid",
    "VH_STORAGE_DEFAULT_ACCESS_KEY_ID": ACCESS_KEY_ID,
    "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY": SECRET_ACCESS_KEY,
}
REQUIRED_VARIABLES = [
    "VH_ACCEPTED_PROTOCOL_VERSIONS",
    "VH_WORKER_AUDIENCE",
    "VH_HUB_TO_WORKER_KEYS",
    "VH_WORKER_TO_HUB_KEY_ID",
    "VH_WORKER_TO_HUB_KEY",
    "VH_CLAIM_URL",
    "VH_CALLBACK_URL",
    "VH_STORAGE_PROFILES",
    "VH_STORAGE_DEFAULT_ENDPOINT",
    "VH_STORAGE_DEFAULT_ACCESS_KEY_ID",
    "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY",
]
URL_VARIABLES = ["VH_CLAIM_URL", "VH_CALLBACK_URL", "VH_STORAGE_DEFAULT_ENDPOINT"]

# The real name of a storage profile's variable holds an entry of VH_STORAGE_PROFILES, a value an operator
# can mistype a secret into. So a refusal names the list, the entry's position and the variable with
# `<NAME>` for the entry's own name; `default` is entry 1 of VALID_ENV's list.
DEFAULT_PROFILE_LEAD_INS = {
    "VH_STORAGE_DEFAULT_ENDPOINT": "entry 1: its VH_STORAGE_<NAME>_ENDPOINT variable ",
    "VH_STORAGE_DEFAULT_REGION": "entry 1: its VH_STORAGE_<NAME>_REGION variable ",
    "VH_STORAGE_DEFAULT_ACCESS_KEY_ID": "entry 1: its VH_STORAGE_<NAME>_ACCESS_KEY_ID variable ",
    "VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY": "entry 1: its VH_STORAGE_<NAME>_SECRET_ACCESS_KEY variable ",
}
# A storage name other than the list's own and the `<NAME>` form: one made from an entry of the list.
DERIVED_STORAGE_NAME = re.compile(r"VH_STORAGE_(?!PROFILES\b|<NAME>_)")
# A second profile, `eu-archive`, after `default`: entry 2 of the list.
TWO_PROFILES_ENV = {
    "VH_STORAGE_PROFILES": "default,eu-archive",
    "VH_STORAGE_EU_ARCHIVE_ENDPOINT": "https://eu.storage.test.invalid",
    "VH_STORAGE_EU_ARCHIVE_REGION": "eu-west-1",
    "VH_STORAGE_EU_ARCHIVE_ACCESS_KEY_ID": "AKIAEUARCHIVE",
    "VH_STORAGE_EU_ARCHIVE_SECRET_ACCESS_KEY": "eu-archive-secret",
}

NOT_AN_HTTPS_URL = "must be an absolute https URL"
CARRIES_USERINFO = "must not carry userinfo"
CARRIES_A_FRAGMENT = "must not carry a fragment"
# The printable ASCII characters no URL holds (RFC 3986), by name.
NOT_URL_CHARACTERS = {
    "space": " ",
    "double-quote": '"',
    "less-than": "<",
    "greater-than": ">",
    "backslash": "\\",
    "caret": "^",
    "backtick": "`",
    "left-brace": "{",
    "pipe": "|",
    "right-brace": "}",
}


def _env(**changes: str | None) -> dict[str, str]:
    """The valid environment with some variables replaced (a value) or removed (None)."""
    env = dict(VALID_ENV)
    for variable, value in changes.items():
        if value is None:
            env.pop(variable, None)
        else:
            env[variable] = value
    return env


def _refusal(env: Mapping[str, str]) -> SettingsInvalid:
    """The refusal of `env`, checked on the way: it shows no value of `env` (the refused one included,
    in upper or lower case) and no storage name made from one.

    A value shorter than MIN_LOOKED_FOR_VALUE_LENGTH is not looked for: an empty string, a single
    character or a word of the fixed rule text is found in a refusal that shows no value at all.
    """
    with pytest.raises(SettingsInvalid) as raised:
        WorkerSettings.from_env(env)
    refusal = raised.value
    text = str(refusal)
    looked_for = [value for value in env.values() if len(value) >= MIN_LOOKED_FOR_VALUE_LENGTH]
    assert [value for value in looked_for if value.lower() in text.lower()] == []
    assert DERIVED_STORAGE_NAME.search(text) is None
    return refusal


def _shown_as(variable: str) -> tuple[str, str]:
    """How a refusal names `variable`: (the variable it shows, what its rule starts with)."""
    if variable in DEFAULT_PROFILE_LEAD_INS:
        return "VH_STORAGE_PROFILES", DEFAULT_PROFILE_LEAD_INS[variable]
    return variable, ""


def _keys(count: int) -> str:
    return ",".join(f"hub-key-{index:02d}={index:02x}" + "ab" * 31 for index in range(count))


class _RecordingEnv(Mapping[str, str]):
    """An environment that records every variable read from it."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.read: set[str] = set()

    def __getitem__(self, variable: str) -> str:
        self.read.add(variable)
        return self._values[variable]

    def get(self, variable: str, default: Any = None) -> Any:
        self.read.add(variable)
        return self._values.get(variable, default)

    def __contains__(self, variable: object) -> bool:
        if isinstance(variable, str):
            self.read.add(variable)
        return variable in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


# --- a valid configuration ---------------------------------------------------------------------


def test_from_env_when_every_variable_is_valid_should_hold_the_parsed_values():
    # Act
    settings = WorkerSettings.from_env(VALID_ENV)

    # Assert
    assert settings.accepted_protocol_versions == ("2.0.0-draft",)
    assert settings.worker_audience == "worker:runpod-test"
    assert dict(settings.hub_to_worker_keys) == {
        "test-hub2wk-k1": bytes.fromhex(HUB_KEY_HEX),
        "test-hub2wk-k2": bytes.fromhex(SECOND_HUB_KEY_HEX),
    }
    assert settings.hub_to_worker_key_ids == ("test-hub2wk-k1", "test-hub2wk-k2")
    assert settings.worker_to_hub_key_id == "test-wk2hub-k1"
    assert settings.worker_to_hub_key == bytes.fromhex(WORKER_KEY_HEX)
    assert settings.claim_url == "https://hub.test.invalid/transcode/claim"
    assert settings.callback_url == "https://hub.test.invalid/transcode/callback"
    assert settings.clock_skew_s == 300
    assert settings.storage_profile("default") == StorageProfile(
        credential_ref="default",
        endpoint="https://storage.test.invalid",
        region="auto",
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
    )


def test_from_env_when_loaded_should_be_immutable():
    # Arrange
    settings = WorkerSettings.from_env(VALID_ENV)

    # Act / Assert
    with pytest.raises(TypeError):
        settings.hub_to_worker_keys["another-key"] = b"x" * 32  # type: ignore[index]
    with pytest.raises(TypeError):
        settings.storage_profiles["another"] = settings.storage_profile("default")  # type: ignore[index]


@pytest.mark.parametrize("variable", REQUIRED_VARIABLES)
def test_from_env_when_a_required_variable_is_missing_should_refuse_naming_it(variable):
    # Arrange
    shown, lead_in = _shown_as(variable)

    # Act
    refusal = _refusal(_env(**{variable: None}))

    # Assert
    assert (refusal.variable, refusal.rule) == (shown, lead_in + "is required")
    assert str(refusal) == f"{shown} {lead_in}is required"


@pytest.mark.parametrize("variable", REQUIRED_VARIABLES)
def test_from_env_when_a_variable_is_empty_should_refuse_naming_it(variable):
    # Arrange
    shown, lead_in = _shown_as(variable)

    # Act
    refusal = _refusal(_env(**{variable: ""}))

    # Assert
    assert (refusal.variable, refusal.rule) == (shown, lead_in + "must not be empty")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("VH_CLAIM_URL", "https://hub.test\t.invalid/claim"),
        ("VH_CALLBACK_URL", "https://hub.test.invalid/call\x7fback"),
        ("VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY", "storage\x00secret"),
        ("VH_STORAGE_DEFAULT_REGION", "eu-wést-1"),
    ],
    ids=["tab-in-the-claim-url", "del-in-the-callback-url", "nul-in-a-storage-secret", "not-ascii-in-a-storage-region"],
)
def test_from_env_when_a_value_is_not_printable_ascii_should_refuse(variable, value):
    # Arrange: each value breaks this rule only, so nothing else refuses it.
    shown, lead_in = _shown_as(variable)

    # Act
    refusal = _refusal(_env(**{variable: value}))

    # Assert
    assert refusal.variable == shown
    assert refusal.rule.startswith(lead_in)
    assert "printable ASCII" in refusal.rule


@pytest.mark.parametrize("value", [" 2.0.0-draft", "2.0.0-draft "], ids=["leading-space", "trailing-space"])
def test_from_env_when_a_value_starts_or_ends_with_a_space_should_refuse(value):
    # Act
    refusal = _refusal(_env(VH_ACCEPTED_PROTOCOL_VERSIONS=value))

    # Assert
    assert refusal.variable == "VH_ACCEPTED_PROTOCOL_VERSIONS"
    assert "start or end with a space" in refusal.rule


# --- VH_ACCEPTED_PROTOCOL_VERSIONS -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        ("3.0.0-draft", "versions.json"),
        ("2.0.0", "versions.json"),
        ("2.0.0-draft,2.0.0-draft", "twice"),
        ("2.0.0-draft,", "empty entry"),
    ],
    ids=["another-major", "the-release-versions-json-does-not-define", "listed-twice", "empty-entry"],
)
def test_from_env_when_the_accepted_versions_break_a_rule_should_refuse(value, rule):
    # Act
    refusal = _refusal(_env(VH_ACCEPTED_PROTOCOL_VERSIONS=value))

    # Assert
    assert refusal.variable == "VH_ACCEPTED_PROTOCOL_VERSIONS"
    assert rule in refusal.rule


# --- VH_WORKER_AUDIENCE ------------------------------------------------------------------------


def test_from_env_when_the_audience_name_has_the_longest_allowed_length_should_load():
    # Arrange
    audience = "worker:" + "a" * 40

    # Act
    settings = WorkerSettings.from_env(_env(VH_WORKER_AUDIENCE=audience))

    # Assert
    assert settings.worker_audience == audience


@pytest.mark.parametrize(
    "value",
    ["hub:cell-test-a", "runpod-test", "worker:RunPod", "worker:a", "worker:" + "a" * 41, "worker:-runpod"],
    ids=["a-hub-audience", "no-prefix", "upper-case", "name-too-short", "name-too-long", "name-starts-with-dash"],
)
def test_from_env_when_the_audience_is_not_a_worker_audience_should_refuse(value):
    # Act
    refusal = _refusal(_env(VH_WORKER_AUDIENCE=value))

    # Assert
    assert refusal.variable == "VH_WORKER_AUDIENCE"


# --- the keys ----------------------------------------------------------------------------------


def test_from_env_when_eight_hub_keys_are_given_should_load_all_of_them():
    # Act
    settings = WorkerSettings.from_env(_env(VH_HUB_TO_WORKER_KEYS=_keys(MAX_HUB_TO_WORKER_KEYS)))

    # Assert
    assert len(settings.hub_to_worker_keys) == MAX_HUB_TO_WORKER_KEYS


def test_max_hub_to_worker_keys_when_compared_with_the_context_schema_should_equal_known_key_ids_max_items():
    # Arrange
    known_key_ids = files.read_json("schemas/trusted-context.schema.json")["properties"]["known_key_ids"]

    # Act
    max_items = next(option["maxItems"] for option in known_key_ids["anyOf"] if option.get("type") == "array")

    # Assert
    assert MAX_HUB_TO_WORKER_KEYS == max_items


@pytest.mark.parametrize(
    ("value", "rule"),
    [
        (_keys(MAX_HUB_TO_WORKER_KEYS + 1), "more than 8"),
        (f"test-hub2wk-k1{HUB_KEY_HEX}", "entry 1: is not `kid=hex`"),
        (f"TEST-K1={HUB_KEY_HEX}", "entry 1: the key id"),
        (f"k1={HUB_KEY_HEX}", "entry 1: the key id"),
        (f"test-hub2wk-k1={HUB_KEY_HEX.upper()}", "entry 1: the key must be lowercase hex"),
        (f"test-hub2wk-k1={HUB_KEY_HEX[:-1]}", "entry 1: the key must be lowercase hex"),
        (f"test-hub2wk-k1={HUB_KEY_HEX[:-2]}", "entry 1: the key must be at least 32 bytes"),
        (f"test-hub2wk-k1={HUB_KEY_HEX},test-hub2wk-k1={SECOND_HUB_KEY_HEX}", "entry 2: repeats a key id"),
        (f"test-hub2wk-k1={HUB_KEY_HEX},,", "empty entry"),
    ],
    ids=[
        "nine-keys",
        "no-separator",
        "key-id-upper-case",
        "key-id-too-short",
        "key-upper-case-hex",
        "key-odd-hex-digits",
        "key-31-bytes",
        "key-id-repeated",
        "empty-entry",
    ],
)
def test_from_env_when_the_hub_keys_break_a_rule_should_refuse_naming_the_entry_and_rule(value, rule):
    # Act
    refusal = _refusal(_env(VH_HUB_TO_WORKER_KEYS=value))

    # Assert
    assert refusal.variable == "VH_HUB_TO_WORKER_KEYS"
    assert rule in refusal.rule


@pytest.mark.parametrize(
    ("changes", "variable", "rule"),
    [
        ({"VH_WORKER_TO_HUB_KEY_ID": "Test-Wk2Hub"}, "VH_WORKER_TO_HUB_KEY_ID", "key_id pattern"),
        ({"VH_WORKER_TO_HUB_KEY_ID": "test-hub2wk-k1"}, "VH_WORKER_TO_HUB_KEY_ID", "Hub-to-worker key"),
        ({"VH_WORKER_TO_HUB_KEY": WORKER_KEY_HEX[:-2]}, "VH_WORKER_TO_HUB_KEY", "at least 32 bytes"),
        ({"VH_WORKER_TO_HUB_KEY": WORKER_KEY_HEX.upper()}, "VH_WORKER_TO_HUB_KEY", "lowercase hex"),
        ({"VH_WORKER_TO_HUB_KEY": HUB_KEY_HEX}, "VH_WORKER_TO_HUB_KEY", "per direction"),
    ],
    ids=["id-pattern", "id-of-a-hub-key", "key-31-bytes", "key-upper-case", "key-of-the-hub"],
)
def test_from_env_when_the_worker_key_breaks_a_rule_should_refuse(changes, variable, rule):
    # Act
    refusal = _refusal(_env(**changes))

    # Assert
    assert refusal.variable == variable
    assert rule in refusal.rule


# --- the URLs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("variable", URL_VARIABLES)
@pytest.mark.parametrize(
    ("value", "rule"),
    [
        ("http://hub.test.invalid/claim", NOT_AN_HTTPS_URL),
        ("ftp://hub.test.invalid/claim", NOT_AN_HTTPS_URL),
        ("hub.test.invalid/claim", NOT_AN_HTTPS_URL),
        ("https:///claim", NOT_AN_HTTPS_URL),
        ("https://hub.test.invalid:99999/claim", NOT_AN_HTTPS_URL),
        ("https://hub.test.invalid:0/claim", NOT_AN_HTTPS_URL),
        ("https://hub.test.invalid:/claim", NOT_AN_HTTPS_URL),
        ("https://hub.test.invalid/a b", NOT_AN_HTTPS_URL),
        ("https://hub.test.invalid\\claim", NOT_AN_HTTPS_URL),
        ("https://evil.example\\hub.test.invalid/claim", NOT_AN_HTTPS_URL),
        ("https://https://hub.test.invalid/claim", NOT_AN_HTTPS_URL),
        ('https://a_b^c|d"e/claim', NOT_AN_HTTPS_URL),
        ("https://a_b.example/claim", NOT_AN_HTTPS_URL),
        ("https://hub%2eexample/claim", NOT_AN_HTTPS_URL),
        ("https://./claim", NOT_AN_HTTPS_URL),
        ("https://-a.example/claim", NOT_AN_HTTPS_URL),
        ("https://a..b/claim", NOT_AN_HTTPS_URL),
        ("https://2130706433/claim", NOT_AN_HTTPS_URL),
        ("https://0x7f.1/claim", NOT_AN_HTTPS_URL),
        ("https://hub.test.0x7f000001/claim", NOT_AN_HTTPS_URL),
        ("https://999.999.999.999/claim", NOT_AN_HTTPS_URL),
        ("https://1.2.3.4.5/claim", NOT_AN_HTTPS_URL),
        ("https://user:password@hub.test.invalid/claim", CARRIES_USERINFO),
        ("https://user@hub.test.invalid/claim", CARRIES_USERINFO),
        ("https://hub.test.invalid/claim#part", CARRIES_A_FRAGMENT),
        ("https://hub.test.invalid/claim#", CARRIES_A_FRAGMENT),
    ],
    ids=[
        "plain-http",
        "another-scheme",
        "relative",
        "no-host",
        "port-out-of-range",
        "port-zero",
        "port-empty",
        "space",
        "backslash-for-the-path-slash",
        "backslash-in-the-authority",
        "scheme-twice",
        "host-of-characters-no-url-holds",
        "host-label-with-an-underscore",
        "host-with-a-percent-escape",
        "host-of-a-dot",
        "label-starting-with-a-hyphen",
        "label-empty",
        "host-of-one-number",
        "host-of-a-hex-number-and-a-number",
        "host-ending-in-a-hex-number",
        "ipv4-address-out-of-range",
        "ipv4-address-of-five-parts",
        "user-and-password",
        "user",
        "fragment",
        "empty-fragment",
    ],
)
def test_from_env_when_a_url_is_not_a_bare_https_url_should_refuse(variable, value, rule):
    # Arrange
    shown, lead_in = _shown_as(variable)

    # Act
    refusal = _refusal(_env(**{variable: value}))

    # Assert
    assert (refusal.variable, refusal.rule) == (shown, lead_in + rule)


@pytest.mark.parametrize("variable", URL_VARIABLES)
@pytest.mark.parametrize("character", list(NOT_URL_CHARACTERS.values()), ids=list(NOT_URL_CHARACTERS))
def test_from_env_when_a_url_holds_a_character_no_url_can_hold_should_refuse(variable, character):
    # Arrange: in the query, which no other rule reads, so only the character can refuse it.
    shown, lead_in = _shown_as(variable)
    url = f"https://hub.test.invalid/claim?next=a{character}b"

    # Act
    refusal = _refusal(_env(**{variable: url}))

    # Assert
    assert (refusal.variable, refusal.rule) == (shown, lead_in + NOT_AN_HTTPS_URL)


@pytest.mark.parametrize(
    "url",
    ["https://hub.test.invalid:8443/claim?x=1", "https://[::1]:8443/claim", "https://127.0.0.1/claim"],
    ids=["name-with-port-and-query", "bracketed-ipv6-with-port", "ipv4-address"],
)
def test_from_env_when_a_url_names_a_host_with_an_optional_port_should_hold_it_unchanged(url):
    # Act
    settings = WorkerSettings.from_env(_env(**{variable: url for variable in URL_VARIABLES}))

    # Assert
    assert (settings.claim_url, settings.callback_url, settings.storage_profile("default").endpoint) == (url, url, url)


# --- VH_CLOCK_SKEW_S ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "seconds"), [("0", 0), ("120", 120), ("600", 600)])
def test_from_env_when_the_clock_skew_is_within_the_context_bounds_should_load_it(value, seconds):
    # Act
    settings = WorkerSettings.from_env(_env(VH_CLOCK_SKEW_S=value))

    # Assert
    assert settings.clock_skew_s == seconds


@pytest.mark.parametrize(
    "value",
    ["601", "-1", "01", "1.5", "abc", "9" * 5_000],
    ids=["above-600", "negative", "leading-zero", "fraction", "not-a-number", "more-digits-than-int-converts"],
)
def test_from_env_when_the_clock_skew_is_not_an_integer_within_bounds_should_refuse(value):
    # Act
    refusal = _refusal(_env(VH_CLOCK_SKEW_S=value))

    # Assert
    assert refusal.variable == "VH_CLOCK_SKEW_S"


# --- the storage profiles ----------------------------------------------------------------------


def test_from_env_when_two_profiles_are_configured_should_map_each_ref_to_its_variables():
    # Arrange
    env = _env(**TWO_PROFILES_ENV)

    # Act
    settings = WorkerSettings.from_env(env)

    # Assert
    archive = settings.storage_profile("eu-archive")
    assert (archive.endpoint, archive.region, archive.access_key_id, archive.secret_access_key) == (
        "https://eu.storage.test.invalid",
        "eu-west-1",
        "AKIAEUARCHIVE",
        "eu-archive-secret",
    )
    assert settings.storage_profile("default").region == "auto"


@pytest.mark.parametrize(
    ("changes", "variable", "rule"),
    [
        ({"VH_STORAGE_PROFILES": "Default"}, "VH_STORAGE_PROFILES", "entry 1 is not a contract credential_ref"),
        ({"VH_STORAGE_PROFILES": "1st"}, "VH_STORAGE_PROFILES", "entry 1 is not a contract credential_ref"),
        ({"VH_STORAGE_PROFILES": "default,default"}, "VH_STORAGE_PROFILES", "entry 2 repeats"),
        ({"VH_STORAGE_PROFILES": "default,"}, "VH_STORAGE_PROFILES", "empty entry"),
        (
            {"VH_STORAGE_DEFAULT_REGION": ""},
            "VH_STORAGE_PROFILES",
            "entry 1: its VH_STORAGE_<NAME>_REGION variable must not be empty",
        ),
    ],
    ids=["ref-upper-case", "ref-starts-with-digit", "ref-repeated", "empty-entry", "region-empty"],
)
def test_from_env_when_a_storage_variable_breaks_a_rule_should_refuse(changes, variable, rule):
    # Act
    refusal = _refusal(_env(**changes))

    # Assert
    assert refusal.variable == variable
    assert rule in refusal.rule


@pytest.mark.parametrize(
    ("changes", "rule"),
    [
        (
            {"VH_STORAGE_EU_ARCHIVE_ENDPOINT": None},
            "entry 2: its VH_STORAGE_<NAME>_ENDPOINT variable is required",
        ),
        (
            {"VH_STORAGE_EU_ARCHIVE_ENDPOINT": "http://eu.storage.test.invalid"},
            "entry 2: its VH_STORAGE_<NAME>_ENDPOINT variable must be an absolute https URL",
        ),
        (
            {"VH_STORAGE_EU_ARCHIVE_REGION": "eu-west-1 "},
            "entry 2: its VH_STORAGE_<NAME>_REGION variable must not start or end with a space",
        ),
        (
            {"VH_STORAGE_EU_ARCHIVE_ACCESS_KEY_ID": ""},
            "entry 2: its VH_STORAGE_<NAME>_ACCESS_KEY_ID variable must not be empty",
        ),
        (
            {"VH_STORAGE_EU_ARCHIVE_SECRET_ACCESS_KEY": None},
            "entry 2: its VH_STORAGE_<NAME>_SECRET_ACCESS_KEY variable is required",
        ),
    ],
    ids=["endpoint-missing", "endpoint-plain-http", "region-trailing-space", "access-key-id-empty", "secret-missing"],
)
def test_from_env_when_a_variable_of_the_second_profile_breaks_a_rule_should_refuse_naming_entry_2(changes, rule):
    # Arrange: `default`, the first entry, is whole.
    env = _env(**{**TWO_PROFILES_ENV, **changes})

    # Act
    refusal = _refusal(env)

    # Assert
    assert (refusal.variable, refusal.rule) == ("VH_STORAGE_PROFILES", rule)
    assert str(refusal) == f"VH_STORAGE_PROFILES {rule}"


def test_from_env_when_two_refs_map_to_the_same_variables_should_refuse():
    # Arrange: `a-b` and `a_b` both read VH_STORAGE_A_B_*.
    env = _env(
        VH_STORAGE_PROFILES="a-b,a_b",
        VH_STORAGE_A_B_ENDPOINT="https://storage.test.invalid",
        VH_STORAGE_A_B_ACCESS_KEY_ID="AKIA",
        VH_STORAGE_A_B_SECRET_ACCESS_KEY="secret",
    )

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == "VH_STORAGE_PROFILES"
    assert "entries 1 and 2 map to the same" in refusal.rule


@pytest.mark.parametrize("credential_ref", ["archive", "", None, 7])
def test_storage_profile_when_the_ref_is_not_configured_should_raise_unknown_credential_ref(credential_ref):
    # Arrange
    settings = WorkerSettings.from_env(VALID_ENV)

    # Act / Assert
    with pytest.raises(UnknownCredentialRef) as raised:
        settings.storage_profile(credential_ref)
    assert isinstance(raised.value, LookupError)
    assert (raised.value.ERROR_CLASS, raised.value.ERROR_CODE) == ("configuration_error", "unknown_credential_ref")


# --- no secret is ever shown -------------------------------------------------------------------


def _everything_shown(refusal: SettingsInvalid) -> str:
    """All a refusal can show: its text, repr and attributes, its formatted traceback with the chain,
    and the exceptions it was raised from or while handling (`from None` hides those, it does not drop them)."""
    behind = [repr(error) for error in (refusal.__context__, refusal.__cause__) if error is not None]
    formatted = "".join(traceback.format_exception(type(refusal), refusal, refusal.__traceback__))
    return "\n".join([str(refusal), repr(refusal), refusal.variable, refusal.rule, formatted, *behind])


def _canaries_in(shown: str) -> list[str]:
    """The canaries whose first CANARY_PREFIX_LENGTH characters `shown` holds, in any case."""
    return [canary for canary in CANARIES if canary[:CANARY_PREFIX_LENGTH].lower() in shown.lower()]


@pytest.mark.parametrize(
    "changes",
    [
        {"VH_HUB_TO_WORKER_KEYS": f"test-hub2wk-k1={HUB_KEY_HEX.upper()}"},
        {"VH_HUB_TO_WORKER_KEYS": f"test-hub2wk-k1={HUB_KEY_HEX[:-2]}"},
        {"VH_HUB_TO_WORKER_KEYS": f"BAD-ID={HUB_KEY_HEX}"},
        {"VH_HUB_TO_WORKER_KEYS": f"test-hub2wk-k1{HUB_KEY_HEX}"},
        {"VH_WORKER_TO_HUB_KEY": WORKER_KEY_HEX.upper()},
        {"VH_WORKER_TO_HUB_KEY": WORKER_KEY_HEX[:-2]},
        {"VH_WORKER_TO_HUB_KEY": HUB_KEY_HEX},
        {"VH_WORKER_TO_HUB_KEY": f" {WORKER_KEY_HEX}"},
        {"VH_STORAGE_DEFAULT_SECRET_ACCESS_KEY": f"{SECRET_ACCESS_KEY}\n"},
        {"VH_STORAGE_DEFAULT_ACCESS_KEY_ID": f"{ACCESS_KEY_ID} "},
        {"VH_CLAIM_URL": f"https://{ACCESS_KEY_ID}:{SECRET_ACCESS_KEY}@hub.test.invalid/claim"},
        {"VH_CLAIM_URL": f"https://hub.test.invalid:{SECRET_ACCESS_KEY}/claim"},
        {"VH_CLOCK_SKEW_S": SECRET_ACCESS_KEY},
        {"VH_STORAGE_DEFAULT_ENDPOINT": f"http://storage.test.invalid/{SECRET_ACCESS_KEY}"},
        {"VH_STORAGE_PROFILES": WORKER_KEY_HEX},
        {"VH_STORAGE_PROFILES": f"default,{WORKER_KEY_HEX}"},
    ],
    ids=[
        "hub-key-upper-case",
        "hub-key-short",
        "hub-key-id-bad",
        "hub-key-no-separator",
        "worker-key-upper-case",
        "worker-key-short",
        "worker-key-of-the-hub",
        "worker-key-leading-space",
        "secret-with-newline",
        "access-key-id-trailing-space",
        "claim-url-with-credentials",
        "claim-url-with-a-secret-for-its-port",
        "clock-skew-holding-a-secret",
        "endpoint-plain-http-with-secret-path",
        "profile-list-holding-a-key",
        "profile-list-holding-a-key-as-its-second-entry",
    ],
)
def test_from_env_when_a_refused_value_holds_a_secret_should_never_show_it(changes):
    # Act
    refusal = _refusal(_env(**changes))

    # Assert
    assert _canaries_in(_everything_shown(refusal)) == []


def test_settings_when_shown_should_hold_no_key_and_no_secret():
    # Arrange
    settings = WorkerSettings.from_env(VALID_ENV)

    # Act
    shown = repr(settings) + str(settings) + repr(settings.storage_profile("default"))

    # Assert
    secrets = list(CANARIES) + [repr(bytes.fromhex(key)) for key in (HUB_KEY_HEX, SECOND_HUB_KEY_HEX, WORKER_KEY_HEX)]
    assert [secret for secret in secrets if secret in shown] == []
    assert "worker:runpod-test" in shown and "test-wk2hub-k1" in shown


# --- the v1 configuration stays apart ----------------------------------------------------------


def test_from_env_when_loading_should_read_only_vh_variables():
    # Arrange
    env = _RecordingEnv(_env(WEBHOOK_SECRET="v1-secret", HUB_CALLBACK_URL="https://v1.test.invalid"))

    # Act
    WorkerSettings.from_env(env)

    # Assert
    assert sorted(variable for variable in env.read if not variable.startswith("VH_")) == []


def test_from_env_when_only_the_v1_variables_are_set_should_refuse_rather_than_fall_back():
    # Arrange
    env = {"WEBHOOK_SECRET": WORKER_KEY_HEX, "FFMPEG_ENCODER": "libx264"}

    # Act
    refusal = _refusal(env)

    # Assert
    assert refusal.variable == "VH_ACCEPTED_PROTOCOL_VERSIONS"
