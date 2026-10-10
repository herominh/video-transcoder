"""The worker's trusted configuration for contract v2, read from the environment and refused unless whole.

Everything here comes from the worker's own configuration, never from a message:

    VH_ACCEPTED_PROTOCOL_VERSIONS   comma list of protocol versions versions.json defines; a draft is
                                    accepted only when listed here, and there is no default
    VH_WORKER_AUDIENCE              this worker's audience, `worker:<name>` (the contract audience)
    VH_HUB_TO_WORKER_KEYS           `kid=hex,kid=hex`: 1 to 8 keys the Hub signs with, distinct ids
    VH_WORKER_TO_HUB_KEY_ID         the id of the one key this worker signs with (not a Hub-to-worker id)
    VH_WORKER_TO_HUB_KEY            that key (not the bytes of a Hub-to-worker key: the check is on the
                                    exact bytes, the paste error it exists for)
    VH_CLAIM_URL, VH_CALLBACK_URL   absolute https URLs without userinfo or fragment (README 15.1, rule 2)
    VH_CLOCK_SKEW_S                 an integer within the trusted context's clock_skew_s bounds;
                                    default limits.json clock_skew_seconds
    VH_STORAGE_PROFILES             comma list of credential_ref names; for each, with NAME the ref in
                                    upper case and `-` as `_`: VH_STORAGE_<NAME>_ENDPOINT (https URL),
                                    VH_STORAGE_<NAME>_REGION (default `auto`), VH_STORAGE_<NAME>_ACCESS_KEY_ID
                                    and VH_STORAGE_<NAME>_SECRET_ACCESS_KEY

A key id is the contract `key_id`, a key at least 32 bytes in lowercase hex. Every value is printable
ASCII without a leading or trailing space. A refusal (SettingsInvalid) names the variable and the rule,
never the value, and neither repr() nor str() of the settings shows a key or a secret. The name of a
storage profile's variable holds an entry of VH_STORAGE_PROFILES, itself a value: its refusal names
VH_STORAGE_PROFILES, the entry's position and the variable as `VH_STORAGE_<NAME>_...`. The v1 Settings
(core/config.py) and its variables are separate and untouched.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit

from . import schema
from .registry import VersionLine, load_registry

VERSIONS_VARIABLE = "VH_ACCEPTED_PROTOCOL_VERSIONS"
AUDIENCE_VARIABLE = "VH_WORKER_AUDIENCE"
HUB_KEYS_VARIABLE = "VH_HUB_TO_WORKER_KEYS"
WORKER_KEY_ID_VARIABLE = "VH_WORKER_TO_HUB_KEY_ID"
WORKER_KEY_VARIABLE = "VH_WORKER_TO_HUB_KEY"
CLAIM_URL_VARIABLE = "VH_CLAIM_URL"
CALLBACK_URL_VARIABLE = "VH_CALLBACK_URL"
CLOCK_SKEW_VARIABLE = "VH_CLOCK_SKEW_S"
STORAGE_PROFILES_VARIABLE = "VH_STORAGE_PROFILES"
STORAGE_VARIABLE_PREFIX = "VH_STORAGE_"
STORAGE_ENDPOINT_SUFFIX = "_ENDPOINT"
STORAGE_REGION_SUFFIX = "_REGION"
STORAGE_ACCESS_KEY_ID_SUFFIX = "_ACCESS_KEY_ID"
STORAGE_SECRET_ACCESS_KEY_SUFFIX = "_SECRET_ACCESS_KEY"

LIST_SEPARATOR = ","
KEY_SEPARATOR = "="
# The trusted context's known_key_ids holds at most 8 ids (trusted-context.schema.json).
MAX_HUB_TO_WORKER_KEYS = 8
MIN_KEY_BYTES = 32
DEFAULT_REGION = "auto"
WORKER_AUDIENCE_PREFIX = "worker:"
HTTPS_SCHEME = "https"
MIN_PORT = 1
MAX_PORT = 65535
STORAGE_NAME_PLACEHOLDER = "<NAME>"

_PRINTABLE_ASCII = re.compile(r"[\x20-\x7e]*")
# Printable ASCII that no URL holds (RFC 3986); URL parsers disagree about some of them (a backslash).
_NOT_URL_CHARACTERS = re.compile(r'[ "<>\\^`{|}]')
_AUTHORITY = re.compile(r"(?P<host>\[[0-9A-Fa-f:.]{2,45}\]|[A-Za-z0-9.-]{1,253})(?::(?P<port>[0-9]{1,5}))?")
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
# A host whose last label is a number is an IPv4 address to an HTTP client, whatever form it takes.
_NUMERIC_LABEL = re.compile(r"[0-9]+|0[xX][0-9A-Fa-f]*")
_LOWERCASE_HEX_PAIRS = re.compile(r"(?:[0-9a-f]{2})+")
_DECIMAL_INTEGER = re.compile(r"0|[1-9][0-9]*")
_COMMON_DEFINITIONS = schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/"
_CLOCK_SKEW_DEFINITION = schema.SCHEMA_BASE_URI + "schemas/trusted-context.schema.json#/properties/clock_skew_s"


class SettingsInvalid(ValueError):
    """A variable is missing or breaks its rule; the message names both, never the value."""

    def __init__(self, variable: str, rule: str) -> None:
        super().__init__(f"{variable} {rule}")
        self.variable = variable
        self.rule = rule


class UnknownCredentialRef(LookupError):
    """A dispatch names a credential_ref this worker holds no storage profile for (contract section 7)."""

    ERROR_CLASS = "configuration_error"
    ERROR_CODE = "unknown_credential_ref"

    def __init__(self, credential_ref: object) -> None:
        super().__init__(f"no storage profile is configured for credential_ref {credential_ref!r}")
        self.credential_ref = credential_ref


@dataclass(frozen=True)
class StorageProfile:
    """The endpoint and credentials one credential_ref names; repr() shows neither credential."""

    credential_ref: str
    endpoint: str
    region: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)


@dataclass(frozen=True)
class WorkerSettings:
    """The validated configuration; build it with from_env(). repr() shows no key and no secret."""

    accepted_protocol_versions: tuple[str, ...]
    worker_audience: str
    hub_to_worker_keys: Mapping[str, bytes] = field(repr=False)
    worker_to_hub_key_id: str
    worker_to_hub_key: bytes = field(repr=False)
    claim_url: str
    callback_url: str
    clock_skew_s: int
    storage_profiles: Mapping[str, StorageProfile]

    @property
    def hub_to_worker_key_ids(self) -> tuple[str, ...]:
        return tuple(self.hub_to_worker_keys)

    def storage_profile(self, credential_ref: str) -> StorageProfile:
        """The profile a dispatch's credential_ref names; UnknownCredentialRef when this worker holds none."""
        profile = self.storage_profiles.get(credential_ref) if isinstance(credential_ref, str) else None
        if profile is None:
            raise UnknownCredentialRef(credential_ref)
        return profile

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> WorkerSettings:
        """Read and check every variable, in the order the module lists them; the first problem raises SettingsInvalid."""
        if not isinstance(env, Mapping):
            raise TypeError("from_env() takes the environment as a mapping")
        accepted_protocol_versions = _accepted_versions(env)
        worker_audience = _worker_audience(env)
        hub_to_worker_keys = _hub_to_worker_keys(env)
        worker_to_hub_key_id = _key_id(WORKER_KEY_ID_VARIABLE, _text(env, WORKER_KEY_ID_VARIABLE))
        if worker_to_hub_key_id in hub_to_worker_keys:
            raise SettingsInvalid(WORKER_KEY_ID_VARIABLE, "must not also name a Hub-to-worker key")
        worker_to_hub_key = _key(WORKER_KEY_VARIABLE, _text(env, WORKER_KEY_VARIABLE))
        if worker_to_hub_key in hub_to_worker_keys.values():
            raise SettingsInvalid(WORKER_KEY_VARIABLE, "must differ from every Hub-to-worker key (keys are per direction)")
        return cls(
            accepted_protocol_versions=accepted_protocol_versions,
            worker_audience=worker_audience,
            hub_to_worker_keys=MappingProxyType(hub_to_worker_keys),
            worker_to_hub_key_id=worker_to_hub_key_id,
            worker_to_hub_key=worker_to_hub_key,
            claim_url=_https_url(CLAIM_URL_VARIABLE, _text(env, CLAIM_URL_VARIABLE)),
            callback_url=_https_url(CALLBACK_URL_VARIABLE, _text(env, CALLBACK_URL_VARIABLE)),
            clock_skew_s=_clock_skew(env),
            storage_profiles=MappingProxyType(_storage_profiles(env)),
        )


def _satisfies(definition_uri: str, value: Any) -> bool:
    return not schema.error_pointers_for_schema({"$ref": definition_uri}, value)


def _text(
    env: Mapping[str, str], variable: str, default: str | None = None, *, named: tuple[str, str] | None = None
) -> str:
    """The variable's value, checked. `named` is the (variable, lead-in) a refusal shows instead of `variable`."""
    shown, where = named if named is not None else (variable, "")
    value = env.get(variable)
    if value is None:
        if default is None:
            raise SettingsInvalid(shown, f"{where}is required")
        return default
    if not isinstance(value, str):
        raise SettingsInvalid(shown, f"{where}must be a string")
    if value == "":
        raise SettingsInvalid(shown, f"{where}must not be empty")
    if _PRINTABLE_ASCII.fullmatch(value) is None:
        raise SettingsInvalid(shown, f"{where}must hold printable ASCII only")
    if value != value.strip():
        raise SettingsInvalid(shown, f"{where}must not start or end with a space")
    return value


def _entries(env: Mapping[str, str], variable: str) -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in _text(env, variable).split(LIST_SEPARATOR))
    if "" in entries:
        raise SettingsInvalid(variable, "has an empty entry")
    return entries


def _key_id(variable: str, key_id: str, where: str = "") -> str:
    if not _satisfies(_COMMON_DEFINITIONS + "key_id", key_id):
        raise SettingsInvalid(variable, f"{where}the key id does not match the contract key_id pattern")
    return key_id


def _key(variable: str, key_hex: str, where: str = "") -> bytes:
    if _LOWERCASE_HEX_PAIRS.fullmatch(key_hex) is None:
        raise SettingsInvalid(variable, f"{where}the key must be lowercase hex digits in pairs")
    key = bytes.fromhex(key_hex)
    if len(key) < MIN_KEY_BYTES:
        raise SettingsInvalid(variable, f"{where}the key must be at least {MIN_KEY_BYTES} bytes")
    return key


def _accepted_versions(env: Mapping[str, str]) -> tuple[str, ...]:
    versions = _entries(env, VERSIONS_VARIABLE)
    defined = load_registry().known_versions(VersionLine.PROTOCOL)
    for position, version in enumerate(versions, start=1):
        if version not in defined:
            raise SettingsInvalid(VERSIONS_VARIABLE, f"entry {position} is not a protocol version versions.json defines")
    if len(set(versions)) != len(versions):
        raise SettingsInvalid(VERSIONS_VARIABLE, "lists a version twice")
    return versions


def _worker_audience(env: Mapping[str, str]) -> str:
    audience = _text(env, AUDIENCE_VARIABLE)
    if not audience.startswith(WORKER_AUDIENCE_PREFIX) or not _satisfies(_COMMON_DEFINITIONS + "audience", audience):
        raise SettingsInvalid(AUDIENCE_VARIABLE, "must be `worker:<name>` matching the contract audience pattern")
    return audience


def _hub_to_worker_keys(env: Mapping[str, str]) -> dict[str, bytes]:
    entries = _entries(env, HUB_KEYS_VARIABLE)
    if len(entries) > MAX_HUB_TO_WORKER_KEYS:
        raise SettingsInvalid(HUB_KEYS_VARIABLE, f"holds more than {MAX_HUB_TO_WORKER_KEYS} keys")
    keys: dict[str, bytes] = {}
    for position, entry in enumerate(entries, start=1):
        where = f"entry {position}: "
        key_id, separator, key_hex = entry.partition(KEY_SEPARATOR)
        if not separator:
            raise SettingsInvalid(HUB_KEYS_VARIABLE, f"{where}is not `kid=hex`")
        _key_id(HUB_KEYS_VARIABLE, key_id, where)
        if key_id in keys:
            raise SettingsInvalid(HUB_KEYS_VARIABLE, f"{where}repeats a key id")
        keys[key_id] = _key(HUB_KEYS_VARIABLE, key_hex, where)
    return keys


def _https_url(variable: str, url: str, where: str = "") -> str:
    """An absolute https URL whose host is a host name (labels of letters, digits and inner hyphens, no
    trailing dot), a dotted-quad IPv4 address or a bracketed IPv6 literal, with an optional port; no
    userinfo, no fragment and no character a URL cannot hold."""
    parts = _split_url(url)
    if parts is None or parts.scheme != HTTPS_SCHEME or _NOT_URL_CHARACTERS.search(url) is not None:
        raise SettingsInvalid(variable, f"{where}must be an absolute https URL")
    if "@" in parts.netloc:
        raise SettingsInvalid(variable, f"{where}must not carry userinfo")
    if "#" in url:
        raise SettingsInvalid(variable, f"{where}must not carry a fragment")
    if not _is_host_with_optional_port(parts.netloc):
        raise SettingsInvalid(variable, f"{where}must be an absolute https URL")
    return url


def _split_url(url: str) -> SplitResult | None:
    """None for what urlsplit() refuses; its error quotes the value, so it never reaches a refusal."""
    try:
        return urlsplit(url)
    except ValueError:
        return None


def _is_host_with_optional_port(authority: str) -> bool:
    match = _AUTHORITY.fullmatch(authority)
    if match is None:
        return False
    host, port = match.group("host"), match.group("port")
    if port is not None and not MIN_PORT <= int(port) <= MAX_PORT:
        return False
    if host.startswith("["):
        return _is_address(ipaddress.IPv6Address, host[1:-1])
    labels = host.split(".")
    if _NUMERIC_LABEL.fullmatch(labels[-1]) is not None:
        return _is_address(ipaddress.IPv4Address, host)
    return all(_DNS_LABEL.fullmatch(label) is not None for label in labels)


def _is_address(address_type: type[ipaddress.IPv4Address] | type[ipaddress.IPv6Address], text: str) -> bool:
    try:
        address_type(text)
    except ValueError:
        return False
    return True


def _clock_skew(env: Mapping[str, str]) -> int:
    if CLOCK_SKEW_VARIABLE not in env:
        return load_registry().clock_skew_seconds
    text = _text(env, CLOCK_SKEW_VARIABLE)
    rule = "must be an integer within the trusted context's clock_skew_s bounds"
    if _DECIMAL_INTEGER.fullmatch(text) is None:
        raise SettingsInvalid(CLOCK_SKEW_VARIABLE, rule)
    try:
        seconds = int(text)
    except ValueError:  # more digits than int() converts
        raise SettingsInvalid(CLOCK_SKEW_VARIABLE, rule) from None
    if not _satisfies(_CLOCK_SKEW_DEFINITION, seconds):
        raise SettingsInvalid(CLOCK_SKEW_VARIABLE, rule)
    return seconds


def storage_variable_name(credential_ref: str) -> str:
    """The NAME of a credential_ref's variables: the ref in upper case, `-` written `_`."""
    return credential_ref.upper().replace("-", "_")


def _storage_profiles(env: Mapping[str, str]) -> dict[str, StorageProfile]:
    profiles: dict[str, StorageProfile] = {}
    positions_by_name: dict[str, int] = {}
    for position, credential_ref in enumerate(_entries(env, STORAGE_PROFILES_VARIABLE), start=1):
        if not _satisfies(_COMMON_DEFINITIONS + "credential_ref", credential_ref):
            raise SettingsInvalid(STORAGE_PROFILES_VARIABLE, f"entry {position} is not a contract credential_ref")
        if credential_ref in profiles:
            raise SettingsInvalid(STORAGE_PROFILES_VARIABLE, f"entry {position} repeats a credential_ref")
        name = storage_variable_name(credential_ref)
        if name in positions_by_name:
            raise SettingsInvalid(
                STORAGE_PROFILES_VARIABLE,
                f"entries {positions_by_name[name]} and {position} map to the same VH_STORAGE_<NAME>_ variables",
            )
        positions_by_name[name] = position
        profiles[credential_ref] = _storage_profile(env, credential_ref, position)
    return profiles


def _storage_profile(env: Mapping[str, str], credential_ref: str, position: int) -> StorageProfile:
    variable_stem = STORAGE_VARIABLE_PREFIX + storage_variable_name(credential_ref)

    def lead_in(suffix: str) -> str:
        # The real name holds the entry, a value an operator can mistype a secret into: never shown.
        return f"entry {position}: its {STORAGE_VARIABLE_PREFIX}{STORAGE_NAME_PLACEHOLDER}{suffix} variable "

    def text(suffix: str, default: str | None = None) -> str:
        return _text(env, variable_stem + suffix, default, named=(STORAGE_PROFILES_VARIABLE, lead_in(suffix)))

    return StorageProfile(
        credential_ref=credential_ref,
        endpoint=_https_url(
            STORAGE_PROFILES_VARIABLE, text(STORAGE_ENDPOINT_SUFFIX), lead_in(STORAGE_ENDPOINT_SUFFIX)
        ),
        region=text(STORAGE_REGION_SUFFIX, DEFAULT_REGION),
        access_key_id=text(STORAGE_ACCESS_KEY_ID_SUFFIX),
        secret_access_key=text(STORAGE_SECRET_ACCESS_KEY_SUFFIX),
    )
