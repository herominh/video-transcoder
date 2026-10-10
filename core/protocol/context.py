"""The trusted context: what the validating side knows from its own records, never from the message.

Construction validates, in this order: the shape (trusted-context.schema.json), a calendar-valid
`now` and accepted versions that versions.json defines -> ContextInvalid; every accepted kind
supported for the role on this channel -> ContextInvalid; every field each accepted (role, kind)
requires present and non-null -> ContextIncomplete. Both are programming errors, never a verdict on
a message.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from . import schema
from .registry import VersionLine, load_registry
from .semantic import parse_timestamp_ms, required_context_fields, supported_channels

CONTEXT_SCHEMA_FILE = "schemas/trusted-context.schema.json"
_EXPECT_PREFIX = "expect."
_MS_PER_SECOND = 1_000


class ContextInvalid(ValueError):
    """The context is malformed or pairs a role with a kind or channel the contract does not define."""


class ContextIncomplete(Exception):
    """A field that the (role, kind) requires is missing or null."""

    def __init__(self, role: str, kind: str, field: str) -> None:
        super().__init__(f"context for role {role!r} and kind {kind!r} lacks required field {field!r}")
        self.role = role
        self.kind = kind
        self.field = field


def _lookup(data: Mapping[str, Any], dotted_field: str) -> Any:
    if not dotted_field.startswith(_EXPECT_PREFIX):
        return data.get(dotted_field)
    expect = data.get("expect")
    if not isinstance(expect, Mapping):
        return None
    return expect.get(dotted_field[len(_EXPECT_PREFIX):])


def _refuse_undefined_versions(snapshot: Mapping[str, Any]) -> None:
    """A context cannot create protocol support: every accepted version must be one versions.json defines."""
    registry = load_registry()
    for field, line in (
        ("accepted_protocol_versions", VersionLine.PROTOCOL),
        ("accepted_manifest_versions", VersionLine.MANIFEST),
    ):
        undefined = set(snapshot.get(field) or ()) - registry.known_versions(line)
        if undefined:
            raise ContextInvalid(f"{field} names versions the contract does not define: {sorted(undefined)}")


def _tuple_or_none(value: Any) -> tuple[Any, ...] | None:
    return None if value is None else tuple(value)


@dataclass(frozen=True)
class TrustedContext:
    role: str
    channel: str
    accepted_message_kinds: tuple[str, ...]
    now: str | None
    clock_skew_s: int | float | None
    accepted_protocol_versions: tuple[str, ...] | None
    accepted_manifest_versions: tuple[str, ...] | None
    expected_audience: str | None
    known_key_ids: tuple[str, ...] | None
    expect: Mapping[str, Any] | None
    last_event_seq: int | float | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrustedContext:
        if not isinstance(data, Mapping):
            raise ContextInvalid("a trusted context is a JSON object")
        snapshot = copy.deepcopy(dict(data))
        pointers = schema.error_pointers(CONTEXT_SCHEMA_FILE, snapshot)
        if pointers:
            raise ContextInvalid(f"context violates {CONTEXT_SCHEMA_FILE} at {pointers[0]!r}")
        if snapshot.get("now") is not None and parse_timestamp_ms(snapshot["now"]) is None:
            raise ContextInvalid("context `now` is not a real calendar instant")
        _refuse_undefined_versions(snapshot)

        role, channel = snapshot["role"], snapshot["channel"]
        kinds = tuple(snapshot["accepted_message_kinds"])
        for kind in kinds:
            channels = supported_channels(role, kind)
            if channels is None:
                raise ContextInvalid(f"role {role!r} never validates {kind!r}")
            if channel not in channels:
                raise ContextInvalid(f"role {role!r} validates {kind!r} only on {channels}, not {channel!r}")
        for kind in kinds:
            for field in required_context_fields(role, kind):
                if _lookup(snapshot, field) is None:
                    raise ContextIncomplete(role, kind, field)

        return cls(
            role=role,
            channel=channel,
            accepted_message_kinds=kinds,
            now=snapshot.get("now"),
            clock_skew_s=snapshot.get("clock_skew_s"),
            accepted_protocol_versions=_tuple_or_none(snapshot.get("accepted_protocol_versions")),
            accepted_manifest_versions=_tuple_or_none(snapshot.get("accepted_manifest_versions")),
            expected_audience=snapshot.get("expected_audience"),
            known_key_ids=_tuple_or_none(snapshot.get("known_key_ids")),
            expect=snapshot.get("expect"),
            last_event_seq=snapshot.get("last_event_seq"),
        )

    def value(self, dotted_field: str) -> Any:
        """A context field by name; `expect.<name>` reads inside `expect`. None when absent."""
        if dotted_field.startswith(_EXPECT_PREFIX):
            return None if self.expect is None else self.expect.get(dotted_field[len(_EXPECT_PREFIX):])
        return getattr(self, dotted_field)

    @property
    def now_ms(self) -> int:
        now_ms = parse_timestamp_ms(self.now)
        if now_ms is None:
            raise ContextIncomplete(self.role, "any", "now")
        return now_ms

    @property
    def skew_ms(self) -> int | float:
        if self.clock_skew_s is None:
            raise ContextIncomplete(self.role, "any", "clock_skew_s")
        return self.clock_skew_s * _MS_PER_SECOND
