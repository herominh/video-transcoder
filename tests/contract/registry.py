"""Contract constants: message kinds, accepted versions, byte limits, reason codes and artifact kinds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Mapping

from . import files


class MessageKind(str, Enum):
    """The six kinds of the contract; generation.manifest is a stored document, the others are messages."""

    TRANSCODE_REQUEST = "transcode.request"
    TRANSCODE_PROGRESS = "transcode.progress"
    TRANSCODE_RESULT_COMPLETED = "transcode.result.completed"
    TRANSCODE_RESULT_FAILED = "transcode.result.failed"
    HUB_ERROR = "hub.error"
    GENERATION_MANIFEST = "generation.manifest"

    @property
    def schema_file(self) -> str:
        return _SCHEMA_FILES[self]


_SCHEMA_FILES: Mapping[MessageKind, str] = {
    MessageKind.TRANSCODE_REQUEST: "schemas/transcode-request.schema.json",
    MessageKind.TRANSCODE_PROGRESS: "schemas/transcode-progress.schema.json",
    MessageKind.TRANSCODE_RESULT_COMPLETED: "schemas/transcode-result-completed.schema.json",
    MessageKind.TRANSCODE_RESULT_FAILED: "schemas/transcode-result-failed.schema.json",
    MessageKind.HUB_ERROR: "schemas/hub-error.schema.json",
    MessageKind.GENERATION_MANIFEST: "schemas/generation-manifest.schema.json",
}

KNOWN_KINDS: frozenset[str] = frozenset(kind.value for kind in MessageKind)


class Role(str, Enum):
    """Who validates; a *_sender role is the self-check before sending."""

    HUB_RECEIVER = "hub_receiver"
    HUB_SENDER = "hub_sender"
    WORKER_RECEIVER = "worker_receiver"
    WORKER_SENDER = "worker_sender"


class Channel(str, Enum):
    """How the payload arrived; selects the freshness rule. `storage` = a manifest read."""

    PUSH = "push"
    POLL = "poll"
    DISPATCH = "dispatch"
    STORAGE = "storage"


class VersionLine(str, Enum):
    PROTOCOL = "protocol"
    MANIFEST = "manifest"


class VersionChannel(str, Enum):
    DRAFT = "draft"
    LIVE = "live"


@dataclass(frozen=True)
class ContractRegistry:
    """Read-only view of versions.json, limits.json, reason-codes.json and artifact-kinds.json."""

    versions: Mapping[str, Any]
    limits: Mapping[str, Any]
    reason_codes: tuple[Mapping[str, Any], ...]
    artifact_kinds: Mapping[str, Any]

    @classmethod
    def load(cls) -> ContractRegistry:
        return cls(
            versions=files.read_json("versions.json"),
            limits=files.read_json("limits.json"),
            reason_codes=tuple(files.read_json("reason-codes.json")["reason_codes"]),
            artifact_kinds=files.read_json("artifact-kinds.json"),
        )

    def accepted_versions(self, line: VersionLine, channel: VersionChannel) -> tuple[str, ...]:
        return tuple(self.versions[VersionLine(line).value][VersionChannel(channel).value])

    def max_bytes(self, kind: MessageKind | str) -> int:
        return int(self.limits["max_bytes"][MessageKind(kind).value])

    def carrier_max_bytes(self, carrier: str) -> int:
        carrier_limits = self.limits["carrier_max_bytes"]
        if carrier not in carrier_limits:
            raise KeyError(f"unknown carrier {carrier!r}")
        return int(carrier_limits[carrier])

    @property
    def max_json_depth(self) -> int:
        return int(self.limits["max_json_depth"])

    @property
    def max_tokens(self) -> int:
        return int(self.limits["max_tokens"])

    @property
    def clock_skew_seconds(self) -> int:
        return int(self.limits["clock_skew_seconds"])

    @property
    def max_segments_per_rendition(self) -> int:
        return int(self.limits["max_segments_per_rendition"])

    def known_versions(self, line: VersionLine) -> frozenset[str]:
        """Every version the tree defines for a line, draft and live together."""
        return frozenset(
            self.accepted_versions(line, VersionChannel.DRAFT) + self.accepted_versions(line, VersionChannel.LIVE)
        )

    def reason_code_values(self) -> tuple[str, ...]:
        return tuple(entry["code"] for entry in self.reason_codes)

    def active_artifact_kinds(self, manifest_version: str) -> tuple[str, ...]:
        return tuple(self.artifact_kinds["manifest_versions"][manifest_version]["active"])

    def reserved_artifact_kinds(self, manifest_version: str) -> tuple[str, ...]:
        return tuple(self.artifact_kinds["manifest_versions"][manifest_version]["reserved"])


@lru_cache(maxsize=1)
def load_registry() -> ContractRegistry:
    """The registry of the tree next to this package (read once per process)."""
    return ContractRegistry.load()
