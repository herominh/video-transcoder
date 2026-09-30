"""Reason codes and validation layers of the transcode contract v2 (mirrors reason-codes.json)."""

from __future__ import annotations

from enum import Enum


class Layer(str, Enum):
    """The pipeline layer that produced a rejection, in evaluation order."""

    SIZE = "size"
    PARSE = "parse"
    VERSION = "version"
    SCHEMA = "schema"
    SEMANTIC = "semantic"


class ReasonCode(str, Enum):
    """Every code of reason-codes.json, in file order (reserved codes included)."""

    PAYLOAD_TOO_LARGE = "payload_too_large"
    MALFORMED_JSON = "malformed_json"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNKNOWN_MESSAGE_KIND = "unknown_message_kind"
    UNEXPECTED_MESSAGE_KIND = "unexpected_message_kind"
    SCHEMA_VIOLATION = "schema_violation"
    INVALID_TIMESTAMP = "invalid_timestamp"
    AUDIENCE_MISMATCH = "audience_mismatch"
    UNKNOWN_KEY_ID = "unknown_key_id"
    TIMESTAMP_IN_FUTURE = "timestamp_in_future"
    TIMESTAMP_OUT_OF_TOLERANCE = "timestamp_out_of_tolerance"
    DISPATCH_EXPIRED = "dispatch_expired"
    IDENTITY_ALIASES_VIDEO_UUID = "identity_aliases_video_uuid"
    GENERATION_EXECUTION_MISMATCH = "generation_execution_mismatch"
    ORG_MISMATCH = "org_mismatch"
    VIDEO_MISMATCH = "video_mismatch"
    CELL_MISMATCH = "cell_mismatch"
    ATTEMPT_MISMATCH = "attempt_mismatch"
    DISPATCH_MISMATCH = "dispatch_mismatch"
    EXECUTION_MISMATCH = "execution_mismatch"
    SOURCE_MISMATCH = "source_mismatch"
    STORAGE_LOCATION_MISMATCH = "storage_location_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    ENCRYPTION_MISMATCH = "encryption_mismatch"
    STALE_PLACEMENT_EPOCH = "stale_placement_epoch"
    STALE_LIFECYCLE_REVISION = "stale_lifecycle_revision"
    STALE_EXECUTION_FENCE = "stale_execution_fence"
    PLACEMENT_EPOCH_AHEAD = "placement_epoch_ahead"
    LIFECYCLE_REVISION_AHEAD = "lifecycle_revision_ahead"
    STALE_EVENT_SEQUENCE = "stale_event_sequence"
    DUPLICATE_ARTIFACT_PATH = "duplicate_artifact_path"
    ARTIFACT_INVENTORY_MISMATCH = "artifact_inventory_mismatch"
    DANGLING_REFERENCE = "dangling_reference"
    THUMBNAIL_DIAGNOSTIC_MISSING = "thumbnail_diagnostic_missing"
    RENDITION_NOT_REQUESTED = "rendition_not_requested"
    LIMIT_EXCEEDED = "limit_exceeded"
    INVALID_SIGNATURE = "invalid_signature"
    DUPLICATE_MESSAGE = "duplicate_message"
    CONFLICTING_TERMINAL_RESULT = "conflicting_terminal_result"
    MANIFEST_DIGEST_MISMATCH = "manifest_digest_mismatch"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
