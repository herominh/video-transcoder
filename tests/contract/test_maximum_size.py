"""The worst-case instance of every kind passes its schema and fits its byte limit.

Worst case = every string at maxLength made of the costliest characters, every array at maxItems,
every integer at its widest bound, every optional member present, measured compact with `/` written
as `\\/` (PHP's json_encode default). A walk over each schema proves the instance is maximal, so a
loosened bound fails here instead of silently outgrowing a byte limit. The manifest worst case (about
7.3 MB) is built here, never committed; the generator of the tree reuses worst_cases() to set
limits.json max_tokens.
"""

import json
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin

import pytest

from tests.contract import files, gate, pipeline, schema
from tests.contract.context import TrustedContext
from tests.contract.reasons import Layer, ReasonCode
from tests.contract.registry import MessageKind, load_registry

UUID = "f" * 8 + "-" + "f" * 4 + "-" + "f" * 4 + "-" + "f" * 4 + "-" + "f" * 12
TIMESTAMP = "2030-12-31T23:59:59.999Z"
VERSION = "9999.9999.9999-draft"
KEY_ID = "k" * 48
CELL_ID = "c" * 40
AUDIENCE = "worker:" + "c" * 40
MAX_SAFE_INTEGER = 2**53 - 1
MAX_SOURCE_BYTES = 32_212_254_720
MAX_DURATION_MS = 21_600_000
MAX_ARTIFACTS = 25_209
REL_PATH = "a" * 32 + "/" + "a" * 31 + "/" + "a" * 31 + "/" + "a" * 31
OBJECT_KEY = "a/" * 255 + "aa"
BOOTSTRAP_TOKEN = "t" * files.read_json("schemas/common.schema.json")["$defs"]["bootstrap_token"]["maxLength"]
SHA256 = "f" * 64
CODECS = "c" * 32 + "," + "c" * 31
RENDITIONS = ["2160p", "1440p", "1080p", "720p", "480p", "360p", "240p"]


def escaped_text(length: int) -> str:
    """Printable ASCII that JSON must escape: two bytes per character."""
    return '"' * length


def worst_cases() -> dict[MessageKind, dict[str, Any]]:
    ext = {("x_" + str(index) + "y" * 29)[:32]: escaped_text(64) for index in range(8)}
    envelope = lambda kind: {  # noqa: E731
        "protocol_version": VERSION,
        "message_kind": kind,
        "message_id": UUID,
        "sent_at": TIMESTAMP,
        "audience": AUDIENCE,
        "key_id": KEY_ID,
    }
    identity = {
        "org_uuid": UUID,
        "video_uuid": UUID,
        "owning_cell_id": CELL_ID,
        "placement_epoch": MAX_SAFE_INTEGER,
        "lifecycle_revision": MAX_SAFE_INTEGER,
        "attempt_id": UUID,
        "dispatch_id": UUID,
    }
    identity_execution = {**identity, "execution_id": UUID, "execution_fence": MAX_SAFE_INTEGER}
    profile = {"id": "p" * 32, "version": 65_535, "sha256": SHA256}
    encryption = {"mode": "aes-128", "media_key_id": UUID}
    diagnostics = [{"code": "d" * 48, "detail": escaped_text(200)} for _ in range(8)]
    encoder = {
        "image_digest": "sha256:" + SHA256,
        "ffmpeg_version": escaped_text(64),
        "video_encoder": "h264_nvenc",
        "preset": "p" * 32,
    }
    source_read = {"source_id": UUID, "bytes_read": MAX_SOURCE_BYTES, "etag_observed": "e" * 128}
    media = {
        "duration_ms": MAX_DURATION_MS,
        "max_width": 3_840,
        "max_height": 3_840,
        "has_audio": False,
        "rendition_count": 7,
    }
    longest_code = max((code.value for code in ReasonCode), key=len)
    rendition = {
        "name": "2160p",
        "width": 3_840,
        "height": 3_840,
        "bandwidth_bps": 999_999_999,
        "average_bandwidth_bps": 999_999_999,
        "codecs": CODECS,
        "playlist_path": REL_PATH,
    }
    # hls_media_playlist (18 characters) with its required `rendition` outweighs hls_master_playlist (19, no
    # rendition), and every artifact then carries the most tokens.
    artifact = {
        "path": REL_PATH,
        "kind": "hls_media_playlist",
        "size_bytes": 4_294_967_296,
        "sha256": SHA256,
        "rendition": "2160p",
    }
    return {
        MessageKind.TRANSCODE_REQUEST: {
            **envelope("transcode.request"),
            "identity": identity,
            "source": {
                "source_id": UUID,
                "location_id": UUID,
                "object_key": OBJECT_KEY,
                "size_bytes": MAX_SOURCE_BYTES,
                "etag": "e" * 128,
            },
            "output": {"location_id": UUID},
            "profile": profile,
            "renditions": RENDITIONS,
            "encryption": encryption,
            "limits": {
                "max_source_bytes": MAX_SOURCE_BYTES,
                "max_source_duration_ms": MAX_DURATION_MS,
                "max_output_bytes": 1_099_511_627_776,
                "max_artifact_count": MAX_ARTIFACTS,
                "max_wall_time_ms": 86_400_000,
            },
            "claim": {"bootstrap_token": BOOTSTRAP_TOKEN, "expires_at": TIMESTAMP},
        },
        MessageKind.TRANSCODE_PROGRESS: {
            **envelope("transcode.progress"),
            "identity": identity_execution,
            "event_seq": 2_147_483_647,
            "stage": "downloading",
            "stage_progress_pct": 100,
            "message": escaped_text(200),
            "counters": {
                "renditions_total": 7,
                "renditions_done": 7,
                "bytes_downloaded": MAX_SAFE_INTEGER,
                "bytes_uploaded": MAX_SAFE_INTEGER,
            },
            "ext": ext,
        },
        MessageKind.TRANSCODE_RESULT_COMPLETED: {
            **envelope("transcode.result.completed"),
            "identity": identity_execution,
            "generation_id": UUID,
            "manifest": {
                "location_id": UUID,
                "path": REL_PATH,
                "sha256": SHA256,
                "size_bytes": 8_388_608,
                "artifact_count": MAX_ARTIFACTS,
                "total_bytes": 1_099_511_627_776,
            },
            "source": source_read,
            "profile": profile,
            "media": media,
            "encoder": encoder,
            "diagnostics": diagnostics,
            "ext": ext,
        },
        MessageKind.TRANSCODE_RESULT_FAILED: {
            **envelope("transcode.result.failed"),
            "identity": identity_execution,
            "error": {
                "class": "source_integrity_mismatch",
                "code": "d" * 48,
                "retryable": False,
                "stage": "downloading",
                "detail": escaped_text(1_000),
            },
            "source": source_read,
            "encoder": encoder,
            "diagnostics": diagnostics,
            "ext": ext,
        },
        MessageKind.HUB_ERROR: {
            "protocol_version": VERSION,
            "message_kind": "hub.error",
            "message_id": UUID,
            "sent_at": TIMESTAMP,
            "error": {
                "code": longest_code,
                "message": escaped_text(200),
                "retryable": False,
                "retry_after_s": 86_400,
                "pointer": ("/" + "a" * 15) * 16,
            },
            "request_message_id": UUID,
            "supported_versions": [f"9999.9999.999{index}-draft" for index in range(8)],
        },
        MessageKind.GENERATION_MANIFEST: {
            "manifest_version": VERSION,
            "document_kind": "generation.manifest",
            "generation_id": UUID,
            "created_at": TIMESTAMP,
            "identity": {
                "org_uuid": UUID,
                "video_uuid": UUID,
                "source_id": UUID,
                "attempt_id": UUID,
                "dispatch_id": UUID,
                "execution_id": UUID,
            },
            "profile": profile,
            "encryption": encryption,
            "media": media,
            "renditions": [rendition] * 7,
            "master_playlist_path": REL_PATH,
            "thumbnail_path": REL_PATH,
            "artifact_count": MAX_ARTIFACTS,
            "total_bytes": 1_099_511_627_776,
            "artifacts": [artifact] * MAX_ARTIFACTS,
            "diagnostics": diagnostics,
        },
        MessageKind.TRANSCODE_CLAIM: {
            **envelope("transcode.claim"),
            "identity": identity,
            "claim": {"bootstrap_token": BOOTSTRAP_TOKEN, "runtime_id": UUID},
        },
        MessageKind.TRANSCODE_CLAIM_GRANTED: {
            **envelope("transcode.claim.granted"),
            "identity": identity_execution,
            "runtime_id": UUID,
            "generation_id": UUID,
            "output": {"location_id": UUID, "prefix": OBJECT_KEY},
        },
        # The cause `refused` makes the largest report: only it carries hub_error_code, which outweighs the
        # longer names of the other causes.
        MessageKind.TRANSCODE_UNCLAIMED: {
            **envelope("transcode.unclaimed"),
            "identity": identity,
            "runtime_id": UUID,
            "cause": "refused",
            "hub_error_code": "d" * 48,
        },
    }


def escaped_size(document: Any) -> int:
    """Compact bytes with every `/` written as `\\/`."""
    return len(json.dumps(document, separators=(",", ":")).replace("/", "\\/").encode("ascii"))


WORST_CASES = worst_cases()


@pytest.mark.parametrize("kind", list(MessageKind), ids=[kind.value for kind in MessageKind])
def test_worst_case_message_when_generated_from_bounds_should_pass_schema_and_fit_its_kind_limit(kind):
    # Arrange
    document = WORST_CASES[kind]

    # Act
    pointers = schema.error_pointers(kind.schema_file, document)
    size = escaped_size(document)

    # Assert
    assert pointers == []
    assert size <= load_registry().max_bytes(kind), size


_JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "null": type(None), "object": dict, "array": list}


@lru_cache(maxsize=None)
def _resolved(base_uri: str, reference: str) -> tuple[str, Any]:
    target_uri = urljoin(base_uri, reference.split("#", 1)[0]) if not reference.startswith("#") else base_uri
    return target_uri, schema.resolve_reference(base_uri, reference)


def _branch(base_uri: str, node: dict) -> tuple[str, dict]:
    return _resolved(base_uri, node["$ref"]) if "$ref" in node else (base_uri, node)


def _matches_type(node: dict, value: Any) -> bool:
    declared = node.get("type")
    if declared is None:
        return True
    types = [declared] if isinstance(declared, str) else declared
    return any(isinstance(value, _JSON_TYPES[name]) and not (name == "integer" and isinstance(value, bool))
               for name in types)


def _maximality_gaps(value: Any, node: dict, base_uri: str, pointer: str, gaps: list[str]) -> None:
    """Record every place where `value` stays below a bound its schema allows (enums and consts excepted)."""
    if "$ref" in node:
        target_uri, target = _resolved(base_uri, node["$ref"])
        _maximality_gaps(value, target, target_uri, pointer, gaps)
    if "anyOf" in node:
        if value is None:
            gaps.append(f"{pointer}: null where a value is allowed")
        branches = [_branch(base_uri, branch) for branch in node["anyOf"]]
        matching = [(uri, branch) for uri, branch in branches if _matches_type(branch, value)]
        for uri, branch in matching[:1]:
            _maximality_gaps(value, branch, uri, pointer, gaps)
    if isinstance(value, str) and "maxLength" in node and "enum" not in node and "const" not in node:
        if len(value) != node["maxLength"]:
            gaps.append(f"{pointer}: string of {len(value)} below maxLength {node['maxLength']}")
    if isinstance(value, int) and not isinstance(value, bool) and "maximum" in node and value != node["maximum"]:
        gaps.append(f"{pointer}: integer {value} below maximum {node['maximum']}")
    if isinstance(value, list):
        if "maxItems" in node and len(value) != node["maxItems"]:
            gaps.append(f"{pointer}: {len(value)} items below maxItems {node['maxItems']}")
        for index, item in enumerate(value):
            if "items" in node:
                _maximality_gaps(item, node["items"], base_uri, f"{pointer}/{index}", gaps)
    if isinstance(value, dict):
        properties = node.get("properties", {})
        gaps.extend(f"{pointer}: optional member {name} absent" for name in properties if name not in value)
        additional = node.get("additionalProperties")
        for name, member in value.items():
            if name in properties:
                _maximality_gaps(member, properties[name], base_uri, f"{pointer}/{name}", gaps)
            elif isinstance(additional, dict):
                _maximality_gaps(member, additional, base_uri, f"{pointer}/{name}", gaps)
            if "propertyNames" in node:
                _maximality_gaps(name, node["propertyNames"], base_uri, f"{pointer}/{name} (name)", gaps)
        if "maxProperties" in node and len(value) != node["maxProperties"]:
            gaps.append(f"{pointer}: {len(value)} members below maxProperties {node['maxProperties']}")


@pytest.mark.parametrize("kind", list(MessageKind), ids=[kind.value for kind in MessageKind])
def test_worst_case_message_when_walked_against_its_schema_should_reach_every_bound(kind):
    # Arrange
    base_uri = schema.SCHEMA_BASE_URI + kind.schema_file
    root = schema.resolve_reference(base_uri, base_uri)
    gaps: list[str] = []

    # Act
    _maximality_gaps(WORST_CASES[kind], root, base_uri, "", gaps)

    # Assert
    assert gaps == []


def test_token_budget_when_compared_with_the_maximal_manifest_should_cover_it_with_little_slack():
    # Arrange
    maximal = json.dumps(WORST_CASES[MessageKind.GENERATION_MANIFEST], separators=(",", ":")).encode("ascii")

    # Act
    tokens = gate.count_tokens(maximal)

    # Assert
    assert tokens <= load_registry().max_tokens <= tokens + 1_024


@pytest.mark.parametrize(
    ("carrier", "schema_file"),
    [
        ("runpod.input", "carriers/runpod-input.schema.json"),
        ("runpod.output", "carriers/runpod-output.schema.json"),
    ],
)
def test_worst_case_carrier_when_message_is_all_escaped_characters_should_fit_its_raw_limit(carrier, schema_file):
    # Arrange: the message length comes from the carrier schema itself.
    registry = load_registry()
    message_length = schema.resolve_reference(schema.SCHEMA_BASE_URI + schema_file, "#/properties/vh_message")[
        "maxLength"
    ]
    value = {"vh_message": escaped_text(message_length), "vh_signature": "hmac-sha256=" + SHA256}

    # Act
    pointers = schema.error_pointers(schema_file, value)
    size = escaped_size(value)

    # Assert
    assert pointers == []
    assert size <= registry.carrier_max_bytes(carrier), size


def _storage_context() -> TrustedContext:
    return TrustedContext.from_dict(files.read_json("fixtures/contexts/ctx-hub-storage.json"))


def test_validate_when_a_manifest_holds_the_most_malformed_artifacts_allowed_should_refuse_it_at_the_schema_layer():
    # Arrange: P14 whose artifacts are 25,209 objects of wrongly typed members (the Hub's memory test input).
    document = files.read_json("fixtures/manifest/P14-encrypted-three-renditions.json")
    document["artifacts"] = [{"path": 0, "kind": 0, "size_bytes": "x", "sha256": 0, "rendition": 0}] * 25_209
    raw = json.dumps(document, separators=(",", ":")).encode("ascii")

    # Act
    verdict = pipeline.validate(raw, _storage_context())

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION)


def test_validate_when_the_maximal_manifest_is_read_should_pass_every_gate_before_the_semantic_layer():
    # Arrange: the worst case under the accepted draft version; its identity is synthetic, so L4 refuses it.
    document = {**WORST_CASES[MessageKind.GENERATION_MANIFEST], "manifest_version": "1.0.0-draft"}
    raw = json.dumps(document, separators=(",", ":")).encode("ascii")

    # Act
    verdict = pipeline.validate(raw, _storage_context())

    # Assert
    assert len(raw) <= load_registry().max_bytes(MessageKind.GENERATION_MANIFEST)
    assert verdict.layer is Layer.SEMANTIC
