"""The trusted context demands exactly the fields README.md lists per (role, kind), and nothing is silently skipped."""

import copy
from typing import Any

import pytest

from tests.contract import files, schema
from tests.contract.context import ContextIncomplete, ContextInvalid, TrustedContext

CONTEXT_SCHEMA_FILE = "schemas/trusted-context.schema.json"
CONTEXT_FILES = [path for path in files.list_tree_files() if path.startswith("fixtures/contexts/")]

REQUEST = "transcode.request"
PROGRESS = "transcode.progress"
COMPLETED = "transcode.result.completed"
FAILED = "transcode.result.failed"
HUB_ERROR = "hub.error"
MANIFEST = "generation.manifest"

MESSAGE_GATE = ["accepted_protocol_versions"]
SIGNED_ENVELOPE = [*MESSAGE_GATE, "expected_audience", "known_key_ids", "now", "clock_skew_s"]
DISPATCH_IDENTITY = [
    "expect.org_uuid",
    "expect.video_uuid",
    "expect.owning_cell_id",
    "expect.placement_epoch",
    "expect.lifecycle_revision",
    "expect.attempt_id",
    "expect.dispatch_id",
]
EXECUTION_IDENTITY = [*DISPATCH_IDENTITY, "expect.execution_id", "expect.execution_fence"]
RECEIVER_REQUIREMENTS = {
    PROGRESS: [*SIGNED_ENVELOPE, *EXECUTION_IDENTITY, "last_event_seq"],
    COMPLETED: [*SIGNED_ENVELOPE, *EXECUTION_IDENTITY, "expect.source_id", "expect.output_location_id",
                "expect.profile"],
    FAILED: [*SIGNED_ENVELOPE, *EXECUTION_IDENTITY, "expect.source_id"],
    MANIFEST: [
        "accepted_manifest_versions",
        "expect.org_uuid",
        "expect.video_uuid",
        "expect.attempt_id",
        "expect.dispatch_id",
        "expect.execution_id",
        "expect.source_id",
        "expect.profile",
        "expect.encryption",
        "expect.renditions",
        "expect.max_artifact_count",
    ],
}
# (role, kind, channel) -> the required context fields stated in README.md.
REQUIRED_FIELDS: dict[tuple[str, str, str], list[str]] = {
    ("worker_receiver", REQUEST, "dispatch"): SIGNED_ENVELOPE,
    ("worker_receiver", HUB_ERROR, "push"): MESSAGE_GATE,
    ("hub_sender", REQUEST, "dispatch"): [
        *SIGNED_ENVELOPE,
        *DISPATCH_IDENTITY,
        "expect.source_id",
        "expect.source_location_id",
        "expect.output_location_id",
        "expect.profile",
        "expect.encryption",
    ],
    **{("hub_receiver", kind, "storage" if kind == MANIFEST else "push"): fields
       for kind, fields in RECEIVER_REQUIREMENTS.items()},
    **{("worker_sender", kind, "storage" if kind == MANIFEST else "push"): fields
       for kind, fields in RECEIVER_REQUIREMENTS.items()},
}
REQUIREMENT_CASES = [
    (role, kind, channel, field)
    for (role, kind, channel), fields in REQUIRED_FIELDS.items()
    for field in fields
]


def _complete_context(role: str, kind: str, channel: str) -> dict[str, Any]:
    """Every context field filled, taken from the fixture contexts."""
    push = files.read_json("fixtures/contexts/ctx-hub-push.json")
    storage = files.read_json("fixtures/contexts/ctx-hub-storage.json")
    return {
        **push,
        "role": role,
        "channel": channel,
        "accepted_message_kinds": [kind],
        "accepted_manifest_versions": storage["accepted_manifest_versions"],
    }


def _without(context: dict[str, Any], field: str) -> dict[str, Any]:
    changed = copy.deepcopy(context)
    if field.startswith("expect."):
        del changed["expect"][field.removeprefix("expect.")]
    else:
        del changed[field]
    return changed


def _nulled(context: dict[str, Any], field: str) -> dict[str, Any]:
    changed = copy.deepcopy(context)
    if field.startswith("expect."):
        changed["expect"][field.removeprefix("expect.")] = None
    else:
        changed[field] = None
    return changed


def _only(context: dict[str, Any], required: list[str]) -> dict[str, Any]:
    kept = {"context_version", "role", "channel", "accepted_message_kinds", *required}
    reduced = {key: value for key, value in context.items() if key in kept}
    expect_fields = {field.removeprefix("expect.") for field in required if field.startswith("expect.")}
    if expect_fields:
        reduced["expect"] = {key: value for key, value in context["expect"].items() if key in expect_fields}
    return reduced


@pytest.mark.parametrize("path", CONTEXT_FILES)
def test_fixture_context_when_loaded_should_satisfy_context_schema(path):
    # Arrange
    data = files.read_json(path)

    # Act
    pointers = schema.error_pointers(CONTEXT_SCHEMA_FILE, data)
    context = TrustedContext.from_dict(data)

    # Assert
    assert pointers == []
    assert context.role == data["role"]


@pytest.mark.parametrize(("role", "kind", "channel", "field"), REQUIREMENT_CASES)
def test_context_when_required_field_missing_for_role_and_kind_should_raise_context_incomplete(
    role, kind, channel, field
):
    # Arrange
    context = _without(_complete_context(role, kind, channel), field)

    # Act / Assert
    with pytest.raises(ContextIncomplete) as raised:
        TrustedContext.from_dict(context)
    assert raised.value.field == field


@pytest.mark.parametrize(("role", "kind", "channel", "field"), REQUIREMENT_CASES)
def test_context_when_required_field_null_for_role_and_kind_should_raise_context_incomplete(
    role, kind, channel, field
):
    # Arrange
    context = _nulled(_complete_context(role, kind, channel), field)

    # Act / Assert
    with pytest.raises(ContextIncomplete) as raised:
        TrustedContext.from_dict(context)
    assert raised.value.field == field


@pytest.mark.parametrize(("role", "kind", "channel"), list(REQUIRED_FIELDS))
def test_context_when_holding_only_the_required_fields_should_load(role, kind, channel):
    # Arrange
    context = _only(_complete_context(role, kind, channel), REQUIRED_FIELDS[(role, kind, channel)])

    # Act
    loaded = TrustedContext.from_dict(context)

    # Assert
    assert loaded.accepted_message_kinds == (kind,)


@pytest.mark.parametrize(
    ("role", "kind", "channel"),
    [
        ("hub_receiver", REQUEST, "dispatch"),
        ("worker_receiver", PROGRESS, "push"),
        ("hub_sender", HUB_ERROR, "push"),
        ("hub_receiver", PROGRESS, "dispatch"),
        ("hub_receiver", MANIFEST, "push"),
        ("hub_receiver", COMPLETED, "storage"),
        ("worker_receiver", REQUEST, "push"),
    ],
)
def test_context_when_role_kind_and_channel_do_not_fit_should_raise_context_invalid(role, kind, channel):
    # Arrange
    context = _complete_context(role, kind, channel)

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)


@pytest.mark.parametrize("value", ["missing", None])
def test_context_when_accepted_message_kinds_absent_should_raise_context_invalid(value):
    # Arrange
    context = _complete_context("hub_receiver", PROGRESS, "push")
    if value == "missing":
        del context["accepted_message_kinds"]
    else:
        context["accepted_message_kinds"] = value

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)


def test_context_when_now_is_not_a_calendar_instant_should_raise_context_invalid():
    # Arrange
    context = {**_complete_context("hub_receiver", PROGRESS, "push"), "now": "2030-02-30T00:00:00.000Z"}

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)


@pytest.mark.parametrize(
    "change",
    [{"unknown_field": 1}, {"clock_skew_s": 601}, {"context_version": 2}, {"accepted_message_kinds": []}],
    ids=["unknown-field", "skew-over-600", "context-version-2", "no-accepted-kind"],
)
def test_context_when_shape_violates_schema_should_raise_context_invalid(change):
    # Arrange
    context = {**_complete_context("hub_receiver", PROGRESS, "push"), **change}

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)


def test_context_when_expect_is_an_empty_object_for_a_worker_receiver_should_load():
    # Arrange
    context = {**files.read_json("fixtures/contexts/ctx-worker-dispatch.json"), "expect": {}}

    # Act
    loaded = TrustedContext.from_dict(context)

    # Assert
    assert loaded.expect == {}


def test_context_when_expect_is_an_empty_object_for_a_hub_receiver_should_raise_context_incomplete():
    # Arrange
    context = {**_complete_context("hub_receiver", PROGRESS, "push"), "expect": {}}

    # Act / Assert
    with pytest.raises(ContextIncomplete) as raised:
        TrustedContext.from_dict(context)
    assert raised.value.field == "expect.org_uuid"


@pytest.mark.parametrize(
    ("field", "versions"),
    [
        ("accepted_protocol_versions", ["3.0.0-draft"]),
        ("accepted_protocol_versions", ["2.0.0-draft", "2.1.0-draft"]),
        ("accepted_protocol_versions", ["2.0.0"]),
        ("accepted_manifest_versions", ["2.0.0"]),
    ],
    ids=["another-major", "an-undefined-minor-beside-the-draft", "the-release", "another-manifest-major"],
)
def test_context_when_an_accepted_version_is_not_defined_by_versions_json_should_raise_context_invalid(
    field, versions
):
    # Arrange: a context must not create protocol support the tree does not define.
    context = {**_complete_context("hub_receiver", PROGRESS, "push"), field: versions}

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)


def test_context_when_accepted_versions_are_the_empty_live_lists_should_load():
    # Arrange
    context = {
        **_complete_context("hub_receiver", PROGRESS, "push"),
        "accepted_protocol_versions": [],
        "accepted_manifest_versions": [],
    }

    # Act
    loaded = TrustedContext.from_dict(context)

    # Assert
    assert loaded.accepted_protocol_versions == ()


@pytest.mark.parametrize(
    "change",
    [
        {"accepted_message_kinds": {"0": PROGRESS}},
        {"known_key_ids": {"0": "test-wk2hub-k1"}},
        {"accepted_protocol_versions": {"0": "2.0.0-draft"}},
    ],
    ids=["kinds-as-a-map", "key-ids-as-a-map", "versions-as-a-map"],
)
def test_context_when_a_list_field_holds_a_map_should_raise_context_invalid(change):
    # Arrange: the Python analogue of a PHP associative array where a list is required.
    context = {**_complete_context("hub_receiver", PROGRESS, "push"), **change}

    # Act / Assert
    with pytest.raises(ContextInvalid):
        TrustedContext.from_dict(context)
