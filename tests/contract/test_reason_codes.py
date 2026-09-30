"""The Python enums, the JSON lists and the schema enums name the same codes and kinds."""

from tests.contract import files
from tests.contract.reasons import Layer, ReasonCode
from tests.contract.registry import ContractRegistry

MANIFEST_DRAFT = "1.0.0-draft"
RESERVED_CODES = {
    "invalid_signature",
    "duplicate_message",
    "conflicting_terminal_result",
    "manifest_digest_mismatch",
    "internal_error",
    "service_unavailable",
}


def _manifest_artifact_kind_enum() -> list[str]:
    manifest_schema = files.read_json("schemas/generation-manifest.schema.json")
    return manifest_schema["properties"]["artifacts"]["items"]["properties"]["kind"]["enum"]


def test_reason_codes_when_compared_should_match_python_enum():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    python_codes = [code.value for code in ReasonCode]

    # Assert
    assert python_codes == list(registry.reason_code_values())


def test_reason_codes_when_scanned_should_reserve_exactly_the_live_path_codes():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    reserved = {entry["code"] for entry in registry.reason_codes if entry["reserved"]}

    # Assert
    assert reserved == RESERVED_CODES


def test_layers_when_compared_should_be_exactly_the_layers_of_non_reserved_codes():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    layers = [entry["layer"] for entry in registry.reason_codes if not entry["reserved"]]

    # Assert
    assert set(layers) == {layer.value for layer in Layer}
    assert [layer.value for layer in Layer] == list(dict.fromkeys(layers))


def test_hub_error_codes_when_compared_should_equal_reason_codes():
    # Arrange
    hub_error_schema = files.read_json("schemas/hub-error.schema.json")

    # Act
    codes = hub_error_schema["properties"]["error"]["properties"]["code"]["enum"]

    # Assert
    assert codes == [code.value for code in ReasonCode]


def test_artifact_kinds_when_compared_should_equal_manifest_schema_enum():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    active = list(registry.active_artifact_kinds(MANIFEST_DRAFT))

    # Assert
    assert active == _manifest_artifact_kind_enum()


def test_reserved_artifact_kinds_when_compared_should_be_absent_from_manifest_schema_enum():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    reserved = set(registry.reserved_artifact_kinds(MANIFEST_DRAFT))

    # Assert
    assert reserved == {"captions", "transcript", "chapters", "storyboard"}
    assert reserved & set(_manifest_artifact_kind_enum()) == set()
