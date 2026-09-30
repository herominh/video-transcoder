"""Schema lints that keep jsonschema (Python) and opis/json-schema (PHP) from silently disagreeing."""

import re
import socket
from typing import Any, Iterator

import pytest

from tests.contract import files, schema

DIALECT = "https://json-schema.org/draft/2020-12/schema"
EXT_DEFINITION = ("schemas/common.schema.json", "/$defs/ext")
ALLOWED_CHARACTER_BANS = {
    "[\\x00-\\x1f\\x7f]",  # control characters
    "[^\\x20-\\x7e]",  # anything but printable ASCII
    "[^\\x09\\x0a\\x0d\\x20-\\x7e]",  # anything but printable ASCII and JSON whitespace (carrier message text)
}
# A backslash may only escape punctuation or introduce a two-digit \xHH; no \d, \w, \s, \b, \p, \u ...
NON_PORTABLE_ESCAPE = re.compile(r"\\(?!x[0-9a-fA-F]{2}|[.\\\-/^$*+?()\[\]{}|])")
SUBSCHEMA_MAP_KEYWORDS = ("properties", "$defs", "patternProperties", "dependentSchemas")
SUBSCHEMA_KEYWORDS = ("items", "additionalProperties", "propertyNames", "not", "if", "then", "else", "contains")
SUBSCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf", "prefixItems")

SCHEMA_FILES = schema.schema_files()


def _subschemas(node: Any, pointer: str = "") -> Iterator[tuple[str, dict]]:
    """Every schema object inside a schema document, with its JSON pointer."""
    if not isinstance(node, dict):
        return
    yield pointer, node
    for keyword in SUBSCHEMA_MAP_KEYWORDS:
        for name, child in node.get(keyword, {}).items():
            yield from _subschemas(child, f"{pointer}/{keyword}/{name}")
    for keyword in SUBSCHEMA_KEYWORDS:
        if isinstance(node.get(keyword), dict):
            yield from _subschemas(node[keyword], f"{pointer}/{keyword}")
    for keyword in SUBSCHEMA_LIST_KEYWORDS:
        for index, child in enumerate(node.get(keyword, [])):
            yield from _subschemas(child, f"{pointer}/{keyword}/{index}")


def _types(node: dict) -> set[str]:
    declared = node.get("type")
    if declared is None:
        return set()
    return {declared} if isinstance(declared, str) else set(declared)


def _all_nodes() -> Iterator[tuple[str, str, dict]]:
    for relative_path in SCHEMA_FILES:
        for pointer, node in _subschemas(files.read_json(relative_path)):
            yield relative_path, pointer, node


def test_schema_files_when_listed_should_declare_2020_12_and_an_id_inside_the_contract_namespace():
    # Act
    headers = {path: files.read_json(path) for path in SCHEMA_FILES}

    # Assert
    assert len(headers) == 11
    for path, document in headers.items():
        assert document["$schema"] == DIALECT, path
        assert document["$id"] == schema.SCHEMA_BASE_URI + path, path


def test_schemas_when_scanned_should_use_no_format_keyword():
    # Act
    offenders = [(path, pointer) for path, pointer, node in _all_nodes() if "format" in node]

    # Assert
    assert offenders == []


def test_patterns_when_scanned_should_use_only_portable_syntax():
    # Arrange
    patterns = [(path, pointer, node["pattern"]) for path, pointer, node in _all_nodes() if "pattern" in node]

    # Act
    offenders = [
        (path, pointer, pattern)
        for path, pointer, pattern in patterns
        if NON_PORTABLE_ESCAPE.search(pattern) or "(?" in pattern
    ]

    # Assert
    assert patterns, "the lint must see the patterns"
    assert offenders == []


def test_every_object_when_scanned_should_forbid_additional_properties_except_ext():
    # Act
    offenders = [
        (path, pointer)
        for path, pointer, node in _all_nodes()
        if "object" in _types(node)
        and (path, pointer) != EXT_DEFINITION
        and node.get("additionalProperties") is not False
    ]

    # Assert
    assert offenders == []


def test_every_string_when_scanned_should_carry_max_length_and_a_character_ban():
    # Act
    offenders = [
        (path, pointer)
        for path, pointer, node in _all_nodes()
        if "string" in _types(node)
        and "enum" not in node
        and "const" not in node
        and ("maxLength" not in node or node.get("not", {}).get("pattern") not in ALLOWED_CHARACTER_BANS)
    ]

    # Assert
    assert offenders == []


def test_every_closed_string_value_when_scanned_should_be_printable_ascii():
    # Arrange
    printable = re.compile(r"[\x20-\x7e]+")

    # Act
    offenders = [
        (path, pointer, value)
        for path, pointer, node in _all_nodes()
        for value in [*node.get("enum", []), *([node["const"]] if "const" in node else [])]
        if isinstance(value, str) and printable.fullmatch(value) is None
    ]

    # Assert
    assert offenders == []


def test_every_array_when_scanned_should_carry_max_items():
    # Act
    offenders = [(path, pointer) for path, pointer, node in _all_nodes() if "array" in _types(node)
                 and "maxItems" not in node]

    # Assert
    assert offenders == []


def test_every_integer_when_scanned_should_carry_minimum_and_maximum_within_2_pow_53():
    # Arrange
    ceiling = 2**53 - 1

    # Act
    offenders = [
        (path, pointer)
        for path, pointer, node in _all_nodes()
        if "integer" in _types(node)
        and not ("minimum" in node and "maximum" in node and -ceiling <= node["minimum"] <= node["maximum"] <= ceiling)
    ]

    # Assert
    assert offenders == []


def test_schema_refs_when_resolved_should_stay_inside_the_contract_tree():
    # Arrange
    references = [
        (path, node["$ref"]) for path, _, node in _all_nodes() if "$ref" in node
    ]

    # Act
    unresolved = []
    for path, reference in references:
        try:
            schema.resolve_reference(schema.SCHEMA_BASE_URI + path, reference)
        except schema.UnresolvableReference:
            unresolved.append((path, reference))

    # Assert
    assert references, "the lint must see the references"
    assert all("://" not in reference or reference.startswith(schema.SCHEMA_BASE_URI) for _, reference in references)
    assert unresolved == []


def test_unknown_remote_ref_when_validated_should_fail_without_touching_the_network(monkeypatch):
    # Arrange: record every attempt instead of raising, so no library can swallow the evidence.
    attempts: list[tuple[str, tuple]] = []

    def recorder(name):
        def record(*args, **kwargs):
            attempts.append((name, args))
            raise OSError(f"network disabled in this test ({name})")

        return record

    monkeypatch.setattr(socket, "create_connection", recorder("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", recorder("getaddrinfo"))
    monkeypatch.setattr(socket.socket, "connect", recorder("socket.connect"))
    remote = {"$ref": "https://schemas.example.invalid/elsewhere.schema.json"}

    # Act
    with pytest.raises(schema.UnresolvableReference):
        schema.error_pointers_for_schema(remote, {})

    # Assert
    assert attempts == []


def test_every_free_text_string_when_scanned_should_also_refuse_urls():
    # Arrange
    printable_ascii_ban = "[^\\x20-\\x7e]"

    # Act
    free_text = [(path, pointer, node) for path, pointer, node in _all_nodes()
                 if node.get("not", {}).get("pattern") == printable_ascii_ban]
    offenders = [(path, pointer) for path, pointer, node in free_text
                 if {"not": {"pattern": "://"}} not in node.get("allOf", [])]

    # Assert
    assert free_text, "the lint must see the free-text strings"
    assert offenders == []
