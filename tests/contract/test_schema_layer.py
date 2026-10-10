"""L3 as README.md section 6 states it: the verdict names the first error the library reports, pointers follow
RFC 6901, and parity with the Hub is on layer and reason (the pointer of a many-violation document is diagnostic)."""

import json

from core.protocol import files, pipeline, schema
from core.protocol.context import TrustedContext
from core.protocol.reasons import Layer, ReasonCode

PUSH = "fixtures/contexts/ctx-hub-push.json"
PROGRESS_PAYLOAD = "fixtures/progress/P04-downloading.json"
PROGRESS_SCHEMA = "schemas/transcode-progress.schema.json"


def _progress() -> dict:
    return files.read_json(PROGRESS_PAYLOAD)


def _wire(document: dict) -> bytes:
    return json.dumps(document, separators=(",", ":")).encode("ascii")


def test_json_pointer_when_a_segment_holds_slash_and_tilde_should_escape_them():
    # Act
    pointer = schema.json_pointer(["ext", "a/b~c", 0])

    # Assert
    assert pointer == "/ext/a~1b~0c/0"


def test_json_pointer_when_the_path_is_empty_should_name_the_root():
    # Act
    pointer = schema.json_pointer([])

    # Assert
    assert pointer == ""


def test_error_pointers_when_an_ext_member_name_holds_slash_and_tilde_should_report_the_escaped_member():
    # Arrange
    document = {**_progress(), "ext": {"a/b~c": []}}

    # Act
    pointers = schema.error_pointers(PROGRESS_SCHEMA, document)

    # Assert
    assert "/ext/a~1b~0c" in pointers


def test_validate_when_a_document_breaks_two_rules_should_name_the_first_error_the_library_reports():
    # Arrange
    document = {**_progress(), "stage_progress_pct": 101, "unexpected": 1}

    # Act
    verdict = pipeline.validate(_wire(document), TrustedContext.from_dict(files.read_json(PUSH)))

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION)
    assert verdict.instance_path == schema.error_pointers(PROGRESS_SCHEMA, document)[0]


def test_first_error_pointer_when_the_document_is_valid_should_be_none():
    # Act
    pointer = schema.first_error_pointer(PROGRESS_SCHEMA, _progress())

    # Assert
    assert pointer is None
