from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from core.failure import (
    ERROR_CLASSES,
    NEVER_RETRYABLE_CLASSES,
    Diagnostic,
    Failure,
    WorkerFailure,
    make_detail,
)

FAILED_RESULT_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "transcode"
    / "v2"
    / "schemas"
    / "transcode-result-failed.schema.json"
)
DETAIL_LIMIT = 1000
SANITIZED_LIMIT = 200
HUGE_TEXT_CHARS = 1024 * 1024
FAST_ENOUGH_S = 2.0


def _failure(**overrides: Any) -> Failure:
    fields: dict[str, Any] = {
        "error_class": "source_unavailable",
        "code": "source_read_failed",
        "retryable": True,
        "detail": "storage answered 503",
    }
    fields.update(overrides)
    return Failure(**fields)


def _contract_error_schema() -> dict[str, Any]:
    schema = json.loads(FAILED_RESULT_SCHEMA.read_text(encoding="utf-8"))
    return schema["properties"]["error"]


def test_error_classes_when_compared_with_contract_v2_should_match_its_enum_exactly() -> None:
    # Arrange
    error_schema = _contract_error_schema()

    # Act
    contract_classes = frozenset(error_schema["properties"]["class"]["enum"])
    contract_never_retryable = frozenset(error_schema["if"]["properties"]["class"]["enum"])

    # Assert
    assert ERROR_CLASSES == contract_classes
    assert NEVER_RETRYABLE_CLASSES == contract_never_retryable


def test_failure_when_every_field_is_valid_should_keep_its_values() -> None:
    # Arrange / Act
    failure = _failure()

    # Assert
    assert (failure.error_class, failure.code, failure.retryable, failure.detail) == (
        "source_unavailable",
        "source_read_failed",
        True,
        "storage answered 503",
    )


def test_failure_when_the_class_is_unknown_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _failure(error_class="network_glitch")


@pytest.mark.parametrize("code", ["ab", "Source_read", "1source", "source-read", "a" * 49, "source_read\n", ""])
def test_failure_when_the_code_breaks_the_pattern_should_raise(code: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _failure(code=code)


@pytest.mark.parametrize("error_class", sorted(NEVER_RETRYABLE_CLASSES))
def test_failure_when_a_never_retryable_class_is_retryable_should_raise(error_class: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _failure(error_class=error_class, retryable=True)


@pytest.mark.parametrize("error_class", sorted(NEVER_RETRYABLE_CLASSES))
def test_failure_when_a_never_retryable_class_is_not_retryable_should_be_accepted(error_class: str) -> None:
    # Arrange / Act
    failure = _failure(error_class=error_class, retryable=False)

    # Assert
    assert failure.retryable is False


def test_failure_when_retryable_is_not_a_bool_should_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _failure(retryable=1)


@pytest.mark.parametrize(
    "detail",
    ["caf\u00e9 closed", "line one\nline two", "tab\there", "bell\x07", "see https://example.test/x", "x" * 1001],
)
def test_failure_when_the_detail_is_not_short_printable_url_free_ascii_should_raise(detail: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        _failure(detail=detail)


def test_failure_when_the_detail_is_exactly_at_the_length_limit_should_be_accepted() -> None:
    # Arrange
    detail = "x" * DETAIL_LIMIT

    # Act
    failure = _failure(detail=detail)

    # Assert
    assert len(failure.detail) == DETAIL_LIMIT


def test_worker_failure_when_raised_should_carry_its_failure() -> None:
    # Arrange
    failure = _failure()

    # Act
    with pytest.raises(WorkerFailure) as raised:
        raise WorkerFailure(failure)

    # Assert
    assert raised.value.failure is failure


def test_make_detail_when_text_holds_non_printable_characters_should_replace_each_with_a_question_mark() -> None:
    # Arrange
    text = "caf\u00e9\nok\x00"

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "caf??ok?"


def test_make_detail_when_text_holds_urls_should_replace_each_with_the_url_marker() -> None:
    # Arrange
    text = "fetch https://bucket.example/key and s3://other failed"

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "fetch [url] and [url] failed"


def test_make_detail_when_text_holds_a_presigned_r2_url_should_keep_no_part_of_it() -> None:
    # Arrange
    text = (
        "HTTP error 403 Forbidden for https://acct1234.r2.cloudflarestorage.com/org-bucket/original-videos/"
        "0192f0aa-1111-7222-8333-944455556666/original.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIAEXAMPLE%2F20261009%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20261009T010203Z"
        "&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=deadbeefcafef00d0123456789abcdef while probing"
    )

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "HTTP error 403 Forbidden for [url] while probing"
    for fragment in ("acct1234", "r2.cloudflarestorage.com", "org-bucket", "original.mp4", "X-Amz", "deadbeef"):
        assert fragment not in detail


def test_make_detail_when_text_holds_a_loopback_url_with_a_token_should_keep_no_part_of_it() -> None:
    # Arrange
    text = "Connection to tcp://127.0.0.1:53211/source.bin?token=s3cr3t-0123 failed: Connection refused"

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "Connection to [url] failed: Connection refused"
    for fragment in ("127.0.0.1", "53211", "source.bin", "token", "s3cr3t"):
        assert fragment not in detail


def test_make_detail_when_a_scheme_separator_has_no_scheme_should_still_break_it() -> None:
    # Arrange
    text = "odd 9://host and \u00e9://other and bare :// end"

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "odd 9:__host and ?:__other and bare :__ end"


def test_make_detail_when_text_is_a_huge_run_of_letters_should_finish_quickly() -> None:
    # Arrange
    text = "a" * HUGE_TEXT_CHARS
    started = time.monotonic()

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "a" * SANITIZED_LIMIT
    assert time.monotonic() - started < FAST_ENOUGH_S


def test_make_detail_when_text_is_long_should_truncate_to_200_characters() -> None:
    # Arrange
    text = "y" * 5000

    # Act
    detail = make_detail(text)

    # Assert
    assert detail == "y" * SANITIZED_LIMIT


def test_make_detail_when_text_is_hostile_should_always_yield_a_valid_failure_detail() -> None:
    # Arrange
    text = ("\u202e:/" + "/" * 3 + "\r\n\x1b[31m" + "\U0001f600") * 100

    # Act
    detail = make_detail(text)
    failure = _failure(detail=detail)

    # Assert
    assert failure.detail == detail
    assert len(detail) <= SANITIZED_LIMIT


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("thumbnail_unavailable", "x" * SANITIZED_LIMIT),
        ("abc", ""),
        ("a" * 48, 'quotes " and \\ backslashes are printable'),
    ],
    ids=["the longest detail", "the shortest code and an empty detail", "the longest code"],
)
def test_diagnostic_when_code_and_detail_fit_the_contract_should_be_accepted(code: str, detail: str) -> None:
    # Arrange / Act
    diagnostic = Diagnostic(code=code, detail=detail)

    # Assert
    assert (diagnostic.code, diagnostic.detail) == (code, detail)


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("ab", "fine"),
        ("a" * 49, "fine"),
        ("Thumbnail", "fine"),
        (None, "fine"),
        ("thumbnail_unavailable", "x" * (SANITIZED_LIMIT + 1)),
        ("thumbnail_unavailable", "a line\nbreak"),
        ("thumbnail_unavailable", "caf\u00e9"),
        ("thumbnail_unavailable", "see https://example.invalid/x"),
        ("thumbnail_unavailable", None),
    ],
    ids=[
        "a code too short",
        "a code too long",
        "an uppercase code",
        "no code",
        "a detail of 201 characters",
        "a control character",
        "a character outside ASCII",
        "a URL",
        "no detail",
    ],
)
def test_diagnostic_when_code_or_detail_breaks_the_contract_should_raise(code: Any, detail: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        Diagnostic(code=code, detail=detail)
