"""The typed failure every worker stage reports: the class, code, retryability and detail of the
transcode contract v2 error object. The job runner adds the stage when it serializes one."""

from __future__ import annotations

import re
from dataclasses import dataclass

ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "source_unavailable",
        "source_integrity_mismatch",
        "input_invalid",
        "input_limits_exceeded",
        "encoder_failed",
        "output_write_failed",
        "resource_exhausted",
        "deadline_exceeded",
        "authority_refused",
        "cancelled",
        "configuration_error",
        "internal_error",
    }
)

NEVER_RETRYABLE_CLASSES: frozenset[str] = frozenset(
    {
        "source_integrity_mismatch",
        "input_invalid",
        "input_limits_exceeded",
        "authority_refused",
        "cancelled",
        "configuration_error",
    }
)

CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,47}")
DETAIL_MAX_CHARS = 1000
SANITIZED_DETAIL_MAX_CHARS = 200
PRINTABLE_ASCII_FIRST = 0x20
PRINTABLE_ASCII_LAST = 0x7E
URL_SCHEME_SEPARATOR = "://"
URL_SCHEME_SEPARATOR_REPLACEMENT = ":__"
UNPRINTABLE_REPLACEMENT = "?"
# A URL-like token: a scheme, "://", then everything up to the next whitespace. The scheme is
# capped so that a long run of letters without "://" costs linear, not quadratic, regex time; a
# longer scheme still matches on its last characters, so its host and path are removed all the same.
URL_SCHEME_MAX_CHARS = 64
URL_TOKEN_PATTERN = re.compile(rf"[A-Za-z][A-Za-z0-9+.-]{{0,{URL_SCHEME_MAX_CHARS - 1}}}://\S*")
URL_MARKER = "[url]"


def _is_printable_ascii(char: str) -> bool:
    return PRINTABLE_ASCII_FIRST <= ord(char) <= PRINTABLE_ASCII_LAST


def _require_valid_detail(detail: object) -> None:
    if not isinstance(detail, str):
        raise ValueError("detail must be a string")
    if len(detail) > DETAIL_MAX_CHARS:
        raise ValueError(f"detail must be at most {DETAIL_MAX_CHARS} characters")
    if not all(_is_printable_ascii(char) for char in detail):
        raise ValueError("detail must hold printable ASCII only")
    if URL_SCHEME_SEPARATOR in detail:
        raise ValueError("detail must not contain a URL")


@dataclass(frozen=True, slots=True)
class Failure:
    error_class: str
    code: str
    retryable: bool
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.error_class, str) or self.error_class not in ERROR_CLASSES:
            raise ValueError(f"unknown error class: {self.error_class!r}")
        if not isinstance(self.code, str) or CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError(f"invalid diagnostic code: {self.code!r}")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a bool")
        if self.retryable and self.error_class in NEVER_RETRYABLE_CLASSES:
            raise ValueError(f"error class {self.error_class} is never retryable")
        _require_valid_detail(self.detail)


class WorkerFailure(Exception):
    """Raised by a stage to end the job with a typed failure."""

    def __init__(self, failure: Failure) -> None:
        if not isinstance(failure, Failure):
            raise TypeError("failure must be a Failure")
        super().__init__(f"{failure.error_class}/{failure.code}: {failure.detail}")
        self.failure = failure


def make_detail(text: str) -> str:
    """Turn arbitrary text (tool output included) into a valid failure detail.

    Every URL-like token (scheme, "://", then everything up to the next whitespace) becomes the
    marker "[url]", which removes the host, path and query string of every ordinary URL (a scheme
    without a letter, or a URL split by unusual whitespace, can still leave parts behind). Then every
    character outside printable ASCII becomes "?", any "://" left over becomes ":__" (a second line
    of defence), and the result is cut to 200 characters.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    without_url_tokens = URL_TOKEN_PATTERN.sub(URL_MARKER, text)
    printable = "".join(
        char if _is_printable_ascii(char) else UNPRINTABLE_REPLACEMENT for char in without_url_tokens
    )
    without_urls = printable.replace(URL_SCHEME_SEPARATOR, URL_SCHEME_SEPARATOR_REPLACEMENT)
    return without_urls[:SANITIZED_DETAIL_MAX_CHARS]
