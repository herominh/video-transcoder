"""Layers L0-L2: raw size, strict parsing and the version/kind gate (standard library only).

L1 judges the WHOLE received text, members that a later duplicate overwrites included: the
structural checks run on the raw text, never on the decoded (last-value-wins) object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .context import TrustedContext
from .reasons import Layer, ReasonCode
from .registry import KNOWN_KINDS, Channel, MessageKind
from .verdict import Verdict

# Tab, LF, CR and printable ASCII: the only bytes a contract payload may hold.
ALLOWED_BYTES = bytes([0x09, 0x0A, 0x0D, *range(0x20, 0x7F)])

# The token budget scan (README section 4, step 2): outside a string literal, `"`, `{`, `[`, the start
# of a number (then every following byte of `0-9+-.eE`) and the start of a literal (`t`, `f` or `n`,
# then every following byte of `a-z`) count 1; inside a string literal `\` skips the next byte.
# Possessive quantifiers keep no backtracking state, so matching a long string literal costs memory
# proportional to nothing but the match itself (a greedy alternation kept state per character).
_TOKEN = re.compile(rb'"(?:[^"\\]++|\\.)*+"?|[{\[]|[-0-9][-0-9+.eE]*+|[tfn][a-z]*+', re.DOTALL)
# Structural tokens of a text already known to be JSON: string literals, brackets and colons.
_STRUCTURE = re.compile(r'"(?:[^"\\]++|\\.)*+"|[{}\[\]:]', re.DOTALL)
_ESCAPE = re.compile(r"\\(?:u([0-9A-Fa-f]{4})|.)", re.DOTALL)
_HIGH_SURROGATES = range(0xD800, 0xDC00)
_LOW_SURROGATES = range(0xDC00, 0xE000)
_NUL_LEADING_NAME = '"\\u0000'


class _Refused(ValueError):
    """A literal the contract refuses while decoding."""


def _refuse_constant(literal: str) -> Any:
    raise _Refused(f"{literal} is not JSON")


def _refuse_non_integer(literal: str) -> Any:
    raise _Refused(f"{literal} is not an integer literal")


def _parse_integer(literal: str) -> int | float:
    """A magnitude never fails L1: a literal too long for int() becomes a float (as PHP does); L3 bounds reject it."""
    try:
        return int(literal)
    except ValueError:
        return float(literal)


def _malformed() -> Verdict:
    return Verdict.reject(Layer.PARSE, ReasonCode.MALFORMED_JSON, None)


def payload_too_large() -> Verdict:
    return Verdict.reject(Layer.SIZE, ReasonCode.PAYLOAD_TOO_LARGE, None)


def within_size(raw: bytes, limit: int) -> bool:
    """L0: the raw byte count, insignificant whitespace included, is at most `limit`."""
    return len(raw) <= limit


def is_contract_ascii(raw: bytes) -> bool:
    """Every byte is tab, LF, CR or printable ASCII."""
    return not bytes(raw).translate(None, ALLOWED_BYTES)


def count_tokens(raw: bytes, stop_after: int | None = None) -> int:
    """Tokens of the budget scan, defined on any byte string; counting stops once it passes `stop_after`."""
    count = 0
    for _ in _TOKEN.finditer(bytes(raw)):
        count += 1
        if stop_after is not None and count > stop_after:
            break
    return count


def _surrogates_paired(literal: str) -> bool:
    """Every \\uD800-\\uDBFF escape is immediately followed by a \\uDC00-\\uDFFF escape; no low escape stands alone."""
    low_expected_at: int | None = None
    for escape in _ESCAPE.finditer(literal):
        code = int(escape.group(1), 16) if escape.group(1) is not None else None
        if low_expected_at is not None:
            if escape.start() != low_expected_at or code is None or code not in _LOW_SURROGATES:
                return False
            low_expected_at = None
        elif code is not None and code in _HIGH_SURROGATES:
            low_expected_at = escape.end()
        elif code is not None and code in _LOW_SURROGATES:
            return False
    return low_expected_at is None


def _structure_is_representable(text: str, max_depth: int) -> bool:
    """Over the whole JSON text: nesting within max_depth, surrogate escapes paired, no member name led by U+0000."""
    depth = 0
    previous_string: str | None = None
    for match in _STRUCTURE.finditer(text):
        token = match.group()
        if token[0] == '"':
            if "\\u" in token and not _surrogates_paired(token):
                return False
            previous_string = token
            continue
        if token in "{[":
            depth += 1
            if depth > max_depth:
                return False
        elif token in "}]":
            depth -= 1
        elif previous_string is not None and previous_string.startswith(_NUL_LEADING_NAME):
            return False
        previous_string = None
    return True


def parse(raw: bytes, max_depth: int, max_tokens: int) -> dict[str, Any] | Verdict:
    """L1: the decoded top-level object, or a rejection (malformed_json; payload_too_large over the token budget)."""
    if not is_contract_ascii(raw):
        return _malformed()
    if count_tokens(raw, stop_after=max_tokens) > max_tokens:
        return payload_too_large()
    text = bytes(raw).decode("ascii")
    try:
        document = json.loads(
            text,
            parse_constant=_refuse_constant,
            parse_float=_refuse_non_integer,
            parse_int=_parse_integer,
        )
    except (ValueError, RecursionError):
        return _malformed()
    if not isinstance(document, dict) or not _structure_is_representable(text, max_depth):
        return _malformed()
    return document


def select_kind(document: Mapping[str, Any], context: TrustedContext) -> MessageKind | Verdict:
    """L2: version first, then kind. A storage context reads a manifest; every other channel a message."""
    if context.channel == Channel.STORAGE.value:
        version_field, kind_field = "manifest_version", "document_kind"
        accepted_versions = context.accepted_manifest_versions or ()
    else:
        version_field, kind_field = "protocol_version", "message_kind"
        accepted_versions = context.accepted_protocol_versions or ()

    version = document.get(version_field)
    if not isinstance(version, str) or version not in accepted_versions:
        return Verdict.reject(Layer.VERSION, ReasonCode.UNSUPPORTED_VERSION, f"/{version_field}")
    kind = document.get(kind_field)
    if not isinstance(kind, str) or kind not in KNOWN_KINDS:
        return Verdict.reject(Layer.VERSION, ReasonCode.UNKNOWN_MESSAGE_KIND, f"/{kind_field}")
    if kind not in context.accepted_message_kinds:
        return Verdict.reject(Layer.VERSION, ReasonCode.UNEXPECTED_MESSAGE_KIND, f"/{kind_field}")
    return MessageKind(kind)
