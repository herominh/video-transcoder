"""L0-L2 as README.md sections 3-5 state them. The HUB_* tables and the L0, budget-ordering and L2 tests copy the
Hub's JsonGateTest, VersionGateTest and TranscodeMessagePipelineTest row for row, so both implementations are held
to the same inputs; the EXTRA_* tables hold Python-side edges beyond them."""

import copy
import json
import tracemalloc
from typing import Any

import pytest

from core.protocol import files, gate, pipeline
from core.protocol.context import TrustedContext
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import MessageKind, load_registry
from core.protocol.verdict import Verdict

PUSH = "fixtures/contexts/ctx-hub-push.json"
STORAGE = "fixtures/contexts/ctx-hub-storage.json"
HUB_CLAIM = "fixtures/contexts/ctx-hub-claim.json"
WORKER_GRANT = "fixtures/contexts/ctx-worker-claim-grant.json"
HUB_UNCLAIMED = "fixtures/contexts/ctx-hub-unclaimed-push.json"
PROGRESS_PAYLOAD = "fixtures/progress/P04-downloading.json"
COMPLETED_PAYLOAD = "fixtures/result-completed/P07-full.json"
FAILED_PAYLOAD = "fixtures/result-failed/P11-minimal-stage-null.json"
MANIFEST_PAYLOAD = "fixtures/manifest/P14-encrypted-three-renditions.json"
MAX_NESTING = 32
LARGEST_PUSH_LIMIT = 16_384
PROGRESS_LIMIT = 4_096


def _context(path: str) -> TrustedContext:
    return TrustedContext.from_dict(files.read_json(path))


def _parse(raw: bytes) -> dict[str, Any] | Verdict:
    return gate.parse(raw, MAX_NESTING, load_registry().max_tokens)


def _is_malformed(result: object) -> bool:
    return isinstance(result, Verdict) and (result.layer, result.reason, result.instance_path) == (
        Layer.PARSE,
        ReasonCode.MALFORMED_JSON,
        None,
    )


def _compact_progress() -> bytes:
    return json.dumps(files.read_json(PROGRESS_PAYLOAD), separators=(",", ":")).encode("ascii")


def _progress_with(tail: bytes) -> bytes:
    return _compact_progress()[:-1] + tail + b"}"


# --- L0 ------------------------------------------------------------------------------------------


def test_validate_when_bytes_exceed_the_largest_accepted_limit_should_refuse_them_before_parsing():
    # Arrange: not JSON at all, so only a size refusal proves nothing parsed it.
    raw = b"x" * (LARGEST_PUSH_LIMIT + 1)

    # Act
    verdict = pipeline.validate(raw, _context(PUSH))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (Layer.SIZE, ReasonCode.PAYLOAD_TOO_LARGE, None)


def test_validate_when_bytes_fit_the_largest_accepted_limit_should_reach_the_parser():
    # Arrange
    raw = b"x" * LARGEST_PUSH_LIMIT

    # Act
    verdict = pipeline.validate(raw, _context(PUSH))

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.PARSE, ReasonCode.MALFORMED_JSON)


def test_validate_when_a_progress_event_exceeds_its_own_limit_but_not_the_largest_should_refuse_it_after_its_kind():
    # Arrange: a valid progress event padded past 4,096 bytes, well within 16,384.
    valid = files.read_bytes(PROGRESS_PAYLOAD)
    padded = valid.rstrip().ljust(PROGRESS_LIMIT + 1, b" ")

    # Act
    verdict_valid = pipeline.validate(valid, _context(PUSH))
    verdict_padded = pipeline.validate(padded, _context(PUSH))

    # Assert
    assert verdict_valid.accepted
    assert (verdict_padded.layer, verdict_padded.reason) == (Layer.SIZE, ReasonCode.PAYLOAD_TOO_LARGE)


def test_validate_when_a_result_is_as_large_as_the_progress_limit_should_not_be_refused_for_size():
    # Arrange: a valid completed result padded to 4,097 bytes, within its own 16,384.
    valid = files.read_bytes(COMPLETED_PAYLOAD)
    padded = valid.rstrip().ljust(max(PROGRESS_LIMIT + 1, len(valid)), b" ")

    # Act
    verdict = pipeline.validate(padded, _context(PUSH))

    # Assert
    assert verdict.accepted


# --- L1: the Hub's JsonGateTest rows, copied row for row ------------------------------------------

HUB_MALFORMED = {
    "nesting 33": b'{"x_deep":' + b"[" * 32 + b"]" * 32 + b"}",
    "a UTF-8 byte order mark before the object": b'\xef\xbb\xbf{"a":1}',
    "the byte 0xFF inside a string": b'{"message":"x\xffy"}',
    "a two-byte character inside a string": b'{"message":"caf\xc3\xa9"}',
    "an overlong encoding of /": b'{"message":"\xc0\xaf"}',
    "an encoded UTF-16 surrogate": b'{"message":"\xed\xa0\x80"}',
    "a code point above U+10FFFF": b'{"message":"\xf4\x90\x80\x80"}',
    "a five-byte lead byte": b'{"message":"\xf8\x88\x80\x80\x80"}',
    "a non-breaking space as whitespace": b'\xc2\xa0{"a":1}',
    "a NUL byte": b'{"a":1}\x00',
    "a backspace byte": b'{"a":"\x08"}',
    "a vertical tab as whitespace": b'\x0b{"a":1}',
    "a form feed as whitespace": b'\x0c{"a":1}',
    "an escape byte": b'{"a":"\x1b"}',
    "a unit separator byte": b'{"a":"\x1f"}',
    "a DEL byte": b'{"a":"\x7f"}',
    "a NaN literal": b'{"stage_progress_pct":NaN}',
    "an Infinity literal": b'{"stage_progress_pct":Infinity}',
    "a -Infinity literal": b'{"stage_progress_pct":-Infinity}',
    "JSON cut mid-array": b'{"artifacts":[{"path":"master.m3u8"},',
    "two top-level values": b'{"a":1}{"b":2}',
    "a top-level array": b'[{"a":1}]',
    "a top-level string": b'"transcode.progress"',
    "a top-level number": b"42",
    "a top-level null": b"null",
    "no bytes at all": b"",
    "whitespace only": b" \n\t",
    "an unpaired surrogate escape in a value": b'{"message":"\\ud800"}',
    "an unpaired low surrogate escape": b'{"message":"\\udc00"}',
    "an unpaired surrogate escape in a member name": b'{"\\ud800":1}',
    "a member name starting with U+0000": b'{"\\u0000x":1}',
    "a member name that is U+0000": b'{"\\u0000":1}',
    "a member name starting with U+0000 in an object inside an array": b'{"a":[{"\\u0000":1}]}',
    "a high surrogate escape followed by another high one": b'{"a":"\\ud800\\ud800\\udc00"}',
    "a high surrogate escape followed by a non-surrogate escape": b'{"a":"\\ud800\\u0041"}',
    "a low surrogate escape before a high one": b'{"a":"\\udc00\\ud800"}',
    "an unpaired uppercase low surrogate escape": b'{"a":"\\uDC00"}',
    "a high surrogate escape ending the string": b'{"a":"x\\ud800"}',
    "a comment": b'{"a":1 /* c */}',
    "a trailing comma": b'{"a":1,}',
    "single quotes": b"{'a':1}",
    "an unescaped tab inside a string": b'{"message":"a\tb"}',
    "a leading zero": b'{"a":01}',
}

HUB_MALFORMED_OVERWRITTEN_MEMBERS = {
    "an unpaired surrogate escape": b'{"message":"\\ud800","message":"ok"}',
    "nesting 33": b'{"ext":' + b"[" * 32 + b"0" + b"]" * 32 + b',"ext":{}}',
    "a member name starting with U+0000": b'{"ext":{"\\u0000x":1},"ext":{}}',
    "a non-ASCII byte": b'{"message":"caf\xc3\xa9","message":"ok"}',
    "a fraction": b'{"event_seq":1.5,"event_seq":2}',
}

HUB_NON_INTEGER_LITERALS = {
    "an integer written with a zero fraction": b'{"stage_progress_pct":50.0}',
    "an exponent": b'{"stage_progress_pct":1e2}',
    "an uppercase exponent": b'{"stage_progress_pct":1E2}',
    "a fraction that rounds to an integer": b'{"execution_fence":1.9999999999999999}',
    "a fraction just above a bound": b'{"stage_progress_pct":100.00000000000000000001}',
    "a plain fraction": b'{"stage_progress_pct":0.5}',
    "a negative zero with a fraction": b'{"a":-0.0}',
    "a negative exponent": b'{"a":1e-2}',
    "an infinite magnitude": b'{"a":1e400}',
    "a fraction inside an array": b'{"a":[1,2,3.5]}',
}

HUB_WELL_FORMED = {
    "JSON whitespace around the object": b' \r\n\t{"a":1}\r\n ',
    "a paired surrogate escape": b'{"message":"\\ud83d\\ude00"}',
    "a paired surrogate escape in uppercase and mixed-case hex": b'{"a":"\\uDBFF\\uDFFF\\uD83d\\uDe00"}',
    "a string value starting with U+0000": b'{"a":"\\u0000x"}',
    "an empty member name": b'{"":1}',
    "U+0000 inside, not leading, a member name": b'{"x\\u0000":1}',
    "a duplicate member name": b'{"a":1,"a":2}',
    "an integer too large for the platform": b'{"a":123456789012345678901234567890}',
    "zero and negative zero": b'{"a":0,"b":-0}',
    "digits, dots and exponents inside strings": b'{"a":"1.5e+10","b":"-0.0"}',
    "nesting 32": b'{"x_deep":' + b"[" * 31 + b"1" + b"]" * 31 + b"}",
}

HUB_TOKEN_COUNTS = {
    "an object with one string member": (b'{"a":"b"}', 3),
    "brackets inside a string count nothing": (b'{"a":"{[1,2,3]}"}', 3),
    "an escaped quote stays inside the string": (b'{"a":"x\\"y"}', 3),
    "an escaped backslash ends before the closing quote": (b'{"a":"\\\\"}', 3),
    "a number runs over its sign, fraction and exponent": (b'{"a":-1.5e+10}', 3),
    "the three literals": (b'{"a":[true,false,null]}', 6),
    "closing brackets, commas, colons and whitespace count nothing": (b'{ "a" : [ ] , "b" : { } }\n', 5),
    "a literal runs over every lowercase letter": (b"[nullnull,truex]", 3),
    "an unterminated string counts one": (b'"abc', 1),
    "a string ending in a lone backslash counts one": (b'"ab\\', 1),
    "bytes outside the starting set count nothing": (b"}]:, abcdeghijklmopqrsuvwxyz+.", 0),
}

# Rows beyond the Hub's tables (Python-side edges of the raw-text scan).
EXTRA_MALFORMED = {
    "a high surrogate escape followed by a letter, then a low one": b'{"message":"\\ud800x\\udc00"}',
    "valid UTF-8 that is not ASCII": b'{"message":"\xc3\xa9"}',
    "an overwritten member holding an exponent": b'{"a":1e2,"a":1}',
}
EXTRA_WELL_FORMED = {
    "an escaped backslash before u": b'{"x":"\\\\ud800"}',
    "U+0000 leading a string value": b'{"x":"\\u0000"}',
}
EXTRA_TOKEN_COUNTS = {
    "a backslash outside a string": (b'\\"x"', 1),
    "t, f and n as one literal": (b"tfn", 1),
    "every literal kind and negative zero": (b"[true,false,null,-0]", 5),
}

AMPLE_TOKENS = 10_000
BUDGET_BOUND_BYTES = 64 * 1024 * 1024


def _read(raw: bytes, max_tokens: int = AMPLE_TOKENS) -> dict[str, Any] | Verdict:
    return gate.parse(raw, MAX_NESTING, max_tokens)


def _is_too_large(result: object) -> bool:
    return isinstance(result, Verdict) and (result.layer, result.reason, result.instance_path) == (
        Layer.SIZE,
        ReasonCode.PAYLOAD_TOO_LARGE,
        None,
    )


@pytest.mark.parametrize(
    "raw",
    [*HUB_MALFORMED.values(), *HUB_MALFORMED_OVERWRITTEN_MEMBERS.values(), *HUB_NON_INTEGER_LITERALS.values(),
     *EXTRA_MALFORMED.values()],
    ids=[*HUB_MALFORMED, *(f"overwritten: {name}" for name in HUB_MALFORMED_OVERWRITTEN_MEMBERS),
         *(f"number: {name}" for name in HUB_NON_INTEGER_LITERALS), *(f"extra: {name}" for name in EXTRA_MALFORMED)],
)
def test_parse_when_bytes_are_malformed_should_refuse_them_at_the_parse_layer(raw):
    # Act
    result = _read(raw)

    # Assert
    assert _is_malformed(result), result


@pytest.mark.parametrize(
    "raw",
    [*HUB_WELL_FORMED.values(), *EXTRA_WELL_FORMED.values()],
    ids=[*HUB_WELL_FORMED, *(f"extra: {name}" for name in EXTRA_WELL_FORMED)],
)
def test_parse_when_bytes_are_well_formed_at_an_edge_should_return_the_object(raw):
    # Act
    result = _read(raw)

    # Assert
    assert isinstance(result, dict), result


def test_parse_when_bytes_hold_an_object_should_keep_empty_objects_and_empty_arrays_apart():
    # Act
    result = _read(b'{"ext":{},"renditions":[],"stage_progress_pct":50}')

    # Assert
    assert result == {"ext": {}, "renditions": [], "stage_progress_pct": 50}
    assert isinstance(result["ext"], dict) and isinstance(result["renditions"], list)


def test_parse_when_a_member_name_repeats_should_keep_the_last_value():
    # Act
    result = _read(b'{"event_seq":4,"event_seq":5}')

    # Assert
    assert result == {"event_seq": 5}


def test_parse_when_nested_thousands_deep_should_refuse_without_recursion_error():
    # Arrange
    raw = b'{"x":' + b"[" * 5_000 + b"]" * 5_000 + b"}"

    # Act
    result = _read(raw)

    # Assert
    assert _is_malformed(result)


def test_validate_when_an_overwritten_member_holds_an_unpaired_surrogate_should_refuse_the_message():
    # Arrange: the surviving value is valid, so only a whole-text check refuses it.
    raw = _progress_with(b',"message":"\\ud800","message":"ok"')

    # Act
    verdict = pipeline.validate(raw, _context(PUSH))

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.PARSE, ReasonCode.MALFORMED_JSON)


def test_validate_when_an_integer_literal_is_too_long_for_int_parsing_should_fail_the_schema_not_the_parse():
    # Arrange
    text = files.read_bytes(FAILED_PAYLOAD).decode("ascii")
    raw = text.replace('"placement_epoch": 7', '"placement_epoch": ' + "9" * 5_000, 1).encode("ascii")

    # Act
    verdict = pipeline.validate(raw, _context(PUSH))

    # Assert
    assert (verdict.layer, verdict.instance_path) == (Layer.SCHEMA, "/identity/placement_epoch")


def test_validate_when_a_paired_surrogate_escape_passes_parsing_should_fail_the_ascii_schema():
    # Arrange
    raw = _progress_with(b',"message":"\\ud83d\\ude00"')

    # Act
    verdict = pipeline.validate(raw, _context(PUSH))

    # Assert
    assert (verdict.layer, verdict.instance_path) == (Layer.SCHEMA, "/message")


# --- L1: the token budget --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [*HUB_TOKEN_COUNTS.values(), *EXTRA_TOKEN_COUNTS.values()],
    ids=[*HUB_TOKEN_COUNTS, *(f"extra: {name}" for name in EXTRA_TOKEN_COUNTS)],
)
def test_count_tokens_when_scanning_text_should_count_by_the_contract_rules(raw, expected):
    # Act
    count = gate.count_tokens(raw)

    # Assert
    assert count == expected


def test_count_tokens_when_a_limit_is_passed_should_stop_just_past_it():
    # Act
    count = gate.count_tokens(b"[" * 1_000_000, stop_after=10)

    # Assert
    assert count == 11


def test_parse_when_the_token_count_equals_the_budget_should_accept_and_one_more_should_refuse_as_too_large():
    # Arrange: { "a" [ 1 true "x\"y" -2 null - eight tokens.
    raw = b'{"a":[1,true,"x\\"y",-2,null]}'

    # Act
    at_budget = _read(raw, max_tokens=8)
    over_budget = _read(raw, max_tokens=7)

    # Assert
    assert isinstance(at_budget, dict)
    assert _is_too_large(over_budget)


def test_parse_when_malformed_text_is_over_the_budget_should_refuse_it_as_too_large_before_decoding():
    # Arrange: six `[` count six tokens whether or not the text is JSON.
    raw = b"[[[[[["

    # Act
    over_budget = _read(raw, max_tokens=5)
    within_budget = _read(raw, max_tokens=6)

    # Assert
    assert _is_too_large(over_budget)
    assert _is_malformed(within_budget)


def test_parse_when_a_fraction_is_over_the_budget_should_refuse_it_as_too_large_first():
    # Arrange: { "a" [ 1.5 2 - five tokens, one of them no integer literal.
    raw = b'{"a":[1.5,2]}'

    # Act
    over_budget = _read(raw, max_tokens=4)
    within_budget = _read(raw, max_tokens=5)

    # Assert
    assert _is_too_large(over_budget)
    assert _is_malformed(within_budget)


def test_parse_when_a_non_ascii_text_is_over_the_budget_should_refuse_it_as_malformed_first():
    # Act
    result = _read(b"[[[[[[\xc3\xa9", max_tokens=3)

    # Assert
    assert _is_malformed(result)


def _manifest_with_filler_tokens(total_tokens: int) -> bytes:
    """P14 whose artifacts array is `[0,0,...]`, sized so the whole text holds `total_tokens` tokens."""
    document = copy.deepcopy(files.read_json(MANIFEST_PAYLOAD))
    document["artifacts"] = []
    base = gate.count_tokens(json.dumps(document, separators=(",", ":")).encode("ascii"))
    document["artifacts"] = [0] * (total_tokens - base)
    return json.dumps(document, separators=(",", ":")).encode("ascii")


def test_validate_when_tokens_equal_the_budget_should_reach_the_schema_layer():
    # Arrange
    raw = _manifest_with_filler_tokens(load_registry().max_tokens)

    # Act
    verdict = pipeline.validate(raw, _context(STORAGE))

    # Assert: decoded and judged by the schema (the array is far over its maxItems).
    assert (verdict.layer, verdict.reason) == (Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION)


def test_validate_when_tokens_exceed_the_budget_should_refuse_for_size_before_decoding():
    # Arrange
    raw = _manifest_with_filler_tokens(load_registry().max_tokens + 1)

    # Act
    verdict = pipeline.validate(raw, _context(STORAGE))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (Layer.SIZE, ReasonCode.PAYLOAD_TOO_LARGE, None)


def test_validate_when_text_over_the_budget_holds_a_non_ascii_byte_should_refuse_it_as_malformed():
    # Arrange: the ASCII check runs before the token count.
    raw = _manifest_with_filler_tokens(load_registry().max_tokens + 1)[:-1] + b',"x":"\xc3\xa9"}'

    # Act
    verdict = pipeline.validate(raw, _context(STORAGE))

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.PARSE, ReasonCode.MALFORMED_JSON)


def _traced_peak_bytes(raw: bytes, context_path: str) -> int:
    tracemalloc.start()
    try:
        pipeline.validate(raw, _context(context_path))
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_validate_when_one_string_literal_fills_the_byte_limit_should_stay_within_64_mib():
    # Arrange: a single 8 MiB string literal, the longest a storage payload may hold.
    limit = load_registry().max_bytes(MessageKind.GENERATION_MANIFEST)
    raw = b'{"a":"' + b"a" * (limit - 8) + b'"}'

    # Act
    peak = _traced_peak_bytes(raw, STORAGE)

    # Assert
    assert len(raw) == limit
    assert peak < BUDGET_BOUND_BYTES, peak


def test_validate_when_an_overwritten_member_holds_a_2_mib_string_should_stay_within_64_mib():
    # Arrange: P14 plus a 2 MiB `diagnostics` string that a later duplicate overwrites (accepted by both sides).
    manifest = json.dumps(files.read_json(MANIFEST_PAYLOAD), separators=(",", ":")).encode("ascii")
    raw = manifest[:-1] + b',"diagnostics":"' + b"a" * 2_097_152 + b'","diagnostics":[]}'

    # Act
    peak = _traced_peak_bytes(raw, STORAGE)

    # Assert
    assert pipeline.validate(raw, _context(STORAGE)).accepted
    assert peak < BUDGET_BOUND_BYTES, peak


# --- L2 --------------------------------------------------------------------------------------------

PROGRESS_HEADER = {"protocol_version": "2.0.0-draft", "message_kind": "transcode.progress"}
MANIFEST_HEADER = {"manifest_version": "1.0.0-draft", "document_kind": "generation.manifest"}
REFUSALS = {
    "no version": (PUSH, {"message_kind": "transcode.progress"}, "unsupported_version", "/protocol_version"),
    "a null version": (PUSH, {**PROGRESS_HEADER, "protocol_version": None}, "unsupported_version",
                       "/protocol_version"),
    "a version that is a number": (PUSH, {**PROGRESS_HEADER, "protocol_version": 2}, "unsupported_version",
                                   "/protocol_version"),
    "the release named without -draft": (PUSH, {**PROGRESS_HEADER, "protocol_version": "2.0.0"},
                                         "unsupported_version", "/protocol_version"),
    "another major": (PUSH, {**PROGRESS_HEADER, "protocol_version": "3.0.0-draft"}, "unsupported_version",
                      "/protocol_version"),
    "the draft in another letter case": (PUSH, {**PROGRESS_HEADER, "protocol_version": "2.0.0-DRAFT"},
                                         "unsupported_version", "/protocol_version"),
    "a wrong version and an unknown kind": (PUSH, {"protocol_version": "9.9.9", "message_kind": "x"},
                                            "unsupported_version", "/protocol_version"),
    "no kind": (PUSH, {"protocol_version": "2.0.0-draft"}, "unknown_message_kind", "/message_kind"),
    "a kind that is a number": (PUSH, {**PROGRESS_HEADER, "message_kind": 7}, "unknown_message_kind",
                                "/message_kind"),
    "an unknown kind": (PUSH, {**PROGRESS_HEADER, "message_kind": "transcode.heartbeat"}, "unknown_message_kind",
                        "/message_kind"),
    "a known kind this receiver does not accept": (PUSH, {**PROGRESS_HEADER, "message_kind": "transcode.request"},
                                                   "unexpected_message_kind", "/message_kind"),
    "a manifest posted to a message receiver": (PUSH, {"protocol_version": "2.0.0-draft",
                                                       "message_kind": "generation.manifest"},
                                                "unexpected_message_kind", "/message_kind"),
    "a claim posted to the Hub callback": (PUSH, {**PROGRESS_HEADER, "message_kind": "transcode.claim"},
                                           "unexpected_message_kind", "/message_kind"),
    "a claim grant posted to the Hub callback": (PUSH, {**PROGRESS_HEADER,
                                                        "message_kind": "transcode.claim.granted"},
                                                 "unexpected_message_kind", "/message_kind"),
    "an unclaimed report where the receiver does not list it": (PUSH, {**PROGRESS_HEADER,
                                                                       "message_kind": "transcode.unclaimed"},
                                                                "unexpected_message_kind", "/message_kind"),
    "a message read from storage": (STORAGE, PROGRESS_HEADER, "unsupported_version", "/manifest_version"),
    "a manifest of another version": (STORAGE, {**MANIFEST_HEADER, "manifest_version": "2.0.0"},
                                      "unsupported_version", "/manifest_version"),
    "a manifest version carrying a message kind": (STORAGE, {**MANIFEST_HEADER,
                                                             "document_kind": "transcode.progress"},
                                                   "unexpected_message_kind", "/document_kind"),
    "a manifest without its kind": (STORAGE, {"manifest_version": "1.0.0-draft",
                                              "message_kind": "generation.manifest"},
                                    "unknown_message_kind", "/document_kind"),
}


@pytest.mark.parametrize(("context", "fields", "reason", "pointer"), list(REFUSALS.values()), ids=list(REFUSALS))
def test_select_when_version_or_kind_is_not_accepted_should_refuse_with_the_reason_at_the_field(
    context, fields, reason, pointer
):
    # Act
    selected = gate.select_kind(dict(fields), _context(context))

    # Assert
    assert isinstance(selected, Verdict)
    assert (selected.layer, selected.reason, selected.instance_path) == (Layer.VERSION, ReasonCode(reason), pointer)


@pytest.mark.parametrize(
    ("context", "fields", "kind"),
    [
        (PUSH, {"protocol_version": "2.0.0-draft", "message_kind": "transcode.result.failed"},
         MessageKind.TRANSCODE_RESULT_FAILED),
        (STORAGE, MANIFEST_HEADER, MessageKind.GENERATION_MANIFEST),
        (PUSH, {**PROGRESS_HEADER, "callback_url": "x"}, MessageKind.TRANSCODE_PROGRESS),
        (HUB_CLAIM, {**PROGRESS_HEADER, "message_kind": "transcode.claim"}, MessageKind.TRANSCODE_CLAIM),
        (WORKER_GRANT, {**PROGRESS_HEADER, "message_kind": "transcode.claim.granted"},
         MessageKind.TRANSCODE_CLAIM_GRANTED),
        (HUB_UNCLAIMED, {**PROGRESS_HEADER, "message_kind": "transcode.unclaimed"},
         MessageKind.TRANSCODE_UNCLAIMED),
    ],
    ids=["a message", "a manifest from storage", "unknown fields left to the schema layer", "a claim at the Hub",
         "a claim grant at the worker", "an unclaimed report at the Hub"],
)
def test_select_when_version_and_kind_are_accepted_should_return_the_kind(context, fields, kind):
    # Act
    selected = gate.select_kind(dict(fields), _context(context))

    # Assert
    assert selected is kind
