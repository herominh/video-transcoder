"""The gate's two scanning patterns match exactly what their possessive originals matched.

The originals need Python 3.11 (possessive quantifiers) and the worker image runs 3.10, so the gate now
uses the unrolled-loop form; this test compiles the originals and compares every match span on 3.11 and
later (it is skipped below, where the originals cannot be compiled)."""

import random
import re
import sys

import pytest

from tests.contract import gate

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="the possessive originals compile only on Python 3.11 and later"
)

ORIGINAL_TOKEN = rb'"(?:[^"\\]++|\\.)*+"?|[{\[]|[-0-9][-0-9+.eE]*+|[tfn][a-z]*+'
ORIGINAL_STRUCTURE = r'"(?:[^"\\]++|\\.)*+"|[{}\[\]:]'
LONG_RUN_CHARS = 100_000
RANDOM_SEED = 20261009
RANDOM_CASES = 3_000
RANDOM_MAX_CHARS = 24
# The characters every pattern treats specially, and a few it does not.
RANDOM_ALPHABET = '"\\{}[]:,-+.eE0123456789tfnaluxz \n\t'
CORPUS = [
    "",
    '""',
    '"',
    '"abc',
    '"abc\\',
    "\\",
    '"\\"',
    '"\\""',
    '"\\\\"',
    '"\\\\\\""',
    '"\\\\\\"',
    '"a\\"b"c"',
    '"\\u00e9\\uD83D\\uDE00"',
    '"\\u"',
    "-0",
    "1e+5",
    "--",
    "-",
    "0.5E-3x",
    "true",
    "nul",
    "null",
    "tfn",
    "falsey",
    '[[{"a":[1,{"b":null}]}]]',
    '{"a":"b","c":[true,false,-1.5e10]}',
    'text outside "strings" and {more} text',
    ' \t\n:,"x":"y"',
    '{"k":"' + "a" * LONG_RUN_CHARS + '"}',
    '"' + "a" * LONG_RUN_CHARS,
    '"' + '\\"' * 1000 + '"',
    '"' + "\\\\" * 1000,
]


def _random_corpus() -> list[str]:
    generator = random.Random(RANDOM_SEED)
    return [
        "".join(generator.choice(RANDOM_ALPHABET) for _ in range(generator.randrange(RANDOM_MAX_CHARS + 1)))
        for _ in range(RANDOM_CASES)
    ]


def _spans(pattern: re.Pattern, text: str | bytes) -> list[tuple[int, int]]:
    return [match.span() for match in pattern.finditer(text)]


def _mismatches(texts: list[str]) -> list[str]:
    """The texts on which either pattern matches differently from its original."""
    original_token = re.compile(ORIGINAL_TOKEN, re.DOTALL)
    original_structure = re.compile(ORIGINAL_STRUCTURE, re.DOTALL)
    return [
        text
        for text in texts
        if _spans(gate._TOKEN, text.encode("ascii")) != _spans(original_token, text.encode("ascii"))
        or _spans(gate._STRUCTURE, text) != _spans(original_structure, text)
    ]


@pytest.mark.parametrize("text", CORPUS)
def test_gate_patterns_when_scanning_a_tricky_text_should_match_exactly_what_the_possessive_originals_matched(text):
    # Arrange
    original_token = re.compile(ORIGINAL_TOKEN, re.DOTALL)
    original_structure = re.compile(ORIGINAL_STRUCTURE, re.DOTALL)
    raw = text.encode("ascii")

    # Act
    token_spans = (_spans(gate._TOKEN, raw), _spans(original_token, raw))
    structure_spans = (_spans(gate._STRUCTURE, text), _spans(original_structure, text))

    # Assert
    assert token_spans[0] == token_spans[1]
    assert structure_spans[0] == structure_spans[1]


def test_gate_patterns_when_scanning_random_short_texts_should_match_exactly_what_the_possessive_originals_matched():
    # Arrange
    texts = _random_corpus()

    # Act
    mismatches = _mismatches(texts)

    # Assert
    assert mismatches == []
