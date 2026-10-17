"""The input's rules beyond its schema (README section 3): each refused once, the contract's valid inputs accepted,
and the configured store's host read strictly. A refusal names the rule, never what the input held."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from core.drive_import import contract
from core.drive_import.input_check import InputRefused, check, storage_host_from

STORAGE_HOST = "storage.example.invalid"
NOW = datetime(2026, 10, 17, 8, 0, 0, tzinfo=timezone.utc)
RUNNING_EXAMPLE = "fixtures/input/valid-running-example.json"
ONE_PART = "fixtures/input/valid-one-part-with-resource-key.json"
LARGEST = "fixtures/input/valid-largest.json"
# The running example's deadline (writes_end_at), as the fixture writes it.
DEADLINE = datetime(2026, 10, 17, 9, 16, 53, tzinfo=timezone.utc)
SIGNATURE_MARK = "test-signature"


def _fixture(name: str = RUNNING_EXAMPLE) -> dict[str, Any]:
    return copy.deepcopy(contract.read_json(name))


def _with_part_url(n: int, change: Callable[[str], str]) -> dict[str, Any]:
    document = _fixture()
    part = next(part for part in document["parts"] if part["n"] == n)
    part["url"] = change(part["url"])
    return document


def _with_receipt_url(change: Callable[[str], str]) -> dict[str, Any]:
    document = _fixture()
    document["receipt_url"] = change(document["receipt_url"])
    return document


def _replace(old: str, new: str) -> Callable[[str], str]:
    def change(url: str) -> str:
        assert old in url
        return url.replace(old, new, 1)

    return change


def _parts_renumbered(numbers: list[int]) -> dict[str, Any]:
    document = _fixture()
    document["parts"] = [
        {"n": number, "url": part["url"].replace(f"partNumber={part['n']}&", f"partNumber={number}&")}
        for part, number in zip(document["parts"], numbers)
    ]
    return document


def _parts_total(total: int) -> dict[str, Any]:
    document = _fixture()
    document["plan"]["parts_total"] = total
    return document


def _size(size: int) -> dict[str, Any]:
    document = _fixture()
    document["expected"]["size"] = size
    return document


def _part_size(part_size: int) -> dict[str, Any]:
    document = _fixture()
    document["plan"]["part_size"] = part_size
    return document


BUCKET = "/videohub-school-a/"
VIDEO = "/0199f1a0-1111-7000-8000-000000000001/"
REFUSED = [
    pytest.param(_with_part_url(2, _replace(STORAGE_HOST, "other.example.invalid")), id="a-part-url-of-another-host"),
    pytest.param(_with_receipt_url(_replace(STORAGE_HOST, "storage.example.invalid.attacker.invalid")),
                 id="the-receipt-url-of-another-host"),
    pytest.param(_with_part_url(1, _replace("https://", "http://")), id="a-part-url-over-http"),
    pytest.param(_with_part_url(1, _replace("https://", "https://user@")), id="a-part-url-with-user-information"),
    pytest.param(_with_part_url(1, _replace("https://", "https://user:secret@")), id="a-part-url-with-a-password"),
    pytest.param(_with_part_url(1, _replace(STORAGE_HOST, STORAGE_HOST + ":8443")), id="a-part-url-with-a-port"),
    pytest.param(_with_part_url(1, _replace(STORAGE_HOST, STORAGE_HOST + ":443")), id="a-part-url-with-the-https-port"),
    pytest.param(_with_part_url(3, _replace(BUCKET, "/videohub-school-b/")), id="a-part-url-of-another-bucket"),
    pytest.param(_with_receipt_url(_replace(BUCKET, "/videohub-school-b/")), id="the-receipt-url-of-another-bucket"),
    pytest.param(_with_part_url(3, _replace(VIDEO, "/0199f1a0-1111-7000-8000-000000000009/")),
                 id="a-part-url-of-another-video"),
    pytest.param(_with_receipt_url(_replace(VIDEO, "/0199f1a0-1111-7000-8000-000000000009/")),
                 id="the-receipt-url-of-another-video"),
    pytest.param(_with_part_url(1, _replace("/original-videos/", "/videos/")), id="a-part-url-outside-the-originals"),
    pytest.param(_with_part_url(1, _replace(VIDEO, VIDEO + "../")), id="a-part-url-with-a-dot-dot-segment"),
    pytest.param(_with_part_url(1, _replace("/original.mp4", "/..")), id="a-part-url-naming-dot-dot"),
    pytest.param(_with_part_url(1, _replace("/original.mp4", "/original%2Emp4")), id="a-part-url-with-an-encoded-name"),
    pytest.param(_with_part_url(1, _replace(VIDEO, "/0199f1a0-1111-7000-8000-000000000001%2F/")),
                 id="a-part-url-with-an-encoded-slash"),
    pytest.param(_with_part_url(1, _replace(STORAGE_HOST + "/", STORAGE_HOST + "//")),
                 id="a-part-url-with-an-empty-segment"),
    pytest.param(_with_part_url(1, _replace("/original.mp4", "/original.exe")), id="a-part-url-naming-another-type"),
    pytest.param(_with_part_url(1, _replace("/original.mp4", "/other.mp4")), id="a-part-url-naming-another-object"),
    pytest.param(_with_part_url(2, _replace("/original.mp4", "/original.mov")),
                 id="the-part-urls-naming-two-originals"),
    pytest.param(_with_part_url(2, _replace("partNumber=2&", "partNumber=3&")), id="a-part-url-naming-another-part"),
    pytest.param(_with_part_url(2, _replace("partNumber=2&", "")), id="a-part-url-naming-no-part"),
    pytest.param(_with_part_url(2, _replace("partNumber=2&", "partNumber=2&partNumber=2&")),
                 id="a-part-url-naming-its-part-twice"),
    pytest.param(_with_part_url(2, _replace("uploadId=upload-1", "uploadId=upload-1&uploadId=upload-2")),
                 id="a-part-url-naming-a-second-upload"),
    pytest.param(_with_part_url(2, _replace("uploadId=upload-1", "uploadId=upload-2")),
                 id="the-part-urls-naming-two-uploads"),
    pytest.param(_with_part_url(2, _replace("uploadId=upload-1&", "")), id="a-part-url-naming-no-upload"),
    pytest.param(_with_part_url(2, _replace("uploadId=upload-1", "uploadId=")), id="a-part-url-naming-an-empty-upload"),
    pytest.param(_with_receipt_url(_replace("?", "?uploadId=upload-1&")), id="the-receipt-url-naming-an-upload"),
    pytest.param(_with_receipt_url(_replace("?", "?partNumber=1&")), id="the-receipt-url-naming-a-part"),
    pytest.param(_with_receipt_url(_replace("0199f1a0-2222-7000-8000-000000000002.json",
                                            "0199f1a0-2222-7000-8000-000000000003.json")),
                 id="the-receipt-url-naming-another-jobs-receipt"),
    pytest.param(_with_receipt_url(_replace("/import-receipt-", "/original.mp4?x=/import-receipt-")),
                 id="the-receipt-url-naming-the-original"),
    pytest.param(_parts_renumbered([*range(1, 15), 14]), id="a-part-number-twice"),
    pytest.param(_parts_renumbered([*range(1, 15), 16]), id="a-part-number-past-the-total"),
    pytest.param(_parts_renumbered(list(range(2, 17))), id="part-numbers-not-from-one"),
    pytest.param(_parts_total(16), id="a-parts-total-above-the-sizes"),
    pytest.param(_parts_total(14), id="a-parts-total-below-the-sizes"),
    pytest.param(_size(15 * 67_108_864 + 1), id="a-size-one-byte-past-the-plan"),
    pytest.param(_size(14 * 67_108_864), id="a-size-one-part-short-of-the-plan"),
    pytest.param(_part_size(67_108_864 // 2), id="a-part-size-not-the-contracts"),
    pytest.param(_with_part_url(1, lambda url: url + "#fragment"), id="a-part-url-with-a-fragment"),
    pytest.param(_with_part_url(1, _replace("/original.mp4", "\\original.mp4")), id="a-part-url-with-a-backslash"),
    pytest.param(_with_receipt_url(lambda url: url + "#"), id="the-receipt-url-with-a-fragment"),
    pytest.param(_with_part_url(1, _replace("?", "? ")), id="a-part-url-with-a-space"),
    pytest.param(_with_part_url(1, _replace("?", "?a-field-without-a-value&")), id="a-part-url-with-a-malformed-query"),
    pytest.param(_with_part_url(1, _replace(STORAGE_HOST, "[1AbCdEfGhIjK]")), id="a-part-url-with-a-bracketed-host"),
    pytest.param(_with_part_url(1, _replace(STORAGE_HOST, STORAGE_HOST + "]")), id="a-part-url-with-a-lone-bracket"),
    pytest.param(_with_receipt_url(_replace(STORAGE_HOST, "[::1]")), id="the-receipt-url-with-an-ipv6-host"),
    pytest.param(_with_part_url(1, _replace("?", "?a=[b]&")), id="a-part-url-with-brackets-in-its-query"),
    pytest.param(_with_part_url(1, _replace("&uploadId", "&&uploadId")), id="a-part-url-with-an-empty-query-field"),
]


@pytest.mark.parametrize("document", REFUSED)
def test_check_when_the_input_breaks_a_rule_beyond_its_schema_should_refuse_it_without_quoting_it(document):
    # Act
    with pytest.raises(InputRefused) as refused:
        check(document, STORAGE_HOST, NOW)

    # Assert: the refusal names a rule; nothing of the URLs, ids or signatures is in it, nor travels with it as the
    # error it was raised in place of (whose text, a parser's, can quote the URL)
    text = str(refused.value)
    assert text != ""
    assert "https://" not in text and SIGNATURE_MARK not in text and STORAGE_HOST not in text
    assert refused.value.__cause__ is None and refused.value.__context__ is None


@pytest.mark.parametrize(
    "deadline",
    [
        pytest.param("2026-02-30T09:16:53Z", id="a-day-the-month-has-not"),
        pytest.param("2026-13-01T00:00:00Z", id="month-13"),
    ],
)
def test_check_when_the_deadline_is_no_calendar_time_should_refuse_it_without_a_context(deadline):
    # Arrange
    document = _fixture()
    document["writes_end_at"] = deadline

    # Act
    with pytest.raises(InputRefused) as refused:
        check(document, STORAGE_HOST, NOW)

    # Assert
    assert refused.value.__context__ is None and refused.value.__cause__ is None


@pytest.mark.parametrize(
    "now",
    [pytest.param(DEADLINE, id="at-the-deadline"), pytest.param(DEADLINE + timedelta(seconds=1), id="after-it")],
)
def test_check_when_the_writer_deadline_has_come_should_refuse_the_input(now):
    # Act / Assert
    with pytest.raises(InputRefused, match="deadline"):
        check(_fixture(), STORAGE_HOST, now)


def test_check_when_now_is_not_an_aware_time_should_raise_a_value_error():
    # Act / Assert: a clock that gives a naive time is a bug of the caller, not a refusal of the input
    with pytest.raises(ValueError):
        check(_fixture(), STORAGE_HOST, NOW.replace(tzinfo=None))


def test_check_when_the_input_is_the_running_example_should_return_what_the_job_acts_on():
    # Act
    checked = check(_fixture(), STORAGE_HOST, NOW)

    # Assert
    assert checked.job == "0199f1a0-2222-7000-8000-000000000002"
    assert checked.file_id == "1AbCdEfGhIjKlMnOpQrStUvWxYz012xyz"
    assert checked.resource_key is None
    assert checked.expected_size == 943_718_400
    assert checked.expected_md5 == "9e107d9d372bb6826bd81d3542a419d6"
    assert checked.expected_sha1 is None
    assert checked.expected_sha256 == "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
    assert checked.expected_head_revision_id == "0B1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
    assert [part.number for part in checked.parts] == list(range(1, 16))
    assert checked.parts[0].url == _fixture()["parts"][0]["url"]
    assert checked.receipt_url == _fixture()["receipt_url"]
    assert checked.writes_end_at == DEADLINE


def test_check_when_the_parts_come_in_another_order_should_return_them_by_number():
    # Arrange
    document = _fixture()
    document["parts"].reverse()

    # Act
    checked = check(document, STORAGE_HOST, NOW)

    # Assert
    assert [part.number for part in checked.parts] == list(range(1, 16))


@pytest.mark.parametrize("name", [RUNNING_EXAMPLE, ONE_PART, LARGEST])
def test_check_when_given_a_valid_fixture_of_the_contract_should_accept_it(name):
    # Arrange
    document = _fixture(name)

    # Act
    checked = check(document, STORAGE_HOST, NOW)

    # Assert
    assert len(checked.parts) == document["plan"]["parts_total"]
    assert checked.resource_key == document["resource_key"]


@pytest.mark.parametrize(
    ("written", "microsecond"),
    [pytest.param(".5Z", 500_000, id="one-digit"), pytest.param(".000123Z", 123, id="six-digits")],
)
def test_check_when_the_deadline_has_a_fraction_should_read_it_as_microseconds(written, microsecond):
    # Arrange
    document = _fixture()
    document["writes_end_at"] = "2026-10-17T09:16:53" + written

    # Act
    checked = check(document, STORAGE_HOST, NOW)

    # Assert
    assert checked.writes_end_at == DEADLINE.replace(microsecond=microsecond)


def test_check_when_the_url_host_is_written_in_capitals_should_accept_the_configured_store():
    # Arrange
    document = _with_part_url(1, _replace(STORAGE_HOST, STORAGE_HOST.upper()))

    # Act
    checked = check(document, STORAGE_HOST, NOW)

    # Assert
    assert checked.parts[0].number == 1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("https://storage.example.invalid", id="a-scheme"),
        pytest.param("storage.example.invalid:443", id="a-port"),
        pytest.param("storage.example.invalid/bucket", id="a-path"),
        pytest.param("user@storage.example.invalid", id="user-information"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(None, id="missing"),
        pytest.param("storage..example.invalid", id="an-empty-label"),
        pytest.param("-storage.example.invalid", id="a-label-starting-with-a-hyphen"),
    ],
)
def test_storage_host_from_when_the_value_is_not_a_bare_host_should_refuse_it_without_quoting_it(value):
    # Act
    with pytest.raises(ValueError) as refused:
        storage_host_from(value)

    # Assert
    assert "IMPORT_STORAGE_HOST" in str(refused.value)
    assert value is None or value.strip() == "" or value not in str(refused.value)


def test_storage_host_from_when_the_value_is_a_bare_host_should_return_it_lowercased_and_trimmed():
    # Act
    host = storage_host_from("  Storage.Example.Invalid ")

    # Assert
    assert host == STORAGE_HOST


def test_check_when_the_integer_members_are_written_as_whole_floats_should_accept_them_as_integers():
    # Arrange: the contract's integers are JSON numbers with an integer value, so 15.0 is one.
    document = _fixture()
    document["expected"]["size"] = float(document["expected"]["size"])
    document["plan"]["part_size"] = float(document["plan"]["part_size"])
    document["plan"]["parts_total"] = float(document["plan"]["parts_total"])
    for part in document["parts"]:
        part["n"] = float(part["n"])

    # Act
    checked = check(document, STORAGE_HOST, NOW)

    # Assert
    assert [part.number for part in checked.parts] == list(range(1, len(document["parts"]) + 1))
    assert all(isinstance(part.number, int) for part in checked.parts)
    assert isinstance(checked.expected_size, int)


def test_check_when_an_integer_member_is_a_fraction_should_refuse_the_input():
    # Arrange
    document = _fixture()
    document["parts"][0]["n"] = 1.5

    # Act / Assert
    with pytest.raises(InputRefused):
        check(document, STORAGE_HOST, NOW)
