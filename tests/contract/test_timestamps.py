"""S1's calendar rule and the millisecond arithmetic of freshness (README.md section 8.1); mirrors the Hub's
ContractTimestampTest."""

import pytest

from core.protocol.semantic import parse_timestamp_ms

REAL_INSTANTS = {
    "the epoch": ("1970-01-01T00:00:00.000Z", 0),
    "the fixtures' now": ("2030-01-01T00:00:00.000Z", 1_893_456_000_000),
    "five minutes before it": ("2029-12-31T23:55:00.000Z", 1_893_455_700_000),
    "a leap day of a leap century": ("2000-02-29T12:34:56.789Z", 951_827_696_789),
    "the millisecond before the epoch": ("1969-12-31T23:59:59.999Z", -1),
    "the first instant of year 1": ("0001-01-01T00:00:00.000Z", -62_135_596_800_000),
    "the last instant of year 9999": ("9999-12-31T23:59:59.999Z", 253_402_300_799_999),
}

NOT_REAL_INSTANTS = {
    "30 February": "2030-02-30T00:00:00.000Z",
    "29 February of a common year": "2031-02-29T00:00:00.000Z",
    "29 February of a century that is no leap year": "2100-02-29T00:00:00.000Z",
    "31 April": "2030-04-31T00:00:00.000Z",
    "month 13": "2030-13-01T00:00:00.000Z",
    "month 00": "2030-00-01T00:00:00.000Z",
    "day 00": "2030-01-00T00:00:00.000Z",
    "year 0000": "0000-01-01T00:00:00.000Z",
    "hour 24": "2030-01-01T24:00:00.000Z",
    "minute 60": "2030-01-01T00:60:00.000Z",
    "a leap second": "2030-06-30T23:59:60.000Z",
    "no milliseconds": "2030-01-01T00:00:00Z",
    "a lowercase zone letter": "2030-01-01T00:00:00.000z",
    "an offset instead of Z": "2030-01-01T00:00:00.000+00:00",
    "a trailing newline": "2030-01-01T00:00:00.000Z\n",
    "non-ASCII digits": "２０３０-01-01T00:00:00.000Z",
    "empty": "",
}


@pytest.mark.parametrize(("timestamp", "expected"), list(REAL_INSTANTS.values()), ids=list(REAL_INSTANTS))
def test_epoch_milliseconds_when_the_timestamp_is_a_real_instant_should_be_exact(timestamp, expected):
    # Act
    milliseconds = parse_timestamp_ms(timestamp)

    # Assert
    assert milliseconds == expected


@pytest.mark.parametrize("timestamp", list(NOT_REAL_INSTANTS.values()), ids=list(NOT_REAL_INSTANTS))
def test_epoch_milliseconds_when_the_timestamp_names_no_real_instant_should_be_none(timestamp):
    # Act
    milliseconds = parse_timestamp_ms(timestamp)

    # Assert
    assert milliseconds is None
