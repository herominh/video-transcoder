"""The import endpoint's configuration: read whole, or every missing or unusable variable named (never its value),
and no repr that shows a key."""

from __future__ import annotations

import pytest

from core.drive_import.settings import ImportSettings
from tests.drive_import.fakes import API_KEY, RECEIPT_KEY, STORAGE_HOST, settings

GOOGLE = "GOOGLE_DRIVE_API_KEY"
RECEIPT = "IMPORT_RECEIPT_KEY"
HOST = "IMPORT_STORAGE_HOST"


def test_from_env_when_every_variable_is_usable_should_hold_them_and_name_nothing_missing():
    # Act
    read = settings()

    # Assert
    assert read.google_api_key == API_KEY
    assert read.receipt_key == RECEIPT_KEY
    assert read.storage_host == STORAGE_HOST
    assert read.missing == ()
    assert read.complete


@pytest.mark.parametrize(
    ("changes", "missing"),
    [
        pytest.param({GOOGLE: None}, (GOOGLE,), id="no-google-key"),
        pytest.param({GOOGLE: "a key with spaces"}, (GOOGLE,), id="a-google-key-with-a-space"),
        pytest.param({GOOGLE: "key\n"}, (GOOGLE,), id="a-google-key-with-a-line-break"),
        pytest.param({GOOGLE: ""}, (GOOGLE,), id="an-empty-google-key"),
        pytest.param({RECEIPT: None}, (RECEIPT,), id="no-receipt-key"),
        pytest.param({RECEIPT: "x" * 31}, (RECEIPT,), id="a-receipt-key-of-31-characters"),
        pytest.param({RECEIPT: "x" * 257}, (RECEIPT,), id="a-receipt-key-of-257-characters"),
        pytest.param({RECEIPT: "x" * 31 + " "}, (RECEIPT,), id="a-receipt-key-with-a-space"),
        pytest.param({RECEIPT: "x" * 32 + "\n"}, (RECEIPT,), id="a-receipt-key-with-a-trailing-line-break"),
        pytest.param({RECEIPT: "x" * 31 + "é"}, (RECEIPT,), id="a-receipt-key-outside-ascii"),
        pytest.param({HOST: None}, (HOST,), id="no-storage-host"),
        pytest.param({HOST: "https://storage.example.invalid"}, (HOST,), id="a-storage-host-with-a-scheme"),
        pytest.param({GOOGLE: None, RECEIPT: None, HOST: None}, (GOOGLE, RECEIPT, HOST), id="nothing"),
    ],
)
def test_from_env_when_a_variable_is_missing_or_unusable_should_name_it_and_never_raise(changes, missing):
    # Act
    read = settings(**changes)

    # Assert
    assert read.missing == missing
    assert not read.complete


@pytest.mark.parametrize("length", [32, 256])
def test_from_env_when_the_receipt_key_is_of_the_hubs_form_should_take_it_as_it_is(length):
    # Arrange: printable ASCII without a space, 32 to 256 characters (the Hub's rule for its setting)
    key = ("!~Kk0" * 60)[:length]

    # Act
    read = settings(**{RECEIPT: key})

    # Assert
    assert read.receipt_key == key and read.missing == ()


def test_repr_when_the_settings_hold_both_keys_should_show_neither():
    # Act
    shown = repr(settings()) + str(settings())

    # Assert
    assert API_KEY not in shown and RECEIPT_KEY not in shown
    assert STORAGE_HOST in shown


def test_from_env_when_given_a_mapping_should_read_only_that_mapping(monkeypatch):
    # Arrange: a whole configuration in the process's environment, none in the mapping
    monkeypatch.setenv(GOOGLE, API_KEY)
    monkeypatch.setenv(RECEIPT, RECEIPT_KEY)
    monkeypatch.setenv(HOST, STORAGE_HOST)

    # Act
    read = ImportSettings.from_env({})

    # Assert
    assert read.missing == (GOOGLE, RECEIPT, HOST)
