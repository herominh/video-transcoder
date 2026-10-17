"""The import endpoint's configuration: three variables, read once per process, none of them in a job's input.

    GOOGLE_DRIVE_API_KEY   the worker's Google key (a RunPod Secret of the import endpoint): printable ASCII
                           with no space, sent in the `X-Goog-Api-Key` header only
    IMPORT_RECEIPT_KEY     the import key, which signs the job's input and its receipt (a RunPod Secret, the same
                           text as the Hub's secret setting `drive_import_receipt_key`): 32 to 256 printable ASCII
                           characters with no space, the Hub's own rule for that setting, used as its bytes
    IMPORT_STORAGE_HOST    the store's host every part and receipt URL must name (input_check.storage_host_from)

ImportSettings.from_env() never raises: a variable that is missing or not usable is recorded by its name in
`missing`, and every job of the process then answers `failed/not_configured` before any request. No repr shows a
key, and nothing here names a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from core.drive_import.input_check import storage_host_from

GOOGLE_API_KEY_VARIABLE = "GOOGLE_DRIVE_API_KEY"
RECEIPT_KEY_VARIABLE = "IMPORT_RECEIPT_KEY"
STORAGE_HOST_VARIABLE = "IMPORT_STORAGE_HOST"
# An API key as a header carries it: printable ASCII, no space (the Hub's own rule for its key).
USABLE_API_KEY = re.compile(r"[\x21-\x7e]+")
# The receipt key as the Hub's setting holds it: 32 to 256 printable ASCII characters, no space (the design, G-23).
USABLE_RECEIPT_KEY = re.compile(r"[\x21-\x7e]{32,256}")


@dataclass(frozen=True)
class ImportSettings:
    """The configuration a job runs with. `missing` names, in a fixed order, every variable that is missing or not
    usable; a job runs only when it is empty. The keys are never part of a repr."""

    google_api_key: str | None = field(repr=False)
    receipt_key: str | None = field(repr=False)
    storage_host: str | None
    missing: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ImportSettings:
        """The settings of `env`; a variable that is missing or not usable becomes None and is named in `missing`."""
        missing: list[str] = []

        google_api_key = env.get(GOOGLE_API_KEY_VARIABLE)
        if google_api_key is None or not USABLE_API_KEY.fullmatch(google_api_key):
            google_api_key = None
            missing.append(GOOGLE_API_KEY_VARIABLE)

        receipt_key = env.get(RECEIPT_KEY_VARIABLE)
        if receipt_key is None or not USABLE_RECEIPT_KEY.fullmatch(receipt_key):
            receipt_key = None
            missing.append(RECEIPT_KEY_VARIABLE)

        storage_host = _storage_host_or_none(env.get(STORAGE_HOST_VARIABLE))
        if storage_host is None:
            missing.append(STORAGE_HOST_VARIABLE)

        return cls(google_api_key, receipt_key, storage_host, tuple(missing))

    @property
    def complete(self) -> bool:
        return not self.missing


def _storage_host_or_none(value: str | None) -> str | None:
    try:
        return storage_host_from(value)
    except ValueError:
        return None
