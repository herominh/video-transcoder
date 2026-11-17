"""The suite runs on two Pythons: the encode image's (3.10, Dockerfile.runpod) and the acquire image's (3.12,
Dockerfile.acquire), which the developer machine's matches.

The Drive acquire's code (core/drive_import/) uses 3.11 features and only ever runs on 3.12, so on a Python below
3.11 the test modules that load it are skipped by name, each reported as one skipped module, its path and the reason
below, never as a collection error. The list is explicit on purpose: a pattern could swallow a test of the encode's
path, which must be collected on 3.10. Everything else, the capability path's and the shared session's tests
(tests/net/) included, runs on both.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import pytest

ACQUIRE_ONLY_REASON = (
    "the Drive acquire runs on Python 3.12 only (Dockerfile.acquire); its code uses 3.11 features"
)
ACQUIRE_ONLY_PYTHON = (3, 11)
TESTS_ROOT = Path(__file__).resolve().parent
# The modules that load core/drive_import/ or the acquire's entry point, relative to tests/. The shared session's own
# tests are tests/net/'s and run on both Pythons; drive_import/test_transport.py holds only the acquire's store PUT.
ACQUIRE_ONLY_MODULES = frozenset({
    "acquire/test_entry.py",
    "acquire/test_failures.py",
    "acquire/test_grants.py",
    "acquire/test_handler.py",
    "acquire/test_messages.py",
    "acquire/test_settings.py",
    "acquire/test_transfer.py",
    "acquire/test_transfer_cases.py",
    "acquire/test_transport.py",
    "drive_import/test_transport.py",
})


def _relative_path(module_path: Path) -> str:
    return module_path.resolve().relative_to(TESTS_ROOT).as_posix()


class _AcquireOnlyModule(pytest.Module):
    """A module that is reported skipped, named by its path, and never imported."""

    def collect(self) -> NoReturn:
        pytest.skip(f"tests/{_relative_path(self.path)}: {ACQUIRE_ONLY_REASON}")


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Module | None:
    if sys.version_info >= ACQUIRE_ONLY_PYTHON:
        return None
    if _relative_path(module_path) not in ACQUIRE_ONLY_MODULES:
        return None
    return _AcquireOnlyModule.from_parent(parent, path=module_path)
