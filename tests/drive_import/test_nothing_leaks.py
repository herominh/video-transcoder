"""Nothing of a key, a URL, the file id, Google's text or an ETag leaves the import job: not in a log line, not in the
output, not in what Python's logging prints when a log handler fails (it prints the error being handled then, and
requests' errors quote the whole URL of their request, a presigned one included)."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import requests

from core.drive_import import job as job_module
from core.drive_import.input_check import PART_SIZE
from core.drive_import.job import run_import
from tests.drive_import.fakes import (
    API_KEY,
    FILE_ID,
    MIB,
    RECEIPT_KEY,
    RECEIPT_URL,
    RESOURCE_KEY,
    SIGNATURE_CANARY,
    STORAGE_HOST,
    THREE_PARTS_SIZE,
    UPLOAD_ID,
    DriveFile,
    Fail,
    FakeClock,
    FakeResponse,
    FakeSession,
    Put,
    Serve,
    google_error,
    import_input,
    part_url,
    settings,
    signed,
)

ONE = DriveFile(MIB)
THREE = DriveFile(THREE_PARTS_SIZE)
# What no line, output or report may hold, whatever else it holds.
NEVER_SHOWN = (
    API_KEY,
    RECEIPT_KEY,
    FILE_ID,
    RESOURCE_KEY,
    SIGNATURE_CANARY,
    UPLOAD_ID,
    STORAGE_HOST,
    "https://",
    "googleapis",
    "X-Amz",
    "Google's own text",
)
LOG_FAILURE_CANARY = "CANARY-quoted-by-a-failing-log"


def _put(n: int, *answers: object) -> Callable[[FakeSession], None]:
    return lambda session: session.puts.setdefault(part_url(n), []).extend(answers)


def _media(*answers: object) -> Callable[[FakeSession], None]:
    return lambda session: session.media.extend(answers)


def _metadata(*answers: object) -> Callable[[FakeSession], None]:
    return lambda session: session.metadata.extend(answers)


def _nothing(session: FakeSession) -> None:
    return None


# Each ending with the file it is run on, its input, what the session answers, and the status or code meant.
SCENARIOS = [
    pytest.param(ONE, import_input(ONE, resource_key=RESOURCE_KEY), _nothing, "imported", id="imported"),
    pytest.param(THREE, import_input(THREE), _media(Serve(break_after=PART_SIZE + MIB), Serve(break_after=MIB)),
                 "imported", id="imported-after-two-resets"),
    pytest.param(ONE, import_input(ONE), _metadata(Fail(), FakeResponse(503)), "imported", id="metadata-tried-again"),
    pytest.param(ONE, import_input(ONE), _put(1, Fail(), Put(500)), "imported", id="put-tried-again"),
    pytest.param(ONE, import_input(ONE), _metadata(FakeResponse(404)), "file_removed", id="file-removed"),
    pytest.param(ONE, import_input(ONE), _media(google_error(403, "downloadQuotaExceeded")), "provider_quota",
                 id="provider-quota"),
    pytest.param(ONE, import_input(ONE), _metadata(google_error(400, "keyInvalid")), "not_configured",
                 id="key-invalid"),
    pytest.param(ONE, import_input(ONE), _media(Serve(extra=MIB)), "size_exceeded", id="size-exceeded"),
    pytest.param(THREE, import_input(THREE),
                 _media(Serve(break_after=PART_SIZE + MIB), Serve(break_after=MIB), Serve(break_after=MIB)),
                 "download_failed", id="three-resets"),
    pytest.param(ONE, import_input(ONE), _put(1, Put(403)), "upload_failed", id="put-refused"),
    pytest.param(ONE, import_input(ONE), _put(1, Fail(), Fail(), Fail(), Fail()), "upload_failed",
                 id="put-without-answer"),
    pytest.param(ONE, import_input(ONE), lambda session: session.puts.setdefault(RECEIPT_URL, []).append(Put(403)),
                 "upload_failed", id="receipt-refused"),
    pytest.param(ONE, dict(import_input(ONE), api_key=API_KEY, file_id=FILE_ID + "/" + SIGNATURE_CANARY), _nothing,
                 "input_refused", id="input-refused-by-its-schema"),
    pytest.param(ONE, signed(dict(import_input(ONE), receipt_url=part_url(1))), _nothing, "input_refused",
                 id="input-refused-beyond-its-schema"),
    pytest.param(ONE, dict(import_input(ONE), receipt_url=part_url(1)), _nothing, "input_refused",
                 id="input-refused-by-its-mac"),
]


def _never_shown(session: FakeSession) -> tuple[str, ...]:
    return (*NEVER_SHOWN, *session.etags_given.values())


def _shown(values: tuple[str, ...], *texts: str) -> list[str]:
    return [value for value in values for text in texts if value in text]


def _run(file: DriveFile, document: object, script: Callable[[FakeSession], None]) -> tuple[dict, FakeSession]:
    clock = FakeClock()
    session = FakeSession(file, clock)
    script(session)
    return run_import(document, settings(), session, clock), session


@pytest.mark.parametrize(("file", "document", "script", "ending"), SCENARIOS)
def test_run_import_when_it_ends_in_any_way_should_log_and_return_nothing_it_must_keep(
    caplog, file, document, script, ending
):
    # Arrange
    caplog.set_level(logging.DEBUG)

    # Act
    output, session = _run(file, document, script)

    # Assert: the ending is the one meant; no line, argument or output shows what it must not, and no line carries
    # an error (whose traceback would end with its text)
    lines = [record.getMessage() for record in caplog.records]
    arguments = [repr(record.args) for record in caplog.records]
    assert ending in (output["status"], output["code"])
    assert lines != []
    assert _shown(_never_shown(session), json.dumps(output), *lines, *arguments) == []
    assert [record for record in caplog.records if record.exc_info or record.exc_text] == []


class _FailingStream(io.TextIOBase):
    """A standard error that fails every write."""

    def write(self, text: str) -> int:
        raise OSError(f"no room left for {LOG_FAILURE_CANARY}")


@contextmanager
def _job_log_through_a_failing_handler() -> Iterator[None]:
    handler = logging.StreamHandler(_FailingStream())
    job_module.logger.addHandler(handler)
    try:
        yield
    finally:
        job_module.logger.removeHandler(handler)


@pytest.mark.parametrize(("file", "document", "script", "ending"), SCENARIOS)
def test_run_import_when_a_log_handler_fails_on_its_lines_should_keep_all_it_must_out_of_standard_error(
    monkeypatch, caplog, capsys, file, document, script, ending
):
    # Arrange: such a handler does not raise; Python's logging prints a report to standard error with every error
    # its failure is chained to, which would be the error being handled when the line was written. Every line of
    # the job reaches the handler
    monkeypatch.setattr(logging, "raiseExceptions", True)
    caplog.set_level(logging.DEBUG, logger=job_module.logger.name)

    # Act
    with _job_log_through_a_failing_handler():
        output, session = _run(file, document, script)
    written = capsys.readouterr().err

    # Assert: the reports were printed, and hold nothing they must not; the job still ended as it would have
    assert LOG_FAILURE_CANARY in written
    assert _shown(_never_shown(session), written) == []
    assert ending in (output["status"], output["code"])


def test_run_import_when_a_transfer_error_quotes_its_url_should_never_log_it(caplog):
    # Arrange: requests' own error text holds the presigned URL and more
    caplog.set_level(logging.DEBUG)
    error = requests.exceptions.ConnectionError(f"Max retries exceeded with url: {part_url(1)} key={API_KEY}")

    # Act
    output, session = _run(ONE, import_input(ONE), _put(1, Fail(error), Fail(error), Fail(error), Fail(error)))

    # Assert
    assert output["code"] == "upload_failed"
    assert _shown(_never_shown(session), *(record.getMessage() for record in caplog.records)) == []
