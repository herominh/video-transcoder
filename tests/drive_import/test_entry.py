"""The import endpoint's RunPod entry point (wrappers/runpod_import_handler.py): what the provider gets for one job,
and what it never gets.

job_handler() is driven with the fake session of tests/drive_import/fakes.py and the real run_import(), or with a
stand-in run_import() that fails on a bug. A failure the job knows is the job's output, never an error; a bug
leaves the handler as the wrapper's own WorkerFault, without a cause or a context, naming only the bug's class, and
is logged once by its class and stack, never its text. build_handler() refuses a tree it cannot trust before it
returns a handler, and builds one whatever the configuration (an incomplete one answers not_configured).
configure_logging() cuts the HTTP libraries out of the process's log; importing the module configures nothing and
loads no SDK; main() is driven with a starter of the test's own, and the file is run in a process of its own with a
stand-in for the SDK first on its module path.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from core.drive_import import contract
from core.drive_import.contract import ImportContractInvalid
from tests.drive_import.fakes import (
    API_KEY,
    MIB,
    RECEIPT_KEY,
    STORAGE_HOST,
    DriveFile,
    FakeClock,
    FakeSession,
    import_input,
    settings,
)
from wrappers import runpod_import_handler
from wrappers.runpod_import_handler import WorkerFault, build_handler, job_handler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENTRY_LOGGER = "wrappers.runpod_import_handler"
LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
HTTP_LIBRARY_LOGGERS = ("urllib3", "requests", "httpx", "httpcore")
CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred",
    "The above exception was the direct cause of the following exception",
)
# The text of a bug, and of the error it is chained to: each quotes what the error choked on.
BUG_CANARY = "CANARY-quoted-by-a-bug"
CHAINED_CANARY = "CANARY-quoted-by-a-chained-error"
BUG_TEXT = f"could not reach https://{STORAGE_HOST}/?X-Amz-Signature={BUG_CANARY}"
CHAINED_TEXT = f"the connection to {CHAINED_CANARY} was reset"
LOG_FAILURE_CANARY = "CANARY-quoted-by-a-failing-log"
# What no error, traceback, log line or output may show.
NEVER_SHOWN = (BUG_CANARY, CHAINED_CANARY, API_KEY, RECEIPT_KEY, LOG_FAILURE_CANARY)
CONFIG_CANARY = "CANARY typed into the configuration"
ONE = DriveFile(MIB)

SDK_MODULE = "runpod"
HANDLER_MEMBER = "handler"
WHILE_THE_HANDLER_IS_BUILT = "the handler is being built"
WHEN_THE_LOOP_IS_STARTED = "the serverless loop was started"
STORAGE_URL_CANARY = f"https://{STORAGE_HOST}/bucket-canary/original-videos/uuid-canary/original.mp4"
# The modules importing the entry point never loads: the SDK, the transcode's and the v1 worker's.
NEVER_IMPORTED = (
    "runpod",
    "core.protocol",
    "core.job",
    "core.api",
    "core.config",
    "core.transcoder",
    "core.storage",
    "core.callback",
    "core.signing",
)
PROGRAM_TIMEOUT_S = 120.0
SDK_RUN_TIMEOUT_S = 60.0
STAND_IN_SDK = f"""\
import json
import logging
import types

logging.basicConfig(level=logging.INFO, format="the SDK's own format: %(message)s", handlers=[logging.StreamHandler()])
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _start(config):
    logging.getLogger("urllib3.connection").warning("Failed to parse headers (url=%s)", {STORAGE_URL_CANARY!r})
    logging.getLogger({ENTRY_LOGGER!r}).info({WHEN_THE_LOOP_IS_STARTED!r})
    print(json.dumps({{"members": sorted(config), "handler_is_callable": callable(config.get({HANDLER_MEMBER!r}))}}))


serverless = types.SimpleNamespace(start=_start)
"""


def _whole_env() -> dict[str, str]:
    return {"GOOGLE_DRIVE_API_KEY": API_KEY, "IMPORT_RECEIPT_KEY": RECEIPT_KEY, "IMPORT_STORAGE_HOST": STORAGE_HOST}


def _shown(values: tuple[str, ...], *texts: str) -> list[str]:
    return [value for value in values for text in texts if value in text]


class _Sessions:
    """Makes the fake session of each job, and keeps them."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.made: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession(ONE, self.clock)
        self.made.append(session)
        return session


def _handler(sessions: _Sessions) -> Callable[[object], dict[str, Any]]:
    return job_handler(settings(), sessions, sessions.clock)


# --- one job ---------------------------------------------------------------------------------------------------------


def test_handler_when_the_job_imports_should_return_the_output_and_close_the_jobs_session():
    # Arrange
    sessions = _Sessions()

    # Act
    output = _handler(sessions)({"id": "job-1", "input": import_input(ONE)})

    # Assert
    (session,) = sessions.made
    assert output == {"contract": "import/v1", "status": "imported", "code": None}
    assert session.closed and session.receipt_puts != []


def test_handler_when_run_import_returns_should_return_that_very_dict(monkeypatch):
    # Arrange
    returned = {"contract": "import/v1", "status": "failed", "code": "deadline"}
    monkeypatch.setattr(runpod_import_handler, "run_import", lambda *arguments: returned)

    # Act
    output = _handler(_Sessions())({"id": "job-1", "input": {}})

    # Assert
    assert output is returned


@pytest.mark.parametrize(
    "job",
    [
        pytest.param({"id": "job-1", "input": {"contract": "import/v1"}}, id="an-input-outside-the-contract"),
        pytest.param({"id": "job-1"}, id="no-input"),
        pytest.param(["not", "a", "job"], id="no-mapping"),
    ],
)
def test_handler_when_the_job_holds_no_input_it_may_act_on_should_return_a_failed_output_not_raise(job):
    # Arrange
    sessions = _Sessions()

    # Act
    output = _handler(sessions)(job)

    # Assert: a known failure is an output, and no request was made
    assert output == {"contract": "import/v1", "status": "failed", "code": "input_refused"}
    assert [session.requests for session in sessions.made] == [[]]


def _fail(*arguments: object) -> dict[str, Any]:
    raise RuntimeError(BUG_TEXT)


def _fail_because_of_another_error(*arguments: object) -> dict[str, Any]:
    try:
        raise ConnectionError(CHAINED_TEXT)
    except ConnectionError as cause:
        raise LookupError(BUG_TEXT) from cause


def _fail_while_handling_another_error(*arguments: object) -> dict[str, Any]:
    try:
        raise ConnectionError(CHAINED_TEXT)
    except ConnectionError:
        raise ValueError(BUG_TEXT)


BUGS = [
    pytest.param(_fail, "the import failed on an unexpected RuntimeError", ["RuntimeError"], id="an-error"),
    pytest.param(_fail_because_of_another_error, "the import failed on an unexpected LookupError",
                 ["LookupError", "ConnectionError"], id="an-error-with-a-cause"),
    pytest.param(_fail_while_handling_another_error, "the import failed on an unexpected ValueError",
                 ["ValueError", "ConnectionError"], id="an-error-raised-while-another-was-handled"),
]


@pytest.mark.parametrize(("bug", "fault_text", "classes"), BUGS)
def test_handler_when_the_job_fails_on_a_bug_should_raise_its_own_error_naming_the_class_only(
    monkeypatch, bug, fault_text, classes
):
    # Arrange
    monkeypatch.setattr(runpod_import_handler, "run_import", bug)
    sessions = _Sessions()

    # Act
    with pytest.raises(WorkerFault) as raised:
        _handler(sessions)({"id": "job-1", "input": import_input(ONE)})

    # Assert: what the provider records of it holds no error's own text and no other error; the session was closed
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == fault_text
    assert _shown(CHAIN_MARKERS, recorded) == []
    assert _shown(NEVER_SHOWN, recorded) == []
    assert [session.closed for session in sessions.made] == [True]


@pytest.mark.parametrize(("bug", "fault_text", "classes"), BUGS)
def test_handler_when_the_job_fails_on_a_bug_should_log_once_its_class_and_stack_without_its_text(
    monkeypatch, caplog, bug, fault_text, classes
):
    # Arrange
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(runpod_import_handler, "run_import", bug)

    # Act
    with pytest.raises(WorkerFault):
        _handler(_Sessions())({"id": "job-1", "input": import_input(ONE)})

    # Assert
    (logged,) = [record for record in caplog.records if record.name == ENTRY_LOGGER and record.levelno == logging.ERROR]
    line = LOG_FORMATTER.format(logged)
    assert [name for name in classes if name not in line] == []
    assert 'File "' in line and ", line " in line and f"in {bug.__name__}" in line
    assert [record for record in caplog.records if record.exc_info] == []
    assert _shown(NEVER_SHOWN, *(record.getMessage() for record in caplog.records)) == []


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        pytest.param(runpod_import_handler, "_stack_report", id="the-line-cannot-be-built"),
        pytest.param(runpod_import_handler.logger, "error", id="the-line-cannot-be-written"),
    ],
)
def test_handler_when_the_log_line_of_a_bug_fails_too_should_still_raise_its_own_error_with_nothing_of_either(
    monkeypatch, owner, name
):
    # Arrange
    def failing_log(*arguments: object, **named: object) -> None:
        raise MemoryError(f"no memory left for {LOG_FAILURE_CANARY}")

    monkeypatch.setattr(owner, name, failing_log)
    monkeypatch.setattr(runpod_import_handler, "run_import", _fail)

    # Act
    with pytest.raises(WorkerFault) as raised:
        _handler(_Sessions())({"id": "job-1", "input": import_input(ONE)})

    # Assert
    recorded = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert str(raised.value) == "the import failed on an unexpected RuntimeError"
    assert _shown(NEVER_SHOWN, recorded) == []


class _FailingStream(io.TextIOBase):
    def write(self, text: str) -> int:
        raise OSError(f"no room left for {LOG_FAILURE_CANARY}")


def test_handler_when_a_log_handler_fails_on_the_line_of_a_bug_should_keep_the_bugs_text_out_of_standard_error(
    monkeypatch, capsys
):
    # Arrange: such a handler does not raise; Python's logging prints a report with every error its failure is
    # chained to
    monkeypatch.setattr(logging, "raiseExceptions", True)
    monkeypatch.setattr(runpod_import_handler, "run_import", _fail_because_of_another_error)
    entry_logger = logging.getLogger(ENTRY_LOGGER)
    failing = logging.StreamHandler(_FailingStream())
    entry_logger.addHandler(failing)

    # Act
    try:
        with pytest.raises(WorkerFault) as raised:
            _handler(_Sessions())({"id": "job-1", "input": import_input(ONE)})
    finally:
        entry_logger.removeHandler(failing)
    written = capsys.readouterr().err

    # Assert
    assert LOG_FAILURE_CANARY in written
    assert _shown((BUG_CANARY, CHAINED_CANARY), written) == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), SystemExit(3)], ids=["an-interrupt", "the-interpreters-exit"])
def test_handler_when_the_process_is_being_stopped_should_let_that_pass_unchanged(monkeypatch, stop):
    # Arrange
    def stopped(*arguments: object) -> dict[str, Any]:
        raise stop

    monkeypatch.setattr(runpod_import_handler, "run_import", stopped)

    # Act
    with pytest.raises(type(stop)) as raised:
        _handler(_Sessions())({"id": "job-1", "input": {}})

    # Assert
    assert raised.value is stop


class _SessionsThatFailToClose(_Sessions):
    """Sessions whose close raises `error`."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    def __call__(self) -> FakeSession:
        session = super().__call__()
        error = self._error

        def close() -> None:
            raise error

        session.close = close  # type: ignore[method-assign]
        return session


def test_handler_when_the_jobs_session_fails_to_close_with_an_os_error_should_still_return_the_output():
    # Arrange: the close's error quotes a host, as a transfer's does
    sessions = _SessionsThatFailToClose(ConnectionResetError(f"reset by {STORAGE_HOST}"))

    # Act
    output = job_handler(settings(), sessions, sessions.clock)({"id": "job-1", "input": import_input(ONE)})

    # Assert
    assert output == {"contract": "import/v1", "status": "imported", "code": None}


def test_handler_when_the_jobs_session_fails_to_close_on_a_bug_should_raise_its_own_error():
    # Arrange
    sessions = _SessionsThatFailToClose(RuntimeError(BUG_TEXT))

    # Act
    with pytest.raises(WorkerFault) as raised:
        job_handler(settings(), sessions, sessions.clock)({"id": "job-1", "input": import_input(ONE)})

    # Assert
    assert str(raised.value) == "the import failed on an unexpected RuntimeError"
    assert raised.value.__cause__ is None and raised.value.__context__ is None


# --- the handler built from the process's start ---------------------------------------------------------------------


def test_build_handler_when_the_tree_and_the_configuration_are_whole_should_return_a_handler_that_answers_jobs():
    # Arrange
    handler = build_handler(_whole_env())

    # Act: an input the schema refuses, so that no request is made
    output = handler({"id": "job-1", "input": {"contract": "import/v1"}})

    # Assert
    assert output == {"contract": "import/v1", "status": "failed", "code": "input_refused"}


def test_build_handler_when_the_configuration_is_not_whole_should_log_the_variables_by_name_and_answer_not_configured(
    caplog,
):
    # Arrange: a key that is not usable, typed with a value no line may show
    caplog.set_level(logging.DEBUG)
    env = dict(_whole_env(), GOOGLE_DRIVE_API_KEY=CONFIG_CANARY)
    del env["IMPORT_RECEIPT_KEY"]

    # Act
    handler = build_handler(env)
    output = handler({"id": "job-1", "input": import_input(ONE)})

    # Assert: the start says which variables, never a value; the job answers before any request
    lines = [record.getMessage() for record in caplog.records if record.name == ENTRY_LOGGER]
    assert output == {"contract": "import/v1", "status": "failed", "code": "not_configured"}
    assert [line for line in lines if "GOOGLE_DRIVE_API_KEY" in line and "IMPORT_RECEIPT_KEY" in line] != []
    assert _shown((CONFIG_CANARY, API_KEY, STORAGE_HOST), *lines) == []


def test_build_handler_when_the_contract_tree_was_tampered_with_should_raise_import_contract_invalid(
    tmp_path, monkeypatch
):
    # Arrange
    tampered = tmp_path / "v1"
    shutil.copytree(contract.IMPORT_CONTRACT_ROOT, tampered)
    schema = tampered / "receipt.schema.json"
    schema.write_bytes(schema.read_bytes() + b" ")
    monkeypatch.setattr(contract, "IMPORT_CONTRACT_ROOT", tampered)

    # Act / Assert
    with pytest.raises(ImportContractInvalid, match="receipt.schema.json"):
        build_handler(_whole_env())


# --- the process's log ---------------------------------------------------------------------------------------------


def _logging_state() -> list[tuple[list[logging.Handler], int, bool]]:
    loggers = (logging.getLogger(), *(logging.getLogger(name) for name in HTTP_LIBRARY_LOGGERS))
    return [(list(logger.handlers), logger.level, logger.propagate) for logger in loggers]


@contextlib.contextmanager
def _logging_nobody_configured() -> Iterator[None]:
    """Inside, the root logger and the HTTP libraries' loggers are as in a process that has just started; on
    leaving, as they were before (a context of the test's body: pytest swaps the root's handlers per phase)."""
    loggers = (logging.getLogger(), *(logging.getLogger(name) for name in HTTP_LIBRARY_LOGGERS))
    state_before = _logging_state()
    for logger in loggers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
    loggers[0].setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, (handlers, level, propagate) in zip(loggers, state_before):
            for configured in list(logger.handlers):
                logger.removeHandler(configured)
                configured.close()
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


@pytest.mark.parametrize("library", HTTP_LIBRARY_LOGGERS)
def test_configure_logging_when_an_http_library_logs_what_it_talks_to_should_keep_it_out_of_the_log(library, capsys):
    # Act
    with _logging_nobody_configured():
        runpod_import_handler.configure_logging()
        logging.getLogger(library).setLevel(logging.WARNING)
        logging.getLogger(f"{library}.connection").warning("Failed to parse headers (url=%s)", STORAGE_URL_CANARY)
        logging.getLogger("core.drive_import.job").info("a line of the job's own")
        written = capsys.readouterr().err

    # Assert
    assert " INFO core.drive_import.job: a line of the job's own\n" in written
    assert STORAGE_URL_CANARY not in written


def test_configure_logging_when_called_should_leave_one_handler_and_the_job_at_info():
    # Arrange
    state_before = _logging_state()

    # Act
    with _logging_nobody_configured():
        runpod_import_handler.configure_logging()
        root_handlers = len(logging.getLogger().handlers)
        job_level = logging.getLogger("core.drive_import.job").getEffectiveLevel()

    # Assert
    assert root_handlers == 1 and job_level == logging.INFO
    assert _logging_state() == state_before


def test_entry_point_when_imported_by_a_fresh_interpreter_should_load_no_sdk_nor_transcode_and_configure_no_log():
    # Arrange
    code = (
        "import logging\n"
        "import sys\n"
        "import wrappers.runpod_import_handler\n"
        f"print(sorted(m for m in sys.modules if any(m == n or m.startswith(n + '.') for n in {NEVER_IMPORTED!r})))\n"
        "print(logging.getLogger().handlers)\n"
        "print(logging.getLogger('urllib3').propagate)\n"
    )
    env = {name: value for name, value in os.environ.items() if name not in _whole_env()}
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Act
    imported = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=PROGRAM_TIMEOUT_S, check=False,
    )

    # Assert
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.splitlines() == ["[]", "[]", "True"]


def test_entry_point_when_imported_with_a_tls_key_log_named_should_drop_it_before_an_https_context_is_made(
    tmp_path,
):
    # Arrange: a process of its own whose environment names a file for the TLS session keys
    key_log = tmp_path / "tls-keys.log"
    code = (
        "import os\n"
        "import wrappers.runpod_import_handler\n"
        "from urllib3.util.ssl_ import create_urllib3_context\n"
        "print('SSLKEYLOGFILE' in os.environ)\n"
        "print(create_urllib3_context().keylog_filename)\n"
    )
    env = {name: value for name, value in os.environ.items() if name not in _whole_env()}
    env["SSLKEYLOGFILE"] = str(key_log)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Act
    imported = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=PROGRAM_TIMEOUT_S, check=False,
    )

    # Assert: the variable is gone, urllib3's HTTPS context factory logs no key, and no file was made
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.splitlines() == ["False", "None"]
    assert not key_log.exists()


# --- the entry point run as the program ------------------------------------------------------------------------------


class _Starter:
    def __init__(self, when_started: Callable[[], None] = lambda: None) -> None:
        self.configurations: list[dict[str, Any]] = []
        self._when_started = when_started

    def __call__(self, configuration: dict[str, Any]) -> None:
        self.configurations.append(configuration)
        self._when_started()


class _EnvironmentReadOnce(Mapping[str, str]):
    """An environment that does `when_first_read` the first time anything is read from it."""

    def __init__(self, variables: Mapping[str, str], when_first_read: Callable[[], None]) -> None:
        self._variables = dict(variables)
        self._when_first_read = when_first_read
        self._was_read = False

    def __getitem__(self, name: str) -> str:
        self._note_the_read()
        return self._variables[name]

    def __iter__(self) -> Iterator[str]:
        self._note_the_read()
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def _note_the_read(self) -> None:
        if not self._was_read:
            self._was_read = True
            self._when_first_read()


def _write_a_line_of_each(words: str) -> None:
    logging.getLogger("urllib3.connection").warning("Failed to parse headers (url=%s)", STORAGE_URL_CANARY)
    logging.getLogger(ENTRY_LOGGER).info(words)


def test_main_when_given_a_starter_should_set_up_the_log_before_the_handler_is_built_and_start_the_loop_once(
    monkeypatch, capsys
):
    # Arrange: the SDK cannot be imported here, installed or not
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)
    env = _EnvironmentReadOnce(_whole_env(), lambda: _write_a_line_of_each(WHILE_THE_HANDLER_IS_BUILT))
    starter = _Starter(lambda: _write_a_line_of_each(WHEN_THE_LOOP_IS_STARTED))

    # Act
    with _logging_nobody_configured():
        runpod_import_handler.main(starter, env)
        written = capsys.readouterr().err

    # Assert: each moment met a log that was set up, and the loop was started once with the handler
    (configuration,) = starter.configurations
    assert f" INFO {ENTRY_LOGGER}: {WHILE_THE_HANDLER_IS_BUILT}\n" in written
    assert f" INFO {ENTRY_LOGGER}: {WHEN_THE_LOOP_IS_STARTED}\n" in written
    assert STORAGE_URL_CANARY not in written
    assert sorted(configuration) == [HANDLER_MEMBER] and callable(configuration[HANDLER_MEMBER])


def test_main_when_the_contract_tree_is_not_the_pinned_one_should_raise_and_never_call_the_starter(
    tmp_path, monkeypatch
):
    # Arrange
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)
    monkeypatch.setattr(contract, "IMPORT_CONTRACT_ROOT", tmp_path / "absent")
    starter = _Starter()

    # Act
    with _logging_nobody_configured(), pytest.raises(ImportContractInvalid):
        runpod_import_handler.main(starter, _whole_env())

    # Assert
    assert starter.configurations == []


def test_entry_point_when_run_as_the_program_should_set_up_the_log_then_import_the_sdk_and_start_it_with_a_handler(
    tmp_path,
):
    # Arrange: a process of its own, a whole configuration, and a stand-in for the SDK first on its module path
    sdk_path = tmp_path / "sdk"
    (sdk_path / SDK_MODULE).mkdir(parents=True)
    (sdk_path / SDK_MODULE / "__init__.py").write_text(STAND_IN_SDK, encoding="utf-8")
    env = {name: value for name, value in os.environ.items() if name not in _whole_env()}
    env.update(_whole_env())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(sdk_path), os.environ.get("PYTHONPATH"))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Act
    started = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "wrappers" / "runpod_import_handler.py")],
        cwd=REPOSITORY_ROOT, env=env, capture_output=True, text=True, timeout=PROGRAM_TIMEOUT_S, check=False,
    )

    # Assert
    assert started.returncode == 0, started.stderr
    assert started.stdout.splitlines() == ['{"members": ["handler"], "handler_is_callable": true}']
    assert f" INFO {ENTRY_LOGGER}: {WHEN_THE_LOOP_IS_STARTED}\n" in started.stderr
    assert STORAGE_URL_CANARY not in started.stderr
    assert _shown((API_KEY, RECEIPT_KEY), started.stdout, started.stderr) == []


# --- through the SDK's own run_job ---------------------------------------------------------------------------------


@pytest.fixture
def run_through_the_sdk() -> Callable[[Callable[[Any], dict[str, Any]], Any], dict[str, Any]]:
    pytest.importorskip("runpod")
    from runpod.serverless.modules.rp_job import run_job

    def run(handler: Callable[[Any], dict[str, Any]], job: Any) -> dict[str, Any]:
        return asyncio.run(asyncio.wait_for(run_job(handler, job), SDK_RUN_TIMEOUT_S))

    return run


def test_sdk_run_job_when_the_job_knows_its_failure_should_record_the_output_and_no_error(run_through_the_sdk):
    # Act
    result = run_through_the_sdk(_handler(_Sessions()), {"id": "job-1", "input": {"contract": "import/v1"}})

    # Assert
    assert result == {"output": {"contract": "import/v1", "status": "failed", "code": "input_refused"}}


def test_sdk_run_job_when_the_job_fails_on_a_bug_should_record_an_error_naming_the_class_and_nothing_of_its_text(
    run_through_the_sdk, monkeypatch, capsys
):
    # Arrange
    monkeypatch.setattr(runpod_import_handler, "run_import", _fail_because_of_another_error)

    # Act
    result = run_through_the_sdk(_handler(_Sessions()), {"id": "job-1", "input": {}})

    # Assert
    recorded = json.loads(result["error"])
    printed = capsys.readouterr()
    assert sorted(result) == ["error"]
    assert recorded["error_message"] == "the import failed on an unexpected LookupError"
    assert _shown(CHAIN_MARKERS, recorded["error_traceback"]) == []
    assert _shown(NEVER_SHOWN, result["error"], printed.out, printed.err) == []
