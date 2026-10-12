"""The version of the installed ffmpeg, as a result's `encoder.ffmpeg_version` states it, and how each
way of not getting it is typed: what a retry cannot cure, a passing shortage, a run that did not end, an
execution whose own deadline passed. The check runs under the execution's deadline; only a version it
got is remembered for the process. Its run of ffmpeg gets nothing of this process's: an environment
cleared of it, an empty working directory of its own, and a cap on what is kept of its output.

The stand-in ffmpegs are small shell scripts named by FFMPEG_PATH, the variable the stages resolve their
ffmpeg from; they run under the same bounded process as the real one (POSIX only). The endings no script
can produce on demand (a failed fork, no temporary directory, a deadline that passes) are put where the
module starts its process, and an execution's deadline is one the test scripts (conftest's
ScriptedDeadline): nothing waits and nothing races."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from core.bounded_process import BoundedResult, Deadline
from core.failure import Failure, WorkerFailure
from core.job import ffmpeg_version as ffmpeg_version_module
from core.job.ffmpeg_version import forget_installed_ffmpeg_version, installed_ffmpeg_version
from core.protocol import schema
from tests.job.conftest import ScriptedDeadline, reported_log_failures

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the version is asked under POSIX process groups")

FFMPEG_PATH_VARIABLE = "FFMPEG_PATH"
ASCII_TEXT_64 = schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/ascii_text_64"
EXECUTABLE = stat.S_IRWXU
UNAVAILABLE = ("configuration_error", "ffmpeg_unavailable", False)
START_FAILED = ("resource_exhausted", "ffmpeg_version_start_failed", True)
RUN_FAILED = ("internal_error", "ffmpeg_version_run_failed", True)
DEADLINE_EXCEEDED = ("deadline_exceeded", "ffmpeg_version_deadline_exceeded", True)
# The check's own limit, whatever an execution has left, and what an execution has left that is held
# to less than it.
CHECK_LIMIT_S = 10.0
LESS_THAN_THE_CHECK_LIMIT_MS = 9_000
PATH_CANARY = "/CANARY/path/in/an/os/error"
# A place for temporary directories that is no directory: nothing can be made in it, by root either.
MISSING_DIRECTORY_NAME = "no-such-temporary-directory"
WORKING_FFMPEG = "echo 'ffmpeg version 7.1.1 Copyright'"
OUTPUT_CAP_BYTES = 16 * 1024
# More than a pipe holds (64 KiB on Linux), so a stand-in that writes this much cannot end before all
# but a pipe's worth of it was read: what is kept of it is then full, however busy the host is.
MORE_THAN_A_PIPE_HOLDS_BYTES = 16 * OUTPUT_CAP_BYTES
# A variable of this process alone, as a storage secret is one.
PROCESS_VARIABLE = "VH_TEST_VARIABLE_OF_THE_PROCESS"
REAL_START = ffmpeg_version_module.run_bounded
TIMED_OUT = BoundedResult(
    returncode=None, stdout=b"", stderr_tail=b"", timed_out=True, stdout_overflow=False, stderr_truncated=False
)
# The ending of a run in which ffmpeg stated its version, for a test that starts no process.
STATED_ITS_VERSION = BoundedResult(
    returncode=0,
    stdout=b"ffmpeg version 7.1.1 Copyright\n",
    stderr_tail=b"",
    timed_out=False,
    stdout_overflow=False,
    stderr_truncated=False,
)
TOOL_TIMEOUT_S = 30
# Far more than any stand-in needs: an execution with this much left never meets its deadline here.
PLENTY_OF_TIME_MS = 600_000


@pytest.fixture(autouse=True)
def fresh_answer() -> Iterator[None]:
    """The answer is kept for the process: every test starts, and leaves, without one."""
    forget_installed_ffmpeg_version()
    yield
    forget_installed_ffmpeg_version()


def _time_left() -> Deadline:
    """The deadline of an execution that has plenty of its wall time left."""
    return Deadline.after_ms(PLENTY_OF_TIME_MS)


def _stand_in(directory: Path, script_body: str) -> Path:
    script = directory / "stand-in-ffmpeg"
    script.write_text(f"#!/bin/sh\n{script_body}\n", encoding="ascii")
    script.chmod(EXECUTABLE)
    return script


def _unavailable(
    monkeypatch: pytest.MonkeyPatch, ffmpeg_path: Path, deadline: Deadline | None = None
) -> WorkerFailure:
    """What asking the ffmpeg at `ffmpeg_path` raises, under `deadline` (plenty of time when none is given)."""
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(ffmpeg_path))
    with pytest.raises(WorkerFailure) as raised:
        installed_ffmpeg_version(_time_left() if deadline is None else deadline)
    return raised.value


def _typed(failure: Failure) -> tuple[str, str, bool]:
    return failure.error_class, failure.code, failure.retryable


def _failure_with(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_body: str = WORKING_FFMPEG,
    deadline: Deadline | None = None,
) -> Failure:
    """The failure of asking the stand-in ffmpeg of `script_body` for its version, under `deadline`."""
    return _unavailable(monkeypatch, _stand_in(tmp_path, script_body), deadline).failure


def _counting_runs(directory: Path) -> tuple[Path, Path]:
    """A working stand-in ffmpeg that leaves one line per run in the second path, which no run has made yet."""
    runs = directory / "runs"
    return _stand_in(directory, f"echo run >> '{runs}'\n{WORKING_FFMPEG}"), runs


def _start_that(outcome: BaseException | BoundedResult) -> Callable[..., BoundedResult]:
    """A stand-in of the bounded process start: it raises `outcome`, or returns it as the run's ending."""

    def start(*arguments: object, **named_arguments: object) -> BoundedResult:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return start


def _start_noting_endings(endings: list[BoundedResult]) -> Callable[..., BoundedResult]:
    """The real bounded process start, which also notes the ending of each run it makes."""

    def start(*arguments: Any, **named_arguments: Any) -> BoundedResult:
        ending = REAL_START(*arguments, **named_arguments)
        endings.append(ending)
        return ending

    return start


def _stating_its_version_in(directory: Path, output_bytes: int) -> Path:
    """A stand-in ffmpeg whose whole output is one version statement of exactly `output_bytes` bytes."""
    padded_to = output_bytes - len("\n")
    return _stand_in(directory, f"printf '%-{padded_to}s\\n' 'ffmpeg version 7.1.1 Copyright'")


def test_version_when_ffmpeg_states_it_should_be_the_token_after_ffmpeg_version(tmp_path, monkeypatch):
    # Arrange
    script = _stand_in(
        tmp_path,
        "echo 'ffmpeg version 7.1.1-full Copyright (c) 2000-2025 the FFmpeg developers'\necho 'built with gcc'",
    )
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert version == "7.1.1-full"


def test_version_when_the_local_ffmpeg_is_asked_should_be_what_its_first_line_states(monkeypatch):
    # Arrange
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    monkeypatch.delenv(FFMPEG_PATH_VARIABLE, raising=False)
    first_line = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, timeout=TOOL_TIMEOUT_S
    ).stdout.decode("ascii", "replace").splitlines()[0]

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert first_line.startswith(f"ffmpeg version {version}")
    assert version != ""
    assert schema.error_pointers_for_schema({"$ref": ASCII_TEXT_64}, version) == []


def test_version_when_asked_twice_should_run_ffmpeg_once(tmp_path, monkeypatch):
    # Arrange: the stand-in leaves one line per run
    runs = tmp_path / "runs"
    script = _stand_in(tmp_path, f"echo run >> '{runs}'\necho 'ffmpeg version 7.1.1 Copyright'")
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    first, second = installed_ffmpeg_version(_time_left()), installed_ffmpeg_version(_time_left())

    # Assert
    assert (first, second) == ("7.1.1", "7.1.1")
    assert runs.read_text(encoding="ascii").splitlines() == ["run"]


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        pytest.param("n7.1-" + "x" * 100, "n7.1-" + "x" * 59, id="longer-than-64-characters"),
        pytest.param("https://mirror.example.invalid/ffmpeg-7.1", "[url]", id="a-url"),
        pytest.param("7.1\\303\\251", "7.1??", id="bytes-outside-ascii"),
    ],
)
def test_version_when_the_stated_token_does_not_fit_the_contract_should_be_made_to_fit(
    tmp_path, monkeypatch, stated, expected
):
    # Arrange
    script = _stand_in(tmp_path, f"printf 'ffmpeg version {stated} Copyright\\n'")
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert version == expected
    assert schema.error_pointers_for_schema({"$ref": ASCII_TEXT_64}, version) == []


@pytest.mark.parametrize(
    "script_body",
    [
        pytest.param("echo 'ffprobe version 7.1.1 Copyright'", id="another-tool"),
        pytest.param("echo 'ffmpeg version '", id="no-version-token"),
        pytest.param("exit 0", id="no-output"),
        pytest.param("echo 'ffmpeg version 7.1.1 Copyright'\nexit 3", id="a-failing-exit"),
        pytest.param("echo 'usage'\necho 'ffmpeg version 7.1.1 Copyright'", id="not-on-the-first-line"),
    ],
)
def test_version_when_ffmpeg_does_not_state_it_should_fail_as_ffmpeg_unavailable(tmp_path, monkeypatch, script_body):
    # Arrange
    script = _stand_in(tmp_path, script_body)

    # Act
    failure = _unavailable(monkeypatch, script).failure

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == UNAVAILABLE


def test_version_when_ffmpeg_is_not_installed_should_fail_as_ffmpeg_unavailable(tmp_path, monkeypatch):
    # Arrange
    missing = tmp_path / "no-such-ffmpeg"

    # Act
    failure = _unavailable(monkeypatch, missing).failure

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == UNAVAILABLE
    assert str(tmp_path) not in failure.detail


def test_version_when_ffmpeg_cannot_be_executed_should_fail_as_ffmpeg_unavailable(tmp_path, monkeypatch):
    # Arrange: an executable file that is no program (no interpreter line, not a binary)
    broken = tmp_path / "broken-ffmpeg"
    broken.write_bytes(b"\x00\x01\x02 not a program")
    broken.chmod(EXECUTABLE)

    # Act
    failure = _unavailable(monkeypatch, broken).failure

    # Assert
    assert (failure.error_class, failure.code) == ("configuration_error", "ffmpeg_unavailable")
    assert str(tmp_path) not in failure.detail


def test_version_when_a_first_ask_failed_should_ask_again_the_next_time(tmp_path, monkeypatch):
    # Arrange: no ffmpeg at first, then one
    _unavailable(monkeypatch, tmp_path / "no-such-ffmpeg")
    script = _stand_in(tmp_path, "echo 'ffmpeg version 7.1.1 Copyright'")
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert version == "7.1.1"


# --- how each way of not getting the version is typed -----------------------------------------------


@pytest.mark.parametrize("status", [126, 127])
def test_version_when_ffmpeg_exits_as_a_program_that_cannot_run_should_fail_as_ffmpeg_unavailable(
    tmp_path, monkeypatch, status
):
    # Act
    failure = _failure_with(monkeypatch, tmp_path, f"exit {status}")

    # Assert
    assert _typed(failure) == UNAVAILABLE


def test_version_when_ffmpeg_may_not_be_executed_should_fail_as_ffmpeg_unavailable(tmp_path, monkeypatch):
    # Arrange: the start is refused for good
    refused = PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(refused))

    # Act
    failure = _failure_with(monkeypatch, tmp_path)

    # Assert
    assert _typed(failure) == UNAVAILABLE
    assert PATH_CANARY not in failure.detail


def test_version_when_ffmpeg_is_gone_when_it_is_started_should_fail_as_ffmpeg_unavailable(tmp_path, monkeypatch):
    # Arrange: the binary was found, and is missing at the start (removed in between, or its interpreter is)
    gone = FileNotFoundError(errno.ENOENT, "No such file or directory", PATH_CANARY)
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(gone))

    # Act
    failure = _failure_with(monkeypatch, tmp_path)

    # Assert
    assert _typed(failure) == ("configuration_error", "ffmpeg_unavailable", False)
    assert PATH_CANARY not in failure.detail


def test_version_when_ffmpeg_writes_more_than_a_version_statement_should_fail_as_ffmpeg_unavailable(
    tmp_path, monkeypatch
):
    # Arrange: a version line, then more output than the cap
    script_body = f"{WORKING_FFMPEG}\nhead -c {4 * OUTPUT_CAP_BYTES} /dev/zero"

    # Act
    failure = _failure_with(monkeypatch, tmp_path, script_body)

    # Assert
    assert _typed(failure) == UNAVAILABLE


@pytest.mark.parametrize(
    ("error_number", "name"),
    [
        pytest.param(errno.EAGAIN, "EAGAIN", id="a-failed-fork"),
        pytest.param(errno.ENOMEM, "ENOMEM", id="no-memory"),
        pytest.param(errno.EMFILE, "EMFILE", id="no-descriptor-left"),
    ],
)
def test_version_when_ffmpeg_cannot_be_started_for_a_passing_shortage_should_fail_retryable(
    tmp_path, monkeypatch, error_number, name
):
    # Arrange
    shortage = OSError(error_number, f"cannot start {PATH_CANARY}")
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(shortage))

    # Act
    failure = _failure_with(monkeypatch, tmp_path)

    # Assert: the errno's name, nothing of the error's own text
    assert _typed(failure) == START_FAILED
    assert name in failure.detail
    assert PATH_CANARY not in failure.detail


def test_version_when_a_reader_thread_cannot_start_should_fail_retryable(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(RuntimeError("can't start new thread")))

    # Act
    failure = _failure_with(monkeypatch, tmp_path)

    # Assert
    assert _typed(failure) == START_FAILED


def test_version_when_no_working_directory_can_be_made_should_fail_retryable_and_not_as_unavailable(
    tmp_path, monkeypatch
):
    # Arrange: what mkdtemp raises when the descriptors or the disk ran out: a FileNotFoundError, too
    script = _stand_in(tmp_path, WORKING_FFMPEG)
    no_directory = FileNotFoundError(errno.ENOENT, f"No usable temporary directory found in {PATH_CANARY}")
    monkeypatch.setattr(ffmpeg_version_module.tempfile, "TemporaryDirectory", _start_that(no_directory))

    # Act
    failure = _unavailable(monkeypatch, script).failure

    # Assert
    assert _typed(failure) == START_FAILED
    assert "ENOENT" in failure.detail
    assert PATH_CANARY not in failure.detail


def test_version_when_no_working_directory_can_be_made_should_say_so_and_not_name_a_failed_start(
    tmp_path, monkeypatch
):
    # Arrange: a working ffmpeg, while temporary directories are to be made where no directory is
    script = _stand_in(tmp_path, WORKING_FFMPEG)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / MISSING_DIRECTORY_NAME))

    # Act
    failure = _unavailable(monkeypatch, script).failure

    # Assert: typed as a failed start is, in words of its own; the errno's name, nothing of the path
    assert _typed(failure) == START_FAILED
    assert "working directory" in failure.detail
    assert "started" not in failure.detail
    assert "ENOENT" in failure.detail
    assert str(tmp_path) not in failure.detail


def test_version_when_no_working_directory_can_be_made_should_log_that_ending_once_and_no_failed_start(
    tmp_path, monkeypatch, caplog
):
    # Arrange
    caplog.set_level(logging.DEBUG, logger=ffmpeg_version_module.__name__)
    script = _stand_in(tmp_path, WORKING_FFMPEG)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / MISSING_DIRECTORY_NAME))

    # Act
    _unavailable(monkeypatch, script)

    # Assert
    (logged,) = [record for record in caplog.records if record.name == ffmpeg_version_module.__name__]
    assert "(no_working_directory)" in logged.getMessage()
    assert "start" not in logged.getMessage()
    assert str(tmp_path) not in caplog.text


def test_version_when_the_run_passes_the_checks_own_limit_with_time_left_should_fail_retryable_as_a_run_failure(
    tmp_path, monkeypatch
):
    # Arrange: the run timed out, and the execution's deadline has not passed
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(TIMED_OUT))
    deadline = ScriptedDeadline(False)

    # Act
    failure = _failure_with(monkeypatch, tmp_path, deadline=deadline)

    # Assert
    assert _typed(failure) == ("internal_error", "ffmpeg_version_run_failed", True)


def test_version_when_ffmpeg_ends_on_a_signal_should_fail_retryable_as_a_run_failure(tmp_path, monkeypatch):
    # Act: the stand-in kills itself, as the kernel kills a process that ran out of memory
    failure = _failure_with(monkeypatch, tmp_path, "kill -KILL $$")

    # Assert
    assert _typed(failure) == RUN_FAILED


def test_version_when_a_first_ask_failed_retryably_should_ask_again_the_next_time(tmp_path, monkeypatch):
    # Arrange: a passing shortage on the first ask only
    script = _stand_in(tmp_path, WORKING_FFMPEG)
    with monkeypatch.context() as during_the_shortage:
        shortage = OSError(errno.EAGAIN, "Resource temporarily unavailable")
        during_the_shortage.setattr(ffmpeg_version_module, "run_bounded", _start_that(shortage))
        first = _unavailable(during_the_shortage, script).failure
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert _typed(first) == START_FAILED
    assert version == "7.1.1"


@pytest.mark.parametrize(
    ("script_body", "start", "ending"),
    [
        pytest.param("exit 127", REAL_START, "not_startable", id="cannot-run"),
        pytest.param("exit 3", REAL_START, "exited_with_a_failure", id="a-failing-exit"),
        pytest.param("echo usage", REAL_START, "no_version_line", id="no-version-line"),
        pytest.param("kill -KILL $$", REAL_START, "ended_on_a_signal", id="a-signal"),
        pytest.param(WORKING_FFMPEG, _start_that(TIMED_OUT), "timed_out", id="a-passed-deadline"),
        pytest.param(
            WORKING_FFMPEG, _start_that(OSError(errno.EAGAIN, f"cannot start {PATH_CANARY}")), "start_failed",
            id="a-passing-shortage",
        ),
    ],
)
def test_version_when_it_cannot_be_determined_should_log_the_ending_once_by_its_name(
    tmp_path, monkeypatch, caplog, script_body, start, ending
):
    # Arrange
    caplog.set_level(logging.DEBUG, logger=ffmpeg_version_module.__name__)
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", start)

    # Act
    _failure_with(monkeypatch, tmp_path, script_body)

    # Assert
    (logged,) = [record for record in caplog.records if record.name == ffmpeg_version_module.__name__]
    assert f"({ending})" in logged.getMessage()
    assert PATH_CANARY not in caplog.text
    assert str(tmp_path) not in caplog.text


# --- an error that a typed failure replaces ---------------------------------------------------------

# The places where the module catches an error and ends the check with a typed failure in its place:
# no working directory can be made; the start is refused for good, by the system itself (a file that is
# no program) or as a stand-in has it; the start meets a passing shortage; a reader thread cannot start.
NO_WORKING_DIRECTORY = "no-working-directory"
NOT_A_PROGRAM = "not-a-program"
REFUSED_FOR_GOOD = "refused-for-good"
A_PASSING_SHORTAGE = "a-passing-shortage"
NO_READER_THREAD = "no-reader-thread"
# What each ends as: the typed failure, the name its log line gives the ending, and the failure's detail.
REPLACED_ERROR_SITES = {
    NO_WORKING_DIRECTORY: (
        START_FAILED,
        "no_working_directory",
        "the empty working directory ffmpeg states its version in could not be made: ENOENT",
    ),
    NOT_A_PROGRAM: (UNAVAILABLE, "not_startable", "ffmpeg could not be started"),
    REFUSED_FOR_GOOD: (UNAVAILABLE, "not_startable", "ffmpeg could not be started"),
    A_PASSING_SHORTAGE: (START_FAILED, "start_failed", "ffmpeg could not be started to state its version: EAGAIN"),
    NO_READER_THREAD: (START_FAILED, "start_failed", "ffmpeg could not be started to state its version: unknown"),
}


def _asked_where_an_error_is_replaced(site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkerFailure:
    """Ask for the version where `site` fails, each time on an error whose own text names a path
    (PATH_CANARY; for the file that is no program, the test's own directory), and return what the ask raised."""
    ffmpeg = _stand_in(tmp_path, WORKING_FFMPEG)
    if site == NO_WORKING_DIRECTORY:
        # What mkdtemp raises when the descriptors or the disk ran out.
        no_directory = FileNotFoundError(errno.ENOENT, f"No usable temporary directory found in {PATH_CANARY}")
        monkeypatch.setattr(ffmpeg_version_module.tempfile, "TemporaryDirectory", _start_that(no_directory))
    if site == NOT_A_PROGRAM:
        # No stand-in fails here: an executable file with no interpreter line that is no binary.
        ffmpeg = tmp_path / "broken-ffmpeg"
        ffmpeg.write_bytes(b"\x00\x01\x02 not a program")
        ffmpeg.chmod(EXECUTABLE)
    if site == REFUSED_FOR_GOOD:
        refused = PermissionError(errno.EACCES, "Permission denied", PATH_CANARY)
        monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(refused))
    if site == A_PASSING_SHORTAGE:
        shortage = OSError(errno.EAGAIN, f"cannot start {PATH_CANARY}")
        monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(shortage))
    if site == NO_READER_THREAD:
        no_thread = RuntimeError(f"can't start new thread for {PATH_CANARY}")
        monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(no_thread))
    return _unavailable(monkeypatch, ffmpeg)


@pytest.mark.parametrize("site", list(REPLACED_ERROR_SITES))
def test_version_when_an_error_is_replaced_by_a_typed_failure_should_carry_nothing_of_that_error(
    tmp_path, monkeypatch, site
):
    # Arrange: what the check ends as at this site
    typed, _, detail = REPLACED_ERROR_SITES[site]

    # Act
    raised = _asked_where_an_error_is_replaced(site, tmp_path, monkeypatch)

    # Assert: the typed failure alone. The error it replaces, whose text names a path, is neither its
    # cause nor its context
    assert (_typed(raised.failure), raised.failure.detail) == (typed, detail)
    assert [told for told in (PATH_CANARY, str(tmp_path)) if told in str(raised)] == []
    assert raised.__cause__ is None and raised.__context__ is None


# A log handler that fails on a line does not raise: Python's logging prints a report of its own to
# standard error, with every error that failure is chained to. A line written while an error is still
# being handled would put that error's own text into the report. `failing_log` makes the handler of the
# check's one line fail; how the check ends is how it ends without it.


@pytest.mark.parametrize("site", list(REPLACED_ERROR_SITES))
def test_version_when_a_log_handler_fails_on_the_line_of_a_replaced_error_should_keep_the_errors_text_out(
    tmp_path, monkeypatch, capsys, failing_log, site
):
    # Arrange: what the check ends as at this site, and the name its line gives that ending
    typed, ending, detail = REPLACED_ERROR_SITES[site]

    # Act
    raised = _asked_where_an_error_is_replaced(site, tmp_path, monkeypatch)

    # Assert: logging did report its failure on that ending's line (a report shows the line's arguments,
    # the ending's name first), with nothing of the path the error named
    written = reported_log_failures(capsys, f"('{ending}', ")
    assert [told for told in (PATH_CANARY, str(tmp_path)) if told in written] == []
    assert (_typed(raised.failure), raised.failure.detail) == (typed, detail)


# --- what the run of ffmpeg is given, and what is kept of its output --------------------------------


def test_version_when_the_process_holds_variables_should_run_ffmpeg_with_none_of_them(tmp_path, monkeypatch):
    # Arrange: the stand-in notes the environment it runs with; this process holds a variable of its
    # own beside the one that names the binary
    seen = tmp_path / "environment"
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(_stand_in(tmp_path, f"env > '{seen}'\n{WORKING_FFMPEG}")))
    monkeypatch.setenv(PROCESS_VARIABLE, "of-this-process")

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert: it noted an environment, and neither variable is in it
    seen_names = {line.partition("=")[0] for line in seen.read_text(encoding="ascii", errors="replace").splitlines()}
    assert version == "7.1.1"
    assert seen_names
    assert not {PROCESS_VARIABLE, FFMPEG_PATH_VARIABLE} & seen_names


def test_version_when_ffmpeg_runs_should_run_it_in_an_empty_directory_that_is_gone_afterwards(tmp_path, monkeypatch):
    # Arrange: the stand-in notes where it runs and what it finds there; this process's own working
    # directory holds files
    where, found = tmp_path / "where", tmp_path / "found"
    listing = f"ls -A > '{found}' || echo 'no listing was made' > '{found}'"
    script = _stand_in(tmp_path, f"pwd > '{where}'\n{listing}\n{WORKING_FFMPEG}")
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))
    monkeypatch.chdir(tmp_path)

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    working_directory = Path(os.fsdecode(where.read_bytes()).rstrip("\n"))
    assert version == "7.1.1"
    assert found.read_bytes() == b""
    assert working_directory.is_absolute()
    assert not working_directory.exists()


def test_version_when_ffmpeg_writes_exactly_the_output_cap_should_still_read_its_version(tmp_path, monkeypatch):
    # Arrange: one version statement, padded to the cap and not a byte more
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(_stating_its_version_in(tmp_path, OUTPUT_CAP_BYTES)))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert version == "7.1.1"


def test_version_when_ffmpeg_writes_one_byte_more_than_the_output_cap_should_fail_as_ffmpeg_unavailable(
    tmp_path, monkeypatch
):
    # Arrange
    script = _stating_its_version_in(tmp_path, OUTPUT_CAP_BYTES + 1)

    # Act
    failure = _unavailable(monkeypatch, script).failure

    # Assert
    assert _typed(failure) == UNAVAILABLE


def test_version_when_ffmpeg_writes_more_than_the_output_cap_to_stderr_should_keep_the_caps_worth_of_it(
    tmp_path, monkeypatch
):
    # Arrange: the stand-in writes far more than the cap to stderr before it states its version
    endings: list[BoundedResult] = []
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_noting_endings(endings))
    script = _stand_in(tmp_path, f"head -c {MORE_THAN_A_PIPE_HOLDS_BYTES} /dev/zero >&2\n{WORKING_FFMPEG}")
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert: what it wrote there costs the check nothing, and the cap's worth of it was kept, no more
    (ending,) = endings
    assert version == "7.1.1"
    assert len(ending.stderr_tail) == OUTPUT_CAP_BYTES


# --- under the execution's deadline -----------------------------------------------------------------


def _start_recording_deadlines(deadlines: list[Deadline]) -> Callable[..., BoundedResult]:
    """A stand-in of the bounded process start that notes the deadline each run is held to and starts
    no process: the run ends at once, having stated a version, however busy the host is."""

    def start(*arguments: object, **named_arguments: object) -> BoundedResult:
        deadline = named_arguments["deadline"]
        assert isinstance(deadline, Deadline)
        deadlines.append(deadline)
        return STATED_ITS_VERSION

    return start


def test_version_when_the_executions_deadline_has_already_passed_should_fail_as_its_deadline_without_starting_ffmpeg(
    tmp_path, monkeypatch
):
    # Arrange: a working ffmpeg that would leave a line if it ran
    script, runs = _counting_runs(tmp_path)

    # Act
    failure = _unavailable(monkeypatch, script, ScriptedDeadline(True)).failure

    # Assert
    assert _typed(failure) == ("deadline_exceeded", "ffmpeg_version_deadline_exceeded", True)
    assert not runs.exists()
    assert str(tmp_path) not in failure.detail


def test_version_when_the_run_times_out_and_the_executions_deadline_has_passed_should_fail_as_its_deadline(
    tmp_path, monkeypatch
):
    # Arrange: the deadline had not passed when the check began, and has when its run is found timed out
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(TIMED_OUT))
    deadline = ScriptedDeadline(False, True)

    # Act
    failure = _failure_with(monkeypatch, tmp_path, deadline=deadline)

    # Assert
    assert _typed(failure) == ("deadline_exceeded", "ffmpeg_version_deadline_exceeded", True)


def test_version_when_it_is_known_should_be_returned_without_a_run_even_after_the_deadline_has_passed(
    tmp_path, monkeypatch
):
    # Arrange: one ask with time left, which runs ffmpeg
    script, runs = _counting_runs(tmp_path)
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))
    first = installed_ffmpeg_version(_time_left())

    # Act
    second = installed_ffmpeg_version(ScriptedDeadline(True))

    # Assert
    assert (first, second) == ("7.1.1", "7.1.1")
    assert runs.read_text(encoding="ascii").splitlines() == ["run"]


def test_version_when_a_first_ask_met_a_passed_deadline_should_ask_again_the_next_time(tmp_path, monkeypatch):
    # Arrange: a first execution whose deadline had passed
    script, runs = _counting_runs(tmp_path)
    first = _unavailable(monkeypatch, script, ScriptedDeadline(True)).failure

    # Act: the next execution has time
    version = installed_ffmpeg_version(_time_left())

    # Assert: the failure was not kept, and the version now is
    assert _typed(first) == DEADLINE_EXCEEDED
    assert version == "7.1.1"
    assert runs.read_text(encoding="ascii").splitlines() == ["run"]


def test_version_when_a_run_timed_out_should_not_be_remembered(tmp_path, monkeypatch):
    # Arrange: a first ask whose run times out on the check's own limit
    script, runs = _counting_runs(tmp_path)
    with monkeypatch.context() as during_the_time_out:
        during_the_time_out.setattr(ffmpeg_version_module, "run_bounded", _start_that(TIMED_OUT))
        first = _unavailable(during_the_time_out, script, ScriptedDeadline(False)).failure
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version(_time_left())

    # Assert
    assert _typed(first) == RUN_FAILED
    assert version == "7.1.1"
    assert runs.read_text(encoding="ascii").splitlines() == ["run"]


def test_version_when_the_execution_has_less_left_than_the_checks_limit_should_hold_the_run_to_its_deadline(
    tmp_path, monkeypatch
):
    # Arrange: an execution with 9 s left, one less than the check's own limit; no process is started,
    # so nothing here waits on the host
    held_to: list[Deadline] = []
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_recording_deadlines(held_to))
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(_stand_in(tmp_path, WORKING_FFMPEG)))
    deadline = Deadline.after_ms(LESS_THAN_THE_CHECK_LIMIT_MS)

    # Act
    version = installed_ffmpeg_version(deadline)

    # Assert
    (run_deadline,) = held_to
    assert version == "7.1.1"
    assert run_deadline.expires_at == deadline.expires_at


def test_version_when_the_execution_has_more_left_than_the_checks_limit_should_hold_the_run_to_that_limit(
    tmp_path, monkeypatch
):
    # Arrange: an execution with ten minutes left
    held_to: list[Deadline] = []
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_recording_deadlines(held_to))
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(_stand_in(tmp_path, WORKING_FFMPEG)))
    deadline = _time_left()
    started = time.monotonic()

    # Act
    version = installed_ffmpeg_version(deadline)
    ended = time.monotonic()

    # Assert: ten seconds from the ask, far sooner than the execution's own deadline
    (run_deadline,) = held_to
    assert version == "7.1.1"
    assert started + CHECK_LIMIT_S <= run_deadline.expires_at <= ended + CHECK_LIMIT_S
    assert run_deadline.expires_at < deadline.expires_at


@pytest.mark.parametrize(
    ("answers", "start"),
    [
        pytest.param((True,), REAL_START, id="passed-before-the-check"),
        pytest.param((False, True), _start_that(TIMED_OUT), id="passed-when-the-run-timed-out"),
    ],
)
def test_version_when_the_executions_deadline_ends_the_check_should_log_the_ending_once_by_its_name(
    tmp_path, monkeypatch, caplog, answers, start
):
    # Arrange
    caplog.set_level(logging.DEBUG, logger=ffmpeg_version_module.__name__)
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", start)

    # Act
    _failure_with(monkeypatch, tmp_path, deadline=ScriptedDeadline(*answers))

    # Assert
    (logged,) = [record for record in caplog.records if record.name == ffmpeg_version_module.__name__]
    assert "(deadline_exceeded)" in logged.getMessage()
    assert str(tmp_path) not in caplog.text


@pytest.mark.parametrize("deadline", [None, 10.0, "soon"], ids=["none", "a-number-of-seconds", "text"])
def test_version_when_asked_without_a_deadline_should_raise_type_error_even_when_it_is_known(
    tmp_path, monkeypatch, deadline
):
    # Arrange: a version is already known
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(_stand_in(tmp_path, WORKING_FFMPEG)))
    installed_ffmpeg_version(_time_left())

    # Act and Assert
    with pytest.raises(TypeError):
        installed_ffmpeg_version(deadline)  # type: ignore[arg-type]
