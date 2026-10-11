"""The version of the installed ffmpeg, as a result's `encoder.ffmpeg_version` states it, and how each
way of not getting it is typed: what a retry cannot cure, a passing shortage, a run that did not end.

The stand-in ffmpegs are small shell scripts named by FFMPEG_PATH, the variable the stages resolve their
ffmpeg from; they run under the same bounded process as the real one (POSIX only). The endings no script
can produce on demand (a failed fork, no temporary directory, a deadline that passes) are put where the
module starts its process: nothing waits and nothing races."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.bounded_process import BoundedResult
from core.failure import Failure, WorkerFailure
from core.job import ffmpeg_version as ffmpeg_version_module
from core.job.ffmpeg_version import installed_ffmpeg_version
from core.protocol import schema

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the version is asked under POSIX process groups")

FFMPEG_PATH_VARIABLE = "FFMPEG_PATH"
ASCII_TEXT_64 = schema.SCHEMA_BASE_URI + "schemas/common.schema.json#/$defs/ascii_text_64"
EXECUTABLE = stat.S_IRWXU
UNAVAILABLE = ("configuration_error", "ffmpeg_unavailable", False)
START_FAILED = ("resource_exhausted", "ffmpeg_version_start_failed", True)
RUN_FAILED = ("internal_error", "ffmpeg_version_run_failed", True)
PATH_CANARY = "/CANARY/path/in/an/os/error"
WORKING_FFMPEG = "echo 'ffmpeg version 7.1.1 Copyright'"
OUTPUT_CAP_BYTES = 16 * 1024
REAL_START = ffmpeg_version_module.run_bounded
TIMED_OUT = BoundedResult(
    returncode=None, stdout=b"", stderr_tail=b"", timed_out=True, stdout_overflow=False, stderr_truncated=False
)
TOOL_TIMEOUT_S = 30


@pytest.fixture(autouse=True)
def fresh_answer() -> Iterator[None]:
    """The answer is kept for the process: every test starts, and leaves, without one."""
    installed_ffmpeg_version.cache_clear()
    yield
    installed_ffmpeg_version.cache_clear()


def _stand_in(directory: Path, script_body: str) -> Path:
    script = directory / "stand-in-ffmpeg"
    script.write_text(f"#!/bin/sh\n{script_body}\n", encoding="ascii")
    script.chmod(EXECUTABLE)
    return script


def _unavailable(monkeypatch: pytest.MonkeyPatch, ffmpeg_path: Path) -> WorkerFailure:
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(ffmpeg_path))
    with pytest.raises(WorkerFailure) as raised:
        installed_ffmpeg_version()
    return raised.value


def _typed(failure: Failure) -> tuple[str, str, bool]:
    return failure.error_class, failure.code, failure.retryable


def _failure_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script_body: str = WORKING_FFMPEG) -> Failure:
    """The failure of asking the stand-in ffmpeg of `script_body` for its version."""
    return _unavailable(monkeypatch, _stand_in(tmp_path, script_body)).failure


def _start_that(outcome: BaseException | BoundedResult) -> Callable[..., BoundedResult]:
    """A stand-in of the bounded process start: it raises `outcome`, or returns it as the run's ending."""

    def start(*arguments: object, **named_arguments: object) -> BoundedResult:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return start


def test_version_when_ffmpeg_states_it_should_be_the_token_after_ffmpeg_version(tmp_path, monkeypatch):
    # Arrange
    script = _stand_in(
        tmp_path,
        "echo 'ffmpeg version 7.1.1-full Copyright (c) 2000-2025 the FFmpeg developers'\necho 'built with gcc'",
    )
    monkeypatch.setenv(FFMPEG_PATH_VARIABLE, str(script))

    # Act
    version = installed_ffmpeg_version()

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
    version = installed_ffmpeg_version()

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
    first, second = installed_ffmpeg_version(), installed_ffmpeg_version()

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
    version = installed_ffmpeg_version()

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
    version = installed_ffmpeg_version()

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


def test_version_when_the_run_passes_its_deadline_should_fail_retryable_as_a_run_failure(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(ffmpeg_version_module, "run_bounded", _start_that(TIMED_OUT))

    # Act
    failure = _failure_with(monkeypatch, tmp_path)

    # Assert
    assert _typed(failure) == RUN_FAILED


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
    version = installed_ffmpeg_version()

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
