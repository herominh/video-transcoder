"""The version of the ffmpeg this process encodes with, for a result's `encoder.ffmpeg_version`.

`ffmpeg -version` runs the way the stages run their tools (core/bounded_process.py: a cleared
environment, an empty working directory, a short deadline, capped output), and the binary is the one
the stages resolve: the FFMPEG_PATH environment variable, else `ffmpeg` on the PATH. The version is the
token after `ffmpeg version ` on the first line, cut to the contract's `ascii_text_64`. The first
version a run states is remembered for the process and returned from then on without a run: the binary
does not change under a running worker.

The check is part of an execution, so it runs under the execution's deadline: its own limit
(VERSION_WALL_MS) only ever shortens that deadline, and with a deadline that has already passed no
process is started.

Without the version no completed result can be built, so the job must not start. Why it could not be
determined is typed as the stages type a tool that cannot be started or run (core/encode.py,
core/probe.py), by one question: would the next attempt, on a healthy host, succeed?

    configuration_error / ffmpeg_unavailable          No, not retryable. No binary, not executable, not a
                                                      program (FileNotFoundError, PermissionError, ENOEXEC,
                                                      exit status 126 or 127); any other non-zero exit; more
                                                      output than the cap; a clean run without a version
                                                      line. What `ffmpeg -version` prints and how it exits
                                                      depends on the image alone: every start meets it again.
    resource_exhausted / ffmpeg_version_start_failed  Yes, retryable. Any other error while starting it (a
                                                      failed fork, no temporary directory, no descriptor
                                                      left): a passing shortage of the worker's own.
    internal_error / ffmpeg_version_run_failed        Yes, retryable. The run passed the check's own limit
                                                      while the execution still had time, or ended on a
                                                      signal: an overloaded or dying host, not the tool.
    deadline_exceeded / ffmpeg_version_deadline_exceeded
                                                      Yes, retryable. The execution's deadline had passed
                                                      before the check began, or had passed when its run
                                                      timed out: the start's time went to its claim or to
                                                      this check, and nothing was read or encoded yet (as
                                                      the runner's wall_time_spent_before_start).

Each failure is logged once, by the name of its ending, and none is kept: the next job asks again.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import tempfile

from core.bounded_process import BoundedResult, Deadline, run_bounded
from core.encode import ENCODER_UNUSABLE_ERRNOS, SHELL_CANNOT_RUN_EXIT_CODES
from core.encode_command import FFMPEG_ENV
from core.failure import Failure, WorkerFailure, make_detail

logger = logging.getLogger(__name__)

FFMPEG_PATH_VARIABLE = "FFMPEG_PATH"
DEFAULT_FFMPEG = "ffmpeg"
VERSION_ARGUMENT = "-version"
VERSION_LINE_PREFIX = "ffmpeg version "
VERSION_WALL_MS = 10_000
VERSION_MAX_OUTPUT_BYTES = 16 * 1024
# The contract's `ascii_text_64`, which `encoder.ffmpeg_version` is.
VERSION_MAX_CHARS = 64
EMPTY_CWD_PREFIX = "ffmpeg-version-"
UNKNOWN_ERRNO_NAME = "unknown"

# The endings that give no version, as a log line names them.
ENDING_NOT_INSTALLED = "not_installed"
ENDING_NOT_STARTABLE = "not_startable"
ENDING_START_FAILED = "start_failed"
ENDING_TIMED_OUT = "timed_out"
ENDING_DEADLINE_EXCEEDED = "deadline_exceeded"
ENDING_ON_A_SIGNAL = "ended_on_a_signal"
ENDING_EXIT_STATUS = "exited_with_a_failure"
ENDING_OUTPUT_TOO_LARGE = "output_too_large"
ENDING_NO_VERSION_LINE = "no_version_line"


def _failed(ending: str, failure: Failure) -> WorkerFailure:
    """The failure of one ending, logged once by the ending's name; its detail is our own words."""
    logger.warning("the ffmpeg version could not be determined (%s): %s", ending, failure.detail)
    return WorkerFailure(failure)


def _unavailable(ending: str, detail: str) -> WorkerFailure:
    return _failed(ending, Failure("configuration_error", "ffmpeg_unavailable", False, detail))


def _start_failed(errno_name: str) -> WorkerFailure:
    detail = f"ffmpeg could not be started to state its version: {errno_name}"
    return _failed(ENDING_START_FAILED, Failure("resource_exhausted", "ffmpeg_version_start_failed", True, detail))


def _run_failed(ending: str, detail: str) -> WorkerFailure:
    return _failed(ending, Failure("internal_error", "ffmpeg_version_run_failed", True, detail))


def _deadline_exceeded() -> WorkerFailure:
    detail = "the execution's deadline passed before ffmpeg stated its version"
    return _failed(
        ENDING_DEADLINE_EXCEEDED, Failure("deadline_exceeded", "ffmpeg_version_deadline_exceeded", True, detail)
    )


def _errno_name(error: OSError) -> str:
    # The errno's symbolic name only: an OSError's own message may carry a path.
    return errno.errorcode.get(error.errno or 0, UNKNOWN_ERRNO_NAME)


# The version a run in this process stated, None until one did. Only a success is ever kept here.
_remembered_version: str | None = None


def installed_ffmpeg_version(deadline: Deadline) -> str:
    """The installed ffmpeg's version, e.g. "7.1.1", determined within the execution's `deadline`.

    The first version a run states is kept for the process and returned from then on without a run,
    whatever the deadline. Otherwise the run is held to the earlier of `deadline` and the check's own
    limit. Raises WorkerFailure when the version cannot be determined, typed as the module's docstring
    says; a failure is not kept.
    """
    global _remembered_version
    if not isinstance(deadline, Deadline):
        raise TypeError("deadline must be a Deadline")
    if _remembered_version is not None:
        return _remembered_version
    if deadline.expired():
        raise _deadline_exceeded()
    executable = shutil.which(os.environ.get(FFMPEG_PATH_VARIABLE) or DEFAULT_FFMPEG)
    if executable is None:
        raise _unavailable(ENDING_NOT_INSTALLED, "ffmpeg is not installed")
    version = _stated_version(_run_version(executable, deadline), deadline)
    _remembered_version = version
    return version


def forget_installed_ffmpeg_version() -> None:
    """Forget the remembered version, so that the next ask runs ffmpeg again. A running worker never
    needs this (its binary does not change); a test that changes the binary does."""
    global _remembered_version
    _remembered_version = None


def _run_version(executable: str, deadline: Deadline) -> BoundedResult:
    """Run `ffmpeg -version` in an empty working directory that lives only while it runs."""
    try:
        with tempfile.TemporaryDirectory(prefix=EMPTY_CWD_PREFIX, ignore_cleanup_errors=True) as empty_dir:
            return _run_in(executable, empty_dir, deadline)
    except OSError as error:
        # The directory could not be made. Always a passing shortage, FileNotFoundError included: mkdtemp
        # reports exhausted descriptors as "No usable temporary directory found". (What _run_in raises
        # is a WorkerFailure, which this lets through.)
        raise _start_failed(_errno_name(error)) from None


def _run_in(executable: str, empty_dir: str, deadline: Deadline) -> BoundedResult:
    try:
        return run_bounded(
            [executable, VERSION_ARGUMENT],
            # The check's own limit, never past the execution's deadline.
            deadline=deadline.within_ms(VERSION_WALL_MS),
            max_stdout_bytes=VERSION_MAX_OUTPUT_BYTES,
            max_stderr_bytes=VERSION_MAX_OUTPUT_BYTES,
            env=FFMPEG_ENV,
            cwd=empty_dir,
        )
    except OSError as error:
        # Also where its output could not be fully read (run_bounded raises after killing the group).
        if isinstance(error, (FileNotFoundError, PermissionError)) or error.errno in ENCODER_UNUSABLE_ERRNOS:
            raise _unavailable(ENDING_NOT_STARTABLE, "ffmpeg could not be started") from None
        raise _start_failed(_errno_name(error)) from None
    except RuntimeError:
        # A reader thread could not start; run_bounded has already killed the group.
        raise _start_failed(UNKNOWN_ERRNO_NAME) from None


def _stated_version(result: BoundedResult, deadline: Deadline) -> str:
    """The version a run of `ffmpeg -version` stated, or the typed failure of the ending it had instead.

    `deadline` is the execution's: a run that timed out once it has passed ran out of the execution's
    time, not of the check's own."""
    if result.timed_out and deadline.expired():
        raise _deadline_exceeded()
    if result.timed_out:
        raise _run_failed(ENDING_TIMED_OUT, f"ffmpeg did not state its version within {VERSION_WALL_MS} ms")
    if result.stdout_overflow:
        raise _unavailable(ENDING_OUTPUT_TOO_LARGE, "ffmpeg wrote more than a version statement")
    returncode = result.returncode
    if returncode is None or returncode < 0:
        raise _run_failed(ENDING_ON_A_SIGNAL, "ffmpeg ended on a signal while stating its version")
    if returncode in SHELL_CANNOT_RUN_EXIT_CODES:
        raise _unavailable(ENDING_NOT_STARTABLE, "ffmpeg could not be started")
    if returncode != 0:
        raise _unavailable(ENDING_EXIT_STATUS, f"ffmpeg -version exited with status {returncode}")
    version = _version_token(result.stdout)
    if version is None:
        raise _unavailable(ENDING_NO_VERSION_LINE, "ffmpeg did not state its version")
    return version


def _version_token(stdout: bytes) -> str | None:
    """The token after `ffmpeg version ` on the first line, made a contract `ascii_text_64`; None without one."""
    first_line = stdout.decode("ascii", "replace").partition("\n")[0]
    if not first_line.startswith(VERSION_LINE_PREFIX):
        return None
    tokens = first_line[len(VERSION_LINE_PREFIX):].split()
    if not tokens:
        return None
    return make_detail(tokens[0])[:VERSION_MAX_CHARS]
