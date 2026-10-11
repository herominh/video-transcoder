"""The RunPod Serverless entry point: one job of the provider is one start of the worker under contract v2.

The image runs this file (`Dockerfile.runpod`). build_handler() runs once, before any job is taken: it
checks the contract tree the image ships and reads the worker's whole configuration
(core.protocol.handler.load_runtime, core.job.config.JobSettings). A tree or a configuration that cannot
be trusted stops the process there (ContractTreeInvalid, SettingsInvalid: each names the file or the
variable and the rule, never a value), so a worker never serves with either.

For each job the SDK calls the handler with its job, a mapping that has `id` and `input`. Nothing of the
job is trusted here: the value of `input` (None when the job is no mapping or has no input) goes to
core.protocol.handler.handle() as it came, and handle() alone decides whether it is a dispatch this
worker may act on. This module looks no further into the job, and logs and keeps none of it. What
handle() returns is the job's output: the carrier `{vh_message, vh_signature}` of the execution's signed
result or of a signed `transcode.unclaimed`.

The provider records a failed job with the error's own text and its whole traceback, which prints every
error the failure is chained to with that error's text too (the SDK's run_job: str(error) and
traceback.format_exc()). An error's text can quote what the error choked on: an endpoint, a path, a
secret, a piece of the job input. So only two errors leave the handler, both the wrapper's own and both
raised once the error that caused them is no longer being handled, so that neither has a cause or a
context and the traceback holds no chain:

    JobRefused    the job input is not a dispatch this worker may act on (README 15.1: nothing of a
                  refused dispatch is trusted and nothing is reported under its identity, so the
                  provider marks the job failed). Its text is the refusal's: the step, the layer and
                  the reason, never content.
    WorkerFault   anything else, a bug. Its text names the error's class and nothing more; the class
                  and the stack are logged once, without the error's text.

Once a storage request was abandoned in this process, at the deadline or at its own ceiling
(core.job.storage._within counts it, in a job that completed too), every returned output also carries
`refresh_worker: True`, until the process ends. It is the process that is marked, not a job: the abandoned
request's helper thread and its connection live until the storage ends the request and belong to no single job
(a read of the preflight is waited for by a thread nobody joins, so it can be given up after the start that
made it has returned), and in a process that goes on taking jobs they would pile up. A job that fails has no
output to carry the member; the next one that returns carries it. The SDK takes that member out of the output
and reports the job with `stopPod`, on which the provider stops the pod after the job, so the Hub still reads
exactly the carrier.

Importing this module builds and configures nothing: the process's logging is set up
(configure_logging), the SDK imported and the handler built only when this file is run.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any

# Ensure project root is on path so `core` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.job.config import JobSettings  # noqa: E402 - after the path insertion, as every `core` import here
from core.job.runner import JobRunner  # noqa: E402
from core.job.storage import abandoned_requests  # noqa: E402
from core.protocol.dispatch import DispatchRefused  # noqa: E402
from core.protocol.handler import JobRunner as RunJob  # noqa: E402 - the callable handle() runs a job with
from core.protocol.handler import Runtime, handle, load_runtime  # noqa: E402

logger = logging.getLogger(__name__)

# The member of the SDK's job that holds the job input, and the member of a handler's output on which
# the SDK has the pod stopped after the job (it takes the member out of the output).
INPUT_MEMBER = "input"
REFRESH_WORKER_MEMBER = "refresh_worker"
# How many errors of a chain (an error, its cause or the error it interrupted, and so on) a bug's log follows.
MAX_CHAINED_ERRORS = 8
# The level and the line format of the process's log, and the logger of the HTTP library under the
# storage clients: the parent of every logger that library writes on.
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
HTTP_LIBRARY_LOGGER = "urllib3"

# What the SDK calls with each job, and what it takes as the job's output.
SdkHandler = Callable[[object], dict[str, Any]]


class JobRefused(Exception):
    """The job input is not a dispatch this worker may act on; the text is the refusal's own: the step,
    the layer and the reason, never content."""


class WorkerFault(Exception):
    """The start failed on a bug; the text names the unexpected error's class and nothing more."""


def configure_logging() -> None:
    """Set up the process's log: INFO and above from every logger, in one line format, and nothing at
    all from the HTTP library under the storage clients. Called when this file is run, never on import.

    urllib3's own lines name what it talks to, and no log line of the worker holds an endpoint, a bucket
    or an object key: "Connection pool is full, discarding connection: <host>" (once abandoned requests
    pile up on one client) and "Certificate did not match expected hostname: <host>" name the storage's
    host; "Failed to parse headers (url=<url>)", on an answer with a malformed header line, and "Retrying
    ... after connection broken ...: <url>" name the request, so the bucket and the object key too.

    The library's whole family of loggers is cut off from the process's log at its parent, not logger by
    logger and not by a level. The library has moved a warning from one of its loggers to another between
    versions, the image pins none, and the SDK's import gives that parent a level of its own, so one set
    here would be lost. The price is that none of the library's lines is in the log; a request that fails
    still ends as an error of the worker's, typed and logged in its own words.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    http_library = logging.getLogger(HTTP_LIBRARY_LOGGER)
    # A handler of its own that writes nothing: a record that meets no handler at all goes to Python's
    # handler of last resort, which writes a warning's text to standard error.
    http_library.addHandler(logging.NullHandler())
    http_library.propagate = False


def build_handler(env: Mapping[str, str]) -> SdkHandler:
    """The handler of a process whose contract tree and whole configuration, read from `env`, can be trusted.

    Raises ContractTreeInvalid for the tree and SettingsInvalid for the configuration, before any job
    could be taken: each names the file or the variable and the rule, never a value.
    """
    runtime = load_runtime(env)
    job_settings = JobSettings.from_env(env)
    runner = JobRunner(job_settings)
    return job_handler(runtime, runner.run)


def job_handler(runtime: Runtime, run_job: RunJob) -> SdkHandler:
    """The handler the SDK calls with each job: one start of the worker with `runtime` and `run_job`.

    While no storage request was ever abandoned in this process, the handler returns the output carrier
    core.protocol.handler.handle() returned, the very dict. Once one was, every output it returns is
    that carrier with `refresh_worker: True` added, until the process ends: the process is marked, not
    the job, because a leftover thread and its connection belong to no single job (the count is read
    when a job returns, so a request given up between two jobs, or in a start that raised, marks the
    next job that returns). It raises JobRefused for a refused dispatch and WorkerFault for any other
    error; an error that is no Exception (the interpreter's exit, an interrupt) passes through untouched.
    """

    def handler(job: object) -> dict[str, Any]:
        # A job that is no mapping, or has no input, is refused by the dispatch's own validation.
        job_input = job.get(INPUT_MEMBER) if isinstance(job, Mapping) else None
        failed: Exception
        try:
            carrier = handle(job_input, runtime=runtime, run_job=run_job)
        except DispatchRefused as refused:
            # Its text names the step, the layer and the reason only (core/protocol/dispatch.py).
            failed = JobRefused(str(refused))
        except Exception as error:  # noqa: BLE001 - a bug leaves as the wrapper's own error, without its text
            # Made before the bug is logged: whatever the logging does, this is the error that leaves.
            failed = WorkerFault(f"the worker failed on an unexpected {type(error).__name__}")
            _log_bug(error)
        else:
            if abandoned_requests() > 0:
                _log_marked_process()
                return {**carrier, REFRESH_WORKER_MEMBER: True}
            return carrier
        # Raised here, after the `except` blocks have ended, so that the error that was caught is not
        # even kept as this one's context: nothing of it travels with what the provider records.
        raise failed

    return handler


def _log_bug(error: Exception) -> None:
    """One error line for a bug: its class and its stack, never its text. It never raises: a line that
    cannot be built or written must not put another error, with the bug as its context, in the bug's place."""
    try:
        # No `exc_info`: a traceback ends with the error's own text. The class and the stack say where.
        logger.error("a start ended on an unexpected %s", _stack_report(error))
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return


def _log_marked_process() -> None:
    """One warning line for an output that asks for a fresh worker. It never raises: a line that cannot be
    written must not cost a finished job its result."""
    try:
        logger.warning("this process abandoned a storage request: asking for a fresh worker")
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return


def _chained_to(error: BaseException) -> BaseException | None:
    """The error Python reports before this one: its cause, else the error it interrupted unless suppressed."""
    if error.__cause__ is not None:
        return error.__cause__
    return None if error.__suppress_context__ else error.__context__


def _stack_report(error: BaseException) -> str:
    """Where an unexpected error came from, without a word of what it says, as core/job/runner.py reports a bug.

    The error's class name and its stack (file names, line numbers and source lines), then the same for
    each error it is chained to. No error's own text is part of it.
    """
    links: list[str] = []
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen and len(seen) < MAX_CHAINED_ERRORS:
        seen.add(id(link))
        links.append(f"{type(link).__name__}\n{''.join(traceback.format_tb(link.__traceback__))}")
        link = _chained_to(link)
    return "chained to ".join(links).rstrip("\n")


if __name__ == "__main__":
    # Here, and before the SDK is imported, so that importing this module configures no logger.
    configure_logging()

    import runpod

    runpod.serverless.start({"handler": build_handler(os.environ)})
