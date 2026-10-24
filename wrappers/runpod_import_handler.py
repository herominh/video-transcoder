"""The RunPod Serverless entry point of the Google Drive import endpoint: one job of the provider is one import
(contract import/v1; core/drive_import/job.py; the Hub's issue #50 item 3b).

The import image runs this file (`Dockerfile.import`). build_handler() runs once, before any job is taken: it checks
the import contract tree the image ships (core.drive_import.contract.verify_import_contract_tree), and a tree that
cannot be trusted stops the process there (ImportContractInvalid names the file and the rule, never content). It
reads the endpoint's configuration once (core.drive_import.settings.ImportSettings.from_env): a variable that is
missing or not usable does not stop the process, it is logged by its name (never its value) and every job then
answers `failed/not_configured` before any request.

For each job the SDK calls the handler with its job, a mapping that has `id` and `input`. Nothing of the job is
trusted here: the value of `input` (None when the job is no mapping or has no input) goes to run_import() as it
came, with a session of its own (closed when the job ends) and the process's clock, and run_import() alone decides
whether it is an input the job may act on. This module looks no further into the job, and logs and keeps none of
it. What run_import() returns is the job's output, {contract, status, code}: every failure the job knows, a refused
input included, is such an output and never an error, so that the Hub reads its code.

The provider records a failed job with the error's own text and its whole traceback, which prints every error the
failure is chained to with that error's text too. A bug's text can quote what it choked on: a URL (a presigned one
is a write capability), a key, a piece of the job input. So the one error that leaves the handler is the wrapper's
own WorkerFault, naming only the bug's class, raised once the bug is no longer being handled, so that it has no
cause and no context and the traceback holds no chain. The bug is logged once by its class and its stack, never
its text, also once it is no longer being handled: when a log handler fails on a line, Python's logging prints a
report of its own to standard error with every error that failure is chained to, which inside the `except` block
would be the bug with its own text (wrappers/runpod_handler.py does the same for the transcode).

Importing this module builds and configures nothing, with one exception made before anything else is imported: it
takes SSLKEYLOGFILE out of the process's environment. The HTTP library writes every TLS session's keys to the file
that variable names (urllib3's create_urllib3_context() reads it when an HTTPS connection creates its context;
requests 2.34.2 does not create a context on import).
What the file does when it is run as the program is main(), which the `__main__` block calls and nothing else: it
sets up the process's logging (configure_logging) before it imports the SDK and before it builds the handler, and
starts the SDK's serverless loop with that handler.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any

# The variable that has the HTTP library write the TLS session keys to a file: out before requests is imported.
TLS_KEY_LOG_VARIABLE = "SSLKEYLOGFILE"
os.environ.pop(TLS_KEY_LOG_VARIABLE, None)

# Ensure project root is on path so `core` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# After the path insertion, as every `core` import here.
from core.drive_import.clock import Clock, SystemClock  # noqa: E402
from core.drive_import.contract import verify_import_contract_tree  # noqa: E402
from core.drive_import.job import run_import  # noqa: E402
from core.drive_import.settings import ImportSettings  # noqa: E402
from core.drive_import.transport import ClosableHttpSession, RequestsSession  # noqa: E402

logger = logging.getLogger(__name__)

# The member of the SDK's job that holds the job input.
INPUT_MEMBER = "input"
# How many errors of a chain (an error, its cause or the error it interrupted, and so on) a bug's log follows.
MAX_CHAINED_ERRORS = 8
# The level and the line format of the process's log, and the HTTP libraries whose loggers are cut off from it:
# each the parent of every logger that library writes on.
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
HTTP_LIBRARY_LOGGERS = ("urllib3", "requests", "httpx", "httpcore")

# What the SDK calls with each job, and what it takes as the job's output.
SdkHandler = Callable[[object], dict[str, Any]]
# What starts the SDK's serverless loop (runpod.serverless.start): it takes the SDK's configuration, whose member
# HANDLER_MEMBER holds the handler.
SdkStarter = Callable[[dict[str, Any]], None]
HANDLER_MEMBER = "handler"
# What makes the session of one job.
SessionFactory = Callable[[], ClosableHttpSession]


class WorkerFault(Exception):
    """The job failed on a bug; the text names the unexpected error's class and nothing more."""


def configure_logging() -> None:
    """Set up the process's log: INFO and above from every logger, in one line format, and nothing at all from the
    HTTP libraries. Called by main(), never on import.

    Their own lines name what they talk to: urllib3's warnings name the host and a request's whole URL ("Retrying
    ... after connection broken ...: <url>", "Failed to parse headers (url=<url>)"), and a presigned URL of the store
    is a write capability. Each library's whole family of loggers is cut off from the process's log at its parent,
    not logger by logger and not by a level (the SDK's import gives urllib3's parent a level of its own, so one set
    here would be lost). The price is that none of their lines is in the log; a request that fails still ends as
    the job's own code, logged in its own words.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    for name in HTTP_LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        # A handler of its own that writes nothing: a record that meets no handler at all goes to Python's handler
        # of last resort, which writes a warning's text to standard error.
        library.addHandler(logging.NullHandler())
        library.propagate = False


def build_handler(env: Mapping[str, str]) -> SdkHandler:
    """The handler of a process whose import contract tree can be trusted, with the configuration read from `env`.

    Raises ImportContractInvalid for the tree before any job could be taken. A configuration that is not whole
    stops nothing: its variables' names are logged, and every job answers `failed/not_configured`.
    """
    verify_import_contract_tree()
    settings = ImportSettings.from_env(env)
    if settings.missing:
        _log_unconfigured(settings.missing)
    return job_handler(settings, RequestsSession, SystemClock())


def job_handler(settings: ImportSettings, new_session: SessionFactory, clock: Clock) -> SdkHandler:
    """The handler the SDK calls with each job: one run_import() with `settings`, a session `new_session` makes for
    the job (closed when it ends) and `clock`. It returns the very dict run_import() returned and raises WorkerFault
    for any error; an error that is no Exception (the interpreter's exit, an interrupt) passes through untouched.
    """

    def handler(job: object) -> dict[str, Any]:
        job_input = job.get(INPUT_MEMBER) if isinstance(job, Mapping) else None
        failed: Exception
        bug: Exception | None = None
        try:
            output = _run_one(job_input, settings, new_session, clock)
        except Exception as error:  # noqa: BLE001 - a bug leaves as the wrapper's own error, without its text
            # Made before the bug is logged: whatever the logging does, this is the error that leaves.
            failed = WorkerFault(f"the import failed on an unexpected {type(error).__name__}")
            bug = error
        else:
            return output
        # Logged here, after the `except` block has ended (see the module's docstring).
        _log_bug(bug)
        # Let go of, as Python lets go of a caught error where its block ends: the bug's stack holds this frame,
        # and this frame would hold the bug.
        bug = None
        # Raised here, after the `except` block has ended, so that the error that was caught is not even kept as
        # this one's context: nothing of it travels with what the provider records.
        raise failed

    return handler


def _run_one(
    job_input: object, settings: ImportSettings, new_session: SessionFactory, clock: Clock
) -> dict[str, Any]:
    session = new_session()
    try:
        return run_import(job_input, settings, session, clock)
    finally:
        _close_session(session)


def _close_session(session: ClosableHttpSession) -> None:
    """Close a job's session. A close that fails with an OSError (a transfer's error, whose text could name a host)
    changes nothing the job decided, so that error is dropped; any other error is a bug and leaves as one."""
    try:
        session.close()
    except OSError:
        return


def _log_unconfigured(missing: tuple[str, ...]) -> None:
    """One warning naming the variables that are missing or not usable, never a value. It never raises."""
    try:
        logger.warning("the import is not configured, every job answers not_configured: %s", ", ".join(missing))
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return


def _log_bug(error: Exception | None) -> None:
    """One error line for a bug: its class and its stack, never its text. It never raises: a line that cannot be
    built or written must not put another error in the place of the wrapper's own. Called once the bug is no longer
    being handled (see the handler)."""
    if error is None:
        return
    try:
        # No `exc_info`: a traceback ends with the error's own text. The class and the stack say where.
        logger.error("an import ended on an unexpected %s", _stack_report(error))
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return


def _chained_to(error: BaseException) -> BaseException | None:
    """The error Python reports before this one: its cause, else the error it interrupted unless suppressed."""
    if error.__cause__ is not None:
        return error.__cause__
    return None if error.__suppress_context__ else error.__context__


def _stack_report(error: BaseException) -> str:
    """Where an unexpected error came from, without a word of what it says: the error's class name and its stack
    (file names, line numbers and source lines), then the same for each error it is chained to."""
    links: list[str] = []
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen and len(seen) < MAX_CHAINED_ERRORS:
        seen.add(id(link))
        links.append(f"{type(link).__name__}\n{''.join(traceback.format_tb(link.__traceback__))}")
        link = _chained_to(link)
    return "chained to ".join(links).rstrip("\n")


def main(start: SdkStarter | None = None, env: Mapping[str, str] | None = None) -> None:
    """All that this file does when it is run as the program: set up the process's log, build the handler from
    `env` (the process's environment when none is given) and start the SDK's serverless loop with it.

    The log is set up first: before the SDK is imported, whose import hangs a handler of its own on a root logger
    that has none, and before the handler is built, so that what the start logs is in the process's log. `start`
    stands in for the SDK's `runpod.serverless.start` in a test; without one the SDK is imported here, after the
    log was set up, and never when this module is imported. What build_handler() raises is raised before the loop
    is started.
    """
    configure_logging()
    if start is None:
        import runpod

        start = runpod.serverless.start
    start({HANDLER_MEMBER: build_handler(os.environ if env is None else env)})


if __name__ == "__main__":
    main()
