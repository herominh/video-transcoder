"""RunPod CPU entry for signed source.acquire requests (protocol 2.1.0)."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

os.environ.pop("SSLKEYLOGFILE", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drive_import.acquire import handle  # noqa: E402
from core.drive_import.transport import ClosableHttpSession, RequestsSession  # noqa: E402
from core.protocol.dispatch import DispatchRefused  # noqa: E402
from core.protocol.runtime import Runtime, load_runtime  # noqa: E402
from core.protocol.settings import AcquireSettings  # noqa: E402
from core.protocol.transport import HttpsTransport, Transport  # noqa: E402

logger = logging.getLogger(__name__)
HTTP_LIBRARY_LOGGERS = ("urllib3", "requests", "httpx", "httpcore")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SdkHandler = Callable[[object], dict[str, str]]
SdkStarter = Callable[[dict[str, Any]], None]
SessionFactory = Callable[[], ClosableHttpSession]
TransportFactory = Callable[[], Transport]


class JobRefused(Exception):
    """A refused dispatch has no identity under which to report."""


class WorkerFault(Exception):
    """An unexpected failure, with no original text or exception chain."""


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    for name in HTTP_LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.addHandler(logging.NullHandler())
        library.propagate = False


def build_handler(env: Mapping[str, str]) -> SdkHandler:
    runtime = load_runtime(env, AcquireSettings)
    return job_handler(runtime, RequestsSession, HttpsTransport)


def job_handler(
    runtime: Runtime[AcquireSettings], new_session: SessionFactory,
    new_transport: TransportFactory | None = None,
) -> SdkHandler:
    def handler(job: object) -> dict[str, str]:
        failure: Exception
        bug: Exception | None = None
        try:
            started = runtime.monotonic()
            runtime_id = runtime.new_uuid()
            current = replace(runtime, transport=new_transport()) if new_transport is not None else runtime
            session = new_session()
            try:
                job_input = job.get("input") if isinstance(job, Mapping) else None
                output = handle(job_input, runtime=current, http=session, started_at=started, runtime_id=runtime_id)
            finally:
                _close(session)
        except DispatchRefused as refused:
            failure = JobRefused(str(refused))
        except Exception as error:
            failure = WorkerFault(f"the acquire failed on an unexpected {type(error).__name__}")
            bug = error
        else:
            return output
        if bug is not None:
            _log_bug(bug)
            bug = None
        # Raise outside the caught error's scope: even a provider's full traceback contains no secret chain.
        raise failure

    return handler


def _close(session: ClosableHttpSession) -> None:
    try:
        session.close()
    except Exception:
        pass


def _log_bug(error: Exception) -> None:
    try:
        stack = "; ".join(f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__))
        logger.error("acquire failed on %s at %s", type(error).__name__, stack)
    except Exception:
        pass


def main(start: SdkStarter | None = None, env: Mapping[str, str] | None = None) -> None:
    configure_logging()
    if start is None:
        import runpod

        start = runpod.serverless.start
    start({"handler": build_handler(os.environ if env is None else env)})


if __name__ == "__main__":
    main()
