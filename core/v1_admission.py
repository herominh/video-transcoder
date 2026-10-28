"""Admission for the retained v1 development adapter and staged Modal entry."""

import os
from typing import NoReturn

DEVELOPMENT_ADAPTER_ENV = "VH_DEVELOPMENT_ADAPTER"
V1_DISABLED_DETAIL = "The v1 execution path is a development-only adapter and is disabled."
MODAL_STAGED_DETAIL = "Modal is staged as the named second candidate; it is not on the adapter contract yet."


class ExecutionDisabled(RuntimeError):
    """An entry point is not admitted to execute a transcode."""


def v1_execution_admitted() -> bool:
    return os.environ.get(DEVELOPMENT_ADAPTER_ENV) == "1"


def require_v1_execution() -> None:
    """Recheck admission at each HTTP request and each pipeline invocation."""
    if not v1_execution_admitted():
        raise ExecutionDisabled(V1_DISABLED_DETAIL)


def refuse_modal_execution() -> NoReturn:
    """Modal stays staged independently of the development adapter's opt-in."""
    raise ExecutionDisabled(MODAL_STAGED_DETAIL)
