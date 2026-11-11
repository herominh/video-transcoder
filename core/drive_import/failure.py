"""How a Drive transfer ends when it does not complete: one code of the closed list the acquire maps to contract
2.1.0's failures (core/protocol/acquire_messages.py) and the stage it ended in. Neither holds anything of what the
transfer read, sent or was answered."""

from __future__ import annotations

from enum import StrEnum


class Code(StrEnum):
    """The code of a failed transfer, which the acquire's failed result carries."""

    FILE_REMOVED = "file_removed"
    LINK_NOT_SHARED = "link_not_shared"
    PROVIDER_QUOTA = "provider_quota"
    PROVIDER_REFUSED = "provider_refused"
    NOT_CONFIGURED = "not_configured"
    SIZE_EXCEEDED = "size_exceeded"
    SIZE_SHORT = "size_short"
    INTEGRITY_FAILED = "integrity_failed"
    SOURCE_CHANGED_AT_PROVIDER = "source_changed_at_provider"
    DEADLINE = "deadline"
    DOWNLOAD_FAILED = "download_failed"
    UPLOAD_FAILED = "upload_failed"
    INPUT_REFUSED = "input_refused"


class Stage(StrEnum):
    """Where a transfer is: the log names it, and the acquire maps it to the failed result's stage."""

    INPUT = "input"
    CONFIGURATION = "configuration"
    METADATA = "metadata"
    DOWNLOAD = "download"
    VERIFY = "verify"
    UPLOAD = "upload"
    # The PUT of the acquire's signed result (named for the import/v1 receipt it replaced).
    RECEIPT = "receipt"


class ImportFailed(Exception):
    """The job ends as `failed` with `code`, at `stage`. Its text is the two names and nothing else."""

    def __init__(self, code: Code, stage: Stage) -> None:
        super().__init__(f"{code.value} at {stage.value}")
        self.code = code
        self.stage = stage
