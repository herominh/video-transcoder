"""How an import job ends when it does not import: one code of the contract's closed list (README section 4) and the
stage it ended in, for the log. Neither holds anything of what the job read, sent or was answered."""

from __future__ import annotations

from enum import StrEnum


class Code(StrEnum):
    """The output's `code` of a failed job (output.schema.json)."""

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
    """Where a job is: the log names it, the output never does."""

    INPUT = "input"
    CONFIGURATION = "configuration"
    METADATA = "metadata"
    DOWNLOAD = "download"
    VERIFY = "verify"
    UPLOAD = "upload"
    RECEIPT = "receipt"


class ImportFailed(Exception):
    """The job ends as `failed` with `code`, at `stage`. Its text is the two names and nothing else."""

    def __init__(self, code: Code, stage: Stage) -> None:
        super().__init__(f"{code.value} at {stage.value}")
        self.code = code
        self.stage = stage
