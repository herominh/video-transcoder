"""One import job (contract import/v1; the Hub's import design, sections 2 and 6, G-10, G-17, G-18): the input
checked, the Drive file's metadata asked again, the file downloaded and written part by part into the Hub's
multipart upload, then one signed receipt beside the original.

run_import() takes everything outside it as a parameter: the input document as RunPod hands it, the settings
(ImportSettings), the session it sends its requests through (transport.HttpSession) and the clock (clock.Clock).
In order:

1. The input against input.schema.json, the settings (all three, or `not_configured` before any request), the
   input's MAC under the import key (contract.input_mac_verifies(), README section 3.1: the worker acts only on an
   input the Hub wrote), then the input's other rules (input_check.check()); any refusal of the input is
   `input_refused`.
2. Google's metadata of the file again (google.change_at_provider()): before anything is written, a file that is no
   longer the pasted one is refused.
3. The download, from the start and, after a reset or a time-out, again from the start of the part in flight
   (`Range`), at most DOWNLOAD_TRIES requests for each part; its bytes counted against the expected size (the first
   byte beyond it is `size_exceeded` at once, fewer at the end `size_short`) and hashed (SHA-256, MD5).
4. Each part of 64 MiB, as soon as it is whole, PUT to its URL (store.put()), the store's ETag kept; the LAST part
   only once the whole stream matched Google's SHA-256 when the input carries one, else its MD5, else nothing
   (`checksum_verified`; a mismatch is `integrity_failed` and the last part is never written).
5. The receipt {contract, job, bytes, checksum_verified, parts, mac}, its MAC under the receipt key
   (contract.receipt_mac()), checked against receipt.schema.json and PUT to the receipt URL.

The output is {contract, status: "imported", code: null} or {contract, status: "failed", code}, checked against
output.schema.json before it is returned. A failure the job knows is an output, never an exception: run_import()
raises only on a bug. Nothing of a URL, the file id, a key, Google's text, an ETag or an answer's body goes into a
log line, an exception's text or the output; a log line holds the job's uuid, the stage, the code, counts and
seconds, and is written only once no error is being handled (a log handler that fails prints the error being
handled, whose text would name a URL).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from core.drive_import import contract, store
from core.drive_import.clock import Clock, StopLine
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.input_check import ImportInput, InputRefused, check
from core.drive_import.settings import ImportSettings
from core.drive_import.transport import HttpSession
from core.drive_import.transfer import DOWNLOAD_CHUNK_BYTES, Part, Tally, TransferPlan, transfer

logger = logging.getLogger(__name__)

IMPORTED = "imported"
FAILED = "failed"
RECEIPT_HEADERS = {"Content-Type": "application/json"}


class SelfCheckFailed(RuntimeError):
    """A document the job built is not one of the contract: a bug, never a job's ending."""


def run_import(
    input_document: object, settings: ImportSettings, http: HttpSession, clock: Clock
) -> dict[str, Any]:
    """One import job, from its input document to its output document (the module says how)."""
    started = clock.monotonic()
    tally = Tally()
    failure: ImportFailed | None = None
    try:
        _import(input_document, settings, http, clock, tally)
    except ImportFailed as failed:
        failure = failed
    output = _output(failure)
    _log_outcome(tally, failure, clock.monotonic() - started)
    return output


def _import(document: object, settings: ImportSettings, http: HttpSession, clock: Clock, tally: Tally) -> None:
    # The schema's first rule is that the input is an object; the isinstance() only says so to the type checker.
    if not isinstance(document, Mapping) or contract.first_error_pointer(contract.INPUT_SCHEMA, document) is not None:
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    tally.job = document["job"]
    api_key, receipt_key, storage_host = settings.google_api_key, settings.receipt_key, settings.storage_host
    if api_key is None or receipt_key is None or storage_host is None:
        raise ImportFailed(Code.NOT_CONFIGURED, Stage.CONFIGURATION)
    # The receipt key is the import key, which signs the input too (README section 3.1).
    if not contract.input_mac_verifies(document, receipt_key):
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    job = _checked_input(document, storage_host, clock)
    stop_line = StopLine.before(job.writes_end_at, clock)
    stop_line.require(Stage.INPUT)

    plan = TransferPlan(
        file_id=job.file_id, resource_key=job.resource_key, expected_size=job.expected_size,
        expected_md5=job.expected_md5, expected_sha1=job.expected_sha1, expected_sha256=job.expected_sha256,
        expected_head_revision_id=job.expected_head_revision_id, part_size=job.part_size,
        parts=tuple(Part(part.number, part.url) for part in job.parts), stop_line=stop_line, reject_url_etags=False,
    )
    result = transfer(plan, api_key, http, tally)
    _write_receipt(job, receipt_key, http, stop_line, tally, [part.to_dict() for part in result.parts], result.checksum_verified)


def _checked_input(document: Mapping[str, Any], storage_host: str, clock: Clock) -> ImportInput:
    """The input's rules beyond its schema (input_check.check()); a refusal is `input_refused`."""
    try:
        checked: ImportInput | None = check(document, storage_host, clock.now())
    except InputRefused:
        # The refusal's text names the rule only; it is answered below, once it is no longer being handled, so
        # that nothing of it travels with the job's own failure.
        checked = None
    if checked is None:
        raise ImportFailed(Code.INPUT_REFUSED, Stage.INPUT)
    return checked


# --- the receipt (README section 5) --------------------------------------------------------------------------------


def _write_receipt(
    job: ImportInput,
    receipt_key: str,
    http: HttpSession,
    stop_line: StopLine,
    tally: Tally,
    parts: list[dict[str, Any]],
    checksum_verified: str,
) -> None:
    receipt: dict[str, Any] = {
        "contract": contract.CONTRACT,
        "job": job.job,
        "bytes": job.expected_size,
        "checksum_verified": checksum_verified,
        "parts": parts,
    }
    receipt[contract.MAC_MEMBER] = contract.receipt_mac(receipt, receipt_key)
    if contract.first_error_pointer(contract.RECEIPT_SCHEMA, receipt) is not None:
        raise SelfCheckFailed("the receipt the job built is not one of the contract")
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")
    store.put(
        http,
        job.receipt_url,
        body,
        RECEIPT_HEADERS,
        stop_line,
        Stage.RECEIPT,
        lambda attempt: _log_failed_try(tally, Stage.RECEIPT, None, attempt),
    )


# --- the output and the log ----------------------------------------------------------------------------------------


def _output(failure: ImportFailed | None) -> dict[str, Any]:
    if failure is None:
        output: dict[str, Any] = {"contract": contract.CONTRACT, "status": IMPORTED, "code": None}
    else:
        output = {"contract": contract.CONTRACT, "status": FAILED, "code": failure.code.value}
    if contract.first_error_pointer(contract.OUTPUT_SCHEMA, output) is not None:
        raise SelfCheckFailed("the output the job built is not one of the contract")
    return output


def _log_outcome(tally: Tally, failure: ImportFailed | None, seconds: float) -> None:
    job = tally.job or "-"
    if failure is None:
        _say(
            logging.INFO,
            "import %s: imported, %d bytes read, %d parts written, %.1f s",
            job, tally.bytes_read, tally.parts_written, seconds,
        )
        return
    _say(
        logging.WARNING,
        "import %s: failed at %s with %s, %d bytes read, %d parts written, %.1f s",
        job, failure.stage.value, failure.code.value, tally.bytes_read, tally.parts_written, seconds,
    )


def _log_failed_try(tally: Tally, stage: Stage, part: int | None, attempt: int) -> None:
    if part is None:
        _say(logging.WARNING, "import %s: %s try %d failed", tally.job or "-", stage.value, attempt)
        return
    _say(logging.WARNING, "import %s: %s of part %d, try %d failed", tally.job or "-", stage.value, part, attempt)


def _say(level: int, message: str, *arguments: object) -> None:
    """One log line. It never raises: a line that cannot be written must not cost the job its ending."""
    try:
        logger.log(level, message, *arguments)
    except Exception:  # noqa: BLE001 - dropped, see the docstring
        # Nothing can be done about a log line that cannot be written: saying so would need the log.
        return
