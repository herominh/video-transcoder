"""The preflight: establish a source's media facts from a bounded, conditional read.

ffprobe never sees storage or the network. It reads one loopback URL that this process serves
from a budgeted range source; every byte it asks for is fetched by `BudgetedSource`, which
enforces the byte, request and time budget independently of ffprobe. ffprobe itself may only
speak HTTP and TCP (to reach that URL) and may only choose a demuxer of the profile, so a
playlist or any other format that opens further URLs is refused before it opens one.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from core.admission import AdmissionDecision, RequestLimits, decide_admission
from core.bounded_process import BoundedResult, Deadline, run_bounded
from core.failure import Failure, WorkerFailure
from core.profile import MediaProfile
from core.source_facts import ProbeInconclusive, SourceFacts, parse_ffprobe_output
from core.source_reader import BudgetedSource, RangeSource, ReadStats

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
URL_TOKEN_BYTES = 32
# libavformat logs this line when no container or stream timestamp gives the duration.
DURATION_ESTIMATED_MARKER = b"Estimating duration from bitrate"
# A shell's "found but not executable" and "not found" exit statuses.
SHELL_CANNOT_RUN_EXIT_CODES = frozenset({126, 127})
# Start errors that mean the configured ffprobe itself cannot run (besides FileNotFoundError and
# PermissionError). Any other start error (EAGAIN, ENOMEM, EMFILE, ENFILE, ...) is a passing shortage
# of the worker's own resources.
PROBER_UNUSABLE_ERRNOS = frozenset({errno.ENOEXEC})
UNKNOWN_ERRNO_NAME = "unknown"
# A handler error other than a dropped connection (a thread that cannot start, a bug): the worker
# failed, not the media, so it outranks whatever the prober then reports.
LOOPBACK_SERVER_FAILURE = Failure(
    error_class="internal_error", code="loopback_server_failed", retryable=True,
    detail="the preflight's loopback server failed",
)
# The child's whole environment: no proxy variable can redirect its one HTTP request.
FFPROBE_ENV = {"LC_ALL": "C", "PATH": os.defpath}
MICROSECONDS_PER_SECOND = 1_000_000
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")

# Codes a parse can end in, and the class each one reports.
INCONCLUSIVE_CLASSES = {
    "duration_unknown": "input_limits_exceeded",
    "frame_rate_unknown": "input_invalid",
    "video_geometry_unknown": "input_invalid",
    "unsupported_rotation": "input_invalid",
    "probe_output_invalid": "input_invalid",
}


@dataclass(frozen=True, slots=True)
class PreflightResult:
    facts: SourceFacts
    decision: AdmissionDecision
    read: ReadStats


def _failure(error_class: str, code: str, detail: str, *, retryable: bool = False) -> WorkerFailure:
    return WorkerFailure(Failure(error_class=error_class, code=code, retryable=retryable, detail=detail))


def run_preflight(
    source: RangeSource,
    *,
    profile: MediaProfile,
    limits: RequestLimits,
    deadline: Deadline,
    ffprobe_path: str | None = None,
) -> PreflightResult:
    """Probe `source` within the profile's budget and decide its admission.

    Returns the facts and the decision when the facts could be established; raises WorkerFailure
    when they could not (storage failure, changed source, budget or deadline exhausted, media
    ffprobe cannot read). A source above the size limit is refused before any read.
    """
    size_bytes = source.size_bytes
    size_limit = min(profile.source.max_source_bytes, limits.max_source_bytes)
    if size_bytes > size_limit:
        raise _failure(
            "input_limits_exceeded", "source_too_large",
            f"source of {size_bytes} bytes exceeds the limit of {size_limit} bytes",
        )
    executable = _resolve_ffprobe(ffprobe_path)
    probe_deadline = deadline.within_ms(profile.probe.max_wall_ms)
    budgeted = BudgetedSource(
        source,
        max_bytes=profile.probe.max_bytes,
        max_requests=profile.probe.max_requests,
        deadline=probe_deadline,
    )

    # Setting up the loopback server and the empty working directory needs descriptors and a
    # thread; running out of either is the worker's passing shortage, never the media's fault.
    # The prober-start mapping inside raises WorkerFailure, which this except lets through.
    try:
        with LoopbackRangeServer(budgeted, chunk_bytes=profile.probe.chunk_bytes) as server:
            with tempfile.TemporaryDirectory(prefix="preflight-", ignore_cleanup_errors=True) as empty_dir:
                try:
                    result = run_bounded(
                        _ffprobe_argv(executable, server.url, profile, probe_deadline),
                        deadline=probe_deadline,
                        max_stdout_bytes=profile.probe.max_output_bytes,
                        max_stderr_bytes=profile.probe.max_stderr_bytes,
                        env=FFPROBE_ENV,
                        cwd=empty_dir,
                    )
                except OSError as error:
                    # Also where ffprobe's output could not be fully read (run_bounded raises after
                    # killing the group): reported with the start failures, same class and retry.
                    raise _prober_start_failure(error) from None
                except RuntimeError as error:
                    # A reader thread could not start; run_bounded has already killed the group.
                    raise _prober_thread_failure(error) from None
            upstream_failure = server.failure
    except (OSError, RuntimeError) as error:
        raise _setup_failure(error) from None

    # A storage or budget failure explains whatever ffprobe printed after it, so it wins.
    if upstream_failure is not None:
        raise WorkerFailure(upstream_failure)
    facts = _facts_from(result)
    decision = decide_admission(facts, size_bytes=size_bytes, profile=profile, limits=limits)
    return PreflightResult(facts=facts, decision=decision, read=budgeted.stats)


def _prober_start_failure(error: OSError) -> WorkerFailure:
    """A broken ffprobe installation is permanent; any other start failure is worth a retry."""
    logger.warning("ffprobe could not be started: %s errno=%s", type(error).__name__, error.errno)
    if isinstance(error, (FileNotFoundError, PermissionError)) or error.errno in PROBER_UNUSABLE_ERRNOS:
        return _failure("configuration_error", "ffprobe_unavailable", "ffprobe could not be started")
    # The errno's symbolic name only: the exception's own message may carry a path.
    errno_name = errno.errorcode.get(error.errno, UNKNOWN_ERRNO_NAME)
    return _failure(
        "resource_exhausted", "prober_start_failed",
        f"ffprobe could not be started: {errno_name}", retryable=True,
    )


def _prober_thread_failure(error: RuntimeError) -> WorkerFailure:
    logger.warning("ffprobe could not be started: %s errno=%s", type(error).__name__, None)
    return _failure(
        "resource_exhausted", "prober_start_failed",
        f"ffprobe could not be started: {UNKNOWN_ERRNO_NAME}", retryable=True,
    )


def _setup_failure(error: OSError | RuntimeError) -> WorkerFailure:
    """Any setup error is a passing shortage, FileNotFoundError included: mkdtemp reports
    exhausted descriptors as "No usable temporary directory found"."""
    error_number = getattr(error, "errno", None)
    logger.warning("preflight setup failed: %s errno=%s", type(error).__name__, error_number)
    errno_name = errno.errorcode.get(error_number, UNKNOWN_ERRNO_NAME)
    return _failure(
        "resource_exhausted", "preflight_setup_failed",
        f"the preflight could not be set up: {errno_name}", retryable=True,
    )


def _resolve_ffprobe(ffprobe_path: str | None) -> str:
    candidate = ffprobe_path or os.environ.get("FFPROBE_PATH") or "ffprobe"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise _failure("configuration_error", "ffprobe_unavailable", "ffprobe is not installed")
    return resolved


def _ffprobe_argv(executable: str, url: str, profile: MediaProfile, deadline: Deadline) -> list[str]:
    io_timeout_us = max(1, int(deadline.remaining_s() * MICROSECONDS_PER_SECOND))
    return [
        executable,
        "-hide_banner",
        "-v", "warning",
        "-protocol_whitelist", "http,tcp",
        "-format_whitelist", ",".join(profile.source.allowed_demuxers),
        "-probesize", str(profile.probe.probesize_bytes),
        "-analyzeduration", str(profile.probe.analyze_duration_us),
        "-rw_timeout", str(io_timeout_us),
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-i", url,
    ]


def _facts_from(result: BoundedResult) -> SourceFacts:
    if result.timed_out:
        raise _failure("deadline_exceeded", "probe_deadline_exceeded", "the preflight ran out of time")
    if result.stdout_overflow:
        raise _failure(
            "input_limits_exceeded", "probe_output_too_large",
            "the prober's description of the source exceeds its limit",
        )
    if result.returncode is None or result.returncode < 0:
        # Killed by a signal (out of memory, a crash): the worker failed, not necessarily the media.
        raise _failure(
            "internal_error", "prober_crashed",
            "the prober ended on a signal", retryable=True,
        )
    if result.returncode in SHELL_CANNOT_RUN_EXIT_CODES:
        raise _failure("configuration_error", "ffprobe_unavailable", "ffprobe could not be started")
    if result.returncode != 0:
        # With the demuxer whitelist, "not media" and "media of an unlisted format" look alike.
        # ffprobe's own words are never forwarded: they hold the loopback URL and its token.
        raise _failure("input_invalid", "unrecognized_media", "the source is not media of a supported format")
    if result.stderr_truncated:
        # The duration-estimate warning may be among the lost lines: never admit on a partial log.
        raise _failure(
            "input_limits_exceeded", "probe_diagnostics_too_large",
            "the prober's warnings about the source exceed their limit",
        )
    try:
        return parse_ffprobe_output(
            result.stdout,
            duration_estimated=DURATION_ESTIMATED_MARKER in result.stderr_tail,
        )
    except ProbeInconclusive as inconclusive:
        error_class = INCONCLUSIVE_CLASSES.get(inconclusive.code, "input_invalid")
        raise _failure(error_class, inconclusive.code, inconclusive.detail) from None


class LoopbackRangeServer:
    """Serves one source at one unguessable loopback URL, in aligned chunks fetched on demand.

    Chunks already fetched are kept, so the prober re-reading a header costs nothing; the
    cache cannot outgrow the budget, because every chunk in it was charged to the budget.
    The first upstream failure is recorded and ends every connection; the caller reads it
    from `failure` after the prober has finished. Fetches hold their own lock, so reading
    `failure` never waits for a stalled fetch.
    """

    def __init__(self, source: BudgetedSource, *, chunk_bytes: int) -> None:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self._source = source
        self._chunk_bytes = chunk_bytes
        self._chunks: dict[int, bytes] = {}
        self._fetch_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._failure: Failure | None = None
        self._path = "/" + secrets.token_urlsafe(URL_TOKEN_BYTES)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("the server is not running")
        port = self._server.server_address[1]
        return f"http://{LOOPBACK_HOST}:{port}{self._path}"

    @property
    def failure(self) -> Failure | None:
        with self._state_lock:
            return self._failure

    @property
    def path(self) -> str:
        return self._path

    @property
    def size_bytes(self) -> int:
        return self._source.size_bytes

    @property
    def chunk_bytes(self) -> int:
        return self._chunk_bytes

    def record_failure(self, failure: Failure) -> None:
        """Keep the first failure; later ones are consequences of it."""
        with self._state_lock:
            if self._failure is None:
                self._failure = failure

    def __enter__(self) -> LoopbackRangeServer:
        server = _LoopbackHTTPServer((LOOPBACK_HOST, 0), _handler_for(self), owner=self)
        thread = threading.Thread(target=server.serve_forever, name="preflight-loopback", daemon=True)
        try:
            thread.start()
        except BaseException:
            # __exit__ never runs when __enter__ raises: release the listening socket here.
            server.server_close()
            raise
        self._server = server
        self._thread = thread
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def chunk(self, index: int) -> bytes:
        """The chunk at `index`, fetched once; raises WorkerFailure on any upstream refusal."""
        with self._fetch_lock:
            recorded = self.failure
            if recorded is not None:
                raise WorkerFailure(recorded)
            cached = self._chunks.get(index)
            if cached is not None:
                return cached
            start = index * self._chunk_bytes
            length = min(self._chunk_bytes, self._source.size_bytes - start)
            try:
                data = self._source.read(start, length)
            except WorkerFailure as failure:
                self.record_failure(failure.failure)
                raise
            self._chunks[index] = data
            return data


class _LoopbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        owner: LoopbackRangeServer,
    ) -> None:
        self._owner = owner
        super().__init__(server_address, handler_class)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The prober drops a connection on every seek; only other errors are worth a line.
        error = sys.exc_info()[1]
        if isinstance(error, OSError):
            return
        logger.error("preflight loopback handler failed: %s", type(error).__name__)
        self._owner.record_failure(LOOPBACK_SERVER_FAILURE)


def parse_range(header: str | None, size_bytes: int) -> tuple[int, int] | None:
    """The inclusive byte range a single-range `Range` header asks for, clamped to the object.

    Returns None for a header this server does not satisfy (several ranges, a malformed value,
    a start beyond the object). A missing header asks for the whole object.
    """
    if header is None:
        return 0, size_bytes - 1
    match = RANGE_PATTERN.match(header.strip())
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if first == "" and last == "":
        return None
    if first == "":
        suffix = int(last)
        if suffix == 0:
            return None
        return max(0, size_bytes - suffix), size_bytes - 1
    start = int(first)
    end = size_bytes - 1 if last == "" else min(int(last), size_bytes - 1)
    if start >= size_bytes or end < start:
        return None
    return start, end


def _handler_for(server: LoopbackRangeServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def log_message(self, format: str, *args: Any) -> None:
            # The request line holds the URL token; nothing of it is logged.
            return

        def _serve(self, *, send_body: bool) -> None:
            if self.path != server.path:
                self._empty(404)
                return
            size = server.size_bytes
            has_range = self.headers.get("Range") is not None
            byte_range = parse_range(self.headers.get("Range"), size)
            if byte_range is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                return
            start, end = byte_range
            self.send_response(206 if has_range else 200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            if has_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Connection", "close")
            self.end_headers()
            if send_body:
                self._stream(start, end)

        def _stream(self, start: int, end: int) -> None:
            chunk_bytes = server.chunk_bytes
            position = start
            while position <= end:
                index = position // chunk_bytes
                try:
                    data = server.chunk(index)
                except WorkerFailure:
                    self._abort()
                    return
                offset = position - index * chunk_bytes
                piece = data[offset: offset + (end - position + 1)]
                try:
                    self.wfile.write(piece)
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    # The prober moved on (a seek opens a new connection); stop fetching.
                    return
                position += len(piece)

        def _abort(self) -> None:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                return

        def _empty(self, status: int) -> None:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

    return Handler
