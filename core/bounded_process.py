"""A subprocess whose whole process group is killed at its deadline, with capped output (POSIX only).

At the deadline the group gets SIGTERM, then SIGKILL after a one-second grace; reaping and the
bounded reader joins can add a few seconds more, so callers reserve that overrun.
An optional monitor, called at a fixed interval while the process runs, can stop it the same way.
Every stderr byte, not only the kept tail, can be scanned for patterns the caller must not miss.
"""

from __future__ import annotations

import enum
import math
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Protocol

MS_PER_SECOND = 1000
TERMINATE_GRACE_S = 1.0
WAIT_POLL_INTERVAL_S = 0.05
READ_POLL_INTERVAL_S = 0.05
READ_CHUNK_BYTES = 64 * 1024
# Bounded waits for the reader threads once the process has ended: first for them to reach
# end of file, then, after asking them to stop, for them to notice.
READER_DRAIN_TIMEOUT_S = 1.0
READER_STOP_TIMEOUT_S = 1.0
REAP_TIMEOUT_S = 1.0
# The bytes of stderr carried from one read into the next, so a pattern split across two reads is
# still found: a stderr pattern must match within this many bytes.
STDERR_PATTERN_OVERLAP_BYTES = 256


def _require_positive_ms(ms: object) -> int:
    if not isinstance(ms, int) or isinstance(ms, bool) or ms < 1:
        raise ValueError(f"milliseconds must be an int of at least 1, got {ms!r}")
    return ms


@dataclass(frozen=True, slots=True)
class Deadline:
    expires_at: float  # a time.monotonic() value

    def __post_init__(self) -> None:
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, (int, float)):
            raise ValueError("expires_at must be a number")
        if not math.isfinite(self.expires_at):
            raise ValueError("expires_at must be finite")

    @classmethod
    def after_ms(cls, ms: int) -> Deadline:
        return cls(time.monotonic() + _require_positive_ms(ms) / MS_PER_SECOND)

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def within_ms(self, ms: int) -> Deadline:
        """The earlier of this deadline and now + ms: a stage budget only ever shortens it."""
        candidate = time.monotonic() + _require_positive_ms(ms) / MS_PER_SECOND
        return Deadline(min(self.expires_at, candidate))


@dataclass(frozen=True, slots=True)
class BoundedResult:
    """The outcome of run_bounded.

    returncode is None whenever the run timed out, overflowed stdout or was stopped by its monitor:
    the process tree was killed (or never started), so no exit status is meaningful.
    """

    returncode: int | None
    stdout: bytes  # at most max_stdout_bytes
    stderr_tail: bytes  # the last max_stderr_bytes bytes of stderr
    timed_out: bool
    stdout_overflow: bool  # stdout exceeded the cap; the process tree was killed
    stderr_truncated: bool  # stderr held more than max_stderr_bytes in total; the tail lost its beginning
    stopped_by_monitor: bool = False  # the monitor asked to stop; the process tree was killed
    stderr_patterns_seen: frozenset[bytes] = frozenset()  # the .pattern of each stderr pattern that matched


class _Sink(Protocol):
    def accept(self, chunk: bytes) -> bool:
        """Store a chunk; False tells the reader to stop reading."""
        ...


class _HeadBuffer:
    """Keeps the first `cap` bytes and flags the first byte beyond them."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._data = bytearray()
        self.overflowed = threading.Event()

    def accept(self, chunk: bytes) -> bool:
        room = self._cap - len(self._data)
        if len(chunk) <= room:
            self._data += chunk
            return True
        self._data += chunk[:room]
        self.overflowed.set()
        return False

    def contents(self) -> bytes:
        return bytes(self._data)


class _TailBuffer:
    """Keeps a rolling tail of the last `cap` bytes, counts every byte seen, and scans every byte
    for `patterns`, carrying an overlap between chunks."""

    def __init__(self, cap: int, patterns: Sequence[re.Pattern[bytes]] = ()) -> None:
        self._cap = cap
        self._data = bytearray()
        self._total_bytes = 0
        self._patterns = tuple(patterns)
        self._carry = b""
        self._seen: set[bytes] = set()
        self._seen_lock = threading.Lock()

    @property
    def truncated(self) -> bool:
        return self._total_bytes > self._cap

    @property
    def patterns_seen(self) -> frozenset[bytes]:
        with self._seen_lock:
            return frozenset(self._seen)

    def accept(self, chunk: bytes) -> bool:
        self._total_bytes += len(chunk)
        self._scan(chunk)
        self._data += chunk
        excess = len(self._data) - self._cap
        if excess > 0:
            del self._data[:excess]
        return True

    def _scan(self, chunk: bytes) -> None:
        if not self._patterns:
            return
        window = self._carry + chunk
        matched = [pattern.pattern for pattern in self._patterns if pattern.search(window) is not None]
        if matched:
            with self._seen_lock:
                self._seen.update(matched)
        self._carry = window[-STDERR_PATTERN_OVERLAP_BYTES:]

    def contents(self) -> bytes:
        return bytes(self._data)


class _Ending(enum.Enum):
    EXITED = "exited"
    TIMED_OUT = "timed_out"
    OVERFLOWED = "overflowed"
    READ_FAILED = "read_failed"
    STOPPED = "stopped"


class _ReadErrors:
    """Errors that lost output while it was still being read, shared by the reader threads."""

    def __init__(self) -> None:
        self._errors: list[OSError] = []
        self.happened = threading.Event()

    def record(self, error: OSError) -> None:
        self._errors.append(error)
        self.happened.set()

    def first(self) -> OSError | None:
        return self._errors[0] if self._errors else None


def _drain(
    fd: int, selector: selectors.BaseSelector, sink: _Sink, stop: threading.Event, read_errors: _ReadErrors
) -> None:
    try:
        while not stop.is_set():
            if not selector.select(timeout=READ_POLL_INTERVAL_S):
                continue
            chunk = os.read(fd, READ_CHUNK_BYTES)
            if not chunk or not sink.accept(chunk):
                return
    except OSError as error:
        # Once stop is set the bounded join has given up and the pipes are closed on purpose;
        # before that, an error means output was lost, and the run must not look complete.
        if not stop.is_set():
            read_errors.record(error)
    finally:
        selector.close()


def _start_reader(
    pipe: IO[bytes] | None, sink: _Sink, stop: threading.Event, read_errors: _ReadErrors
) -> threading.Thread:
    """Start a reader; its selector is made here, so a failure to make it raises to the caller."""
    if pipe is None:
        raise RuntimeError("subprocess pipe was not created")
    fd = pipe.fileno()
    selector = selectors.DefaultSelector()
    try:
        selector.register(fd, selectors.EVENT_READ)
        reader = threading.Thread(target=_drain, args=(fd, selector, sink, stop, read_errors), daemon=True)
        reader.start()
    except BaseException:
        selector.close()
        raise
    return reader


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        # The group is already gone (or holds only zombies): nothing left to signal.
        return


class _Monitor:
    """Calls the caller's monitor at most once per interval, the first time one interval after start."""

    def __init__(self, check: Callable[[], bool] | None, interval_ms: int | None) -> None:
        self._check = check
        self._interval_s = 0.0 if interval_ms is None else interval_ms / MS_PER_SECOND
        self._next_call_at = time.monotonic() + self._interval_s

    def asks_to_stop(self) -> bool:
        if self._check is None or time.monotonic() < self._next_call_at:
            return False
        keep_running = self._check()
        self._next_call_at = time.monotonic() + self._interval_s
        if not isinstance(keep_running, bool):
            raise TypeError("the monitor must return a bool")
        return not keep_running


def _wait_for_ending(
    process: subprocess.Popen[bytes],
    deadline: Deadline,
    overflowed: threading.Event,
    read_failed: threading.Event,
    monitor: _Monitor,
) -> _Ending:
    while True:
        if overflowed.is_set():
            return _Ending.OVERFLOWED
        if read_failed.is_set():
            return _Ending.READ_FAILED
        if monitor.asks_to_stop():
            return _Ending.STOPPED
        remaining = deadline.remaining_s()
        if remaining <= 0:
            return _Ending.EXITED if process.poll() is not None else _Ending.TIMED_OUT
        try:
            process.wait(timeout=min(remaining, WAIT_POLL_INTERVAL_S))
        except subprocess.TimeoutExpired:
            continue
        return _Ending.EXITED


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        _signal_group(process.pid, signal.SIGKILL)


def _reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # A SIGKILLed process that still has not ended is stuck in the kernel; never hang on it.
        return


def _finish_readers(readers: Sequence[threading.Thread], stop: threading.Event) -> None:
    drain_until = time.monotonic() + READER_DRAIN_TIMEOUT_S
    for reader in readers:
        reader.join(timeout=max(0.0, drain_until - time.monotonic()))
    # A grandchild that escaped into its own session may still hold a pipe open.
    stop.set()
    for reader in readers:
        reader.join(timeout=READER_STOP_TIMEOUT_S)


def _close_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _validated_argv(argv: object) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise ValueError("argv must be a sequence of strings")
    command = list(argv)
    if not command or not all(isinstance(part, str) for part in command):
        raise ValueError("argv must be a non-empty sequence of strings")
    return command


def _validated_env(env: object) -> dict[str, str]:
    if not isinstance(env, Mapping):
        raise ValueError("env must be a mapping of strings")
    environment = dict(env)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise ValueError("env must map strings to strings")
    return environment


def _require_cap(name: str, cap: object) -> None:
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError(f"{name} must be an int of at least 1, got {cap!r}")


def _validated_fds(fds: object) -> tuple[int, ...]:
    if isinstance(fds, (str, bytes)) or not isinstance(fds, Sequence):
        raise ValueError("pass_fds must be a sequence of descriptors")
    descriptors = tuple(fds)
    if not all(isinstance(fd, int) and not isinstance(fd, bool) and fd >= 0 for fd in descriptors):
        raise ValueError("pass_fds must hold non-negative ints")
    return descriptors


def _validated_patterns(patterns: object) -> tuple[re.Pattern[bytes], ...]:
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
        raise ValueError("stderr_patterns must be a sequence of compiled bytes patterns")
    compiled = tuple(patterns)
    if not all(isinstance(pattern, re.Pattern) and isinstance(pattern.pattern, bytes) for pattern in compiled):
        raise ValueError("stderr_patterns must hold compiled bytes patterns")
    return compiled


def _require_monitor(monitor: object, interval_ms: object) -> None:
    if monitor is None and interval_ms is None:
        return
    if monitor is None or interval_ms is None:
        raise ValueError("monitor and monitor_interval_ms must be given together")
    if not callable(monitor):
        raise ValueError("monitor must be callable")
    _require_cap("monitor_interval_ms", interval_ms)


def run_bounded(
    argv: Sequence[str],
    *,
    deadline: Deadline,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    env: Mapping[str, str],
    cwd: str,
    monitor: Callable[[], bool] | None = None,
    monitor_interval_ms: int | None = None,
    pass_fds: Sequence[int] = (),
    stderr_patterns: Sequence[re.Pattern[bytes]] = (),
) -> BoundedResult:
    """Run argv in its own session; kill the whole process group at the deadline or on overflow.

    The child sees exactly `env`, nothing inherited. Output contents are never logged.

    With `monitor` (and `monitor_interval_ms`, given together), the monitor is called at most once
    per interval while the process runs; True lets it continue, False stops it as a deadline would.

    Raises OSError when the process cannot be started, or when its output could not be fully
    read (a reader could not be set up, or a read failed); the process group is dead by then.
    Raises RuntimeError when a reader thread cannot be started, likewise after the cleanup.
    Whatever the monitor raises propagates, likewise after the cleanup.

    `pass_fds` are the only descriptors the child inherits besides its standard streams.

    Every stderr byte is scanned for `stderr_patterns` (each must match within
    STDERR_PATTERN_OVERLAP_BYTES bytes); the result names those that matched, even when the kept
    tail has lost them.
    """
    command = _validated_argv(argv)
    _require_cap("max_stdout_bytes", max_stdout_bytes)
    _require_cap("max_stderr_bytes", max_stderr_bytes)
    environment = _validated_env(env)
    if not isinstance(deadline, Deadline):
        raise ValueError("deadline must be a Deadline")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("cwd must be a non-empty string")
    _require_monitor(monitor, monitor_interval_ms)
    inherited_fds = _validated_fds(pass_fds)
    patterns = _validated_patterns(stderr_patterns)
    if deadline.expired():
        return BoundedResult(
            returncode=None,
            stdout=b"",
            stderr_tail=b"",
            timed_out=True,
            stdout_overflow=False,
            stderr_truncated=False,
        )

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd=cwd,
        start_new_session=True,
        close_fds=True,
        pass_fds=inherited_fds,
    )
    output_monitor = _Monitor(monitor, monitor_interval_ms)
    stdout_buffer = _HeadBuffer(max_stdout_bytes)
    stderr_buffer = _TailBuffer(max_stderr_bytes, patterns)
    stop_reading = threading.Event()
    read_errors = _ReadErrors()
    readers: list[threading.Thread] = []
    ending = _Ending.TIMED_OUT
    try:
        readers.append(_start_reader(process.stdout, stdout_buffer, stop_reading, read_errors))
        readers.append(_start_reader(process.stderr, stderr_buffer, stop_reading, read_errors))
        ending = _wait_for_ending(
            process, deadline, stdout_buffer.overflowed, read_errors.happened, output_monitor
        )
        if ending is not _Ending.EXITED:
            _terminate_group(process)
    finally:
        # start_new_session makes the child's pid its process group id; this SIGKILL also removes
        # stragglers a normally exiting process left behind in its group.
        _signal_group(process.pid, signal.SIGKILL)
        _reap(process)
        _finish_readers(readers, stop_reading)
        _close_pipes(process)

    lost_output = read_errors.first()
    if lost_output is not None:
        raise lost_output

    stdout_overflow = stdout_buffer.overflowed.is_set()
    timed_out = ending is _Ending.TIMED_OUT
    stopped_by_monitor = ending is _Ending.STOPPED
    killed = timed_out or stdout_overflow or stopped_by_monitor
    return BoundedResult(
        returncode=None if killed else process.returncode,
        stdout=stdout_buffer.contents(),
        stderr_tail=stderr_buffer.contents(),
        timed_out=timed_out,
        stdout_overflow=stdout_overflow,
        stderr_truncated=stderr_buffer.truncated,
        stopped_by_monitor=stopped_by_monitor,
        stderr_patterns_seen=stderr_buffer.patterns_seen,
    )
