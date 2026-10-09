from __future__ import annotations

import errno
import os
import re
import selectors
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from core.bounded_process import BoundedResult, Deadline, run_bounded

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX only")

SHELL = "/bin/sh"
BASE_ENV = {"PATH": "/bin:/usr/bin"}
GENEROUS_DEADLINE_MS = 10_000
SHORT_DEADLINE_MS = 500
RETURN_WITHIN_S = 5.0
REAP_POLL_LIMIT_S = 5.0
REAP_POLL_INTERVAL_S = 0.05
ONE_KIB = 1024
CAP = 4096
SMALL_STDERR_BYTES = 100
ZERO_BYTES_WRITTEN = 100_000
# A grandchild that calls setsid() leaves the process group, keeps the pipes open and sleeps.
ESCAPING_GRANDCHILD = (
    "import os, sys, time\n"
    "os.setsid()\n"
    "descriptor = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC)\n"
    "os.write(descriptor, str(os.getpid()).encode())\n"
    "os.close(descriptor)\n"
    "time.sleep(30)\n"
)


def _run_shell(
    script: str,
    tmp_path: Path,
    *,
    deadline_ms: int = GENEROUS_DEADLINE_MS,
    max_stdout: int = CAP,
    max_stderr: int = CAP,
    args: tuple[str, ...] = (),
) -> BoundedResult:
    return run_bounded(
        [SHELL, "-c", script, "sh", *args],
        deadline=Deadline.after_ms(deadline_ms),
        max_stdout_bytes=max_stdout,
        max_stderr_bytes=max_stderr,
        env=BASE_ENV,
        cwd=str(tmp_path),
    )


def _run_python(code: str, tmp_path: Path, *, max_stderr: int = CAP) -> BoundedResult:
    return run_bounded(
        [sys.executable, "-c", code],
        deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
        max_stdout_bytes=CAP,
        max_stderr_bytes=max_stderr,
        env={},
        cwd=str(tmp_path),
    )


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_recorded_process(pid_file: Path) -> None:
    """Remove a process the test let escape; it may already be gone."""
    if not pid_file.exists() or not pid_file.read_text(encoding="ascii").strip():
        return
    pid = _read_pid(pid_file)
    if _is_alive(pid):
        os.kill(pid, signal.SIGKILL)


def _wait_until_gone(pid: int) -> bool:
    give_up_at = time.monotonic() + REAP_POLL_LIMIT_S
    while time.monotonic() < give_up_at:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(REAP_POLL_INTERVAL_S)
    return False


def _read_pid(pid_file: Path) -> int:
    return int(pid_file.read_text(encoding="ascii").strip())


def test_run_bounded_when_the_process_exits_normally_should_return_its_output_and_status(tmp_path: Path) -> None:
    # Arrange
    script = "printf out; printf err >&2; exit 3"

    # Act
    result = _run_shell(script, tmp_path)

    # Assert
    assert result.returncode == 3
    assert result.stdout == b"out"
    assert result.stderr_tail == b"err"
    assert result.timed_out is False
    assert result.stdout_overflow is False


def test_run_bounded_when_the_deadline_passes_should_kill_the_whole_process_group(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "child.pid"
    script = 'sleep 60 & echo $! > "$1"; sleep 60'
    started = time.monotonic()

    # Act
    result = _run_shell(script, tmp_path, deadline_ms=SHORT_DEADLINE_MS, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert result.timed_out is True
    assert result.returncode is None
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_stdout_exceeds_its_cap_should_stop_at_the_cap_and_kill_the_process(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "yes.pid"
    script = 'echo $$ > "$1"; exec yes'

    # Act
    result = _run_shell(script, tmp_path, max_stdout=ONE_KIB, args=(str(pid_file),))

    # Assert
    assert result.stdout_overflow is True
    assert result.returncode is None
    assert result.timed_out is False
    assert len(result.stdout) == ONE_KIB
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_stdout_fills_the_cap_exactly_should_not_report_overflow(tmp_path: Path) -> None:
    # Arrange
    script = "printf 0123456789"

    # Act
    result = _run_shell(script, tmp_path, max_stdout=10)

    # Assert
    assert result.stdout == b"0123456789"
    assert result.stdout_overflow is False
    assert result.returncode == 0


def test_run_bounded_when_stderr_is_long_should_keep_only_its_tail(tmp_path: Path) -> None:
    # Arrange
    script = 'i=0; while [ $i -lt 100 ]; do printf "line%03d\\n" $i >&2; i=$((i+1)); done'

    # Act
    result = _run_shell(script, tmp_path, max_stderr=16)

    # Assert
    assert result.stderr_tail == b"line098\nline099\n"
    assert result.stderr_truncated is True
    assert result.returncode == 0


def test_run_bounded_when_stderr_exceeds_its_cap_should_report_it_truncated_and_keep_the_last_bytes(
    tmp_path: Path,
) -> None:
    # Arrange
    code = f"import sys; sys.stderr.write('a' * {ONE_KIB} + 'b' * {ONE_KIB})"

    # Act
    result = _run_python(code, tmp_path, max_stderr=ONE_KIB)

    # Assert
    assert result.stderr_truncated is True
    assert result.stderr_tail == b"b" * ONE_KIB


def test_run_bounded_when_stderr_fits_its_cap_should_not_report_it_truncated(tmp_path: Path) -> None:
    # Arrange
    code = f"import sys; sys.stderr.write('c' * {SMALL_STDERR_BYTES})"

    # Act
    result = _run_python(code, tmp_path, max_stderr=ONE_KIB)

    # Assert
    assert result.stderr_truncated is False
    assert result.stderr_tail == b"c" * SMALL_STDERR_BYTES


def test_run_bounded_when_the_process_exits_leaving_a_background_child_should_kill_the_child(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "background.pid"
    script = 'sleep 60 & echo $! > "$1"; exit 0'
    started = time.monotonic()

    # Act
    result = _run_shell(script, tmp_path, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert result.returncode == 0
    assert result.timed_out is False
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_a_grandchild_escapes_and_holds_the_pipes_should_still_return(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "escaped.pid"
    script = '"$1" -c "$2" "$3" & while [ ! -s "$3" ]; do sleep 0.05; done; exit 0'
    started = time.monotonic()

    # Act
    try:
        result = _run_shell(script, tmp_path, args=(sys.executable, ESCAPING_GRANDCHILD, str(pid_file)))
        elapsed = time.monotonic() - started
        escaped_pid = _read_pid(pid_file)
        escaped_was_alive = _is_alive(escaped_pid)
    finally:
        _kill_recorded_process(pid_file)

    # Assert
    assert escaped_was_alive is True
    assert result.returncode == 0
    assert result.timed_out is False
    assert elapsed < RETURN_WITHIN_S


def test_run_bounded_when_the_group_ignores_sigterm_should_escalate_to_sigkill(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "stubborn.pid"
    script = 'trap "" TERM; sleep 60 & echo $! > "$1"; sleep 60'
    started = time.monotonic()

    # Act
    result = _run_shell(script, tmp_path, deadline_ms=SHORT_DEADLINE_MS, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert result.timed_out is True
    assert result.returncode is None
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_a_finite_writer_overflows_stdout_should_report_overflow_without_status(
    tmp_path: Path,
) -> None:
    # Arrange
    script = f"head -c {ZERO_BYTES_WRITTEN} /dev/zero; exit 0"

    # Act
    result = _run_shell(script, tmp_path, max_stdout=ONE_KIB)

    # Assert
    assert result.stdout_overflow is True
    assert result.returncode is None
    assert len(result.stdout) == ONE_KIB


def _wait_until_written(path: Path) -> None:
    give_up_at = time.monotonic() + REAP_POLL_LIMIT_S
    while time.monotonic() < give_up_at and not (path.exists() and path.read_text(encoding="ascii").strip()):
        time.sleep(REAP_POLL_INTERVAL_S)


def _selector_factory_failing_on_call(failing_call: int, *, once_written: Path) -> Any:
    """A DefaultSelector stand-in whose n-th call fails as an exhausted descriptor table does.

    It fails only once the child has recorded its background process, so the test sees that
    process die with the group rather than never start.
    """
    real_factory = selectors.DefaultSelector
    calls = 0

    def factory() -> selectors.BaseSelector:
        nonlocal calls
        calls += 1
        if calls == failing_call:
            _wait_until_written(once_written)
            raise OSError(errno.EMFILE, "Too many open files")
        return real_factory()

    return factory


def test_run_bounded_when_a_reader_selector_cannot_be_created_should_raise_and_kill_the_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    pid_file = tmp_path / "background.pid"
    script = 'sleep 60 & echo $! > "$1"; sleep 60'
    monkeypatch.setattr(
        "core.bounded_process.selectors.DefaultSelector", _selector_factory_failing_on_call(2, once_written=pid_file)
    )
    started = time.monotonic()

    # Act
    with pytest.raises(OSError) as raised:
        _run_shell(script, tmp_path, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert raised.value.errno == errno.EMFILE
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_a_read_fails_should_raise_instead_of_reporting_a_normal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    pid_file = tmp_path / "child.pid"
    script = 'echo $$ > "$1"; echo started; exec sleep 60'
    real_read = os.read

    def read_failing_on_reader_threads(fd: int, length: int) -> bytes:
        # Popen itself reads its error pipe on the calling thread; only the readers fail.
        if threading.current_thread() is not threading.main_thread():
            raise OSError(errno.EIO, "I/O error")
        return real_read(fd, length)

    monkeypatch.setattr("core.bounded_process.os.read", read_failing_on_reader_threads)
    started = time.monotonic()

    # Act
    with pytest.raises(OSError) as raised:
        _run_shell(script, tmp_path, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert raised.value.errno == errno.EIO
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


@pytest.mark.parametrize(
    "bad_arguments",
    [
        {"env": [("PATH", "/bin")]},
        {"env": {"PATH": 1}},
        {"env": {1: "value"}},
        {"cwd": ""},
        {"deadline": 5.0},
    ],
)
def test_run_bounded_when_env_cwd_or_deadline_is_invalid_should_raise(
    bad_arguments: dict[str, Any], tmp_path: Path
) -> None:
    # Arrange
    arguments: dict[str, Any] = {
        "deadline": Deadline.after_ms(GENEROUS_DEADLINE_MS),
        "max_stdout_bytes": CAP,
        "max_stderr_bytes": CAP,
        "env": BASE_ENV,
        "cwd": str(tmp_path),
    }
    arguments.update(bad_arguments)

    # Act / Assert
    with pytest.raises(ValueError):
        run_bounded([SHELL, "-c", "true"], **arguments)


def test_run_bounded_when_given_an_environment_should_expose_exactly_that_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("PARENT_ONLY_SECRET", "must-not-leak")

    # Act
    result = run_bounded(
        ["/usr/bin/env"],
        deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
        max_stdout_bytes=CAP,
        max_stderr_bytes=CAP,
        env={"ONLY_VISIBLE": "yes"},
        cwd=str(tmp_path),
    )

    # Assert
    assert result.stdout == b"ONLY_VISIBLE=yes\n"


def test_run_bounded_when_the_deadline_has_already_passed_should_start_nothing(tmp_path: Path) -> None:
    # Arrange
    marker = tmp_path / "started"
    expired = Deadline(expires_at=time.monotonic() - 1)

    # Act
    result = run_bounded(
        [SHELL, "-c", 'touch "$1"', "sh", str(marker)],
        deadline=expired,
        max_stdout_bytes=CAP,
        max_stderr_bytes=CAP,
        env=BASE_ENV,
        cwd=str(tmp_path),
    )

    # Assert
    assert result.timed_out is True
    assert result.returncode is None
    assert (result.stdout, result.stderr_tail) == (b"", b"")
    assert not marker.exists()


@pytest.mark.parametrize("argv", [[], "/bin/sh", [SHELL, 1]])
def test_run_bounded_when_argv_is_not_a_non_empty_list_of_strings_should_raise(argv: object, tmp_path: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        run_bounded(
            argv,  # type: ignore[arg-type]
            deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
            max_stdout_bytes=CAP,
            max_stderr_bytes=CAP,
            env=BASE_ENV,
            cwd=str(tmp_path),
        )


@pytest.mark.parametrize(("max_stdout", "max_stderr"), [(0, CAP), (CAP, 0), (-1, CAP), (True, CAP)])
def test_run_bounded_when_a_cap_is_below_one_should_raise(max_stdout: int, max_stderr: int, tmp_path: Path) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        run_bounded(
            [SHELL, "-c", "true"],
            deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
            max_stdout_bytes=max_stdout,
            max_stderr_bytes=max_stderr,
            env=BASE_ENV,
            cwd=str(tmp_path),
        )


def test_deadline_within_ms_when_the_stage_budget_is_longer_should_keep_the_original_deadline() -> None:
    # Arrange
    deadline = Deadline.after_ms(100)

    # Act
    stage = deadline.within_ms(60_000)

    # Assert
    assert stage.expires_at == deadline.expires_at


def test_deadline_within_ms_when_the_stage_budget_is_shorter_should_shorten_the_deadline() -> None:
    # Arrange
    deadline = Deadline.after_ms(60_000)

    # Act
    stage = deadline.within_ms(100)

    # Assert
    assert stage.expires_at < deadline.expires_at


def test_deadline_remaining_s_when_expired_should_be_zero_not_negative() -> None:
    # Arrange
    deadline = Deadline(expires_at=time.monotonic() - 5)

    # Act
    remaining = deadline.remaining_s()

    # Assert
    assert remaining == 0.0
    assert deadline.expired() is True


@pytest.mark.parametrize("ms", [0, -5, True])
def test_deadline_after_ms_when_ms_is_below_one_should_raise(ms: int) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        Deadline.after_ms(ms)


MONITOR_INTERVAL_MS = 20
LONG_MONITOR_INTERVAL_MS = 60_000
SHORT_RUN_SCRIPT = "sleep 0.3; exit 0"


class _MonitorFailure(Exception):
    pass


def _pid_recorded(pid_file: Path) -> bool:
    return pid_file.exists() and bool(pid_file.read_text(encoding="ascii").strip())


def _run_monitored(
    script: str, tmp_path: Path, monitor: Any, *, interval_ms: int = MONITOR_INTERVAL_MS, args: tuple[str, ...] = ()
) -> BoundedResult:
    return run_bounded(
        [SHELL, "-c", script, "sh", *args],
        deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
        max_stdout_bytes=CAP,
        max_stderr_bytes=CAP,
        env=BASE_ENV,
        cwd=str(tmp_path),
        monitor=monitor,
        monitor_interval_ms=interval_ms,
    )


def test_run_bounded_when_the_monitor_asks_to_stop_should_kill_the_whole_process_group(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "child.pid"
    script = 'sleep 60 & echo $! > "$1"; sleep 60'
    started = time.monotonic()

    # Act
    result = _run_monitored(script, tmp_path, lambda: not _pid_recorded(pid_file), args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert result.stopped_by_monitor is True
    assert result.timed_out is False
    assert result.returncode is None
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_the_monitor_keeps_answering_true_should_let_the_process_exit_normally(
    tmp_path: Path,
) -> None:
    # Arrange
    calls: list[float] = []

    def monitor() -> bool:
        calls.append(time.monotonic())
        return True

    # Act
    result = _run_monitored(SHORT_RUN_SCRIPT, tmp_path, monitor)

    # Assert
    assert result.returncode == 0
    assert result.stopped_by_monitor is False
    assert result.timed_out is False
    assert calls


def test_run_bounded_when_the_interval_is_longer_than_the_run_should_never_call_the_monitor(tmp_path: Path) -> None:
    # Arrange
    calls: list[float] = []

    def monitor() -> bool:
        calls.append(time.monotonic())
        return False

    # Act
    result = _run_monitored(SHORT_RUN_SCRIPT, tmp_path, monitor, interval_ms=LONG_MONITOR_INTERVAL_MS)

    # Assert
    assert result.returncode == 0
    assert result.stopped_by_monitor is False
    assert calls == []


def test_run_bounded_when_no_monitor_is_given_should_report_no_monitor_stop(tmp_path: Path) -> None:
    # Arrange
    script = "exit 0"

    # Act
    result = _run_shell(script, tmp_path)

    # Assert
    assert result.returncode == 0
    assert result.stopped_by_monitor is False


def test_run_bounded_when_the_monitor_raises_should_propagate_and_kill_the_group(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "child.pid"
    script = 'sleep 60 & echo $! > "$1"; sleep 60'

    def monitor() -> bool:
        if _pid_recorded(pid_file):
            raise _MonitorFailure("the output could not be measured")
        return True

    started = time.monotonic()

    # Act
    with pytest.raises(_MonitorFailure):
        _run_monitored(script, tmp_path, monitor, args=(str(pid_file),))
    elapsed = time.monotonic() - started

    # Assert
    assert elapsed < RETURN_WITHIN_S
    assert _wait_until_gone(_read_pid(pid_file))


def test_run_bounded_when_the_monitor_returns_a_non_bool_should_raise_and_kill_the_group(tmp_path: Path) -> None:
    # Arrange
    pid_file = tmp_path / "child.pid"
    script = 'sleep 60 & echo $! > "$1"; sleep 60'

    def monitor() -> Any:
        return None if _pid_recorded(pid_file) else True

    # Act
    with pytest.raises(TypeError):
        _run_monitored(script, tmp_path, monitor, args=(str(pid_file),))

    # Assert
    assert _wait_until_gone(_read_pid(pid_file))


@pytest.mark.parametrize(
    ("monitor", "interval_ms"),
    [
        (lambda: True, None),
        (None, MONITOR_INTERVAL_MS),
        (lambda: True, 0),
        (lambda: True, -1),
        (lambda: True, True),
        (lambda: True, 1.5),
        ("not callable", MONITOR_INTERVAL_MS),
    ],
)
def test_run_bounded_when_the_monitor_arguments_are_invalid_should_raise(
    monitor: Any, interval_ms: Any, tmp_path: Path
) -> None:
    # Arrange
    marker = tmp_path / "started"

    # Act
    with pytest.raises(ValueError):
        _run_monitored('touch "$1"', tmp_path, monitor, interval_ms=interval_ms, args=(str(marker),))

    # Assert
    assert not marker.exists()


REFUSAL = re.compile(rb"exceeds specified max pixel count")
OTHER_REFUSAL = re.compile(rb"Frame size [0-9]{1,10}x[0-9]{1,10} exceeds limit")
SPLIT_REFUSAL_WRITER = (
    "import sys, time\n"
    "sys.stderr.buffer.write(b'Picture size 9x9 exceeds specified ma'); sys.stderr.buffer.flush()\n"
    "time.sleep(0.3)\n"
    "sys.stderr.buffer.write(b'x pixel count 4\\n'); sys.stderr.buffer.flush()\n"
)
EARLY_REFUSAL_WRITER = (
    "import sys\n"
    "sys.stderr.buffer.write(b'Frame size 128x128 exceeds limit 4096\\n')\n"
    "sys.stderr.buffer.write(b'w' * 200_000)\n"
)
NO_REFUSAL_WRITER = "import sys\nsys.stderr.buffer.write(b'Picture size 9x9 is fine\\n' * 100)\n"
PASSED_DESCRIPTOR_READER = (
    "import os, sys\n"
    "sys.stdout.write(os.read(int(sys.argv[1]), 64).decode())\n"
)
DESCRIPTOR_PROBE = (
    "import os, sys\n"
    "try:\n"
    "    os.fstat(int(sys.argv[1]))\n"
    "    sys.stdout.write('inherited')\n"
    "except OSError:\n"
    "    sys.stdout.write('closed')\n"
)
PIPE_MESSAGE = b"the key"


def _run_scanning(code: str, tmp_path: Path, *, max_stderr: int = CAP) -> BoundedResult:
    return run_bounded(
        [sys.executable, "-c", code],
        deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
        max_stdout_bytes=CAP,
        max_stderr_bytes=max_stderr,
        env={},
        cwd=str(tmp_path),
        stderr_patterns=(REFUSAL, OTHER_REFUSAL),
    )


def test_run_bounded_when_a_stderr_pattern_is_split_across_two_writes_should_still_find_it(tmp_path: Path) -> None:
    # Arrange
    code = SPLIT_REFUSAL_WRITER

    # Act
    result = _run_scanning(code, tmp_path)

    # Assert
    assert result.returncode == 0
    assert result.stderr_patterns_seen == frozenset({REFUSAL.pattern})


def test_run_bounded_when_a_stderr_pattern_is_far_before_the_kept_tail_should_still_report_it(
    tmp_path: Path,
) -> None:
    # Arrange
    code = EARLY_REFUSAL_WRITER

    # Act
    result = _run_scanning(code, tmp_path, max_stderr=ONE_KIB)

    # Assert
    assert result.stderr_truncated is True
    assert b"exceeds limit" not in result.stderr_tail
    assert result.stderr_patterns_seen == frozenset({OTHER_REFUSAL.pattern})


def test_run_bounded_when_no_stderr_pattern_matches_should_report_none(tmp_path: Path) -> None:
    # Arrange
    code = NO_REFUSAL_WRITER

    # Act
    result = _run_scanning(code, tmp_path)

    # Assert
    assert result.returncode == 0
    assert result.stderr_patterns_seen == frozenset()


@pytest.mark.parametrize("patterns", ["exceeds", [b"exceeds"], [re.compile("exceeds")], None])
def test_run_bounded_when_stderr_patterns_are_not_compiled_bytes_patterns_should_raise(
    patterns: Any, tmp_path: Path
) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        run_bounded(
            [SHELL, "-c", "true"],
            deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
            max_stdout_bytes=CAP,
            max_stderr_bytes=CAP,
            env=BASE_ENV,
            cwd=str(tmp_path),
            stderr_patterns=patterns,
        )


def _pipe_holding(message: bytes) -> tuple[int, int]:
    read_end, write_end = os.pipe()
    os.write(write_end, message)
    os.close(write_end)
    return read_end, write_end


def _run_with_descriptor(code: str, descriptor: int, tmp_path: Path, pass_fds: tuple[int, ...]) -> BoundedResult:
    return run_bounded(
        [sys.executable, "-c", code, str(descriptor)],
        deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
        max_stdout_bytes=CAP,
        max_stderr_bytes=CAP,
        env={},
        cwd=str(tmp_path),
        pass_fds=pass_fds,
    )


def test_run_bounded_when_a_descriptor_is_passed_should_let_the_child_read_it(tmp_path: Path) -> None:
    # Arrange
    read_end, _ = _pipe_holding(PIPE_MESSAGE)

    # Act
    try:
        result = _run_with_descriptor(PASSED_DESCRIPTOR_READER, read_end, tmp_path, (read_end,))
    finally:
        os.close(read_end)

    # Assert
    assert result.returncode == 0
    assert result.stdout == PIPE_MESSAGE


def test_run_bounded_when_a_descriptor_is_not_passed_should_not_let_the_child_inherit_it(tmp_path: Path) -> None:
    # Arrange
    read_end, _ = _pipe_holding(PIPE_MESSAGE)
    os.set_inheritable(read_end, True)

    # Act
    try:
        result = _run_with_descriptor(DESCRIPTOR_PROBE, read_end, tmp_path, ())
    finally:
        os.close(read_end)

    # Assert
    assert result.stdout == b"closed"


@pytest.mark.parametrize("pass_fds", ["3", [-1], [True], [1.5], None, 3])
def test_run_bounded_when_pass_fds_is_not_a_sequence_of_descriptors_should_raise(
    pass_fds: Any, tmp_path: Path
) -> None:
    # Arrange
    marker = tmp_path / "started"

    # Act
    with pytest.raises(ValueError):
        run_bounded(
            [SHELL, "-c", 'touch "$1"', "sh", str(marker)],
            deadline=Deadline.after_ms(GENEROUS_DEADLINE_MS),
            max_stdout_bytes=CAP,
            max_stderr_bytes=CAP,
            env=BASE_ENV,
            cwd=str(tmp_path),
            pass_fds=pass_fds,
        )

    # Assert
    assert not marker.exists()
