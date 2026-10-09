"""The preflight end to end: a range source, the loopback server, a real isolated ffprobe."""

from __future__ import annotations

import dataclasses
import errno
import logging
import selectors
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from core.admission import RequestLimits
from core.bounded_process import Deadline
from core.failure import Failure, WorkerFailure
from core.probe import LoopbackRangeServer, parse_range, run_preflight
from core.profile import PILOT_PROFILE, MediaProfile
from core.source_reader import BudgetedSource

GIB = 1024 ** 3
KIB = 1024
TEN_GIB = 10 * GIB
OPEN_LIMITS = RequestLimits(max_source_bytes=TEN_GIB, max_source_duration_ms=21_600_000)
TEST_DEADLINE_MS = 60_000

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")


class MemorySource:
    """A range source over bytes in memory, recording every read."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.reads: list[tuple[int, int]] = []

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    def read(self, start: int, length: int) -> bytes:
        self.reads.append((start, length))
        return self._data[start:start + length]


class FileSource(MemorySource):
    def __init__(self, path: Path) -> None:
        super().__init__(path.read_bytes())


class SparseSource:
    """A virtual object of `size_bytes`: real bytes first, zeros after them."""

    def __init__(self, prefix: bytes, size_bytes: int) -> None:
        self._prefix = prefix
        self._size = size_bytes
        self.reads: list[tuple[int, int]] = []

    @property
    def size_bytes(self) -> int:
        return self._size

    def read(self, start: int, length: int) -> bytes:
        self.reads.append((start, length))
        real = self._prefix[start:start + length]
        return real + bytes(length - len(real))


class ChangingSource(FileSource):
    """The object changes after its first read."""

    def read(self, start: int, length: int) -> bytes:
        if self.reads:
            raise WorkerFailure(Failure(
                error_class="source_integrity_mismatch", code="source_changed",
                retryable=False, detail="the object's ETag differs from the recorded one",
            ))
        return super().read(start, length)


class StallingSource(FileSource):
    """Every read waits until the test releases it."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.release = threading.Event()

    def read(self, start: int, length: int) -> bytes:
        self.release.wait(timeout=30)
        return super().read(start, length)


class Canary:
    """A loopback listener that must never be contacted."""

    def __init__(self) -> None:
        self.hits = 0
        canary = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                canary.hits += 1
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_CONNECT = do_GET

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def canary() -> Iterator[Canary]:
    listener = Canary()
    try:
        yield listener
    finally:
        listener.close()


def _profile(**probe_changes: int) -> MediaProfile:
    return dataclasses.replace(PILOT_PROFILE, probe=dataclasses.replace(PILOT_PROFILE.probe, **probe_changes))


def _preflight(source: Any, profile: MediaProfile = PILOT_PROFILE, limits: RequestLimits = OPEN_LIMITS) -> Any:
    return run_preflight(source, profile=profile, limits=limits, deadline=Deadline.after_ms(TEST_DEADLINE_MS))


def _failure_of(source: Any, profile: MediaProfile = PILOT_PROFILE, limits: RequestLimits = OPEN_LIMITS) -> Failure:
    with pytest.raises(WorkerFailure) as raised:
        _preflight(source, profile, limits)
    return raised.value.failure


def _refusal_codes(result: Any) -> list[str]:
    return [refusal.code for refusal in result.decision.refusals]


def test_preflight_when_the_source_is_an_ordinary_mp4_should_admit_it_with_its_facts(media):
    # Arrange
    source = FileSource(media.mp4_faststart)

    # Act
    result = _preflight(source)

    # Assert
    assert result.decision.admissible
    video = result.facts.video_streams[0]
    assert (video.display_width, video.display_height) == (320, 240)
    assert len(result.facts.audio_streams) == 1
    assert 2000 <= result.facts.duration_ms <= 2100
    assert result.read.bytes_read <= source.size_bytes
    assert result.read.requests == len(source.reads)


def test_preflight_when_the_mp4_index_is_at_the_end_should_reach_it_by_range(media):
    # Arrange
    source = FileSource(media.mp4_moov_at_end)
    profile = _profile(chunk_bytes=4 * KIB)

    # Act
    result = _preflight(source, profile)

    # Assert
    assert result.decision.admissible
    assert 2000 <= result.facts.duration_ms <= 2100
    assert max(start for start, _ in source.reads) > source.size_bytes // 2


def test_preflight_when_the_source_is_ten_gib_should_read_no_more_than_the_byte_budget(media):
    # Arrange
    source = SparseSource(media.mp4_faststart.read_bytes(), TEN_GIB)

    # Act
    result = _preflight(source)

    # Assert
    assert result.decision.admissible
    assert result.read.bytes_read <= PILOT_PROFILE.probe.max_bytes
    assert sum(length for _, length in source.reads) == result.read.bytes_read


def test_preflight_when_the_source_exceeds_the_size_limit_should_refuse_before_any_read(media):
    # Arrange
    source = SparseSource(media.mp4_faststart.read_bytes(), TEN_GIB + 1)

    # Act
    failure = _failure_of(source)

    # Assert
    assert (failure.error_class, failure.code) == ("input_limits_exceeded", "source_too_large")
    assert source.reads == []


def test_preflight_when_the_request_limit_is_below_the_profile_should_refuse_by_the_request(media):
    # Arrange
    source = FileSource(media.mp4_faststart)
    limits = RequestLimits(max_source_bytes=source.size_bytes - 1, max_source_duration_ms=1_000)

    # Act
    failure = _failure_of(source, limits=limits)

    # Assert
    assert failure.code == "source_too_large"
    assert source.reads == []


def test_preflight_when_the_duration_exceeds_the_request_limit_should_refuse_it(media):
    # Arrange
    source = FileSource(media.mp4_faststart)
    limits = RequestLimits(max_source_bytes=TEN_GIB, max_source_duration_ms=1_000)

    # Act
    result = _preflight(source, limits=limits)

    # Assert
    assert _refusal_codes(result) == ["duration_exceeded"]


def test_preflight_when_the_source_is_a_playlist_should_refuse_it_without_any_outbound_connection(tmp_path, canary):
    # Arrange
    base = f"http://127.0.0.1:{canary.port}"
    playlist = tmp_path / "index.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n"
        f'#EXT-X-KEY:METHOD=AES-128,URI="{base}/key"\n'
        f"#EXTINF:6.0,\n{base}/segment0.ts\n#EXT-X-ENDLIST\n",
    )
    source = FileSource(playlist)

    # Act
    failure = _failure_of(source)

    # Assert
    assert (failure.error_class, failure.code) == ("input_invalid", "unrecognized_media")
    assert canary.hits == 0


def test_preflight_when_a_proxy_is_configured_should_never_use_it(media, canary, monkeypatch):
    # Arrange
    for variable in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(variable, f"http://127.0.0.1:{canary.port}")
    source = FileSource(media.mp4_faststart)

    # Act
    result = _preflight(source)

    # Assert
    assert result.decision.admissible
    assert canary.hits == 0


@pytest.mark.parametrize("fixture_name", ["malformed", "truncated_mp4"])
def test_preflight_when_the_bytes_are_not_media_should_fail_as_unrecognized(media, fixture_name):
    # Arrange
    source = FileSource(getattr(media, fixture_name))

    # Act
    failure = _failure_of(source)

    # Assert
    assert (failure.error_class, failure.code) == ("input_invalid", "unrecognized_media")
    assert "127.0.0.1" not in failure.detail


def test_preflight_when_the_budget_cannot_reach_the_index_should_fail_closed(media):
    # Arrange
    source = FileSource(media.mp4_moov_at_end)
    profile = _profile(max_bytes=8 * KIB, chunk_bytes=4 * KIB, probesize_bytes=4 * KIB)

    # Act
    failure = _failure_of(source, profile)

    # Assert
    assert (failure.error_class, failure.code) == ("input_limits_exceeded", "probe_budget_exceeded")
    assert sum(length for _, length in source.reads) <= 8 * KIB


def test_preflight_when_the_source_changes_during_the_probe_should_fail_as_source_changed(media):
    # Arrange
    source = ChangingSource(media.mp4_moov_at_end)
    profile = _profile(chunk_bytes=4 * KIB)

    # Act
    failure = _failure_of(source, profile)

    # Assert
    assert (failure.error_class, failure.code) == ("source_integrity_mismatch", "source_changed")


def test_preflight_when_storage_stalls_should_end_at_its_deadline(media):
    # Arrange
    source = StallingSource(media.mp4_faststart)
    profile = _profile(max_wall_ms=1_500)
    started = time.monotonic()

    # Act
    try:
        failure = _failure_of(source, profile)
    finally:
        source.release.set()

    # Assert
    assert (failure.error_class, failure.code) == ("deadline_exceeded", "probe_deadline_exceeded")
    assert time.monotonic() - started < 10


@pytest.mark.parametrize(
    ("fixture_name", "expected_codes"),
    [
        ("subsecond", []),
        ("silent", []),
        ("portrait", []),
        ("rotated_90", []),
        ("anamorphic", []),
        ("mkv", []),
        ("mpegts", []),
        ("high_fps", ["frame_rate_exceeded"]),
        ("many_audio", ["too_many_audio_streams"]),
        ("interlaced", ["interlaced_unsupported"]),
        ("hdr_pq", ["hdr_unsupported"]),
    ],
)
def test_preflight_when_given_each_media_shape_should_decide_by_the_profile(media, fixture_name, expected_codes):
    # Arrange
    source = FileSource(getattr(media, fixture_name))

    # Act
    result = _preflight(source)

    # Assert
    assert _refusal_codes(result) == expected_codes


def test_preflight_when_the_source_is_rotated_should_report_its_display_geometry(media):
    # Arrange
    source = FileSource(media.rotated_90)

    # Act
    result = _preflight(source)

    # Assert
    video = result.facts.video_streams[0]
    assert (video.display_width, video.display_height) == (240, 320)


def test_preflight_when_the_source_is_subsecond_should_keep_its_fractional_duration(media):
    # Arrange
    source = FileSource(media.subsecond)

    # Act
    result = _preflight(source)

    # Assert
    assert 400 <= result.facts.duration_ms < 500


def test_preflight_when_the_source_is_webm_should_admit_it(media):
    # Arrange
    if media.webm is None:
        pytest.skip("this ffmpeg cannot write VP9/Opus")
    source = FileSource(media.webm)

    # Act
    result = _preflight(source)

    # Assert
    assert result.decision.admissible


def test_preflight_when_the_source_is_an_mp3_should_refuse_its_container(media):
    # Arrange
    source = FileSource(media.audio_with_cover)

    # Act
    failure = _failure_of(source)

    # Assert
    assert (failure.error_class, failure.code) == ("input_invalid", "unrecognized_media")


def test_preflight_when_the_only_picture_is_a_cover_should_refuse_it_as_no_video(tmp_path):
    # Arrange
    cover = tmp_path / "cover.png"
    m4a = tmp_path / "with-cover.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=1:duration=1",
         "-frames:v", "1", str(cover)],
        check=True, capture_output=True, timeout=60,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-i", str(cover),
         "-map", "0:a", "-map", "1:v", "-c:a", "aac", "-c:v", "png", "-disposition:v:0", "attached_pic",
         str(m4a)],
        check=True, capture_output=True, timeout=60,
    )
    source = FileSource(m4a)

    # Act
    result = _preflight(source)

    # Assert
    assert result.facts.attached_picture_count == 1
    assert _refusal_codes(result) == ["no_video_stream"]


def test_preflight_when_no_duration_is_recorded_should_refuse_it(tmp_path):
    # Arrange
    live_webm = tmp_path / "live.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=2",
         "-c:v", "libx264", "-f", "matroska", "-live", "1", str(live_webm)],
        check=True, capture_output=True, timeout=60,
    )
    source = FileSource(live_webm)

    # Act
    try:
        result = _preflight(source)
    except WorkerFailure as failure:
        codes = [failure.failure.code]
    else:
        codes = _refusal_codes(result)

    # Assert
    assert codes == ["duration_unknown"]


def _live_matroska_with_ac3(path: Path, *, noisy_tags: int) -> Path:
    """One video frame and 20 s of AC3 audio, with no recorded duration: ffprobe estimates it.

    Each noisy tag holds an invalid UTF-8 byte, so ffprobe's JSON writer warns about it after the
    duration-estimate warning, pushing that warning towards the front of stderr.
    """
    tags: list[bytes] = []
    for number in range(noisy_tags):
        tags += [b"-metadata", b"TAG%d=" % number + b"x" * 900 + b"\xff" + str(number).encode()]
    subprocess.run(
        [b"ffmpeg", b"-v", b"error", b"-y",
         b"-f", b"lavfi", b"-i", b"testsrc2=size=160x120:rate=25:duration=2",
         b"-f", b"lavfi", b"-i", b"sine=frequency=440:duration=20",
         b"-frames:v", b"1", b"-c:v", b"libx264", b"-c:a", b"ac3", *tags,
         b"-f", b"matroska", b"-live", b"1", str(path).encode()],
        check=True, capture_output=True, timeout=60,
    )
    return path


def test_preflight_when_the_duration_is_only_estimated_should_refuse_it(tmp_path):
    # Arrange
    source = FileSource(_live_matroska_with_ac3(tmp_path / "estimated.mkv", noisy_tags=0))

    # Act
    result = _preflight(source)

    # Assert
    assert result.facts.duration_estimated
    assert "duration_unknown" in _refusal_codes(result)


def test_preflight_when_warnings_overflow_the_diagnostics_cap_should_fail_closed(tmp_path):
    # Arrange
    source = FileSource(_live_matroska_with_ac3(tmp_path / "noisy.mkv", noisy_tags=12))
    profile = _profile(max_stderr_bytes=4 * KIB)

    # Act
    failure = _failure_of(source, profile)

    # Assert
    assert (failure.error_class, failure.code) == ("input_limits_exceeded", "probe_diagnostics_too_large")


def _stub_prober(tmp_path: Path, body: str) -> str:
    stub = tmp_path / "ffprobe-stub"
    stub.write_text("#!/bin/sh\n" + body + "\n")
    stub.chmod(0o755)
    return str(stub)


def _failure_with_prober(media: Any, prober: str) -> Failure:
    with pytest.raises(WorkerFailure) as raised:
        run_preflight(
            FileSource(media.mp4_faststart), profile=PILOT_PROFILE, limits=OPEN_LIMITS,
            deadline=Deadline.after_ms(TEST_DEADLINE_MS), ffprobe_path=prober,
        )
    return raised.value.failure


def test_preflight_when_the_prober_is_killed_by_a_signal_should_fail_retryable_not_as_bad_media(media, tmp_path):
    # Arrange
    prober = _stub_prober(tmp_path, "kill -9 $$")

    # Act
    failure = _failure_with_prober(media, prober)

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("internal_error", "prober_crashed", True)


def test_preflight_when_the_prober_cannot_run_should_fail_as_configuration(media, tmp_path):
    # Arrange
    prober = _stub_prober(tmp_path, "exit 127")

    # Act
    failure = _failure_with_prober(media, prober)

    # Assert
    assert (failure.error_class, failure.code) == ("configuration_error", "ffprobe_unavailable")


def test_preflight_when_the_prober_is_missing_should_fail_as_configuration(media, tmp_path):
    # Act
    failure = _failure_with_prober(media, str(tmp_path / "no-such-ffprobe"))

    # Assert
    assert (failure.error_class, failure.code) == ("configuration_error", "ffprobe_unavailable")


def test_preflight_when_the_prober_is_not_an_executable_format_should_fail_as_configuration(media, tmp_path):
    # Arrange
    prober = tmp_path / "ffprobe-not-a-binary"
    prober.write_bytes(b"\x7fELF-not-really")
    prober.chmod(0o755)

    # Act
    failure = _failure_with_prober(media, str(prober))

    # Assert
    assert (failure.error_class, failure.code) == ("configuration_error", "ffprobe_unavailable")
    assert failure.retryable is False
    assert failure.detail == "ffprobe could not be started"


@pytest.mark.parametrize(
    ("start_error", "errno_name"),
    [
        (BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable"), "EAGAIN"),
        (OSError(errno.EMFILE, "Too many open files"), "EMFILE"),
    ],
)
def test_preflight_when_the_prober_cannot_start_for_lack_of_resources_should_fail_retryable(
    media, monkeypatch, caplog, start_error, errno_name
):
    # Arrange
    def refuse_to_start(*args: Any, **kwargs: Any) -> None:
        raise start_error

    monkeypatch.setattr("core.bounded_process.subprocess.Popen", refuse_to_start)
    caplog.set_level(logging.WARNING, logger="core.probe")

    # Act
    failure = _failure_with_prober(media, shutil.which("ffprobe"))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("resource_exhausted", "prober_start_failed", True)
    assert failure.detail == f"ffprobe could not be started: {errno_name}"
    assert [record.getMessage() for record in caplog.records if record.name == "core.probe"] == [
        f"ffprobe could not be started: {type(start_error).__name__} errno={start_error.errno}"
    ]


def _raising(error: BaseException) -> Any:
    """A stand-in for a constructor or function that fails the way an exhausted worker does."""
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    return fail


@pytest.mark.parametrize(
    ("target", "setup_error", "errno_name"),
    [
        pytest.param(
            "core.probe.tempfile.TemporaryDirectory", OSError(errno.EMFILE, "Too many open files"), "EMFILE",
            id="temp-dir-emfile",
        ),
        pytest.param(
            "core.probe._LoopbackHTTPServer", OSError(errno.EMFILE, "Too many open files"), "EMFILE",
            id="loopback-socket-emfile",
        ),
        pytest.param(
            "core.probe.tempfile.TemporaryDirectory",
            FileNotFoundError(errno.ENOENT, "No usable temporary directory found"),
            "ENOENT",
            id="mkdtemp-exhaustion-reported-as-not-found",
        ),
    ],
)
def test_preflight_when_its_setup_runs_out_of_resources_should_fail_retryable_not_as_configuration(
    media, monkeypatch, caplog, target, setup_error, errno_name
):
    # Arrange
    monkeypatch.setattr(target, _raising(setup_error))
    caplog.set_level(logging.WARNING, logger="core.probe")

    # Act
    failure = _failure_of(FileSource(media.mp4_faststart))

    # Assert
    assert (failure.error_class, failure.code) == ("resource_exhausted", "preflight_setup_failed")
    assert failure.retryable is True
    assert failure.detail == f"the preflight could not be set up: {errno_name}"
    assert [record.getMessage() for record in caplog.records if record.name == "core.probe"] == [
        f"preflight setup failed: {type(setup_error).__name__} errno={setup_error.errno}"
    ]


def test_preflight_when_a_loopback_handler_thread_cannot_start_should_fail_retryable_not_as_bad_media(
    media, monkeypatch
):
    # Arrange
    def refuse_thread(self: Any, request: Any, client_address: Any) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("core.probe._LoopbackHTTPServer.process_request", refuse_thread)

    # Act
    failure = _failure_of(FileSource(media.mp4_faststart))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("internal_error", "loopback_server_failed", True)


def test_preflight_when_a_reader_thread_cannot_start_should_fail_retryable(media, monkeypatch, caplog):
    # Arrange
    monkeypatch.setattr("core.bounded_process._start_reader", _raising(RuntimeError("can't start new thread")))
    caplog.set_level(logging.WARNING, logger="core.probe")

    # Act
    failure = _failure_of(FileSource(media.mp4_faststart))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("resource_exhausted", "prober_start_failed", True)
    assert failure.detail == "ffprobe could not be started: unknown"
    assert [record.getMessage() for record in caplog.records if record.name == "core.probe"] == [
        "ffprobe could not be started: RuntimeError errno=None"
    ]


def test_preflight_when_a_reader_selector_cannot_be_created_should_fail_retryable_not_as_bad_media(
    media, monkeypatch
):
    # Arrange
    real_factory = selectors.DefaultSelector
    calls = 0

    def factory_failing_on_second_call() -> selectors.BaseSelector:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_factory()

    monkeypatch.setattr("core.bounded_process.selectors.DefaultSelector", factory_failing_on_second_call)

    # Act
    failure = _failure_of(FileSource(media.mp4_faststart))

    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("resource_exhausted", "prober_start_failed", True)
    assert failure.detail == "ffprobe could not be started: EMFILE"


# The loopback server on its own.

@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, (0, 99)),
        ("bytes=0-", (0, 99)),
        ("bytes=10-19", (10, 19)),
        ("bytes=90-500", (90, 99)),
        ("bytes=-10", (90, 99)),
        ("bytes=-500", (0, 99)),
        ("bytes=100-", None),
        ("bytes=20-10", None),
        ("bytes=0-1,5-6", None),
        ("bytes=-0", None),
        ("bytes=-", None),
        ("items=0-1", None),
    ],
)
def test_parse_range_when_given_a_range_header_should_satisfy_only_single_ranges_inside_the_object(header, expected):
    # Act / Assert
    assert parse_range(header, 100) == expected


def _serve(data: bytes) -> tuple[LoopbackRangeServer, MemorySource]:
    inner = MemorySource(data)
    budgeted = BudgetedSource(inner, max_bytes=len(data) * 4, max_requests=100, deadline=Deadline.after_ms(10_000))
    return LoopbackRangeServer(budgeted, chunk_bytes=16), inner


def test_loopback_server_when_asked_another_path_should_answer_not_found():
    # Arrange
    server, _ = _serve(b"x" * 100)

    # Act
    with server:
        wrong = server.url.rsplit("/", 1)[0] + "/guess"
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(wrong, timeout=5)

    # Assert
    assert raised.value.code == 404


def test_loopback_server_when_ranges_overlap_should_fetch_each_chunk_once():
    # Arrange
    data = bytes(range(100))
    server, inner = _serve(data)

    # Act
    with server:
        first = urllib.request.urlopen(
            urllib.request.Request(server.url, headers={"Range": "bytes=10-40"}), timeout=5)
        body_first = first.read()
        again = urllib.request.urlopen(
            urllib.request.Request(server.url, headers={"Range": "bytes=20-30"}), timeout=5)
        body_again = again.read()

    # Assert
    assert first.status == 206
    assert first.headers["Content-Range"] == "bytes 10-40/100"
    assert body_first == data[10:41]
    assert body_again == data[20:31]
    assert inner.reads == [(0, 16), (16, 16), (32, 16)]


def test_loopback_server_when_the_range_starts_beyond_the_object_should_refuse_it():
    # Arrange
    server, inner = _serve(b"x" * 100)

    # Act
    with server:
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(
                urllib.request.Request(server.url, headers={"Range": "bytes=100-"}), timeout=5)

    # Assert
    assert raised.value.code == 416
    assert inner.reads == []
