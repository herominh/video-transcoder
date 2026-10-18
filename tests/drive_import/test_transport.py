"""The real session (RequestsSession) against a server of the test's own on the loopback address: it follows no
redirect and reads no redirect's body, reads no proxy from the environment, never reads a PUT's answer body, sends a
BoundedBody whole with its length or stops it at the stop line, and its watchdog ends at the stop line a request
that a trickling or silent server would hold far longer, a TLS handshake whose server never answers included, and
lets no connection whose name resolution or TCP connect straddled the stop line go on, so that the job answers
`deadline` within a moment of it. The server speaks plain HTTP and, with a throwaway self-signed certificate for
127.0.0.1 made by the test (`cryptography`) and trusted by the session, HTTPS, the scheme the job uses. Nothing leaves
the host."""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import socket
import socketserver
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from core.drive_import import google, store
from core.drive_import.clock import DeadlineReached, StopLine, SystemClock
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.job import run_import
from core.drive_import.transport import TRANSFER_ERRORS, BoundedBody, RequestsSession, read_small_answer
from tests.drive_import.fakes import MIB, DriveFile, FakeClock, import_input, settings

TIMEOUT = (5.0, 5.0)
# A read time-out far longer than any test waits: only the watchdog can end a read before it.
LONG_TIMEOUT = (5.0, 60.0)
PAYLOAD = bytes(range(256)) * 400  # 100 KiB, in more blocks than one
ETAG = '"loopback-etag"'
DEAD_PROXY = "http://127.0.0.1:9"
# How far the stop line of a watchdog test lies, and how soon after it the request must have ended.
STOP_LINE_S = 1.0
ENDED_WITHIN_S = 2.5
# How long a holding or trickling server keeps a connection at most, so that no test can hang the suite.
SERVER_HOLDS_AT_MOST_S = 15.0
TRICKLE_EVERY_S = 0.1
DECLARED_LENGTH = 1_000_000
# How a trickling answer's body is framed (the loopback server's `_trickle()`).
KEPT_ALIVE = "kept-alive"
CONNECTION_CLOSE = "connection-close"
HTTP_1_0 = "http-1.0"
CLOSE_DELIMITED = "close-delimited"
# The framings in which http.client hands the connection's socket over to the answer.
HANDED_OVER = [
    pytest.param(CONNECTION_CLOSE, id="connection-close"),
    pytest.param(HTTP_1_0, id="http-1.0"),
    pytest.param(CLOSE_DELIMITED, id="close-delimited"),
]
ONE = DriveFile(MIB)


@dataclass
class _Seen:
    """What the loopback server was asked: each request's method, path and headers, and the bodies it read."""

    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    bodies: list[bytes] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)


def _handler_for(seen: _Seen) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the standard library's name
            return

        def _answer(self, status: int, headers: dict[str, str], body: bytes = b"") -> None:
            self.send_response(status)
            for name, value in {"Content-Length": str(len(body)), **headers}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _hold(self) -> None:
            """Send nothing more, until the test ends or the client goes."""
            seen.stop.wait(SERVER_HOLDS_AT_MOST_S)

        def _trickle(self, framing: str = KEPT_ALIVE) -> None:
            """A head, then one byte of the body at a time, each well within any time-out. The body is framed by a
            declared length on a connection kept alive, by a declared length on a connection the answer ends
            (`Connection: close`, or an HTTP/1.0 answer), or by the close itself (no length): in the last three
            http.client hands the connection's socket over to the answer."""
            if framing == HTTP_1_0:
                self.protocol_version = "HTTP/1.0"
            self.send_response(200)
            if framing != CLOSE_DELIMITED:
                self.send_header("Content-Length", str(DECLARED_LENGTH))
            if framing in (CONNECTION_CLOSE, CLOSE_DELIMITED):
                self.send_header("Connection", "close")
            self.end_headers()
            started = time.monotonic()
            try:
                while not seen.stop.wait(TRICKLE_EVERY_S) and time.monotonic() - started < SERVER_HOLDS_AT_MOST_S:
                    self.wfile.write(b"{")
            except OSError:
                # The client shut the connection: what the test waits for.
                return

        def do_GET(self) -> None:  # noqa: N802 - the standard library's name
            seen.requests.append(("GET", self.path, dict(self.headers)))
            path, query = urlsplit(self.path).path, urlsplit(self.path).query
            if path == "/redirect":
                self._answer(302, {"Location": f"http://127.0.0.1:{self.server.server_port}/target"})
            elif path == "/redirect-with-a-long-body":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/target")
                self.send_header("Content-Length", str(DECLARED_LENGTH))
                self.end_headers()
                self._hold()
            elif path == "/trickle":
                self._trickle(query.removeprefix("framing=") or KEPT_ALIVE)
            elif path.startswith("/drive/v3/files/"):
                self._drive(query)
            else:
                self._answer(200, {}, b"the target")

        def _drive(self, query: str) -> None:
            behaviour = self.server.drive_behaviour  # type: ignore[attr-defined]
            framing = self.server.framing  # type: ignore[attr-defined]
            downloading = "alt=media" in query
            if behaviour == ("media-trickles" if downloading else "metadata-trickles"):
                self._trickle(framing)
            elif downloading:
                self._answer(200, {}, ONE.bytes())
            else:
                self._answer(200, {"Content-Type": "application/json"}, json.dumps(ONE.metadata()).encode("utf-8"))

        def do_PUT(self) -> None:  # noqa: N802 - the standard library's name
            seen.requests.append(("PUT", self.path, dict(self.headers)))
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            seen.bodies.append(body)
            if len(body) < length:
                return
            if self.path == "/redirect":
                self._answer(307, {"Location": f"http://127.0.0.1:{self.server.server_port}/target"})
            elif self.path == "/hold":
                self._hold()
            elif self.path == "/a-long-answer":
                self.send_response(200)
                self.send_header("ETag", ETAG)
                self.send_header("Content-Length", str(DECLARED_LENGTH))
                self.end_headers()
                self._hold()
            else:
                self._answer(200, {"ETag": ETAG})

    return Handler


@dataclass(frozen=True)
class _Tls:
    """A throwaway self-signed certificate for 127.0.0.1, its key, and a server context of them."""

    certificate: Path
    key: Path

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, self.key)
        return context


@pytest.fixture(scope="module")
def tls(tmp_path_factory: pytest.TempPathFactory) -> _Tls:
    directory = tmp_path_factory.mktemp("tls")
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = dt.datetime.now(dt.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=True, crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    tls_files = _Tls(directory / "certificate.pem", directory / "key.pem")
    tls_files.certificate.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    tls_files.key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return tls_files


class _LoopbackServer(ThreadingHTTPServer):
    """The standard library's server without its look-up of the host's name (HTTPServer.server_bind asks
    socket.getfqdn(), which can wait on name resolution for half a minute), speaking TLS when it has a context: each
    connection's handshake in its own thread, after `handshake_delay` seconds of silence."""

    drive_behaviour = "answers"
    framing = "kept-alive"
    tls_context: ssl.SSLContext | None = None
    handshake_delay = 0.0
    stop: threading.Event | None = None

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def finish_request(self, request: Any, client_address: Any) -> None:
        if self.tls_context is None:
            super().finish_request(request, client_address)
            return
        if self.handshake_delay and self.stop is not None and self.stop.wait(self.handshake_delay):
            return
        try:
            secured = self.tls_context.wrap_socket(request, server_side=True)
        except OSError:
            # The client went before the handshake ended: what a test of a held handshake waits for.
            return
        try:
            super().finish_request(secured, client_address)
        finally:
            secured.close()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A connection the client shut is what most of these tests make: nothing to report.
        return


@contextmanager
def _loopback_server(
    drive_behaviour: str = "answers",
    tls: _Tls | None = None,
    handshake_delay: float = 0.0,
    framing: str = "kept-alive",
) -> Iterator[tuple[str, _Seen]]:
    seen = _Seen()
    server = _LoopbackServer(("127.0.0.1", 0), _handler_for(seen))
    server.drive_behaviour = drive_behaviour
    server.framing = framing
    server.tls_context = tls.server_context() if tls is not None else None
    server.handshake_delay = handshake_delay
    server.stop = seen.stop
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    scheme = "https" if tls is not None else "http"
    try:
        yield f"{scheme}://127.0.0.1:{server.server_port}", seen
    finally:
        seen.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def session(tls: _Tls) -> Iterator[RequestsSession]:
    made = RequestsSession(ca_bundle=str(tls.certificate))
    try:
        yield made
    finally:
        made.close()


@pytest.fixture(params=["http", "https"])
def server_tls(request: pytest.FixtureRequest, tls: _Tls) -> _Tls | None:
    """The TLS of a test's server: none (HTTP) or the throwaway certificate (HTTPS), one run of the test each."""
    return tls if request.param == "https" else None


class _TickingClock(FakeClock):
    """A clock that moves a second each time it is read: a stop line on it passes after so many reads."""

    def monotonic(self) -> float:
        self.advance(1.0)
        return super().monotonic()


def _far_stop_line() -> StopLine:
    clock = SystemClock()
    return StopLine(clock, clock.monotonic() + 3600.0)


def _near_stop_line() -> StopLine:
    clock = SystemClock()
    return StopLine(clock, clock.monotonic() + STOP_LINE_S)


def _ended_with_a_transfer_error(work: Callable[[], object]) -> tuple[bool, float]:
    """Whether `work` raised a transfer error, and the seconds it took."""
    started = time.monotonic()
    failed = False
    try:
        work()
    except TRANSFER_ERRORS:
        failed = True
    return failed, time.monotonic() - started


# --- no redirect, no body read that the job did not ask for -----------------------------------------------------------


def test_get_when_the_server_answers_a_redirect_should_return_it_without_following(session):
    # Act
    with _loopback_server() as (base, seen):
        response = session.get(base + "/redirect", {}, TIMEOUT, True, stop_line=_far_stop_line())
        response.close()

    # Assert
    assert response.status_code == 302
    assert [path for _, path, _ in seen.requests] == ["/redirect"]


def test_get_when_a_redirect_declares_a_body_that_never_comes_should_return_at_once_without_reading_it(session):
    # Arrange: requests reads the body of a redirect it does not follow, unless the session sees no redirect

    # Act
    with _loopback_server() as (base, seen):
        started = time.monotonic()
        response = session.get(base + "/redirect-with-a-long-body", {}, TIMEOUT, True, stop_line=_far_stop_line())
        elapsed = time.monotonic() - started
        response.close()

    # Assert
    assert response.status_code == 302
    assert elapsed < 1.0
    assert [path for _, path, _ in seen.requests] == ["/redirect-with-a-long-body"]


def test_put_when_the_server_answers_a_redirect_should_return_it_without_following(session):
    # Act
    with _loopback_server() as (base, seen):
        response = session.put(
            base + "/redirect",
            BoundedBody(PAYLOAD, _far_stop_line()),
            {"Content-Length": str(len(PAYLOAD))},
            TIMEOUT,
            stop_line=_far_stop_line(),
        )
        response.close()

    # Assert
    assert response.status_code == 307
    assert [path for _, path, _ in seen.requests] == ["/redirect"]


def test_put_when_its_answer_declares_a_body_that_never_comes_should_return_its_head_at_once(session):
    # Act
    with _loopback_server() as (base, _):
        started = time.monotonic()
        response = session.put(
            base + "/a-long-answer",
            BoundedBody(PAYLOAD, _far_stop_line()),
            {"Content-Length": str(len(PAYLOAD))},
            TIMEOUT,
            stop_line=_far_stop_line(),
        )
        elapsed = time.monotonic() - started
        response.close()

    # Assert: the status and the ETag are there, the body was never waited for
    assert response.status_code == 200 and response.headers["ETag"] == ETAG
    assert elapsed < 1.0


def test_put_when_given_a_bounded_body_should_send_it_whole_with_its_length(session):
    # Act
    with _loopback_server() as (base, seen):
        response = session.put(
            base + "/part",
            BoundedBody(PAYLOAD, _far_stop_line()),
            {"Content-Length": str(len(PAYLOAD))},
            TIMEOUT,
            stop_line=_far_stop_line(),
        )
        response.close()

    # Assert
    ((_, _, headers),) = seen.requests
    assert response.status_code == 200 and response.headers["etag"] == ETAG
    assert seen.bodies == [PAYLOAD]
    assert headers["Content-Length"] == str(len(PAYLOAD))
    assert "Transfer-Encoding" not in headers


def test_put_when_the_stop_line_passes_while_the_body_is_sent_should_raise_deadline_reached_and_stop_it(session):
    # Arrange: the body's stop line passes after a few blocks
    clock = _TickingClock()
    stop_line = StopLine(clock, clock.monotonic() + 3.0)

    # Act
    with _loopback_server() as (base, seen):
        with pytest.raises(DeadlineReached) as raised:
            session.put(
                base + "/part",
                BoundedBody(PAYLOAD, stop_line),
                {"Content-Length": str(len(PAYLOAD))},
                TIMEOUT,
                stop_line=_far_stop_line(),
            )

    # Assert: the error is the body's own, not one of the HTTP library's, and less than the body went out
    assert type(raised.value) is DeadlineReached
    assert [len(body) < len(PAYLOAD) for body in seen.bodies] in ([True], [])


def test_session_when_the_environment_names_a_proxy_should_not_use_it(monkeypatch, session):
    # Arrange: a proxy on a port nobody listens on, for every scheme
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, DEAD_PROXY)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)

    # Act
    with _loopback_server() as (base, seen):
        response = session.get(base + "/target", {}, TIMEOUT, True, stop_line=_far_stop_line())
        body = read_small_answer(response)
        response.close()

    # Assert: the server itself was asked
    assert response.status_code == 200 and body == b"the target"
    assert [path for _, path, _ in seen.requests] == ["/target"]


# --- the watchdog at the stop line ---------------------------------------------------------------------------------

# Seconds a name resolution or a TLS handshake a test holds takes: past the stop line, well before any time-out.
HELD_S = 1.5
HELD_HANDSHAKE_S = 5.0
# A connect time-out far longer than any test waits, for a handshake only the watchdog can end.
LONG_CONNECT_TIMEOUT = (30.0, 60.0)


def _ended(work: Callable[[], object]) -> tuple[BaseException | None, float]:
    """What `work` raised (a transfer error or the session's refusal at the stop line), and the seconds it took."""
    started = time.monotonic()
    raised: BaseException | None = None
    try:
        work()
    except (DeadlineReached, *TRANSFER_ERRORS) as error:
        raised = error
    return raised, time.monotonic() - started


def test_watchdog_when_a_download_trickles_past_the_stop_line_should_end_its_read_at_the_stop_line(
    session, server_tls
):
    # Arrange: a body of a byte every tenth of a second, so that no read time-out ever passes
    with _loopback_server(tls=server_tls) as (base, _):
        response = session.get(base + "/trickle", {}, LONG_TIMEOUT, True, stop_line=_near_stop_line())

        # Act
        failed, elapsed = _ended_with_a_transfer_error(lambda: list(response.iter_content(MIB)))
        response.close()

    # Assert
    assert failed
    assert STOP_LINE_S * 0.9 <= elapsed < ENDED_WITHIN_S


def test_watchdog_when_a_small_answer_trickles_past_the_stop_line_should_end_its_read_at_the_stop_line(
    session, server_tls
):
    # Arrange
    with _loopback_server(tls=server_tls) as (base, _):
        started = time.monotonic()
        response = session.get(base + "/trickle", {}, LONG_TIMEOUT, True, stop_line=_near_stop_line())

        # Act
        body = read_small_answer(response)
        elapsed = time.monotonic() - started
        response.close()

    # Assert: the read failed (no body), at the stop line
    assert body is None
    assert elapsed < ENDED_WITHIN_S


def test_watchdog_when_a_puts_answer_never_comes_should_end_the_put_at_the_stop_line(session, server_tls):
    # Act
    with _loopback_server(tls=server_tls) as (base, _):
        failed, elapsed = _ended_with_a_transfer_error(
            lambda: session.put(
                base + "/hold",
                BoundedBody(PAYLOAD, _far_stop_line()),
                {"Content-Length": str(len(PAYLOAD))},
                LONG_TIMEOUT,
                stop_line=_near_stop_line(),
            )
        )

    # Assert
    assert failed
    assert elapsed < ENDED_WITHIN_S


def test_store_put_when_the_stores_answer_never_comes_should_end_deadline_at_the_stop_line(session, server_tls):
    # Arrange: the stop line on the process's clock, a second away
    stop_line = _near_stop_line()

    # Act
    with _loopback_server(tls=server_tls) as (base, _):
        started = time.monotonic()
        with pytest.raises(ImportFailed) as raised:
            store.put(session, base + "/hold", PAYLOAD, {}, stop_line, Stage.UPLOAD, lambda attempt: None)
        elapsed = time.monotonic() - started

    # Assert
    assert raised.value.code is Code.DEADLINE
    assert elapsed < ENDED_WITHIN_S


def test_watchdog_when_a_tls_handshake_is_held_past_the_stop_line_should_end_it_at_the_stop_line(session, tls):
    # Arrange: a server that says nothing for five seconds after the TCP connect, under a connect time-out of 30 s

    # Act
    with _loopback_server(tls=tls, handshake_delay=HELD_HANDSHAKE_S) as (base, _):
        raised, elapsed = _ended(
            lambda: session.get(base + "/target", {}, LONG_CONNECT_TIMEOUT, True, stop_line=_near_stop_line())
        )

    # Assert: the handshake ended with the stop line, not with the server's silence
    assert raised is not None
    assert elapsed < ENDED_WITHIN_S


def _resolving_slowly(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Every name resolution of the process takes `seconds` before it answers."""
    resolve = socket.getaddrinfo

    def slow(*arguments: Any, **named: Any) -> Any:
        time.sleep(seconds)
        return resolve(*arguments, **named)

    monkeypatch.setattr(socket, "getaddrinfo", slow)


def test_watchdog_when_a_name_resolution_straddles_the_stop_line_should_let_the_connection_go_no_further(
    monkeypatch, session, server_tls
):
    # Arrange: the resolution ends half a second after the stop line; the server would answer at once
    with _loopback_server(tls=server_tls) as (base, seen):
        _resolving_slowly(monkeypatch, HELD_S)

        # Act
        raised, elapsed = _ended(lambda: session.get(base + "/target", {}, TIMEOUT, True, stop_line=_near_stop_line()))

    # Assert: the session refused to go on with the connection, and the server was asked nothing
    assert isinstance(raised, DeadlineReached)
    assert elapsed < HELD_S + 1.0
    assert seen.requests == []


def test_session_when_its_stop_line_has_passed_should_refuse_a_request_before_resolving_its_name(
    monkeypatch, session
):
    # Arrange: the watchdog has fired; from now on every name resolution takes 1.5 s, so only a refusal made before
    # the request reaches its connection can be quick
    with _loopback_server() as (base, seen):
        session.get(base + "/target", {}, TIMEOUT, True, stop_line=_near_stop_line()).close()
        time.sleep(STOP_LINE_S + 0.2)
        _resolving_slowly(monkeypatch, HELD_S)

        # Act: to another host's name, so that no connection of the pool could carry it
        raised, elapsed = _ended(
            lambda: session.get(base.replace("127.0.0.1", "localhost") + "/target", {}, TIMEOUT, True,
                                stop_line=_far_stop_line())
        )

    # Assert
    assert isinstance(raised, DeadlineReached)
    assert elapsed < 0.5
    assert len(seen.requests) == 1


def _drive_on_loopback(monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    monkeypatch.setattr(google, "FILES_URL", base + "/drive/v3/files/")


def _input_whose_stop_line_is_near(clock: SystemClock) -> dict[str, Any]:
    # The writer deadline 121 s away, so the stop line 1 s away.
    return import_input(ONE, writes_end_at=clock.now() + timedelta(seconds=120 + STOP_LINE_S))


DEADLINE_OUTPUT = {"contract": "import/v1", "status": "failed", "code": "deadline"}


@pytest.mark.parametrize(
    "behaviour",
    [pytest.param("metadata-trickles", id="the-metadata-trickles"), pytest.param("media-trickles", id="the-download")],
)
def test_run_import_when_google_trickles_past_the_stop_line_should_answer_deadline_at_the_stop_line(
    monkeypatch, session, server_tls, behaviour
):
    # Arrange: Google stands on the loopback server
    clock = SystemClock()
    document = _input_whose_stop_line_is_near(clock)

    # Act
    with _loopback_server(behaviour, tls=server_tls) as (base, seen):
        _drive_on_loopback(monkeypatch, base)
        started = time.monotonic()
        output = run_import(document, settings(), session, clock)
        elapsed = time.monotonic() - started

    # Assert: the job ended with the stop line, and wrote nothing
    assert output == DEADLINE_OUTPUT
    assert elapsed < ENDED_WITHIN_S
    assert [method for method, _, _ in seen.requests if method == "PUT"] == []


def test_run_import_when_the_name_resolution_straddles_the_stop_line_should_answer_deadline(
    monkeypatch, session, server_tls
):
    # Arrange
    clock = SystemClock()
    document = _input_whose_stop_line_is_near(clock)

    # Act
    with _loopback_server(tls=server_tls) as (base, seen):
        _drive_on_loopback(monkeypatch, base)
        _resolving_slowly(monkeypatch, HELD_S)
        started = time.monotonic()
        output = run_import(document, settings(), session, clock)
        elapsed = time.monotonic() - started

    # Assert: no request reached Google's stand-in
    assert output == DEADLINE_OUTPUT
    assert elapsed < HELD_S + 1.0
    assert seen.requests == []


def test_run_import_when_googles_tls_handshake_is_held_past_the_stop_line_should_answer_deadline(
    monkeypatch, session, tls
):
    # Arrange
    clock = SystemClock()
    document = _input_whose_stop_line_is_near(clock)

    # Act
    with _loopback_server(tls=tls, handshake_delay=HELD_HANDSHAKE_S) as (base, seen):
        _drive_on_loopback(monkeypatch, base)
        started = time.monotonic()
        output = run_import(document, settings(), session, clock)
        elapsed = time.monotonic() - started

    # Assert
    assert output == DEADLINE_OUTPUT
    assert elapsed < ENDED_WITHIN_S
    assert seen.requests == []


# --- answers that end their connection: http.client hands the socket over to them -----------------------------------


def _read_all(response: Any) -> tuple[int, float]:
    """The bytes read of `response` before it ended or failed, and the seconds that took."""
    started = time.monotonic()
    read = 0
    try:
        for chunk in response.iter_content(MIB):
            read += len(chunk)
    except TRANSFER_ERRORS:
        # The watchdog shut the socket: the body is cut there.
        pass
    return read, time.monotonic() - started


@pytest.mark.parametrize("framing", HANDED_OVER)
def test_watchdog_when_an_answer_that_ends_its_connection_trickles_should_end_its_read_at_the_stop_line(
    session, server_tls, framing
):
    # Arrange: a body of a byte every tenth of a second; http.client closes the connection object once the head is
    # read and hands its socket over to the answer
    with _loopback_server(tls=server_tls) as (base, _):
        response = session.get(
            base + "/trickle?framing=" + framing, {}, LONG_TIMEOUT, True, stop_line=_near_stop_line()
        )

        # Act
        read, elapsed = _read_all(response)
        response.close()

    # Assert: the read ended at the stop line, short of the body (a body framed by its length is cut with an error,
    # one framed by the close ends where the shutdown closed it)
    assert elapsed < ENDED_WITHIN_S
    assert read < DECLARED_LENGTH


@pytest.mark.parametrize("framing", HANDED_OVER)
@pytest.mark.parametrize(
    "behaviour",
    [pytest.param("metadata-trickles", id="the-metadata"), pytest.param("media-trickles", id="the-download")],
)
def test_run_import_when_google_trickles_an_answer_that_ends_its_connection_should_answer_deadline_at_the_stop_line(
    monkeypatch, session, server_tls, behaviour, framing
):
    # Arrange
    clock = SystemClock()
    document = _input_whose_stop_line_is_near(clock)

    # Act
    with _loopback_server(behaviour, tls=server_tls, framing=framing) as (base, seen):
        _drive_on_loopback(monkeypatch, base)
        started = time.monotonic()
        output = run_import(document, settings(), session, clock)
        elapsed = time.monotonic() - started

    # Assert
    assert output == DEADLINE_OUTPUT
    assert elapsed < ENDED_WITHIN_S
    assert [method for method, _, _ in seen.requests if method == "PUT"] == []


# --- a connect over several addresses ---------------------------------------------------------------------------------

# Addresses of the documentation range (RFC 5737) a stubbed name resolution answers with: the stubbed connect never
# lets a packet go to them, and holds each for its whole time-out, as a host that does not answer would.
SILENT_ADDRESSES = [f"192.0.2.{n}" for n in range(1, 6)]
SILENT_HOST = "silent.example.invalid"


@dataclass
class _Connects:
    """When each connect to a silent address began, on the monotonic clock."""

    started: list[float] = field(default_factory=list)


def _silent_addresses(monkeypatch: pytest.MonkeyPatch) -> _Connects:
    """The silent host resolves to five addresses, and a connect to any of them lasts its whole time-out."""
    resolve = socket.getaddrinfo
    connect = socket.socket.connect
    connects = _Connects()

    def resolved(host: Any, port: Any, *arguments: Any, **named: Any) -> Any:
        if host != SILENT_HOST:
            return resolve(host, port, *arguments, **named)
        tcp = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "")
        return [(*tcp, (address, port)) for address in SILENT_ADDRESSES]

    def connected(sock: socket.socket, address: Any) -> None:
        if not (isinstance(address, tuple) and address[0] in SILENT_ADDRESSES):
            connect(sock, address)
            return
        connects.started.append(time.monotonic())
        time.sleep(sock.gettimeout() or 0.0)
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "getaddrinfo", resolved)
    monkeypatch.setattr(socket.socket, "connect", connected)
    return connects


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_connect_when_every_address_is_silent_should_share_the_stop_line_and_start_no_connect_after_it(
    monkeypatch, session, scheme
):
    # Arrange: a connect time-out of 30 s, which urllib3 would give each of the five addresses
    connects = _silent_addresses(monkeypatch)
    stop_line = _near_stop_line()
    stop_line_at = time.monotonic() + stop_line.left_s()

    # Act
    raised, elapsed = _ended(
        lambda: session.get(f"{scheme}://{SILENT_HOST}/target", {}, LONG_CONNECT_TIMEOUT, True, stop_line=stop_line)
    )

    # Assert: the addresses shared the stop line's time, and none was tried once it had passed
    assert isinstance(raised, DeadlineReached)
    assert elapsed < ENDED_WITHIN_S
    assert connects.started != [] and max(connects.started) < stop_line_at


def test_run_import_when_googles_addresses_are_all_silent_should_answer_deadline_at_the_stop_line(monkeypatch, session):
    # Arrange: urllib3 would give each address the whole time-out the job asks (cut to the stop line: about 1 s)
    _silent_addresses(monkeypatch)
    _drive_on_loopback(monkeypatch, f"https://{SILENT_HOST}")
    clock = SystemClock()

    # Act
    started = time.monotonic()
    output = run_import(_input_whose_stop_line_is_near(clock), settings(), session, clock)
    elapsed = time.monotonic() - started

    # Assert
    assert output == DEADLINE_OUTPUT
    assert elapsed < ENDED_WITHIN_S


def test_watchdog_when_requests_end_before_the_stop_line_should_leave_them_whole(session):
    # Act: a request whose stop line is near, but which the server answers at once
    with _loopback_server() as (base, _):
        response = session.get(base + "/target", {}, TIMEOUT, True, stop_line=_near_stop_line())
        body = read_small_answer(response)
        response.close()

    # Assert
    assert body == b"the target"


def test_close_when_a_watchdog_is_armed_should_let_its_timer_go():
    # Arrange
    made = RequestsSession()
    with _loopback_server() as (base, _):
        made.get(base + "/target", {}, TIMEOUT, True, stop_line=_far_stop_line()).close()
    timers_before = [thread for thread in threading.enumerate() if isinstance(thread, threading.Timer)]

    # Act
    made.close()
    for thread in timers_before:
        thread.join(1.0)

    # Assert: the session's timer was armed, and its close let it go
    assert timers_before != []
    assert [thread for thread in timers_before if thread.is_alive()] == []


# --- the bounded body ----------------------------------------------------------------------------------------------


def test_bounded_body_when_read_in_blocks_should_give_its_bytes_in_order_and_say_where_it_is():
    # Arrange
    body = BoundedBody(b"0123456789", _far_stop_line())

    # Act
    blocks = [body.read(4), body.read(4), body.read(4), body.read(4)]

    # Assert
    assert blocks == [b"0123", b"4567", b"89", b""]
    assert len(body) == 10 and body.tell() == 10


def test_bounded_body_when_the_stop_line_has_passed_should_refuse_to_give_a_block():
    # Arrange
    clock = FakeClock()
    body = BoundedBody(b"0123456789", StopLine(clock, clock.monotonic() - 1.0))

    # Act / Assert
    with pytest.raises(DeadlineReached):
        body.read(4)


def test_bounded_body_when_the_stop_line_passes_after_its_last_block_should_still_say_it_ended():
    # Arrange: every byte was given before the stop line passed
    clock = FakeClock()
    body = BoundedBody(b"0123", StopLine(clock, clock.monotonic() + 1.0))
    first = body.read(16)
    clock.advance(2.0)

    # Act
    end = body.read(16)

    # Assert
    assert (first, end) == (b"0123", b"")
