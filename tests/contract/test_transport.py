"""The real HTTP carrier, against servers this module starts on 127.0.0.1 and nothing else.

One POST and its answer: the exact body and its signature header go out, the status, the body and the
answer's signature header come back; no redirect is followed, an oversized answer is cut, the server's
certificate is always verified, and the environment's proxy and CA settings change nothing. A request
without an answer raises TransportFailed, whose text is a bare class name. The transport does not judge
the scheme (the settings do), so plain http://127.0.0.1 addresses are fine here.
"""

from __future__ import annotations

import gzip
import ipaddress
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Mapping

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from core.protocol.transport import HttpAnswer, HttpsTransport, TransportFailed

LOOPBACK = "127.0.0.1"
ANY_FREE_PORT = 0
SIGNATURE_HEADER = "X-VH-Signature"
REQUEST_SIGNATURE = "hmac-sha256=" + "1a" * 32
ANSWER_SIGNATURE = "hmac-sha256=" + "2b" * 32
SIGNATURE_HEX_DIGITS = 64
# A value no failure may show, placed in the request's body.
CANARY = "CANARY-transport-request-body"
REQUEST_BODY = b'{"canary":"' + CANARY.encode("ascii") + b'"}'
ANSWER_BODY = b'{"answer":"from the local server"}'
TIMEOUT_S = 5.0
SHORT_TIMEOUT_S = 0.3
# A test that waits for an answer that never comes must still end quickly.
NO_ANSWER_TEST_BUDGET_S = 3.0
SERVER_POLL_INTERVAL_S = 0.01
HOLD_LIMIT_S = 10.0
MAX_BODY_BYTES = 4096
READ_CHUNK_BYTES = 4096
LARGE_ANSWER_BYTES = 1024 * 1024
LARGE_ANSWER = b"x" * LARGE_ANSWER_BYTES
HTTP_OK = 200
HTTP_FOUND = 302
REDIRECT_STATUSES = [301, HTTP_FOUND, 303, 307, 308]
# What an HTTP client that prepares a redirect's next request would fail on: a URL no parser reads, and
# bytes that are not UTF-8 (the server writes a header's characters as single bytes).
UNREADABLE_LOCATIONS = {"no-url": "http://[", "not-utf-8": "/caf\xe9\xff"}
HTTP_CONFLICT = 409
FIRST_PATH = "/transcode/claim"
SECOND_PATH = "/somewhere/else"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")
CA_BUNDLE_VARIABLES = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
CERTIFICATE_VALIDITY_MARGIN = timedelta(days=1)
BARE_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Received:
    """One request the local server received."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class Reply:
    """What the local server answers on one path; `hold` keeps the request unanswered until the server stops."""

    status: int = HTTP_OK
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    hold: bool = False


class LocalServer:
    """An HTTP server on 127.0.0.1 (a free port) in a thread; the test sets its reply per path and reads what it received."""

    def __init__(self, tls: ssl.SSLContext | None = None) -> None:
        self.received: list[Received] = []
        self.replies: dict[str, Reply] = {}
        self._released = threading.Event()
        self._scheme = "http" if tls is None else "https"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the name http.server looks up
                owner._serve(self)

            do_GET = do_POST

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class's name
                return None

        self._server = ThreadingHTTPServer((LOOPBACK, ANY_FREE_PORT), Handler)
        if tls is not None:
            self._server.socket = tls.wrap_socket(self._server.socket, server_side=True)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": SERVER_POLL_INTERVAL_S}, daemon=True
        )

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def url(self, path: str = "/") -> str:
        return f"{self._scheme}://{LOOPBACK}:{self.port}{path}"

    def paths(self) -> list[str]:
        return [request.path for request in self.received]

    def __enter__(self) -> LocalServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._released.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        headers = {name.lower(): value for name, value in handler.headers.items()}
        self.received.append(Received(handler.command, handler.path, headers, body))
        reply = self.replies.get(handler.path, Reply())
        if reply.hold:
            self._released.wait(HOLD_LIMIT_S)
            return
        handler.send_response(reply.status)
        for name, value in reply.headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(reply.body)))
        handler.end_headers()
        try:
            handler.wfile.write(reply.body)
        except (BrokenPipeError, ConnectionResetError):
            return  # the client stopped reading: what the cut-off test is about


@dataclass(frozen=True)
class SelfSigned:
    """A self-signed certificate for 127.0.0.1 and its key, as PEM files."""

    certificate: Path
    key: Path


def _write_self_signed(directory: Path) -> SelfSigned:
    """A certificate a client accepts for https://127.0.0.1 if, and only if, it trusts this very certificate."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, LOOPBACK)])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - CERTIFICATE_VALIDITY_MARGIN)
        .not_valid_after(now + CERTIFICATE_VALIDITY_MARGIN)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(LOOPBACK))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    files = SelfSigned(certificate=directory / "self-signed.pem", key=directory / "self-signed.key")
    files.certificate.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    files.key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return files


def _closed_port() -> int:
    """A port of 127.0.0.1 on which nothing listens: bound once, then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, ANY_FREE_PORT))
        return probe.getsockname()[1]


def _post(url: str, *, timeout_s: float = TIMEOUT_S, max_body_bytes: int = MAX_BODY_BYTES) -> HttpAnswer:
    return HttpsTransport().post(
        url, REQUEST_BODY, REQUEST_SIGNATURE, timeout_s=timeout_s, max_body_bytes=max_body_bytes
    )


def _failure(url: str, *, timeout_s: float = TIMEOUT_S) -> TransportFailed:
    with pytest.raises(TransportFailed) as raised:
        _post(url, timeout_s=timeout_s)
    return raised.value


@pytest.fixture
def server() -> Iterator[LocalServer]:
    with LocalServer() as local:
        yield local


@pytest.fixture
def self_signed(tmp_path: Path) -> SelfSigned:
    return _write_self_signed(tmp_path)


@pytest.fixture
def tls_server(self_signed: SelfSigned) -> Iterator[LocalServer]:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(self_signed.certificate), keyfile=str(self_signed.key))
    with LocalServer(tls=context) as local:
        yield local


@pytest.fixture
def no_proxy_exemption(monkeypatch: pytest.MonkeyPatch) -> None:
    """No NO_PROXY in the environment: a client that reads the proxy variables would use them for 127.0.0.1 too."""
    for variable in NO_PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


# --- one POST and its answer --------------------------------------------------------------------


def test_post_when_the_server_answers_should_send_the_exact_body_and_signature_and_return_the_answer(server):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_CONFLICT, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH))

    # Assert
    (request,) = server.received
    assert (request.method, request.path, request.body) == ("POST", FIRST_PATH, REQUEST_BODY)
    assert request.headers["x-vh-signature"] == REQUEST_SIGNATURE
    assert request.headers["content-type"] == "application/json"
    assert (answer.status, answer.body, answer.signature) == (HTTP_CONFLICT, ANSWER_BODY, ANSWER_SIGNATURE)


def test_post_when_the_answer_has_no_signature_header_should_return_none_for_the_signature(server):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert answer.signature is None


def test_post_when_the_server_redirects_should_return_the_redirect_and_never_request_its_target(server):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, {"Location": server.url(SECOND_PATH)})
    server.replies[SECOND_PATH] = Reply(HTTP_OK, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH))

    # Assert
    assert answer.status == HTTP_FOUND
    assert server.paths() == [FIRST_PATH]


@pytest.mark.parametrize("status", REDIRECT_STATUSES)
def test_post_when_a_redirect_carries_a_body_far_above_the_cap_should_return_it_cut_like_any_answer(server, status):
    # Arrange: a client that prepares the redirect's next request reads the whole body first.
    server.replies[FIRST_PATH] = Reply(status, {"Location": server.url(SECOND_PATH)}, LARGE_ANSWER)
    server.replies[SECOND_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert answer.status == status
    assert MAX_BODY_BYTES < len(answer.body) <= MAX_BODY_BYTES + READ_CHUNK_BYTES
    assert server.paths() == [FIRST_PATH]


def test_post_when_a_redirect_is_sent_compressed_should_return_its_bytes_as_sent(server):
    # Arrange
    compressed = gzip.compress(LARGE_ANSWER)
    headers = {"Location": server.url(SECOND_PATH), "Content-Encoding": "gzip"}
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, headers, compressed)

    # Act
    answer = _post(server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert (answer.status, answer.body) == (HTTP_FOUND, compressed)


@pytest.mark.parametrize("location", list(UNREADABLE_LOCATIONS.values()), ids=list(UNREADABLE_LOCATIONS))
def test_post_when_a_redirect_names_a_location_that_cannot_be_read_should_return_the_redirect(server, location):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, {"Location": location}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH))

    # Assert: the location is never read, so it cannot fail the POST.
    assert (answer.status, answer.body) == (HTTP_FOUND, ANSWER_BODY)


def test_post_when_the_answer_is_far_above_the_cap_should_return_little_more_than_the_cap(server):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, LARGE_ANSWER)

    # Act
    answer = _post(server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert: more than the cap, so the caller sees an oversized answer, and at most one read past it.
    assert MAX_BODY_BYTES < len(answer.body) <= MAX_BODY_BYTES + READ_CHUNK_BYTES


def test_post_when_the_answer_is_sent_compressed_should_ask_for_no_encoding_and_return_the_bytes_as_sent(server):
    # Arrange: a megabyte that compresses to far less than the cap; a decoder would inflate it first.
    compressed = gzip.compress(LARGE_ANSWER)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {"Content-Encoding": "gzip"}, compressed)

    # Act
    answer = _post(server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    (request,) = server.received
    assert request.headers["accept-encoding"] == "identity"
    assert answer.body == compressed


def test_post_when_the_answer_is_exactly_the_cap_should_return_all_of_it(server):
    # Arrange
    body = b"y" * MAX_BODY_BYTES
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, body)

    # Act
    answer = _post(server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert answer.body == body


# --- no answer ----------------------------------------------------------------------------------


def _assert_shows_nothing_of_the_request(failure: TransportFailed, port: int) -> None:
    text = str(failure)
    assert BARE_CLASS_NAME.fullmatch(text) is not None
    assert [value for value in (LOOPBACK, str(port), CANARY, REQUEST_SIGNATURE) if value in text] == []
    assert failure.__cause__ is None
    assert failure.__context__ is None


def test_post_when_nothing_listens_on_the_port_should_raise_transport_failed_naming_only_a_class():
    # Arrange
    port = _closed_port()

    # Act
    failure = _failure(f"http://{LOOPBACK}:{port}{FIRST_PATH}")

    # Assert
    _assert_shows_nothing_of_the_request(failure, port)


def test_post_when_the_server_accepts_and_never_answers_should_raise_transport_failed_within_the_time_out(server):
    # Arrange
    server.replies[FIRST_PATH] = Reply(hold=True)
    started = time.monotonic()

    # Act
    failure = _failure(server.url(FIRST_PATH), timeout_s=SHORT_TIMEOUT_S)
    waited_s = time.monotonic() - started

    # Assert
    _assert_shows_nothing_of_the_request(failure, server.port)
    assert server.paths() == [FIRST_PATH]
    assert waited_s < NO_ANSWER_TEST_BUDGET_S


# --- the environment changes nothing ------------------------------------------------------------


def test_post_when_the_environment_names_a_proxy_where_nothing_listens_should_still_reach_the_server(
    server, monkeypatch, no_proxy_exemption
):
    # Arrange
    dead_proxy = f"http://{LOOPBACK}:{_closed_port()}"
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, dead_proxy)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert server.paths() == [FIRST_PATH]


def test_plain_requests_when_the_environment_names_that_dead_proxy_should_fail(server, monkeypatch, no_proxy_exemption):
    # Arrange: the control of the test above: a client that does read the environment cannot reach the server.
    dead_proxy = f"http://{LOOPBACK}:{_closed_port()}"
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, dead_proxy)

    # Act / Assert
    with pytest.raises(requests.ConnectionError):
        requests.post(server.url(FIRST_PATH), data=REQUEST_BODY, timeout=TIMEOUT_S)
    assert server.paths() == []


# --- the server's certificate -------------------------------------------------------------------


def test_post_when_the_servers_certificate_is_self_signed_should_raise_transport_failed_before_sending(tls_server):
    # Act
    failure = _failure(tls_server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, tls_server.port)
    assert tls_server.received == []


def test_post_when_the_environment_names_that_certificate_as_a_ca_bundle_should_still_not_trust_it(
    tls_server, self_signed, monkeypatch
):
    # Arrange
    for variable in CA_BUNDLE_VARIABLES:
        monkeypatch.setenv(variable, str(self_signed.certificate))

    # Act
    failure = _failure(tls_server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, tls_server.port)
    assert tls_server.received == []


def test_plain_requests_when_the_environment_names_that_certificate_as_a_ca_bundle_should_trust_it(
    tls_server, self_signed, monkeypatch
):
    # Arrange: the control of the two tests above: the certificate is right for this server, and a client
    # that does read the environment trusts it. Only the missing trust makes the transport fail.
    for variable in CA_BUNDLE_VARIABLES:
        monkeypatch.setenv(variable, str(self_signed.certificate))
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable in NO_PROXY_VARIABLES:
        monkeypatch.setenv(variable, LOOPBACK)
    tls_server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    response = requests.post(tls_server.url(FIRST_PATH), data=REQUEST_BODY, timeout=TIMEOUT_S)

    # Assert
    assert (response.status_code, response.content) == (HTTP_OK, ANSWER_BODY)
    assert tls_server.paths() == [FIRST_PATH]


# --- what an answer shows -----------------------------------------------------------------------


def test_http_answer_when_shown_with_repr_should_show_neither_its_body_nor_its_signature():
    # Arrange
    answer = HttpAnswer(status=HTTP_OK, body=REQUEST_BODY, signature=ANSWER_SIGNATURE)

    # Act
    shown = repr(answer)

    # Assert
    assert str(HTTP_OK) in shown
    assert [value for value in (CANARY, ANSWER_SIGNATURE, ANSWER_SIGNATURE[-SIGNATURE_HEX_DIGITS:]) if value in shown] == []
