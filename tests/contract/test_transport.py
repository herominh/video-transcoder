"""The real HTTP carrier, against servers this module starts on 127.0.0.1 and nothing else.

One POST and its answer, over https only: the exact body and its signature header go out, the status, the
body and the answer's signature header come back; no redirect is followed, an oversized answer is cut and no
content encoding is decoded. An answer is whole and plain or it is no answer: its head is a status line and
header lines of HTTP/1.x, each ended by CRLF and closed by an empty line, within a size and a count; its
body is delimited by one plain length or by chunks of plain hexadecimal sizes, never by both or by anything
else; and a cut or garbled connection (in the head, in the body, or as the only end a body has) is never
returned. The whole exchange ends within its time-out, counted from one deadline: the addresses of the
host share it for the connect (one no socket can be made for is passed over like one that refuses),
neither a TLS handshake nor an answer that trickles outlasts it, and what fails at or after it is named
DeadlinePassed. The address and the signature are visible ASCII, or nothing is sent. The server's
certificate and the address it names are always verified, and nothing in the environment (a proxy, a CA
file or directory, a netrc, a TLS key log) changes anything. A request without a whole answer raises
TransportFailed, whose text is a bare name. A transport is built with an `ssl.SSLContext` or with none.

Every server here answers on 127.0.0.1 under a certificate the test generates, and the transport gets its
own verifying context with that certificate added as a CA: the default HttpsTransport() trusts certifi's
bundle only, so it refuses every server of this module (a test of its own, below). Where a test needs a
host with several addresses, one that does not answer or one no socket can be made for, it steers the
resolver and the socket class the transport's module sees (the `network` fixture): nothing ever leaves
127.0.0.1.
"""

from __future__ import annotations

import enum
import errno
import gzip
import hashlib
import http
import ipaddress
import math
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from core.protocol import transport as transport_module
from core.protocol.transport import HttpAnswer, HttpsTransport, TransportFailed, verifying_tls_context

LOOPBACK = "127.0.0.1"
# An address no server of this module is reached at: only a certificate names it.
OTHER_ADDRESS = "127.0.0.2"
# The loopback address of IPv6, as a host an address names: no socket of this module ever connects to it.
IPV6_HOST = "::1"
HTTPS_PORT = 443
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
# An answer that is refused for what it says is refused as soon as that is read, long before any time-out.
REFUSED_WITHIN_S = 1.0
# Time-outs that are no positive, finite number of seconds.
UNUSABLE_TIMEOUTS = {
    "zero": 0,
    "zero-as-a-float": 0.0,
    "negative": -1.0,
    "not-a-number": math.nan,
    "infinite": math.inf,
    "true": True,
    "a-string": "5",
    "none": None,
}
# Bodies that are not the exact bytes of a message.
NOT_BYTES = {"a-string": "{}", "a-bytearray": bytearray(b"{}"), "none": None}
SERVER_POLL_INTERVAL_S = 0.002
HOLD_LIMIT_S = 10.0
# No read or write of the local server waits longer than this for the client.
SERVER_SOCKET_TIMEOUT_S = 5.0
MAX_BODY_BYTES = 4096
# What the stand-in peers of this module read at once.
PEER_READ_BYTES = 4096
# The limits of an answer's head: its bytes, the closing empty line included, and its header lines.
MAX_HEAD_BYTES = 32 * 1024
MAX_HEADER_LINES = 64
LARGE_ANSWER_BYTES = 1024 * 1024
LARGE_ANSWER = b"x" * LARGE_ANSWER_BYTES
# More than the buffers between a server and a client that has stopped reading can ever hold.
BEYOND_ANY_BUFFER_BYTES = 32 * LARGE_ANSWER_BYTES
# Once a POST has returned, the server sees its connection closed at once: this long is far more than enough.
CLOSE_SEEN_WITHIN_S = 1.0
HTTP_OK = 200
HTTP_NO_CONTENT = 204
HTTP_FOUND = 302
HTTP_NOT_MODIFIED = 304
REDIRECT_STATUSES = [301, HTTP_FOUND, 303, 307, 308]
# The statuses whose answer has no body by definition: its length is zero without a header saying so.
BODILESS_STATUSES = [HTTP_NO_CONTENT, HTTP_NOT_MODIFIED]
# What an HTTP client that prepares a redirect's next request would fail on: a URL no parser reads, and
# bytes that are not UTF-8 (the server writes a header's characters as single bytes).
UNREADABLE_LOCATIONS = {"no-url": "http://[", "not-utf-8": "/caf\xe9\xff"}
HTTP_CONFLICT = 409
FIRST_PATH = "/transcode/claim"
SECOND_PATH = "/somewhere/else"
PATH_WITH_A_QUERY = f"{FIRST_PATH}?a=1&b=two"
USER_AGENT = "video-transcoder-worker"
HEADER_CHARSET = "latin-1"
# The body of a chunked answer goes out in chunks of this size, closed by the final, empty chunk.
CHUNK_BYTES = 16
FINAL_CHUNK = b"0\r\n\r\n"

# An answer that trickles: a few bytes, a pause far shorter than the POST's time-out, again and again.
TRICKLE_PIECE_BYTES = 4
TRICKLE_PAUSE_S = 0.02
# The time-out of a POST whose answer trickles: fifteen pauses long, so no single read ever times out.
DEADLINE_S = 0.3
# The server trickles for this long, then sends the rest at once: a POST that is never cut still ends.
TRICKLE_LASTS_S = 1.5
# A trickling answer is cut at the time-out: within twice of it, long before the trickle ends by itself.
CUT_WITHIN_S = 2 * DEADLINE_S
# More than the server trickles in TRICKLE_LASTS_S (TRICKLE_PIECE_BYTES every TRICKLE_PAUSE_S).
TRICKLED_BODY = b"t" * 4096
FILLER_HEADER = "X-Filler"
FILLER = "f" * 1024
# A trickle that ends by itself, some thirty pieces long: slow, and well within the time-out of its POST.
SOON_TRICKLED_BODY = b"s" * (30 * TRICKLE_PIECE_BYTES)
SHORT_FILLER = "f" * 60
SLOW_ANSWER_TIMEOUT_S = 2.0
DEADLINE_PASSED = "DeadlinePassed"
INCOMPLETE_HEAD = "IncompleteHead"
OVERSIZED_HEAD = "OversizedHead"
MALFORMED_ANSWER = "MalformedAnswer"
INCOMPLETE_BODY = "IncompleteBody"
UNFRAMED_BODY = "UnframedBody"
AMBIGUOUS_FRAMING = "AmbiguousFraming"
NO_ADDRESS = "NoAddress"
CONTENT_LENGTH = "Content-Length"
TRANSFER_ENCODING = "Transfer-Encoding"
CHUNKED = "chunked"
# The start of a TLS record: a handshake message, in the framing of TLS 1.2, announcing its largest size.
# A client waits for all of it before it reads any, so a peer that sends it slowly stalls the handshake.
TLS_RECORD_START = b"\x16\x03\x03\x40\x00"
# The time-out of a POST to a host with two addresses, and how long a POST may take when neither answers:
# about the time-out, and nowhere near one time-out for each.
SHARED_TIMEOUT_S = 0.6
BOTH_SILENT_WITHIN_S = 1.5 * SHARED_TIMEOUT_S
# A connection that is taken only after half its POST's time-out, by a peer that then never speaks TLS:
# the POST ends at the time-out, not that half later.
LATE_TIMEOUT_S = 0.6
LATE_CONNECT_S = LATE_TIMEOUT_S / 2
LATE_CUT_WITHIN_S = LATE_TIMEOUT_S + LATE_CONNECT_S / 2
# A value no failure may show, placed in the error of a resolver that fails.
RESOLVER_CANARY = "CANARY-transport-resolver"
# Another, placed in the error of a socket that cannot be made, and the name such a failure goes by:
# its error's class.
SOCKET_CANARY = "CANARY-transport-socket"
NO_SOCKET = "OSError"
# The lines of an answer's head a connection can be cut in, each named by how it starts.
STATUS_LINE = "HTTP/"
CONNECTION_HEADER = "Connection"
# Interim answers: none of them is the answer, and no request of the transport asks for one.
INTERIM_STATUSES = [100, 101, 102, 103]
NO_FINAL_ANSWER = "NoFinalAnswer"

# Values no error may show, placed in the address.
URL_CANARY = "CANARY-transport-address"
URL_USER = "CANARY-user"
URL_PASSWORD = "CANARY-password"
ADDRESS_PATH = f"{FIRST_PATH}?trace={URL_CANARY}"
# Addresses the transport refuses to post to; the first three name a server that would answer plain HTTP.
# From the space in the host on, each is an https address that is not visible ASCII all through, as it
# is given: a URL parser takes a tab, a line break and a space in front out before anyone can look.
NOT_HTTPS_ADDRESSES = {
    "plain-http": "http://{host}:{port}{path}",
    "another-scheme": "ftp://{host}:{port}{path}",
    "no-scheme": "//{host}:{port}{path}",
    "https-without-a-host": "https://:{port}{path}",
    "https-with-a-port-that-is-no-number": "https://{host}:{port}x{path}",
    "https-with-brackets-around-no-address": "https://[{host}x]:{port}{path}",
    "https-with-port-zero": "https://{host}:0{path}",
    "https-with-a-user-and-a-password": "https://" + URL_USER + ":" + URL_PASSWORD + "@{host}:{port}{path}",
    "https-with-a-user": "https://" + URL_USER + "@{host}:{port}{path}",
    "https-with-a-space-in-the-host": "https://{host} x:{port}{path}",
    "https-with-a-control-character-in-the-host": "https://{host}\x01:{port}{path}",
    "https-with-a-character-that-is-not-ascii-in-the-host": "https://{host}\xe9:{port}{path}",
    "https-with-a-space-in-the-path": "https://{host}:{port}/transcode/a claim",
    "https-with-a-character-that-is-not-ascii-in-the-path": "https://{host}:{port}/transcode/cl\xe9im",
    "https-with-a-character-that-is-not-ascii-in-the-query": "https://{host}:{port}{path}&name=caf\xe9",
    "https-with-a-line-feed-in-the-host": "https://{host}\n:{port}{path}",
    "https-with-a-tab-in-the-host": "https://{host}\t:{port}{path}",
    "https-with-a-carriage-return-in-the-path": "https://{host}:{port}/transcode/cl\raim",
    "https-with-a-space-in-front": " https://{host}:{port}{path}",
    "https-with-a-space-behind": "https://{host}:{port}{path} ",
}
# Addresses that are no string at all.
NOT_STRINGS = {"none": None, "bytes": f"https://{LOOPBACK}{FIRST_PATH}".encode("ascii")}
# What follows a line break in a header's value would be a header of its own if the break went out.
INJECTED_HEADER = "X-Injected: 1"
# Signatures that are no header value: anything but one or more characters of visible ASCII, in a string.
NOT_HEADER_VALUES = {
    "a-line-feed": REQUEST_SIGNATURE + "\n" + INJECTED_HEADER,
    "a-carriage-return": REQUEST_SIGNATURE + "\r" + INJECTED_HEADER,
    "both": REQUEST_SIGNATURE + "\r\n" + INJECTED_HEADER,
    "a-line-feed-then-a-space": REQUEST_SIGNATURE + "\n " + INJECTED_HEADER,
    "a-carriage-return-then-a-space": REQUEST_SIGNATURE + "\r " + INJECTED_HEADER,
    "both-then-a-tab": REQUEST_SIGNATURE + "\r\n\t" + INJECTED_HEADER,
    "a-line-feed-at-its-end": REQUEST_SIGNATURE + "\n",
    "a-space": REQUEST_SIGNATURE + " " + REQUEST_SIGNATURE,
    "a-space-at-its-start": " " + REQUEST_SIGNATURE,
    "a-tab": REQUEST_SIGNATURE + "\t" + REQUEST_SIGNATURE,
    "a-delete-character": REQUEST_SIGNATURE + "\x7f",
    "a-character-that-is-not-ascii": REQUEST_SIGNATURE + "\xe9",
    "an-empty-string": "",
    "none": None,
    "bytes": REQUEST_SIGNATURE.encode("ascii"),
}

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")
# The variables through which an environment names a CA: the first two to `requests`, the others to OpenSSL.
REQUESTS_CA_VARIABLES = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
OPENSSL_CA_FILE_VARIABLE = "SSL_CERT_FILE"
OPENSSL_CA_DIRECTORY_VARIABLE = "SSL_CERT_DIR"
OPENSSL_CA_VARIABLES = (OPENSSL_CA_FILE_VARIABLE, OPENSSL_CA_DIRECTORY_VARIABLE)
KEY_LOG_VARIABLE = "SSLKEYLOGFILE"
NETRC_VARIABLE = "NETRC"
NETRC_LOGIN = "canary-netrc-login"
NETRC_PASSWORD = "canary-netrc-password"
OWNER_ONLY = 0o600
CERTIFICATE_VALIDITY_MARGIN = timedelta(days=1)
# Lowercase with single spaces: OpenSSL's canonical form of the name is then the name itself (_hashed_name).
CERTIFICATE_NAME = "transport test of {address}"
SUBJECT_HASH_BYTES = 4
DER_HEADER_BYTES = 2
BARE_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Received:
    """One request the local server received; `line` is its request line as it came, less the line end."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    line: str


class Framing(enum.Enum):
    """How the local server delimits the body of an answer on the wire.

    NONE adds no header of the server's own: the body then ends with the connection, unless the reply's own
    headers say otherwise.
    """

    CONTENT_LENGTH = "content-length"
    CHUNKED = "chunked"
    NONE = "none"


class Trickle(enum.Enum):
    """The part of an answer the local server writes a few bytes at a time."""

    HEADERS = "headers"
    BODY = "body"


@dataclass(frozen=True)
class Reply:
    """What the local server answers on one path.

    `hold` keeps the request unanswered until the server stops. `cut_after` ends the answer early: the
    server sends that many bytes of the body and closes the connection (a chunked body then lacks its final
    chunk). `head_cut_in` ends it earlier still: the server closes in the middle of the line of the head that
    starts with this text (a header's name, or STATUS_LINE), so the empty line that ends the header block
    never comes. `trickle` names the part that goes out TRICKLE_PIECE_BYTES at a time. `says_close` adds
    `Connection: close`, as a server does when the request asked for it (every request of the transport does).
    `awaits_close` keeps the server on the connection after the answer, until the client closes it.

    For answers no honest server sends: `transfer_encoding` is what a chunked reply calls its coding,
    `raw_head` goes out as it is in place of the status line and the headers, `raw_payload` after the head
    as it is in place of the framed body, and a header may come twice under two spellings of its name
    (_twice()).
    """

    status: int = HTTP_OK
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    hold: bool = False
    framing: Framing = Framing.CONTENT_LENGTH
    cut_after: int | None = None
    head_cut_in: str | None = None
    trickle: Trickle | None = None
    says_close: bool = True
    awaits_close: bool = False
    transfer_encoding: str = CHUNKED
    raw_head: bytes | None = None
    raw_payload: bytes | None = None


def _twice(name: str, first: str, second: str) -> dict[str, str]:
    """One header sent twice: a header's name is the same in any letter case, and a mapping holds both spellings."""
    return {name: first, name.lower(): second}


def _head_of(*lines: bytes) -> bytes:
    """A head of these lines, each ended as HTTP ends a line, and the empty line that closes it."""
    return b"".join(line + b"\r\n" for line in (*lines, b""))


def _status_line(reply: Reply) -> bytes:
    return f"HTTP/1.1 {reply.status} {http.HTTPStatus(reply.status).phrase}\r\n".encode(HEADER_CHARSET)


def _header_block(reply: Reply) -> bytes:
    """The headers of a reply and the empty line that ends them, the reply's own headers first."""
    framing = {
        Framing.CONTENT_LENGTH: {CONTENT_LENGTH: str(len(reply.body))},
        Framing.CHUNKED: {TRANSFER_ENCODING: reply.transfer_encoding},
        Framing.NONE: {},
    }[reply.framing]
    closing = {CONNECTION_HEADER: "close"} if reply.says_close else {}
    lines = [f"{name}: {value}" for name, value in {**reply.headers, **framing, **closing}.items()]
    return "".join(f"{line}\r\n" for line in [*lines, ""]).encode(HEADER_CHARSET)


def _cut_head(reply: Reply, line: str) -> bytes:
    """The head of a reply up to the middle of the line that starts with `line`: all a cut connection delivers."""
    head = _status_line(reply) + _header_block(reply)
    line_starts = head.index(line.encode(HEADER_CHARSET))
    line_ends = head.index(b"\r\n", line_starts)
    return head[: (line_starts + line_ends) // 2]


def _payload(reply: Reply) -> bytes:
    """The body of a reply as it goes on the wire: all of it, or what comes before the cut."""
    if reply.raw_payload is not None:
        return reply.raw_payload
    whole = reply.cut_after is None
    body = reply.body if whole else reply.body[: reply.cut_after]
    if reply.framing is not Framing.CHUNKED:
        return body
    chunks = [body[start : start + CHUNK_BYTES] for start in range(0, len(body), CHUNK_BYTES)]
    encoded = b"".join(b"%x\r\n%s\r\n" % (len(chunk), chunk) for chunk in chunks)
    return encoded + (FINAL_CHUNK if whole else b"")


class LocalServer:
    """An HTTP server on 127.0.0.1 (a free port) in a thread, over TLS when it is given a context.

    The test sets its reply per path and reads what it received; `connections` counts the connections it
    accepted, whatever became of them, and `whole_answers` the answers a client let it send to their last
    byte. `client_gone` is set once the server has seen a client close its connection (while an answer was
    still being written, or after it under `awaits_close`).
    """

    def __init__(self, tls: ssl.SSLContext | None = None) -> None:
        self.received: list[Received] = []
        self.replies: dict[str, Reply] = {}
        self.connections = 0
        self.whole_answers = 0
        self.client_gone = threading.Event()
        self._stopping = threading.Event()
        self._scheme = "http" if tls is None else "https"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            timeout = SERVER_SOCKET_TIMEOUT_S

            def handle(self) -> None:
                try:
                    if isinstance(self.connection, ssl.SSLSocket):
                        self.connection.do_handshake()
                    super().handle()
                except OSError:
                    return  # the client refused the certificate or left: what several tests are about

            def do_POST(self) -> None:  # noqa: N802 - the name http.server looks up
                owner._serve(self)

            do_GET = do_POST

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class's name
                return None

        class Listener(ThreadingHTTPServer):
            # server_close() then waits for every answer in progress: no test leaves a thread behind.
            daemon_threads = False

            def get_request(self) -> tuple[socket.socket, Any]:
                connection, address = super().get_request()
                owner.connections += 1
                if tls is not None:
                    # The handshake happens in the connection's own thread, never in the accepting one.
                    connection = tls.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
                return connection, address

        self._server = Listener((LOOPBACK, ANY_FREE_PORT), Handler)
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
        self._stopping.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        headers = {name.lower(): value for name, value in handler.headers.items()}
        self.received.append(Received(handler.command, handler.path, headers, body, handler.requestline))
        handler.close_connection = True
        reply = self.replies.get(handler.path, Reply())
        if reply.hold:
            self._stopping.wait(HOLD_LIMIT_S)
            return
        try:
            self._answer(handler, reply)
        except TimeoutError:
            return  # the client neither read on nor closed its connection
        except OSError:
            self.client_gone.set()  # it stopped reading or cut the connection: what several tests are about

    def _answer(self, handler: BaseHTTPRequestHandler, reply: Reply) -> None:
        """Write the reply as far as it goes; returning closes the connection."""
        if reply.head_cut_in is not None:
            handler.wfile.write(_cut_head(reply, reply.head_cut_in))
            return
        if reply.raw_head is not None:
            handler.wfile.write(reply.raw_head)
        else:
            self._write(handler.wfile, _status_line(reply), trickled=False)
            self._write(handler.wfile, _header_block(reply), trickled=reply.trickle is Trickle.HEADERS)
        self._write(handler.wfile, _payload(reply), trickled=reply.trickle is Trickle.BODY)
        self.whole_answers += 1
        if reply.awaits_close and handler.connection.recv(1) == b"":
            self.client_gone.set()

    def _write(self, wire: BinaryIO, data: bytes, *, trickled: bool) -> None:
        """Write `data`; trickled, a piece at a time until the trickle has lasted its limit, then the rest at once."""
        sent = 0
        if trickled:
            gives_up_at = time.monotonic() + TRICKLE_LASTS_S
            while sent < len(data) and time.monotonic() < gives_up_at and not self._stopping.is_set():
                wire.write(data[sent : sent + TRICKLE_PIECE_BYTES])
                sent += TRICKLE_PIECE_BYTES
                self._stopping.wait(TRICKLE_PAUSE_S)
        wire.write(data[sent:])


@dataclass(frozen=True)
class SelfSigned:
    """A self-signed certificate for one address and its key, as PEM files.

    `hashed_directory` holds the certificate under the name OpenSSL looks it up by in a CA directory.
    """

    certificate: Path
    key: Path
    hashed_directory: Path


def _hashed_name(name: x509.Name) -> str:
    """The file name under which OpenSSL finds a CA of this subject in a directory: its subject hash, then `.0`.

    The hash is the first four bytes, little-endian, of the SHA-1 of the name's canonical encoding, which
    for a name of one lowercase, single-spaced text value (CERTIFICATE_NAME) is its DER encoding less the
    outer header. The control test of the CA directory proves the result is a name OpenSSL reads.
    """
    canonical = name.public_bytes()[DER_HEADER_BYTES:]
    digest = hashlib.sha1(canonical, usedforsecurity=False).digest()
    return f"{int.from_bytes(digest[:SUBJECT_HASH_BYTES], 'little'):08x}.0"


def _write_self_signed(directory: Path, address: str = LOOPBACK) -> SelfSigned:
    """A certificate a client accepts for https://<address> if, and only if, it trusts this very certificate."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CERTIFICATE_NAME.format(address=address))])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - CERTIFICATE_VALIDITY_MARGIN)
        .not_valid_after(now + CERTIFICATE_VALIDITY_MARGIN)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(address))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM)
    files = SelfSigned(
        certificate=directory / "self-signed.pem", key=directory / "self-signed.key", hashed_directory=directory / "hashed"
    )
    files.hashed_directory.mkdir(parents=True)
    files.certificate.write_bytes(pem)
    (files.hashed_directory / _hashed_name(name)).write_bytes(pem)
    files.key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return files


def _serving(identity: SelfSigned) -> ssl.SSLContext:
    """The context of a server that presents this certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(identity.certificate), keyfile=str(identity.key))
    return context


def _trusting(certificate: Path) -> ssl.SSLContext:
    """The transport's own verifying context, with this certificate as one more CA."""
    context = verifying_tls_context()
    context.load_verify_locations(cafile=str(certificate))
    return context


def _closed_port() -> int:
    """A port of 127.0.0.1 on which nothing listens: bound once, then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, ANY_FREE_PORT))
        return probe.getsockname()[1]


def _post(
    transport: HttpsTransport, url: str, *, timeout_s: float = TIMEOUT_S, max_body_bytes: int = MAX_BODY_BYTES
) -> HttpAnswer:
    return transport.post(url, REQUEST_BODY, REQUEST_SIGNATURE, timeout_s=timeout_s, max_body_bytes=max_body_bytes)


def _failure(transport: HttpsTransport, url: str, *, timeout_s: float = TIMEOUT_S) -> TransportFailed:
    with pytest.raises(TransportFailed) as raised:
        _post(transport, url, timeout_s=timeout_s)
    return raised.value


def _transport_trusting(identity: SelfSigned) -> HttpsTransport:
    return HttpsTransport(tls_context=_trusting(identity.certificate))


@pytest.fixture(scope="module")
def self_signed(tmp_path_factory: pytest.TempPathFactory) -> SelfSigned:
    return _write_self_signed(tmp_path_factory.mktemp("loopback"))


@pytest.fixture
def server(self_signed: SelfSigned) -> Iterator[LocalServer]:
    """A server over TLS, under the certificate for 127.0.0.1."""
    with LocalServer(tls=_serving(self_signed)) as local:
        yield local


@pytest.fixture(scope="module")
def transport(self_signed: SelfSigned) -> HttpsTransport:
    """The transport, trusting the certificate of `server`: one for the module, since a POST leaves nothing in it.

    A test about the environment builds its own, after it has set the environment.
    """
    return _transport_trusting(self_signed)


@pytest.fixture
def plain_server() -> Iterator[LocalServer]:
    """A server without TLS: what an address that is not https would reach."""
    with LocalServer() as local:
        yield local


@pytest.fixture
def named_for_another_address(tmp_path: Path) -> SelfSigned:
    return _write_self_signed(tmp_path, address=OTHER_ADDRESS)


@pytest.fixture
def misnamed_server(named_for_another_address: SelfSigned) -> Iterator[LocalServer]:
    """A server reached at 127.0.0.1 whose certificate names another address only."""
    with LocalServer(tls=_serving(named_for_another_address)) as local:
        yield local


@pytest.fixture
def named_for_the_ipv6_host(tmp_path: Path) -> SelfSigned:
    return _write_self_signed(tmp_path, address=IPV6_HOST)


@pytest.fixture
def ipv6_named_server(named_for_the_ipv6_host: SelfSigned) -> Iterator[LocalServer]:
    """A server on 127.0.0.1 whose certificate names `::1`: where a test's resolver sends that host."""
    with LocalServer(tls=_serving(named_for_the_ipv6_host)) as local:
        yield local


@pytest.fixture
def no_proxy_exemption(monkeypatch: pytest.MonkeyPatch) -> None:
    """No NO_PROXY in the environment: a client that reads the proxy variables would use them for 127.0.0.1 too."""
    for variable in NO_PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """No proxy for 127.0.0.1, whatever the developer's shell holds: for the controls that call `requests`."""
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    for variable in NO_PROXY_VARIABLES:
        monkeypatch.setenv(variable, LOOPBACK)


@pytest.fixture
def no_ca_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No variable names a CA: a test then sets the one it is about."""
    for variable in (*REQUESTS_CA_VARIABLES, *OPENSSL_CA_VARIABLES):
        monkeypatch.delenv(variable, raising=False)


def _as_the_bundle_of_certifi(monkeypatch: pytest.MonkeyPatch, identity: SelfSigned) -> None:
    """Make certifi's bundle, as the transport's module finds it, this one certificate."""
    monkeypatch.setattr(transport_module.certifi, "where", lambda: str(identity.certificate))


class Network:
    """The resolver and the way to the ports of 127.0.0.1, as a test wants the transport to find them.

    Unsteered, the resolver knows the address 127.0.0.1 and no name at all, so nothing is ever looked up
    outside this machine; `resolved` lists the hosts it was asked for and `ports_asked` their ports.
    `resolves_to()` makes it answer every question with these ports of 127.0.0.1, in order (none: a host
    without an address), and `resolution_fails()` with an error. A port in `silent` answers no attempt to
    connect: the attempt lasts its whole time-out and then times out, as when the packets are dropped,
    and `given_s` records that time-out. A port in `late` takes the connection only after so many seconds.
    A port in `unopenable` is given as an address of `::1`, a family no socket can be made for here, as on
    a host that cannot open it: `sockets_refused` counts the sockets that were asked for and not made.
    """

    def __init__(self) -> None:
        self.resolved: list[str] = []
        self.ports_asked: list[int] = []
        self.silent: set[int] = set()
        self.late: dict[int, float] = {}
        self.given_s: list[float] = []
        self.unopenable: set[int] = set()
        self.sockets_refused = 0
        self._ports: list[int] | None = None
        self._fails = False

    def resolves_to(self, *ports: int) -> None:
        self._ports = list(ports)

    def resolution_fails(self) -> None:
        self._fails = True

    def addresses(self, host: str, port: int, *arguments: object, **options: object) -> list[tuple[Any, ...]]:
        """What socket.getaddrinfo() answers."""
        self.resolved.append(host)
        self.ports_asked.append(port)
        if self._fails or (self._ports is None and host != LOOPBACK):
            raise socket.gaierror(socket.EAI_NONAME, RESOLVER_CANARY)
        ports = [port] if self._ports is None else self._ports
        return [self._address(each) for each in ports]

    def _address(self, port: int) -> tuple[Any, ...]:
        if port in self.unopenable:
            return (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (IPV6_HOST, port, 0, 0))
        return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (LOOPBACK, port))


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> Network:
    """The resolver and the socket class of the transport's module, replaced by ones the test steers."""
    steered = Network()

    class SteeredSocket(socket.socket):
        def __init__(self, family: int = -1, *arguments: Any, **options: Any) -> None:
            if family == socket.AF_INET6 and steered.unopenable:
                steered.sockets_refused += 1
                raise OSError(errno.EAFNOSUPPORT, SOCKET_CANARY)
            super().__init__(family, *arguments, **options)

        def connect(self, address: Any) -> None:
            port = address[1]
            if port in steered.silent:
                patience_s = self.gettimeout()
                steered.given_s.append(patience_s)
                time.sleep(patience_s)
                raise TimeoutError("timed out")
            time.sleep(steered.late.get(port, 0.0))
            super().connect(address)

    monkeypatch.setattr(transport_module.socket, "getaddrinfo", steered.addresses)
    monkeypatch.setattr(transport_module.socket, "socket", SteeredSocket)
    return steered


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A port of 127.0.0.1 that takes a connection and never sends a byte: a listening socket nobody serves."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, ANY_FREE_PORT))
        listener.listen()
        yield listener.getsockname()[1]


def _trickle_a_handshake(listener: socket.socket, stopping: threading.Event) -> None:
    """Take one connection and what its client says first, then send the start of a TLS record and trickle on."""
    try:
        connection, _ = listener.accept()
    except OSError:
        return  # no client came before the listener closed
    with connection:
        try:
            connection.recv(PEER_READ_BYTES)
            connection.sendall(TLS_RECORD_START)
            gives_up_at = time.monotonic() + TRICKLE_LASTS_S
            while time.monotonic() < gives_up_at and not stopping.is_set():
                connection.sendall(bytes(TRICKLE_PIECE_BYTES))
                stopping.wait(TRICKLE_PAUSE_S)
        except OSError:
            return  # the client cut the connection: what the test is about


@pytest.fixture
def trickling_handshake_port() -> Iterator[int]:
    """A port of 127.0.0.1 whose TLS handshake never ends: a few bytes of its first record, again and again."""
    stopping = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, ANY_FREE_PORT))
        listener.listen()
        listener.settimeout(SERVER_SOCKET_TIMEOUT_S)
        peer = threading.Thread(target=_trickle_a_handshake, args=(listener, stopping), daemon=True)
        peer.start()
        yield listener.getsockname()[1]
        stopping.set()
        peer.join()


# --- one POST and its answer --------------------------------------------------------------------


def test_post_when_the_server_answers_should_send_the_exact_body_and_signature_and_return_the_answer(server, transport):
    # Arrange: an address with a query, which is part of what the server is asked for.
    server.replies[PATH_WITH_A_QUERY] = Reply(HTTP_CONFLICT, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(PATH_WITH_A_QUERY))

    # Assert
    (request,) = server.received
    assert (request.method, request.path, request.body) == ("POST", PATH_WITH_A_QUERY, REQUEST_BODY)
    assert request.headers["host"] == f"{LOOPBACK}:{server.port}"
    assert request.headers["x-vh-signature"] == REQUEST_SIGNATURE
    assert request.headers["content-type"] == "application/json"
    assert request.headers["user-agent"] == USER_AGENT
    assert (answer.status, answer.body, answer.signature) == (HTTP_CONFLICT, ANSWER_BODY, ANSWER_SIGNATURE)


@pytest.mark.parametrize(
    ("path_in_the_address", "asked_for"),
    [(PATH_WITH_A_QUERY, PATH_WITH_A_QUERY), ("", "/"), ("?a=1", "/?a=1")],
    ids=["a-path-and-a-query", "no-path", "a-query-and-no-path"],
)
def test_post_when_it_asks_should_send_one_http_1_1_request_line_and_say_it_closes_and_takes_json(
    server, transport, path_in_the_address, asked_for
):
    # Act
    _post(transport, f"https://{LOOPBACK}:{server.port}{path_in_the_address}")

    # Assert: the path and the query as the address has them, on a connection the transport does not keep
    (request,) = server.received
    assert request.line == f"POST {asked_for} HTTP/1.1"
    assert request.headers["connection"] == "close"
    assert request.headers["accept"] == "application/json"


@pytest.mark.parametrize("authority", [LOOPBACK, f"{LOOPBACK}:{HTTPS_PORT}"], ids=["no-port", "the-https-port"])
def test_post_when_the_address_names_no_port_but_the_https_one_should_connect_to_it_and_name_the_host_alone(
    server, transport, network, authority
):
    # Arrange: port 443 of 127.0.0.1 is, for this test, where its server answers.
    network.resolves_to(server.port)

    # Act
    answer = _post(transport, f"https://{authority}{FIRST_PATH}")

    # Assert
    (request,) = server.received
    assert answer.status == HTTP_OK
    assert (network.resolved, network.ports_asked) == ([LOOPBACK], [HTTPS_PORT])
    assert request.headers["host"] == LOOPBACK


def test_post_when_the_host_is_an_ipv6_address_should_name_it_in_brackets_in_the_host_header(
    ipv6_named_server, named_for_the_ipv6_host, network
):
    # Arrange: the host `::1` is, for this test, where its server answers on 127.0.0.1.
    network.resolves_to(ipv6_named_server.port)
    transport = _transport_trusting(named_for_the_ipv6_host)
    authority = f"[{IPV6_HOST}]:{ipv6_named_server.port}"

    # Act
    answer = _post(transport, f"https://{authority}{FIRST_PATH}")

    # Assert
    (request,) = ipv6_named_server.received
    assert answer.status == HTTP_OK
    assert network.resolved == [IPV6_HOST]
    assert request.headers["host"] == authority


def test_post_when_the_answer_has_no_signature_header_should_return_none_for_the_signature(server, transport):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert answer.signature is None


# How an answer may write its one signature header.
ONE_SIGNATURE_HEADER = {
    "as-the-contract-writes-it": {SIGNATURE_HEADER: ANSWER_SIGNATURE},
    "its-name-in-small-letters": {SIGNATURE_HEADER.lower(): ANSWER_SIGNATURE},
    "its-name-in-capitals": {SIGNATURE_HEADER.upper(): ANSWER_SIGNATURE},
    "its-value-between-spaces-and-tabs": {SIGNATURE_HEADER: f" \t{ANSWER_SIGNATURE}\t "},
}


@pytest.mark.parametrize("headers", list(ONE_SIGNATURE_HEADER.values()), ids=list(ONE_SIGNATURE_HEADER))
def test_post_when_the_answer_has_one_signature_header_should_return_its_value_without_the_white_space_around_it(
    server, transport, headers
):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, headers, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert answer.signature == ANSWER_SIGNATURE


@pytest.mark.parametrize("second", [ANSWER_SIGNATURE, REQUEST_SIGNATURE], ids=["the-same-twice", "two-that-differ"])
def test_post_when_the_answer_has_two_signature_headers_should_return_none_for_the_signature(
    server, transport, second
):
    # Arrange
    headers = _twice(SIGNATURE_HEADER, ANSWER_SIGNATURE, second)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, headers, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert: two signatures name none, so what the answer carries cannot verify.
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert answer.signature is None


@pytest.mark.parametrize(
    "transfer_encoding",
    [CHUNKED, "Chunked", "CHUNKED", f"  {CHUNKED}", f"{CHUNKED}  ", f"{CHUNKED}\t"],
    ids=["chunked", "capitalized", "in-capitals", "padded-in-front", "padded-behind", "with-a-tab-behind"],
)
def test_post_when_the_body_is_chunked_should_return_all_of_it(server, transport, transfer_encoding):
    # Arrange: longer than one chunk, so the body takes several; the coding is `chunked` in any letter case.
    body = ANSWER_BODY * CHUNK_BYTES
    headers = {SIGNATURE_HEADER: ANSWER_SIGNATURE}
    server.replies[FIRST_PATH] = Reply(
        HTTP_OK, headers, body, framing=Framing.CHUNKED, transfer_encoding=transfer_encoding
    )

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body, answer.signature) == (HTTP_OK, body, ANSWER_SIGNATURE)


@pytest.mark.parametrize("status", BODILESS_STATUSES)
def test_post_when_a_status_that_has_no_body_comes_without_a_content_length_should_return_it_with_an_empty_body(
    server, transport, status
):
    # Arrange: nothing delimits a body, and none is needed: the status itself says there is none.
    server.replies[FIRST_PATH] = Reply(status, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, framing=Framing.NONE)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body, answer.signature) == (status, b"", ANSWER_SIGNATURE)


def test_post_when_a_304_announces_the_length_of_what_it_does_not_send_should_return_it_with_an_empty_body(
    server, transport
):
    # Arrange: a 304 may carry the length the full answer would have; it never carries that body.
    headers = {CONTENT_LENGTH: str(len(ANSWER_BODY)), SIGNATURE_HEADER: ANSWER_SIGNATURE}
    server.replies[FIRST_PATH] = Reply(HTTP_NOT_MODIFIED, headers, framing=Framing.NONE)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body, answer.signature) == (HTTP_NOT_MODIFIED, b"", ANSWER_SIGNATURE)


def test_post_when_the_server_redirects_should_return_the_redirect_and_never_request_its_target(server, transport):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, {"Location": server.url(SECOND_PATH)})
    server.replies[SECOND_PATH] = Reply(HTTP_OK, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert answer.status == HTTP_FOUND
    assert server.paths() == [FIRST_PATH]


@pytest.mark.parametrize("status", REDIRECT_STATUSES)
def test_post_when_a_redirect_carries_a_body_far_above_the_cap_should_return_it_cut_like_any_answer(
    server, transport, status
):
    # Arrange: a client that prepares the redirect's next request reads the whole body first.
    server.replies[FIRST_PATH] = Reply(status, {"Location": server.url(SECOND_PATH)}, LARGE_ANSWER)
    server.replies[SECOND_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert answer.status == status
    assert answer.body == LARGE_ANSWER[: MAX_BODY_BYTES + 1]
    assert server.paths() == [FIRST_PATH]


def test_post_when_a_redirect_is_sent_compressed_should_return_its_bytes_as_sent(server, transport):
    # Arrange
    compressed = gzip.compress(LARGE_ANSWER)
    headers = {"Location": server.url(SECOND_PATH), "Content-Encoding": "gzip"}
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, headers, compressed)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert (answer.status, answer.body) == (HTTP_FOUND, compressed)


@pytest.mark.parametrize("location", list(UNREADABLE_LOCATIONS.values()), ids=list(UNREADABLE_LOCATIONS))
def test_post_when_a_redirect_names_a_location_that_cannot_be_read_should_return_the_redirect(
    server, transport, location
):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_FOUND, {"Location": location}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert: the location is never read, so it cannot fail the POST.
    assert (answer.status, answer.body) == (HTTP_FOUND, ANSWER_BODY)


# Answers far above the cap, in every way a body is delimited. The last announces the largest chunk a size
# line can, and sends a megabyte of it: nothing waits for the rest.
FAR_ABOVE_THE_CAP = {
    "a-content-length": Reply(HTTP_OK, {}, LARGE_ANSWER),
    "many-small-chunks": Reply(HTTP_OK, {}, LARGE_ANSWER, framing=Framing.CHUNKED),
    "one-chunk": Reply(
        HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=b"%x\r\n%s\r\n" % (LARGE_ANSWER_BYTES, LARGE_ANSWER) + FINAL_CHUNK
    ),
    "a-chunk-of-the-largest-size": Reply(HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=b"ffffffff\r\n" + LARGE_ANSWER),
}


@pytest.mark.parametrize("reply", list(FAR_ABOVE_THE_CAP.values()), ids=list(FAR_ABOVE_THE_CAP))
def test_post_when_the_answer_is_far_above_the_cap_should_return_one_byte_more_than_the_cap(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert: one byte above the cap is all the caller needs to see an oversized answer.
    assert answer.body == LARGE_ANSWER[: MAX_BODY_BYTES + 1]


def test_post_when_the_answer_is_sent_compressed_should_ask_for_no_encoding_and_return_the_bytes_as_sent(
    server, transport
):
    # Arrange: a megabyte that compresses to far less than the cap; a decoder would inflate it first.
    compressed = gzip.compress(LARGE_ANSWER)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {"Content-Encoding": "gzip"}, compressed)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    (request,) = server.received
    assert request.headers["accept-encoding"] == "identity"
    assert answer.body == compressed


@pytest.mark.parametrize("framing", [Framing.CONTENT_LENGTH, Framing.CHUNKED], ids=["content-length", "chunked"])
def test_post_when_the_answer_is_exactly_the_cap_should_return_all_of_it(server, transport, framing):
    # Arrange
    body = b"y" * MAX_BODY_BYTES
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, body, framing=framing)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)

    # Assert
    assert answer.body == body


# --- https only ---------------------------------------------------------------------------------


@pytest.mark.parametrize("address", list(NOT_HTTPS_ADDRESSES.values()), ids=list(NOT_HTTPS_ADDRESSES))
def test_post_when_the_address_is_not_a_plain_https_address_should_raise_value_error_and_open_no_connection(
    plain_server, transport, network, address
):
    # Arrange
    url = address.format(host=LOOPBACK, port=plain_server.port, path=ADDRESS_PATH)
    unshown = (LOOPBACK, str(plain_server.port), URL_CANARY, URL_USER, URL_PASSWORD)

    # Act
    with pytest.raises(ValueError) as raised:
        _post(transport, url, timeout_s=SHORT_TIMEOUT_S)

    # Assert: a wiring bug, told without the address, before its host is so much as looked up.
    text = str(raised.value)
    assert [value for value in unshown if value in text] == []
    assert network.resolved == []
    assert plain_server.connections == 0
    assert plain_server.received == []


@pytest.mark.parametrize("address", list(NOT_STRINGS.values()), ids=list(NOT_STRINGS))
def test_post_when_the_address_is_no_string_should_raise_value_error(transport, network, address):
    # Arrange: the resolver is the test's own, so a look-up would show.
    looked_up = network.resolved

    # Act
    with pytest.raises(ValueError):
        _post(transport, address, timeout_s=SHORT_TIMEOUT_S)

    # Assert
    assert looked_up == []


# --- no answer ----------------------------------------------------------------------------------


def _assert_shows_nothing_of_the_request(failure: TransportFailed, port: int) -> None:
    text = str(failure)
    assert BARE_CLASS_NAME.fullmatch(text) is not None
    assert [value for value in (LOOPBACK, str(port), CANARY, REQUEST_SIGNATURE) if value in text] == []
    assert failure.__cause__ is None
    assert failure.__context__ is None


def test_post_when_nothing_listens_on_the_port_should_raise_transport_failed_naming_only_a_class(transport):
    # Arrange
    port = _closed_port()

    # Act
    failure = _failure(transport, f"https://{LOOPBACK}:{port}{FIRST_PATH}")

    # Assert
    _assert_shows_nothing_of_the_request(failure, port)


def test_post_when_the_server_accepts_and_never_answers_should_raise_deadline_passed_at_the_time_out(
    server, transport
):
    # Arrange
    server.replies[FIRST_PATH] = Reply(hold=True)
    started = time.monotonic()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH), timeout_s=DEADLINE_S)
    waited_s = time.monotonic() - started

    # Assert: named by the clock, whichever of the socket's own time-out and the watchdog ended the wait.
    assert str(failure) == DEADLINE_PASSED
    assert DEADLINE_S / 2 < waited_s < CUT_WITHIN_S
    assert server.paths() == [FIRST_PATH]
    _assert_shows_nothing_of_the_request(failure, server.port)


# --- the head of an answer: whole, plain and within its limits, or no answer --------------------

OK_LINE = b"HTTP/1.1 200 OK"
NO_BODY_LINE = b"Content-Length: 0"
BODY_LENGTH_LINE = b"Content-Length: %d" % len(ANSWER_BODY)
# Heads a lenient reader would take for a `200` without a body. None is of the grammar: a status line of
# HTTP/1.0 or 1.1 with a status of three digits, header lines `name: value`, CRLF after each line.
# Nearly a whole head's allowance of spaces in one header value.
LONG_RUN_OF_SPACES = 32_000
MALFORMED_HEADS = {
    "a-status-with-a-sign": _head_of(b"HTTP/1.1 +200 OK", NO_BODY_LINE),
    "a-status-with-an-underscore": _head_of(b"HTTP/1.1 2_00 OK", NO_BODY_LINE),
    "a-status-of-two-digits": _head_of(b"HTTP/1.1 20 OK", NO_BODY_LINE),
    "a-status-of-four-digits": _head_of(b"HTTP/1.1 2000 OK", NO_BODY_LINE),
    "two-spaces-before-the-status": _head_of(b"HTTP/1.1  200 OK", NO_BODY_LINE),
    "http-0.9": _head_of(b"HTTP/0.9 200 OK", NO_BODY_LINE),
    "http-2.0": _head_of(b"HTTP/2.0 200 OK", NO_BODY_LINE),
    "http-in-small-letters": _head_of(b"http/1.1 200 OK", NO_BODY_LINE),
    "a-status-line-ended-by-a-bare-line-feed": OK_LINE + b"\n" + _head_of(NO_BODY_LINE),
    "a-header-line-ended-by-a-bare-line-feed": OK_LINE + b"\r\n" + NO_BODY_LINE + b"\n\r\n",
    "a-head-closed-by-a-bare-line-feed": OK_LINE + b"\r\n" + NO_BODY_LINE + b"\r\n\n",
    "a-carriage-return-inside-a-line": _head_of(OK_LINE, b"X-Odd: one\rtwo", NO_BODY_LINE),
    "a-header-folded-with-a-space": _head_of(OK_LINE, b"X-Folded: one", b" two", NO_BODY_LINE),
    "a-header-folded-with-a-tab": _head_of(OK_LINE, b"X-Folded: one", b"\ttwo", NO_BODY_LINE),
    "a-line-without-a-colon": _head_of(OK_LINE, b"Content-Length 0"),
    "a-space-before-the-colon": _head_of(OK_LINE, b"Content-Length : 0"),
    "a-header-without-a-name": _head_of(OK_LINE, b": 0", NO_BODY_LINE),
    "a-control-character-in-a-value": _head_of(OK_LINE, b"X-Odd: one\x00two", NO_BODY_LINE),
    # A reader that looked for the end of this white space by trial would run for seconds, past any
    # time-out of this file: the failure would then be named after the deadline, not after the line.
    "a-control-character-behind-a-long-run-of-spaces": _head_of(
        OK_LINE, b"X-Odd:" + b" " * LONG_RUN_OF_SPACES + b"\x00", NO_BODY_LINE
    ),
}


@pytest.mark.parametrize("head", list(MALFORMED_HEADS.values()), ids=list(MALFORMED_HEADS))
def test_post_when_a_line_of_the_head_is_not_of_the_grammar_should_raise_malformed_answer(server, transport, head):
    # Arrange
    server.replies[FIRST_PATH] = Reply(raw_head=head)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == MALFORMED_ANSWER
    _assert_shows_nothing_of_the_request(failure, server.port)


# Heads that are of the grammar in a form a server may well choose.
PLAIN_HEADS = {
    "no-reason-phrase": _head_of(b"HTTP/1.1 200", BODY_LENGTH_LINE),
    "an-empty-reason-phrase": _head_of(b"HTTP/1.1 200 ", BODY_LENGTH_LINE),
    "http-1.0": _head_of(b"HTTP/1.0 200 OK", BODY_LENGTH_LINE),
    "no-space-after-the-colon": _head_of(OK_LINE, BODY_LENGTH_LINE.replace(b": ", b":")),
    "spaces-and-tabs-around-a-value": _head_of(OK_LINE, BODY_LENGTH_LINE.replace(b": ", b": \t ") + b" \t "),
    "a-name-in-capitals": _head_of(OK_LINE, BODY_LENGTH_LINE.upper()),
    "a-header-without-a-value": _head_of(OK_LINE, b"X-Empty:", BODY_LENGTH_LINE),
    "bytes-above-ascii-in-a-value": _head_of(OK_LINE, b"X-Latin: caf\xe9", BODY_LENGTH_LINE),
}


@pytest.mark.parametrize("head", list(PLAIN_HEADS.values()), ids=list(PLAIN_HEADS))
def test_post_when_the_head_is_of_the_grammar_in_another_plain_form_should_return_the_answer(server, transport, head):
    # Arrange
    server.replies[FIRST_PATH] = Reply(raw_head=head, raw_payload=ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)


def _head_cut_in_its_signature(sent_before: Mapping[str, str], status: int = HTTP_OK) -> Reply:
    """An answer cut in the middle of its signature header, the headers `sent_before` it having come whole."""
    headers = {**sent_before, SIGNATURE_HEADER: ANSWER_SIGNATURE}
    return Reply(status, headers, ANSWER_BODY, framing=Framing.NONE, head_cut_in=SIGNATURE_HEADER)


# Answers whose connection is cut before the empty line that closes the head, wherever the cut falls and
# whatever had come before it: the last three had already said that they carry no body.
HEADS_CUT_SHORT = {
    "before-its-first-byte": Reply(raw_head=b""),
    "in-the-status-line": Reply(HTTP_OK, {}, ANSWER_BODY, head_cut_in=STATUS_LINE),
    "in-its-first-header": Reply(HTTP_OK, {FILLER_HEADER: SHORT_FILLER}, ANSWER_BODY, head_cut_in=FILLER_HEADER),
    "in-the-signature-header": _head_cut_in_its_signature({}),
    "in-the-line-end-of-a-header": Reply(raw_head=OK_LINE + b"\r\n" + BODY_LENGTH_LINE + b"\r"),
    "before-the-closing-empty-line": Reply(raw_head=OK_LINE + b"\r\n" + BODY_LENGTH_LINE + b"\r\n"),
    "in-the-closing-empty-line": Reply(raw_head=OK_LINE + b"\r\n" + BODY_LENGTH_LINE + b"\r\n\r"),
    "after-a-content-length": _head_cut_in_its_signature({CONTENT_LENGTH: str(len(ANSWER_BODY))}),
    "after-transfer-encoding-chunked": _head_cut_in_its_signature({TRANSFER_ENCODING: CHUNKED}),
    "after-a-content-length-of-zero": _head_cut_in_its_signature({CONTENT_LENGTH: "0"}),
    "of-a-204": _head_cut_in_its_signature({}, status=HTTP_NO_CONTENT),
    "of-a-304": _head_cut_in_its_signature({}, status=HTTP_NOT_MODIFIED),
}


@pytest.mark.parametrize("reply", list(HEADS_CUT_SHORT.values()), ids=list(HEADS_CUT_SHORT))
def test_post_when_the_head_is_cut_before_its_closing_empty_line_should_raise_incomplete_head(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: never an answer, though a status line and headers may have come.
    assert str(failure) == INCOMPLETE_HEAD
    _assert_shows_nothing_of_the_request(failure, server.port)


def _answer_with_a_head_of(size: int) -> Reply:
    """An answer whose head is exactly `size` bytes long, its closing empty line included."""
    unfilled = Reply(HTTP_OK, {FILLER_HEADER: ""}, ANSWER_BODY)
    room = size - len(_status_line(unfilled) + _header_block(unfilled))
    return Reply(HTTP_OK, {FILLER_HEADER: "f" * room}, ANSWER_BODY)


def _answer_with_header_lines(count: int) -> Reply:
    """An answer with exactly `count` header lines, those the server adds for the framing included."""
    own_lines = _header_block(Reply(HTTP_OK, {}, ANSWER_BODY)).count(b"\r\n") - 1
    fillers = {f"{FILLER_HEADER}-{number}": "f" for number in range(count - own_lines)}
    return Reply(HTTP_OK, fillers, ANSWER_BODY)


# Answers whose head is at one of its two limits, and ones just beyond or far beyond them.
HEADS_AT_THEIR_LIMITS = {
    "all-the-bytes-allowed": _answer_with_a_head_of(MAX_HEAD_BYTES),
    "all-the-header-lines-allowed": _answer_with_header_lines(MAX_HEADER_LINES),
}
HEADS_BEYOND_THEIR_LIMITS = {
    "one-byte-more-than-allowed": _answer_with_a_head_of(MAX_HEAD_BYTES + 1),
    "one-header-line-of-a-megabyte": Reply(HTTP_OK, {FILLER_HEADER: "f" * LARGE_ANSWER_BYTES}, ANSWER_BODY),
    "a-status-line-of-a-megabyte": Reply(raw_head=_head_of(OK_LINE + b" " + LARGE_ANSWER, NO_BODY_LINE)),
    "one-header-line-more-than-allowed": _answer_with_header_lines(MAX_HEADER_LINES + 1),
}


@pytest.mark.parametrize("reply", list(HEADS_AT_THEIR_LIMITS.values()), ids=list(HEADS_AT_THEIR_LIMITS))
def test_post_when_the_head_is_exactly_at_a_limit_should_return_the_answer(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)


@pytest.mark.parametrize("reply", list(HEADS_BEYOND_THEIR_LIMITS.values()), ids=list(HEADS_BEYOND_THEIR_LIMITS))
def test_post_when_the_head_is_beyond_a_limit_should_raise_oversized_head(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == OVERSIZED_HEAD
    _assert_shows_nothing_of_the_request(failure, server.port)


@pytest.mark.parametrize("status", INTERIM_STATUSES)
def test_post_when_the_answer_is_an_interim_one_should_raise_no_final_answer(server, transport, status):
    # Arrange: an interim answer whose own framing is whole; the final answer is what the POST is for.
    server.replies[FIRST_PATH] = Reply(status, {}, b"")

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == NO_FINAL_ANSWER


# --- the body of an answer: delimited in one plain way and whole, or no answer ------------------


@pytest.mark.parametrize(
    "cut_after",
    [0, len(ANSWER_BODY) // 2, len(ANSWER_BODY) - 1],
    ids=["before-its-first-byte", "half-way", "one-byte-short"],
)
def test_post_when_the_server_closes_before_the_announced_length_should_raise_incomplete_body(
    server, transport, cut_after
):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY, cut_after=cut_after)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == INCOMPLETE_BODY
    _assert_shows_nothing_of_the_request(failure, server.port)


ONE_CHUNK = b"5\r\nhello\r\n"
# Chunked bodies whose connection is cut, each by what had come by then. From the last chunk on, all the
# data is there and the answer is still not whole: the empty line that ends it has not come.
CHUNKED_BODIES_CUT_SHORT = {
    "inside-a-size-line": b"5",
    "after-a-size-line": b"5\r\n",
    "inside-the-data-of-a-chunk": b"5\r\nhel",
    "before-the-line-end-of-a-chunk": b"5\r\nhello",
    "inside-the-line-end-of-a-chunk": b"5\r\nhello\r",
    "before-the-last-chunk": ONE_CHUNK,
    "inside-the-last-chunk": ONE_CHUNK + b"0",
    "before-the-final-empty-line": ONE_CHUNK + b"0\r\n",
    "inside-the-final-empty-line": ONE_CHUNK + b"0\r\n\r",
}


@pytest.mark.parametrize("chunks", list(CHUNKED_BODIES_CUT_SHORT.values()), ids=list(CHUNKED_BODIES_CUT_SHORT))
def test_post_when_a_chunked_body_is_cut_short_should_raise_incomplete_body(server, transport, chunks):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=chunks)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == INCOMPLETE_BODY
    _assert_shows_nothing_of_the_request(failure, server.port)


# Answers whose body nothing delimits but the end of the connection: no length, and no chunks.
UNFRAMED_ANSWERS = {
    "a-small-body": Reply(HTTP_OK, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, ANSWER_BODY, framing=Framing.NONE),
    "a-body-far-above-the-cap": Reply(HTTP_OK, {}, LARGE_ANSWER, framing=Framing.NONE),
    "no-body-at-all": Reply(HTTP_OK, {SIGNATURE_HEADER: ANSWER_SIGNATURE}, framing=Framing.NONE),
    "a-redirect-without-a-body": Reply(HTTP_FOUND, {"Location": SECOND_PATH}, framing=Framing.NONE),
    "http-1.0-without-a-length": Reply(raw_head=_head_of(b"HTTP/1.0 200 OK"), raw_payload=ANSWER_BODY),
}


@pytest.mark.parametrize("reply", list(UNFRAMED_ANSWERS.values()), ids=list(UNFRAMED_ANSWERS))
def test_post_when_only_the_end_of_the_connection_ends_the_body_should_raise_unframed_body(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: a connection that was cut would look the same, so this is no answer.
    assert str(failure) == UNFRAMED_BODY
    _assert_shows_nothing_of_the_request(failure, server.port)


ANSWER_LENGTH = str(len(ANSWER_BODY))
# Answers whose framing is not one plain thing. A lenient reader takes each of the first four lengths for
# the body's true length, and can follow each of the codings.
AMBIGUOUS_FRAMINGS = {
    "two-content-lengths-that-agree": Reply(
        HTTP_OK, _twice(CONTENT_LENGTH, ANSWER_LENGTH, ANSWER_LENGTH), ANSWER_BODY, framing=Framing.NONE
    ),
    "two-content-lengths-that-differ": Reply(
        HTTP_OK, _twice(CONTENT_LENGTH, ANSWER_LENGTH, "1"), ANSWER_BODY, framing=Framing.NONE
    ),
    "a-content-length-with-a-sign": Reply(HTTP_OK, {CONTENT_LENGTH: f"+{ANSWER_LENGTH}"}, ANSWER_BODY, framing=Framing.NONE),
    "a-content-length-with-an-underscore": Reply(
        HTTP_OK, {CONTENT_LENGTH: "_".join(ANSWER_LENGTH)}, ANSWER_BODY, framing=Framing.NONE
    ),
    "a-content-length-with-a-space-inside": Reply(
        HTTP_OK, {CONTENT_LENGTH: " ".join(ANSWER_LENGTH)}, ANSWER_BODY, framing=Framing.NONE
    ),
    "a-negative-content-length": Reply(HTTP_OK, {CONTENT_LENGTH: f"-{ANSWER_LENGTH}"}, ANSWER_BODY, framing=Framing.NONE),
    "a-content-length-that-is-no-number": Reply(HTTP_OK, {CONTENT_LENGTH: "many"}, ANSWER_BODY, framing=Framing.NONE),
    "an-empty-content-length": Reply(HTTP_OK, {CONTENT_LENGTH: ""}, ANSWER_BODY, framing=Framing.NONE),
    "a-content-length-of-nineteen-digits": Reply(HTTP_OK, {CONTENT_LENGTH: "9" * 19}, ANSWER_BODY, framing=Framing.NONE),
    "an-empty-transfer-encoding": Reply(HTTP_OK, {}, ANSWER_BODY, framing=Framing.CHUNKED, transfer_encoding=""),
    "a-coding-before-chunked": Reply(
        HTTP_OK, {}, ANSWER_BODY, framing=Framing.CHUNKED, transfer_encoding=f"gzip, {CHUNKED}"
    ),
    "a-coding-that-is-not-chunked": Reply(HTTP_OK, {}, ANSWER_BODY, framing=Framing.CHUNKED, transfer_encoding="gzip"),
    "chunked-announced-twice": Reply(
        HTTP_OK, _twice(TRANSFER_ENCODING, CHUNKED, CHUNKED), ANSWER_BODY, framing=Framing.NONE, raw_payload=FINAL_CHUNK
    ),
    "chunks-and-a-content-length": Reply(HTTP_OK, {CONTENT_LENGTH: ANSWER_LENGTH}, ANSWER_BODY, framing=Framing.CHUNKED),
    "a-204-that-announces-chunks": Reply(HTTP_NO_CONTENT, {}, framing=Framing.CHUNKED),
    "a-304-that-announces-chunks": Reply(HTTP_NOT_MODIFIED, {}, framing=Framing.CHUNKED),
}


@pytest.mark.parametrize("reply", list(AMBIGUOUS_FRAMINGS.values()), ids=list(AMBIGUOUS_FRAMINGS))
def test_post_when_the_body_is_not_delimited_in_one_plain_way_should_raise_ambiguous_framing(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: two readers could take two different bodies from it, so this is no answer.
    assert str(failure) == AMBIGUOUS_FRAMING
    _assert_shows_nothing_of_the_request(failure, server.port)


# Chunked bodies a lenient reader would take for `hello`. None is of the grammar: a size line of one to
# eight hexadecimal digits and nothing else, CRLF after it and after the data, and after the last chunk
# the empty line at once.
MALFORMED_CHUNKS = {
    "a-size-of-minus-one": b"-1\r\nhello\r\n" + FINAL_CHUNK,
    "a-negative-size": b"-5\r\nhello\r\n" + FINAL_CHUNK,
    "a-negative-size-after-a-whole-chunk": ONE_CHUNK + b"-5\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-with-a-plus-sign": b"+5\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-with-a-prefix": b"0x5\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-with-an-underscore": b"0_5\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-with-an-extension": b"5;name=value\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-with-a-short-extension": b"5;a\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-of-nine-digits": b"000000005\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-that-is-no-number": b"zz\r\nhello\r\n" + FINAL_CHUNK,
    "a-space-before-the-size": b" 5\r\nhello\r\n" + FINAL_CHUNK,
    "a-space-behind-the-size": b"5 \r\nhello\r\n" + FINAL_CHUNK,
    "an-empty-size-line": b"\r\nhello\r\n" + FINAL_CHUNK,
    "a-size-line-ended-by-a-bare-line-feed": b"5\nhello\r\n" + FINAL_CHUNK,
    "data-with-no-line-end-behind-it": b"5\r\nhello" + FINAL_CHUNK,
    "data-with-two-other-bytes-behind-it": b"5\r\nhello??" + FINAL_CHUNK,
    "data-ended-by-a-bare-line-feed": b"5\r\nhello\n" + FINAL_CHUNK,
    "a-trailer-after-the-last-chunk": ONE_CHUNK + b"0\r\nX-Trailer: 1\r\n\r\n",
    "a-last-chunk-ended-by-a-bare-line-feed": ONE_CHUNK + b"0\n\r\n",
    "a-body-closed-by-a-bare-line-feed": ONE_CHUNK + b"0\r\n\n",
}


@pytest.mark.parametrize("chunks", list(MALFORMED_CHUNKS.values()), ids=list(MALFORMED_CHUNKS))
def test_post_when_a_chunked_body_is_not_of_the_grammar_should_raise_malformed_answer(server, transport, chunks):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=chunks)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == MALFORMED_ANSWER
    _assert_shows_nothing_of_the_request(failure, server.port)


def test_post_when_chunk_sizes_are_plain_hexadecimal_in_any_case_should_return_the_body(server, transport):
    # Arrange: ten bytes announced as `A` and as `a`, five under leading zeros, and a last chunk of three zeros.
    ten = b"0123456789"
    chunks = b"A\r\n" + ten + b"\r\na\r\n" + ten + b"\r\n00000005\r\nhello\r\n000\r\n\r\n"
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=chunks)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ten + ten + b"hello")


def test_post_when_a_chunk_announces_a_size_of_minus_one_should_refuse_it_at_once_and_read_no_further(
    server, transport
):
    # Arrange: far more follows the size line than any buffer on the way holds, so only a client that goes
    # on reading lets the server send its last byte. Such a size makes a lenient reader read to the end.
    endless = b"-1\r\n" + bytes(BEYOND_ANY_BUFFER_BYTES)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, framing=Framing.CHUNKED, raw_payload=endless)
    started = time.monotonic()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))
    waited_s = time.monotonic() - started

    # Assert
    assert str(failure) == MALFORMED_ANSWER
    assert waited_s < REFUSED_WITHIN_S
    assert server.whole_answers == 0


@pytest.mark.parametrize(
    "error", [RuntimeError(CANARY), KeyError(CANARY), AssertionError(CANARY)], ids=lambda error: type(error).__name__
)
def test_post_when_reading_the_answer_fails_in_a_way_nobody_foresaw_should_raise_transport_failed_naming_only_a_class(
    server, transport, monkeypatch, error
):
    # Arrange: the transport's own reader breaks in the middle of the exchange, whatever the reason.
    def broken(reader: BinaryIO, max_body_bytes: int) -> HttpAnswer:
        raise error

    monkeypatch.setattr(transport_module, "_read_answer", broken)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: no answer, and nothing of the error but its class.
    assert str(failure) == type(error).__name__
    _assert_shows_nothing_of_the_request(failure, server.port)


# --- nothing is left open -----------------------------------------------------------------------


@pytest.fixture
def kept_sockets(monkeypatch: pytest.MonkeyPatch) -> list[ssl.SSLSocket]:
    """Every TLS socket the transport connects, kept alive past its POST.

    A socket nobody refers to is closed as soon as CPython drops it, so that alone shows nothing: only with
    the socket still alive does a closed connection prove that the transport closed it.
    """
    kept: list[ssl.SSLSocket] = []
    connected = transport_module._connected

    def keeping(*arguments: Any) -> ssl.SSLSocket:
        tls_socket = connected(*arguments)
        kept.append(tls_socket)
        return tls_socket

    monkeypatch.setattr(transport_module, "_connected", keeping)
    return kept


def test_post_when_the_answer_was_cut_at_the_cap_should_have_closed_its_connection_when_it_returns(
    server, transport, kept_sockets
):
    # Arrange: the server stays on the connection, most of its answer unread, until the client closes it.
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, LARGE_ANSWER, awaits_close=True)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=MAX_BODY_BYTES)
    closed = server.client_gone.wait(CLOSE_SEEN_WITHIN_S)

    # Assert: closed by the transport itself, since its socket is still alive here.
    assert len(answer.body) < LARGE_ANSWER_BYTES
    assert len(kept_sockets) == 1
    assert closed


# --- the whole exchange ends within the time-out ------------------------------------------------

# Each answer is still trickling when the time-out passes, and no single read of it ever waits that long.
# The bodies of the first pair belong to answers that leave the connection open; those of the second to
# answers that say `Connection: close`, which a server does for a request that asked for it.
TRICKLING_ANSWERS = [
    pytest.param(Reply(HTTP_OK, {FILLER_HEADER: FILLER}, ANSWER_BODY, trickle=Trickle.HEADERS), id="in-its-headers"),
    pytest.param(
        Reply(HTTP_OK, {"Content-Length": "0", FILLER_HEADER: FILLER}, framing=Framing.NONE, trickle=Trickle.HEADERS),
        id="in-its-headers-after-a-content-length-of-zero",
    ),
    pytest.param(
        Reply(HTTP_OK, {}, TRICKLED_BODY, trickle=Trickle.BODY, says_close=False), id="in-a-body-of-announced-length"
    ),
    pytest.param(
        Reply(HTTP_OK, {}, TRICKLED_BODY, framing=Framing.CHUNKED, trickle=Trickle.BODY, says_close=False),
        id="in-a-chunked-body",
    ),
    pytest.param(
        Reply(HTTP_OK, {}, TRICKLED_BODY, trickle=Trickle.BODY),
        id="in-a-body-of-announced-length-after-connection-close",
    ),
    pytest.param(
        Reply(HTTP_OK, {}, TRICKLED_BODY, framing=Framing.CHUNKED, trickle=Trickle.BODY),
        id="in-a-chunked-body-after-connection-close",
    ),
]


@pytest.mark.parametrize("reply", TRICKLING_ANSWERS)
def test_post_when_the_answer_still_trickles_at_the_time_out_should_be_cut_there_as_deadline_passed(
    server, transport, reply
):
    # Arrange
    server.replies[FIRST_PATH] = reply
    started = time.monotonic()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH), timeout_s=DEADLINE_S)
    waited_s = time.monotonic() - started

    # Assert: cut at the time-out, not when the server gave up trickling.
    assert str(failure) == DEADLINE_PASSED
    assert waited_s < CUT_WITHIN_S
    _assert_shows_nothing_of_the_request(failure, server.port)


# Slow answers that end by themselves: each takes a good part of its POST's time-out, and none all of it.
TRICKLES_THAT_END_IN_TIME = [
    pytest.param(Reply(HTTP_OK, {FILLER_HEADER: SHORT_FILLER}, ANSWER_BODY, trickle=Trickle.HEADERS), id="in-its-headers"),
    pytest.param(Reply(HTTP_OK, {}, SOON_TRICKLED_BODY, trickle=Trickle.BODY), id="in-its-body"),
]


@pytest.mark.parametrize("reply", TRICKLES_THAT_END_IN_TIME)
def test_post_when_the_answer_trickles_and_ends_within_the_time_out_should_return_all_of_it(server, transport, reply):
    # Arrange
    server.replies[FIRST_PATH] = reply

    # Act
    answer = _post(transport, server.url(FIRST_PATH), timeout_s=SLOW_ANSWER_TIMEOUT_S)

    # Assert: the time-out bounds the exchange, it does not shorten it.
    assert (answer.status, answer.body) == (HTTP_OK, reply.body)


def test_post_when_the_answer_is_whole_only_after_the_time_out_should_raise_deadline_passed(
    server, transport, monkeypatch
):
    # Arrange: the transport's reader takes the whole answer, then outlasts the time-out before it hands it on.
    read_answer = transport_module._read_answer

    def late(reader: BinaryIO, max_body_bytes: int) -> HttpAnswer:
        answer = read_answer(reader, max_body_bytes)
        time.sleep(CUT_WITHIN_S)
        return answer

    monkeypatch.setattr(transport_module, "_read_answer", late)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH), timeout_s=DEADLINE_S)

    # Assert: an answer that comes late is no answer.
    assert str(failure) == DEADLINE_PASSED


@pytest.mark.parametrize("timeout_s", list(UNUSABLE_TIMEOUTS.values()), ids=list(UNUSABLE_TIMEOUTS))
def test_post_when_the_time_out_is_no_positive_finite_number_should_raise_value_error_and_open_no_connection(
    server, transport, timeout_s
):
    # Arrange: a server that would answer at once.
    server.replies[ADDRESS_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    with pytest.raises(ValueError) as raised:
        _post(transport, server.url(ADDRESS_PATH), timeout_s=timeout_s)

    # Assert: a wiring bug, told without the address.
    text = str(raised.value)
    assert [value for value in (LOOPBACK, str(server.port), URL_CANARY) if value in text] == []
    assert server.connections == 0
    assert server.received == []


# What is no cap on an answer's body: the cap is a number of bytes, zero or more.
UNUSABLE_BODY_CAPS = {"minus-one": -1, "minus-two": -2, "true": True, "a-fraction": 4096.5, "a-string": "4096", "none": None}


@pytest.mark.parametrize("max_body_bytes", list(UNUSABLE_BODY_CAPS.values()), ids=list(UNUSABLE_BODY_CAPS))
def test_post_when_the_largest_body_is_no_number_of_bytes_should_raise_value_error_and_open_no_connection(
    server, transport, max_body_bytes
):
    # Arrange: a server that would answer at once.
    server.replies[ADDRESS_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    with pytest.raises(ValueError):
        _post(transport, server.url(ADDRESS_PATH), max_body_bytes=max_body_bytes)

    # Assert: a wiring bug, found before anything is sent.
    assert server.connections == 0
    assert server.received == []


def test_post_when_the_largest_body_is_zero_should_return_one_byte_of_a_body_that_is_not_empty(server, transport):
    # Arrange
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH), max_body_bytes=0)

    # Assert: one byte above the cap, enough for the caller to see an answer too large.
    assert answer.body == ANSWER_BODY[:1]


@pytest.mark.parametrize("body", list(NOT_BYTES.values()), ids=list(NOT_BYTES))
def test_post_when_the_body_is_not_bytes_should_raise_type_error_and_open_no_connection(server, transport, body):
    # Arrange: a server that would answer at once.
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    with pytest.raises(TypeError):
        transport.post(
            server.url(FIRST_PATH), body, REQUEST_SIGNATURE, timeout_s=TIMEOUT_S, max_body_bytes=MAX_BODY_BYTES
        )

    # Assert: a message is signed over its exact bytes, so nothing else is ever sent in their place.
    assert server.connections == 0
    assert server.received == []


# The fixtures of the two peers that take a connection and never finish the TLS handshake.
STALLED_HANDSHAKES = {
    "a-peer-that-says-nothing": "silent_port",
    "a-peer-that-trickles-its-first-record": "trickling_handshake_port",
}


@pytest.fixture(params=list(STALLED_HANDSHAKES.values()), ids=list(STALLED_HANDSHAKES))
def stalled_handshake_port(request: pytest.FixtureRequest) -> int:
    """A port of 127.0.0.1 that takes the connection and never finishes the TLS handshake."""
    return request.getfixturevalue(request.param)


def test_post_when_the_tls_handshake_never_ends_should_raise_deadline_passed_at_the_time_out(
    stalled_handshake_port, transport
):
    # Arrange
    url = f"https://{LOOPBACK}:{stalled_handshake_port}{FIRST_PATH}"
    started = time.monotonic()

    # Act
    failure = _failure(transport, url, timeout_s=DEADLINE_S)
    waited_s = time.monotonic() - started

    # Assert: the handshake did stall, and was cut at the time-out; named by the clock, whichever of the
    # handshake's own time-out and the watchdog ended it.
    assert str(failure) == DEADLINE_PASSED
    assert DEADLINE_S / 2 < waited_s < CUT_WITHIN_S
    _assert_shows_nothing_of_the_request(failure, stalled_handshake_port)


def test_post_when_the_connection_is_taken_late_and_the_handshake_never_ends_should_still_end_at_the_time_out(
    silent_port, transport, network
):
    # Arrange: half the time-out goes by before the address takes the connection.
    network.late[silent_port] = LATE_CONNECT_S
    url = f"https://{LOOPBACK}:{silent_port}{FIRST_PATH}"
    started = time.monotonic()

    # Act
    failure = _failure(transport, url, timeout_s=LATE_TIMEOUT_S)
    waited_s = time.monotonic() - started

    # Assert: the handshake had what the connect left of the time-out, not a whole time-out of its own.
    assert str(failure) == DEADLINE_PASSED
    assert LATE_CONNECT_S < waited_s < LATE_CUT_WITHIN_S
    _assert_shows_nothing_of_the_request(failure, silent_port)


# --- the addresses of a host share the time-out -------------------------------------------------


def test_post_when_the_first_address_of_the_host_refuses_should_reach_the_next(server, transport, network):
    # Arrange
    network.resolves_to(_closed_port(), server.port)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert network.resolved == [LOOPBACK]


def test_post_when_no_socket_can_be_made_for_the_first_address_should_reach_the_next(server, transport, network):
    # Arrange: the host's first address is of a family this host cannot open
    unopenable = _closed_port()
    network.unopenable.add(unopenable)
    network.resolves_to(unopenable, server.port)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert: passed over like an address that refuses the connection
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert network.sockets_refused == 1
    assert server.paths() == [FIRST_PATH]


def test_post_when_no_socket_can_be_made_for_any_address_should_raise_transport_failed_named_after_the_error(
    server, transport, network
):
    # Arrange
    unopenable = [_closed_port(), _closed_port()]
    network.unopenable.update(unopenable)
    network.resolves_to(*unopenable)

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: every address was tried, and the failure goes by its error's class, as a refused connection does
    assert str(failure) == NO_SOCKET
    assert network.sockets_refused == len(unopenable)
    assert server.connections == 0
    _assert_shows_nothing_of_the_request(failure, server.port)


def test_post_when_the_first_address_does_not_answer_should_leave_the_next_its_part_of_the_time_out(
    server, transport, network
):
    # Arrange
    unanswering = _closed_port()
    network.silent.add(unanswering)
    network.resolves_to(unanswering, server.port)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)
    started = time.monotonic()

    # Act
    answer = _post(transport, server.url(FIRST_PATH), timeout_s=SHARED_TIMEOUT_S)
    waited_s = time.monotonic() - started

    # Assert: the address that does not answer was given half the time-out at most, the other the rest.
    (given_s,) = network.given_s
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert 0 < given_s <= SHARED_TIMEOUT_S / 2
    assert waited_s < SHARED_TIMEOUT_S


def test_post_when_no_address_of_the_host_answers_should_raise_deadline_passed_within_about_one_time_out(
    transport, network
):
    # Arrange
    unanswering = [_closed_port(), _closed_port()]
    network.silent.update(unanswering)
    network.resolves_to(*unanswering)
    url = f"https://{LOOPBACK}:{unanswering[0]}{FIRST_PATH}"
    started = time.monotonic()

    # Act
    failure = _failure(transport, url, timeout_s=SHARED_TIMEOUT_S)
    waited_s = time.monotonic() - started

    # Assert: both were tried, in one time-out between them and not one each.
    assert str(failure) == DEADLINE_PASSED
    assert len(network.given_s) == len(unanswering)
    assert waited_s < BOTH_SILENT_WITHIN_S
    _assert_shows_nothing_of_the_request(failure, unanswering[0])


def test_post_when_the_host_has_no_address_should_raise_no_address(server, transport, network):
    # Arrange
    network.resolves_to()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    assert str(failure) == NO_ADDRESS
    assert server.connections == 0


def test_post_when_the_host_cannot_be_resolved_should_raise_transport_failed_naming_only_a_class(
    server, transport, network
):
    # Arrange
    network.resolution_fails()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert: the resolver's own words stay out of it.
    _assert_shows_nothing_of_the_request(failure, server.port)
    assert network.resolved == [LOOPBACK]
    assert server.connections == 0


# --- the signature is one header ----------------------------------------------------------------


@pytest.mark.parametrize("signature", list(NOT_HEADER_VALUES.values()), ids=list(NOT_HEADER_VALUES))
def test_post_when_the_signature_is_not_a_header_value_should_raise_value_error_and_open_no_connection(
    server, transport, signature
):
    # Arrange: a server that would answer at once.
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    with pytest.raises(ValueError) as raised:
        transport.post(
            server.url(FIRST_PATH), REQUEST_BODY, signature, timeout_s=TIMEOUT_S, max_body_bytes=MAX_BODY_BYTES
        )

    # Assert: a wiring bug, told without the value it refuses.
    text = str(raised.value)
    assert [value for value in (repr(signature), REQUEST_SIGNATURE, INJECTED_HEADER) if value in text] == []
    assert server.connections == 0
    assert server.received == []


# --- the environment changes nothing ------------------------------------------------------------


def test_post_when_the_environment_names_a_proxy_where_nothing_listens_should_still_reach_the_server(
    server, self_signed, monkeypatch, no_proxy_exemption
):
    # Arrange
    dead_proxy = f"http://{LOOPBACK}:{_closed_port()}"
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, dead_proxy)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(_transport_trusting(self_signed), server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert server.paths() == [FIRST_PATH]


def test_plain_requests_when_the_environment_names_that_dead_proxy_should_fail(
    server, self_signed, monkeypatch, no_proxy_exemption
):
    # Arrange: the control of the test above: a client that does read the environment cannot reach the server.
    dead_proxy = f"http://{LOOPBACK}:{_closed_port()}"
    for variable in PROXY_VARIABLES:
        monkeypatch.setenv(variable, dead_proxy)

    # Act / Assert
    with pytest.raises(requests.ConnectionError):
        requests.post(
            server.url(FIRST_PATH), data=REQUEST_BODY, timeout=TIMEOUT_S, verify=str(self_signed.certificate)
        )
    assert server.paths() == []


def _write_netrc(directory: Path) -> Path:
    """A netrc file that holds a login for 127.0.0.1."""
    netrc = directory / "netrc"
    netrc.write_text(f"machine {LOOPBACK} login {NETRC_LOGIN} password {NETRC_PASSWORD}\n", encoding="ascii")
    netrc.chmod(OWNER_ONLY)
    return netrc


def test_post_when_the_environment_names_a_netrc_with_a_login_for_the_server_should_send_no_authorization(
    server, self_signed, monkeypatch, tmp_path
):
    # Arrange
    monkeypatch.setenv(NETRC_VARIABLE, str(_write_netrc(tmp_path)))

    # Act
    _post(_transport_trusting(self_signed), server.url(FIRST_PATH))

    # Assert
    (request,) = server.received
    assert "authorization" not in request.headers


def test_plain_requests_when_the_environment_names_that_netrc_should_send_its_login(
    server, self_signed, monkeypatch, tmp_path, no_proxy
):
    # Arrange: the control of the test above: a client that does read the environment sends the login.
    monkeypatch.setenv(NETRC_VARIABLE, str(_write_netrc(tmp_path)))

    # Act
    requests.post(server.url(FIRST_PATH), data=REQUEST_BODY, timeout=TIMEOUT_S, verify=str(self_signed.certificate))

    # Assert
    (request,) = server.received
    assert request.headers["authorization"].startswith("Basic ")


def test_post_when_the_environment_names_a_tls_key_log_should_write_no_key_to_it(
    server, self_signed, monkeypatch, tmp_path
):
    # Arrange: the variable is set before the transport's own context is built.
    key_log = tmp_path / "tls-keys.log"
    monkeypatch.setenv(KEY_LOG_VARIABLE, str(key_log))
    transport = _transport_trusting(self_signed)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert not key_log.exists() or key_log.read_bytes() == b""


def test_a_context_that_reads_the_environment_when_it_names_a_tls_key_log_should_write_the_keys_to_it(
    server, self_signed, monkeypatch, tmp_path
):
    # Arrange: the control of the test above: the standard library's default context honours the variable,
    # so the same POST to the same server leaves the session's keys in the file.
    key_log = tmp_path / "tls-keys.log"
    monkeypatch.setenv(KEY_LOG_VARIABLE, str(key_log))
    transport = HttpsTransport(tls_context=ssl.create_default_context(cafile=str(self_signed.certificate)))

    # Act
    _post(transport, server.url(FIRST_PATH))

    # Assert
    assert key_log.read_bytes() != b""


# --- the server's certificate -------------------------------------------------------------------


def _name_as_a_ca(monkeypatch: pytest.MonkeyPatch, variable: str, identity: SelfSigned) -> None:
    """Make `variable` name the certificate the way that variable is read: a directory for one, the file for the rest."""
    directory = variable == OPENSSL_CA_DIRECTORY_VARIABLE
    monkeypatch.setenv(variable, str(identity.hashed_directory if directory else identity.certificate))


def test_post_when_the_servers_certificate_is_self_signed_should_raise_transport_failed_before_sending(server):
    # Arrange: the transport as the worker builds it, trusting certifi's bundle and nothing else.
    transport = HttpsTransport()

    # Act
    failure = _failure(transport, server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, server.port)
    assert server.received == []


def test_post_when_the_environment_names_that_certificate_as_a_ca_should_still_not_trust_it(
    server, self_signed, monkeypatch
):
    # Arrange: every variable at once, set before the transport's own context is built.
    for variable in (*REQUESTS_CA_VARIABLES, *OPENSSL_CA_VARIABLES):
        _name_as_a_ca(monkeypatch, variable, self_signed)

    # Act
    failure = _failure(HttpsTransport(), server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, server.port)
    assert server.received == []


@pytest.mark.parametrize("variable", REQUESTS_CA_VARIABLES)
def test_plain_requests_when_the_environment_names_that_certificate_as_a_ca_bundle_should_trust_it(
    server, self_signed, monkeypatch, no_proxy, no_ca_in_the_environment, variable
):
    # Arrange: a control of the two tests above: the certificate is right for this server, and a client
    # that does read the environment trusts it. Only the missing trust makes the transport fail.
    _name_as_a_ca(monkeypatch, variable, self_signed)
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    response = requests.post(server.url(FIRST_PATH), data=REQUEST_BODY, timeout=TIMEOUT_S)

    # Assert
    assert (response.status_code, response.content) == (HTTP_OK, ANSWER_BODY)
    assert server.paths() == [FIRST_PATH]


@pytest.mark.parametrize("variable", OPENSSL_CA_VARIABLES)
def test_a_context_that_reads_the_environment_when_it_names_that_certificate_as_a_ca_should_trust_it(
    server, self_signed, monkeypatch, no_ca_in_the_environment, variable
):
    # Arrange: the other control: the standard library's default context loads the CAs these two
    # variables name, and with it the same transport reaches the same server.
    _name_as_a_ca(monkeypatch, variable, self_signed)
    transport = HttpsTransport(tls_context=ssl.create_default_context())
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)


def test_post_when_the_certificate_names_another_address_should_refuse_it_though_its_issuer_is_trusted(
    misnamed_server, named_for_another_address
):
    # Arrange
    transport = _transport_trusting(named_for_another_address)

    # Act
    failure = _failure(transport, misnamed_server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, misnamed_server.port)
    assert misnamed_server.received == []


def test_a_context_that_checks_no_name_when_the_certificate_names_another_address_should_reach_the_server(
    misnamed_server, named_for_another_address
):
    # Arrange: the control of the test above: the certificate is trusted, and only the address it names
    # stands between the transport and this server.
    context = _trusting(named_for_another_address.certificate)
    context.check_hostname = False
    misnamed_server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(HttpsTransport(tls_context=context), misnamed_server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)


# --- the transport as the worker builds it: no context given ------------------------------------


def test_post_when_the_transport_is_built_without_a_context_should_trust_what_the_bundle_of_certifi_holds(
    server, self_signed, monkeypatch
):
    # Arrange: the bundle is the test's certificate when the transport is built.
    _as_the_bundle_of_certifi(monkeypatch, self_signed)
    transport = HttpsTransport()
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)


def test_post_when_the_transport_is_built_without_a_context_should_refuse_a_certificate_that_names_another_address(
    misnamed_server, named_for_another_address, monkeypatch
):
    # Arrange: the bundle holds the server's own certificate, so only the address it names can refuse it.
    _as_the_bundle_of_certifi(monkeypatch, named_for_another_address)
    transport = HttpsTransport()

    # Act
    failure = _failure(transport, misnamed_server.url(FIRST_PATH))

    # Assert
    _assert_shows_nothing_of_the_request(failure, misnamed_server.port)
    assert misnamed_server.received == []


def test_post_when_the_transport_is_built_without_a_context_should_write_no_key_to_the_tls_key_log(
    server, self_signed, monkeypatch, tmp_path
):
    # Arrange: the environment names a key log before the transport is built.
    key_log = tmp_path / "tls-keys.log"
    monkeypatch.setenv(KEY_LOG_VARIABLE, str(key_log))
    _as_the_bundle_of_certifi(monkeypatch, self_signed)
    transport = HttpsTransport()
    server.replies[FIRST_PATH] = Reply(HTTP_OK, {}, ANSWER_BODY)

    # Act
    answer = _post(transport, server.url(FIRST_PATH))

    # Assert
    assert (answer.status, answer.body) == (HTTP_OK, ANSWER_BODY)
    assert not key_log.exists() or key_log.read_bytes() == b""


# --- the context a transport is built with ------------------------------------------------------

# What is no TLS context, each a way to wire a transport wrongly: a transport that took one of these as
# it is would fail every POST as if the Hub never answered.
NOT_TLS_CONTEXTS = {
    "the-name-of-a-ca-file": "ca-bundle.pem",
    "true": True,
    "false": False,
    "zero": 0,
    "the-function-that-makes-one": verifying_tls_context,
    "the-class-of-one": ssl.SSLContext,
}


@pytest.mark.parametrize("tls_context", list(NOT_TLS_CONTEXTS.values()), ids=list(NOT_TLS_CONTEXTS))
def test_https_transport_when_built_with_what_is_no_tls_context_should_raise_type_error(tls_context):
    # Act / Assert
    with pytest.raises(TypeError):
        HttpsTransport(tls_context=tls_context)


# --- what an answer shows -----------------------------------------------------------------------


def test_http_answer_when_shown_with_repr_should_show_neither_its_body_nor_its_signature():
    # Arrange
    answer = HttpAnswer(status=HTTP_OK, body=REQUEST_BODY, signature=ANSWER_SIGNATURE)

    # Act
    shown = repr(answer)

    # Assert
    assert str(HTTP_OK) in shown
    assert [value for value in (CANARY, ANSWER_SIGNATURE, ANSWER_SIGNATURE[-SIGNATURE_HEX_DIGITS:]) if value in shown] == []
