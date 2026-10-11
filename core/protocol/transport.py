"""The HTTP carrier of contract v2 (README section 12): a message is the body, its signature the
`X-VH-Signature` header, of the request for what the worker sends and of the response for a claim grant.

One POST and its answer, nothing more, written on sockets directly so that each of these holds by
construction, not by an HTTP library's defaults or leniency:

- https only: a claim body can fetch the media key and the answer can carry it (README 15.1, rule 2), so
  an address that is not a plain https URL (visible ASCII, a host, no userinfo, a port of 1 or more) is a
  wiring bug (ValueError), whatever the settings let through;
- the server's certificate and host name are verified against the CA bundle of the `certifi` package, and
  nothing is read from the environment: no proxy, no other CA file, no netrc, no TLS key log;
- no redirect is followed and no `Location` is read: a 3xx is an answer like any other;
- the body is the bytes as they were sent: no content encoding is asked for and none is decoded (a message
  is signed over its exact bytes), so a compressed answer simply does not verify;
- the body is cut just above `max_body_bytes` and the head at MAX_HEAD_BYTES, so an oversized answer is
  refused by its size, never held in memory;
- an answer is whole and plain or it is no answer. Its head is a status line and header lines of HTTP/1.x,
  each ended by CRLF, closed by an empty line; its body is delimited by one plain `Content-Length` or by
  `Transfer-Encoding: chunked` alone, with chunk sizes in plain hexadecimal and nothing after the last
  chunk. A head or a body cut short, a body delimited only by the end of the connection or in two ways,
  an interim answer (a 1xx), and any line that is not exactly of that grammar are all TransportFailed, so
  that a cut or garbled connection is never mistaken for what the Hub said;
- the whole exchange ends within `timeout_s`, counted from one deadline: the addresses of the host share
  what is left of it for the TCP connect (a first address that does not answer leaves the others their
  part), and from the TLS handshake on a watchdog shuts the socket at the deadline, so that a handshake
  or an answer that trickles is cut like one that hangs. Name resolution is the one step nothing here
  can cut short: a POST that outlasts its time-out there fails as soon as the name resolves;
- no retry: the caller owns that.

No body, of a request or of an answer, is ever logged or put in an error (README 15.1, rule 5): a failure
shows a name, nothing else.
"""

from __future__ import annotations

import contextlib
import math
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO, Mapping, Protocol
from urllib.parse import urlsplit

import certifi

SIGNATURE_HEADER = "X-VH-Signature"
CONTENT_TYPE = "application/json"
NO_CONTENT_ENCODING = "identity"
USER_AGENT = "video-transcoder-worker"
HTTPS_SCHEME = "https"
HTTPS_PORT = 443
CHUNKED = "chunked"
LINE_END = b"\r\n"
MAX_HEAD_BYTES = 32 * 1024
MAX_HEADER_LINES = 64
FAILURE_DEADLINE_PASSED = "DeadlinePassed"
FAILURE_NO_ADDRESS = "NoAddress"
FAILURE_INCOMPLETE_HEAD = "IncompleteHead"
FAILURE_OVERSIZED_HEAD = "OversizedHead"
FAILURE_MALFORMED_ANSWER = "MalformedAnswer"
FAILURE_NO_FINAL_ANSWER = "NoFinalAnswer"
FAILURE_AMBIGUOUS_FRAMING = "AmbiguousFraming"
FAILURE_UNFRAMED_BODY = "UnframedBody"
FAILURE_INCOMPLETE_BODY = "IncompleteBody"
FIRST_FINAL_STATUS = 200
# Statuses whose answer has no body by definition (RFC 9110): such an answer announces no chunks.
_BODILESS_STATUSES = frozenset({204, 304})
# What this transport puts in a request line or a header: visible ASCII, so no space, line break or fold.
_VISIBLE_ASCII = re.compile(r"[\x21-\x7e]+")
_STATUS_LINE = re.compile(rb"HTTP/1\.[01] ([1-9][0-9]{2})(?: [\x09\x20-\x7e\x80-\xff]*)?")
# A header line is a name (a token), a colon and a value (visible characters, spaces and tabs). Each part
# has a pattern of its own and the white space around a value is stripped in code: one pattern for the
# whole line would have to guess where that white space ends, and backtrack over every guess.
_HEADER_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_HEADER_VALUE = re.compile(rb"[\x09\x20-\x7e\x80-\xff]*")
_HEADER_WHITE_SPACE = b"\t "
_PLAIN_LENGTH = re.compile(r"[0-9]{1,18}")
# A chunk size in plain hexadecimal: no sign, no prefix, no extension; 8 digits hold any body this reads.
_CHUNK_SIZE = re.compile(rb"[0-9A-Fa-f]{1,8}")
_CHUNK_SIZE_LINE_BYTES = 8 + len(LINE_END)
_HEADER_CHARSET = "latin-1"


class TransportFailed(Exception):
    """The request got no whole, plain answer (connection, TLS, time-out, a cut or garbled answer); the text is a name only."""


@dataclass(frozen=True)
class HttpAnswer:
    """What came back: the status, the body's bytes (at most `max_body_bytes` plus one) and the signature header."""

    status: int
    body: bytes = field(repr=False)
    signature: str | None = field(repr=False)


class Transport(Protocol):
    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        """POST the signed message; raise TransportFailed when no whole answer comes within `timeout_s`."""


def verifying_tls_context() -> ssl.SSLContext:
    """Certificates and host names verified against certifi's bundle; nothing taken from the environment."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # verifies the certificate and the host name by default
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(cafile=certifi.where())
    return context


class _CutShort(Exception):
    """The exchange ended without a whole, plain answer; the text is the failure's name."""


class _Watchdog:
    """Shuts the socket it holds `seconds` after it starts, whatever the exchange is doing at that moment."""

    def __init__(self, seconds: float) -> None:
        self._held: socket.socket | None = None
        self._timer = threading.Timer(seconds, self._cut)
        self._timer.daemon = True
        self.fired = False

    def hold(self, sock: socket.socket) -> None:
        self._held = sock

    def _cut(self) -> None:
        self.fired = True
        sock = self._held
        if sock is not None:
            # The plain socket's own shutdown, so no TLS state is touched from this thread; a socket that is
            # already closed has nothing left to cut.
            with contextlib.suppress(OSError):
                socket.socket.shutdown(sock, socket.SHUT_RDWR)

    def __enter__(self) -> _Watchdog:
        self._timer.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._timer.cancel()


@dataclass(frozen=True)
class _Target:
    host: str
    port: int
    path_and_query: str

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == HTTPS_PORT else f"{host}:{self.port}"


class HttpsTransport:
    """The real carrier. `tls_context` is for tests that serve from 127.0.0.1 under a certificate of their own."""

    def __init__(self, tls_context: ssl.SSLContext | None = None) -> None:
        self._tls_context = tls_context if tls_context is not None else verifying_tls_context()

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        target = _https_target(url)
        if not isinstance(body, bytes):
            raise TypeError("the body of a POST is the message's exact bytes")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < math.inf:
            raise ValueError("the time-out of a POST is a positive, finite number of seconds")
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes < 0:
            raise ValueError("the largest body of an answer is a number of bytes, zero or more")
        if not isinstance(signature, str) or _VISIBLE_ASCII.fullmatch(signature) is None:
            raise ValueError("the signature is not a header value")
        request = _request_bytes(target, body, signature)
        deadline = time.monotonic() + timeout_s
        watchdog = _Watchdog(timeout_s)
        failure: str
        try:
            with watchdog:
                return _exchange(self._tls_context, watchdog, deadline, target, request, max_body_bytes)
        except _CutShort as cut:
            failure = str(cut)
        except Exception as error:  # noqa: BLE001 - whatever fails while talking to the peer is no answer
            failure = type(error).__name__
        if watchdog.fired or time.monotonic() >= deadline:
            # By the clock, not by which of the socket's own time-out and the watchdog happened to win.
            failure = FAILURE_DEADLINE_PASSED
        # Raised outside the handlers: such an error can quote the address and keeps what it failed on.
        raise TransportFailed(failure)


def _https_target(url: str) -> _Target:
    """Where an https URL points; ValueError (never quoting it) for anything else."""
    target = _split_https(url)
    if target is None:
        raise ValueError("the transport posts to plain https addresses only")
    return target


def _split_https(url: object) -> _Target | None:
    """None for what is no plain https URL; the parser's own errors quote the address, so none leaves here."""
    # Checked on the address as given: the parser silently removes a tab or a line break.
    if not isinstance(url, str) or _VISIBLE_ASCII.fullmatch(url) is None:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    has_userinfo = parts.username is not None or parts.password is not None
    if parts.scheme != HTTPS_SCHEME or not host or has_userinfo or (port is not None and port < 1):
        return None
    path_and_query = parts.path or "/"
    if parts.query:
        path_and_query = f"{path_and_query}?{parts.query}"
    return _Target(host=host, port=port if port is not None else HTTPS_PORT, path_and_query=path_and_query)


def _request_bytes(target: _Target, body: bytes, signature: str) -> bytes:
    """The whole request: every part is visible ASCII already, so nothing can break a line."""
    head = (
        f"POST {target.path_and_query} HTTP/1.1\r\n"
        f"Host: {target.host_header}\r\n"
        f"User-Agent: {USER_AGENT}\r\n"
        f"Content-Type: {CONTENT_TYPE}\r\n"
        f"Accept: {CONTENT_TYPE}\r\n"
        f"Accept-Encoding: {NO_CONTENT_ENCODING}\r\n"
        f"Connection: close\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{SIGNATURE_HEADER}: {signature}\r\n"
        f"\r\n"
    )
    return head.encode("ascii") + body


def _time_left(deadline: float) -> float:
    """Seconds until the deadline; the exchange is cut short when none is left."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise _CutShort(FAILURE_DEADLINE_PASSED)
    return left


def _exchange(
    tls_context: ssl.SSLContext,
    watchdog: _Watchdog,
    deadline: float,
    target: _Target,
    request: bytes,
    max_body_bytes: int,
) -> HttpAnswer:
    tls_socket = _connected(tls_context, watchdog, deadline, target)
    try:
        tls_socket.sendall(request)
        with tls_socket.makefile("rb") as reader:
            answer = _read_answer(reader, max_body_bytes)
        # An answer that is whole only after the deadline came too late, whether or not the watchdog ran yet.
        _time_left(deadline)
        return answer
    finally:
        tls_socket.close()


def _connected(tls_context: ssl.SSLContext, watchdog: _Watchdog, deadline: float, target: _Target) -> ssl.SSLSocket:
    """A TLS connection to the host, its certificate and name verified, made within what the deadline leaves."""
    plain = _tcp_connected(target.host, target.port, deadline)
    try:
        tls_socket = tls_context.wrap_socket(plain, server_hostname=target.host, do_handshake_on_connect=False)
    except BaseException:
        plain.close()
        raise
    try:
        # From here on the watchdog can cut: a handshake that trickles ends at the deadline like one that hangs.
        watchdog.hold(tls_socket)
        tls_socket.settimeout(_time_left(deadline))
        tls_socket.do_handshake()
        tls_socket.settimeout(_time_left(deadline))
    except BaseException:
        tls_socket.close()
        raise
    return tls_socket


def _tcp_connected(host: str, port: int, deadline: float) -> socket.socket:
    """A TCP connection to the first address of the host that takes one.

    Each address gets an equal part of the time left, so that one address that does not answer cannot use
    up what the others need.
    """
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    failure: Exception = _CutShort(FAILURE_NO_ADDRESS)
    for position, (family, kind, protocol, _, address) in enumerate(addresses):
        left = _time_left(deadline)
        sock = socket.socket(family, kind, protocol)
        sock.settimeout(left / (len(addresses) - position))
        try:
            sock.connect(address)
        except OSError as error:
            sock.close()
            failure = error
            continue
        return sock
    raise failure


# --- reading the answer: a strict reader of the little HTTP/1.x an answer needs -------------------------


class _Head:
    """What is left of the head's size allowance while its lines are read."""

    def __init__(self) -> None:
        self.bytes_left = MAX_HEAD_BYTES


def _line(reader: BinaryIO, limit: int, incomplete: str, oversized: str) -> bytes:
    """One line without its CRLF, of at most `limit` bytes with it; cut short otherwise."""
    line = reader.readline(limit)
    if line.endswith(LINE_END):
        return line[: -len(LINE_END)]
    if line.endswith(b"\n"):
        # A line ended by a bare line feed is not of the grammar.
        raise _CutShort(FAILURE_MALFORMED_ANSWER)
    # No line end within the limit is a line too long; no line end before the end is an answer cut short.
    raise _CutShort(oversized if len(line) >= limit else incomplete)


def _head_line(reader: BinaryIO, head: _Head) -> bytes:
    if head.bytes_left <= 0:
        raise _CutShort(FAILURE_OVERSIZED_HEAD)
    line = _line(reader, head.bytes_left, FAILURE_INCOMPLETE_HEAD, FAILURE_OVERSIZED_HEAD)
    head.bytes_left -= len(line) + len(LINE_END)
    return line


def _read_answer(reader: BinaryIO, max_body_bytes: int) -> HttpAnswer:
    head = _Head()
    status_line = _STATUS_LINE.fullmatch(_head_line(reader, head))
    if status_line is None:
        raise _CutShort(FAILURE_MALFORMED_ANSWER)
    status = int(status_line.group(1))
    headers = _read_headers(reader, head)
    if status < FIRST_FINAL_STATUS:
        # An interim answer is not the answer, and no request of this transport asks for one.
        raise _CutShort(FAILURE_NO_FINAL_ANSWER)
    signatures = headers.get(SIGNATURE_HEADER.lower(), [])
    return HttpAnswer(
        status=status,
        body=_read_body(reader, status, headers, max_body_bytes),
        # Two signature headers name no signature: the verification then fails on its own.
        signature=signatures[0] if len(signatures) == 1 else None,
    )


def _read_headers(reader: BinaryIO, head: _Head) -> dict[str, list[str]]:
    """Every header's values by its lower-case name, up to the empty line that closes the head."""
    headers: dict[str, list[str]] = {}
    for _ in range(MAX_HEADER_LINES + 1):
        line = _head_line(reader, head)
        if not line:
            return headers
        name, colon, value = line.partition(b":")
        if not colon or _HEADER_NAME.fullmatch(name) is None or _HEADER_VALUE.fullmatch(value) is None:
            raise _CutShort(FAILURE_MALFORMED_ANSWER)
        stripped = value.strip(_HEADER_WHITE_SPACE)
        headers.setdefault(name.decode(_HEADER_CHARSET).lower(), []).append(stripped.decode(_HEADER_CHARSET))
    raise _CutShort(FAILURE_OVERSIZED_HEAD)


def _read_body(reader: BinaryIO, status: int, headers: Mapping[str, list[str]], max_body_bytes: int) -> bytes:
    """The body, delimited in exactly one plain way; at most `max_body_bytes` plus one byte of it."""
    lengths = headers.get("content-length", [])
    encodings = headers.get("transfer-encoding", [])
    bodiless = status in _BODILESS_STATUSES
    if lengths and (len(lengths) != 1 or _PLAIN_LENGTH.fullmatch(lengths[0]) is None):
        raise _CutShort(FAILURE_AMBIGUOUS_FRAMING)
    if encodings and ([encoding.lower() for encoding in encodings] != [CHUNKED] or lengths or bodiless):
        raise _CutShort(FAILURE_AMBIGUOUS_FRAMING)
    if bodiless:
        return b""
    if encodings:
        return _read_chunked(reader, max_body_bytes)
    if not lengths:
        raise _CutShort(FAILURE_UNFRAMED_BODY)
    length = int(lengths[0])
    # One byte above the cap is enough for the caller to see an oversized answer.
    return _read_exactly(reader, min(length, max_body_bytes + 1))


def _read_exactly(reader: BinaryIO, count: int) -> bytes:
    data = reader.read(count)
    if len(data) != count:
        raise _CutShort(FAILURE_INCOMPLETE_BODY)
    return data


def _read_chunked(reader: BinaryIO, max_body_bytes: int) -> bytes:
    body = bytearray()
    while True:
        size_line = _line(reader, _CHUNK_SIZE_LINE_BYTES, FAILURE_INCOMPLETE_BODY, FAILURE_MALFORMED_ANSWER)
        if _CHUNK_SIZE.fullmatch(size_line) is None:
            raise _CutShort(FAILURE_MALFORMED_ANSWER)
        size = int(size_line, 16)
        if size == 0:
            # No trailer: the empty line that ends the body comes right after the last chunk, and a line
            # of two bytes can only be that empty line.
            _line(reader, len(LINE_END), FAILURE_INCOMPLETE_BODY, FAILURE_MALFORMED_ANSWER)
            return bytes(body)
        body += _read_exactly(reader, min(size, max_body_bytes + 1 - len(body)))
        if len(body) > max_body_bytes:
            # Oversized: what was read is enough for the caller to refuse it, and nothing more is read.
            return bytes(body)
        if _read_exactly(reader, len(LINE_END)) != LINE_END:
            raise _CutShort(FAILURE_MALFORMED_ANSWER)
