"""What the import job sends its requests through: a small session interface, its real implementation over
`requests`, and the request body that stops at the stop line.

The job calls only `get(url, headers, timeout, stream, stop_line=...)` and `put(url, data, headers, timeout,
stop_line=...)` and reads only an answer's `status_code`, its `headers` (case-insensitive), `iter_content(chunk_size)`
and `close()`; tests hand it a fake of their own. The real session (RequestsSession) is a `requests.Session` that
reads nothing from the environment (`trust_env=False`: no proxy, no netrc, no other certificate bundle than certifi's)
and follows no redirect: `allow_redirects=False` on every call, and a session whose `get_redirect_target()` names
none, so that requests does not even read a 3xx's body to release its connection (it does that for a 3xx with a
`Location` when it does not follow it). A 3xx comes back to the job as the answer it is, its body unread. A PUT is
sent with `stream=True`, so that its answer's body is never read: the job reads its status and its ETag and closes it.

Every request carries the job's stop line, and the session's watchdog holds every request to it. The session's
connections (a connection class of this module's own, kept by the pools of the session's adapter) give the watchdog
the socket that carries each of them, for as long as that socket lives: the TCP socket of an HTTP connection, and the
TLS socket of an HTTPS one once its handshake is done, both held weakly, so that the watchdog still reaches a socket
http.client handed over to an answer that ends the connection (`Connection: close`, HTTP/1.0, a body framed by the
close); and between an HTTPS connection's TCP connect and the end of its handshake, while the TLS socket takes the TCP
socket's descriptor over, a duplicate of that descriptor, closed once the TLS socket is kept. A timer armed just after
the stop line latches the session's expiry, under the lock that keeps those sockets, and shuts each of them (the
base class's `socket.shutdown()`, from the timer's thread), so that a read or a write blocked on one (a TLS handshake
whose server never answers, a download that trickles, an answer whose head or body never comes) ends at once, and
the job, finding its stop line passed, answers `deadline`. Once the latch is set, a socket kept later is shut as it
is kept, a connection goes no further with its setup after its TCP connect and after its TLS handshake, and the
session starts no request at all (each raises DeadlineReached). A connection connects within the stop line too: it
resolves the host's name once and tries each of its addresses only while the session has not expired, with the
connect time-out cut to what is left of the stop line, so that several addresses share that time instead of each
taking a whole time-out. What nothing cuts short is the name resolution itself, which no time-out of the HTTP
library bounds and which has no socket to shut; it ends where the latch is read next.

A transfer that fails (no connection, a reset, a time-out, an answer cut short, a socket shut at the stop line)
raises an OSError: requests' own errors are OSErrors (RequestException derives from IOError). Their text names the
request's URL, so the job never logs, keeps or re-raises one: it notes that the try failed and goes on outside the
`except` block.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import weakref
from typing import Any, Iterator, Mapping, Protocol

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util.connection import _set_socket_options, allowed_gai_family

from core.drive_import.clock import DeadlineReached, StopLine

# What a failed transfer raises (requests.RequestException is an OSError).
TRANSFER_ERRORS: tuple[type[BaseException], ...] = (OSError,)
# How many bytes of an answer the job reads when it reads one whole (Google's metadata, Google's error document).
SMALL_ANSWER_MAX_BYTES = 65_536
_SMALL_ANSWER_CHUNK_BYTES = 8_192
# The HTTP library makes no request again by itself: every try is the job's.
NO_RETRIES = 0
# How far apart two readings of one stop line may lie and still name the same instant for the watchdog, and how long
# after the stop line the watchdog fires.
_WATCHDOG_SLACK_S = 0.05
_WATCHDOG_AFTER_S = 0.01
# A connect's time-out when no request has named a stop line yet (every request of the job names one).
_NO_STOP_LINE_CONNECT_S = 10.0


class HttpResponse(Protocol):
    """The part of an answer the job reads."""

    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class HttpSession(Protocol):
    """The two requests the job makes. `timeout` is (connect seconds, read seconds); `stop_line` is the job's, which
    the session holds the request to while it is in flight, its answer's body included. Both raise DeadlineReached
    once the session refuses at the stop line (a request asked after it, or a connection whose setup it overtook) and
    when a request body stops at the stop line (BoundedBody), a transfer error (an OSError) for every other request that
    got no answer."""

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        stream: bool,
        *,
        stop_line: StopLine,
    ) -> HttpResponse: ...

    def put(
        self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *, stop_line: StopLine
    ) -> HttpResponse: ...


class ClosableHttpSession(HttpSession, Protocol):
    """A session the entry point makes for one job and closes when the job ends."""

    def close(self) -> None: ...


class _Sockets:
    """What a session's watchdog shuts, and the session's expiry.

    Two kinds of socket are kept, both weakly: the socket object that carries a connection (the TCP socket of an
    HTTP connection; the TLS socket of an HTTPS one, once its handshake is done), for as long as that object lives,
    which is as long as the connection or the answer it handed the socket to (http.client gives an answer that ends
    the connection, `Connection: close`, HTTP/1.0 or a body framed by the close, the socket itself) can read it; and,
    for an HTTPS connection between its TCP connect and the end of its handshake, a duplicate of the TCP socket's
    descriptor, which the TLS socket takes over while the handshake runs. The expiry is latched under the same lock,
    with the stop line's instant, so that no socket kept after the watchdog fired escapes it and no connect starts
    after it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sockets: weakref.WeakSet[socket.socket] = weakref.WeakSet()
        self._expired = False
        self._due: float | None = None

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expired

    def arm(self, due: float) -> None:
        """The stop line's instant, on the monotonic clock: the earliest one a request named."""
        with self._lock:
            self._due = due if self._due is None else min(self._due, due)

    def keep(self, sock: socket.socket) -> None:
        """Watch `sock` for as long as it lives; shut it at once when the session has expired."""
        with self._lock:
            self._sockets.add(sock)
            expired = self._expired
        if expired:
            _shut(sock)

    def keep_duplicate(self, sock: socket.socket) -> socket.socket:
        """A duplicate of `sock`'s descriptor, watched (shut at once when the session has expired) until its caller
        closes it."""
        duplicate = sock.dup()
        self.keep(duplicate)
        return duplicate

    def connect_timeout(self, configured: object) -> float:
        """The connect time-out of the next address: the configured one cut to what is left of the stop line;
        DeadlineReached once the session has expired or nothing is left."""
        with self._lock:
            expired, due = self._expired, self._due
        left = None if due is None else due - time.monotonic()
        if expired or (left is not None and left <= 0):
            raise DeadlineReached()
        bounded = float(configured) if isinstance(configured, (int, float)) else None
        if left is None:
            return bounded if bounded is not None else _NO_STOP_LINE_CONNECT_S
        return left if bounded is None else min(bounded, left)

    def expire(self) -> None:
        """Latch the expiry and shut every socket kept: a thread blocked on one returns at once."""
        with self._lock:
            self._expired = True
            sockets = list(self._sockets)
        for sock in sockets:
            _shut(sock)


def _shut(sock: socket.socket) -> None:
    """Shut `sock`'s connection both ways. The base class's own shutdown, so that a TLS socket's is not called: that
    one lets go of its TLS state first, which a thread still reading it would then meet as an error of another kind."""
    try:
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        # Closed already, or its connection ended: nothing is left to shut.
        return


def _connect_within(connection: HTTPConnection, sockets: _Sockets) -> socket.socket:
    """The TCP socket of `connection`, made as urllib3's create_connection() makes it but held to the stop line: the
    name resolved once (which nothing bounds), then each of its addresses tried only while the session has not
    expired, with the connect time-out cut to what is left of the stop line, so that the addresses share that time
    instead of each taking a whole time-out. Its failures are urllib3's own, without the error they come from, but for a socket that cannot be created
    (EMFILE, ENOBUFS, EAFNOSUPPORT), which is raised as the OSError it is."""
    host = connection._dns_host
    if host.startswith("["):
        host = host.strip("[]")
    try:
        addresses: list[Any] | None = socket.getaddrinfo(
            host, connection.port, allowed_gai_family(), socket.SOCK_STREAM
        )
    except socket.gaierror:
        addresses = None
    if addresses is None:
        raise NameResolutionError(connection.host, connection, socket.gaierror("the host's name did not resolve"))
    timed_out = False
    for family, kind, protocol, _, address in addresses:
        timeout = sockets.connect_timeout(connection.timeout)
        sock = socket.socket(family, kind, protocol)
        try:
            _set_socket_options(sock, connection.socket_options)
            sock.settimeout(timeout)
            if connection.source_address:
                sock.bind(connection.source_address)
            sock.connect(address)
        except TimeoutError:
            sock.close()
            timed_out = True
            continue
        except OSError:
            sock.close()
            continue
        sys.audit("http.client.connect", connection, connection.host, connection.port)
        return sock
    if timed_out:
        raise ConnectTimeoutError(connection, "the connection to the host timed out")
    raise NewConnectionError(connection, "no address of the host took the connection")


def _watched_pools(sockets: _Sockets) -> dict[str, type[HTTPConnectionPool]]:
    """Pool classes whose connections connect within the stop line and keep, through `sockets`, the socket that
    carries each of them, and go no further with their setup once the session has expired."""

    class WatchedHTTPConnection(HTTPConnection):
        def _new_conn(self) -> socket.socket:
            sock = _connect_within(self, sockets)
            # This socket carries the connection for its whole life, the answer's body included.
            sockets.keep(sock)
            if sockets.expired:
                sock.close()
                raise DeadlineReached()
            return sock

    class WatchedHTTPSConnection(HTTPSConnection):
        _handshake_watch: socket.socket | None = None

        def _new_conn(self) -> socket.socket:
            sock = _connect_within(self, sockets)
            # The TLS socket takes this socket's descriptor over during the handshake: a duplicate of it is watched
            # until the TLS socket itself is kept, after the handshake.
            self._handshake_watch = sockets.keep_duplicate(sock)
            if sockets.expired:
                sock.close()
                self._end_handshake_watch()
                raise DeadlineReached()
            return sock

        def connect(self) -> None:
            try:
                super().connect()
                sockets.keep(self.sock)
            finally:
                # Only now is the duplicate not needed: the TLS socket that carries the connection is kept.
                self._end_handshake_watch()
            # The handshake may have ended because the watchdog shut it, or just as the session expired.
            if sockets.expired:
                raise DeadlineReached()

        def _end_handshake_watch(self) -> None:
            watch, self._handshake_watch = self._handshake_watch, None
            if watch is not None:
                watch.close()

    class WatchedHTTPConnectionPool(HTTPConnectionPool):
        ConnectionCls = WatchedHTTPConnection

    class WatchedHTTPSConnectionPool(HTTPSConnectionPool):
        ConnectionCls = WatchedHTTPSConnection

    return {"http": WatchedHTTPConnectionPool, "https": WatchedHTTPSConnectionPool}


class _WatchedAdapter(HTTPAdapter):
    """requests' adapter, its pools making connections whose sockets the session's watchdog knows."""

    def __init__(self, pools: dict[str, type[HTTPConnectionPool]]) -> None:
        # Set before the base class builds its pool manager, which init_poolmanager() then changes.
        self._watched_pool_classes = pools
        super().__init__(max_retries=NO_RETRIES)

    def init_poolmanager(self, *arguments: Any, **named: Any) -> None:
        super().init_poolmanager(*arguments, **named)
        self.poolmanager.pool_classes_by_scheme = self._watched_pool_classes


class _NoRedirectSession(requests.Session):
    """A session that sees no redirect in any answer: requests neither follows one nor reads its body."""

    def get_redirect_target(self, resp: requests.Response) -> str | None:
        return None


class RequestsSession:
    """The real session: one `requests.Session`, nothing read from the environment, no redirect followed, every
    request held to the job's stop line by the session's watchdog.

    `ca_bundle` replaces the certificates a server's is verified against (certifi's) for a test's own server only.
    """

    def __init__(self, ca_bundle: str | None = None) -> None:
        self._sockets = _Sockets()
        self._session = _NoRedirectSession()
        self._session.trust_env = False
        if ca_bundle is not None:
            self._session.verify = ca_bundle
        adapter = _WatchedAdapter(_watched_pools(self._sockets))
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._watchdog_lock = threading.Lock()
        self._watchdog: threading.Timer | None = None
        self._watchdog_due: float | None = None

    def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: tuple[float, float],
        stream: bool,
        *,
        stop_line: StopLine,
    ) -> HttpResponse:
        self._watch(stop_line)
        return self._session.get(url, headers=dict(headers), timeout=timeout, stream=stream, allow_redirects=False)

    def put(
        self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *, stop_line: StopLine
    ) -> HttpResponse:
        self._watch(stop_line)
        return self._session.put(
            url, data=data, headers=dict(headers), timeout=timeout, stream=True, allow_redirects=False
        )

    def close(self) -> None:
        with self._watchdog_lock:
            if self._watchdog is not None:
                self._watchdog.cancel()
            self._watchdog = None
            self._watchdog_due = None
        self._session.close()

    def _watch(self, stop_line: StopLine) -> None:
        """Refuse a request once the session has expired (DeadlineReached); else arm the watchdog at `stop_line`,
        unless it is armed for that instant or an earlier one already. It fires a moment after the stop line, so that
        the job, reading the stop line on its own clock, finds it passed when a request ends there."""
        if self._sockets.expired:
            raise DeadlineReached()
        seconds_left = max(0.0, stop_line.left_s()) + _WATCHDOG_AFTER_S
        due = time.monotonic() + seconds_left
        self._sockets.arm(due)
        with self._watchdog_lock:
            if self._watchdog_due is not None and self._watchdog_due <= due + _WATCHDOG_SLACK_S:
                return
            if self._watchdog is not None:
                self._watchdog.cancel()
            watchdog = threading.Timer(seconds_left, self._sockets.expire)
            watchdog.daemon = True
            self._watchdog, self._watchdog_due = watchdog, due
            watchdog.start()


class BoundedBody:
    """A request body that gives its bytes in the blocks the HTTP library asks for, and raises DeadlineReached
    instead of a block once the stop line has passed, so that a slow upload ends at the stop line.

    It is a file-like object on purpose: urllib3 sends one as blocks of 16 KiB, each written within the socket's
    time-out, where it would send `bytes` with one sendall() that the time-out bounds as a whole. Its length
    (`__len__` less `tell()`) is the Content-Length requests computes; the job sets the same header itself.
    """

    def __init__(self, payload: bytes | bytearray, stop_line: StopLine) -> None:
        self._view = memoryview(payload)
        self._position = 0
        self._stop_line = stop_line

    def read(self, size: int = -1) -> bytes:
        """The next block of at most `size` bytes (all that is left when `size` is negative or None); b"" at the end,
        which is no request to send and so is given whatever the time."""
        if self._position < len(self._view) and self._stop_line.passed():
            raise DeadlineReached()
        end = len(self._view) if size is None or size < 0 else min(len(self._view), self._position + size)
        block = bytes(self._view[self._position:end])
        self._position = end
        return block

    def tell(self) -> int:
        return self._position

    def __len__(self) -> int:
        return len(self._view)


def read_small_answer(response: HttpResponse) -> bytes | None:
    """The whole body of an answer the job reads in full, or None when it is longer than SMALL_ANSWER_MAX_BYTES or
    its transfer failed. It reads at most one chunk past the limit. The limit counts the bytes as the HTTP library
    gives them, decoded, so an answer in a content coding is never handed to it (google.py)."""
    body = bytearray()
    failed = False
    try:
        for chunk in response.iter_content(_SMALL_ANSWER_CHUNK_BYTES):
            body += chunk
            if len(body) > SMALL_ANSWER_MAX_BYTES:
                return None
    except TRANSFER_ERRORS:
        failed = True
    return None if failed else bytes(body)


def close_quietly(response: HttpResponse) -> None:
    """Close an answer. A close that fails changes nothing the job decided, so its error is dropped (its text could
    name the request)."""
    try:
        response.close()
    except TRANSFER_ERRORS:
        return
