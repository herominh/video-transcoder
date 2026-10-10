"""The HTTP carrier of contract v2 (README section 12): a message is the body, its signature the
`X-VH-Signature` header, of the request for what the worker sends and of the response for a claim grant.

One POST and its answer, nothing more: no retry (the caller owns that), no redirect (the claim route never
redirects, and following one would send a claim body somewhere the configuration does not name; a 3xx is
an answer like any other, its `Location` never read), the server's certificate always verified, the environment's proxy, CA and netrc settings ignored, and the
answer's body cut just above the largest message the contract defines, so that an oversized answer is
refused by its size, never held in memory. The body is the bytes as they were sent: the request asks for
no content encoding and none is ever decoded (a message is signed over its exact bytes, and a decoder
would inflate an answer before any cap), so a compressed answer simply does not verify. No body, of a
request or of an answer, is ever logged or put in an error (README 15.1, rule 5). A response that
trickles is bounded per read, not as a whole: the provider's execution time-out is the outer bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import requests
from urllib3.exceptions import HTTPError as Urllib3Error

SIGNATURE_HEADER = "X-VH-Signature"
CONTENT_TYPE = "application/json"
NO_CONTENT_ENCODING = "identity"
CONNECT_TIMEOUT_S = 5.0
_READ_CHUNK_BYTES = 4096


class TransportFailed(Exception):
    """The request got no answer (connection, TLS or time-out); the text is the failure's class name only."""


@dataclass(frozen=True)
class HttpAnswer:
    """What came back: the status, the body's bytes (at most `max_body_bytes` plus one chunk) and the signature header."""

    status: int
    body: bytes = field(repr=False)
    signature: str | None = field(repr=False)


class Transport(Protocol):
    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        """POST the signed message; raise TransportFailed when no answer comes."""


class _NoRedirectSession(requests.Session):
    """A session to which no answer is a redirect.

    With `allow_redirects=False` requests still prepares the redirect's next request: it parses `Location`
    (raising for a malformed one) and reads the whole body, decoded, to free the connection. A redirect is
    never followed here, so its target is never looked at and its body is read like any other, cut and undecoded.
    """

    def get_redirect_target(self, resp: requests.Response) -> str | None:
        return None


class HttpsTransport:
    """The real carrier, over `requests`."""

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        headers = {
            "Content-Type": CONTENT_TYPE,
            "Accept": CONTENT_TYPE,
            "Accept-Encoding": NO_CONTENT_ENCODING,
            SIGNATURE_HEADER: signature,
        }
        failure: str | None = None
        try:
            with _NoRedirectSession() as session:
                session.trust_env = False
                with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=(min(CONNECT_TIMEOUT_S, timeout_s), timeout_s),
                    allow_redirects=False,
                    verify=True,
                    stream=True,
                ) as response:
                    return HttpAnswer(
                        status=response.status_code,
                        body=_read_capped(response, max_body_bytes),
                        signature=response.headers.get(SIGNATURE_HEADER),
                    )
        except (OSError, Urllib3Error) as error:  # a requests error is an OSError
            failure = type(error).__name__
        # Raised outside the handler: such an error can quote the URL and keeps the request it failed on.
        raise TransportFailed(failure)


def _read_capped(response: requests.Response, max_body_bytes: int) -> bytes:
    """The body's first bytes, undecoded: reading stops with the chunk that passes `max_body_bytes`."""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.raw.stream(_READ_CHUNK_BYTES, decode_content=False):
        chunks.append(chunk)
        size += len(chunk)
        if size > max_body_bytes:
            break
    return b"".join(chunks)
