"""A scripted answer of the shared HTTP session (core/net/), for the tests of every path that speaks through it: the
Drive acquire's (tests/acquire/, tests/drive_import/fakes.py) and the capability path's (tests/job/).

Not a test module. It imports nothing of either runtime, so it loads on the encode image's Python (3.10) as well as on
the acquire image's (3.12).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from requests.structures import CaseInsensitiveDict


class FakeResponse:
    """An answer: a status, headers (case-insensitive) and a body given as chunks, possibly lazy."""

    def __init__(
        self,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        chunks: Callable[[int], Iterator[bytes]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = CaseInsensitiveDict(dict(headers or {}))
        self._chunks = chunks if chunks is not None else (lambda size: _split(body, size))
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        return self._chunks(chunk_size)

    def close(self) -> None:
        self.closed = True


def _split(body: bytes, size: int) -> Iterator[bytes]:
    for start in range(0, len(body), size):
        yield body[start:start + size]
