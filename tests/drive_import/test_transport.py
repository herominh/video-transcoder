"""The acquire's store PUT (core/drive_import/store.py) through the shared session, against the loopback server of
tests/net/test_transport.py, whose tests hold the session itself. Acquire code: left out on a Python below 3.11
(tests/conftest.py)."""

from __future__ import annotations

import time

import pytest

from core.drive_import import store
from core.drive_import.failure import Code, ImportFailed, Stage
from tests.net.test_transport import (  # noqa: F401 - session, server_tls and tls are the fixtures the test uses
    ENDED_WITHIN_S, PAYLOAD, _loopback_server, _near_stop_line, server_tls, session, tls,
)


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
