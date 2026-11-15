"""The source read through the presigned GET URL of protocol 2.2.0 (core/source_reader.py UrlRangeSource): ranged,
conditional reads, a fresh URL once when the store refuses it, and every other answer failing closed as the S3
reader's do. The store and the Hub are fakes; no network."""

from __future__ import annotations

import pytest

from core.failure import WorkerFailure
from core.net.clock import stop_line_within
from core.protocol.capabilities_client import CapabilityClient
from core.source_reader import UrlRangeSource
from tests.contract.v2_support import FakeClock, uuid_source
from tests.drive_import.fakes import FakeResponse
from tests.job.capability_fakes import (
    SOURCE_ETAG,
    SOURCE_URL_S,
    FakeHub,
    FakeStore,
    capability_claimed,
    capability_settings,
)

SOURCE = bytes(range(256)) * 16


class _Setup:
    def __init__(self, *, etag: str = SOURCE_ETAG, lease_s: int = 900) -> None:
        self.claimed = capability_claimed(request_changes={"/source/size_bytes": len(SOURCE)})
        self.clock = FakeClock()
        self.hub = FakeHub(self.claimed, self.clock, lease_s=lease_s)
        self.store = FakeStore(self.hub, SOURCE, etag=etag)
        self.client = CapabilityClient(
            dispatch=self.claimed.dispatch, grant=self.claimed.grant, runtime_id=self.claimed.runtime_id,
            settings=capability_settings(), transport=self.hub, clock=self.clock, monotonic=self.clock.monotonic,
            sleep=self.clock.sleep, new_message_id=uuid_source(), time_left=lambda: 3600.0,
        )
        self.source = UrlRangeSource(
            self.store, size_bytes=len(SOURCE), etag=SOURCE_ETAG, stop_line=stop_line_within(lambda: 3600.0),
            monotonic=self.clock.monotonic, ask_url=self._ask,
        )

    def _ask(self):
        return self.client.ask(self.client.stage or "probing", source=True).source

    def source_asks(self) -> int:
        return sum(1 for request in self.hub.requests if request["source"])


def _failure(call) -> WorkerFailure:
    with pytest.raises(WorkerFailure) as raised:
        call()
    return raised.value


def test_read_when_the_url_is_granted_should_read_the_exact_range_conditional_on_the_recorded_etag():
    # Arrange
    setup = _Setup()
    # Act
    data = setup.source.read(100, 300)
    # Assert
    assert data == SOURCE[100:400]
    (get,) = setup.store.gets
    assert get["headers"] == {"Range": "bytes=100-399", "If-Match": f'"{SOURCE_ETAG}"', "Accept-Encoding": "identity"}
    assert get["url"] in setup.hub.issued


def test_read_when_a_url_was_handed_to_it_should_use_it_and_ask_for_none():
    # Arrange
    setup = _Setup()
    setup.source.use(setup.client.ask("probing", source=True).source)
    # Act
    setup.source.read(0, 10)
    setup.source.read(10, 10)
    # Assert
    assert setup.source_asks() == 1


def test_read_when_the_store_refuses_the_url_once_should_ask_once_for_a_fresh_one_and_read():
    # Arrange
    setup = _Setup()
    setup.store.get_answers.append(403)
    # Act
    data = setup.source.read(0, 64)
    # Assert
    assert data == SOURCE[:64]
    assert setup.source_asks() == 2
    assert setup.store.gets[0]["url"] != setup.store.gets[1]["url"]


def test_read_when_the_url_went_past_its_time_and_its_fresh_one_is_refused_should_fail_as_source_url_refused():
    # Arrange
    setup = _Setup(lease_s=2 * SOURCE_URL_S)
    setup.source.read(0, 10)
    setup.clock.advance(SOURCE_URL_S)
    setup.store.get_answers.append(403)
    # Act
    failure = _failure(lambda: setup.source.read(10, 10)).failure
    # Assert
    assert failure.code == "source_url_refused"
    assert setup.source_asks() == 2


def test_read_when_the_store_refuses_the_fresh_url_too_should_fail_retryable_as_source_url_refused():
    # Arrange
    setup = _Setup()
    setup.store.get_answers.extend([403, 403])
    # Act
    failure = _failure(lambda: setup.source.read(0, 64)).failure
    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("source_unavailable", "source_url_refused", True)
    assert setup.source_asks() == 2


def test_read_when_the_url_is_past_its_usable_time_should_ask_for_a_fresh_one_before_reading():
    # Arrange: a lease longer than the source URL's usable time.
    setup = _Setup(lease_s=2 * SOURCE_URL_S)
    setup.source.read(0, 10)
    setup.clock.advance(SOURCE_URL_S)
    # Act
    setup.source.read(10, 10)
    # Assert
    assert setup.source_asks() == 2
    assert all(get["url"] in setup.hub.issued for get in setup.store.gets)


def test_read_when_the_object_changed_should_fail_as_source_changed():
    # Arrange
    setup = _Setup(etag="another-etag")
    # Act
    failure = _failure(lambda: setup.source.read(0, 10)).failure
    # Assert
    assert (failure.error_class, failure.code, failure.retryable) == ("source_integrity_mismatch", "source_changed", False)


@pytest.mark.parametrize("answer,code", [
    (FakeResponse(200, {"Content-Length": "10"}, SOURCE[:10]), "range_not_honoured"),
    (FakeResponse(206, {"ETag": f'"{SOURCE_ETAG}"', "Content-Range": f"bytes 0-19/{len(SOURCE)}",
                        "Content-Length": "20"}, SOURCE[:20]), "range_not_honoured"),
    (FakeResponse(206, {"ETag": f'"{SOURCE_ETAG}"', "Content-Range": "bytes 0-9/99", "Content-Length": "10"},
                  SOURCE[:10]), "source_size_mismatch"),
    (FakeResponse(206, {"ETag": '"other"', "Content-Range": f"bytes 0-9/{len(SOURCE)}", "Content-Length": "10"},
                  SOURCE[:10]), "source_changed"),
    (FakeResponse(206, {"ETag": f'"{SOURCE_ETAG}"', "Content-Range": f"bytes 0-9/{len(SOURCE)}",
                        "Content-Length": "10", "Content-Encoding": "gzip"}, SOURCE[:10]), "range_not_honoured"),
    (FakeResponse(206, {"ETag": f'"{SOURCE_ETAG}"', "Content-Range": f"bytes 0-9/{len(SOURCE)}",
                        "Content-Length": "10"}, SOURCE[:20]), "range_not_honoured"),
    (FakeResponse(307, {"Location": "https://elsewhere.test.invalid/x"}), "range_not_honoured"),
    (FakeResponse(412), "source_changed"),
    (FakeResponse(404), "source_missing"),
    (FakeResponse(416), "source_size_mismatch"),
], ids=["a-200", "a-wider-range", "another-size", "another-etag", "a-content-coding", "a-longer-body",
        "a-redirect", "a-412", "a-404", "a-416"])
def test_read_when_the_answer_is_not_the_asked_range_should_fail_closed(answer, code):
    # Arrange
    setup = _Setup()
    setup.store.get_answers.append(answer)
    # Act
    failure = _failure(lambda: setup.source.read(0, 10)).failure
    # Assert
    assert failure.code == code
    assert failure.retryable is False
    assert answer.closed


def test_read_when_the_store_fails_for_now_should_fail_retryable_as_the_s3_reader_does():
    # Arrange
    setup = _Setup()
    setup.store.get_answers.append(FakeResponse(503))
    # Act
    failure = _failure(lambda: setup.source.read(0, 10)).failure
    # Assert
    assert (failure.code, failure.retryable) == ("storage_error", True)


def test_read_when_the_stop_line_has_passed_should_send_nothing():
    # Arrange
    setup = _Setup()
    passed = UrlRangeSource(
        setup.store, size_bytes=len(SOURCE), etag=SOURCE_ETAG, stop_line=stop_line_within(lambda: 0.0),
        monotonic=setup.clock.monotonic, ask_url=setup._ask,
    )
    # Act
    failure = _failure(lambda: passed.read(0, 10)).failure
    # Assert
    assert (failure.error_class, failure.code) == ("deadline_exceeded", "source_read_deadline_exceeded")
    assert setup.store.gets == []
