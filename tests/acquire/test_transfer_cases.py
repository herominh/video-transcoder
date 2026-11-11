"""The Drive transfer's own rules, through a whole acquire: Google's refusals and how they map, the metadata asked
again before any write, redirects never followed, the download's tries and its resumption, the size and the
checksums, and the store's PUTs with their tries and ETags. Google and the store are tests/drive_import/fakes.py's
session; nothing opens a connection. These cases ran through the import/v1 job until its retirement (TD-184) and
hold the same transfer code, which only the acquire runs now."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import requests

from core.drive_import.clock import StopLine
from core.drive_import.failure import Code, ImportFailed, Stage
from core.drive_import.transfer import DOWNLOAD_CHUNK_BYTES, Part, TransferPlan, TransferResult, transfer
from tests.acquire.support import PART_SIZE, SIGNATURE_CANARY, Scenario, grant_for
from tests.drive_import.fakes import (
    API_KEY, FILE_ID, HEAD_REVISION, METADATA_URL, MEDIA_URL, MIB, REDIRECT_TARGET, RESOURCE_KEY, STORAGE_HOST, DriveFile,
    Fail, FakeClock, FakeResponse, FakeSession, Late, Put, Serve, google_error, json_answer, part_url, redirect,
)

TWO_PARTS = PART_SIZE + 37
COMPLETED = "source.acquire.result.completed"


def _outcome(scenario: Scenario) -> dict[str, Any]:
    return json.loads(scenario.run()["vh_message"])


def _code(scenario: Scenario) -> str | None:
    outcome = _outcome(scenario)
    return outcome["error"]["code"] if "error" in outcome else None


def _part_urls(scenario: Scenario) -> list[str]:
    """The part URLs of the first grant, as the scenario's Hub signs them at the claim."""
    grant = grant_for(scenario.request, "88888888-8888-7888-8888-888888888888", scenario.clock.now())
    return [part["url"] for part in grant["upload"]["parts"]]


def _asked_the_redirect_target(scenario: Scenario) -> bool:
    return any(request.url.startswith(REDIRECT_TARGET) for request in scenario.http.requests)


# --- what is asked of Google ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("resource_key", [RESOURCE_KEY, None])
def test_acquire_when_it_asks_google_should_name_the_file_by_its_id_and_send_the_key_in_a_header_only(
    resource_key: str | None,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.request["external"]["resource_key"] = resource_key
    # Act
    outcome = _outcome(scenario)
    # Assert: the metadata, then the bytes, of one fixed host; the key in its header and in no URL
    metadata, media = scenario.http.of("GET")
    assert outcome["message_kind"] == COMPLETED
    assert (metadata.url, media.url) == (METADATA_URL, MEDIA_URL)
    assert metadata.headers["X-Goog-Api-Key"] == API_KEY and media.headers["X-Goog-Api-Key"] == API_KEY
    assert media.headers["Accept-Encoding"] == "identity" and "Range" not in media.headers
    assert [request.url for request in scenario.http.requests if API_KEY in request.url] == []
    resource_keys = [request.headers.get("X-Goog-Drive-Resource-Keys") for request in (metadata, media)]
    expected = f"{FILE_ID}/{resource_key}" if resource_key is not None else None
    assert resource_keys == [expected, expected]


def test_acquire_when_it_writes_should_send_each_part_whole_with_its_length_and_content_md5_once_it_is_whole() -> None:
    # Arrange
    scenario = Scenario(TWO_PARTS)
    # Act
    outcome = _outcome(scenario)
    # Assert: the first part as soon as it is whole, the last only once the whole stream was read
    first, last = scenario.http.part_puts
    assert outcome["message_kind"] == COMPLETED
    assert [(put.length, put.md5) for put in (first, last)] == [
        (PART_SIZE, scenario.file.part_md5(1)), (37, scenario.file.part_md5(2)),
    ]
    assert PART_SIZE <= first.served_before <= PART_SIZE + DOWNLOAD_CHUNK_BYTES
    assert last.served_before == TWO_PARTS
    for put in (first, last):
        assert put.headers["Content-Length"] == str(put.length)
        assert base64.b64decode(put.headers["Content-MD5"]).hex() == put.md5


# --- redirects are never followed ----------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_acquire_when_google_answers_the_download_with_a_redirect_should_fail_without_asking_its_target(
    status: int,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.media = [redirect(status)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert len(scenario.http.media_gets) == 1
    assert not _asked_the_redirect_target(scenario) and scenario.http.part_puts == []


def test_acquire_when_google_answers_the_metadata_with_a_redirect_should_fail_before_any_download() -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [redirect(302)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert [request.url for request in scenario.http.of("GET")] == [METADATA_URL]


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_acquire_when_the_store_answers_a_part_put_with_a_redirect_should_fail_without_asking_its_target(
    status: int,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[_part_urls(scenario)[0]] = [FakeResponse(status, {"Location": REDIRECT_TARGET})]
    # Act
    code = _code(scenario)
    # Assert: one try, and nothing sent where the redirect pointed
    assert code == "upload_failed"
    assert len(scenario.http.part_puts) == 1
    assert not _asked_the_redirect_target(scenario)


# --- Google's refusals ---------------------------------------------------------------------------------------------

GOOGLE_REFUSALS = [
    pytest.param(lambda: FakeResponse(404), "file_removed", id="404"),
    pytest.param(lambda: google_error(403, "downloadQuotaExceeded", "usageLimits"), "provider_quota", id="403-quota"),
    pytest.param(lambda: google_error(403, "userRateLimitExceeded"), "provider_quota", id="403-rate"),
    pytest.param(lambda: FakeResponse(429), "provider_quota", id="429"),
    pytest.param(lambda: google_error(400, "keyInvalid"), "not_configured", id="400-key-invalid"),
    pytest.param(lambda: google_error(403, "accessNotConfigured"), "not_configured", id="403-api-not-enabled"),
    pytest.param(lambda: google_error(403, "forbidden", "usageLimits"), "not_configured", id="403-key-forbidden"),
    pytest.param(lambda: google_error(403, "forbidden", "global", detail="API_KEY_SERVICE_BLOCKED"), "not_configured",
                 id="403-error-info-key-blocked"),
    pytest.param(lambda: google_error(400, detail="API_KEY_INVALID"), "not_configured",
                 id="400-error-info-key-invalid"),
    pytest.param(lambda: google_error(403, "cannotDownloadAbusiveFile"), "provider_refused", id="403-abusive-file"),
    pytest.param(lambda: google_error(403, "insufficientFilePermissions"), "link_not_shared", id="403-not-shared"),
    pytest.param(lambda: google_error(403, "notFound"), "link_not_shared", id="403-not-found"),
    pytest.param(lambda: google_error(403, "aReasonOfALaterVersion"), "provider_refused", id="403-unknown-reason"),
    pytest.param(lambda: google_error(400, "rateLimitExceeded"), "provider_refused", id="400-rate-is-no-quota"),
    pytest.param(lambda: FakeResponse(403, body=b"<html>no JSON</html>"), "provider_refused", id="403-no-json"),
    pytest.param(lambda: FakeResponse(403, body=b"[" * 70_000), "provider_refused", id="403-body-too-long"),
    pytest.param(lambda: FakeResponse(403, body=b"[" * 60_000), "provider_refused", id="403-body-nested-too-deep"),
    pytest.param(lambda: json_answer(403, {"error": {"errors": [{"reason": ["a"]}], "details": [{"reason": []}]}}),
                 "provider_refused", id="403-reasons-that-are-no-text"),
    pytest.param(lambda: json_answer(403, {"error": "a text"}), "provider_refused", id="403-error-that-is-no-object"),
    pytest.param(lambda: FakeResponse(401), "download_failed", id="401"),
    pytest.param(lambda: FakeResponse(410), "download_failed", id="410"),
]


@pytest.mark.parametrize(("answer", "code"), GOOGLE_REFUSALS)
def test_acquire_when_google_refuses_the_metadata_should_end_with_its_code_before_any_download(answer, code) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [answer()]
    # Act
    ended = _code(scenario)
    # Assert
    assert ended == code
    assert [request.url for request in scenario.http.of("GET")] == [METADATA_URL]
    assert scenario.http.part_puts == []


@pytest.mark.parametrize(("answer", "code"), GOOGLE_REFUSALS)
def test_acquire_when_google_refuses_the_download_should_end_with_its_code_and_write_nothing(answer, code) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.media = [answer()]
    # Act
    ended = _code(scenario)
    # Assert: one request of the bytes, no try again
    assert ended == code
    assert len(scenario.http.media_gets) == 1
    assert scenario.http.part_puts == []


def test_acquire_when_google_refuses_a_resumed_download_should_end_with_its_code_and_write_no_more() -> None:
    # Arrange: a reset in the second part, then Google's quota on the request that resumes it
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = [Serve(break_after=PART_SIZE + 10), google_error(403, "downloadQuotaExceeded")]
    first_part = _part_urls(scenario)[:1]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "provider_quota"
    assert [put.url for put in scenario.http.part_puts] == first_part


class _BodyNeverToRead:
    """An answer's body the transfer must never read: each read is counted and raises. The HTTP library would decode
    a body in a content coding whole before any limit counted it (a stacked `gzip, gzip` of a few kilobytes can come
    to gigabytes)."""

    def __init__(self) -> None:
        self.reads = 0

    def __call__(self, chunk_size: int) -> Iterator[bytes]:
        return self

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        self.reads += 1
        raise AssertionError("the body of an answer in a content coding was read")


@pytest.mark.parametrize("coding", ["gzip", "gzip, gzip"])
@pytest.mark.parametrize("status", [400, 403])
@pytest.mark.parametrize("request_kind", ["metadata", "media"])
def test_acquire_when_google_refuses_in_a_content_coding_should_end_provider_refused_without_reading_the_body(
    request_kind: str, status: int, coding: str,
) -> None:
    # Arrange: Google's refusal comes in a content coding, though the transfer asked for none
    scenario = Scenario()
    body = _BodyNeverToRead()
    setattr(scenario.http, request_kind, [
        FakeResponse(status, {"Content-Type": "application/json", "Content-Encoding": coding}, chunks=body),
    ])
    # Act
    code = _code(scenario)
    # Assert: the code of a refusal whose reasons went unread, its body never read, nothing written
    assert code == "provider_refused"
    assert body.reads == 0
    assert scenario.http.part_puts == []


# --- the metadata again before the first write ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        pytest.param({"size": "a lot"}, "source_changed_at_provider", id="a-size-that-is-no-number"),
        pytest.param({"size": True}, "source_changed_at_provider", id="a-size-that-is-a-boolean"),
        pytest.param({"trashed": True}, "file_removed", id="in-the-trash"),
        pytest.param({"trashed": None}, "file_removed", id="in-the-trash-as-a-null"),
        pytest.param({"capabilities": {"canDownload": False}}, "link_not_shared", id="no-longer-downloadable"),
        pytest.param({"capabilities": {"canDownload": None}}, "link_not_shared", id="downloadable-as-a-null"),
    ],
)
def test_acquire_when_the_metadata_again_is_not_the_pasted_files_should_refuse_before_any_download(
    changes: dict[str, Any], code: str,
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata_document.update(changes)
    # Act
    ended = _code(scenario)
    # Assert: nothing was downloaded, nothing written
    assert ended == code
    assert [request.url for request in scenario.http.of("GET")] == [METADATA_URL]


@pytest.mark.parametrize(
    "metadata_change",
    [
        pytest.param(lambda metadata: metadata.update(sha256Checksum=metadata["sha256Checksum"].upper()),
                     id="the-recorded-sha256-in-capitals"),
        pytest.param(lambda metadata: metadata.update(size=int(metadata["size"])), id="the-size-as-a-number"),
        pytest.param(lambda metadata: [metadata.pop(name) for name in ("capabilities", "trashed")],
                     id="nothing-said-of-downloading"),
    ],
)
def test_acquire_when_the_metadata_again_says_the_same_in_another_form_should_complete(metadata_change) -> None:
    # Arrange
    scenario = Scenario()
    metadata_change(scenario.http.metadata_document)
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED


def test_acquire_when_google_gives_values_now_that_the_dispatch_did_not_carry_should_not_compare_them() -> None:
    # Arrange: the paste recorded no SHA-1, no SHA-256 and no head revision; Google now gives other ones
    scenario = Scenario()
    scenario.request["external"]["expected"].update(sha1=None, sha256=None, head_revision_id=None)
    scenario.http.metadata_document.update(sha1Checksum="1" * 40, sha256Checksum="2" * 64, headRevisionId="0BNew")
    # Act
    outcome = _outcome(scenario)
    # Assert: completed, verified by the MD5 the dispatch carried
    assert outcome["message_kind"] == COMPLETED
    assert outcome["checksum_verified"] == "md5"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(lambda: json_answer(200, ["a", "list"]), id="no-json-object"),
        pytest.param(lambda: FakeResponse(200, {"Content-Encoding": "gzip"}, body=b"\x1f\x8b"), id="a-content-coding"),
    ],
)
def test_acquire_when_the_metadata_is_not_a_plain_json_object_should_end_download_failed(answer) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [answer()]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert scenario.http.media_gets == []


TRY_AGAIN_ON_GOOGLE = [
    pytest.param(lambda: FakeResponse(500), id="500"),
    pytest.param(lambda: FakeResponse(503), id="503"),
    pytest.param(lambda: FakeResponse(408), id="408"),
    pytest.param(lambda: Fail(), id="no-answer"),
    pytest.param(lambda: Fail(requests.exceptions.ReadTimeout("read timed out")), id="a-time-out"),
    pytest.param(lambda: FakeResponse(200, body=b"{" + b" " * 70_000 + b"}"), id="a-body-too-long-to-read"),
]


@pytest.mark.parametrize("first", TRY_AGAIN_ON_GOOGLE)
def test_acquire_when_the_metadata_fails_once_should_ask_again_after_a_wait_and_complete(first) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [first()]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert len(scenario.http.of("GET", METADATA_URL)) == 2
    assert scenario.clock.slept == [0.5]


def test_acquire_when_the_metadata_fails_three_times_should_end_download_failed() -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.metadata = [FakeResponse(500) for _ in range(3)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert len(scenario.http.of("GET", METADATA_URL)) == 3 and scenario.http.media_gets == []
    assert scenario.clock.slept == [0.5, 2.0]


# --- the download: its size, its checksums, its tries ---------------------------------------------------------------


def test_acquire_when_google_sends_twice_the_declared_bytes_should_end_size_exceeded_within_one_chunk() -> None:
    # Arrange
    scenario = Scenario(MIB)
    scenario.http.media = [Serve(extra=MIB)]
    # Act
    code = _code(scenario)
    # Assert: it stopped reading at the chunk that went past the size, and wrote nothing
    assert code == "size_exceeded"
    assert scenario.http.bytes_served <= MIB + DOWNLOAD_CHUNK_BYTES
    assert scenario.http.part_puts == []


def test_acquire_when_google_sends_bytes_past_the_last_part_should_never_write_the_last_part() -> None:
    # Arrange
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = [Serve(extra=1)]
    first_part = _part_urls(scenario)[:1]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "size_exceeded"
    assert [put.url for put in scenario.http.part_puts] == first_part


@pytest.mark.parametrize("checksum", ["sha256", "md5"])
def test_acquire_when_the_stream_does_not_match_the_recorded_checksum_should_end_integrity_failed(checksum: str) -> None:
    # Arrange: Google serves other bytes of the same size under the metadata of the pasted file
    scenario = Scenario(TWO_PARTS)
    if checksum == "md5":
        scenario.request["external"]["expected"]["sha256"] = None
    scenario.http.file = DriveFile(TWO_PARTS, seed=99)
    first_part = _part_urls(scenario)[:1]
    # Act
    code = _code(scenario)
    # Assert: the part before the last was written as it came; the last never
    assert code == "integrity_failed"
    assert [put.url for put in scenario.http.part_puts] == first_part


def test_acquire_when_the_stream_resets_in_the_first_part_should_download_again_from_the_start() -> None:
    # Arrange
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = [Serve(break_after=3 * MIB)]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert ["Range" in request.headers for request in scenario.http.media_gets] == [False, False]
    assert [put.md5 for put in scenario.http.part_puts] == [scenario.file.part_md5(n) for n in (1, 2)]


def test_acquire_when_the_stream_resets_three_times_in_one_part_should_end_download_failed() -> None:
    # Arrange: three breaks in the second part, each of its three requests
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = [Serve(break_after=PART_SIZE + 10), Serve(break_after=10), Serve(break_after=10)]
    first_part = _part_urls(scenario)[:1]
    # Act
    code = _code(scenario)
    # Assert: the first part was written, the second never; no fourth request
    assert code == "download_failed"
    assert len(scenario.http.media_gets) == 3
    assert [put.url for put in scenario.http.part_puts] == first_part
    assert scenario.clock.slept == [0.5, 2.0]


def test_acquire_when_two_parts_each_reset_twice_should_count_each_parts_tries_apart_and_complete() -> None:
    # Arrange: the first part breaks twice; its third request goes on into the second part, which breaks twice too
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = [
        Serve(break_after=MIB), Serve(break_after=MIB), Serve(break_after=PART_SIZE + 10), Serve(break_after=10),
    ]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert [request.headers.get("Range") for request in scenario.http.media_gets] == [
        None, None, None, f"bytes={PART_SIZE}-", f"bytes={PART_SIZE}-",
    ]


@pytest.mark.parametrize("first", TRY_AGAIN_ON_GOOGLE[:5] + [
    pytest.param(lambda: Fail(requests.exceptions.ConnectTimeout("connect timed out")), id="a-connect-time-out"),
])
def test_acquire_when_the_download_fails_before_its_first_byte_should_ask_again_and_complete(first) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.media = [first()]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert len(scenario.http.media_gets) == 2


def test_acquire_when_the_download_fails_three_times_before_its_first_byte_should_end_download_failed() -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.media = [FakeResponse(503) for _ in range(3)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert len(scenario.http.media_gets) == 3


@pytest.mark.parametrize(
    "answers",
    [
        pytest.param([Serve(break_after=PART_SIZE + 10), Serve(offset=PART_SIZE + 1)], id="a-206-from-another-byte"),
        pytest.param([Serve(break_after=PART_SIZE + 10), Serve(status=200)], id="a-200-to-a-resumed-request"),
        pytest.param([Serve(status=206)], id="a-206-to-the-first-request"),
        pytest.param([FakeResponse(204)], id="a-204"),
        pytest.param([FakeResponse(200, {"Content-Encoding": "gzip"}, body=b"\x1f\x8b compressed")],
                     id="the-bytes-in-a-content-coding"),
    ],
)
def test_acquire_when_the_download_answers_another_shape_than_asked_should_end_download_failed(answers) -> None:
    # Arrange
    scenario = Scenario(TWO_PARTS)
    scenario.http.media = list(answers)
    # Act
    code = _code(scenario)
    # Assert
    assert code == "download_failed"
    assert len(scenario.http.media_gets) == len(answers)
    assert len(scenario.http.part_puts) <= 1


# --- the parts written --------------------------------------------------------------------------------------------


def test_acquire_when_a_part_put_fails_twice_with_500_should_write_it_on_its_third_try_and_complete() -> None:
    # Arrange
    scenario = Scenario()
    url = _part_urls(scenario)[0]
    scenario.http.puts[url] = [Put(500), Put(500)]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert len(scenario.http.of("PUT", url)) == 3
    assert scenario.clock.slept == [0.5, 2.0]
    assert outcome["parts"] == [{"n": 1, "etag": scenario.file.part_md5(1)}]


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(lambda: Put(503), id="503"),
        pytest.param(lambda: Put(408), id="408"),
        pytest.param(lambda: Put(429), id="429"),
        pytest.param(lambda: Fail(), id="no-answer"),
        pytest.param(lambda: Fail(requests.exceptions.ReadTimeout("read timed out")), id="a-time-out"),
    ],
)
def test_acquire_when_a_part_put_fails_once_in_a_way_worth_another_try_should_write_it_again(first) -> None:
    # Arrange
    scenario = Scenario()
    url = _part_urls(scenario)[0]
    scenario.http.puts[url] = [first()]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["message_kind"] == COMPLETED
    assert len(scenario.http.of("PUT", url)) == 2


def test_acquire_when_a_part_put_fails_four_times_should_end_upload_failed_after_the_three_waits() -> None:
    # Arrange
    scenario = Scenario()
    url = _part_urls(scenario)[0]
    scenario.http.puts[url] = [Put(500) for _ in range(4)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "upload_failed"
    assert len(scenario.http.of("PUT", url)) == 4
    assert scenario.clock.slept == [0.5, 2.0, 8.0]


@pytest.mark.parametrize("status", [400, 403, 404, 409, 411, 201, 204])
def test_acquire_when_the_store_refuses_a_part_put_should_end_upload_failed_at_once(status: int) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[_part_urls(scenario)[0]] = [Put(status)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "upload_failed"
    assert len(scenario.http.part_puts) == 1
    assert scenario.clock.slept == []


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-etag"),
        pytest.param({"ETag": '""'}, id="an-empty-etag"),
        pytest.param({"ETag": '"an etag"'}, id="an-etag-with-a-space"),
        pytest.param({"ETag": 'W/"weak"'}, id="a-weak-etag"),
        pytest.param({"ETag": '"tab\there"'}, id="an-etag-with-a-control-character"),
    ],
)
def test_acquire_when_a_part_puts_200_carries_no_etag_of_the_results_form_should_end_upload_failed(
    headers: dict[str, str],
) -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[_part_urls(scenario)[0]] = [Put(200, headers=headers)]
    # Act
    code = _code(scenario)
    # Assert
    assert code == "upload_failed"


def test_acquire_when_the_store_gives_an_etag_without_quotes_should_keep_it_as_it_is() -> None:
    # Arrange
    scenario = Scenario()
    scenario.http.puts[_part_urls(scenario)[0]] = [Put(200, headers={"etag": "abc-123"})]
    # Act
    outcome = _outcome(scenario)
    # Assert
    assert outcome["parts"] == [{"n": 1, "etag": "abc-123"}]


def test_acquire_when_the_wall_deadline_passes_while_a_part_is_sent_should_cut_its_body_and_end_deadline() -> None:
    # Arrange: each block of the part's body takes a second, and the wall deadline is a minute away
    scenario = Scenario(MIB)
    scenario.http.puts[_part_urls(scenario)[0]] = [Put(seconds_per_block=1.0)]
    # Act
    outcome = _outcome(scenario)
    # Assert: the body stopped at the deadline, and nothing was asked of the store after it
    (put,) = scenario.http.part_puts
    assert (outcome["error"]["code"], outcome["error"]["stage"]) == ("deadline", "uploading")
    assert put.cut and put.length < MIB
    assert scenario.http.result_puts == []


# --- the stop line, at the transfer itself -----------------------------------------------------------------------
# A whole acquire answers `deadline` for every failure once its wall deadline has passed, so the transfer's own rule
# (a try that fails after the stop line is the stop line's, never its own code) is held here, on transfer() alone.

THREE_PARTS = 2 * PART_SIZE + MIB
STOP_S = 60.0


def _transfer(
    file: DriveFile, *, stop_s: float = STOP_S, script: Callable[[FakeSession, FakeClock], None] = lambda s, c: None,
) -> tuple[TransferResult | ImportFailed, FakeSession, FakeClock, StopLine]:
    """One transfer of `file`, its stop line `stop_s` away on a fake clock, after `script` set up the answers."""
    clock = FakeClock()
    session = FakeSession(file, clock)
    script(session, clock)
    parts = tuple(Part(n, part_url(n)) for n in range(1, -(-file.size // PART_SIZE) + 1))
    stop_line = StopLine(clock, clock.monotonic() + stop_s)
    plan = TransferPlan(FILE_ID, None, file.size, file.md5, None, file.sha256, HEAD_REVISION, PART_SIZE, parts, stop_line)
    try:
        ended: TransferResult | ImportFailed = transfer(plan, API_KEY, session)
    except ImportFailed as failure:
        ended = failure
    return ended, session, clock, stop_line


def _ended_with(ended: object, code: Code, stage: Stage) -> bool:
    return isinstance(ended, ImportFailed) and (ended.code, ended.stage) == (code, stage)


PAST_THE_STOP_LINE_S = STOP_S + 1


def test_transfer_when_far_from_its_stop_line_should_give_each_request_its_own_time_outs() -> None:
    # Arrange: two whole parts and a short last one, the stop line far away
    file = DriveFile(THREE_PARTS)
    # Act
    ended, session, _, _ = _transfer(file, stop_s=3600.0)
    # Assert: connect 10 s; Google's metadata 30 s and bytes 60 s; a PUT at least 60 s, a second per MiB beyond
    assert isinstance(ended, TransferResult)
    assert [request.timeout for request in session.of("GET")] == [(10.0, 30.0), (10.0, 60.0)]
    assert [request.timeout for request in session.of("PUT")] == [(10.0, 64.0), (10.0, 64.0), (10.0, 60.0)]


def test_transfer_when_its_stop_line_is_near_should_cut_every_requests_time_outs_to_what_is_left() -> None:
    # Act: the stop line 30 s away
    ended, session, _, _ = _transfer(DriveFile(THREE_PARTS), stop_s=30.0)
    # Assert
    assert isinstance(ended, TransferResult)
    assert {request.timeout for request in session.requests} == {(10.0, 30.0)}


def test_transfer_when_it_asks_should_hand_every_request_its_stop_line_and_close_every_answer() -> None:
    # Act
    ended, session, _, stop_line = _transfer(DriveFile(THREE_PARTS))
    # Assert: the session is the caller's to close; every answer is closed by the transfer
    assert isinstance(ended, TransferResult)
    assert len(session.stop_lines) == len(session.requests)
    assert all(given is stop_line for given in session.stop_lines)
    assert not session.closed
    assert session.answers and all(answer.closed for answer in session.answers)


def test_transfer_when_its_stop_line_has_passed_at_the_start_should_end_deadline_without_a_request() -> None:
    # Act
    ended, session, _, _ = _transfer(DriveFile(MIB), stop_s=-1.0)
    # Assert
    assert _ended_with(ended, Code.DEADLINE, Stage.INPUT)
    assert session.requests == []


def test_transfer_when_the_metadata_fails_after_the_stop_line_should_end_deadline() -> None:
    # Arrange: the third try's 503 comes once the clock has passed the stop line
    def script(session: FakeSession, clock: FakeClock) -> None:
        session.metadata += [FakeResponse(503), FakeResponse(503), Late(FakeResponse(503), PAST_THE_STOP_LINE_S)]

    # Act
    ended, session, _, _ = _transfer(DriveFile(MIB), script=script)
    # Assert: not `download_failed`, though the tries are used up
    assert _ended_with(ended, Code.DEADLINE, Stage.METADATA)
    assert session.media_gets == []


def test_transfer_when_the_last_download_try_breaks_after_the_stop_line_should_end_deadline() -> None:
    # Arrange: the second part's third request breaks once the clock has passed the stop line
    def script(session: FakeSession, clock: FakeClock) -> None:
        session.media += [
            Serve(break_after=PART_SIZE + MIB), Serve(break_after=MIB),
            Serve(break_after=MIB, seconds_before_break=PAST_THE_STOP_LINE_S),
        ]

    # Act
    ended, session, _, _ = _transfer(DriveFile(THREE_PARTS), script=script)
    # Assert
    assert _ended_with(ended, Code.DEADLINE, Stage.DOWNLOAD)
    assert len(session.media_gets) == 3


@pytest.mark.parametrize("more", [b"", b"x" * MIB], ids=["the-stream-ends-cleanly", "a-chunk-comes-after-it"])
def test_transfer_when_the_stop_line_passes_while_the_stream_is_read_should_end_deadline_without_writing(
    more: bytes,
) -> None:
    # Arrange: half the file, then the clock passes the stop line; the stream ends there (a body framed by the
    # connection's close, which the watchdog's shutdown ends without an error) or gives another chunk
    file = DriveFile(MIB)

    def script(session: FakeSession, clock: FakeClock) -> None:
        def chunks(chunk_size: int) -> Iterator[bytes]:
            yield file.bytes(0, MIB // 2)
            clock.advance(PAST_THE_STOP_LINE_S)
            if more:
                yield more

        session.media.append(FakeResponse(200, chunks=chunks))

    # Act
    ended, session, _, _ = _transfer(file, script=script)
    # Assert: `deadline`, never `size_short`, and nothing written
    assert isinstance(ended, ImportFailed) and ended.code is Code.DEADLINE
    assert session.of("PUT") == []


def test_transfer_when_the_last_put_try_fails_after_the_stop_line_should_end_deadline() -> None:
    # Arrange
    def script(session: FakeSession, clock: FakeClock) -> None:
        session.puts[part_url(1)] = [Put(500), Put(500), Put(500), Put(500, seconds_after=PAST_THE_STOP_LINE_S)]

    # Act
    ended, session, _, _ = _transfer(DriveFile(MIB), script=script)
    # Assert: not `upload_failed`, though the tries are used up
    assert _ended_with(ended, Code.DEADLINE, Stage.UPLOAD)
    assert len(session.of("PUT")) == 4


def test_transfer_when_a_wait_before_another_try_would_pass_the_stop_line_should_end_deadline_without_waiting() -> None:
    # Arrange: the first PUT fails 0.4 s before the stop line
    def script(session: FakeSession, clock: FakeClock) -> None:
        session.puts[part_url(1)] = [Put(500, seconds_after=STOP_S - 0.4)]

    # Act
    ended, session, clock, _ = _transfer(DriveFile(MIB), script=script)
    # Assert
    assert _ended_with(ended, Code.DEADLINE, Stage.UPLOAD)
    assert len(session.of("PUT")) == 1 and clock.slept == []


def test_transfer_when_the_stop_line_passes_before_the_last_part_should_end_deadline_and_ask_nothing_more() -> None:
    # Arrange: the second part's PUT takes the clock past the stop line
    def script(session: FakeSession, clock: FakeClock) -> None:
        session.puts[part_url(2)] = [Put(seconds_after=PAST_THE_STOP_LINE_S)]

    # Act
    ended, session, _, _ = _transfer(DriveFile(THREE_PARTS), script=script)
    # Assert: what was written stays; nothing more is asked
    assert isinstance(ended, ImportFailed) and ended.code is Code.DEADLINE
    assert [put.url for put in session.of("PUT")] == [part_url(1), part_url(2)]
    assert session.requests[-1].url == part_url(2)


# --- what never leaves an acquire --------------------------------------------------------------------------------

# What no log line, log argument or failed result may hold, whatever else it holds.
NEVER_SHOWN = (API_KEY, FILE_ID, RESOURCE_KEY, SIGNATURE_CANARY, STORAGE_HOST, "https://", "googleapis", "X-Amz",
               "Google's own text")


@pytest.mark.parametrize(
    ("size", "script", "code"),
    [
        pytest.param(TWO_PARTS, lambda http: http.media.extend([Serve(break_after=PART_SIZE + 10), Serve(break_after=10)]),
                     None, id="completed-after-two-resets"),
        pytest.param(37, lambda http: http.metadata.extend([Fail(), FakeResponse(503)]), None,
                     id="metadata-tried-again"),
        pytest.param(37, lambda http: http.media.append(google_error(403, "downloadQuotaExceeded")), "provider_quota",
                     id="provider-quota"),
        pytest.param(37, lambda http: http.metadata.append(google_error(400, "keyInvalid")), "not_configured",
                     id="key-invalid"),
        pytest.param(MIB, lambda http: http.media.append(Serve(extra=MIB)), "size_exceeded", id="size-exceeded"),
        pytest.param(TWO_PARTS, lambda http: http.media.extend(
            [Serve(break_after=PART_SIZE + 10), Serve(break_after=10), Serve(break_after=10)]),
            "download_failed", id="three-resets"),
    ],
)
def test_acquire_when_it_ends_in_any_way_should_log_nothing_it_must_keep(
    caplog: pytest.LogCaptureFixture, size: int, script: Callable[[Any], None], code: str | None,
) -> None:
    # Arrange
    caplog.set_level(logging.DEBUG)
    scenario = Scenario(size)
    script(scenario.http)
    # Act
    outcome = _outcome(scenario)
    # Assert: the ending is the one meant; no line or argument shows a secret, a URL or a store's ETag, a failed
    # result shows neither, and no line carries an error (whose traceback would end with its text)
    lines = [record.getMessage() for record in caplog.records] + [repr(record.args) for record in caplog.records]
    etags = tuple(scenario.http.etags_given.values())
    assert (outcome["error"]["code"] if code else outcome["message_kind"]) == (code or COMPLETED)
    assert [value for value in (*NEVER_SHOWN, *etags) for line in lines if value in line] == []
    if code:
        assert [value for value in (*NEVER_SHOWN, *etags) if value in json.dumps(outcome)] == []
    assert [record for record in caplog.records if record.exc_info or record.exc_text] == []
