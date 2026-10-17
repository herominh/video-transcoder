"""One import job end to end on the fake session (tests/drive_import/fakes.py): what it asks Google, what it writes
to the store, its receipt, and the code of every way it can end. No connection is opened and no real time passes.

The files are the contract's sizes: parts of 64 MiB, made lazily. THREE is the running example of these tests
(two whole parts and a last one of 1 MiB), ONE a file of one part.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import timedelta
from typing import Any, Callable

import pytest
import requests

from core.drive_import import contract
from core.drive_import.input_check import PART_SIZE
from core.drive_import.job import DOWNLOAD_CHUNK_BYTES, run_import
from tests.drive_import.fakes import (
    API_KEY,
    FILE_ID,
    HEAD_REVISION,
    JOB_UUID,
    MEDIA_URL,
    METADATA_URL,
    MIB,
    RECEIPT_KEY,
    RECEIPT_URL,
    REDIRECT_TARGET,
    RESOURCE_KEY,
    START,
    THREE_PARTS_SIZE,
    DriveFile,
    Fail,
    FakeClock,
    FakeResponse,
    FakeSession,
    Late,
    Put,
    Serve,
    google_error,
    import_input,
    json_answer,
    part_url,
    redirect,
    settings,
    signed,
)

THREE = DriveFile(THREE_PARTS_SIZE)
ONE = DriveFile(MIB)
# Bytes of another file of the same size: what Google serves when the file changed under the same metadata.
OTHER_THREE = DriveFile(THREE_PARTS_SIZE, seed=99)
OTHER_ONE = DriveFile(MIB, seed=99)
IMPORTED = {"contract": "import/v1", "status": "imported", "code": None}
# In _run(): the file's own input.
OWN_INPUT = object()


def failed(code: str) -> dict[str, Any]:
    return {"contract": "import/v1", "status": "failed", "code": code}


def _run(
    file: DriveFile,
    *,
    document: Any = OWN_INPUT,
    script: Callable[[FakeSession], None] = lambda session: None,
    served: DriveFile | None = None,
    metadata: dict[str, Any] | None = None,
    configuration: Any = None,
    clock: FakeClock | None = None,
) -> tuple[dict[str, Any], FakeSession, FakeClock]:
    """Run one job for `file` (its input, unless `document`), Google serving `served` (default: `file`) with
    `metadata` (default: `file`'s), after `script` set up the session's answers."""
    clock = clock or FakeClock()
    session = FakeSession(served or file, clock, metadata if metadata is not None else file.metadata())
    script(session)
    output = run_import(
        import_input(file) if document is OWN_INPUT else document,
        settings() if configuration is None else configuration,
        session,
        clock,
    )
    assert contract.first_error_pointer(contract.OUTPUT_SCHEMA, output) is None
    return output, session, clock


def _part_sizes(file: DriveFile) -> list[int]:
    total = -(-file.size // PART_SIZE)
    return [PART_SIZE] * (total - 1) + [file.size - (total - 1) * PART_SIZE]


# --- the running example -------------------------------------------------------------------------------------------


def test_run_import_when_the_file_downloads_whole_should_write_every_part_and_a_receipt_whose_mac_verifies():
    # Act
    output, session, _ = _run(THREE)

    # Assert: imported; every part written with its own bytes; the receipt names the store's ETags and verifies
    receipt = session.receipt()
    assert output == IMPORTED
    assert [(put.url, put.length, put.md5) for put in session.part_puts] == [
        (part_url(n), size, THREE.part_md5(n)) for n, size in enumerate(_part_sizes(THREE), start=1)
    ]
    assert receipt == {
        "contract": "import/v1",
        "job": JOB_UUID,
        "bytes": THREE.size,
        "checksum_verified": "sha256",
        "parts": [{"n": n, "etag": session.etags_given[part_url(n)]} for n in (1, 2, 3)],
        "mac": contract.receipt_mac(receipt, RECEIPT_KEY),
    }
    assert contract.first_error_pointer(contract.RECEIPT_SCHEMA, receipt) is None
    assert session.requests[-1].url == RECEIPT_URL


def test_run_import_when_it_writes_should_write_each_part_once_whole_and_the_last_after_the_whole_stream():
    # Act
    _, session, _ = _run(THREE)

    # Assert: a part is written as soon as it is whole, the last only once the whole file was read
    first, second, last = session.part_puts
    assert PART_SIZE <= first.served_before <= PART_SIZE + DOWNLOAD_CHUNK_BYTES
    assert 2 * PART_SIZE <= second.served_before <= 2 * PART_SIZE + DOWNLOAD_CHUNK_BYTES
    assert last.served_before == THREE.size
    assert session.receipt_puts[0].served_before == THREE.size


def test_run_import_when_it_writes_should_send_each_body_with_its_length_and_its_content_md5():
    # Act
    _, session, _ = _run(THREE)

    # Assert
    for put in session.of("PUT"):
        assert put.headers["Content-Length"] == str(put.length)
        assert base64.b64decode(put.headers["Content-MD5"]).hex() == put.md5
    assert session.receipt_puts[0].headers["Content-Type"] == "application/json"


def test_run_import_when_it_asks_google_should_name_the_file_by_its_id_and_send_the_key_in_a_header_only():
    # Act
    _, session, _ = _run(ONE)

    # Assert: the metadata, then the bytes, of one fixed host; the key in its header and in no URL
    metadata, media = session.of("GET")
    assert (metadata.url, media.url) == (METADATA_URL, MEDIA_URL)
    assert metadata.headers["X-Goog-Api-Key"] == API_KEY and media.headers["X-Goog-Api-Key"] == API_KEY
    assert "X-Goog-Drive-Resource-Keys" not in metadata.headers and "Range" not in media.headers
    assert media.headers["Accept-Encoding"] == "identity"
    assert [request.url for request in session.requests if API_KEY in request.url] == []


def test_run_import_when_the_link_has_a_resource_key_should_send_it_with_the_files_id_to_google():
    # Act
    _, session, _ = _run(ONE, document=import_input(ONE, resource_key=RESOURCE_KEY))

    # Assert
    assert [request.headers["X-Goog-Drive-Resource-Keys"] for request in session.of("GET")] == [
        f"{FILE_ID}/{RESOURCE_KEY}",
        f"{FILE_ID}/{RESOURCE_KEY}",
    ]


def test_run_import_when_far_from_its_stop_line_should_give_each_request_its_own_time_outs():
    # Act
    _, session, _ = _run(THREE)

    # Assert: connect 10 s; Google's metadata 30 s and bytes 60 s; a PUT at least 60 s, a second per MiB beyond
    assert [request.timeout for request in session.of("GET")] == [(10.0, 30.0), (10.0, 60.0)]
    put_timeouts = [request.timeout for request in session.of("PUT")]
    assert put_timeouts == [(10.0, 64.0), (10.0, 64.0), (10.0, 60.0), (10.0, 60.0)]


@pytest.mark.parametrize(
    ("carried", "verified"),
    [
        pytest.param({"md5": True, "sha256": True}, "sha256", id="sha256-when-both"),
        pytest.param({"md5": True, "sha256": False}, "md5", id="md5-without-sha256"),
        pytest.param({"md5": False, "sha256": False}, "none", id="none-without-either"),
    ],
)
def test_run_import_when_the_input_carries_checksums_should_verify_the_strongest_and_say_so_in_the_receipt(
    carried, verified
):
    # Act
    output, session, _ = _run(ONE, document=import_input(ONE, **carried))

    # Assert
    assert output == IMPORTED
    assert session.receipt()["checksum_verified"] == verified


def test_run_import_when_the_integer_members_are_written_as_floats_should_import_the_file():
    # Arrange: JSON's 1048576.0 is an integer member (README section 2), which Python reads as a float; the MAC the
    # Hub computed over the integers still verifies (the canonical form writes 1048576.0 as 1048576)
    document = import_input(ONE)
    document["expected"]["size"] = float(ONE.size)
    document["plan"] = {"part_size": float(PART_SIZE), "parts_total": 1.0}
    document["parts"][0]["n"] = 1.0

    # Act
    output, session, _ = _run(ONE, document=document)

    # Assert
    assert output == IMPORTED
    assert session.receipt()["bytes"] == ONE.size and isinstance(session.receipt()["bytes"], int)


# --- redirects are never followed ----------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_run_import_when_google_answers_the_download_with_a_redirect_should_fail_without_asking_its_target(status):
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.media.append(redirect(status)))

    # Assert
    assert output == failed("download_failed")
    assert len(session.media_gets) == 1
    assert [request for request in session.requests if request.url.startswith(REDIRECT_TARGET)] == []
    assert session.of("PUT") == []


def test_run_import_when_google_answers_the_metadata_with_a_redirect_should_fail_before_any_download():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.metadata.append(redirect(302)))

    # Assert
    assert output == failed("download_failed")
    assert [request.url for request in session.requests] == [METADATA_URL]


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_run_import_when_the_store_answers_a_put_with_a_redirect_should_fail_without_asking_its_target(status):
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.puts.setdefault(part_url(1), []).append(
        FakeResponse(status, {"Location": REDIRECT_TARGET})
    ))

    # Assert: one try, no receipt, and nothing sent where the redirect pointed
    assert output == failed("upload_failed")
    assert len(session.part_puts) == 1 and session.receipt_puts == []
    assert [request for request in session.requests if request.url.startswith(REDIRECT_TARGET)] == []


# --- the size and the checksums ------------------------------------------------------------------------------------


def test_run_import_when_google_sends_twice_the_declared_bytes_should_end_size_exceeded_within_one_chunk():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.media.append(Serve(extra=ONE.size)))

    # Assert: it stopped reading at the chunk that went past the size, and wrote nothing
    assert output == failed("size_exceeded")
    assert session.bytes_served <= ONE.size + DOWNLOAD_CHUNK_BYTES
    assert session.of("PUT") == []


def test_run_import_when_google_sends_bytes_past_the_last_part_should_never_write_the_last_part():
    # Act
    output, session, _ = _run(THREE, script=lambda session: session.media.append(Serve(extra=1)))

    # Assert
    assert output == failed("size_exceeded")
    assert [put.url for put in session.part_puts] == [part_url(1), part_url(2)]
    assert session.receipt_puts == []


def test_run_import_when_google_sends_fewer_bytes_than_declared_should_end_size_short_without_the_last_part():
    # Arrange: Google's stream ends cleanly, one byte short
    short = DriveFile(THREE.size - 1)

    # Act
    output, session, _ = _run(THREE, served=short, metadata=THREE.metadata())

    # Assert
    assert output == failed("size_short")
    assert [put.url for put in session.part_puts] == [part_url(1), part_url(2)]
    assert session.receipt_puts == []


@pytest.mark.parametrize(
    "carried",
    [pytest.param({"sha256": True}, id="against-the-sha256"), pytest.param({"sha256": False}, id="against-the-md5")],
)
def test_run_import_when_the_stream_does_not_match_googles_checksum_should_end_integrity_failed_without_the_last_part(
    carried,
):
    # Arrange: Google serves other bytes of the same size under the metadata of the pasted file
    document = import_input(THREE, **carried)

    # Act
    output, session, _ = _run(THREE, document=document, served=OTHER_THREE, metadata=THREE.metadata())

    # Assert: the parts before the last were written as they came; the last never, nor any receipt
    assert output == failed("integrity_failed")
    assert [put.url for put in session.part_puts] == [part_url(1), part_url(2)]
    assert session.receipt_puts == []


def test_run_import_when_a_one_part_file_does_not_match_its_checksum_should_write_nothing():
    # Act
    output, session, _ = _run(ONE, served=OTHER_ONE, metadata=ONE.metadata())

    # Assert
    assert output == failed("integrity_failed")
    assert session.of("PUT") == []


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
def test_run_import_when_google_refuses_the_metadata_should_end_with_its_code_before_any_download(answer, code):
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.metadata.append(answer()))

    # Assert
    assert output == failed(code)
    assert [request.url for request in session.requests] == [METADATA_URL]


@pytest.mark.parametrize(("answer", "code"), GOOGLE_REFUSALS)
def test_run_import_when_google_refuses_the_download_should_end_with_its_code_and_write_nothing(answer, code):
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.media.append(answer()))

    # Assert: one request of the bytes, no try again
    assert output == failed(code)
    assert len(session.media_gets) == 1
    assert session.of("PUT") == []


def test_run_import_when_google_refuses_a_resumed_download_should_end_with_its_code_and_write_no_more():
    # Arrange: a reset in the second part, then Google's quota on the request that resumes it
    def script(session: FakeSession) -> None:
        session.media += [Serve(break_after=PART_SIZE + MIB), google_error(403, "downloadQuotaExceeded")]

    # Act
    output, session, _ = _run(THREE, script=script)

    # Assert
    assert output == failed("provider_quota")
    assert [put.url for put in session.part_puts] == [part_url(1)]


# --- the metadata again before the first write (G-10) --------------------------------------------------------------


# In _changed_metadata(): the field is left out (where None sets it to null).
REMOVED = object()


def _changed_metadata(**changes: Any) -> dict[str, Any]:
    metadata = THREE.metadata()
    for name, value in changes.items():
        if value is REMOVED:
            metadata.pop(name)
        else:
            metadata[name] = value
    return metadata


@pytest.mark.parametrize(
    ("metadata", "document", "code"),
    [
        pytest.param(_changed_metadata(size=str(THREE.size + 1)), OWN_INPUT, "source_changed_at_provider",
                     id="another-size"),
        pytest.param(_changed_metadata(size=REMOVED), OWN_INPUT, "source_changed_at_provider", id="no-size"),
        pytest.param(_changed_metadata(size="a lot"), OWN_INPUT, "source_changed_at_provider",
                     id="a-size-that-is-no-number"),
        pytest.param(_changed_metadata(sha256Checksum="0" * 64), OWN_INPUT, "source_changed_at_provider",
                     id="the-recorded-sha256-changed"),
        pytest.param(_changed_metadata(md5Checksum="0" * 32), OWN_INPUT, "source_changed_at_provider",
                     id="the-recorded-md5-changed"),
        pytest.param(_changed_metadata(md5Checksum=REMOVED), OWN_INPUT, "source_changed_at_provider",
                     id="the-recorded-md5-no-longer-given"),
        pytest.param(_changed_metadata(sha1Checksum="0" * 40), import_input(THREE, sha1=True),
                     "source_changed_at_provider", id="the-recorded-sha1-changed"),
        pytest.param(_changed_metadata(headRevisionId="0BAnotherRevision"), OWN_INPUT, "source_changed_at_provider",
                     id="the-recorded-head-revision-changed"),
        pytest.param(_changed_metadata(trashed=True), OWN_INPUT, "file_removed", id="in-the-trash"),
        pytest.param(_changed_metadata(trashed=None), OWN_INPUT, "file_removed", id="in-the-trash-as-a-null"),
        pytest.param(_changed_metadata(capabilities={"canDownload": False}), OWN_INPUT, "link_not_shared",
                     id="no-longer-downloadable"),
        pytest.param(_changed_metadata(capabilities={"canDownload": None}), OWN_INPUT, "link_not_shared",
                     id="downloadable-as-a-null"),
    ],
)
def test_run_import_when_the_metadata_again_is_not_the_pasted_files_should_refuse_before_any_download(
    metadata, document, code
):
    # Act
    output, session, _ = _run(THREE, document=document, metadata=metadata)

    # Assert: nothing was downloaded, nothing written
    assert output == failed(code)
    assert [request.url for request in session.requests] == [METADATA_URL]


def test_run_import_when_the_recorded_checksum_comes_in_capitals_now_should_find_it_unchanged():
    # Act
    output, _, _ = _run(ONE, metadata=dict(ONE.metadata(), sha256Checksum=ONE.sha256.upper()))

    # Assert
    assert output == IMPORTED


def test_run_import_when_google_gives_values_now_that_the_input_did_not_carry_should_not_compare_them():
    # Arrange: the paste recorded no SHA-1, no SHA-256 and no head revision; Google now gives other ones
    document = import_input(ONE, sha1=False, sha256=False, head_revision=False)
    metadata = dict(ONE.metadata(), sha1Checksum="1" * 40, sha256Checksum="2" * 64, headRevisionId="0BNewRevision")

    # Act
    output, session, _ = _run(ONE, document=document, metadata=metadata)

    # Assert: imported, verified by the MD5 the input carried
    assert output == IMPORTED
    assert session.receipt()["checksum_verified"] == "md5"


def test_run_import_when_the_metadata_says_nothing_of_downloading_should_import_the_file():
    # Act
    output, _, _ = _run(ONE, metadata=_without(ONE.metadata(), "capabilities", "trashed"))

    # Assert
    assert output == IMPORTED


def _without(document: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: value for name, value in document.items() if name not in names}


def test_run_import_when_the_metadata_is_no_json_object_should_end_download_failed():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.metadata.append(json_answer(200, ["a", "list"])))

    # Assert
    assert output == failed("download_failed")
    assert session.media_gets == []


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(lambda: FakeResponse(500), id="500"),
        pytest.param(lambda: FakeResponse(503), id="503"),
        pytest.param(lambda: FakeResponse(408), id="408"),
        pytest.param(lambda: Fail(), id="no-answer"),
        pytest.param(lambda: Fail(requests.exceptions.ReadTimeout("read timed out")), id="a-time-out"),
    ],
)
def test_run_import_when_the_metadata_fails_once_should_ask_again_after_a_wait_and_import(first):
    # Act
    output, session, clock = _run(ONE, script=lambda session: session.metadata.append(first()))

    # Assert
    assert output == IMPORTED
    assert len(session.of("GET", METADATA_URL)) == 2
    assert clock.slept == [0.5]


def test_run_import_when_the_metadata_fails_three_times_should_end_download_failed():
    # Arrange
    def script(session: FakeSession) -> None:
        session.metadata.extend(FakeResponse(500) for _ in range(3))

    # Act
    output, session, clock = _run(ONE, script=script)

    # Assert
    assert output == failed("download_failed")
    assert len(session.of("GET", METADATA_URL)) == 3 and session.media_gets == []
    assert clock.slept == [0.5, 2.0]


# --- the download resumed (G-17) -----------------------------------------------------------------------------------


def test_run_import_when_the_stream_resets_in_a_part_should_resume_from_that_parts_first_byte_and_import():
    # Arrange: the stream breaks 5 MiB into the second part
    def script(session: FakeSession) -> None:
        session.media.append(Serve(break_after=PART_SIZE + 5 * MIB))

    # Act
    output, session, clock = _run(THREE, script=script)

    # Assert: the second request asks from the second part's first byte; every part holds its own bytes
    first, resumed = session.media_gets
    assert output == IMPORTED
    assert "Range" not in first.headers and resumed.headers["Range"] == f"bytes={PART_SIZE}-"
    assert [put.md5 for put in session.part_puts] == [THREE.part_md5(n) for n in (1, 2, 3)]
    assert clock.slept == [0.5]


def test_run_import_when_the_stream_resets_in_the_first_part_should_download_again_from_the_start():
    # Act
    output, session, _ = _run(THREE, script=lambda session: session.media.append(Serve(break_after=3 * MIB)))

    # Assert
    assert output == IMPORTED
    assert ["Range" in request.headers for request in session.media_gets] == [False, False]
    assert [put.md5 for put in session.part_puts] == [THREE.part_md5(n) for n in (1, 2, 3)]


def test_run_import_when_the_stream_resets_three_times_in_one_part_should_end_download_failed():
    # Arrange: three breaks in the second part, each of its three requests
    def script(session: FakeSession) -> None:
        session.media += [Serve(break_after=PART_SIZE + MIB), Serve(break_after=MIB), Serve(break_after=MIB)]

    # Act
    output, session, clock = _run(THREE, script=script)

    # Assert: the first part was written, the second never; no fourth request
    assert output == failed("download_failed")
    assert len(session.media_gets) == 3
    assert [put.url for put in session.part_puts] == [part_url(1)]
    assert session.receipt_puts == []
    assert clock.slept == [0.5, 2.0]


def test_run_import_when_two_parts_each_reset_twice_should_count_each_parts_tries_apart_and_import():
    # Arrange: the first part breaks twice; its third request goes on into the second part, which breaks twice too
    def script(session: FakeSession) -> None:
        session.media += [
            Serve(break_after=MIB),
            Serve(break_after=MIB),
            Serve(break_after=PART_SIZE + MIB),
            Serve(break_after=MIB),
        ]

    # Act
    output, session, _ = _run(THREE, script=script)

    # Assert
    assert output == IMPORTED
    assert [request.headers.get("Range") for request in session.media_gets] == [
        None, None, None, f"bytes={PART_SIZE}-", f"bytes={PART_SIZE}-"
    ]


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(lambda: FakeResponse(500), id="500"),
        pytest.param(lambda: FakeResponse(502), id="502"),
        pytest.param(lambda: FakeResponse(408), id="408"),
        pytest.param(lambda: Fail(), id="no-answer"),
        pytest.param(lambda: Fail(requests.exceptions.ConnectTimeout("connect timed out")), id="a-time-out"),
    ],
)
def test_run_import_when_the_download_fails_before_its_first_byte_should_ask_again_and_import(first):
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.media.append(first()))

    # Assert
    assert output == IMPORTED
    assert len(session.media_gets) == 2


def test_run_import_when_the_download_fails_three_times_before_its_first_byte_should_end_download_failed():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.media.extend(FakeResponse(503) for _ in range(3)))

    # Assert
    assert output == failed("download_failed")
    assert len(session.media_gets) == 3


@pytest.mark.parametrize(
    "answers",
    [
        pytest.param([Serve(break_after=PART_SIZE + MIB), Serve(offset=PART_SIZE + 1)], id="a-206-from-another-byte"),
        pytest.param([Serve(break_after=PART_SIZE + MIB), Serve(status=200)], id="a-200-to-a-resumed-request"),
        pytest.param([Serve(status=206)], id="a-206-to-the-first-request"),
        pytest.param([FakeResponse(204)], id="a-204"),
    ],
)
def test_run_import_when_the_download_answers_another_range_than_asked_should_end_download_failed(answers):
    # Act
    output, session, _ = _run(THREE, script=lambda session: session.media.extend(answers))

    # Assert
    assert output == failed("download_failed")
    assert len(session.media_gets) == len(answers)
    assert session.receipt_puts == []


def test_run_import_when_google_sends_the_bytes_in_a_content_coding_should_end_download_failed():
    # Arrange
    def script(session: FakeSession) -> None:
        session.media.append(FakeResponse(200, {"Content-Encoding": "gzip"}, body=b"\x1f\x8b compressed"))

    # Act
    output, session, _ = _run(ONE, script=script)

    # Assert
    assert output == failed("download_failed")
    assert session.of("PUT") == []


# --- the parts written (G-18) --------------------------------------------------------------------------------------


def _puts_of_part(n: int, *answers: Put | Fail | FakeResponse) -> Callable[[FakeSession], None]:
    def script(session: FakeSession) -> None:
        session.puts.setdefault(part_url(n), []).extend(answers)

    return script


def test_run_import_when_a_put_fails_twice_with_500_should_write_it_on_its_third_try_and_import():
    # Act
    output, session, clock = _run(ONE, script=_puts_of_part(1, Put(500), Put(500)))

    # Assert
    assert output == IMPORTED
    assert len(session.of("PUT", part_url(1))) == 3
    assert clock.slept == [0.5, 2.0]
    assert session.receipt()["parts"] == [{"n": 1, "etag": ONE.part_md5(1)}]


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(Put(503), id="503"),
        pytest.param(Put(408), id="408"),
        pytest.param(Put(429), id="429"),
        pytest.param(Fail(), id="no-answer"),
        pytest.param(Fail(requests.exceptions.ReadTimeout("read timed out")), id="a-time-out"),
    ],
)
def test_run_import_when_a_put_fails_once_in_a_way_worth_another_try_should_write_it_again(first):
    # Act
    output, session, _ = _run(ONE, script=_puts_of_part(1, first))

    # Assert
    assert output == IMPORTED
    assert len(session.of("PUT", part_url(1))) == 2


def test_run_import_when_a_put_fails_four_times_should_end_upload_failed_after_the_three_waits():
    # Act
    output, session, clock = _run(ONE, script=_puts_of_part(1, Put(500), Put(500), Put(500), Put(500)))

    # Assert
    assert output == failed("upload_failed")
    assert len(session.of("PUT", part_url(1))) == 4
    assert clock.slept == [0.5, 2.0, 8.0]
    assert session.receipt_puts == []


@pytest.mark.parametrize("status", [400, 403, 404, 409, 411, 201, 204])
def test_run_import_when_the_store_refuses_a_put_should_end_upload_failed_at_once(status):
    # Act
    output, session, clock = _run(ONE, script=_puts_of_part(1, Put(status)))

    # Assert
    assert output == failed("upload_failed")
    assert len(session.part_puts) == 1 and session.receipt_puts == []
    assert clock.slept == []


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-etag"),
        pytest.param({"ETag": '""'}, id="an-empty-etag"),
        pytest.param({"ETag": '"an etag"'}, id="an-etag-with-a-space"),
        pytest.param({"ETag": 'W/"weak"'}, id="a-weak-etag"),
        pytest.param({"ETag": '"' + "a" * 129 + '"'}, id="an-etag-too-long"),
        pytest.param({"ETag": '"tab\there"'}, id="an-etag-with-a-control-character"),
    ],
)
def test_run_import_when_a_puts_200_carries_no_etag_of_the_receipts_form_should_end_upload_failed(headers):
    # Act
    output, session, _ = _run(ONE, script=_puts_of_part(1, Put(200, headers=headers)))

    # Assert
    assert output == failed("upload_failed")
    assert session.receipt_puts == []


def test_run_import_when_the_store_gives_an_etag_without_quotes_should_keep_it_as_it_is():
    # Act
    output, session, _ = _run(ONE, script=_puts_of_part(1, Put(200, headers={"etag": "abc-123"})))

    # Assert
    assert output == IMPORTED
    assert session.receipt()["parts"] == [{"n": 1, "etag": "abc-123"}]


def test_run_import_when_the_receipt_cannot_be_written_should_end_upload_failed():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.puts.setdefault(RECEIPT_URL, []).append(Put(403)))

    # Assert
    assert output == failed("upload_failed")
    assert len(session.receipt_puts) == 1


def test_run_import_when_the_receipt_put_fails_once_with_500_should_write_it_again_and_import():
    # Act
    output, session, _ = _run(ONE, script=lambda session: session.puts.setdefault(RECEIPT_URL, []).append(Put(500)))

    # Assert
    assert output == IMPORTED
    assert len(session.receipt_puts) == 2


# --- the stop line (writes_end_at − 120 s) -------------------------------------------------------------------------


def test_run_import_when_the_stop_line_passes_before_the_last_part_should_end_deadline_without_it_or_a_receipt():
    # Arrange: the second part's PUT takes the clock past the stop line
    clock = FakeClock()
    passing = clock.stop_line_in() + 1

    # Act
    output, session, _ = _run(THREE, clock=clock, script=_puts_of_part(2, Put(seconds_after=passing)))

    # Assert: what was written stays; nothing more is asked
    assert output == failed("deadline")
    assert [put.url for put in session.part_puts] == [part_url(1), part_url(2)]
    assert session.receipt_puts == []
    assert session.requests[-1].url == part_url(2)


def test_run_import_when_the_stop_line_passes_while_a_part_is_sent_should_cut_its_body_and_end_deadline():
    # Arrange: each block of the first part's body takes a second, and the stop line is 30 s away
    clock = FakeClock()
    document = import_input(ONE, writes_end_at=START + timedelta(seconds=150))

    # Act
    output, session, _ = _run(ONE, document=document, clock=clock, script=_puts_of_part(1, Put(seconds_per_block=1.0)))

    # Assert: the body stopped at the stop line, and nothing was asked after
    (put,) = session.of("PUT")
    assert output == failed("deadline")
    assert put.cut and put.length < ONE.size
    assert session.requests[-1] is put


def test_run_import_when_the_stop_line_has_passed_at_the_start_should_end_deadline_without_a_request():
    # Arrange: the writer deadline is 100 s away, the stop line 20 s behind
    document = import_input(ONE, writes_end_at=START + timedelta(seconds=100))

    # Act
    output, session, _ = _run(ONE, document=document)

    # Assert
    assert output == failed("deadline")
    assert session.requests == []


def test_run_import_when_a_wait_before_another_try_would_pass_the_stop_line_should_end_deadline_without_waiting():
    # Arrange: the stop line is 0.4 s away, and the first PUT fails
    document = import_input(ONE, writes_end_at=START + timedelta(seconds=120.4))

    # Act
    output, session, clock = _run(ONE, document=document, script=_puts_of_part(1, Put(500)))

    # Assert
    assert output == failed("deadline")
    assert len(session.of("PUT")) == 1 and clock.slept == []


def test_run_import_when_the_stop_line_is_near_should_cut_every_requests_time_outs_to_what_is_left():
    # Arrange: the stop line is 30 s away
    document = import_input(THREE, writes_end_at=START + timedelta(seconds=150))

    # Act
    output, session, _ = _run(THREE, document=document)

    # Assert
    assert output == IMPORTED
    assert {request.timeout for request in session.requests} == {(10.0, 30.0)}


def test_run_import_when_a_chunk_comes_after_the_stop_line_should_end_deadline_without_writing_it():
    # Arrange: the stop line passes while the stream is read
    clock = FakeClock()

    def chunks(chunk_size: int):
        yield ONE.bytes(0, MIB // 2)
        clock.advance(clock.stop_line_in() + 1)
        yield ONE.bytes(MIB // 2)

    # Act
    output, session, _ = _run(
        ONE, clock=clock, script=lambda session: session.media.append(FakeResponse(200, chunks=chunks))
    )

    # Assert
    assert output == failed("deadline")
    assert session.of("PUT") == []


# --- the configuration and the input -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param({"GOOGLE_DRIVE_API_KEY": None}, id="no-google-key"),
        pytest.param({"IMPORT_RECEIPT_KEY": None}, id="no-receipt-key"),
        pytest.param({"IMPORT_STORAGE_HOST": None}, id="no-storage-host"),
    ],
)
def test_run_import_when_the_configuration_is_not_whole_should_end_not_configured_without_a_request(missing):
    # Act
    output, session, _ = _run(ONE, configuration=settings(**missing))

    # Assert
    assert output == failed("not_configured")
    assert session.requests == []


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(None, id="no-input"),
        pytest.param([], id="a-list"),
        pytest.param("import/v1", id="a-string"),
        pytest.param(dict(import_input(ONE), api_key=API_KEY), id="a-member-the-contract-does-not-name"),
        pytest.param(dict(import_input(ONE), contract="import/v2"), id="another-contract"),
        pytest.param(_without(import_input(ONE), "receipt_url"), id="no-receipt-url"),
    ],
)
def test_run_import_when_the_input_is_not_the_contracts_should_end_input_refused_without_a_request(document):
    # Act
    output, session, _ = _run(ONE, document=document, configuration=settings(GOOGLE_DRIVE_API_KEY=None))

    # Assert: the schema is checked first, before the configuration
    assert output == failed("input_refused")
    assert session.requests == []


def test_run_import_when_a_url_points_at_another_store_should_end_input_refused_without_a_request():
    # Arrange: signed again, so that the URL's own rule refuses it, not the MAC
    document = import_input(ONE)
    document["parts"][0]["url"] = document["parts"][0]["url"].replace("storage.example.invalid", "evil.example.invalid")
    document = signed(document)

    # Act
    output, session, _ = _run(ONE, document=document)

    # Assert
    assert output == failed("input_refused")
    assert session.requests == []


def test_run_import_when_the_configuration_is_not_whole_and_a_url_is_wrong_should_say_not_configured():
    # Arrange: the URLs are checked against the configured store, so a missing store comes first
    document = import_input(ONE)
    document["parts"][0]["url"] = document["parts"][0]["url"].replace("storage.example.invalid", "evil.example.invalid")

    # Act
    output, _, _ = _run(ONE, document=document, configuration=settings(IMPORT_STORAGE_HOST=None))

    # Assert
    assert output == failed("not_configured")


def test_run_import_when_the_writer_deadline_has_passed_should_end_input_refused():
    # Arrange
    document = import_input(ONE, writes_end_at=START - timedelta(seconds=1))

    # Act
    output, session, _ = _run(ONE, document=document)

    # Assert
    assert output == failed("input_refused")
    assert session.requests == []


# --- the input's MAC (README section 3.1) --------------------------------------------------------------------------


def _changed(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """A signed input of THREE with `change` made after it was signed: its MAC is the one of the input before."""
    document = import_input(THREE)
    change(document)
    return document


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        target = document
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return change


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(_changed(_set(("file_id",), "1AnotherFileOfTheSameSize0123456")), id="another-drive-file"),
        pytest.param(_changed(_set(("expected", "sha256"), OTHER_THREE.sha256)), id="another-expected-sha256"),
        pytest.param(_changed(_set(("expected", "md5"), None)), id="no-expected-md5"),
        pytest.param(_changed(_set(("resource_key",), "0-AnotherResourceKey")), id="a-resource-key-added"),
        pytest.param(_changed(_set(("parts", 0, "url"), part_url(1) + "&x=1")), id="another-part-url"),
        pytest.param(_changed(_set(("receipt_url",), RECEIPT_URL + "&x=1")), id="another-receipt-url"),
        pytest.param(_changed(_set(("writes_end_at",), "2026-10-17T10:00:00Z")), id="a-later-deadline"),
        pytest.param(_changed(_set(("mac",), "0" * 64)), id="another-mac"),
        pytest.param(signed(import_input(THREE), key="another-import-key-of-32-characters-or-more"),
                     id="signed-with-another-key"),
    ],
)
def test_run_import_when_the_input_is_not_the_one_the_hub_signed_should_end_input_refused_without_a_request(document):
    # Act
    output, session, _ = _run(THREE, document=document)

    # Assert: nothing of the input was acted on
    assert output == failed("input_refused")
    assert session.requests == []


def test_run_import_when_the_input_has_another_files_facts_under_a_real_jobs_urls_should_be_refused():
    # Arrange: SEC-01's case: a job's own URLs, deadline and uuid, with another file of the same size and its
    # checksums, which the worker would otherwise verify against and sign a receipt for
    document = import_input(THREE)
    document["file_id"] = "1AnotherFileOfTheSameSize0123456"
    document["expected"]["md5"] = OTHER_THREE.md5
    document["expected"]["sha256"] = OTHER_THREE.sha256

    # Act
    output, session, _ = _run(THREE, document=document, served=OTHER_THREE, metadata=OTHER_THREE.metadata())

    # Assert
    assert output == failed("input_refused")
    assert session.requests == [] and session.receipt_puts == []


def test_run_import_when_the_receipt_key_is_missing_should_end_not_configured_before_the_mac_is_read():
    # Act: an input with a MAC that would not verify, and no key to check it with
    output, session, _ = _run(
        ONE, document=dict(import_input(ONE), mac="0" * 64), configuration=settings(IMPORT_RECEIPT_KEY=None)
    )

    # Assert
    assert output == failed("not_configured")
    assert session.requests == []


# --- a try that ends after the stop line is the stop line's ----------------------------------------------------------


def test_run_import_when_the_last_download_try_breaks_after_the_stop_line_should_end_deadline():
    # Arrange: the part's third request breaks once the clock has passed the stop line
    clock = FakeClock()
    passing = clock.stop_line_in() + 1

    def script(session: FakeSession) -> None:
        session.media += [
            Serve(break_after=PART_SIZE + MIB),
            Serve(break_after=MIB),
            Serve(break_after=MIB, seconds_before_break=passing),
        ]

    # Act
    output, session, _ = _run(THREE, clock=clock, script=script)

    # Assert
    assert output == failed("deadline")
    assert len(session.media_gets) == 3


def test_run_import_when_the_last_put_try_fails_after_the_stop_line_should_end_deadline():
    # Arrange
    clock = FakeClock()
    passing = clock.stop_line_in() + 1

    # Act
    output, session, _ = _run(
        ONE, clock=clock, script=_puts_of_part(1, Put(500), Put(500), Put(500), Put(500, seconds_after=passing))
    )

    # Assert
    assert output == failed("deadline")


def test_run_import_when_the_metadata_fails_after_the_stop_line_should_end_deadline():
    # Arrange: the third try's 503 comes once the clock has passed the stop line
    clock = FakeClock()
    passing = clock.stop_line_in() + 1

    def script(session: FakeSession) -> None:
        session.metadata += [FakeResponse(503), FakeResponse(503), Late(FakeResponse(503), passing)]

    # Act
    output, _, _ = _run(ONE, clock=clock, script=script)

    # Assert
    assert output == failed("deadline")


def test_run_import_when_a_download_ends_cleanly_after_the_stop_line_should_end_deadline_not_size_short():
    # Arrange: a body framed by the connection's close, which the watchdog's shutdown ends without an error
    clock = FakeClock()

    def chunks(chunk_size: int):
        yield ONE.bytes(0, MIB // 2)
        clock.advance(clock.stop_line_in() + 1)

    # Act
    output, _, _ = _run(
        ONE, clock=clock, script=lambda session: session.media.append(FakeResponse(200, chunks=chunks))
    )

    # Assert
    assert output == failed("deadline")


def test_run_import_when_it_asks_should_hand_every_request_its_stop_line():
    # Act
    _, session, clock = _run(THREE)

    # Assert: one stop line for the job, the writer deadline less 120 s, handed with every request
    assert len(session.stop_lines) == len(session.requests)
    assert {round(stop_line.left_s(), 3) for stop_line in session.stop_lines} == {round(clock.stop_line_in(), 3)}


def test_run_import_when_the_session_was_given_should_never_close_it():
    # Act: the entry point owns the session; the job only uses it
    _, session, _ = _run(ONE)

    # Assert
    assert not session.closed
    assert [answer.closed for answer in session.answers] == [True] * len(session.answers)


def test_drive_file_when_made_lazily_should_hash_as_its_whole_bytes():
    # Act / Assert: the test's own file stand-in is what its checksums say
    assert hashlib.sha256(ONE.bytes()).hexdigest() == ONE.sha256
    assert hashlib.md5(ONE.bytes()).hexdigest() == ONE.md5 == ONE.part_md5(1)
    assert ONE.metadata()["headRevisionId"] == HEAD_REVISION
