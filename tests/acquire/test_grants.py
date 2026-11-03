"""Grant authority is checked in full before any PUT capability can be used."""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.protocol.acquire_claim import AcquireGrant, claim_execution
from core.protocol.acquire_dispatch import accept_dispatch
from core.protocol.transport import HttpAnswer
from tests.acquire.support import MAX_BYTES, PART_SIZE, SIGNATURE_CANARY, Scenario, wire
from tests.contract.v2_support import HUB_KEY, RUNTIME_ID, signed_carrier

URL_REJECTIONS = {
    "http": ("https://", "http://"),
    "other-host": ("storage.example.invalid", "other.example.invalid"),
    "uppercase-host": ("storage.example.invalid", "STORAGE.example.invalid"),
    "port": ("storage.example.invalid/", "storage.example.invalid:443/"),
    "userinfo": ("https://", "https://secret@"),
    "fragment": ("&partNumber=1", "&partNumber=1#secret"),
    "backslash": ("/original.mp4", "/original\\.mp4"),
    "brackets": ("storage.example.invalid", "[storage.example.invalid]"),
    "encoded-path": ("/sources/", "/%73ources/"),
    "traversal": ("/sources/", "/sources/../"),
    "alternate-key": ("/original.mp4", "/original.mov"),
    "invalid-bucket": ("/test-source-bucket/", "/invalid_bucket/"),
    "missing-query": ("?", "/"),
    "bad-percent": ("Opaque%2BId", "Opaque%2XId"),
    "repeated-name": ("&partNumber=1", "&partNumber=1&partNumber=1"),
    "encoded-duplicate-name": ("&partNumber=1", "&partNumber=1&part%4Eumber=1"),
    "wrong-part": ("&partNumber=1", "&partNumber=2"),
    "leading-zero-part": ("&partNumber=1", "&partNumber=01"),
    "empty-upload": ("uploadId=Opaque%2BId", "uploadId="),
    "no-upload": ("&uploadId=Opaque%2BId", ""),
    "invalid-date": ("20300101T000000Z", "20300230T000000Z"),
    "future-date": ("20300101T000000Z", "20300101T001000Z"),
    "wrong-expiry": ("X-Amz-Expires=1320", "X-Amz-Expires=1321"),
    "zero-expiry": ("X-Amz-Expires=1320", "X-Amz-Expires=0"),
    "leading-zero-expiry": ("X-Amz-Expires=1320", "X-Amz-Expires=01320"),
    "over-week-expiry": ("X-Amz-Expires=1320", "X-Amz-Expires=604801"),
    "no-expiry": ("X-Amz-Expires=1320&", ""),
    "no-date": ("X-Amz-Date=20300101T000000Z&", ""),
    "duplicate-date": ("X-Amz-Date=20300101T000000Z", "X-Amz-Date=20300101T000000Z&X-Amz-Date=20300101T000000Z"),
    "no-equals": ("&partNumber=1", "&partNumber"),
    "empty-name": ("&partNumber=1", "&partNumber=1&=value"),
    "space": (SIGNATURE_CANARY, "space in signature"),
}


@pytest.mark.parametrize("rule", URL_REJECTIONS)
def test_acquire_grant_when_a_part_url_breaks_authority_should_report_invalid_answer_and_do_no_work(rule: str) -> None:
    # Arrange
    scenario = Scenario()
    before, after = URL_REJECTIONS[rule]

    def change(grant: dict[str, Any]) -> None:
        url = grant["upload"]["parts"][0]["url"]
        assert before in url
        grant["upload"]["parts"][0]["url"] = url.replace(before, after)

    scenario.hub.modify_grant = change
    # Act
    report = json.loads(scenario.run()["vh_message"])
    # Assert
    assert report["message_kind"] == "transcode.unclaimed"
    assert report["cause"] == "invalid_answer"
    assert scenario.http.requests == []


@pytest.mark.parametrize("condition", [
    "result-bucket", "result-key", "result-upload", "result-part", "result-expiry", "result-host", "result-percent",
    "location", "prefix", "runtime", "execution-alias", "version", "empty-parts", "duplicate-part",
    "float-part", "float-fence", "too-many-parts", "unordered-parts", "mixed-bucket", "mixed-upload",
])
def test_acquire_grant_when_a_binding_is_invalid_should_discard_all_capabilities(condition: str) -> None:
    # Arrange
    scenario = Scenario()
    if condition in ("unordered-parts", "mixed-bucket", "mixed-upload"):
        scenario.request["external"]["expected"]["size"] = PART_SIZE + 1
        scenario.request["plan"]["parts_total"] = 2

    def change(grant: dict[str, Any]) -> None:
        upload = grant["upload"]
        if condition.startswith("result-"):
            old, new = {
                "result-bucket": ("/test-source-bucket/", "/other-bucket/"),
                "result-key": ("/acquire-result-", "/result-"),
                "result-upload": ("?", "?uploadId=Opaque%2BId&"),
                "result-part": ("?", "?partNumber=1&"),
                "result-expiry": ("X-Amz-Expires=1320", "X-Amz-Expires=1321"),
                "result-host": ("storage.example.invalid", "other.invalid"),
                "result-percent": (SIGNATURE_CANARY, "%xy"),
            }[condition]
            upload["result_url"] = upload["result_url"].replace(old, new)
        elif condition == "location":
            grant["output"]["location_id"] = grant["identity"]["dispatch_id"]
        elif condition == "prefix":
            grant["output"]["prefix"] += "/other"
        elif condition == "runtime":
            grant["runtime_id"] = RUNTIME_ID
        elif condition == "execution-alias":
            grant["identity"]["execution_id"] = grant["identity"]["video_uuid"]
        elif condition == "version":
            grant["protocol_version"] = "2.0.0"
        elif condition == "empty-parts":
            upload["parts"] = []
        elif condition == "duplicate-part":
            upload["parts"] *= 2
        elif condition == "float-part":
            upload["parts"][0]["n"] = 1.0
        elif condition == "float-fence":
            grant["identity"]["execution_fence"] = 2.0
        elif condition == "too-many-parts":
            upload["parts"] *= 161
        elif condition == "unordered-parts":
            upload["parts"].reverse()
        else:
            old, new = (("/test-source-bucket/", "/other-bucket/") if condition == "mixed-bucket"
                        else ("uploadId=Opaque%2BId", "uploadId=Different"))
            upload["parts"][1]["url"] = upload["parts"][1]["url"].replace(old, new)

    scenario.hub.modify_grant = change
    # Act
    report = json.loads(scenario.run()["vh_message"])
    # Assert
    assert report["cause"] == "invalid_answer"
    assert scenario.http.requests == []


def test_acquire_claim_when_the_grant_has_160_maximum_urls_should_accept_with_only_the_acquire_answer_cap() -> None:
    # Arrange
    scenario = Scenario()
    scenario.request["external"]["expected"]["size"] = MAX_BYTES
    scenario.request["plan"]["parts_total"] = 160
    dispatch = accept_dispatch(signed_carrier(wire(scenario.request), HUB_KEY), settings=scenario.runtime.settings, now=scenario.clock.now())

    def maximize(grant: dict[str, Any]) -> None:
        for part in grant["upload"]["parts"]:
            part["url"] += "&opaque=" + "/" * (4096 - len(part["url"]) - len("&opaque="))
        url = grant["upload"]["result_url"]
        grant["upload"]["result_url"] += "&opaque=" + "/" * (4096 - len(url) - len("&opaque="))

    scenario.hub.modify_grant = maximize
    # Act
    grant = claim_execution(
        dispatch, settings=scenario.runtime.settings, runtime_id=RUNTIME_ID, transport=scenario.hub,
        clock=scenario.clock.now, monotonic=scenario.clock.monotonic, sleep=scenario.clock.sleep,
        new_message_id=scenario.runtime.new_uuid, wall_time_ends_at=scenario.clock.monotonic() + 60,
    )
    # Assert
    assert isinstance(grant, AcquireGrant)
    assert len(grant.parts) == 160
    assert all(len(part.url) == 4096 for part in grant.parts)
    assert len(wire(scenario.hub.grants[0])) > 4096
    assert scenario.hub.claims[0].cap == 2_097_152
    assert not scenario.http.requests
