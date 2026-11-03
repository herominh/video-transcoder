"""Signed Hub answers and Google/storage fakes; no external traffic."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from core.drive_import.clock import StopLine
from core.protocol import files, signing
from core.protocol.messages import timestamp
from core.protocol.runtime import Runtime
from core.protocol.settings import AcquireSettings
from core.protocol.transport import HttpAnswer
from tests.contract.v2_support import (
    CALLBACK_URL, CLAIM_URL, FIXTURE_NOW, HUB_KEY, WORKER_KEY, signed_carrier, uuid_source, worker_env,
)
from tests.drive_import.fakes import (
    API_KEY, FILE_ID, HEAD_REVISION, RESOURCE_KEY, STORAGE_HOST, DriveFile, FakeClock, FakeResponse, FakeSession,
)
from wrappers.runpod_acquire_handler import job_handler

PART_SIZE = 67_108_864
MAX_BYTES = 10_737_418_240
WALL_MS = 60_000
EXPIRY_S = 60
WRITE_ALLOWANCE_S = 1_200
EXECUTION = "55555555-5555-7555-8555-555555555555"
SIGNATURE_CANARY = "ACQUIRE-PUT-CAPABILITY-CANARY"


def environment() -> dict[str, str]:
    env = {name: value for name, value in worker_env().items() if not name.startswith("VH_STORAGE_")}
    env.update(VH_GOOGLE_DRIVE_API_KEY=API_KEY, VH_ACQUIRE_STORAGE_HOST=STORAGE_HOST, VH_CLOCK_SKEW_S="300")
    return env


def settings() -> AcquireSettings:
    # Draft support is explicit in this test's object, never in runtime environment parsing.
    return replace(AcquireSettings.from_env(environment()), accepted_protocol_versions=("2.0.0", "2.1.0"))


def wire(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def request_for(file: DriveFile) -> dict[str, Any]:
    document = files.read_json("fixtures/acquire-request/P30-minimum.json")
    document["external"] = {
        "provider": "google_drive", "file_id": FILE_ID, "resource_key": RESOURCE_KEY,
        "expected": {"size": file.size, "md5": file.md5, "sha1": file.sha1, "sha256": file.sha256,
                     "head_revision_id": HEAD_REVISION},
    }
    document["plan"]["parts_total"] = (file.size + PART_SIZE - 1) // PART_SIZE
    document["limits"]["max_wall_time_ms"] = WALL_MS
    document["claim"]["expires_at"] = timestamp(FIXTURE_NOW + timedelta(seconds=EXPIRY_S))
    return document


def grant_for(request: Mapping[str, Any], runtime_id: str, now: datetime, revision: int = 1) -> dict[str, Any]:
    grant = files.read_json("fixtures/acquire-grant/P35-one-part.json")
    grant["runtime_id"] = runtime_id
    grant["sent_at"] = timestamp(now)
    grant["identity"] = {**request["identity"], "execution_id": EXECUTION, "execution_fence": 2}
    prefix = request["destination"]["object_key"].rsplit("/", 1)[0]
    grant["output"] = {"location_id": request["destination"]["location_id"], "prefix": prefix}
    expires = EXPIRY_S + request["limits"]["max_wall_time_ms"] // 1000 + WRITE_ALLOWANCE_S
    elapsed = int((now - FIXTURE_NOW).total_seconds())
    query = (f"X-Amz-Date={now.strftime('%Y%m%dT%H%M%SZ')}&X-Amz-Expires={expires - elapsed}"
             f"&X-Amz-Signature={SIGNATURE_CANARY}-{revision}")
    base = f"https://{STORAGE_HOST}/test-source-bucket/"
    grant["upload"] = {
        "parts": [{"n": n, "url": f"{base}{request['destination']['object_key']}?{query}&uploadId=Opaque%2BId&partNumber={n}"}
                  for n in range(1, request["plan"]["parts_total"] + 1)],
        "result_url": f"{base}{prefix}/acquire-result-{EXECUTION}.json?{query}",
    }
    return grant


@dataclass(frozen=True)
class Post:
    url: str = field(repr=False)
    body: bytes = field(repr=False)
    signature: str = field(repr=False)
    timeout: float
    cap: int


class Hub:
    def __init__(self, request: dict[str, Any], clock: FakeClock) -> None:
        self.request = request
        self.clock = clock
        self.posts: list[Post] = []
        self.grants: list[dict[str, Any]] = []
        self.claim_actions: list[HttpAnswer | Exception | Callable[[dict[str, Any]], HttpAnswer]] = []
        self.callback_actions: list[HttpAnswer | Exception] = []
        self.modify_grant: Callable[[dict[str, Any]], None] = lambda grant: None

    def post(self, url: str, body: bytes, signature: str, *, timeout_s: float, max_body_bytes: int) -> HttpAnswer:
        assert signing.verify(body, WORKER_KEY, signature)
        self.posts.append(Post(url, body, signature, timeout_s, max_body_bytes))
        actions = self.callback_actions if url == CALLBACK_URL else self.claim_actions
        if url == CLAIM_URL:
            claim = json.loads(body)
            grant = grant_for(self.request, claim["claim"]["runtime_id"], self.clock.now(), len(self.grants) + 1)
            self.modify_grant(grant)
            self.grants.append(copy.deepcopy(grant))
            default = self.signed_grant(grant)
        else:
            assert url == CALLBACK_URL
            grant = {}
            default = HttpAnswer(200, b"", None)
        action = actions.pop(0) if actions else default
        if isinstance(action, Exception):
            raise action
        return action(grant) if callable(action) else action

    @staticmethod
    def signed_grant(grant: Mapping[str, Any]) -> HttpAnswer:
        raw = wire(grant)
        return HttpAnswer(200, raw, signing.sign(raw, HUB_KEY))

    @property
    def claims(self) -> list[Post]:
        return [post for post in self.posts if post.url == CLAIM_URL]

    @property
    def callbacks(self) -> list[Post]:
        return [post for post in self.posts if post.url == CALLBACK_URL]


class _Tee:
    def __init__(self, body: Any) -> None:
        self.body = body
        self.seen = bytearray()

    def read(self, size: int) -> bytes:
        block = self.body.read(size)
        self.seen.extend(block)
        return block


class Session(FakeSession):
    def put(self, url: str, data: Any, headers: Mapping[str, str], timeout: tuple[float, float], *,
            stop_line: StopLine) -> FakeResponse:
        if "acquire-result-" not in url:
            return super().put(url, data, headers, timeout, stop_line=stop_line)
        tee = _Tee(data)
        answer = super().put(url, tee, headers, timeout, stop_line=stop_line)
        self.requests[-1].body = bytes(tee.seen)
        return answer

    @property
    def part_puts(self) -> list[Any]:
        return [request for request in self.of("PUT") if "acquire-result-" not in request.url]

    @property
    def result_puts(self) -> list[Any]:
        return [request for request in self.of("PUT") if "acquire-result-" in request.url]


class Scenario:
    def __init__(self, size: int = 37) -> None:
        self.file = DriveFile(size)
        self.clock = FakeClock(FIXTURE_NOW)
        self.request = request_for(self.file)
        self.hub = Hub(self.request, self.clock)
        self.http = Session(self.file, self.clock)
        self.runtime = Runtime(settings(), self.hub, self.clock.now, self.clock.monotonic, self.clock.sleep, uuid_source())

    def run(self, carrier: object | None = None) -> dict[str, str]:
        value = signed_carrier(wire(self.request), HUB_KEY) if carrier is None else carrier
        return job_handler(self.runtime, lambda: self.http)({"id": "provider-id-is-not-a-runtime", "input": value})

    def urls(self) -> tuple[str, str]:
        grant = grant_for(self.request, "88888888-8888-7888-8888-888888888888", self.clock.now())
        return grant["upload"]["parts"][0]["url"], grant["upload"]["result_url"]
