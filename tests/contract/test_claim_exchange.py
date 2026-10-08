"""The three kinds of the claim exchange (README.md sections 7, 8 and 15): a grant binds the runtime that claimed (S29)
and a prefix whose last segment is the generation (S30); the worker takes the execution from the grant where the Hub's
self-check compares it with its records; an unclaimed report is read from whichever start sent it; a runtime id is
never the video's UUID (S7). The fixture catalog pins one case per check; these are the boundaries it leaves, and they
copy the Hub's ClaimExchangeTest row for row."""

import json
from typing import Any

import pytest

from tests.contract import files, pipeline
from tests.contract.context import TrustedContext
from tests.contract.reasons import Layer, ReasonCode
from tests.contract.registry import load_registry
from tests.contract.verdict import Verdict

CLAIM = "fixtures/claim/P23-claim-of-the-accepted-dispatch.json"
GRANT = "fixtures/claim-granted/P24-grant-to-the-claiming-runtime.json"
UNCLAIMED_REFUSED = "fixtures/unclaimed/P25-refused-with-claim-conflict.json"
UNCLAIMED_UNREACHABLE = "fixtures/unclaimed/P26-unreachable-polled-three-hours-late.json"
CLAIM_SENT_301S_AGO = "fixtures/claim/N120-sent-301s-ago.json"  # N120: the claim of P23 sent 301 s before now
# N123: the grant of P24 whose prefix goes on past the generation (`<generation>/hls`).
GRANT_PREFIX_PAST_THE_GENERATION = "fixtures/claim-granted/N123-prefix-not-ending-in-generation.json"
HUB_CLAIM = "fixtures/contexts/ctx-hub-claim.json"
WORKER_GRANT = "fixtures/contexts/ctx-worker-claim-grant.json"
HUB_SENDER_GRANT = "fixtures/contexts/ctx-hub-sender-claim-grant.json"
HUB_UNCLAIMED_PUSH = "fixtures/contexts/ctx-hub-unclaimed-push.json"
HUB_UNCLAIMED_POLL = "fixtures/contexts/ctx-hub-unclaimed-poll.json"
VIDEO = "22222222-2222-7222-8222-222222222222"
GENERATION = "55555555-5555-7555-8555-555555555555"
OTHER_EXECUTION = "56565656-5656-7656-8656-565656565656"
OTHER_RUNTIME = "89898989-8989-7989-8989-898989898989"


def _validate(message: dict[str, Any], context: dict[str, Any]) -> Verdict:
    raw = json.dumps(message, separators=(",", ":")).encode("ascii")
    return pipeline.validate(raw, TrustedContext.from_dict(context))


def _grant_for_execution(execution_id: str, fence: int) -> dict[str, Any]:
    """The catalog's grant for another execution: its identity, its generation and its prefix all name it."""
    grant = files.read_json(GRANT)
    grant["identity"] = {**grant["identity"], "execution_id": execution_id, "execution_fence": fence}
    grant["generation_id"] = execution_id
    grant["output"]["prefix"] = f"test/media/{VIDEO}/{execution_id}"
    return grant


def test_grant_when_its_prefix_is_the_generation_alone_should_be_accepted():
    # Arrange: a prefix without a slash is its own last segment.
    grant = files.read_json(GRANT)
    grant["output"]["prefix"] = GENERATION

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert verdict.accepted, verdict


@pytest.mark.parametrize(
    "prefix",
    [
        "media",
        f"{GENERATION}/hls",
        f"test/media/x{GENERATION}",
        f"test/media/{GENERATION}.tmp",
        f"test/media/{VIDEO}",
    ],
    ids=[
        "one segment that is not the generation",
        "the generation as the first segment only",
        "a last segment that only ends with the generation",
        "a last segment that only starts with the generation",
        "the video in place of the generation",
    ],
)
def test_grant_when_the_last_segment_of_its_prefix_is_not_the_generation_should_be_refused_at_the_prefix(prefix):
    # Arrange
    grant = files.read_json(GRANT)
    grant["output"]["prefix"] = prefix

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.SEMANTIC,
        ReasonCode.GENERATION_PREFIX_MISMATCH,
        "/output/prefix",
    )


def test_grant_when_its_runtime_and_its_prefix_are_both_wrong_should_report_the_runtime_first():
    # Arrange: S29 runs before S30.
    grant = files.read_json(GRANT)
    grant["runtime_id"] = OTHER_RUNTIME
    grant["output"]["prefix"] = f"test/media/{VIDEO}"

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert (verdict.reason, verdict.instance_path) == (ReasonCode.RUNTIME_MISMATCH, "/runtime_id")


def test_grant_when_the_worker_validates_it_should_take_the_execution_from_the_grant():
    # Arrange: a grant consistent in itself for an execution the worker has never heard of, at any fence.
    grant = _grant_for_execution(OTHER_EXECUTION, 9)

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert verdict.accepted, verdict


def test_grant_when_the_worker_reads_an_execution_that_is_not_the_generation_should_be_refused_at_the_generation():
    # Arrange: another execution id (not the video's), the generation and the prefix left as granted.
    grant = files.read_json(GRANT)
    grant["identity"]["execution_id"] = OTHER_EXECUTION

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.SEMANTIC,
        ReasonCode.GENERATION_EXECUTION_MISMATCH,
        "/generation_id",
    )


@pytest.mark.parametrize(
    ("sent_at", "reason"),
    [
        ("2029-12-31T23:54:59.000Z", "timestamp_out_of_tolerance"),
        ("2030-01-01T00:05:01.000Z", "timestamp_in_future"),
        ("2030-02-30T00:00:00.000Z", "invalid_timestamp"),
    ],
    ids=["301 s before now", "301 s after now", "a day the calendar does not have"],
)
def test_grant_when_the_worker_reads_a_sent_at_outside_the_push_bounds_or_the_calendar_should_be_refused_at_sent_at(
    sent_at, reason
):
    # Arrange
    grant = {**files.read_json(GRANT), "sent_at": sent_at}

    # Act
    verdict = _validate(grant, files.read_json(WORKER_GRANT))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (Layer.SEMANTIC, ReasonCode(reason), "/sent_at")


def test_grant_when_the_hub_self_checks_the_one_it_minted_should_be_accepted():
    # Arrange
    grant = files.read_json(GRANT)

    # Act
    verdict = _validate(grant, files.read_json(HUB_SENDER_GRANT))

    # Assert
    assert verdict.accepted, verdict


def test_grant_when_the_hub_self_checks_a_prefix_that_goes_on_past_the_generation_should_be_refused_at_the_prefix():
    # Arrange: the grant the worker refuses in the catalog (N123), before the Hub sends it.
    raw = files.read_bytes(GRANT_PREFIX_PAST_THE_GENERATION)

    # Act
    verdict = pipeline.validate(raw, TrustedContext.from_dict(files.read_json(HUB_SENDER_GRANT)))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.SEMANTIC,
        ReasonCode.GENERATION_PREFIX_MISMATCH,
        "/output/prefix",
    )


@pytest.mark.parametrize(
    ("execution_id", "fence", "reason", "pointer"),
    [
        (OTHER_EXECUTION, 2, "execution_mismatch", "/identity/execution_id"),
        (GENERATION, 1, "stale_execution_fence", "/identity/execution_fence"),
        (GENERATION, 3, "execution_mismatch", "/identity/execution_fence"),
    ],
    ids=["another execution id", "a fence below the recorded one", "a fence above the recorded one"],
)
def test_grant_when_the_hub_self_checks_an_execution_it_did_not_record_should_be_refused(
    execution_id, fence, reason, pointer
):
    # Arrange: the grant is consistent in itself, so only the Hub's records can refuse it.
    grant = _grant_for_execution(execution_id, fence)

    # Act
    verdict = _validate(grant, files.read_json(HUB_SENDER_GRANT))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (Layer.SEMANTIC, ReasonCode(reason), pointer)


@pytest.mark.parametrize(
    ("payload", "hub_context"),
    [(CLAIM, HUB_CLAIM), (UNCLAIMED_REFUSED, HUB_UNCLAIMED_PUSH), (UNCLAIMED_UNREACHABLE, HUB_UNCLAIMED_POLL)],
    ids=["a claim", "an unclaimed report by callback", "an unclaimed report in provider output"],
)
def test_worker_self_check_when_the_hub_would_accept_the_message_should_accept_it_too(payload, hub_context):
    # Arrange: the Hub receiver's context, held by the worker as its self-check before sending.
    context = {**files.read_json(hub_context), "role": "worker_sender"}

    # Act
    verdict = pipeline.validate(files.read_bytes(payload), TrustedContext.from_dict(context))

    # Assert
    assert verdict.accepted, verdict


def test_worker_self_check_when_its_claim_was_sent_301_seconds_ago_should_be_out_of_tolerance():
    # Arrange: the claim the Hub refuses as stale in the catalog (N120), under the worker's own self-check.
    context = {**files.read_json(HUB_CLAIM), "role": "worker_sender"}

    # Act
    verdict = pipeline.validate(files.read_bytes(CLAIM_SENT_301S_AGO), TrustedContext.from_dict(context))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.SEMANTIC,
        ReasonCode.TIMESTAMP_OUT_OF_TOLERANCE,
        "/sent_at",
    )


@pytest.mark.parametrize(
    ("payload", "context"),
    [(GRANT, WORKER_GRANT), (UNCLAIMED_REFUSED, HUB_UNCLAIMED_PUSH)],
    ids=["a claim grant", "an unclaimed report"],
)
def test_runtime_id_when_it_is_the_video_uuid_should_be_refused_as_an_alias_at_the_runtime_id(payload, context):
    # Arrange: a runtime id defaulted from the video; the claim's own case is the catalog's N132.
    message = {**files.read_json(payload), "runtime_id": VIDEO}

    # Act
    verdict = _validate(message, files.read_json(context))

    # Assert
    assert (verdict.layer, verdict.reason, verdict.instance_path) == (
        Layer.SEMANTIC,
        ReasonCode.IDENTITY_ALIASES_VIDEO_UUID,
        "/runtime_id",
    )


def test_claim_when_its_organization_and_its_runtime_both_alias_the_video_should_name_the_organization_first():
    # Arrange: the runtime id is the last id S7 compares.
    claim = files.read_json(CLAIM)
    claim["identity"]["org_uuid"] = VIDEO
    claim["claim"]["runtime_id"] = VIDEO

    # Act
    verdict = _validate(claim, files.read_json(HUB_CLAIM))

    # Assert
    assert (verdict.reason, verdict.instance_path) == (ReasonCode.IDENTITY_ALIASES_VIDEO_UUID, "/identity/org_uuid")


def test_unclaimed_when_sent_by_a_start_other_than_the_recorded_runtime_should_be_accepted():
    # Arrange: the starts that report unclaimed are the ones that were not granted; no check compares their runtime.
    context = files.read_json(HUB_UNCLAIMED_PUSH)
    context["expect"]["runtime_id"] = OTHER_RUNTIME

    # Act
    verdict = pipeline.validate(files.read_bytes(UNCLAIMED_REFUSED), TrustedContext.from_dict(context))

    # Assert
    assert verdict.accepted, verdict


def test_unclaimed_when_three_hours_old_on_the_push_channel_should_be_out_of_tolerance():
    # Arrange: the report the poll channel accepts (P26), delivered by callback instead.
    report = files.read_json(UNCLAIMED_UNREACHABLE)

    # Act
    verdict = _validate(report, files.read_json(HUB_UNCLAIMED_PUSH))

    # Assert
    assert (verdict.reason, verdict.instance_path) == (ReasonCode.TIMESTAMP_OUT_OF_TOLERANCE, "/sent_at")


@pytest.mark.parametrize(
    ("cause", "hub_error_code"),
    [("refused", None), ("unreachable", "claim_conflict"), ("invalid_answer", "claim_conflict")],
    ids=[
        "refused without the code of the refusal",
        "unreachable with a code no answer gave",
        "an invalid answer with a code",
    ],
)
def test_unclaimed_when_hub_error_code_does_not_fit_its_cause_should_be_a_schema_violation(cause, hub_error_code):
    # Arrange
    report = {**files.read_json(UNCLAIMED_REFUSED), "cause": cause}
    del report["hub_error_code"]
    if hub_error_code is not None:
        report["hub_error_code"] = hub_error_code

    # Act
    verdict = _validate(report, files.read_json(HUB_UNCLAIMED_PUSH))

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION)


def test_unclaimed_when_refused_with_a_code_this_version_does_not_list_should_be_accepted():
    # Arrange: a later Hub may refuse with a code this version's reason-codes.json does not hold.
    report = {**files.read_json(UNCLAIMED_REFUSED), "hub_error_code": "quota_exhausted"}

    # Act
    verdict = _validate(report, files.read_json(HUB_UNCLAIMED_PUSH))

    # Assert
    assert "quota_exhausted" not in load_registry().reason_code_values()
    assert verdict.accepted, verdict
