"""B03 accepts no version for live use: the draft documents are refused by any live context."""

import pytest

from core.protocol import files, pipeline
from core.protocol.context import TrustedContext
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import ContractRegistry, VersionChannel, VersionLine

POSITIVE_CASES = [case for case in files.read_json("fixtures/cases.json")["cases"]
                  if case["expect"]["outcome"] == "accept"]


def test_live_versions_when_loaded_should_be_empty():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    live = (
        registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.LIVE),
        registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.LIVE),
    )

    # Assert
    assert live == ((), ())


def test_draft_versions_when_loaded_should_be_exactly_the_two_b03_drafts():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    drafts = (
        registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.DRAFT),
        registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.DRAFT),
    )

    # Assert
    assert drafts == (("2.0.0-draft",), ("1.0.0-draft",))


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=[case["id"] for case in POSITIVE_CASES])
def test_draft_message_when_validated_under_live_context_should_be_rejected_as_unsupported_version(case):
    # Arrange
    registry = ContractRegistry.load()
    context_data = files.read_json(case["context"])
    context_data["accepted_protocol_versions"] = list(
        registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.LIVE)
    )
    context_data["accepted_manifest_versions"] = list(
        registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.LIVE)
    )
    live_context = TrustedContext.from_dict(context_data)

    # Act
    verdict = pipeline.validate(files.read_bytes(case["payload"]), live_context)

    # Assert
    assert (verdict.layer, verdict.reason) == (Layer.VERSION, ReasonCode.UNSUPPORTED_VERSION)
