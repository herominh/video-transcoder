"""Every live version is accepted, and no version is a draft; a draft would be supported only by a context built for tests."""

from dataclasses import replace

import pytest

from core.protocol import context as context_module
from core.protocol import files, gate, pipeline
from core.protocol.context import TrustedContext
from core.protocol.reasons import Layer, ReasonCode
from core.protocol.registry import ContractRegistry, MessageKind, VersionChannel, VersionLine

POSITIVE_CASES = [case for case in files.read_json("fixtures/cases.json")["cases"]
                  if case["expect"]["outcome"] == "accept" and "carrier" not in case]


def test_live_versions_when_loaded_should_be_exactly_both_protocol_minors_and_manifest_1():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    live = (
        registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.LIVE),
        registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.LIVE),
    )

    # Assert
    assert live == (("2.0.0", "2.1.0"), ("1.0.0",))


def test_draft_versions_when_loaded_should_be_empty():
    # Arrange
    registry = ContractRegistry.load()

    # Act
    drafts = (
        registry.accepted_versions(VersionLine.PROTOCOL, VersionChannel.DRAFT),
        registry.accepted_versions(VersionLine.MANIFEST, VersionChannel.DRAFT),
    )

    # Assert
    assert drafts == ((), ())


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=[case["id"] for case in POSITIVE_CASES])
def test_live_message_when_validated_under_live_context_should_be_accepted(case):
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
    assert verdict.accepted


@pytest.mark.parametrize(
    ("line", "version", "context_path", "header", "kind"),
    [
        (VersionLine.PROTOCOL, "2.1.0-draft", "fixtures/contexts/ctx-hub-push.json",
         {"protocol_version": "2.1.0-draft", "message_kind": "transcode.progress"}, MessageKind.TRANSCODE_PROGRESS),
        (VersionLine.MANIFEST, "1.1.0-draft", "fixtures/contexts/ctx-hub-storage.json",
         {"manifest_version": "1.1.0-draft", "document_kind": "generation.manifest"}, MessageKind.GENERATION_MANIFEST),
    ],
)
def test_draft_version_when_listed_only_in_a_test_registry_should_need_a_draft_context(
    monkeypatch, line, version, context_path, header, kind
):
    # Arrange: no shipped file defines a draft; only this test's registry and context do.
    shipped = ContractRegistry.load()
    versions = {name: {**channels, "draft": []} for name, channels in shipped.versions.items()}
    versions[line.value]["draft"] = [version]
    registry = replace(shipped, versions=versions)
    monkeypatch.setattr(context_module, "load_registry", lambda: registry)
    data = files.read_json(context_path)
    field = f"accepted_{line.value}_versions"
    live = TrustedContext.from_dict({**data, field: list(registry.accepted_versions(line, VersionChannel.LIVE))})
    draft = TrustedContext.from_dict({**data, field: list(registry.accepted_versions(line, VersionChannel.DRAFT))})

    # Act
    refused = gate.select_kind(header, live)
    accepted = gate.select_kind(header, draft)

    # Assert
    assert (refused.layer, refused.reason) == (Layer.VERSION, ReasonCode.UNSUPPORTED_VERSION)
    assert accepted is kind
    assert ContractRegistry.load().accepted_versions(line, VersionChannel.DRAFT) == ()
