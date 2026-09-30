"""The five-layer validation pipeline of the transcode contract v2: the first failing layer wins."""

from __future__ import annotations

from . import gate, schema, semantic
from .context import TrustedContext
from .reasons import Layer, ReasonCode
from .registry import ContractRegistry, load_registry
from .verdict import Verdict


def validate(raw: bytes, context: TrustedContext, registry: ContractRegistry | None = None) -> Verdict:
    """Validate the exact received bytes: L0 size, L1 parse (and token budget), L2 version/kind, L0 kind size,
    L3 schema (stopping at the first error), L4 semantic."""
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("validate() takes the exact received bytes")
    if not isinstance(context, TrustedContext):
        raise TypeError("validate() takes a TrustedContext")
    contract = registry if registry is not None else load_registry()
    payload = bytes(raw)

    ceiling = max(contract.max_bytes(kind) for kind in context.accepted_message_kinds)
    if not gate.within_size(payload, ceiling):
        return gate.payload_too_large()

    document = gate.parse(payload, contract.max_json_depth, contract.max_tokens)
    if isinstance(document, Verdict):
        return document

    kind = gate.select_kind(document, context)
    if isinstance(kind, Verdict):
        return kind
    if not gate.within_size(payload, contract.max_bytes(kind)):
        return gate.payload_too_large()

    pointer = schema.first_error_pointer(kind.schema_file, document)
    if pointer is not None:
        return Verdict.reject(Layer.SCHEMA, ReasonCode.SCHEMA_VIOLATION, pointer)

    failure = semantic.first_failure(kind, document, context, contract)
    if failure is not None:
        return Verdict.reject(Layer.SEMANTIC, failure.reason, failure.instance_path)
    return Verdict.accept()
