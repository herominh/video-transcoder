"""Acquire terminals use their own bounded provider carrier and retain exact bytes."""

import json

from core.protocol import acquire_messages, schema
from core.protocol.acquire_dispatch import accept_dispatch
from core.protocol.messages import ExecutionIdentity
from core.protocol.outbox import runpod_output, seal_message
from core.protocol.registry import MessageKind, load_registry
from tests.acquire.support import EXECUTION, MAX_BYTES, Scenario, wire
from tests.contract.v2_support import HUB_KEY, signed_carrier


def test_acquire_terminal_when_160_etags_need_maximal_escaping_should_fit_only_the_acquire_carrier() -> None:
    # Arrange
    scenario = Scenario()
    scenario.request["external"]["expected"]["size"] = MAX_BYTES
    scenario.request["plan"]["parts_total"] = 160
    dispatch = accept_dispatch(signed_carrier(wire(scenario.request), HUB_KEY), settings=scenario.runtime.settings, now=scenario.clock.now())
    identity = ExecutionIdentity.of(dispatch.identity, execution_id=EXECUTION, execution_fence=2)
    parts = [{"n": n, "etag": "\\" * 128} for n in range(1, 161)]
    message = acquire_messages.result_completed(
        protocol_version=dispatch.protocol_version, message_id=scenario.runtime.new_uuid(), sent_at=scenario.clock.now(),
        key_id=scenario.runtime.settings.worker_to_hub_key_id, identity=identity,
        bytes_read=MAX_BYTES, checksum_verified="sha256", parts=parts,
    )
    context = acquire_messages.sender_context(MessageKind.SOURCE_ACQUIRE_RESULT_COMPLETED, dispatch, identity,
                                               settings=scenario.runtime.settings, now=scenario.clock.now())
    # Act
    sealed = seal_message(message, context, settings=scenario.runtime.settings)
    carrier = runpod_output(sealed)
    serialized = wire(carrier)
    # Assert
    registry = load_registry()
    assert len(sealed.body) > registry.max_bytes(MessageKind.TRANSCODE_RESULT_COMPLETED)
    assert len(sealed.body) <= registry.max_bytes(MessageKind.SOURCE_ACQUIRE_RESULT_COMPLETED)
    assert registry.carrier_max_bytes("runpod.output") < len(serialized) <= registry.carrier_max_bytes("runpod.acquire.output")
    assert schema.first_error_pointer("carriers/runpod-acquire-output.schema.json", carrier) is None
    assert schema.first_error_pointer("carriers/runpod-output.schema.json", carrier) is not None
    assert carrier["vh_message"].encode("ascii") == sealed.body
    assert json.loads(sealed.body)["parts"] == parts
