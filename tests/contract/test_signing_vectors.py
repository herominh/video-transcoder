"""Known-answer vectors pin the exact-bytes signature both sides compute (HMAC-SHA256, per-direction keys)."""

import hashlib
import hmac
import json

import pytest

from core.protocol import files, schema, signing
from core.protocol.registry import load_registry

VECTORS = files.read_json("signing/vectors.json")
KEYS = {entry["key_id"]: bytes.fromhex(entry["key_hex"]) for entry in VECTORS["keys"]}
VECTORS_BY_ID = {vector["id"]: vector for vector in VECTORS["vectors"]}
REQUIRED_CATEGORIES = {
    "valid",
    "body_byte_changed",
    "reserialized",
    "v1_signing_input",
    "other_direction_key",
    "malformed_value",
}


def _vector_bytes(vector: dict) -> bytes:
    return files.read_bytes(vector["message_file"])


def test_signing_prefix_when_read_should_be_the_23_byte_domain_prefix():
    # Act
    prefix = VECTORS["signing_input_prefix"].encode("ascii")

    # Assert
    assert prefix == signing.SIGNING_PREFIX == bytes.fromhex(VECTORS["signing_input_prefix_hex"])
    assert len(prefix) == 23


def test_vectors_when_scanned_should_cover_every_required_category_with_test_only_keys():
    # Act
    categories = {vector["category"] for vector in VECTORS["vectors"]}

    # Assert
    assert categories == REQUIRED_CATEGORIES
    assert all(key_id.startswith("test-") for key_id in KEYS)
    assert {entry["direction"] for entry in VECTORS["keys"]} == {"worker_to_hub", "hub_to_worker"}


@pytest.mark.parametrize(
    "vector", [vector for vector in VECTORS["vectors"] if vector["valid"]], ids=lambda vector: vector["id"]
)
def test_signature_when_computed_over_vector_bytes_should_equal_expected(vector):
    # Arrange
    message = _vector_bytes(vector)

    # Act
    value = signing.sign(message, KEYS[vector["key_id"]])

    # Assert
    assert value == vector["value"]


@pytest.mark.parametrize(
    "vector", [vector for vector in VECTORS["vectors"] if vector["valid"]], ids=lambda vector: vector["id"]
)
def test_valid_vector_when_its_message_is_read_should_name_the_key_it_verifies_with(vector):
    # Arrange
    message = json.loads(_vector_bytes(vector))

    # Act
    named_key = message["key_id"]

    # Assert: a live receiver picks the key by the message's own key_id.
    assert named_key == vector["key_id"]


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda vector: vector["id"])
def test_vector_when_verified_should_yield_its_expected_validity(vector):
    # Arrange
    message = _vector_bytes(vector)

    # Act
    valid = signing.verify(message, KEYS[vector["key_id"]], vector["value"])

    # Assert
    assert valid is vector["valid"]


def test_v1_signing_input_vector_when_recomputed_should_be_hmac_over_timestamp_dot_body():
    # Arrange: the live v1 input is `timestamp + "." + body`, without the v2 domain prefix.
    vector = next(vector for vector in VECTORS["vectors"] if vector["category"] == "v1_signing_input")
    v1_input = vector["v1_timestamp"].encode("ascii") + b"." + _vector_bytes(vector)

    # Act
    recomputed = "hmac-sha256=" + hmac.new(KEYS[vector["key_id"]], v1_input, hashlib.sha256).hexdigest()

    # Assert
    assert vector["v1_timestamp"].isdigit()
    assert recomputed == vector["value"]
    assert signing.verify(_vector_bytes(vector), KEYS[vector["key_id"]], vector["value"]) is False


def test_other_direction_vector_when_recomputed_should_be_the_body_signed_with_the_hub_to_worker_key():
    # Arrange
    vector = next(vector for vector in VECTORS["vectors"] if vector["category"] == "other_direction_key")
    hub_to_worker = next(entry for entry in VECTORS["keys"] if entry["direction"] == "hub_to_worker")
    verifying = next(entry for entry in VECTORS["keys"] if entry["key_id"] == vector["key_id"])

    # Act
    recomputed = signing.sign(_vector_bytes(vector), bytes.fromhex(hub_to_worker["key_hex"]))

    # Assert
    assert verifying["direction"] == "worker_to_hub"
    assert recomputed == vector["value"]


def test_signature_when_one_body_byte_changes_should_not_verify():
    # Arrange
    original = VECTORS_BY_ID["V01"]
    changed = _vector_bytes(original)[:-2] + b"X" + _vector_bytes(original)[-1:]

    # Act
    valid = signing.verify(changed, KEYS[original["key_id"]], original["value"])

    # Assert
    assert valid is False


def test_signature_when_body_is_reserialized_should_not_verify():
    # Arrange
    original = VECTORS_BY_ID["V01"]
    reserialized = json.dumps(json.loads(_vector_bytes(original)), indent=4).encode("ascii")

    # Act
    valid = signing.verify(reserialized, KEYS[original["key_id"]], original["value"])

    # Assert
    assert valid is False


@pytest.mark.parametrize("value", [None, 7, b"hmac-sha256=", "", "hmac-sha256=", "hmac-sha256=" + "g" * 64])
def test_verify_when_value_is_malformed_should_return_false_without_raising(value):
    # Arrange
    message = _vector_bytes(VECTORS_BY_ID["V01"])

    # Act
    valid = signing.verify(message, KEYS["test-wk2hub-k1"], value)

    # Assert
    assert valid is False


@pytest.mark.parametrize("key", [b"", "not-bytes"])
def test_sign_when_key_is_empty_or_not_bytes_should_raise(key):
    # Act / Assert
    with pytest.raises(ValueError):
        signing.sign(b"{}", key)


@pytest.mark.parametrize("example", VECTORS["carrier_examples"], ids=lambda example: example["id"])
def test_carrier_example_when_validated_should_satisfy_its_schema_and_raw_limit(example):
    # Arrange
    serialized = json.dumps(example["value"], separators=(",", ":")).encode("ascii")

    # Act
    pointers = schema.error_pointers(example["schema"], example["value"])

    # Assert
    assert pointers == []
    assert len(serialized) <= load_registry().carrier_max_bytes(example["carrier"])


@pytest.mark.parametrize("example", VECTORS["carrier_examples"], ids=lambda example: example["id"])
def test_carrier_example_when_decoded_should_return_the_vector_bytes_and_verify(example):
    # Arrange
    vector = VECTORS_BY_ID[example["vector"]]

    # Act
    decoded = example["value"]["vh_message"].encode("ascii")

    # Assert
    assert decoded == _vector_bytes(vector)
    assert signing.verify(decoded, KEYS[vector["key_id"]], example["value"]["vh_signature"]) is True
