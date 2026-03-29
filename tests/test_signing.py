from unittest import mock

from core.signing import sign_request, verify_request

TEST_SECRET = "test-secret-key"
TEST_TIMESTAMP = 1700000000
TEST_BODY = b'{"uuid":"abc-123","status":"ready"}'
EXPECTED_SIGNATURE = (
    "sha256=9f474bec573dbe80567db419017df368157f097c8ebccf765aaa393bc6feaea9"
)


class TestSignRequest:
    @mock.patch("core.signing.time")
    def test_sign_should_produce_correct_hmac(self, mock_time):
        mock_time.time.return_value = TEST_TIMESTAMP

        headers = sign_request(TEST_BODY, TEST_SECRET)

        assert headers["X-Signature"] == EXPECTED_SIGNATURE
        assert headers["X-Timestamp"] == str(TEST_TIMESTAMP)

    @mock.patch("core.signing.time")
    def test_sign_and_verify_roundtrip(self, mock_time):
        mock_time.time.return_value = TEST_TIMESTAMP

        headers = sign_request(TEST_BODY, TEST_SECRET)
        result = verify_request(
            body=TEST_BODY,
            signature_header=headers["X-Signature"],
            timestamp_header=headers["X-Timestamp"],
            secret=TEST_SECRET,
        )

        assert result is True


class TestVerifyRequest:
    @mock.patch("core.signing.time")
    def test_verify_when_valid_should_return_true(self, mock_time):
        mock_time.time.return_value = TEST_TIMESTAMP

        result = verify_request(
            body=TEST_BODY,
            signature_header=EXPECTED_SIGNATURE,
            timestamp_header=str(TEST_TIMESTAMP),
            secret=TEST_SECRET,
        )

        assert result is True

    @mock.patch("core.signing.time")
    def test_verify_when_wrong_signature_should_return_false(self, mock_time):
        mock_time.time.return_value = TEST_TIMESTAMP

        result = verify_request(
            body=TEST_BODY,
            signature_header="sha256=wrong",
            timestamp_header=str(TEST_TIMESTAMP),
            secret=TEST_SECRET,
        )

        assert result is False

    def test_verify_when_missing_headers_should_return_false(self):
        assert verify_request(TEST_BODY, None, None, TEST_SECRET) is False
        assert verify_request(TEST_BODY, EXPECTED_SIGNATURE, None, TEST_SECRET) is False
        assert verify_request(TEST_BODY, None, str(TEST_TIMESTAMP), TEST_SECRET) is False

    @mock.patch("core.signing.time")
    def test_verify_when_stale_timestamp_should_return_false(self, mock_time):
        # 10 minutes later — beyond 5-min tolerance.
        mock_time.time.return_value = TEST_TIMESTAMP + 600

        result = verify_request(
            body=TEST_BODY,
            signature_header=EXPECTED_SIGNATURE,
            timestamp_header=str(TEST_TIMESTAMP),
            secret=TEST_SECRET,
        )

        assert result is False

    @mock.patch("core.signing.time")
    def test_verify_when_within_tolerance_should_return_true(self, mock_time):
        # 4 minutes later — within 5-min tolerance.
        mock_time.time.return_value = TEST_TIMESTAMP + 240

        result = verify_request(
            body=TEST_BODY,
            signature_header=EXPECTED_SIGNATURE,
            timestamp_header=str(TEST_TIMESTAMP),
            secret=TEST_SECRET,
        )

        assert result is True
