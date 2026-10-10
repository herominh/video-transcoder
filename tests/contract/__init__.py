"""Tests of the transcode contract v2 validator and signing (core/protocol/).

They check the contract tree copy in tests/contracts/transcode/v2/ (a byte-identical
copy of the Hub's laravel/resources/contracts/transcode/v2/) against SHA256SUMS, run
every fixture through the validator and every signing vector through the signer.
"""
