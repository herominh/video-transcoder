"""Transcode contract v2 (draft): the worker-side validator and the exact-bytes message signing.

The contract tree lives in tests/contracts/transcode/v2/ (a byte-identical copy of the Hub's
laravel/resources/contracts/transcode/v2/, shipped in the RunPod image at the same relative
path); its README.md is the normative rule text this package implements. files.verify_contract_tree()
is the fail-closed start-up check of that tree. The handler does not call this package yet: D1
wires it in. The live v1 signer (core/signing.py) is unrelated.
"""
