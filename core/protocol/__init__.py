"""Transcode contract v2 (draft): the worker-side validator and the exact-bytes message signing.

The contract tree lives in tests/contracts/transcode/v2/ (a byte-identical copy of the Hub's
laravel/resources/contracts/transcode/v2/, shipped in the RunPod image at the same relative
path); its README.md is the normative rule text this package implements. files.verify_contract_tree()
is the fail-closed start-up check of that tree. settings.WorkerSettings is the worker's trusted
configuration (the VH_* variables); messages builds, encodes and seals what the worker sends and
builds the contexts it validates with. handler.handle() is one start of the worker for one dispatch
(README section 15.1): dispatch accepts the RunPod job input (inbound: the signature first, then the
validation, then the key the body names), claim runs the claim exchange over transport, and outbox
self-checks, signs and carries what the worker sends back. The RunPod entry point
(wrappers/runpod_handler.py) calls handle() with the job runner (core/job/) for every job; this package
is on the RunPod entry path only (the Modal app and the v1 dev server never import it). The v1 signer
(core/signing.py) is unrelated.
"""
