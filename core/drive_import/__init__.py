"""The Google Drive import job (contract import/v1; the Hub's issue #50 item 3b).

One RunPod job of the import endpoint downloads one Drive file and writes it, part by part, into the multipart
upload the Hub opened, through the presigned URLs of the job's input, then writes one receipt signed with the
receipt key. The contract tree lives in tests/contracts/import/v1/ (a byte-identical copy of the Hub's
laravel/resources/contracts/import/v1/, shipped in the import image at the same relative path); its README.md is
the normative rule text. contract.verify_import_contract_tree() is the fail-closed start-up check of that tree;
input_check holds the input's rules beyond its schema; job.run_import() is one job. The entry point is
wrappers/runpod_import_handler.py. This package imports nothing of the transcode contract (core.protocol), of the
job runner (core.job) or of the v1 modules, and they import nothing of it (tests/test_runtime_scope.py).
"""
