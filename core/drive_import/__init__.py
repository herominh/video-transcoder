"""The Google Drive acquire (contract v2, protocol 2.1.0, `source.acquire`; the Hub's issue #51 item 19).

One RunPod job of the acquire endpoint downloads one Drive file and writes it, part by part, into the multipart
upload the Hub opened, through the presigned URLs of its claim grant, then writes its signed result through the
grant's result URL. acquire.handle() is one start (wrappers/runpod_acquire_handler.py is the entry point); the
transfer itself is transfer.py over google.py, store.py and transport.py, held to the start's wall deadline by
clock.StopLine. Only acquire.py reaches the contract (core.protocol); the transcode's job runner (core.job) and the
v1 modules are not reached, and the transcode entry reaches none of this package (tests/test_runtime_scope.py).
The first Drive import, contract import/v1, is retired on both sides (TD-184).
"""
