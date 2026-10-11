"""The worker's job under contract v2: a claimed execution run to its result.

core.protocol.handler.handle() accepts a dispatch and claims it; what it hands on is a ClaimedExecution,
and this package turns that into the execution's one ending, a sealed `transcode.result.completed` or
`transcode.result.failed`. Only the RunPod entry path imports it: it builds on core.protocol, which the
Modal app and the v1 dev server never load (tests/test_runtime_scope.py).

    config          JobSettings: what the job needs beyond the protocol's settings (the image digest
                    a result reports, the scratch directory)
    storage         the two ways the job touches a bucket, each with its own client of an access
                    profile: download_source() reads the source object through conditional range reads
                    (storage_client()), upload_generation() stores exactly the manifest's files under
                    the granted prefix, the manifest last (upload_client())
    ffmpeg_version  the version of the installed ffmpeg, for a result's `encoder` block
    progress        ProgressReporter: best-effort `transcode.progress` messages
    scratch         the lock an execution holds on its scratch directory, and
                    remove_leftover_scratch(), for the entry point to call at process start: it
                    removes the scratch directories a killed process left behind
    runner          JobRunner: the stages in order (preflight and plan, download, encode, package,
                    upload), every ending typed, the scratch directory always removed; it takes one
                    client factory for the source and one for the output
"""
