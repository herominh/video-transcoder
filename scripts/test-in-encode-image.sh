#!/usr/bin/env bash
# Runs the worker's test suite inside an encode image (Dockerfile.runpod), on that image's own Python and installed
# packages, against this checkout's code. Usage: scripts/test-in-encode-image.sh <image reference> [pytest options]
#
# It always runs the whole suite (`tests/`): extra arguments are pytest OPTIONS added to that run (`-q -rs`, `-x`, `-k`);
# a path given among them does not narrow the run.
#
# The checkout is mounted read-only at /src (never over the image's own /app copy) and is the working directory, so
# `core`, `wrappers` and `tests` are the checkout's. Its `.env`, when there is one, is hidden behind an empty file: no
# test reads it, and no secret goes into the container. Nothing is written into the checkout: no pytest cache, no
# bytecode. /tmp is a writable tmpfs (the scratch directories of the job tests live there). The container has no
# network and runs an init process (the process-group tests need one). The image is amd64, so on an arm64 machine it
# runs emulated and slowly. The image must already be on this machine: the script never pulls one.
set -euo pipefail

readonly IMAGE_PLATFORM="linux/amd64"
readonly MOUNT_POINT="/src"
readonly TMP_OPTIONS="rw,exec,nosuid,nodev,mode=1777,size=4g"
readonly USAGE_EXIT=2

if [[ $# -lt 1 || -z "${1}" ]]; then
    echo "usage: scripts/test-in-encode-image.sh <image reference> [pytest options added to the whole-suite run]" >&2
    exit "${USAGE_EXIT}"
fi
image="$1"
shift

if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "test-in-encode-image: the image ${image} was not found locally, or the container engine did not answer" \
        "(this script never pulls an image)" >&2
    exit "${USAGE_EXIT}"
fi

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
hidden_files=()
if [[ -e "${repository_root}/.env" ]]; then
    hidden_files=(--mount "type=bind,source=/dev/null,target=${MOUNT_POINT}/.env,readonly")
fi

started_at="${SECONDS}"
status=0
# The image's entrypoint prints NVIDIA's banner; bash in its place prints the interpreter's version, then hands the
# process to pytest, whose status is the container's.
docker run --rm --init --network none --pull never \
    --platform "${IMAGE_PLATFORM}" \
    --mount "type=bind,source=${repository_root},target=${MOUNT_POINT},readonly" \
    ${hidden_files[@]+"${hidden_files[@]}"} \
    --workdir "${MOUNT_POINT}" \
    --tmpfs "/tmp:${TMP_OPTIONS}" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /bin/bash \
    "${image}" \
    -c 'python3 --version && exec python3 -m pytest -p no:cacheprovider tests/ "$@"' test-in-encode-image "$@" \
    || status=$?

echo "test-in-encode-image: ${image}: exit ${status} after $((SECONDS - started_at)) s"
exit "${status}"
