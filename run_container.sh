#!/bin/bash
# Run the ants-nidm BIDS App container on a BIDS dataset.
#
# This wrapper exists because the container needs a handful of non-obvious
# runtime arguments to work outside the cluster it was developed on:
#   - a writable /work bind (the image sets TMPDIR=/work, which is read-only
#     inside a SIF unless something writable is mounted over it)
#   - read-only input / writable output binds with absolute paths
#   - a sensible thread count passed through to the app
#
# Usage:
#   ./run_container.sh <image> <bids_dir> <output_dir> <participant-label> [extra app args...]
#
#   <image>  path to ants-nidm_bidsapp.sif (Apptainer/Singularity), or a
#            Docker image name such as ants-nidm_bidsapp:latest
#
# Examples:
#   ./run_container.sh ants-nidm_bidsapp.sif  /data/bids /data/derivatives/ants-nidm 01
#   ./run_container.sh ants-nidm_bidsapp:latest /data/bids /data/derivatives/ants-nidm 01 --num-threads 8 --verbose
#
# Anything after the participant label is passed to the app unchanged
# (e.g. --session-label, --nidm-input-dir, --method quick, --num-threads N).

set -euo pipefail

if [ "$#" -lt 4 ]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,24p'
    exit 1
fi

IMAGE="$1"
BIDS_DIR="$(cd "$2" && pwd)"
OUTPUT_DIR="$3"
PARTICIPANT="$4"
shift 4

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ants-nidm-work.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

# Default to 4 threads unless the caller passes their own --num-threads
THREAD_ARGS=()
case " $* " in
    *" --num-threads "*) ;;
    *) THREAD_ARGS=(--num-threads 4) ;;
esac

if [ -f "$IMAGE" ]; then
    # Apptainer / Singularity image file
    RUNNER=""
    command -v apptainer >/dev/null 2>&1 && RUNNER=apptainer
    [ -z "$RUNNER" ] && command -v singularity >/dev/null 2>&1 && RUNNER=singularity
    if [ -z "$RUNNER" ]; then
        echo "ERROR: '$IMAGE' is a file but neither apptainer nor singularity is installed." >&2
        exit 1
    fi
    exec "$RUNNER" run \
        --userns --no-home \
        -B "$BIDS_DIR":"$BIDS_DIR":ro \
        -B "$OUTPUT_DIR":"$OUTPUT_DIR" \
        -B "$WORK_DIR":/work \
        "$IMAGE" \
        "$BIDS_DIR" "$OUTPUT_DIR" participant \
        --participant-label "$PARTICIPANT" \
        "${THREAD_ARGS[@]}" "$@"
else
    # Docker image name
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: '$IMAGE' is not a file and docker is not installed." >&2
        exit 1
    fi
    exec docker run --rm \
        -u "$(id -u):$(id -g)" \
        -v "$BIDS_DIR":"$BIDS_DIR":ro \
        -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
        -v "$WORK_DIR":/work \
        "$IMAGE" \
        "$BIDS_DIR" "$OUTPUT_DIR" participant \
        --participant-label "$PARTICIPANT" \
        "${THREAD_ARGS[@]}" "$@"
fi
