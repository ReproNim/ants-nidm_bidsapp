#!/bin/bash
# Run the ants-nidm BIDS App container on a BIDS dataset.
#
# This wrapper exists because the container needs a handful of non-obvious
# runtime arguments to work outside the cluster it was developed on:
#   - a writable /work bind (the image sets TMPDIR=/work, which is read-only
#     inside a SIF unless something writable is mounted over it)
#   - read-only input / writable output binds with absolute paths
#   - a sensible thread count passed through to the app
#   - --nv so the GPU is visible inside the container (registration is
#     GPU-only; without it the app exits early with a device error)
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
# Anything after the participant label is passed to the app (e.g.
# --session-label, --method quick); --nidm-input-dir is also mounted read-only.

set -euo pipefail
# An exported CDPATH would make the `cd` calls below pick another directory
# and print it.
unset CDPATH

if [ "$#" -lt 4 ]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,24p'
    exit 1
fi

# An empty or blank directory argument is an unset variable in the caller.
# `cd ""` succeeds and stays put, so an empty <bids_dir> would become the
# current directory before the app could reject it, and a blank <output_dir>
# would be created here.
if [ -z "${2//[[:space:]]/}" ]; then
    echo "ERROR: bids_dir must not be empty" >&2
    exit 2
fi
if [ -z "${3//[[:space:]]/}" ]; then
    echo "ERROR: output_dir must not be empty" >&2
    exit 2
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

# An explicit --nidm-input-dir is mounted read-only at its absolute host
# path, since the app rejects a directory it cannot see. A value that is not
# a directory here is passed on unchanged, and the app reports it.
# argparse also accepts an unambiguous prefix such as --nidm.
is_nidm_flag() {
    case "$1" in --nidm*) ;; *) return 1 ;; esac
    case "--nidm-input-dir" in "$1"*) return 0 ;; esac
    return 1
}
APP_ARGS=()
NIDM_DIR=""
while [ "$#" -gt 0 ]; do
    key="${1%%=*}"
    if ! is_nidm_flag "$key"; then
        APP_ARGS+=("$1"); shift; continue
    fi
    if [ "$key" != "$1" ]; then
        nidm_value="${1#*=}"; shift
    elif [ "$#" -ge 2 ]; then
        nidm_value="$2"; shift 2
    else
        APP_ARGS+=("$1"); shift; continue
    fi
    if [ -n "$nidm_value" ] && nidm_abs="$(cd "$nidm_value" 2>/dev/null && pwd)"; then
        nidm_value="$nidm_abs"
        NIDM_DIR="$nidm_abs"
    fi
    APP_ARGS+=(--nidm-input-dir "$nidm_value")
done

if [ -f "$IMAGE" ]; then
    # Apptainer / Singularity image file
    RUNNER=""
    command -v apptainer >/dev/null 2>&1 && RUNNER=apptainer
    [ -z "$RUNNER" ] && command -v singularity >/dev/null 2>&1 && RUNNER=singularity
    if [ -z "$RUNNER" ]; then
        echo "ERROR: '$IMAGE' is a file but neither apptainer nor singularity is installed." >&2
        exit 1
    fi
    # FireANTs registration needs the GPU visible inside the container.
    APPTAINER_ARGS=(run --userns --no-home --nv)
    NIDM_BIND=()
    [ -n "$NIDM_DIR" ] && NIDM_BIND=(-B "$NIDM_DIR":"$NIDM_DIR":ro)
    exec "$RUNNER" "${APPTAINER_ARGS[@]}" \
        -B "$BIDS_DIR":"$BIDS_DIR":ro \
        -B "$OUTPUT_DIR":"$OUTPUT_DIR" \
        -B "$WORK_DIR":/work \
        ${NIDM_BIND[@]+"${NIDM_BIND[@]}"} \
        "$IMAGE" \
        "$BIDS_DIR" "$OUTPUT_DIR" participant \
        --participant-label "$PARTICIPANT" \
        ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} ${APP_ARGS[@]+"${APP_ARGS[@]}"}
else
    # Docker image name
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: '$IMAGE' is not a file and docker is not installed." >&2
        exit 1
    fi
    NIDM_BIND=()
    [ -n "$NIDM_DIR" ] && NIDM_BIND=(-v "$NIDM_DIR":"$NIDM_DIR":ro)
    # FireANTs registration needs the GPU visible inside the container.
    exec docker run --rm \
        --gpus all \
        -u "$(id -u):$(id -g)" \
        -v "$BIDS_DIR":"$BIDS_DIR":ro \
        -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
        -v "$WORK_DIR":/work \
        ${NIDM_BIND[@]+"${NIDM_BIND[@]}"} \
        "$IMAGE" \
        "$BIDS_DIR" "$OUTPUT_DIR" participant \
        --participant-label "$PARTICIPANT" \
        ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} ${APP_ARGS[@]+"${APP_ARGS[@]}"}
fi
