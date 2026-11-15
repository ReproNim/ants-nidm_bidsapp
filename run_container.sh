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

# Default to 4 threads unless the caller passes their own --num-threads
THREAD_ARGS=()
case " $* " in
    *" --num-threads "*) ;;
    *) THREAD_ARGS=(--num-threads 4) ;;
esac

# An explicit --nidm-input-dir is mounted read-only at its resolved host
# path (see below), since the app rejects a directory it cannot see. A value that is not
# a directory here is passed on unchanged, and the app reports it.
# argparse also accepts an unambiguous prefix such as --ni; --n is ambiguous
# with --num-threads, so argparse rejects it.
is_nidm_flag() {
    [ "${#1}" -ge 4 ] || return 1
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
        NIDM_IDX=$(( ${#APP_ARGS[@]} + 1 ))
    fi
    APP_ARGS+=(--nidm-input-dir "$nidm_value")
done

# A directory whose resolved path is inside the resolved <bids_dir> is visible
# through the BIDS mount at the same suffix. Anything else is mounted, and
# passed on, at its resolved path: a mount at a path with a symlink in it can
# land on the link inside another mount (the BIDS dir, the working directory,
# /tmp), and Apptainer then mounts it where the container cannot see it.
NIDM_MOUNT=""
if [ -n "$NIDM_DIR" ]; then
    nidm_phys="$(cd "$NIDM_DIR" && pwd -P)"
    bids_phys="$(cd "$BIDS_DIR" && pwd -P)"
    case "$nidm_phys/" in
        "$bids_phys/"*) APP_ARGS[NIDM_IDX]="$BIDS_DIR${nidm_phys#"$bids_phys"}" ;;
        *) APP_ARGS[NIDM_IDX]="$nidm_phys"; NIDM_MOUNT="$nidm_phys" ;;
    esac
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# The NIDM input must not be the output directory or lie inside it: the app
# would read its own outputs as input, and a read-only mount over them would
# make the outputs, written only after the GPU work, fail. Checked on both the
# given and the resolved paths, so a symlinked alias is caught too. Checked
# after mkdir -p: `cd` reads a path like nonexist/../out as out, but only once
# nonexist exists.
if [ -n "$NIDM_DIR" ]; then
    out_phys="$(cd "$OUTPUT_DIR" && pwd -P)"
    nidm_phys="$(cd "$NIDM_DIR" && pwd -P)"
    case "$NIDM_DIR/" in "$OUTPUT_DIR/"*) nidm_in_output=1 ;; *) nidm_in_output="" ;; esac
    case "$nidm_phys/" in "$out_phys/"*) nidm_in_output=1 ;; esac
    if [ -n "$nidm_in_output" ]; then
        echo "ERROR: --nidm-input-dir $NIDM_DIR is the output directory or inside it" \
             "(resolved: $nidm_phys; output resolved: $out_phys). The NIDM input" \
             "must not share a directory with the outputs: give it from a directory" \
             "outside $OUTPUT_DIR." >&2
        exit 2
    fi
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ants-nidm-work.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

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
    [ -n "$NIDM_MOUNT" ] && NIDM_BIND=(-B "$NIDM_MOUNT":"$NIDM_MOUNT":ro)
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
    [ -n "$NIDM_MOUNT" ] && NIDM_BIND=(-v "$NIDM_MOUNT":"$NIDM_MOUNT":ro)
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
