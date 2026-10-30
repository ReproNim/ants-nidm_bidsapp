#!/bin/bash
# Issue #16: does computing --method quick's centre-of-mass init on the plain
# images (instead of the intensity + mask batches) change the segmentation?
#
# One array task per subject. Each task runs --method quick three times in the
# same SIF, with src/ taken from a git ref and bound over /opt/src:
#   old      OLD_REF  (moments on the 2-channel masked-loss batches)
#   new      NEW_REF  (moments on the plain single-channel images)
#   old_rep  OLD_REF  again -- run-to-run noise floor for the old/new difference
# scripts/diag_quick_moments_init.py then compares the three dsegs with each
# other and with a fusion dseg of the same subject.
#
# Usage (from the repo root, partition on the command line):
#   sbatch -p <partition> --array=0-<n-1> -o <log_dir>/qmi_%A_%a.out \
#     scripts/diag_quick_moments_init.sbatch \
#     <sif> <bids_dir> <out_root> <old_ref> <new_ref> <participant> [<participant> ...]
#
# Outputs: <out_root>/<variant>/sub-<id>/... and <out_root>/src/task<N>/{old,new}/
# (the exact src/ each variant ran, with the commit in SRC_COMMIT).
#SBATCH --job-name=ants-quick-moments
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --constraint=rocky8

set -euo pipefail

SIF="$1"; BIDS="$2"; OUT_ROOT="$3"; OLD_REF="$4"; NEW_REF="$5"
shift 5
SUBJECTS=("$@")
SUB="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
REPO="$(pwd)"

module load apptainer/1.5.2
nvidia-smi -L

# Export src/ (and the NIDM converter submodule at the commit the ref pins)
# from git, so each variant runs exactly one commit and edits to the working
# tree after submission cannot leak in.
export_src() {
    local ref="$1" dest="$2"
    local sha sub_sha
    sha="$(git -C "$REPO" rev-parse "${ref}^{commit}")"
    sub_sha="$(git -C "$REPO" rev-parse "${sha}:src/ants_seg_to_nidm")"
    rm -rf "$dest"; mkdir -p "$dest"
    git -C "$REPO" archive "$sha" src | tar -x -C "$dest" --strip-components=1
    rm -rf "${dest}/ants_seg_to_nidm"; mkdir -p "${dest}/ants_seg_to_nidm"
    git -C "${REPO}/src/ants_seg_to_nidm" archive "$sub_sha" \
        | tar -x -C "${dest}/ants_seg_to_nidm"
    echo "$sha" > "${dest}/SRC_COMMIT"
}

TASK_SRC="${OUT_ROOT}/src/task${SLURM_ARRAY_TASK_ID}"
export_src "$OLD_REF" "${TASK_SRC}/old"
export_src "$NEW_REF" "${TASK_SRC}/new"
echo "old=$(cat "${TASK_SRC}/old/SRC_COMMIT") new=$(cat "${TASK_SRC}/new/SRC_COMMIT")"

for VARIANT in old new old_rep; do
    case "$VARIANT" in
        old|old_rep) SRC="${TASK_SRC}/old" ;;
        new) SRC="${TASK_SRC}/new" ;;
    esac
    OUT="${OUT_ROOT}/${VARIANT}"
    WORK="${OUT_ROOT}/work/${VARIANT}_${SUB}"
    mkdir -p "$OUT" "$WORK"
    echo "=== sub-${SUB} variant=${VARIANT} start $(date -Is) ==="
    apptainer run --nv \
        --bind "${SRC}:/opt/src:ro" \
        --bind "${BIDS}:/bids:ro" \
        --bind "${OUT}:/out" \
        --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" \
        --method quick \
        --device cuda \
        --num-threads 8 \
        --skip-nidm \
        --verbose
    rm -rf "$WORK"
    test -s "${OUT}/sub-${SUB}/anat/sub-${SUB}_space-orig_dseg.nii.gz" \
        || { echo "MISSING dseg for ${VARIANT}"; exit 1; }
    echo "=== sub-${SUB} variant=${VARIANT} done $(date -Is) ==="
done
echo "ALL VARIANTS OK sub-${SUB}"
