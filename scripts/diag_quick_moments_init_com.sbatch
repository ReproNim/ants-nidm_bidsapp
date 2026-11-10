#!/bin/bash
# CPU companion to diag_quick_moments_init.sbatch: the centre-of-mass init
# translation quick gets with the old and the new moments inputs (#16).
# No GPU needed: the moments step is cheap on CPU; the brain extraction
# dominates the runtime.
#
# Usage (from the repo root, partition on the command line):
#   sbatch -p <partition> --array=0-<n-1> -o <log_dir>/qmicom_%A_%a.out \
#     scripts/diag_quick_moments_init_com.sbatch <sif> <bids_dir> <out_root> <ref> <participant> ...
#SBATCH --job-name=ants-quick-moments-com
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --constraint=rocky8

set -euo pipefail
SIF="$1"; BIDS="$2"; OUT_ROOT="$3"; REF="$4"; shift 4
SUBJECTS=("$@")
SUB="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
REPO="$(pwd)"
module load apptainer/1.5.2

SHA="$(git -C "$REPO" rev-parse "${REF}^{commit}")"
SNAP="${OUT_ROOT}/com_src/task${SLURM_ARRAY_TASK_ID}"
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$REPO" archive "$SHA" src scripts/diag_quick_moments_init_com.py | tar -x -C "$SNAP"
echo "commit=${SHA}"
WORK="${OUT_ROOT}/com_work/sub-${SUB}"
mkdir -p "$WORK" "${OUT_ROOT}/com"
export OMP_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
apptainer exec \
    --bind "${SNAP}/src:/opt/src:ro" \
    --bind "${SNAP}/scripts:/opt/diag:ro" \
    --bind "${BIDS}:/bids:ro" \
    --bind "${WORK}:/work" \
    --bind "${OUT_ROOT}/com:/com" \
    "$SIF" python /opt/diag/diag_quick_moments_init_com.py \
        "/bids/sub-${SUB}/anat/sub-${SUB}_T1w.nii.gz" /work "/com/sub-${SUB}.json"
rm -rf "$WORK"
