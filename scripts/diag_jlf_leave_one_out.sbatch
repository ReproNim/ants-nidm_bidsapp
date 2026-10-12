#!/bin/bash
# Leave-one-out fusion accuracy against the OASIS-TRT-20 manual labels; see
# scripts/diag_jlf_leave_one_out.py. One array task per held-out atlas.
#
# Usage (from the repo root):
#   sbatch scripts/diag_jlf_leave_one_out.sbatch <sif> <out_dir>
#
# A copy of the repo's src/ taken when the task starts is bound over the
# container's /opt/src, so the harness and the fusion it measures come from the
# same commit, which the log records.
#SBATCH --job-name=diag-jlf-loo
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=1-20
#SBATCH --output=logs/diag_jlf_loo_%A_%a.out

set -euo pipefail
SIF="$1"; OUT="$2"
FOLD="$SLURM_ARRAY_TASK_ID"

module load apptainer/1.5.2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short src/ scripts/diag_jlf_leave_one_out.py

SNAP="$(mktemp -d "${TMPDIR:-/tmp}/loo-src-XXXXXX")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loo-work-XXXXXX")"
trap 'rm -rf "$SNAP" "$WORK"' EXIT
cp -a src/. "$SNAP/"
cp scripts/diag_jlf_leave_one_out.py "$WORK/"

mkdir -p "$OUT"
apptainer exec --nv \
    --env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 \
    --bind "${SNAP}:/opt/src:ro" \
    --bind "${WORK}:/work" \
    --bind "${OUT}:/out" \
    "$SIF" python3 /work/diag_jlf_leave_one_out.py --fold "$FOLD" --out /out --num-threads 8
