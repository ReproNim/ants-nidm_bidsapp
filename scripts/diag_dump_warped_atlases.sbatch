#!/bin/bash
# Dump one subject's warped atlases; see scripts/diag_dump_warped_atlases.py.
#
# Usage:
#   sbatch scripts/diag_dump_warped_atlases.sbatch <sif> <bids_dir> <out_dir> <participant>
#
# The repo's src/ is bound over the container's, so this runs the working-tree
# code rather than whatever the SIF was built from.
# No partition is set here: pass one with `sbatch -p <partition>`.
#SBATCH --job-name=diag-dump-warped
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diag_dump_warped_%j.out

set -euo pipefail
SIF="$1"; BIDS="$2"; OUT="$3"; SUB="$4"

module load apptainer/1.5.2
nvidia-smi
git rev-parse HEAD
git status --short src/

mkdir -p "$OUT"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/diag-dump-XXXXXX")"
apptainer exec --nv \
    --bind "$(pwd)/src:/opt/src:ro" \
    --bind "$(pwd)/scripts/diag_dump_warped_atlases.py:/scripts/diag_dump_warped_atlases.py:ro" \
    --bind "${BIDS}:/bids:ro" \
    --bind "${OUT}:/out" \
    --bind "${WORK}:/work" \
    "$SIF" python3 /scripts/diag_dump_warped_atlases.py \
        --bids /bids --participant "$SUB" --out /out --num-threads 8
rm -rf "$WORK"
