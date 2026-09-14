#!/bin/bash
# Run scripts/diag_folding.py inside the FireANTs container on a GPU node.
# The repo's src/ is bound over /opt/src so code changes are exercised without
# rebuilding the SIF (the installed `ants-nidm` package is an editable install
# of /opt/src, so the bind is what python imports).
# Usage: sbatch scripts/diag_folding.sbatch <sif> <out_dir> [participant]
#SBATCH --job-name=ants-fireants-diag-folding
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/diag_folding_%j.out
set -euo pipefail
SIF="${1:?usage: diag_folding.sbatch <sif> <out_dir> [participant]}"
OUT="${2:?usage: diag_folding.sbatch <sif> <out_dir> [participant]}"
SUB="${3:-0051456}"
SCRIPTS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts"
DATASET=/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech
module load apptainer/1.5.2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$OUT"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/diag-XXXXXX")"
apptainer exec --nv \
    --bind "${DATASET}/sourcedata/raw:/bids:ro" \
    --bind "${DATASET}/derivatives/babs-ants-nidm:/reference:ro" \
    --bind "${SCRIPTS}:/scripts:ro" --bind "${OUT}:/diag" --bind "${WORK}:/work" \
    --bind "${SCRIPTS}/../src:/opt/src:ro" \
    "$SIF" python3 /scripts/diag_folding.py \
    --t1w "/bids/sub-${SUB}/anat/sub-${SUB}_T1w.nii.gz" \
    --reference-dseg "/reference/sub-${SUB}/anat/sub-${SUB}_space-orig_dseg.nii.gz" \
    --out /diag --batch-size 5 --threads 32
rm -rf "$WORK"
