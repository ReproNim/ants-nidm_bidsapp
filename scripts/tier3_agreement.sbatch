#!/bin/bash
# Tier-3: run 3 Caltech subjects through the FireANTs container, then compare
# against the delivered ANTs derivative.
#SBATCH --job-name=ants-fireants-tier3
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tier3_%j.out

set -euo pipefail

SIF="${1:?usage: tier3_agreement.sbatch <sif> <out_dir>}"

# Compute nodes have no system apptainer; only the module provides one. Do not
# rely on the submitting shell having loaded it -- sbatch exports that
# environment, which is why this script once worked and then failed with
# "apptainer: command not found" from a fresh shell.
module load apptainer/1.5.2
OUT="${2:?usage: tier3_agreement.sbatch <sif> <out_dir>}"

# Under sbatch, BASH_SOURCE is the spooled copy in /var/spool/slurmd, so the
# scripts dir is located from the submit directory (sbatch from the repo root).
SCRIPTS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts"
test -f "$SCRIPTS/tier3_agreement.py" || { echo "tier3_agreement.py not in $SCRIPTS; sbatch from the repo root" >&2; exit 2; }

DATASET=/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech
BIDS="${DATASET}/sourcedata/raw"
REFERENCE="${DATASET}/derivatives/babs-ants-nidm"
SUBJECTS=(0051456 0051457 0051458)

nvidia-smi
mkdir -p "$OUT"

for SUB in "${SUBJECTS[@]}"; do
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/tier3-${SUB}-XXXXXX")"
    apptainer run --nv \
        --bind "${BIDS}:/bids:ro" --bind "${OUT}:/out" --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" --method fusion --device cuda \
        --num-threads 8 --skip-nidm --verbose
    rm -rf "$WORK"
done

# The comparison runs INSIDE the container, via `apptainer exec`.
#
# It used to run on the compute node's host interpreter, which has no
# guaranteed nibabel or numpy and certainly no fireants -- so the expected
# outcome was ModuleNotFoundError, and the failure mode if numpy and nibabel
# happened to resolve was worse: surface_metrics silently NaN-filled every
# surface-Dice and Hausdorff value that spec section 8 requires be reported,
# and the run still printed TIER 3 PASS. tier3_agreement.py now refuses to
# start without fireants.scripts.evalutils rather than degrading.
apptainer exec \
    --bind "${OUT}:/out" \
    --bind "${REFERENCE}:/reference:ro" \
    --bind "${SCRIPTS}:/scripts:ro" \
    "$SIF" python3 /scripts/tier3_agreement.py \
    --new /out --reference /reference \
    --subjects "${SUBJECTS[@]}" --out /out/tier3_report.csv
