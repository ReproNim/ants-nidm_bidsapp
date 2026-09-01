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
OUT="${2:?usage: tier3_agreement.sbatch <sif> <out_dir>}"

DATASET=/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech
BIDS="${DATASET}/sourcedata"
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

python3 scripts/tier3_agreement.py \
    --new "$OUT" --reference "$REFERENCE" \
    --subjects "${SUBJECTS[@]}" --out "${OUT}/tier3_report.csv"
