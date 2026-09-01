#!/bin/bash
# Tier-2 smoke: one real subject, both methods, on a GPU node.
#
# Proves the seam holds end to end -- FireANTs registers, ANTs fuses, outputs
# land in the per-subject layout with NIDM alongside. It does NOT check
# segmentation quality; that is tier 3.
#
# Usage:
#   sbatch scripts/tier2_smoke.sbatch <sif> <bids_dir> <out_dir> <participant>
#SBATCH --job-name=ants-fireants-tier2
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tier2_smoke_%j.out

set -euo pipefail

SIF="$1"; BIDS="$2"; OUT="$3"; SUB="$4"

nvidia-smi
mkdir -p "$OUT"

for METHOD in fusion quick; do
    echo "=== method=${METHOD} ==="
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/tier2-${METHOD}-XXXXXX")"
    # Apptainer does not reliably auto-create a missing host-side bind source;
    # create it before the bind, not after -- otherwise this can fail at the
    # bind step, before the app even starts, on a GPU allocation.
    mkdir -p "${OUT}/${METHOD}"
    apptainer run --nv \
        --bind "${BIDS}:/bids:ro" \
        --bind "${OUT}/${METHOD}:/out" \
        --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" \
        --method "$METHOD" \
        --device cuda \
        --num-threads 8 \
        --skip-nidm \
        --verbose
    rm -rf "$WORK"

    SUBDIR="${OUT}/${METHOD}/sub-${SUB}"
    for REQUIRED in "anat" "stats" "processing_summary.json"; do
        test -e "${SUBDIR}/${REQUIRED}" \
            || { echo "MISSING ${SUBDIR}/${REQUIRED}"; exit 1; }
    done
    test -s "${SUBDIR}/stats/sub-${SUB}_antsbrainvols.csv" \
        || { echo "empty brainvols for ${METHOD}"; exit 1; }
    echo "=== ${METHOD} OK ==="
done

echo "TIER 2 PASS"
