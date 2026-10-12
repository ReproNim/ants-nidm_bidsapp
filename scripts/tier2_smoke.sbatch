#!/bin/bash
# Tier-2 smoke: one real subject, both methods, on a GPU node.
#
# Proves the seam holds end to end -- FireANTs registers, ANTs fuses, outputs
# land in the per-subject layout with nidm.ttl and ants_cde.ttl alongside
# It does NOT check segmentation quality; the leave-one-out gate does.
#
# Usage:
#   sbatch scripts/tier2_smoke.sbatch <sif> <bids_dir> <out_dir> <participant>
#
# With SRC_FROM_REPO=1 in the environment, a copy of the repo's src/
# (including the ants_seg_to_nidm submodule) taken when the job starts is bound
# over the container's /opt/src, so a code change can be checked end to end
# without rebuilding the SIF. It is a copy, not the live tree: the NIDM
# conversion and the second method start long after the first, and edits made
# in between would otherwise run a mix of old and new code.
#SBATCH --job-name=ants-fireants-tier2
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tier2_smoke_%j.out

set -euo pipefail

SIF="$1"; BIDS="$2"; OUT="$3"; SUB="$4"

# Compute nodes have no system apptainer; only the module provides one. Do not
# rely on the submitting shell having loaded it -- sbatch exports that
# environment, which is why this script once worked and then failed with
# "apptainer: command not found" from a fresh shell.
module load apptainer/1.5.2

nvidia-smi
mkdir -p "$OUT"

SRC_BIND=()
if [[ "${SRC_FROM_REPO:-0}" == "1" ]]; then
    git rev-parse HEAD
    git status --short src/
    SRC_SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/tier2-src-XXXXXX")"
    cp -a src/. "$SRC_SNAPSHOT/"
    trap 'rm -rf "$SRC_SNAPSHOT"' EXIT
    SRC_BIND=(--bind "${SRC_SNAPSHOT}:/opt/src:ro")
fi

for METHOD in fusion quick; do
    echo "=== method=${METHOD} ==="
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/tier2-${METHOD}-XXXXXX")"
    # Apptainer does not reliably auto-create a missing host-side bind source;
    # create it before the bind, not after -- otherwise this can fail at the
    # bind step, before the app even starts, on a GPU allocation.
    mkdir -p "${OUT}/${METHOD}"
    # NIDM conversion is deliberately NOT skipped. Tier 2 exists
    # to confirm outputs land in the per-subject layout *with nidm.ttl
    # and ants_cde.ttl present*; --skip-nidm turned the one thing this tier is
    # supposed to prove into something it never exercised. The app falls back
    # to producing a fresh NIDM graph when no input NIDM is found, so no extra
    # bind is needed -- pass --nidm-input-dir if you want to augment one.
    apptainer run --nv \
        "${SRC_BIND[@]}" \
        --bind "${BIDS}:/bids:ro" \
        --bind "${OUT}/${METHOD}:/out" \
        --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" \
        --method "$METHOD" \
        --device cuda \
        --num-threads 8 \
        --verbose
    rm -rf "$WORK"

    SUBDIR="${OUT}/${METHOD}/sub-${SUB}"
    for REQUIRED in "anat" "stats" "processing_summary.json"; do
        test -e "${SUBDIR}/${REQUIRED}" \
            || { echo "MISSING ${SUBDIR}/${REQUIRED}"; exit 1; }
    done
    test -s "${SUBDIR}/stats/sub-${SUB}_antsbrainvols.csv" \
        || { echo "empty brainvols for ${METHOD}"; exit 1; }
    # Both TTLs, and both non-empty: ants_seg_to_nidm writes the shared CDE
    # vocabulary next to nidm.ttl rather than merging it in, so a missing
    # ants_cde.ttl leaves every ants_* predicate unresolvable.
    for TTL in "nidm.ttl" "ants_cde.ttl"; do
        test -s "${SUBDIR}/${TTL}" \
            || { echo "MISSING OR EMPTY ${SUBDIR}/${TTL}"; exit 1; }
    done
    wc -c "${SUBDIR}/nidm.ttl" "${SUBDIR}/ants_cde.ttl"
    echo "=== ${METHOD} OK ==="
done

echo "TIER 2 PASS"
