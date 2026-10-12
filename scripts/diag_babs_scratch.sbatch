#!/bin/bash
# One subject through --method fusion with the container flags BABS uses
# (examples/babs-ants-nidm-bids-study.yaml): --userns --no-home
# --writable-tmpfs and nothing bound over /work. The image sets TMPDIR=/work,
# which is then Apptainer's size-capped tmpfs overlay, so the app must detect
# that and move its scratch to <output_dir>/tmp. tier2_smoke.sbatch binds a
# real /work and never exercises this path.
#
# Usage (from the repo root):
#   sbatch scripts/diag_babs_scratch.sbatch <sif> <bids_dir> <out_dir> <participant> \
#       [<session>] [<nidm_input_dir>]
# A session is passed the way BABS passes it: positional level `participant`
# plus --session-label. A NIDM input dir is bound read-only at /nidm.
# No partition is set here: pass one with `sbatch -p <partition>`.
#SBATCH --job-name=ants-babs-scratch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/diag_babs_scratch_%j.out

set -euo pipefail
SIF="$1"; BIDS="$2"; OUT="$3"; SUB="$4"; SES="${5:-}"; NIDM="${6:-}"

extra_binds=(); extra_args=()
[[ -n "$SES" ]] && extra_args+=(--session-label "$SES")
if [[ -n "$NIDM" ]]; then
    extra_binds+=(--bind "${NIDM}:/nidm:ro"); extra_args+=(--nidm-input-dir /nidm)
fi

module load apptainer/1.5.2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"

apptainer run --nv --userns --no-home --writable-tmpfs \
    --bind "${BIDS}:/bids:ro" \
    --bind "${OUT}:/out" \
    "${extra_binds[@]}" \
    "$SIF" /bids /out participant \
    --participant-label "$SUB" \
    --method fusion \
    --device cuda \
    --num-threads 8 \
    --skip-bids-validation \
    "${extra_args[@]}" \
    --verbose 2>&1 | tee "${OUT}/app.log"

SUBDIR="${OUT}/sub-${SUB}"; STEM="sub-${SUB}"
if [[ -n "$SES" ]]; then SUBDIR="${SUBDIR}/ses-${SES}"; STEM="${STEM}_ses-${SES}"; fi
fail=0
grep -q "TMPDIR scratch .* using /out/tmp" "${OUT}/app.log" \
    && echo "OK fallback warning logged" || { echo "FAIL no fallback warning"; fail=1; }
for f in nidm.ttl ants_cde.ttl "stats/${STEM}_antsbrainvols.csv" "stats/${STEM}_antslabelstats.csv" \
         "anat/${STEM}_space-orig_dseg.nii.gz" "anat/${STEM}_space-orig_dseg.json"; do
    test -s "${SUBDIR}/${f}" && echo "OK ${f}" || { echo "FAIL missing ${f}"; fail=1; }
done
if [[ -n "$NIDM" ]]; then
    grep -q "Found NIDM input file" "${OUT}/app.log" \
        && echo "OK NIDM input used" || { echo "FAIL NIDM input not found"; fail=1; }
fi
grep -q '"input_file"' "${SUBDIR}/processing_summary.json" 2>/dev/null \
    && echo "OK input recorded: $(grep -o '"input_file": "[^"]*"' "${SUBDIR}/processing_summary.json")" \
    || { echo "FAIL input_file missing from processing_summary.json"; fail=1; }
grep -qi "no space left" "${OUT}/app.log" && { echo "FAIL ENOSPC in log"; fail=1; } || echo "OK no ENOSPC"
echo "left in ${OUT}/tmp (outside sub-${SUB}/, so not in the BABS zip):"
du -sh "${OUT}/tmp" 2>/dev/null || echo "  none"
[[ $fail == 0 ]] && echo "BABS SCRATCH PASS" || { echo "BABS SCRATCH FAIL"; exit 1; }
