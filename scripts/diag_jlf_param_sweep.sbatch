#!/bin/bash
# CPU-only JLF parameter sweep; see scripts/diag_jlf_param_sweep.py.
#
# Usage:
#   sbatch scripts/diag_jlf_param_sweep.sbatch <sif> <dump_dir> <out_dir>
#SBATCH --job-name=diag-jlf-sweep
#SBATCH --partition=pi_satra
#SBATCH --array=0-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/diag_jlf_sweep_%A_%a.out

set -euo pipefail
SIF="$1"; DUMP="$2"; OUT="$3"
VARIANTS=(baseline rsearch0 rad1 beta1 usecor normtarget normtarget_usecor)
V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

module load apptainer/1.5.2
mkdir -p "$OUT"
apptainer exec \
    --env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 \
    --bind "$(pwd)/scripts/diag_jlf_param_sweep.py:/scripts/diag_jlf_param_sweep.py:ro" \
    --bind "${DUMP}:/dump:ro" \
    --bind "${OUT}:/out" \
    "$SIF" python3 /scripts/diag_jlf_param_sweep.py /dump /out "$V"
