#!/bin/bash
# Build the FireANTs container on a CPU node.
#
# The login node can build (module apptainer/1.5.2 + --fakeroot works there),
# but this build pulls a ~4 GB CUDA devel base and compiles FireANTs' nine .cu
# fused-op sources with nvcc, so it runs as a Slurm job. No GPU is needed at
# build time: TORCH_CUDA_ARCH_LIST in the %post names the targets explicitly.
#
# Usage:
#   sbatch -p <partition> scripts/build_sif.sbatch <out.sif>
# Name the output by version (e.g. ants-nidm_bidsapp-0.2.0.sif) so it never
# overwrites an image that produced a delivered derivative.
# No partition is set here: pass one with `sbatch -p <partition>`.
#SBATCH --job-name=ants-fireants-sifbuild
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sifbuild_%j.out

set -euo pipefail

# Under sbatch, BASH_SOURCE is the spooled copy in /var/spool/slurmd, so the
# repo is located from the submit directory (run `sbatch` from the repo root).
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
test -f "$REPO/Singularity" || { echo "no Singularity in $REPO; sbatch from the repo root" >&2; exit 2; }
OUT="${1:?usage: sbatch -p <partition> scripts/build_sif.sbatch <out.sif>}"
set -a; [[ -f .env ]] && . ./.env; set +a   # paths live in the gitignored .env
SCRATCH="${SCRATCH:?set SCRATCH in .env to a large scratch directory}"

# 1.5.2, not 1.1.9: the canonical Aug 30 SIF carries the label
# usage.apptainer.version: 1.5.2-1.el8, so this is the version known to get
# through this %post rootless (1.1.9 died at apt's privilege drop).
module load apptainer/1.5.2
# Some site modules export an APPTAINER_BINDPATH (e.g. a site-wide /nfs), which breaks
# `apptainer build` ("mount /nfs destination doesn't exist").
unset APPTAINER_BINDPATH
export APPTAINER_TMPDIR="${SCRATCH}/apptainer_tmp"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

cd "$REPO"
echo "host:    $(hostname)"
echo "repo:    $REPO"
echo "branch:  $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "submod:  $(git submodule status | awk '{print $1, $2}')"
echo "dirty:   $(git status --porcelain | grep -v '^?? logs/' || echo none)"
echo "output:  $OUT"
# Stamped into the image as org.opencontainers.image.revision.
GIT_COMMIT="$(git rev-parse HEAD)$(git diff --quiet HEAD -- || echo -dirty)"
echo "commit:  $GIT_COMMIT"
echo "started: $(date -Is)"
apptainer --version

# Build to a partial name so an interrupted job never leaves a plausible .sif.
PARTIAL="${OUT}.partial"
rm -f "$PARTIAL"
apptainer build --fakeroot --build-arg GIT_COMMIT="$GIT_COMMIT" "$PARTIAL" Singularity
mv "$PARTIAL" "$OUT"
echo "built:   $(date -Is)  $(du -h "$OUT" | cut -f1)"

# Post-build checks. The %post already asserts the fused ops import (set -e),
# so a finished SIF implies it passed; re-check from outside anyway, plus the
# other things tier 2 will need. None of this needs a GPU.
echo "=== post-build checks ==="
rev=$(apptainer inspect --labels "$OUT" | awk '$1 == "org.opencontainers.image.revision:" {print $2}')
echo "revision label: $rev"
test "$rev" = "$GIT_COMMIT"
apptainer exec "$OUT" python3 -c "import torch, fireants_fused_ops; print('fused ops OK (post-build)')"
# FFO_AVAILABLE is the flag src/registration/fireants.py reads at runtime to
# decide between the fused generic-label sampler and the one-hot fallback, so
# assert the exact signal the app will see rather than only the raw import.
apptainer exec "$OUT" python3 -c "from fireants.interpolator import FFO_AVAILABLE; assert FFO_AVAILABLE, 'FFO_AVAILABLE is False'; print('FFO_AVAILABLE=True')"
apptainer exec "$OUT" python3 -c "
import torch, fireants, ants
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('fireants', getattr(fireants, '__version__', '?'))
print('antspy', ants.__version__)
"
apptainer exec "$OUT" bash -c 'test -x /opt/ants-bin/antsBrainExtraction.sh && antsRegistration --version | head -2 && command -v bc'
apptainer exec "$OUT" bash -c 'ls /opt/data/OASIS-TRT-20_brains | wc -l; ls /opt/data/OASIS-TRT-20_DKT31_CMA_labels_v2 | wc -l; ls /opt/data/OASIS-30_Atropos_template | head -3'
apptainer run "$OUT" --help | head -5
echo "ALL CHECKS PASSED $(date -Is)"
