"""Which joint_label_fusion setting produces the tissue-dependent size bias?

On sub-0051456 (Oct 3), a plain majority vote over the 20 warped atlases
matches FreeSurfer (brain stem 0.89x, cerebellar WM 0.95x, pallidum 0.91x) while
production joint_label_fusion on the same atlases does not (0.39x, 0.19x,
0.39x; hippocampus and caudate ~1.5x). This reruns the fusion on the atlases
saved by diag_dump_warped_atlases.py, changing one parameter at a time from
production, so each variant can be scored with

    python scripts/diag_warped_atlas_qc.py <dump_dir> --participant <id> \
        --fusion-dseg <out>/<variant>/fusion_dseg.nii.gz

Variants (production: beta=4, rad=2, rho=0.01, r_search=3, usecor=False,
max_lab_plus_one=True):
  baseline      production settings (checks the fusion is reproducible)
  rsearch0      r_search=0: no search for the best-matching patch nearby
  rad1          rad=1: 3x3x3 patches instead of 5x5x5
  beta1         beta=1: weaker sharpening of the intensity weights
  usecor        usecor=True: correlation instead of mean squared difference
  normtarget    production settings, but the target rescaled linearly so its
                in-mask median equals the atlases' median in-mask median. The
                N4 target's median is ~388 against ~1100 for the atlases, and
                mean squared difference is not scale-invariant
  normtarget_usecor  both of the above

Result on sub-0051456 (Oct 3, jobs 24713328 and 24715225; scored JSON
recorded as jlf_sweep_<variant>_sub-0051456_2026-10-03.json),
volume / FreeSurfer (Dice with FreeSurfer):

              brain stem   R cblm WM    R pallidum   R caudate    R lat vent
  baseline    0.39 (0.54)  0.19 (0.30)  0.39 (0.24)  1.70 (0.56)  1.46 (0.80)
  rad1        0.37 (0.52)  0.16 (0.26)  0.33 (0.24)  1.49 (0.54)  1.66 (0.74)
  beta1       0.39 (0.54)  0.19 (0.29)  0.39 (0.24)  1.76 (0.54)  1.50 (0.77)
  rsearch0    0.89 (0.90)  0.99 (0.81)  0.91 (0.86)  0.81 (0.77)  0.88 (0.91)
  usecor      0.90 (0.91)  0.93 (0.82)  0.81 (0.84)  0.87 (0.81)  0.93 (0.94)
  normtarget  0.96 (0.91)  1.04 (0.80)  0.87 (0.63)  1.29 (0.77)  1.09 (0.94)
  plain vote  0.89 (0.90)  0.95 (0.81)  0.91 (0.85)  1.23 (0.58)  0.50 (0.66)

baseline reproduces the production fusion voxel for voxel. The bias comes from
the intensity-scale mismatch acting through mean-squared-difference patch
search: removing the search, switching to a scale-invariant similarity, or
rescaling the target each remove most of it. One subject only.

Must run inside the app container; see diag_jlf_param_sweep.sbatch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import ants
import numpy as np

PRODUCTION = dict(beta=4.0, rad=2, rho=0.01, usecor=False, r_search=3,
                  nonnegative=False, no_zeroes=False, max_lab_plus_one=True)
VARIANTS = {
    "baseline": {},
    "rsearch0": {"r_search": 0},
    "rad1": {"rad": 1},
    "beta1": {"beta": 1.0},
    "usecor": {"usecor": True},
    "normtarget": {},
    "normtarget_usecor": {"usecor": True},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump_dir")
    ap.add_argument("out_dir")
    ap.add_argument("variant", choices=sorted(VARIANTS))
    a = ap.parse_args()

    d = Path(a.dump_dir)
    n = len(json.loads((d / "manifest.json").read_text())["atlases"])
    target = ants.image_read(str(d / "target_masked_brain.nii.gz"))
    mask = ants.image_read(str(d / "brain_mask.nii.gz"))
    atlases = [ants.image_read(str(d / f"atlas-{i:02d}_brain.nii.gz")) for i in range(1, n + 1)]
    labels = [ants.image_read(str(d / f"atlas-{i:02d}_labels.nii.gz")) for i in range(1, n + 1)]

    if a.variant.startswith("normtarget"):
        m = mask.numpy() > 0
        atlas_med = float(np.median([np.median(x.numpy()[m]) for x in atlases]))
        scale = atlas_med / float(np.median(target.numpy()[m]))
        target = target * scale
        print(f"rescaled target by {scale:.3f} (atlas median {atlas_med:.1f})")

    params = {**PRODUCTION, **VARIANTS[a.variant]}
    t0 = time.time()
    fusion = ants.joint_label_fusion(target_image=target, target_image_mask=mask,
                                     atlas_list=atlases, label_list=labels, verbose=False,
                                     **params)
    out = Path(a.out_dir) / a.variant
    out.mkdir(parents=True, exist_ok=True)
    ants.image_write(fusion["segmentation"], str(out / "fusion_dseg.nii.gz"))
    (out / "params.json").write_text(json.dumps(
        {"variant": a.variant, "params": params, "seconds": round(time.time() - t0)}, indent=1))
    print("done", a.variant, params, round(time.time() - t0), "s")


if __name__ == "__main__":
    sys.exit(main())
