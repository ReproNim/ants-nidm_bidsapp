"""Where do the voxels max_lab_plus_one=True drops actually go, and why?

Question (2026-09-26): with the background class on, the brain stem halves,
the optic chiasm vanishes and ~8 other structures shrink. Is the flag
under-segmenting, or was flag-off over-segmenting?

Works entirely from a finished fusion run's saved probseg maps, so no GPU:

* flag-off counterfactual under the SAME registration = argmax over the real
  label probabilities only (adding the background class does not change the
  real-label posteriors, it only adds a competitor);
* sum(p_L), the posterior-expected volume, as the estimate the vote implies;
* for the voxels the flag drops: mean background vs label probability, and
  whether they sit in a 2-voxel shell of the structure (boundary) or in its
  inferior 20% (truncation);
* share of each structure's probability mass near the brain-mask edge (mask
  truncation).

Result on sub-0051456, sep25 tier-2 (FireANTs):
  brain stem  flag-on 8873, flag-off 19044, mass 9877; dropped voxels mean
  bg 0.86 vs p_L 0.12, 63% in the boundary shell; <1% of mass near the mask
  edge. Every loser's mass matches the flag-on count, not the flag-off one.
Pass --compare <second anat dir> (e.g. the delivered Aug ANTs-registered run)
to put the posterior masses of two registrations side by side: brain stem
11045 (ANTs) vs 9877 (FireANTs), pallidum 644 vs 651, i.e. engine-independent.

    python scripts/diag_jlf_posterior_boundary.py <anat_dir> [--compare <anat_dir>]
"""
import argparse
import glob
import re

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

LOSERS = [69, 63, 64, 35, 61, 62, 40, 41, 55, 56, 4, 47, 48, 38, 39]


def load(anat):
    seg_img = nib.load(glob.glob(f"{anat}/*_dseg.nii.gz")[0])
    seg = np.asarray(seg_img.dataobj).astype(int)
    files = sorted(glob.glob(f"{anat}/*_probseg.nii.gz"),
                   key=lambda f: int(re.search(r"label-(\d+)_", f)[1]))
    ids = np.array([int(re.search(r"label-(\d+)_", f)[1]) for f in files])
    P = np.stack([np.asarray(nib.load(f).dataobj, dtype=np.float32) for f in files])
    return seg, seg_img.affine, ids, P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("anat_dir")
    ap.add_argument("--compare", help="second run's anat dir: compare posterior masses")
    a = ap.parse_args()

    seg, aff, ids, P = load(a.anat_dir)
    S = P.sum(0)
    mask = S > 0
    bg = np.clip(1 - S, 0, 1)
    off = np.where(mask, ids[P.argmax(0)], 0)
    edge = ndi.distance_transform_edt(mask)
    si = list(nib.aff2axcodes(aff)).index("S") if "S" in nib.aff2axcodes(aff) else 2
    print(f"{len(ids)} maps, max sum(p) {S.max():.4f}, mask voxels {mask.sum()}")
    print(f"{'label':>5} {'on':>7} {'off':>7} {'mass':>8} | dropped: {'n':>6} {'bg':>5} "
          f"{'pL':>5} {'shell':>5} {'inf20':>5} | mass@edge")
    for L in LOSERS:
        if L not in ids:
            continue
        pl = P[np.where(ids == L)[0][0]]
        cf = off == L
        lost = cf & (seg != L)
        dist = ndi.distance_transform_edt(cf)
        zs = np.where(cf)[si]
        row = f"{L:5d} {int((seg == L).sum()):7d} {int(cf.sum()):7d} {pl.sum():8.0f} | "
        if lost.any():
            inf = (np.where(lost)[si] <= np.quantile(zs, 0.2)).mean()
            row += (f"{int(lost.sum()):8d} {bg[lost].mean():5.2f} {pl[lost].mean():5.2f} "
                    f"{(dist[lost] <= 2).mean():5.2f} {inf:5.2f}")
        else:
            row += f"{0:8d} {'':>5} {'':>5} {'':>5} {'':>5}"
        row += f" | {pl[edge <= 2].sum() / max(pl.sum(), 1e-9):.3f}"
        print(row)

    if a.compare:
        _, _, ids2, P2 = load(a.compare)
        print(f"\nposterior mass: {a.anat_dir} vs {a.compare}")
        for L in LOSERS:
            if L in ids and L in ids2:
                print(f"{L:5d} {P[np.where(ids == L)[0][0]].sum():9.0f} "
                      f"{P2[np.where(ids2 == L)[0][0]].sum():9.0f}")
        print(f"total mass mL {P.sum() / 1000:.0f} vs {P2.sum() / 1000:.0f}")


if __name__ == "__main__":
    main()
