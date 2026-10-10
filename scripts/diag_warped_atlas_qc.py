"""Are the warped atlases correct, and does the size bias come before or in fusion?

Runs on the output of diag_dump_warped_atlases.py (CPU only). Two questions:

1. Are the 20 warped atlases trustworthy inputs?
   A  geometry      every warped file is on the target grid
   B  label set     warping neither loses nor invents label ids
   C  brain fit     Dice(warped atlas brain > 0, target brain mask)
   D  intensity     NCC between warped atlas and target inside the mask
   E  FreeSurfer    per-structure Dice of each warped atlas's labels with the
                    subject's own FreeSurfer aseg (resampled to the target grid,
                    ids mapped through label_vocabulary). FreeSurfer is an
                    independent segmentation of the same scan, so a well-aligned
                    atlas should overlap it about as well as the two protocols
                    overlap each other (bijection overlaps 0.4-0.7)

2. Where does the tissue-dependent size bias enter?
   F  per-atlas     each warped label's volume vs FreeSurfer, median over atlases
   G  plain vote    unweighted majority vote over the 20 warped label maps, with
                    background as a candidate, vs FreeSurfer
   H  fusion        production joint_label_fusion on the same atlases vs FreeSurfer
   If F and G are near FreeSurfer but H is not, the intensity weighting in the
   fusion causes the bias. If F is already biased, registration or label
   interpolation does.

Also reports fusion_dseg against an earlier normal run of the same subject
(--reference-dseg) to confirm the dump reproduces production.

    set -a; . ./.env; set +a
    python scripts/diag_warped_atlas_qc.py <dump_dir> --participant 0051456 \
        [--reference-dseg <anat>/sub-..._space-orig_dseg.nii.gz] [--json out.json]
        [--fusion-dseg <sweep_dir>/<variant>/fusion_dseg.nii.gz]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.antspy.label_vocabulary import NEUROMORPHOMETRICS_TO_ASEG  # noqa: E402

STRUCTS = {35: "BrainStem", 40: "R CblmWM", 41: "L CblmWM", 55: "R Pall", 56: "L Pall",
           61: "R VDC", 62: "L VDC", 47: "R Hippo", 48: "L Hippo", 36: "R Caud",
           37: "L Caud", 57: "R Putamen", 58: "L Putamen", 59: "R Thal", 60: "L Thal",
           51: "R LatVent", 52: "L LatVent", 4: "3rdVent"}


def arr(path, dtype=None):
    a = np.asarray(nib.load(str(path)).dataobj)
    return a.astype(dtype) if dtype else a


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def ncc(a, b, m):
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump_dir")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--freesurfer", default=os.path.join(
        os.environ.get("CALTECH_DERIVATIVES", ""), "babs-freesurfer-nidm"))
    ap.add_argument("--atlas-labels", default=str(REPO / "data/OASIS-TRT-20_DKT31_CMA_labels_v2"))
    ap.add_argument("--reference-dseg")
    ap.add_argument("--fusion-dseg", help="score this segmentation as H instead of "
                    "<dump_dir>/fusion_dseg.nii.gz (e.g. a diag_jlf_param_sweep.py variant)")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dump_dir)
    res = {}

    timg = nib.load(str(d / "target_masked_brain.nii.gz"))
    target = np.asarray(timg.dataobj, dtype=np.float32)
    mask = arr(d / "brain_mask.nii.gz") > 0
    man = json.loads((d / "manifest.json").read_text())
    n = len(man["atlases"])

    aseg_path = Path(a.freesurfer) / f"sub-{a.participant}/freesurfer/mri/aseg.mgz"
    aseg = np.asarray(resample_from_to(nib.load(str(aseg_path)), timg, order=0).dataobj).astype(int)
    fs = {L: aseg == NEUROMORPHOMETRICS_TO_ASEG[L] for L in STRUCTS}
    fs_vol = {L: int(m.sum()) for L, m in fs.items()}
    res["fs_brain_overlap"] = float(dice(aseg > 0, mask))
    print(f"FreeSurfer aseg resampled to target grid; Dice(aseg>0, brain mask) "
          f"= {res['fs_brain_overlap']:.3f}")

    # --- Part 1: per-atlas checks
    geom_ok, rows, per_atlas_vol, per_atlas_dice = True, [], {L: [] for L in STRUCTS}, {L: [] for L in STRUCTS}
    labs = []
    for i in range(1, n + 1):
        bimg = nib.load(str(d / f"atlas-{i:02d}_brain.nii.gz"))
        limg = nib.load(str(d / f"atlas-{i:02d}_labels.nii.gz"))
        ok = (bimg.shape == timg.shape and limg.shape == timg.shape
              and np.allclose(bimg.affine, timg.affine, atol=1e-3)
              and np.allclose(limg.affine, timg.affine, atol=1e-3))
        geom_ok &= ok
        b = np.asarray(bimg.dataobj, dtype=np.float32)
        lab = np.rint(np.asarray(limg.dataobj)).astype(np.int32)
        labs.append(lab)
        native = arr(Path(a.atlas_labels) / man["labels"][i - 1]).astype(np.int32)
        nid, wid = set(np.unique(native)), set(np.unique(lab))
        for L in STRUCTS:
            m = lab == L
            per_atlas_vol[L].append(int(m.sum()))
            per_atlas_dice[L].append(float(dice(m, fs[L])))
        rows.append(dict(atlas=i, geometry_ok=bool(ok), lost_ids=sorted(int(x) for x in nid - wid),
                         new_ids=sorted(int(x) for x in wid - nid),
                         brain_dice=float(dice(b > 0, mask)), ncc=ncc(b, target, mask)))
    res["atlases"] = rows
    print(f"\nA geometry on target grid: {'all ok' if geom_ok else 'MISMATCH'}")
    print(f"{'atlas':>5} {'brainDice':>9} {'NCC':>6}  lost/new label ids")
    for r in rows:
        print(f"{r['atlas']:5d} {r['brain_dice']:9.3f} {r['ncc']:6.3f}  {r['lost_ids']} / {r['new_ids']}")

    # --- Part 2: where does the bias enter
    print("\nG building plain majority vote (background is a candidate)...", flush=True)
    stack_ids = sorted(set().union(*(np.unique(x).tolist() for x in labs)))
    best_c = np.zeros(target.shape, np.uint8)
    best_l = np.zeros(target.shape, np.int32)
    for L in stack_ids:
        c = np.zeros(target.shape, np.uint8)
        for lab in labs:
            c += lab == L
        upd = c > best_c
        best_c[upd], best_l[upd] = c[upd], L
    vote = np.where(mask, best_l, 0)
    fusion = np.rint(arr(a.fusion_dseg or d / "fusion_dseg.nii.gz")).astype(np.int32)

    print(f"\n{'structure':>10} {'FS vox':>7} | F per-atlas: {'vol/FS':>6} {'Dice':>5} "
          f"| G vote: {'vol/FS':>6} {'Dice':>5} | H fusion: {'vol/FS':>6} {'Dice':>5}")
    res["structures"] = {}
    for L, name in STRUCTS.items():
        f = fs_vol[L]
        pv = np.median(per_atlas_vol[L]) / f
        pdice = np.median(per_atlas_dice[L])
        vv, vd = (vote == L).sum() / f, dice(vote == L, fs[L])
        hv, hd = (fusion == L).sum() / f, dice(fusion == L, fs[L])
        res["structures"][name] = dict(label=L, fs_vox=f, per_atlas_ratio=float(pv),
                                       per_atlas_dice=float(pdice), vote_ratio=float(vv),
                                       vote_dice=float(vd), fusion_ratio=float(hv),
                                       fusion_dice=float(hd))
        print(f"{name:>10} {f:7d} | {pv:19.2f} {pdice:5.2f} | {vv:14.2f} {vd:5.2f} "
              f"| {hv:16.2f} {hd:5.2f}")

    if a.reference_dseg:
        ref = np.rint(arr(a.reference_dseg)).astype(np.int32)
        print("\nfusion_dseg vs reference run (percent difference in voxels)")
        res["reproduction"] = {}
        for L, name in STRUCTS.items():
            r, h = int((ref == L).sum()), int((fusion == L).sum())
            pct = 100 * (h - r) / r if r else float("nan")
            res["reproduction"][name] = pct
            print(f"{name:>10} {r:7d} {h:7d} {pct:+6.1f}%")

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
