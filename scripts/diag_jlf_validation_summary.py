"""Which joint_label_fusion fix holds up across subjects? Score every variant vs FreeSurfer.

diag_jlf_param_sweep.py found on sub-0051456 that the fused label map is
mis-sized by tissue type (WM-like structures shrink, GM/CSF grow) and that
usecor=True, r_search=0 or rescaling the target each remove the bias. This
scores those variants on several subjects at once, against each subject's own
FreeSurfer aparc.DKTatlas+aseg, and adds what diag_warped_atlas_qc.py leaves
out: all 37 mapped subcortical labels (not 18) and the 62 DKT cortical parcels,
so a fix that repairs subcortical sizes at the cost of cortex shows up.

Per subject, the FreeSurfer reference and the plain majority vote are built
once; each variant's fusion_dseg is then only compared. Subcortical ids are
mapped through label_vocabulary; cortical DKT ids are the same in both.

Metrics per label: Dice with FreeSurfer, and log2(volume / FreeSurfer volume).
Groups summarise them; |log2 ratio| is the size error, the signed mean is the
direction of the bias. The two protocols differ (bijection overlaps 0.4-0.7 on
some structures), so absolute Dice is capped below 1. Compare variants with
each other, not with 1.

    set -a; . ./.env; set +a
    python scripts/diag_jlf_validation_summary.py \
        --subject 0051456=<dump_dir> --subject 0051457=<dump_dir> ... \
        [--variants baseline rsearch0 usecor normtarget normtarget_usecor] \
        [--json out.json]

Each <dump_dir> is a diag_dump_warped_atlases.py output with the variants under
<dump_dir>/sweep/<variant>/fusion_dseg.nii.gz. Missing variants are skipped.

Result on 6 Caltech subjects (0051456/57/63/70/78/86; dumps + sweeps from
jobs 24711231, 24719700-24719711; JSON in
docs/superpowers/specs/analysis/jlf_validation_6sub_2026-10-10.json),
group mean Dice with FreeSurfer (mean |log2 volume ratio|):

              WM-like      GM subcort   CSF          cortex
  vote        0.85 (0.15)  0.82 (0.28)  0.80 (0.36)  0.56 (0.41)
  baseline    0.41 (1.46)  0.59 (0.34)  0.47 (1.60)  0.34 (0.34)
  rsearch0    0.85 (0.16)  0.85 (0.27)  0.86 (0.17)  0.67 (0.19)
  usecor      0.86 (0.18)  0.86 (0.25)  0.89 (0.11)  0.72 (0.27)
  normtarget  0.75 (0.26)  0.76 (0.34)  0.86 (0.26)  0.66 (0.57)

normtarget_usecor equals usecor (correlation is scale-invariant). usecor has
the highest mean Dice on all 6 subjects (0.69-0.76 vs baseline 0.36-0.40) and
was adopted in production; rsearch0 sizes WM-like structures and cortex
slightly better but overlaps less everywhere. normtarget fixes sizes only
partly and drops pallidum Dice to 0.54-0.57.
"""
import argparse
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

KEY = {35: "BrainStem", 40: "R CblmWM", 41: "L CblmWM", 55: "R Pall", 56: "L Pall",
       61: "R VDC", 62: "L VDC", 47: "R Hippo", 48: "L Hippo", 36: "R Caud",
       37: "L Caud", 57: "R Putamen", 58: "L Putamen", 59: "R Thal", 60: "L Thal",
       51: "R LatVent", 52: "L LatVent", 4: "3rdVent"}
GROUPS = {
    "wm_like": [35, 40, 41, 55, 56, 61, 62],
    "gm_subcort": [36, 37, 47, 48, 57, 58, 59, 60],
    "csf": [4, 51, 52],
    "subcort_all": sorted(NEUROMORPHOMETRICS_TO_ASEG),
    "cortex": None,  # filled with the DKT ids present in the atlases
}
DEFAULT_VARIANTS = ["baseline", "rsearch0", "usecor", "normtarget", "normtarget_usecor"]


def load(path, dtype=np.int32):
    return np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(dtype)


def plain_vote(dump, n, mask):
    """Unweighted majority vote over the warped atlases, background a candidate."""
    stack = np.stack([load(dump / f"atlas-{i:02d}_labels.nii.gz")[mask] for i in range(1, n + 1)])
    ids = np.unique(stack)
    best_c = np.zeros(stack.shape[1], np.uint8)
    best_l = np.zeros(stack.shape[1], np.int32)
    for L in ids:
        c = (stack == L).sum(0).astype(np.uint8)
        upd = c > best_c
        best_c[upd], best_l[upd] = c[upd], L
    out = np.zeros(mask.shape, np.int32)
    out[mask] = best_l
    return out, set(ids.tolist())


def score(seg, ref, labels):
    out = {}
    for L, refid in labels.items():
        s, r = seg == L, ref == refid
        sv, rv = int(s.sum()), int(r.sum())
        if rv == 0:
            continue
        out[L] = dict(vox=sv, fs_vox=rv, dice=float(2 * np.logical_and(s, r).sum() / (sv + rv)),
                      log2_ratio=float(np.log2(sv / rv)) if sv else float("-inf"))
    return out


def group_stats(per_label, ids):
    rows = [per_label[L] for L in ids if L in per_label]
    lr = np.array([r["log2_ratio"] for r in rows])
    lr = np.clip(lr, -5, 5)  # a vanished label counts as 32x too small, not -inf
    return dict(n=len(rows), dice=float(np.mean([r["dice"] for r in rows])),
                abs_log2=float(np.mean(np.abs(lr))), signed_log2=float(np.mean(lr)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", action="append", required=True, metavar="ID=DUMP_DIR")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--freesurfer", default=os.path.join(
        os.environ.get("CALTECH_DERIVATIVES", ""), "babs-freesurfer-nidm"))
    ap.add_argument("--json")
    a = ap.parse_args()

    res = {"variants": a.variants, "subjects": {}}
    for spec in a.subject:
        sid, dump = spec.split("=", 1)
        dump = Path(dump)
        timg = nib.load(str(dump / "target_masked_brain.nii.gz"))
        mask = load(dump / "brain_mask.nii.gz") > 0
        n = len(json.loads((dump / "manifest.json").read_text())["atlases"])
        ref_img = nib.load(str(Path(a.freesurfer) / f"sub-{sid}/freesurfer/mri/aparc.DKTatlas+aseg.mgz"))
        ref = np.asarray(resample_from_to(ref_img, timg, order=0).dataobj).astype(np.int32)

        print(f"sub-{sid}: plain vote over {n} atlases...", flush=True)
        vote, atlas_ids = plain_vote(dump, n, mask)
        cortex = sorted(L for L in atlas_ids if 1000 < L < 3000)
        labels = {**NEUROMORPHOMETRICS_TO_ASEG, **{L: L for L in cortex}}
        groups = {**GROUPS, "cortex": cortex}

        subj = {}
        methods = [("vote", vote)] + [
            (v, dump / "sweep" / v / "fusion_dseg.nii.gz") for v in a.variants]
        for name, seg in methods:
            if not isinstance(seg, np.ndarray):
                if not seg.exists():
                    print(f"  {name}: missing, skipped")
                    continue
                seg = load(seg)
            per = score(seg, ref, labels)
            subj[name] = dict(labels={str(k): v for k, v in per.items()},
                              groups={g: group_stats(per, ids) for g, ids in groups.items()})
        res["subjects"][sid] = subj

    methods = ["vote"] + a.variants
    subs = res["subjects"]

    def agg(m, g, k):
        vals = [subs[s][m]["groups"][g][k] for s in subs if m in subs[s]]
        return (np.mean(vals), len(vals)) if vals else (np.nan, 0)

    print("\nGroup means over subjects: Dice / mean|log2 vol ratio| / signed log2 (n subjects)")
    print(f"{'method':>18} " + " ".join(f"{g:>22}" for g in GROUPS))
    res["summary"] = {}
    for m in methods:
        cells, res["summary"][m] = [], {}
        for g in GROUPS:
            d, nd = agg(m, g, "dice")
            e, _ = agg(m, g, "abs_log2")
            sg, _ = agg(m, g, "signed_log2")
            res["summary"][m][g] = dict(dice=d, abs_log2=e, signed_log2=sg, n_subjects=nd)
            cells.append(f"{d:.3f}/{e:.2f}/{sg:+.2f} ({nd})")
        print(f"{m:>18} " + " ".join(f"{c:>22}" for c in cells))

    print("\nKey structures, median over subjects: vol/FS (Dice)")
    print(f"{'structure':>10} " + " ".join(f"{m:>17}" for m in methods))
    res["key_structures"] = {}
    for L, name in KEY.items():
        cells, res["key_structures"][name] = [], {}
        for m in methods:
            rows = [subs[s][m]["labels"].get(str(L)) for s in subs if m in subs[s]]
            rows = [r for r in rows if r]
            if not rows:
                cells.append(f"{'-':>17}")
                continue
            ratio = float(np.median([2 ** r["log2_ratio"] for r in rows]))
            dc = float(np.median([r["dice"] for r in rows]))
            res["key_structures"][name][m] = dict(ratio=ratio, dice=dc, n=len(rows))
            cells.append(f"{ratio:9.2f} ({dc:.2f})")
        print(f"{name:>10} " + " ".join(cells))

    print("\nPer subject, mean Dice over subcortical + cortical labels (best marked *)")
    res["per_subject_best"] = {}
    for s in subs:
        ds = {m: np.mean([r["dice"] for r in subs[s][m]["labels"].values()])
              for m in methods if m in subs[s]}
        best = max((m for m in ds if m != "vote"), key=ds.get, default=None)
        res["per_subject_best"][s] = best
        print(f"sub-{s} " + " ".join(f"{m}={v:.3f}{'*' if m == best else ''}" for m, v in ds.items()))

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
