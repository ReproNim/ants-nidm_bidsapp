"""Are fusion volumes mis-sized, and does max_lab_plus_one cause it?

Question (2026-09-26): compare three estimators of each subcortical volume
against FreeSurfer's aseg.stats for the same subjects -- an independent
reference not built from the OASIS-TRT-20 atlases:

  off  -- argmax over real labels (what the delivered August data used)
  on   -- argmax with a background class of 1 - sum(p) (emulates
          max_lab_plus_one=True; within ~3-8% of a real flag-on run on the
          sep25 tier-2 subject)
  mass -- sum(p_L), the posterior-expected volume

Result on the 38 delivered Caltech subjects (ANTs registration), median
estimate/FreeSurfer (off / on / mass):
  too small: pallidum ~0.5/0.35/0.4, brain stem 0.84/0.44/0.49,
             cerebellar WM ~0.45/0.37/0.43
  too large: hippocampus ~1.9/1.5/1.35, caudate ~1.7/1.3/1.3,
             lateral ventricles ~4.3/3.5/3.3, 3rd ventricle 4.0/3.5/2.9
Most also sit outside the atlas min-max. White-matter-intensity structures
shrink, grey-matter/CSF structures grow, for every estimator -- the bias is in
the fused posterior, not in the flag. Results:
docs/superpowers/specs/analysis/jlf_vs_freesurfer_2026-09-26.json

Note: label_vocabulary maps 38/39 to aseg 45/6 (Cerebellum-Exterior), which
aseg.stats never reports, so cerebellar cortex is not compared here.

    set -a; . ./.env; set +a      # provides CALTECH_DERIVATIVES
    python scripts/diag_jlf_vs_freesurfer.py collect --out results.json   # ~1 min/subject
    python scripts/diag_jlf_vs_freesurfer.py report  results.json
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import os

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.antspy.label_vocabulary import NEUROMORPHOMETRICS_TO_ASEG  # noqa: E402

# Site derivatives root (holds babs-ants-nidm/ and babs-freesurfer-nidm/); set
# in the gitignored .env, e.g. CALTECH_DERIVATIVES=<path>/site-Caltech/derivatives
SITE = os.environ.get("CALTECH_DERIVATIVES", "")
RANGES = Path(__file__).resolve().parent.parent / "src/antspy/atlas_label_volume_ranges.json"
STRUCTS = {4: "3rdVent", 35: "BrainStem", 40: "R CblmWM", 41: "L CblmWM",
           47: "R Hippo", 48: "L Hippo", 55: "R Pall", 56: "L Pall",
           61: "R VDC", 62: "L VDC", 57: "R Putamen", 58: "L Putamen",
           59: "R Thal", 60: "L Thal", 36: "R Caud", 37: "L Caud",
           51: "R LatVent", 52: "L LatVent", 69: "OpticChi"}


def aseg_stats(path):
    vols = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        vols[int(t[1])] = float(t[3])
    return vols


def estimators(anat):
    files = sorted(glob.glob(f"{anat}/*_probseg.nii.gz"),
                   key=lambda f: int(re.search(r"label-(\d+)_", f)[1]))
    ids = np.array([int(re.search(r"label-(\d+)_", f)[1]) for f in files])
    P = np.stack([np.asarray(nib.load(f).dataobj, dtype=np.float32) for f in files])
    S = P.sum(0)
    m = S > 0
    a = P.argmax(0)
    off = np.where(m, ids[a], 0)
    on = np.where(m & (P.max(0) > 1 - S), ids[a], 0)
    out = {}
    for L in STRUCTS:
        i = np.where(ids == L)[0]
        if len(i):
            out[L] = dict(off=int((off == L).sum()), on=int((on == L).sum()),
                          mass=float(P[i[0]].sum()))
    return out


def collect(args):
    res = {}
    for d in sorted(glob.glob(f"{args.ants}/sub-*/anat")):
        sub = d.split("sub-")[1].split("/")[0]
        fs = Path(f"{args.freesurfer}/sub-{sub}/freesurfer/stats/aseg.stats")
        if not fs.exists():
            print(sub, "no aseg.stats", flush=True)
            continue
        fsv = aseg_stats(fs)
        r = estimators(d)
        for L in r:
            r[L]["fs"] = fsv.get(NEUROMORPHOMETRICS_TO_ASEG[L])
        res[sub] = r
        print(sub, "ok", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


def report(args):
    from scipy.stats import pearsonr
    R = json.load(open(args.results))
    rng = {int(k): v for k, v in json.load(open(RANGES))["labels"].items()}
    print(f"n={len(R)}  ratio=median(est/FS)  r=Pearson vs FS  inR=frac within atlas min-max")
    print(f"{'structure':>10} | {'off':>5} {'r':>5} {'inR':>4} | {'on':>5} {'r':>5} {'inR':>4} "
          f"| {'mass':>5} {'r':>5} {'inR':>4}")
    for L, name in STRUCTS.items():
        rows = [R[s][str(L)] for s in R if str(L) in R[s] and R[s][str(L)].get("fs")]
        if not rows:
            continue
        fs = np.array([x["fs"] for x in rows])
        lo, hi = rng[L]["min"], rng[L]["max"]
        cells = []
        for k in ("off", "on", "mass"):
            v = np.array([x[k] for x in rows], float)
            r = pearsonr(v, fs)[0] if v.std() > 0 else float("nan")
            cells.append(f"{np.median(v / fs):5.2f} {r:5.2f} {((v >= lo) & (v <= hi)).mean():4.2f}")
        print(f"{name:>10} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--ants", default=f"{SITE}/babs-ants-nidm" if SITE else None)
    c.add_argument("--freesurfer", default=f"{SITE}/babs-freesurfer-nidm" if SITE else None)
    c.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("results")
    a = ap.parse_args()
    if a.cmd == "collect" and not (a.ants and a.freesurfer):
        ap.error("set CALTECH_DERIVATIVES (see .env) or pass --ants and --freesurfer")
    collect(a) if a.cmd == "collect" else report(a)


if __name__ == "__main__":
    main()
