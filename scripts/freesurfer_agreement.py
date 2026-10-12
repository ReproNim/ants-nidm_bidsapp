"""Report how the app's delivered segmentation agrees with FreeSurfer on real scans.

Report only: this prints numbers and exits 0 whatever they are. Accuracy is
gated by the leave-one-out test against the OASIS-TRT-20 manual labels
(scripts/diag_jlf_leave_one_out.py), which has a ground truth. FreeSurfer is a
different protocol, so agreement with it is capped below 1 and is not a pass
criterion; it is here because the atlases share one intensity scale and real
scans do not, and the Oct 2026 fusion bias showed only on real scans.

For each method directory under <app_out> (fusion/, quick/, or the directory
itself when it holds sub-<id>/ directly), per subject:
  - the space-orig dseg against FreeSurfer's aparc.DKTatlas+aseg, resampled
    to the dseg grid: Dice and log2(volume / FreeSurfer volume) for the 37
    mapped subcortical labels and the DKT cortical parcels, summarised by
    the groups of diag_jlf_validation_summary.py;
  - BVOL (brain mask volume) against FreeSurfer's BrainSegVol.

fusion emits Neuromorphometrics subcortical ids and quick emits aseg ids
(see src/antspy/label_vocabulary.py), so quick's map is scored on aseg ids
directly and reported under the same Neuromorphometrics keys.

For comparison, the fusion with usecor=True on 6 Caltech subjects (offline
dumps, recorded as jlf_validation_6sub_2026-10-10.json)
had group Dice WM-like 0.86, GM subcortical 0.86, CSF 0.89, cortex 0.72, and
BVOL within 0.8-4.2% of BrainSegVol.

    set -a; . ./.env; set +a
    python scripts/freesurfer_agreement.py <app_out> --subject 0051456 [...] \
        [--freesurfer <dir with sub-<id>/freesurfer/>] [--json out.json]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_jlf_validation_summary import GROUPS, KEY, group_stats, load, score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.antspy.label_vocabulary import NEUROMORPHOMETRICS_TO_ASEG  # noqa: E402


def method_dirs(app_out):
    found = {m: app_out / m for m in ("fusion", "quick") if (app_out / m).is_dir()}
    return found or {app_out.name: app_out}


def find_one(root, pattern):
    hits = sorted(root.glob(pattern))
    return hits[0] if len(hits) == 1 else None


def brain_seg_vol(fs_subject):
    for line in (fs_subject / "stats" / "aseg.stats").read_text().splitlines():
        if line.startswith("# Measure BrainSeg, BrainSegVol,"):
            return float(line.split(",")[3])
    raise ValueError(f"no BrainSegVol in {fs_subject}/stats/aseg.stats")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("app_out", type=Path)
    ap.add_argument("--subject", action="append", required=True)
    ap.add_argument("--freesurfer", type=Path, default=Path(os.environ.get(
        "CALTECH_DERIVATIVES", "")) / "babs-freesurfer-nidm")
    ap.add_argument("--json")
    a = ap.parse_args()

    res = {}
    for method, mdir in method_dirs(a.app_out).items():
        res[method] = {}
        for sid in (s.removeprefix("sub-") for s in a.subject):
            sdir = mdir / f"sub-{sid}"
            dseg = find_one(sdir, "**/anat/*_space-orig_dseg.nii.gz")
            vols = find_one(sdir, "**/stats/*_antsbrainvols.csv")
            fs = a.freesurfer / f"sub-{sid}" / "freesurfer"
            if dseg is None or not fs.is_dir():
                print(f"{method} sub-{sid}: dseg or FreeSurfer missing, skipped")
                continue
            img = nib.load(str(dseg))
            ref = np.asarray(resample_from_to(
                nib.load(str(fs / "mri" / "aparc.DKTatlas+aseg.mgz")), img, order=0
            ).dataobj).astype(np.int32)
            seg = load(dseg)
            cortex = sorted(int(L) for L in np.unique(seg) if 1000 < L < 3000)
            if method == "quick":
                # aseg ids in the map: score them as-is, key the result by the
                # Neuromorphometrics id so both methods share GROUPS and KEY.
                pairs = {nm: (aseg, aseg) for nm, aseg in NEUROMORPHOMETRICS_TO_ASEG.items()}
            else:
                pairs = {nm: (nm, aseg) for nm, aseg in NEUROMORPHOMETRICS_TO_ASEG.items()}
            pairs.update({L: (L, L) for L in cortex})
            per = {key: score(seg, ref, {own: fsid}).get(own) for key, (own, fsid) in pairs.items()}
            per = {k: v for k, v in per.items() if v}

            row = dict(groups={g: group_stats(per, ids) for g, ids in
                               {**GROUPS, "cortex": cortex}.items()},
                       key={name: per.get(L) for L, name in KEY.items()})
            fs_bv = brain_seg_vol(fs)
            if vols is not None:
                v = next(csv.DictReader(vols.open()))
                row["volumes"] = dict(BVOL=float(v["BVOL"]), LABELVOL=float(v.get("LABELVOL", "nan")),
                                      fs_BrainSegVol=fs_bv,
                                      bvol_vs_fs=float(v["BVOL"]) / fs_bv - 1)
            res[method][sid] = row

            g = row["groups"]
            print(f"{method} sub-{sid}: Dice / mean|log2 vol ratio| vs FreeSurfer")
            print("   " + "  ".join(f"{k} {g[k]['dice']:.2f}/{g[k]['abs_log2']:.2f}"
                                    for k in ("wm_like", "gm_subcort", "csf", "cortex")))
            print("   " + "  ".join(f"{n} {r['vox'] / r['fs_vox']:.2f}x" for n, r in row["key"].items()
                                    if r and n in ("BrainStem", "L LatVent", "R LatVent", "L Pall")))
            if "volumes" in row:
                vv = row["volumes"]
                print(f"   BVOL {vv['BVOL'] / 1e6:.3f} L vs FreeSurfer BrainSegVol "
                      f"{fs_bv / 1e6:.3f} L ({vv['bvol_vs_fs']:+.1%}); LABELVOL {vv['LABELVOL'] / 1e6:.3f} L")

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
