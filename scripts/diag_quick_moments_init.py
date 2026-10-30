"""Compare --method quick before and after the centre-of-mass init change (#16).

Reads the outputs of scripts/diag_quick_moments_init.sbatch:

    <runs>/{old,new,old_rep}/sub-<id>/anat/sub-<id>_space-orig_dseg.nii.gz

and a fusion dseg per subject (same subject, same original grid). For each
subject it reports

  * per-label Dice of old vs new, and of old vs old_rep (run-to-run noise
    floor: the same code twice);
  * mean Dice of each quick variant against fusion, separately for cortical
    labels (DKT ids >= 1000, identical in both vocabularies) and subcortical
    labels (fusion's Neuromorphometrics ids renumbered to aseg through
    src/antspy/label_vocabulary.py; fusion ids outside that bijection are
    dropped rather than compared by number);
  * the folding fraction each run logged, when a log is given;
  * the distance between the old and new centre-of-mass init translations,
    from scripts/diag_quick_moments_init_com.sbatch (--com-dir).

Usage:
    python scripts/diag_quick_moments_init.py --runs <runs> --fusion-dir <dir> \
        [--com-dir <dir>] [--logs <glob>] \
        --out <csv_prefix> 0051456 0051457 ...
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antspy.label_vocabulary import (  # noqa: E402
    CORTICAL_ID_FLOOR,
    NEUROMORPHOMETRICS_TO_ASEG,
)

VARIANTS = ("old", "new", "old_rep")


def load_labels(path):
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj).astype(np.int64)


def dice(a, b, label):
    x, y = a == label, b == label
    denom = x.sum() + y.sum()
    return float("nan") if denom == 0 else 2.0 * np.logical_and(x, y).sum() / denom


def fusion_in_aseg(fusion):
    """Cortical ids copied, subcortical renumbered via the bijection, rest 0."""
    out = np.zeros_like(fusion)
    cortical = fusion >= CORTICAL_ID_FLOOR
    out[cortical] = fusion[cortical]
    for nmm, aseg in NEUROMORPHOMETRICS_TO_ASEG.items():
        out[fusion == nmm] = aseg
    return out


def folding_from_log(log_paths, sub):
    """Map variant -> folding percentage logged for the template warp."""
    found = {}
    for path in log_paths:
        text = Path(path).read_text(errors="replace")
        if f"sub-{sub} variant=" not in text:
            continue
        for variant in VARIANTS:
            # A requeued (preempted) task can leave an earlier, unfinished
            # attempt in the same log; only the last completed block counts.
            blocks = re.findall(
                rf"=== sub-{sub} variant={variant} start(?:(?!=== sub-{sub} variant={variant} start).)*?"
                rf"=== sub-{sub} variant={variant} done",
                text, re.S,
            )
            if blocks:
                m = re.search(r"Non-positive Jacobian determinants, the template: ([0-9.]+)%",
                              blocks[-1])
                found[variant] = float(m.group(1)) if m else float("nan")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--fusion-dir", required=True, type=Path)
    ap.add_argument("--com-dir", default=None, type=Path,
                    help="sub-<id>.json files from diag_quick_moments_init_com.sbatch")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()

    log_paths = sorted(glob.glob(args.logs)) if args.logs else []
    summary_rows, label_rows = [], []

    for sub in args.subjects:
        seg = {}
        for v in VARIANTS:
            img, seg[v] = load_labels(
                args.runs / v / f"sub-{sub}" / "anat" / f"sub-{sub}_space-orig_dseg.nii.gz")
        # The sbatch runs old, new, old_rep in that order and a requeued task
        # reruns all three, so out-of-order mtimes mean a dseg survived from an
        # earlier (preempted) attempt.
        mtimes = [(args.runs / v / f"sub-{sub}" / "anat" / f"sub-{sub}_space-orig_dseg.nii.gz")
                  .stat().st_mtime for v in VARIANTS]
        if mtimes != sorted(mtimes):
            raise SystemExit(f"sub-{sub}: dseg mtimes out of run order {mtimes}")
        fimg, fusion = load_labels(args.fusion_dir / f"sub-{sub}_space-orig_dseg.nii.gz")
        if fusion.shape != seg["old"].shape or not np.allclose(fimg.affine, img.affine, atol=1e-3):
            raise SystemExit(f"sub-{sub}: fusion dseg is not on the quick dseg grid")
        fusion_aseg = fusion_in_aseg(fusion)

        quick_labels = sorted((set(np.unique(seg["old"])) | set(np.unique(seg["new"]))) - {0})
        for lab in quick_labels:
            label_rows.append({
                "subject": sub, "label": int(lab),
                "dice_old_new": dice(seg["old"], seg["new"], lab),
                "dice_old_oldrep": dice(seg["old"], seg["old_rep"], lab),
            })

        row = {"subject": sub}
        for pair, (a, b) in {"old_new": ("old", "new"), "old_oldrep": ("old", "old_rep")}.items():
            d = [dice(seg[a], seg[b], lab) for lab in quick_labels]
            row[f"mean_dice_{pair}"] = float(np.nanmean(d))
            row[f"min_dice_{pair}"] = float(np.nanmin(d))
            row[f"voxels_changed_{pair}"] = int((seg[a] != seg[b]).sum())

        fusion_labels = set(np.unique(fusion_aseg)) - {0}
        cortical = sorted(l for l in fusion_labels if l >= CORTICAL_ID_FLOOR)
        subcortical = sorted(l for l in fusion_labels if l < CORTICAL_ID_FLOOR)
        for v in VARIANTS:
            row[f"cort_dice_vs_fusion_{v}"] = float(np.nanmean(
                [dice(seg[v], fusion_aseg, l) for l in cortical]))
            row[f"subc_dice_vs_fusion_{v}"] = float(np.nanmean(
                [dice(seg[v], fusion_aseg, l) for l in subcortical]))
        row["n_cortical"], row["n_subcortical"] = len(cortical), len(subcortical)

        com = args.com_dir / f"sub-{sub}.json" if args.com_dir else None
        if com and com.exists():
            d = json.loads(com.read_text())
            row["init_shift_mm"] = d["translation_diff_norm"]

        for v, pct in folding_from_log(log_paths, sub).items():
            row[f"folding_pct_{v}"] = pct
        summary_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary_rows), ("labels", label_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(f"{args.out}_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    cols = ["subject", "mean_dice_old_new", "mean_dice_old_oldrep",
            "cort_dice_vs_fusion_old", "cort_dice_vs_fusion_new",
            "subc_dice_vs_fusion_old", "subc_dice_vs_fusion_new", "init_shift_mm",
            "folding_pct_old", "folding_pct_new"]
    print("\t".join(cols))
    for r in summary_rows:
        print("\t".join(f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                        for c in cols))


if __name__ == "__main__":
    main()
