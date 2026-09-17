#!/usr/bin/env python3
"""Characterise why some labels are structurally hard for joint label fusion,
using properties of the OASIS-TRT-20 atlas geometry alone -- computed with no
knowledge of any pipeline's Dice, so it cannot be reverse-engineered from a
particular run's failures.

Motivation (2026-09-14/17): after the tier-3 tuning campaign in
docs/superpowers/specs/2026-09-01-fireants-engine-design.md moved primary-label
median Dice from 0.81 (SyN) to 0.85 (Greedy), four labels still missed the
per-label floor on every one of the three Caltech subjects tested, under both
registration algorithms: 40, 41 (primary tier) and 49, 50 (already relaxed
tier). Two atlas-geometry hypotheses were tested, run against all 20
OASIS-TRT-20 DKT31+CMA atlas label volumes:

  1. NEIGHBOUR DOMINANCE: for each label, the volume of its single largest
     touching neighbour, relative to its own volume (median across the 20
     atlases). Correlates with tier-3 Dice across all 93 gated labels
     (Pearson r=-0.63, Spearman rho=-0.55, p<0.0001) -- a real, general
     effect. But labels 40/41 rank only 46th/47th of 93 on this metric: many
     labels with a FAR more dominant neighbour (ratio 18-30 vs 40/41's ~3.5)
     still score 0.80-0.88 Dice. This does NOT single out 40/41 -- reported
     here as a negative result, not silently dropped.

  2. BOUNDARY CONCENTRATION: for each label, the fraction of its total
     touching-boundary voxels that belong to a single dominant neighbour,
     rather than being spread across several neighbours (median across the
     20 atlases). Weaker overall correlation (Pearson r=-0.26, Spearman
     rho=-0.35, p=0.0006) but DOES single out exactly the four persistently
     failing labels: 40, 41, 50, 49 rank 5th, 6th, 7th, 9th of 93 by this
     metric (concentration 0.74-0.81), tightly clustered at the top. The
     mechanism: most structures share their boundary across several
     neighbours, so per-neighbour JLF voting error partially cancels; these
     four have 74-81% of their entire boundary contested by ONE competitor,
     so voting error there compounds rather than averaging out. Not a
     perfect rule (labels 57/58 also concentrate >0.83 but score >=0.90) --
     reported with that caveat, not oversold.

Both metrics are computed from atlas anatomy only and are independent of any
registration algorithm's output, so they cannot have been tuned to fit the
2026-09-14 tier-3 result -- see [[fireants-registration-root-causes-sep7]].

Usage:
  scripts/label_boundary_analysis.py --atlas-dir /opt/data/OASIS-TRT-20_DKT31_CMA_labels_v2 \
      --tier3-report /path/to/tier3_report.csv --out results.json

Runs on CPU with nibabel + numpy + scipy only; no GPU, no FireANTs, no ANTsPy.
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


def compute_atlas_metrics(atlas_dir):
    """Per-atlas neighbour-dominance ratio and boundary-concentration ratio,
    aggregated across atlases as the median (min 10 atlases must carry the
    label id for it to be reported).
    """
    atlas_files = sorted(glob.glob(str(Path(atlas_dir) / "*.nii.gz")))
    if not atlas_files:
        raise SystemExit(f"no .nii.gz files under {atlas_dir}")

    dominance_per_atlas, concentration_per_atlas = [], []
    for f in atlas_files:
        a = np.asarray(nib.load(f).dataobj).astype(np.int32)
        ids, counts = np.unique(a, return_counts=True)
        vol = dict(zip(ids.tolist(), counts.tolist()))
        vol.pop(0, None)

        # One-voxel shift along each axis gives every touching label pair in
        # a single vectorised pass, rather than one binary_dilation per label
        # (the latter is O(n_labels) dilations over the full volume and does
        # not finish in a reasonable wall-clock time for ~100 labels).
        touch = defaultdict(lambda: defaultdict(int))
        for axis in range(3):
            s1, s2 = [slice(None)] * 3, [slice(None)] * 3
            s1[axis], s2[axis] = slice(1, None), slice(None, -1)
            A, B = a[tuple(s1)], a[tuple(s2)]
            diff = (A != B) & (A != 0) & (B != 0)
            pa, pb = A[diff], B[diff]
            pairs, cnts = np.unique(np.stack([pa, pb], axis=1), axis=0, return_counts=True)
            for (l1, l2), c in zip(pairs.tolist(), cnts.tolist()):
                touch[l1][l2] += c
                touch[l2][l1] += c

        dominance, concentration = {}, {}
        for i, neighbours in touch.items():
            if vol.get(i, 0) < 50:
                continue
            total_border = sum(neighbours.values())
            biggest_neighbour_vol = max(vol.get(n, 0) for n in neighbours)
            dominance[i] = biggest_neighbour_vol / vol[i]
            concentration[i] = max(neighbours.values()) / total_border
        dominance_per_atlas.append(dominance)
        concentration_per_atlas.append(concentration)

    def aggregate(per_atlas):
        all_ids = sorted(set().union(*[set(d) for d in per_atlas]))
        return {
            i: float(np.median([d[i] for d in per_atlas if i in d]))
            for i in all_ids
            if sum(i in d for d in per_atlas) >= 10
        }

    return aggregate(dominance_per_atlas), aggregate(concentration_per_atlas), len(atlas_files)


def load_median_dice(tier3_report_csv):
    rows = list(csv.DictReader(open(tier3_report_csv)))
    by_label = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(float(r["dice"]))
    return {label: float(np.median(v)) for label, v in by_label.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-dir", required=True,
                     help="OASIS-TRT-20_DKT31_CMA_labels_v2 directory")
    ap.add_argument("--tier3-report", help="tier3_report.csv, to correlate against Dice")
    ap.add_argument("--out", default="label_boundary_analysis.json")
    args = ap.parse_args()

    dominance, concentration, n_atlases = compute_atlas_metrics(args.atlas_dir)
    print(f"computed from {n_atlases} atlas label volumes")

    result = {"n_atlases": n_atlases, "dominance_ratio": dominance, "boundary_concentration": concentration}

    if args.tier3_report:
        from scipy.stats import spearmanr

        median_dice = load_median_dice(args.tier3_report)
        result["median_dice"] = median_dice
        for name, metric in [("dominance_ratio", dominance), ("boundary_concentration", concentration)]:
            common = sorted(set(metric) & set(median_dice))
            x = np.array([metric[l] for l in common])
            y = np.array([median_dice[l] for l in common])
            pearson = float(np.corrcoef(x, y)[0, 1])
            rho, p = spearmanr(x, y)
            ranked = sorted(common, key=lambda l: -metric[l])
            rank_of = {l: i + 1 for i, l in enumerate(ranked)}
            print(f"\n{name}: n={len(common)} Pearson r={pearson:.3f} Spearman rho={rho:.3f} p={p:.4g}")
            for lab in [40, 41, 49, 50]:
                if lab in metric:
                    print(f"  label {lab}: value={metric[lab]:.3f} rank={rank_of[lab]}/{len(common)} "
                          f"median_dice={median_dice.get(lab, float('nan')):.3f}")
            result[f"{name}_correlation"] = {
                "n": len(common), "pearson_r": pearson, "spearman_rho": float(rho), "p": float(p),
                "rank": {str(l): rank_of[l] for l in common},
            }

    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
