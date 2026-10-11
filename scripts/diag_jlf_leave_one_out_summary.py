"""Summarise the leave-one-out folds written by diag_jlf_leave_one_out.py.

Reads <loo_dir>/atlas-NN/scores.json for every fold present and reports, for
the production fusion and the old usecor=False fusion:
  - mean Dice with the manual labels per structure group and overall,
  - mean |log2(volume / manual volume)| per group (size error) and its signed
    mean (direction),
  - the per-fold intensity medians, which say whether the folds reproduce the
    target/atlas scale mismatch seen on real scans.

With --gate it is the accuracy gate for a release: every fold of the
production fusion must reach GATE_MIN_DICE in every structure group, and the
script exits 1 otherwise.

    python scripts/diag_jlf_leave_one_out_summary.py <loo_dir> [--json out.json] [--gate]

Result on 20 folds (jobs 25557911/12/14, Oct 11; N4 put each held-out target
at an in-mask median of 3000-5600 against ~1130 for the warped atlases, so the
folds do reproduce the scale mismatch): production overall Dice 0.78-0.82 per
fold, old usecor=False fusion 0.31-0.38. Mean Dice / worst fold by group,
production: WM-like 0.889/0.877, GM subcortical 0.896/0.852, CSF 0.846/0.785,
all subcortical 0.776/0.734, cortex 0.823/0.810, all 0.805/0.782.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_jlf_validation_summary import GROUPS, KEY  # noqa: E402


#: Per-fold Dice floors for --gate: 0.03 below the worst production fold of
#: the Oct 11 run in each group, rounded down, so GPU nondeterminism does not
#: trip it while a regression of the size of the usecor=False bias (overall
#: Dice 0.31-0.38) fails every group.
GATE_MIN_DICE = {"wm_like": 0.84, "gm_subcort": 0.82, "csf": 0.75,
                 "subcort_all": 0.70, "cortex": 0.78, "all": 0.75}


def group_ids(scores):
    ids = {int(k) for k in scores}
    groups = {g: [L for L in v if L in ids] for g, v in GROUPS.items() if v is not None}
    groups["cortex"] = sorted(L for L in ids if 1000 < L < 3000)
    groups["all"] = sorted(ids)
    return groups


def stats(scores, ids):
    rows = [scores[str(L)] for L in ids if str(L) in scores]
    lr = np.clip([np.log2(r["vox"] / r["ref_vox"]) if r["vox"] else -5 for r in rows], -5, 5)
    return dict(dice=float(np.mean([r["dice"] for r in rows])),
                abs_log2=float(np.mean(np.abs(lr))), signed_log2=float(np.mean(lr)), n=len(rows))


def gate(folds, variant="production"):
    """One failure dict per (fold, group) below GATE_MIN_DICE; [] passes."""
    failures = []
    for f in folds:
        scores = f["scores"][variant]
        for g, ids in group_ids(scores).items():
            if not ids:
                continue
            d = stats(scores, ids)["dice"]
            if d < GATE_MIN_DICE[g]:
                failures.append(dict(fold=f["fold"], group=g, dice=d, floor=GATE_MIN_DICE[g]))
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("loo_dir")
    ap.add_argument("--json")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 unless every production fold meets GATE_MIN_DICE")
    a = ap.parse_args()

    folds = [json.loads(p.read_text()) for p in sorted(Path(a.loo_dir).glob("atlas-*/scores.json"))]
    if not folds:
        sys.exit(f"no folds under {a.loo_dir}")
    variants = list(folds[0]["scores"])
    res = {"n_folds": len(folds), "folds": [f["fold"] for f in folds], "summary": {}, "per_fold": {}}

    print(f"{len(folds)} folds: {res['folds']}")
    print("\nintensity medians in the mask (target after N4 / warped atlases):")
    for f in folds:
        print(f"  fold {f['fold']:2d}: target {f['target_median']:7.1f}  atlases "
              f"{np.median(f['atlas_medians']):7.1f}  ratio "
              f"{np.median(f['atlas_medians']) / f['target_median']:.2f}")

    groups = group_ids(folds[0]["scores"][variants[0]])
    print("\nmean over folds: Dice / mean|log2 vol ratio| / signed log2")
    print(f"{'variant':>11} " + " ".join(f"{g:>20}" for g in groups))
    for v in variants:
        res["summary"][v] = {}
        cells = []
        for g, ids in groups.items():
            per = [stats(f["scores"][v], ids) for f in folds]
            agg = {k: float(np.mean([p[k] for p in per])) for k in ("dice", "abs_log2", "signed_log2")}
            agg["fold_min_dice"] = float(min(p["dice"] for p in per))
            res["summary"][v][g] = agg
            cells.append(f"{agg['dice']:.3f}/{agg['abs_log2']:.2f}/{agg['signed_log2']:+.2f}")
        print(f"{v:>11} " + " ".join(f"{c:>20}" for c in cells))

    print("\nkey structures, median over folds: vol/manual (Dice)")
    print(f"{'structure':>10} " + " ".join(f"{v:>16}" for v in variants))
    res["key_structures"] = {}
    for L, name in KEY.items():
        cells, res["key_structures"][name] = [], {}
        for v in variants:
            rows = [f["scores"][v].get(str(L)) for f in folds]
            rows = [r for r in rows if r and r["ref_vox"]]
            ratio = float(np.median([r["vox"] / r["ref_vox"] for r in rows]))
            d = float(np.median([r["dice"] for r in rows]))
            res["key_structures"][name][v] = dict(ratio=ratio, dice=d, n=len(rows))
            cells.append(f"{ratio:8.2f} ({d:.2f})")
        print(f"{name:>10} " + " ".join(cells))

    for f in folds:
        res["per_fold"][f["fold"]] = {v: stats(f["scores"][v], groups["all"])["dice"] for v in variants}
    if a.gate:
        failures = gate(folds)
        res["gate"] = dict(floors=GATE_MIN_DICE, failures=failures, passed=not failures)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    if a.gate:
        for x in failures:
            print(f"GATE FAIL fold {x['fold']} {x['group']}: Dice {x['dice']:.3f} < {x['floor']}")
        print(f"LEAVE-ONE-OUT GATE {'FAIL' if failures else 'PASS'} ({len(folds)} folds)")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
