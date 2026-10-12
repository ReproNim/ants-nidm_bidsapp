"""Re-derive and validate the fusion<->quick label correspondence from data.

Background
----------
2026-09-17/18 concluded that ``--method quick`` was broken by a
"label-vocabulary mismatch", with ``quick``'s pre-fused template file blamed as
the odd one out. This script was written on 2026-09-23 to check that, and the
direction turned out to be inverted:

* ``quick`` emits standard FreeSurfer aseg subcortical ids -- which is exactly
  what ``src/ants_seg_to_nidm``'s ``FreeSurferColorLUT.txt`` expects.
* ``fusion`` emits Neuromorphometrics / BrainCOLOR ids inherited from the
  per-atlas ``OASIS-TRT-20_DKT31_CMA_labels_v2/`` files -- which that LUT
  mis-names.

So it is the primary, already-delivered method whose NIDM structure names are
wrong, not the secondary one.

What it checks
--------------
1. **Atlas side.** The id sets of the pre-fused template file and of the union
   of the 20 per-atlas files. No registration or subject data involved, so any
   difference here is purely a vocabulary difference.
2. **Subject side.** Given a ``fusion`` and a ``quick`` dseg of the *same*
   subject in the *same* space, every entry of
   ``src/antspy/label_vocabulary.NEUROMORPHOMETRICS_TO_ASEG`` is re-measured by
   reciprocal maximum overlap, and the measured Dice is compared against the
   recorded ``EVIDENCE``.
3. **Nameability.** How many ids of each vocabulary the shipped
   FreeSurferColorLUT can name, and how many it names implausibly.

Exit status is non-zero if an ``overlap`` entry no longer reproduces, so this
can gate a rebuild of the table.

Pure nibabel/numpy: no GPU, no ANTs, no container needed.

Usage
-----
    python scripts/diag_label_vocabulary_correspondence.py \\
        --data-dir data \\
        --fusion-dseg  <run>/fusion/sub-XX/anat/sub-XX_space-orig_dseg.nii.gz \\
        --quick-dseg   <run>/quick/sub-XX/anat/sub-XX_space-orig_dseg.nii.gz \\
        --out label_vocabulary_correspondence.json

``--fusion-dseg``/``--quick-dseg`` may be omitted to run the atlas-side and
nameability checks alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.antspy.label_vocabulary import (  # noqa: E402
    EVIDENCE,
    NEUROMORPHOMETRICS_TO_ASEG,
)

# Names that cannot denote a structure a brain parcellation should contain.
# Their presence in a delivered TTL is the visible tip of the mismatch.
IMPLAUSIBLE = (
    "Porg", "Aorg", "Stellate", "F3orb", "Interior", "undetermined",
    "Lesion", "Line-", "mOg", "pOg", "5th-Ventricle", "Cerebral-Exterior",
)

# An overlap entry is considered reproduced if it still lands on the recorded
# target and its Dice has not collapsed. The tolerance is loose on purpose:
# the point is to catch a changed *correspondence*, not to pin segmentation
# quality, which legitimately moves with the registration engine.
DICE_FLOOR_RATIO = 0.6


def load_lut(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path.exists():
        return names
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def ids_of(path: str | Path) -> set[int]:
    arr = np.asarray(nib.load(str(path)).dataobj).astype(int)
    return set(np.unique(arr).tolist()) - {0}


def atlas_side(data_dir: Path) -> dict:
    prefused = data_dir / "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz"
    per_atlas = sorted(glob.glob(str(data_dir / "OASIS-TRT-20_DKT31_CMA_labels_v2" / "*.nii.gz")))
    if not prefused.exists() or not per_atlas:
        return {"available": False}

    t_ids = ids_of(prefused)
    union: set[int] = set()
    for p in per_atlas:
        union |= ids_of(p)

    return {
        "available": True,
        "n_atlases": len(per_atlas),
        "prefused_ids": len(t_ids),
        "per_atlas_union_ids": len(union),
        "shared": len(t_ids & union),
        "prefused_only": sorted(t_ids - union),
        "per_atlas_only": sorted(union - t_ids),
    }


def subject_side(fusion_dseg: Path, quick_dseg: Path) -> dict:
    fi = nib.load(str(fusion_dseg))
    f = np.asarray(fi.dataobj).astype(int)
    q = np.asarray(nib.load(str(quick_dseg)).dataobj).astype(int)
    if f.shape != q.shape:
        raise SystemExit(
            f"dsegs are not in the same grid: {f.shape} vs {q.shape}. "
            "They must be the same subject from the same run."
        )

    qvol = {i: int((q == i).sum()) for i in set(np.unique(q).tolist()) - {0}}

    # Reverse best-match, used for the reciprocity test.
    best_q: dict[int, int] = {}
    for j in qvol:
        cand = f[q == j]
        cand = cand[cand != 0]
        if cand.size:
            vals, cnts = np.unique(cand, return_counts=True)
            best_q[j] = int(vals[cnts.argmax()])

    rows = []
    regressions = []
    for src, dst in sorted(NEUROMORPHOMETRICS_TO_ASEG.items()):
        kind, recorded = EVIDENCE[src]
        mask = f == src
        n = int(mask.sum())
        row = {
            "neuromorphometrics": src, "aseg": dst, "evidence": kind,
            "recorded_dice": recorded, "fusion_voxels": n,
            "quick_voxels": qvol.get(dst, 0),
        }
        if n == 0:
            row["status"] = "absent-in-this-subject"
            rows.append(row)
            continue

        cand = q[mask]
        cand = cand[cand != 0]
        if cand.size == 0:
            row["status"] = "no-overlap"
            rows.append(row)
            continue

        vals, cnts = np.unique(cand, return_counts=True)
        top = int(vals[cnts.argmax()])
        dice = 2 * int(cnts.max()) / (n + qvol.get(top, 0))
        row.update(
            measured_best_match=top,
            measured_dice=round(dice, 3),
            reciprocal=best_q.get(top) == src,
            hits_recorded_target=top == dst,
        )

        if kind == "overlap":
            ok = top == dst and recorded is not None and dice >= recorded * DICE_FLOOR_RATIO
            row["status"] = "reproduced" if ok else "REGRESSED"
            if not ok:
                regressions.append(row)
        else:
            # centroid/protocol entries are not expected to win on overlap;
            # they are reported for inspection only.
            row["status"] = "not-overlap-derived"
        rows.append(row)

    # Cortical range must be an identity mapping in both vocabularies.
    cort_same = cort_tot = 0
    for i in sorted(i for i in set(np.unique(f).tolist()) if i >= 1000):
        cand = q[f == i]
        cand = cand[cand != 0]
        if not cand.size:
            continue
        vals, cnts = np.unique(cand, return_counts=True)
        cort_tot += 1
        cort_same += int(vals[cnts.argmax()]) == i

    return {
        "available": True,
        "entries": rows,
        "regressions": [r["neuromorphometrics"] for r in regressions],
        "cortical_identity": {"matching": cort_same, "total": cort_tot},
    }


def nameability(lut: dict[int, str], data_dir: Path) -> dict:
    out: dict[str, dict] = {}
    sources = {
        "quick_prefused_template": data_dir
        / "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz",
    }
    per_atlas = sorted(glob.glob(str(data_dir / "OASIS-TRT-20_DKT31_CMA_labels_v2" / "*.nii.gz")))
    if per_atlas:
        sources["fusion_per_atlas_example"] = Path(per_atlas[0])

    for tag, path in sources.items():
        if not path.exists():
            continue
        ids = ids_of(path)
        absent = sorted(i for i in ids if i not in lut)
        implausible = sorted(
            i for i in ids
            if i in lut and any(tok.lower() in lut[i].lower() for tok in IMPLAUSIBLE)
        )
        out[tag] = {
            "n_ids": len(ids),
            "absent_from_lut": absent,
            "implausibly_named": {str(i): lut[i] for i in implausible},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--fusion-dseg", type=Path)
    ap.add_argument("--quick-dseg", type=Path)
    ap.add_argument("--lut", type=Path, default=Path(
        "src/ants_seg_to_nidm/ants_seg_to_nidm/mapping_data/FreeSurferColorLUT.txt"))
    ap.add_argument("--out", type=Path, default=Path("label_vocabulary_correspondence.json"))
    args = ap.parse_args()

    result: dict = {"atlas_side": atlas_side(args.data_dir)}

    lut = load_lut(args.lut)
    result["nameability"] = nameability(lut, args.data_dir) if lut else {"lut": "not found"}

    if bool(args.fusion_dseg) != bool(args.quick_dseg):
        ap.error("--fusion-dseg and --quick-dseg must be given together")

    if args.fusion_dseg:
        result["subject_side"] = subject_side(args.fusion_dseg, args.quick_dseg)
    else:
        result["subject_side"] = {"available": False}

    a = result["atlas_side"]
    if a.get("available"):
        print(f"atlas side: pre-fused {a['prefused_ids']} ids, "
              f"per-atlas union {a['per_atlas_union_ids']} ids, {a['shared']} shared")
        print(f"  only in pre-fused (quick): {a['prefused_only']}")
        print(f"  only in per-atlas (fusion): {a['per_atlas_only']}")

    for tag, info in result["nameability"].items():
        if not isinstance(info, dict) or "n_ids" not in info:
            continue
        print(f"\n{tag}: {info['n_ids']} ids, "
              f"{len(info['absent_from_lut'])} absent from LUT, "
              f"{len(info['implausibly_named'])} implausibly named")
        for i, name in list(info["implausibly_named"].items())[:10]:
            print(f"    {i:>5}  {name}")

    s = result["subject_side"]
    rc = 0
    if s.get("available"):
        ci = s["cortical_identity"]
        print(f"\nsubject side: cortical identity {ci['matching']}/{ci['total']}")
        print(f"{'nmm':>5} {'aseg':>5} {'kind':>10} {'rec':>5} {'meas':>5} {'recip':>6}  status")
        for r in s["entries"]:
            print(f"{r['neuromorphometrics']:>5} {r['aseg']:>5} {r['evidence']:>10} "
                  f"{str(r.get('recorded_dice','-')):>5} {str(r.get('measured_dice','-')):>5} "
                  f"{str(r.get('reciprocal','-')):>6}  {r['status']}")
        if s["regressions"]:
            print(f"\nREGRESSED entries: {s['regressions']}")
            rc = 1
        if ci["total"] and ci["matching"] != ci["total"]:
            print("cortical ids no longer map to themselves")
            rc = 1

    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
