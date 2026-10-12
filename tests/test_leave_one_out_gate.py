"""The leave-one-out gate: every fold must reach each group's Dice floor
against the held-out atlas's manual labels.

It replaces a gate that compared the app with a delivered derivative, which
was itself wrong. Manual labels are a ground truth; a derivative is not.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "loo_summary", ROOT / "scripts" / "diag_jlf_leave_one_out_summary.py")
loo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loo)


def fold(dice_by_label, fold_id=1):
    """A fold whose 'production' variant scores each label at the given Dice,
    with the fused volume equal to the manual one."""
    scores = {str(L): {"vox": 100, "ref_vox": 100, "dice": d} for L, d in dice_by_label.items()}
    return {"fold": fold_id, "scores": {"production": scores}}


def all_labels(d):
    ids = {L for ids in loo.GROUPS.values() if ids for L in ids} | {1002, 2002}
    return {L: d for L in ids}


def test_a_fold_at_production_accuracy_passes():
    assert loo.gate([fold(all_labels(0.9))], "production") == []


def test_a_fold_below_a_group_floor_fails_and_names_it():
    labels = all_labels(0.9)
    for L in loo.GROUPS["csf"]:
        labels[L] = 0.5
    failures = loo.gate([fold(labels, fold_id=7)], "production")
    assert any(f["fold"] == 7 and f["group"] == "csf" for f in failures)


def test_the_old_intensity_biased_fusion_fails():
    """usecor=False scored 0.31-0.38 overall Dice on every fold."""
    failures = loo.gate([fold(all_labels(0.35))], "production")
    assert {f["group"] for f in failures} >= {"all", "cortex", "wm_like"}


def test_every_group_has_a_floor():
    groups = set(loo.group_ids(fold(all_labels(0.9))["scores"]["production"]))
    assert groups <= set(loo.GATE_MIN_DICE), groups - set(loo.GATE_MIN_DICE)
