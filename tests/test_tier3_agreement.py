"""Failure paths of the tier-3 comparison harness.

Both of these used to be silent or unhelpful: a missing FireANTs evaluation
harness NaN-filled every surface metric and still printed TIER 3 PASS, and a
run where every subject shape-mismatched died on a bare `rows[0]` IndexError
that threw away the diagnostics explaining why.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tier3():
    """Load scripts/tier3_agreement.py fresh, inside a test.

    Not at module level: tests/test_run.py replaces sys.modules['nibabel']
    with a MagicMock at collection time, and this file is collected after it.
    conftest's autouse _restore_real_nibabel has run by the time a test body
    executes, so importing here binds the real nibabel.
    """
    spec = importlib.util.spec_from_file_location(
        "tier3_agreement", ROOT / "scripts" / "tier3_agreement.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seg(path, shape):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.int16)
    data[0, 0, 0] = 1002
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _reference_tree(root, subject, shape):
    stats = root / f"sub-{subject}" / "stats" / f"sub-{subject}_antslabelstats.csv"
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text("Label,Volume_mm3\n1002,5000.0\n")
    _seg(root / f"sub-{subject}" / "anat" / f"sub-{subject}_space-orig_dseg.nii.gz", shape)


def test_all_subjects_shape_mismatching_reports_why(tier3, tmp_path, monkeypatch, capsys):
    """`rows[0]` raised a bare IndexError and discarded `failures`."""
    monkeypatch.setattr(tier3, "load_evalutils", lambda: None)

    reference = tmp_path / "reference"
    new = tmp_path / "new"
    _reference_tree(reference, "0051456", (4, 5, 6))
    _seg(new / "sub-0051456" / "anat" / "sub-0051456_space-orig_dseg.nii.gz", (4, 5, 7))

    monkeypatch.setattr(sys, "argv", [
        "tier3_agreement.py",
        "--new", str(new),
        "--reference", str(reference),
        "--subjects", "0051456",
        "--out", str(tmp_path / "report.csv"),
    ])

    with pytest.raises(SystemExit) as exc:
        tier3.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "TIER 3 FAIL" in out
    # The diagnostics that used to be thrown away with the IndexError.
    assert "shape (4, 5, 7) != reference (4, 5, 6)" in out
    assert not (tmp_path / "report.csv").exists()


def test_missing_evalutils_stops_the_run_instead_of_nan_filling(tier3, monkeypatch):
    """Surface Dice and Hausdorff are required for every gated
    label, so an absent harness invalidates the report -- it must not become a
    column of NaNs under a TIER 3 PASS banner."""
    monkeypatch.setitem(sys.modules, "fireants.scripts.evalutils", None)

    with pytest.raises(SystemExit) as exc:
        tier3.load_evalutils()

    message = str(exc.value)
    assert "fireants.scripts.evalutils" in message
    # Names the actual fix: run it inside the container.
    assert "apptainer exec" in message


def test_boundary_concentration_relaxed_labels_are_not_gated_as_primary(tier3):
    """Labels 40/41 clear PRIMARY_MIN_VOL on median volume alone but are moved
    to the relaxed tier on atlas-geometry evidence (boundary concentration on
    a single dominant neighbour; see scripts/label_boundary_analysis.py and
    the constant's docstring in tier3_agreement.py). This locks that decision
    in as tested behaviour rather than a comment that can silently drift.
    """
    stats = {
        40: {"median": 7000.0, "sd": 1500.0, "presence": 1.0},
        41: {"median": 8000.0, "sd": 1500.0, "presence": 1.0},
        4: {"median": 7000.0, "sd": 1500.0, "presence": 1.0},  # ordinary primary control
    }
    assert stats[40]["median"] >= tier3.PRIMARY_MIN_VOL
    assert stats[41]["median"] >= tier3.PRIMARY_MIN_VOL

    excluded = {l for l, s in stats.items() if s["presence"] < tier3.PRESENCE_FRACTION}
    primary = {l for l, s in stats.items()
               if l not in excluded and s["median"] >= tier3.PRIMARY_MIN_VOL
               and l not in tier3.BOUNDARY_CONCENTRATION_RELAXED}
    relaxed = {l for l, s in stats.items()
               if l not in excluded
               and (tier3.RELAXED_MIN_VOL <= s["median"] < tier3.PRIMARY_MIN_VOL
                    or l in tier3.BOUNDARY_CONCENTRATION_RELAXED)}

    assert 40 in relaxed and 40 not in primary
    assert 41 in relaxed and 41 not in primary
    assert 4 in primary and 4 not in relaxed
