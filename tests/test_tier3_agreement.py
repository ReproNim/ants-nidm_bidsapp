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
    """Spec section 8 requires surface Dice and Hausdorff for every gated
    label, so an absent harness invalidates the report -- it must not become a
    column of NaNs under a TIER 3 PASS banner."""
    monkeypatch.setitem(sys.modules, "fireants.scripts.evalutils", None)

    with pytest.raises(SystemExit) as exc:
        tier3.load_evalutils()

    message = str(exc.value)
    assert "fireants.scripts.evalutils" in message
    # Names the actual fix: run it inside the container.
    assert "apptainer exec" in message
