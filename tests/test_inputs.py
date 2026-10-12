"""Choice of the anatomical input when a subject or session has several."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.antspy.inputs import rank_anat_inputs, select_anat_input  # noqa: E402


def _touch(anat, *names):
    anat.mkdir(parents=True, exist_ok=True)
    for n in names:
        (anat / n).write_bytes(b"")


def test_plain_name_single_session(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_T1w.nii.gz", "sub-01_T1w.json")
    assert select_anat_input(tmp_path, "01").name == "sub-01_T1w.nii.gz"


def test_run_entity_is_matched(tmp_path):
    # ABIDE II longitudinal sites name every file with run-1.
    anat = tmp_path / "sub-50002" / "ses-1" / "anat"
    _touch(anat, "sub-50002_ses-1_run-1_T1w.nii.gz")
    assert select_anat_input(tmp_path, "50002", "1").name == "sub-50002_ses-1_run-1_T1w.nii.gz"


def test_prefers_no_acq_over_acq(tmp_path):
    anat = tmp_path / "sub-51209" / "ses-1" / "anat"
    _touch(anat, "sub-51209_ses-1_acq-hires_run-1_T1w.nii.gz",
           "sub-51209_ses-1_run-1_T1w.nii.gz")
    assert select_anat_input(tmp_path, "51209", "1").name == "sub-51209_ses-1_run-1_T1w.nii.gz"


def test_lowest_run_wins_numerically(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_run-10_T1w.nii.gz", "sub-01_run-2_T1w.nii.gz")
    assert select_anat_input(tmp_path, "01").name == "sub-01_run-2_T1w.nii.gz"


def test_acq_only_files_are_still_used(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_acq-mprage_T1w.nii.gz")
    assert select_anat_input(tmp_path, "01").name == "sub-01_acq-mprage_T1w.nii.gz"


def test_other_subjects_sessions_and_suffixes_are_ignored(tmp_path):
    anat = tmp_path / "sub-01" / "ses-1" / "anat"
    _touch(anat, "sub-01_ses-1_T2w.nii.gz", "sub-010_ses-1_T1w.nii.gz",
           "sub-01_ses-10_T1w.nii.gz", "sub-01_ses-1_desc-x_T1w.json")
    assert select_anat_input(tmp_path, "01", "1") is None


def test_missing_directory_returns_none(tmp_path):
    assert select_anat_input(tmp_path, "99", "1") is None


def test_choice_and_alternatives_are_logged(tmp_path, caplog):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_acq-hires_T1w.nii.gz", "sub-01_T1w.nii.gz")
    with caplog.at_level(logging.INFO):
        select_anat_input(tmp_path, "01", logger=logging.getLogger("t"))
    assert "sub-01_T1w.nii.gz" in caplog.text
    assert "sub-01_acq-hires_T1w.nii.gz" in caplog.text


@pytest.mark.parametrize("ext", [".nii.gz", ".nii"])
def test_both_nifti_extensions(tmp_path, ext):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, f"sub-01_T1w{ext}")
    assert select_anat_input(tmp_path, "01").name == f"sub-01_T1w{ext}"


def test_rank_lists_all_candidates_best_first(tmp_path):
    anat = tmp_path / "sub-01" / "ses-1" / "anat"
    _touch(anat, "sub-01_ses-1_acq-hires_run-1_T1w.nii.gz", "sub-01_ses-1_run-1_T1w.nii.gz")
    assert [p.name for p in rank_anat_inputs(tmp_path, "01", "1")] == [
        "sub-01_ses-1_run-1_T1w.nii.gz", "sub-01_ses-1_acq-hires_run-1_T1w.nii.gz"]
    assert rank_anat_inputs(tmp_path, "02", "1") == []
