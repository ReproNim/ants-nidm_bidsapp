"""Choice of the anatomical input when a subject or session has several."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.antspy.inputs import choose_anat_input, rank_anat_inputs, select_anat_input  # noqa: E402


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


def _nifti(path, zooms):
    import nibabel as nib
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), np.diag(list(zooms) + [1]))
    nib.save(img, str(path))


def test_choose_skips_images_coarser_than_the_limit(tmp_path):
    # UCLA_Long's acq-hires is 1.5x1.5x4 mm; a coarse best-ranked image gives
    # way to the next one that passes.
    anat = tmp_path / "sub-01" / "ses-1" / "anat"
    _nifti(anat / "sub-01_ses-1_run-1_T1w.nii.gz", (1.5, 1.5, 4.0))
    _nifti(anat / "sub-01_ses-1_run-2_T1w.nii.gz", (1.0, 1.0, 1.0))
    choice = choose_anat_input(tmp_path, "01", "1", max_voxel_size=2.0)
    assert choice.chosen.name == "sub-01_ses-1_run-2_T1w.nii.gz"
    assert [p.name for p, _ in choice.too_coarse] == ["sub-01_ses-1_run-1_T1w.nii.gz"]
    assert choice.too_coarse[0][1] == (1.5, 1.5, 4.0)
    assert choice.not_used == []


def test_choose_returns_none_when_every_image_is_too_coarse(tmp_path):
    anat = tmp_path / "sub-01" / "ses-1" / "anat"
    _nifti(anat / "sub-01_ses-1_acq-hires_run-1_T1w.nii.gz", (1.5, 1.5, 4.0))
    choice = choose_anat_input(tmp_path, "01", "1", max_voxel_size=2.0)
    assert choice.chosen is None
    assert len(choice.too_coarse) == 1


def test_choose_without_limit_keeps_rank_order(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_acq-hires_T1w.nii.gz", "sub-01_T1w.nii.gz")
    choice = choose_anat_input(tmp_path, "01")
    assert choice.chosen.name == "sub-01_T1w.nii.gz"
    assert [p.name for p in choice.not_used] == ["sub-01_acq-hires_T1w.nii.gz"]
    assert choice.too_coarse == []


def test_unreadable_header_ranks_after_readable_images(tmp_path):
    # A corrupt best-ranked file must not shadow a usable run-2 image.
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_T1w.nii.gz")
    _nifti(anat / "sub-01_run-2_T1w.nii.gz", (1.0, 1.0, 1.0))
    choice = choose_anat_input(tmp_path, "01", max_voxel_size=2.0)
    assert choice.chosen.name == "sub-01_run-2_T1w.nii.gz"
    assert [p.name for p in choice.not_used] == ["sub-01_T1w.nii.gz"]


def test_unreadable_header_is_still_chosen_when_alone(tmp_path):
    # With nothing else, it is chosen so that loading it fails loudly.
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat, "sub-01_T1w.nii.gz")
    choice = choose_anat_input(tmp_path, "01", max_voxel_size=2.0)
    assert choice.chosen.name == "sub-01_T1w.nii.gz"
    assert choice.too_coarse == []


def test_recorded_input_is_read_from_the_dseg_sidecar(tmp_path):
    """--skip-ants names the image the existing segmentation recorded, even
    when the current rule would now choose another one."""
    import json

    from src.antspy.inputs import recorded_anat_input

    bids = tmp_path / "bids"
    _touch(bids / "sub-01" / "ses-1" / "anat",
           "sub-01_ses-1_T1w.nii.gz", "sub-01_ses-1_acq-hires_T1w.nii.gz")
    subject_dir = tmp_path / "out" / "sub-01" / "ses-1"
    (subject_dir / "anat").mkdir(parents=True)
    (subject_dir / "anat" / "sub-01_ses-1_space-orig_dseg.json").write_text(json.dumps(
        {"Sources": ["bids:raw:sub-01/ses-1/anat/sub-01_ses-1_acq-hires_T1w.nii.gz"]}))

    choice = recorded_anat_input(subject_dir, bids)
    assert choice.chosen == bids / "sub-01/ses-1/anat/sub-01_ses-1_acq-hires_T1w.nii.gz"
    assert choice.not_used == [] and choice.too_coarse == []


@pytest.mark.parametrize("content", [None, "{}", '{"Sources": ["bids:other:x"]}', "not json"])
def test_recorded_input_is_none_without_a_usable_sources_entry(tmp_path, content):
    from src.antspy.inputs import recorded_anat_input

    subject_dir = tmp_path / "sub-01"
    (subject_dir / "anat").mkdir(parents=True)
    if content is not None:
        (subject_dir / "anat" / "sub-01_space-orig_dseg.json").write_text(content)
    assert recorded_anat_input(subject_dir, tmp_path) is None


def test_bids_relative_without_a_bids_dir_gives_the_name():
    from src.antspy.inputs import bids_relative

    assert bids_relative("/data/sub-01/anat/sub-01_T1w.nii.gz", None) == "sub-01_T1w.nii.gz"
