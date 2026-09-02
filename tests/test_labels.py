"""Contiguous label remapping.

DKT31 label ids are sparse (4, 11, 15, 30, 1001-1035, 2001-2035). FireANTs
one-hots with max_label+1 channels, so warping them unremapped would allocate
2036 channels instead of 97.
"""
import nibabel as nib
import numpy as np
import pytest

from src.registration.labels import build_label_map


def _write_labels(tmp_path, name, values):
    data = np.array(values, dtype=np.int16).reshape(-1, 1, 1)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_map_is_contiguous_and_starts_at_background(tmp_path):
    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 11, 2035])
    lm = build_label_map([a])
    assert lm.forward[0] == 0, "background must stay 0"
    assert sorted(lm.forward.values()) == [0, 1, 2, 3]
    assert lm.max_index == 3


def test_map_spans_the_union_across_atlases(tmp_path):
    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 11])
    b = _write_labels(tmp_path, "b.nii.gz", [0, 11, 2035])
    lm = build_label_map([a, b])
    assert set(lm.forward) == {0, 4, 11, 2035}
    assert lm.max_index == 3


# The remap half of the round trip is tested in tests/test_fireants_units.py
# against `_seg_preprocessor`, the torch implementation production actually
# uses. The numpy `LabelMap.remap_array` that used to be tested here was never
# called by anything but these tests.


def test_background_is_never_dropped(tmp_path):
    """background_seg_label=-1 keeps a channel for 0; if 0 were missing from
    the map the inverse lookup would shift every id by one."""
    a = _write_labels(tmp_path, "a.nii.gz", [4, 11])
    lm = build_label_map([a])
    assert lm.forward[0] == 0


def test_unknown_value_in_restore_is_an_error(tmp_path):
    a = _write_labels(tmp_path, "a.nii.gz", [0, 4])
    lm = build_label_map([a])
    with pytest.raises(IndexError):
        lm.restore_array(np.array([99]))
