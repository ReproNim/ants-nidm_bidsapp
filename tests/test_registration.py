"""FireANTs backend tests on tiny CPU volumes.

Registration quality is not under test here -- that is tier 2 and tier 3 on a
GPU. What is under test is the wiring most likely to be silently wrong: label
shape handling, remapping round trips, and the geometry of what comes back.
"""
import numpy as np
import pytest

fireants = pytest.importorskip("fireants")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_fireants

from src.registration.fireants import labels_from_warp_result  # noqa: E402
from src.registration.labels import LabelMap  # noqa: E402


def _label_map(ids):
    ordered = sorted(set(ids) | {0})
    return LabelMap(
        forward={v: i for i, v in enumerate(ordered)},
        inverse=np.array(ordered, dtype=np.int32),
    )


def test_argmax_shaped_result_is_restored_to_original_ids():
    """With FFO active the warp returns [1, 1, H, W, D] remapped indices."""
    lm = _label_map([0, 4, 2035])
    tensor = torch.tensor([[[[[0, 1, 2]]]]], dtype=torch.float32)
    out = labels_from_warp_result(tensor, lm)
    np.testing.assert_array_equal(out, np.array([[[0, 4, 2035]]]))


def test_onehot_shaped_result_is_argmaxed_then_restored():
    """Without FFO, fireants/io/image.py:145-147 forces is_onehot=True and the
    warp returns [1, C, H, W, D]. Handing that to joint_label_fusion unchanged
    would be a well-formed disaster."""
    lm = _label_map([0, 4, 2035])
    onehot = torch.zeros((1, 3, 1, 1, 3), dtype=torch.float32)
    onehot[0, 0, 0, 0, 0] = 1.0   # channel 0 -> id 0
    onehot[0, 2, 0, 0, 1] = 1.0   # channel 2 -> id 2035
    onehot[0, 1, 0, 0, 2] = 1.0   # channel 1 -> id 4
    out = labels_from_warp_result(onehot, lm)
    np.testing.assert_array_equal(out, np.array([[[0, 2035, 4]]]))


def test_result_contains_only_ids_present_in_the_map():
    lm = _label_map([0, 4, 11, 2035])
    onehot = torch.rand((1, 4, 2, 2, 2))
    out = labels_from_warp_result(onehot, lm)
    assert set(np.unique(out)).issubset({0, 4, 11, 2035})


def test_folding_fraction_is_zero_for_an_identity_warp():
    """A registration of an image to itself must not fold."""
    import nibabel as nib

    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration

    from src.registration.fireants import folding_fraction

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/blob.nii.gz"
        blob = np.zeros((16, 16, 16), dtype=np.float32)
        blob[6:10, 6:10, 6:10] = 1.0
        nib.save(nib.Nifti1Image(blob, np.eye(4)), path)
        batch = BatchedImages([Image.load_file(path, device="cpu")])
        reg = AffineRegistration([2, 1], [10, 5], batch, batch,
                                 optimizer="Adam", optimizer_lr=1e-3,
                                 cc_kernel_size=3)
        reg.optimize()
        assert folding_fraction(reg, batch, batch) < 0.001


def test_unexpected_channel_count_is_rejected():
    """A one-hot whose channel count does not match the map means the label
    map and the loaded image disagree -- fail loudly rather than argmax into
    the wrong ids."""
    lm = _label_map([0, 4, 2035])
    with pytest.raises(ValueError, match="channel"):
        labels_from_warp_result(torch.zeros((1, 7, 1, 1, 1)), lm)
