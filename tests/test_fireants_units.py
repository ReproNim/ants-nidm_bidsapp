"""Unit tests for src/registration/fireants.py helpers.

Separate from tests/test_registration.py because that module is skipped whole
when `fireants` is not importable (CI installs neither fireants nor torch via
`pip install -e .`). Everything here needs at most torch, and the pure
functions need nothing, so the coverage survives a minimal environment.
"""
import numpy as np
import pytest

from src.registration.fireants import (
    _chunks,
    _force_onehot,
    _seg_preprocessor,
    folding_fractions_from_det,
)
from src.registration.labels import build_label_map

import nibabel as nib


def _write_labels(tmp_path, name, values):
    data = np.array(values, dtype=np.int16).reshape(-1, 1, 1)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


# --------------------------------------------------------------------------
# _chunks -- batch_size chunking must return the same number of atlases in the
# same order as a single full batch.
# --------------------------------------------------------------------------

def test_chunks_of_zero_or_negative_is_one_full_batch():
    """batch_size=0 means "all at once". It was the production default until
    tier 2 (2026-09-07) showed 20 atlases at once OOM an 80 GB A100."""
    items = list(range(5))
    assert _chunks(items, 0) == [items]
    assert _chunks(items, -1) == [items]
    assert _chunks(items, None) == [items]


def test_chunk_size_at_or_above_the_input_is_one_full_batch():
    items = list(range(5))
    assert _chunks(items, 5) == [items]
    assert _chunks(items, 99) == [items]


def test_uneven_final_chunk_is_kept_not_dropped():
    """20 atlases at batch_size=6 is 6+6+6+2. Dropping or padding the last
    chunk silently changes how many atlases reach joint_label_fusion."""
    assert _chunks(list(range(20)), 6) == [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19],
    ]


def test_chunking_preserves_count_and_order():
    items = [f"atlas{i}" for i in range(20)]
    for size in (0, 1, 3, 6, 7, 19, 20, 21):
        flat = [x for chunk in _chunks(items, size) for x in chunk]
        assert flat == items, f"batch_size={size} changed count or order"


def test_chunks_returns_copies_not_views_of_the_input():
    items = [1, 2, 3]
    out = _chunks(items, 0)
    out[0].append(4)
    assert items == [1, 2, 3]


# --------------------------------------------------------------------------
# _force_onehot -- finding: `--device cpu --label-interpolation genericlabel`
# crashes in the shipped container unless one-hot is requested explicitly.
# --------------------------------------------------------------------------

def test_cpu_with_fused_ops_built_must_request_onehot_explicitly():
    """The shipped container has fused ops. `Image.__init__` only forces
    is_onehot=True when the module-level `fi.use_ffo` is false, which it is
    not there -- so on CPU the image keeps interpolation_mode='genericlabel'
    and GridSampleDispatcher.__call__ raises
    RuntimeError("genericlabel interpolation requires fused ops") once
    augment_registry_backend has flipped _use_ffo off for the CPU tensors.
    """
    assert _force_onehot("cpu", True) is True


def test_cuda_with_fused_ops_uses_the_fused_generic_label_sampler():
    """The production default: the one combination where genericlabel works."""
    assert _force_onehot("cuda", True) is False


def test_missing_fused_ops_is_onehot_on_either_device():
    """FireANTs forces this itself; saying so explicitly makes the degraded
    path deliberate rather than incidental."""
    assert _force_onehot("cuda", False) is True
    assert _force_onehot("cpu", False) is True


# --------------------------------------------------------------------------
# folding_fractions_from_det -- the folding gate applies to every atlas
# registration, not to the batch mean.
# --------------------------------------------------------------------------

def test_folding_is_reported_per_batch_element_not_as_one_mean():
    """A single atlas folding on 2% of its voxels must be reported as 2%.

    Flattening [N, ...] to -1 and taking one mean turns that into a 0.02/20 =
    0.1% batch mean at batch_size=0 (all twenty), which sits on the gate rather
    than tripping it.
    """
    torch = pytest.importorskip("torch")

    n, voxels = 20, 1000
    det = torch.ones((n, voxels), dtype=torch.float32)
    det[7, :20] = -1.0  # 2% of atlas 7 folds

    fractions = folding_fractions_from_det(det)

    assert len(fractions) == n
    assert fractions[7] == pytest.approx(0.02)
    assert all(f == 0.0 for i, f in enumerate(fractions) if i != 7)
    # The bug this pins: the batch mean is 20x smaller than the real value.
    assert sum(fractions) / n == pytest.approx(0.001)


def test_folding_reduces_over_spatial_axes_of_a_real_shaped_tensor():
    torch = pytest.importorskip("torch")

    det = torch.ones((2, 4, 4, 4), dtype=torch.float32)
    det[1] = -1.0
    assert folding_fractions_from_det(det) == [0.0, 1.0]


def test_zero_determinants_count_as_folding():
    """det == 0 is a collapsed voxel, not a valid one."""
    torch = pytest.importorskip("torch")

    det = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    assert folding_fractions_from_det(det) == [0.5]


# --------------------------------------------------------------------------
# _seg_preprocessor -- the remap production actually runs. FireANTs applies it
# to the torch tensor at fireants/io/image.py:151, before max_seg_label
# clamping and before one-hotting.
# --------------------------------------------------------------------------

def test_seg_preprocessor_round_trips_through_restore_array(tmp_path):
    """The mechanism the spec calls out: sparse DKT31 ids -> contiguous
    indices -> back to the original ids after the warp."""
    torch = pytest.importorskip("torch")

    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 11, 30, 1001, 2035])
    lm = build_label_map([a])

    original = np.array([[0, 4], [2035, 1001]], dtype=np.int32)
    remapped = _seg_preprocessor(lm)(torch.tensor(original, dtype=torch.float32))

    np.testing.assert_array_equal(
        lm.restore_array(remapped.numpy().round().astype(np.int64)), original
    )


def test_seg_preprocessor_output_is_dense_enough_for_onehot(tmp_path):
    """The whole point: max index must be small, not 2035.

    integer_to_onehot allocates max_label + 1 channels, so an unremapped 2035
    would ask for 2036 channels (~136 GB at 256^3).
    """
    torch = pytest.importorskip("torch")

    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 2035])
    lm = build_label_map([a])
    out = _seg_preprocessor(lm)(torch.tensor([2035.0]))
    assert out.max().item() == 2
    assert lm.max_index == 2


def test_seg_preprocessor_leaves_ids_absent_from_the_map_at_zero(tmp_path):
    """`out = torch.zeros_like(array)` means an id the map does not know
    becomes background rather than an out-of-range index that would blow up
    restore_array. Pinned so the behaviour is a decision, not an accident."""
    torch = pytest.importorskip("torch")

    a = _write_labels(tmp_path, "a.nii.gz", [0, 4])
    lm = build_label_map([a])
    out = _seg_preprocessor(lm)(torch.tensor([4.0, 999.0]))
    assert out.tolist() == [1.0, 0.0]


# --------------------------------------------------------------------------
# Centre-of-mass init inputs. MomentsRegistration sums over channels
# (fireants 1.5.0 registration/moments.py, optimize_helper), so the masked-loss
# batches -- intensity plus a trailing brain-mask channel -- would weight the
# centre of mass by intensity + mask. Both registrations must give moments the
# plain single-channel images. Image.concatenate (inside _with_mask_channel)
# mutates the Image in place while BatchedImages snapshots its tensor when
# built, so the plain batches must exist before the mask channel is added; the
# fakes reproduce both. Runs without torch or fireants.
# --------------------------------------------------------------------------

def test_template_registration_gives_moments_the_plain_images(tmp_path, monkeypatch):
    import sys
    import types

    from src.registration import fireants as mod

    class FakeImage:
        def __init__(self, path):
            self.path = path
            self.itk_image = object()
            self.channels = 1

        @classmethod
        def load_file(cls, path, device=None):
            return cls(path)

    class FakeBatch:
        def __init__(self, images):
            self.images = list(images)
            # fireants' BatchedImages snapshots the tensor at construction.
            self.channels = [i.channels for i in self.images]

    class FakeAffine:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self):
            pass

    fake_io = types.ModuleType("fireants.io")
    fake_io.Image, fake_io.BatchedImages = FakeImage, FakeBatch
    fake_affine = types.ModuleType("fireants.registration.affine")
    fake_affine.AffineRegistration = FakeAffine
    monkeypatch.setitem(sys.modules, "fireants", types.ModuleType("fireants"))
    monkeypatch.setitem(sys.modules, "fireants.io", fake_io)
    monkeypatch.setitem(
        sys.modules, "fireants.registration", types.ModuleType("fireants.registration")
    )
    monkeypatch.setitem(sys.modules, "fireants.registration.affine", fake_affine)

    seen = {}

    def record_moments(fixed_batch, moving_batch):
        seen["fixed"], seen["moving"] = fixed_batch, moving_batch
        return None

    monkeypatch.setattr(mod, "LOSS_MASKED", True)
    monkeypatch.setattr(mod, "_moments_init", record_moments)
    def add_mask_channel(img, *args):
        # Like Image.concatenate: mutates and returns the same Image.
        img.channels += 1
        return img

    monkeypatch.setattr(mod, "_with_mask_channel", add_mask_channel)
    monkeypatch.setattr(mod, "geometry_of", lambda img: None)
    monkeypatch.setattr(mod, "write_nifti", lambda img, path: path)
    monkeypatch.setattr(mod, "_assert_inbound", lambda *a: None)
    monkeypatch.setattr(mod, "_deformable_registration", lambda *a: FakeAffine())
    monkeypatch.setattr(mod, "folding_fraction", lambda *a: [0.0])
    monkeypatch.setattr(mod, "_refuse_folding", lambda *a: None)

    mod.register_to_template(object(), object(), temp_dir=tmp_path, device="cpu")

    assert seen["fixed"].channels == [1]
    assert seen["moving"].channels == [1]
    assert seen["fixed"].images[0].path.endswith("tpl_subject.nii.gz")
    assert seen["moving"].images[0].path.endswith("tpl_template.nii.gz")
