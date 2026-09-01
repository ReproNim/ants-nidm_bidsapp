# FireANTs Registration Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every ANTs registration and transform-application call in the BIDS app with GPU-accelerated FireANTs, keeping ANTs for N4, brain extraction, and joint-label-fusion voting.

**Architecture:** A new `src/registration/` package owns all registration. Images cross between ANTsPy and FireANTs exclusively as NIfTI files on `TMPDIR` — no in-memory array conversion — guarded by an explicit geometry check on every crossing. `wrapper.py`'s twenty-atlas registration loop collapses to one `warp_atlases()` call; the subject↔template registration becomes a `TemplateWarp` object that never leaves memory and never serialises to ANTs transform format.

**Tech Stack:** Python 3.10, FireANTs 1.5.0, PyTorch ≥2.3.0 (CUDA in container, CPU in CI), ANTsPy (`antspyx`), ANTs 2.5.4 binaries, nibabel, pytest, Apptainer/Docker, BABS/SLURM.

**Spec:** [`docs/superpowers/specs/2026-09-01-fireants-engine-design.md`](../specs/2026-09-01-fireants-engine-design.md)

## Global Constraints

- **Branch:** `fireants-engine`, already created off `main`.
- **FireANTs version:** `1.5.0`. **PyTorch:** `>=2.3.0`. **ANTs binaries:** `2.5.4` (unchanged).
- **Never pass `orientation`, `spacing`, `direction`, `origin`, or `center` to `fireants.io.Image`.** `orientation` applies `sitk.DICOMOrient` and rewrites the header; the others override geometry read from the file. All must be left unset.
- **No in-memory ANTsImage↔FireANTs conversion.** Every crossing is a NIfTI file under `self.temp_dir`. Anything that pulls `.numpy()` off an ANTsImage and rebuilds a header on the other side is a plan violation.
- **`assert_geometry_match` runs in production, not only under pytest.**
- **ANTs stays for:** `n4_bias_field_correction`, `antsBrainExtraction.sh`, `joint_label_fusion`, `image_read`/`image_write`. Do not touch these.
- **Label warping returns two possible shapes.** With FFO active, `[1, 1, H, W, D]` labels. Without it (no `fused_ops`, or any CPU tensor), `fireants/io/image.py:145-147` forces `is_onehot=True` and the result is `[1, C, H, W, D]` one-hot. `warp_atlases` must detect and argmax. Never hand a one-hot volume to `ants.joint_label_fusion`.
- **Registration hyperparameters** come from FireANTs' own `scripts/pairwise/oasis.py`, not from invention: affine `scales=[4,2,1], iterations=[200,100,50]`; SyN `scales=[4,2,1], iterations=[200,100,50], deformation_type='compositive', optimizer='Adam'`.
- **Tests that need FireANTs installed** are marked `@pytest.mark.requires_fireants` and skip cleanly when the import fails, so the existing mocked suite keeps running everywhere.
- **Existing mocked test suite must keep passing.** `tests/test_wrapper.py` stubs `sys.modules['ants']` at import time; do not break that.
- Commit after every task. Run `pytest -m "not ci_incompatible" tests/` before each commit.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/registration/__init__.py` | Public API (`warp_atlases`, `register_to_template`, `apply_warp`, `TemplateWarp`); lazily imports the FireANTs backend so CI can import the package without torch |
| `src/registration/handoff.py` | The ANTs↔FireANTs file boundary: `Geometry`, `geometry_of`, `assert_geometry_match`, `write_nifti`, `read_nifti`. No torch, no fireants — importable anywhere |
| `src/registration/device.py` | `resolve_device`, `DeviceUnavailable`. Injectable CUDA probe so it tests without torch |
| `src/registration/fireants.py` | All FireANTs calls. The only module that imports `fireants` or `torch` |
| `tests/test_handoff.py` | Geometry guard + NIfTI round trip (nibabel only, always runs in CI) |
| `tests/test_device.py` | Device resolution (no torch needed) |
| `tests/test_registration.py` | FireANTs behaviour on tiny CPU volumes, marked `requires_fireants` |
| `scripts/tier2_smoke.sbatch` | One-subject GPU end-to-end for both methods |
| `scripts/tier3_agreement.py` | Dice / volume / Jacobian comparison against the delivered derivative |
| `scripts/tier3_agreement.sbatch` | SLURM wrapper for the above on `pi_satra` |
| `requirements-test.txt` | CPU torch + fireants for CI |

**Modified:**

| File | Change |
|---|---|
| `src/antspy/wrapper.py:415-439` | Fusion loop → one `warp_atlases()` call |
| `src/antspy/wrapper.py:802-869` | Affine+SyN → `register_to_template()`; `TemplateToSubjectTransforms` → `TemplateWarp` |
| `src/antspy/wrapper.py:355-378` | `ants.apply_transforms` → `apply_warp()` |
| `src/antspy/wrapper.py:23-30` | Constructor gains `device`, `fireants_batch_size`, `label_interpolation` |
| `src/run.py:198-248` | Three new CLI flags |
| `src/run.py:288-296` | Pass them to `ANTsSegmentation` |
| `tests/test_wrapper.py:197-214` | Structural invtransforms assertion → behavioural direction test |
| `requirements.txt`, `Dockerfile`, `Singularity`, `run_container.sh` | torch + fireants + `fused_ops`, `--nv` |

---

### Task 1: Geometry guard and the NIfTI boundary

**Files:**
- Create: `src/registration/__init__.py` (empty for now)
- Create: `src/registration/handoff.py`
- Test: `tests/test_handoff.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Geometry` (namedtuple: `shape`, `spacing`, `origin`, `direction`, all tuples of float except `shape` which is a tuple of int); `geometry_of(img) -> Geometry`; `GeometryMismatch(RuntimeError)`; `assert_geometry_match(a, b, tol=1e-4, context="") -> None`; `write_nifti(img, path) -> Path`; `read_nifti(path, expect=None) -> ants.ANTsImage`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_handoff.py`:

```python
"""Geometry guard and NIfTI boundary tests.

These use nibabel only -- no antspyx, no torch, no fireants -- so they run in
CI unconditionally. They validate the claim the whole design rests on: that a
NIfTI file carries the affine faithfully across libraries.
"""
import numpy as np
import nibabel as nib
import pytest

from src.registration.handoff import (
    Geometry,
    GeometryMismatch,
    assert_geometry_match,
    geometry_of,
)


def _img(affine, shape=(4, 5, 6)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return nib.Nifti1Image(data, affine)


# A deliberately nasty affine: anisotropic spacing, non-identity direction,
# non-zero origin. An identity affine would hide exactly the bugs we care about.
OBLIQUE = np.array([
    [0.0, -1.2, 0.0, 10.0],
    [0.9, 0.0, 0.0, -20.0],
    [0.0, 0.0, 3.0, 5.5],
    [0.0, 0.0, 0.0, 1.0],
])


def test_geometry_of_extracts_spacing_from_oblique_affine():
    g = geometry_of(_img(OBLIQUE))
    assert g.shape == (4, 5, 6)
    np.testing.assert_allclose(g.spacing, (0.9, 1.2, 3.0), rtol=1e-6)


def test_geometry_survives_a_disk_round_trip(tmp_path):
    src = _img(OBLIQUE)
    path = tmp_path / "oblique.nii.gz"
    nib.save(src, str(path))
    assert_geometry_match(geometry_of(src), geometry_of(nib.load(str(path))))


def test_reoriented_image_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    flipped = OBLIQUE.copy()
    flipped[:3, :3] = -flipped[:3, :3]
    with pytest.raises(GeometryMismatch, match="direction"):
        assert_geometry_match(a, geometry_of(_img(flipped)))


def test_changed_spacing_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    rescaled = OBLIQUE.copy()
    rescaled[:3, :3] *= 2.0
    with pytest.raises(GeometryMismatch, match="spacing"):
        assert_geometry_match(a, geometry_of(_img(rescaled)))


def test_shifted_origin_is_rejected():
    a = geometry_of(_img(OBLIQUE))
    shifted = OBLIQUE.copy()
    shifted[0, 3] += 1.0
    with pytest.raises(GeometryMismatch, match="origin"):
        assert_geometry_match(a, geometry_of(_img(shifted)))


def test_differing_shape_is_rejected():
    with pytest.raises(GeometryMismatch, match="shape"):
        assert_geometry_match(
            geometry_of(_img(OBLIQUE)), geometry_of(_img(OBLIQUE, shape=(4, 5, 7)))
        )


def test_context_string_appears_in_the_error():
    a = geometry_of(_img(OBLIQUE))
    shifted = OBLIQUE.copy()
    shifted[0, 3] += 1.0
    with pytest.raises(GeometryMismatch, match="warped atlas 3"):
        assert_geometry_match(a, geometry_of(_img(shifted)), context="warped atlas 3")


def test_tolerance_admits_float_noise():
    a = geometry_of(_img(OBLIQUE))
    noisy = OBLIQUE.copy()
    noisy[0, 3] += 1e-9
    assert_geometry_match(a, geometry_of(_img(noisy)))


def test_geometry_of_accepts_itk_style_objects():
    """ANTsImage and SimpleITK images expose spacing/origin/direction directly.

    A stub stands in so this runs without antspyx installed.
    """

    class FakeANTsImage:
        shape = (4, 5, 6)
        spacing = (0.9, 1.2, 3.0)
        origin = (-10.0, -20.0, 5.5)
        direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    g = geometry_of(FakeANTsImage())
    assert g.shape == (4, 5, 6)
    np.testing.assert_allclose(g.spacing, (0.9, 1.2, 3.0), rtol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_handoff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.registration'`

- [ ] **Step 3: Write minimal implementation**

Create `src/registration/__init__.py` as an empty file.

Create `src/registration/handoff.py`:

```python
"""File-based handoff between ANTsPy and FireANTs.

Every image crossing between the two libraries goes through a NIfTI file on
disk. Nothing here rebuilds a header from loose parts: NIfTI carries the affine
and both ANTsPy and SimpleITK are well-tested readers of it. Pulling a numpy
array out of an ANTsImage and reattaching spacing/origin/direction on the other
side is the failure mode this module exists to make impossible.

See docs/superpowers/specs/2026-09-01-fireants-engine-design.md section 4.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np

Geometry = namedtuple("Geometry", "shape spacing origin direction")

# ITK (ANTs, SimpleITK) uses LPS; nibabel/NIfTI affines are RAS. Negating the
# first two axes converts between them.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


class GeometryMismatch(RuntimeError):
    """Two images that should share a voxel grid do not."""


def _from_affine(affine, shape):
    affine = np.asarray(affine, dtype=float)
    lps = _RAS_TO_LPS @ affine[:3, :3]
    spacing = np.linalg.norm(lps, axis=0)
    spacing[spacing == 0.0] = 1.0
    direction = lps / spacing
    origin = _RAS_TO_LPS @ affine[:3, 3]
    return Geometry(
        tuple(int(s) for s in shape),
        tuple(float(s) for s in spacing),
        tuple(float(o) for o in origin),
        tuple(float(d) for d in np.asarray(direction).ravel()),
    )


def geometry_of(img):
    """Extract a comparable geometry from an ANTsImage, a SimpleITK image, or a
    nibabel image. Results are always in LPS so the three are comparable.
    """
    if hasattr(img, "affine"):  # nibabel
        return _from_affine(img.affine, img.shape)
    if hasattr(img, "GetSpacing"):  # SimpleITK
        return Geometry(
            tuple(int(s) for s in reversed(img.GetSize())),
            tuple(float(s) for s in img.GetSpacing()),
            tuple(float(o) for o in img.GetOrigin()),
            tuple(float(d) for d in img.GetDirection()),
        )
    # ANTsImage
    return Geometry(
        tuple(int(s) for s in img.shape),
        tuple(float(s) for s in img.spacing),
        tuple(float(o) for o in img.origin),
        tuple(float(d) for d in np.asarray(img.direction).ravel()),
    )


def assert_geometry_match(a, b, tol=1e-4, context=""):
    """Raise GeometryMismatch unless a and b describe the same voxel grid.

    Called on every crossing of the ANTs/FireANTs boundary in production, not
    only under pytest. A silently reoriented volume yields a well-formed
    segmentation, a well-formed nidm.ttl, and exit 0.
    """
    where = f" [{context}]" if context else ""
    if tuple(a.shape) != tuple(b.shape):
        raise GeometryMismatch(f"shape differs{where}: {a.shape} != {b.shape}")
    for field in ("spacing", "origin", "direction"):
        va, vb = np.asarray(getattr(a, field)), np.asarray(getattr(b, field))
        if not np.allclose(va, vb, atol=tol, rtol=0.0):
            raise GeometryMismatch(
                f"{field} differs{where} (atol={tol}): {tuple(va)} != {tuple(vb)}"
            )


def write_nifti(img, path):
    """Write an ANTsImage to `path` for FireANTs to read. Returns the path."""
    import ants

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(img, str(path))
    return path


def read_nifti(path, expect=None, context=""):
    """Read a NIfTI back as an ANTsImage, verifying geometry against `expect`.

    `expect` is a Geometry (typically of the fixed/target image). Passing it is
    the point of this function; omitting it makes the read unguarded.
    """
    import ants

    img = ants.image_read(str(path))
    if expect is not None:
        assert_geometry_match(expect, geometry_of(img), context=context or str(path))
    return img
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_handoff.py -v`
Expected: 9 passed

- [ ] **Step 5: Confirm the existing suite still passes**

Run: `pytest -m "not ci_incompatible" tests/ -q`
Expected: no new failures versus the pre-task baseline

- [ ] **Step 6: Commit**

```bash
git add src/registration/__init__.py src/registration/handoff.py tests/test_handoff.py
git commit -m "Add geometry guard and file-based ANTs/FireANTs boundary

Every image crossing between ANTsPy and FireANTs goes through a NIfTI file
rather than an in-memory array plus a hand-rebuilt header. assert_geometry_match
runs on every crossing in production, not only under pytest, because a silently
reoriented volume produces a well-formed segmentation and exit 0.

Tests use nibabel only, so they run in CI without antspyx or torch, and use an
oblique affine with anisotropic spacing -- an identity affine would hide the
bugs this guards."
```

---

### Task 2: Device resolution

**Files:**
- Create: `src/registration/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DeviceUnavailable(RuntimeError)`; `resolve_device(requested: str, cuda_available: Callable[[], bool] | None = None) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_device.py`:

```python
"""Device resolution tests.

The CUDA probe is injectable so these run without torch installed.
"""
import pytest

from src.registration.device import DeviceUnavailable, resolve_device


def test_cuda_resolves_when_a_gpu_is_present():
    assert resolve_device("cuda", cuda_available=lambda: True) == "cuda"


def test_cuda_without_a_gpu_raises_rather_than_falling_back():
    """Silent CPU fallback under a BABS array turns a fast job into a
    many-hour one while still exiting 0."""
    with pytest.raises(DeviceUnavailable) as exc:
        resolve_device("cuda", cuda_available=lambda: False)
    assert "--device" in str(exc.value)
    assert "cpu" in str(exc.value)


def test_cpu_is_honoured_and_never_probes_for_cuda():
    def explode():
        raise AssertionError("must not probe for CUDA when cpu was requested")

    assert resolve_device("cpu", cuda_available=explode) == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps", cuda_available=lambda: True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.registration.device'`

- [ ] **Step 3: Write minimal implementation**

Create `src/registration/device.py`:

```python
"""Compute-device selection for the FireANTs registration path."""

VALID_DEVICES = ("cuda", "cpu")


class DeviceUnavailable(RuntimeError):
    """A GPU was requested and none is visible."""


def _torch_cuda_available():
    import torch

    return torch.cuda.is_available()


def resolve_device(requested, cuda_available=None):
    """Return the device to run FireANTs on, or raise.

    Deliberately does not fall back to CPU when CUDA is missing. FireANTs on CPU
    is slower than the ANTs code it replaces, so a silent fallback inside a BABS
    array produces hundreds of many-hour jobs that all exit 0. `--device cpu`
    remains available as an explicit opt-in for tests and small debugging runs.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {requested!r}; expected one of {VALID_DEVICES}"
        )
    if requested == "cpu":
        return "cpu"
    probe = cuda_available or _torch_cuda_available
    if not probe():
        raise DeviceUnavailable(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Submit to a GPU partition (e.g. --partition pi_satra --gres=gpu:1) "
            "and run the container with `apptainer --nv`, or pass --device cpu "
            "for a small debugging run. Not falling back to CPU automatically: "
            "FireANTs on CPU is slower than the ANTs path it replaces."
        )
    return "cuda"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_device.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/registration/device.py tests/test_device.py
git commit -m "Add device resolution that refuses to fall back to CPU

--device cuda with no visible GPU raises before any processing rather than
degrading. FireANTs on CPU is slower than the ANTs code being replaced, so a
silent fallback inside a BABS array yields hundreds of many-hour jobs that all
exit 0 -- the failure shape this repo has already been bitten by.

The CUDA probe is injectable so the tests run without torch installed."
```

---

### Task 3: Label remapping

**Files:**
- Create: `src/registration/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LabelMap` (frozen dataclass with `forward: dict[int,int]`, `inverse: numpy.ndarray`, `max_index: int`); `build_label_map(label_paths: Sequence[Path]) -> LabelMap`; `LabelMap.remap_array(a: numpy.ndarray) -> numpy.ndarray`; `LabelMap.restore_array(a: numpy.ndarray) -> numpy.ndarray`.

**Why this exists:** `fireants.utils.imageutils.integer_to_onehot` allocates `max_label + 1` channels. DKT31 ids run to 2035, so an unremapped one-hot is 2036 channels — about 136 GB at 256³. Remapping to 0…96 makes it 97.

- [ ] **Step 1: Write the failing test**

Create `tests/test_labels.py`:

```python
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


def test_round_trip_restores_original_ids(tmp_path):
    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 11, 30, 1001, 2035])
    lm = build_label_map([a])
    original = np.array([[0, 4], [2035, 1001]], dtype=np.int32)
    np.testing.assert_array_equal(
        lm.restore_array(lm.remap_array(original)), original
    )


def test_remap_output_is_dense_enough_for_onehot(tmp_path):
    """The whole point: max index must be small, not 2035."""
    a = _write_labels(tmp_path, "a.nii.gz", [0, 4, 2035])
    lm = build_label_map([a])
    assert lm.remap_array(np.array([2035])).max() == 2
    assert lm.max_index == 2


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.registration.labels'`

- [ ] **Step 3: Write minimal implementation**

Create `src/registration/labels.py`:

```python
"""Contiguous remapping of sparse anatomical label ids.

fireants.utils.imageutils.integer_to_onehot allocates max_label + 1 channels.
DKT31 ids are sparse and reach 2035, so warping them directly would allocate
2036 channels -- roughly 136 GB at 256^3. Remapping the ~97 distinct ids onto
0..96 makes the same operation cost 97 channels.

Background (0) is always mapped to 0 and always present, because FireANTs is
called with background_seg_label=-1, which builds num_labels = max_label + 1
and skips no channel. Channel c is therefore remapped label c and the inverse
is a plain lookup. Calling FireANTs with background_seg_label=0 would drop a
channel and shift every index by one.
"""
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class LabelMap:
    forward: dict
    inverse: np.ndarray

    @property
    def max_index(self):
        return int(self.inverse.size - 1)

    def remap_array(self, a):
        out = np.zeros_like(a, dtype=np.int32)
        for original, index in self.forward.items():
            out[a == original] = index
        return out

    def restore_array(self, a):
        a = np.asarray(a, dtype=np.int64)
        if a.size and (a.max() > self.max_index or a.min() < 0):
            raise IndexError(
                f"remapped index out of range [0, {self.max_index}]: "
                f"saw [{a.min()}, {a.max()}]"
            )
        return self.inverse[a]


def build_label_map(label_paths):
    """Build a contiguous map from the union of label ids across all atlases.

    Uses nibabel rather than ANTsPy so this is importable and testable without
    antspyx, and reads with `dataobj` to avoid loading full volumes into memory.
    """
    values = {0}
    for path in label_paths:
        img = nib.load(str(Path(path)))
        values.update(int(v) for v in np.unique(np.asarray(img.dataobj)))
    ordered = sorted(values)
    forward = {original: index for index, original in enumerate(ordered)}
    inverse = np.array(ordered, dtype=np.int32)
    assert all(inverse[forward[v]] == v for v in ordered), "label map is not invertible"
    return LabelMap(forward=forward, inverse=inverse)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/registration/labels.py tests/test_labels.py
git commit -m "Remap sparse DKT31 label ids onto a contiguous range

integer_to_onehot allocates max_label + 1 channels. DKT31 ids reach 2035, so
one-hotting them unremapped costs 2036 channels -- about 136 GB at 256 cubed.
Remapping the 97 distinct ids to 0..96 makes the same operation cost 97.

Background is pinned to index 0 and always present because FireANTs is called
with background_seg_label=-1, under which no channel is skipped and channel c
is remapped label c."
```

---

### Task 4: FireANTs backend — atlas warping

**Files:**
- Create: `src/registration/fireants.py`
- Create: `requirements-test.txt`
- Create: `pytest.ini`
- Test: `tests/test_registration.py`

**Interfaces:**
- Consumes: `src.registration.handoff.{geometry_of, assert_geometry_match, write_nifti, read_nifti}`; `src.registration.labels.{LabelMap, build_label_map}`; `src.registration.device.resolve_device`.
- Produces: `warp_atlases(target, target_mask, atlas_paths, label_paths, *, temp_dir, device="cuda", batch_size=0, label_interpolation="genericlabel", logger=None) -> tuple[list, list]` returning `(warped_atlas_images, warped_label_images)` as ANTsImages; `labels_from_warp_result(tensor, label_map) -> numpy.ndarray`; `folding_fraction(registration, fixed_batch, moving_batch) -> float`; `MAX_FOLDING_FRACTION = 0.001`.

- [ ] **Step 1: Add the test dependency and marker**

Create `requirements-test.txt`:

```
# CPU-only torch for CI. The container installs the CUDA build instead; see
# Dockerfile / Singularity. Kept separate so CI does not pull ~5 GB of nvidia
# wheels for tests that run on 16^3 volumes.
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.3.0
fireants==1.5.0
SimpleITK>=2.2.1
```

Create `pytest.ini` at the repo root:

```ini
[pytest]
markers =
    ci_incompatible: requires a full ANTs/template environment, skipped in CI
    requires_fireants: requires fireants + torch importable; skipped otherwise
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_registration.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_registration.py -v`
Expected: FAIL — `ImportError: cannot import name 'labels_from_warp_result'` (or the whole module skips if fireants is not installed locally; install `requirements-test.txt` first)

- [ ] **Step 4: Write minimal implementation**

Create `src/registration/fireants.py`:

```python
"""FireANTs registration backend.

The only module in this package that imports `fireants` or `torch`. Everything
else stays importable without them so CI can run the geometry, label, and
device tests unconditionally.

Design constraints enforced here (spec section 4):
  * Images cross to and from ANTsPy as NIfTI files, never as arrays with a
    hand-rebuilt header. Registration outputs are tensors, so their header is
    copied wholesale from the fixed image via SimpleITK CopyInformation --
    copied, not reconstructed from spacing/origin/direction parts.
  * fireants.io.Image is never given orientation / spacing / direction /
    origin / center. `orientation` applies sitk.DICOMOrient and rewrites the
    header; the rest override geometry that should come from the file.
  * Labels are remapped to a contiguous range before one-hotting, because
    integer_to_onehot allocates max_label + 1 channels and DKT31 ids reach 2035.
"""
import logging
from pathlib import Path

import numpy as np

from src.registration.handoff import (
    assert_geometry_match,
    geometry_of,
    read_nifti,
    write_nifti,
)
from src.registration.labels import build_label_map

# Hyperparameters from FireANTs' own scripts/pairwise/oasis.py rather than
# invented. SyN rather than Greedy, as the closer analogue to the SyNRA being
# replaced.
AFFINE_SCALES = [4, 2, 1]
AFFINE_ITERATIONS = [200, 100, 50]
SYN_SCALES = [4, 2, 1]
SYN_ITERATIONS = [200, 100, 50]
SYN_LR = 0.5
CC_KERNEL_SIZE = 5
SMOOTH_GRAD_SIGMA = 1.0
SMOOTH_WARP_SIGMA = 0.5

# Spec section 8 criterion 7. A folding warp is provably bad independently of
# Dice; the ANTs path being replaced had no equivalent check.
MAX_FOLDING_FRACTION = 0.001


def labels_from_warp_result(tensor, label_map):
    """Turn a FireANTs label-warp result into original-id labels.

    The result has one of two shapes depending on whether fused ops are active:
      [1, 1, H, W, D]  argmax indices, when interpolation_mode == 'genericlabel'
      [1, C, H, W, D]  one-hot, when fireants/io/image.py:145-147 forced
                       is_onehot=True because FFO was unavailable

    Both are remapped indices, never original ids.
    """
    array = tensor.detach().cpu().numpy()
    channels = array.shape[1]
    if channels == 1:
        indices = np.squeeze(array, axis=(0, 1)).round().astype(np.int64)
    else:
        expected = label_map.max_index + 1
        if channels != expected:
            raise ValueError(
                f"one-hot warp result has {channels} channels but the label map "
                f"has {expected}; the label map and the loaded image disagree"
            )
        indices = np.argmax(array[0], axis=0).astype(np.int64)
    return label_map.restore_array(indices)


def _seg_preprocessor(label_map):
    """Remap sparse ids to contiguous indices, on the torch tensor FireANTs
    hands us (fireants/io/image.py:151, applied before max_seg_label clamping).
    """
    import torch

    def preprocess(array):
        out = torch.zeros_like(array)
        for original, index in label_map.forward.items():
            out[array == original] = float(index)
        return out

    return preprocess


def _tensor_to_nifti(tensor, reference_itk, path):
    """Write a registration output tensor using the fixed image's header.

    CopyInformation copies the header object wholesale; it does not rebuild one
    from spacing/origin/direction values. The result is in fixed space by
    construction, so the fixed image's header is the correct one.
    """
    import SimpleITK as sitk

    array = np.asarray(tensor)
    while array.ndim > 3:
        array = array[0]
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(reference_itk)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path))
    return path


def _chunks(items, size):
    if not size or size <= 0 or size >= len(items):
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def warp_atlases(
    target,
    target_mask,
    atlas_paths,
    label_paths,
    *,
    temp_dir,
    device="cuda",
    batch_size=0,
    label_interpolation="genericlabel",
    logger=None,
):
    """Register every atlas to `target` and return warped atlases and labels.

    `target_mask` is accepted because ants.joint_label_fusion needs it
    downstream; it is deliberately not wired into the similarity loss. Both
    sides of the metric are already skull-stripped (`masked_brain` is
    N4 * BrainExtractionMask, and the OASIS-TRT-20 atlases are brain-only), so a
    mask would change nothing while adding a difference from the reference.

    Returns (warped_atlas_images, warped_label_images) as ANTsImages.
    """
    import SimpleITK as sitk
    import torch
    from fireants.interpolator import fireants_interpolator
    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.syn import SyNRegistration

    log = logger or logging.getLogger("ants-nidm.registration")
    atlas_paths = [Path(p) for p in atlas_paths]
    label_paths = [Path(p) for p in label_paths]
    if len(atlas_paths) != len(label_paths):
        raise ValueError(
            f"{len(atlas_paths)} atlases but {len(label_paths)} label volumes"
        )

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_geom = geometry_of(target)

    label_map = build_label_map(label_paths)
    log.info(
        "Label map: %d distinct ids -> 0..%d (one-hot would otherwise need %d "
        "channels)",
        len(label_map.forward),
        label_map.max_index,
        int(label_map.inverse.max()) + 1,
    )

    from fireants.interpolator import FFO_AVAILABLE

    log.info(
        "FireANTs device=%s fused_ops=%s label_interpolation=%s",
        device,
        FFO_AVAILABLE,
        label_interpolation,
    )
    if label_interpolation == "genericlabel" and (
        not FFO_AVAILABLE or device != "cuda"
    ):
        log.warning(
            "genericlabel requested but fused ops are unavailable (FFO=%s, "
            "device=%s); FireANTs will fall back to one-hot, costing %d channels",
            FFO_AVAILABLE,
            device,
            label_map.max_index + 1,
        )

    target_path = write_nifti(target, temp_dir / "target.nii.gz")
    fixed = Image.load_file(str(target_path), device=device)
    fixed_batch = BatchedImages([fixed])
    reference_itk = fixed.itk_image

    warped_atlases, warped_labels = [], []
    index = 0
    for chunk in _chunks(list(zip(atlas_paths, label_paths)), batch_size):
        moving_batch = BatchedImages(
            [Image.load_file(str(a), device=device) for a, _ in chunk]
        )
        log.info("Registering %d atlases (affine)", len(chunk))
        affine = AffineRegistration(
            AFFINE_SCALES,
            AFFINE_ITERATIONS,
            fixed_batch,
            moving_batch,
            optimizer="Adam",
            optimizer_lr=3e-3,
            cc_kernel_size=CC_KERNEL_SIZE,
        )
        affine.optimize()

        log.info("Registering %d atlases (SyN)", len(chunk))
        syn = SyNRegistration(
            SYN_SCALES,
            SYN_ITERATIONS,
            fixed_batch,
            moving_batch,
            deformation_type="compositive",
            optimizer="Adam",
            optimizer_lr=SYN_LR,
            cc_kernel_size=CC_KERNEL_SIZE,
            smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
            smooth_warp_sigma=SMOOTH_WARP_SIGMA,
            init_affine=affine.get_affine_matrix().detach(),
        )
        syn.optimize()

        moved = syn.evaluate(fixed_batch, moving_batch)

        folding = folding_fraction(syn, fixed_batch, moving_batch)
        log.info("Non-positive Jacobian determinants: %.4f%%", folding * 100.0)
        if folding > MAX_FOLDING_FRACTION:
            raise RuntimeError(
                f"registration folded: {folding * 100:.3f}% of voxels have a "
                f"non-positive Jacobian determinant (limit "
                f"{MAX_FOLDING_FRACTION * 100:.3f}%). The warp is not "
                f"diffeomorphic; the segmentation would be wrong in a way Dice "
                f"against a single reference may not reveal."
            )

        if label_interpolation == "genericlabel":
            label_batch = BatchedImages(
                [
                    Image.load_file(
                        str(lp),
                        device=device,
                        is_segmentation=True,
                        background_seg_label=-1,
                        max_seg_label=label_map.max_index,
                        seg_preprocessor=_seg_preprocessor(label_map),
                    )
                    for _, lp in chunk
                ]
            )
            moved_labels = syn.evaluate(fixed_batch, label_batch)
        elif label_interpolation == "nearest":
            label_batch = BatchedImages(
                [Image.load_file(str(lp), device=device) for _, lp in chunk]
            )
            coords = syn.get_warp_parameters(fixed_batch, label_batch)
            moved_labels = fireants_interpolator(
                label_batch(), **coords, mode="nearest", align_corners=True
            )
        else:
            raise ValueError(
                f"unknown label_interpolation {label_interpolation!r}; "
                "expected 'genericlabel' or 'nearest'"
            )

        for k in range(len(chunk)):
            atlas_out = _tensor_to_nifti(
                moved[k].detach().cpu().numpy(),
                reference_itk,
                temp_dir / f"warped_atlas_{index:02d}.nii.gz",
            )
            if label_interpolation == "genericlabel":
                ids = labels_from_warp_result(moved_labels[k : k + 1], label_map)
            else:
                ids = (
                    moved_labels[k].detach().cpu().numpy().squeeze().round().astype(np.int64)
                )
            label_out = _tensor_to_nifti(
                ids.astype(np.float32),
                reference_itk,
                temp_dir / f"warped_labels_{index:02d}.nii.gz",
            )

            warped_atlases.append(
                read_nifti(atlas_out, expect=target_geom, context=f"warped atlas {index}")
            )
            warped_labels.append(
                read_nifti(label_out, expect=target_geom, context=f"warped labels {index}")
            )
            index += 1

        del affine, syn, moved, moved_labels, moving_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info("Warped %d atlases", len(warped_atlases))
    return warped_atlases, warped_labels


def folding_fraction(registration, fixed_batch, moving_batch):
    """Fraction of voxels where the warp's Jacobian determinant is <= 0.

    A folding warp is provably bad independently of Dice, and the ANTs path
    being replaced had no such check. Spec section 8 criterion 7 gates this at
    0.1%. Mirrors the diffeomorphism check in FireANTs' own
    tutorials/tutorial1_basic_usage.py.
    """
    import torch
    from fireants.utils.imageutils import jacobian

    warp = registration.get_warped_coordinates(fixed_batch, moving_batch)
    jac = jacobian(warp).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac).reshape(-1)
    return float((det <= 0).float().mean().item())
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pip install -r requirements-test.txt
pytest tests/test_registration.py -v
```
Expected: 4 passed

- [ ] **Step 6: Confirm nothing else broke**

Run: `pytest -m "not ci_incompatible" tests/ -q`
Expected: no new failures

- [ ] **Step 7: Commit**

```bash
git add src/registration/fireants.py requirements-test.txt pytest.ini tests/test_registration.py
git commit -m "Add FireANTs atlas warping with explicit label-shape handling

warp_atlases registers all atlases against the target in batches and returns
ANTsImages that ants.joint_label_fusion consumes unchanged. Every crossing is a
NIfTI file, geometry-checked against the target on read.

The label result has two possible shapes: [1, 1, H, W, D] argmax indices when
fused ops are active, or [1, C, H, W, D] one-hot when fireants/io/image.py
forced is_onehot=True because FFO was unavailable. Both are remapped indices,
never original ids. Handing the one-hot form to joint_label_fusion unchanged
would produce a well-formed, wrong result, so the shape is detected explicitly
and a channel-count mismatch raises rather than argmaxing into wrong ids.

Registration output tensors get their header from SimpleITK CopyInformation on
the fixed image -- copied wholesale, not rebuilt from parts."
```

---

### Task 5: Template registration and TemplateWarp

**Files:**
- Modify: `src/registration/fireants.py` (append)
- Test: `tests/test_registration.py` (append)

**Interfaces:**
- Consumes: everything from Task 4.
- Produces: `TemplateWarp(registration, subject_batch, template_batch, reference_itk, device)` with a `warp_parameters()` method and public `reference_itk` / `device` attributes; `register_to_template(subject, template, *, temp_dir, device="cuda", logger=None) -> TemplateWarp`; `apply_warp(warp, moving, reference, *, temp_dir, interpolation="genericlabel", label_map=None, logger=None) -> ants.ANTsImage`.

**Direction contract:** `register_to_template` registers subject → template, but `apply_warp` moves **template-space images into subject space**. That is the only direction any consumer needs, and it is the direction `TemplateWarp` exposes. `wrapper.py:839-851` documents an 11.5 L brain volume caused by getting this backwards with `fwdtransforms`/`invtransforms`. `TemplateWarp` exposes one direction so a caller cannot pick the wrong attribute.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registration.py`:

```python
def test_template_warp_moves_template_space_into_subject_space(tmp_path):
    """Behavioural direction guard, replacing the structural invtransforms
    assertion in tests/test_wrapper.py.

    A marker placed in template space must land at the predicted subject-space
    voxel. If the direction is inverted it lands on the opposite side and this
    fails. This is the regression guard for the 11.5 L whole-FOV bug.
    """
    import nibabel as nib

    from src.registration.fireants import apply_warp, register_to_template

    ants = pytest.importorskip("ants")

    shape = (24, 24, 24)
    shift = 4  # voxels along x

    subject = np.zeros(shape, dtype=np.float32)
    subject[8 + shift : 16 + shift, 8:16, 8:16] = 1.0
    template = np.zeros(shape, dtype=np.float32)
    template[8:16, 8:16, 8:16] = 1.0

    sub_path = tmp_path / "subject.nii.gz"
    tpl_path = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(subject, np.eye(4)), str(sub_path))
    nib.save(nib.Nifti1Image(template, np.eye(4)), str(tpl_path))

    sub_img = ants.image_read(str(sub_path))
    tpl_img = ants.image_read(str(tpl_path))

    warp = register_to_template(
        sub_img, tpl_img, temp_dir=tmp_path / "work", device="cpu"
    )

    marker = np.zeros(shape, dtype=np.float32)
    marker[8:16, 8:16, 8:16] = 1.0
    marker_path = tmp_path / "marker.nii.gz"
    nib.save(nib.Nifti1Image(marker, np.eye(4)), str(marker_path))

    moved = apply_warp(
        warp,
        ants.image_read(str(marker_path)),
        sub_img,
        temp_dir=tmp_path / "work",
        interpolation="nearest",
    )
    out = moved.numpy()

    centre = float(np.mean(np.nonzero(out > 0.5)[0]))
    subject_centre = float(np.mean(np.nonzero(subject > 0.5)[0]))
    template_centre = float(np.mean(np.nonzero(template > 0.5)[0]))
    assert abs(centre - subject_centre) < abs(centre - template_centre), (
        f"template marker landed at x={centre:.1f}; subject box is at "
        f"x={subject_centre:.1f} and template box at x={template_centre:.1f}. "
        "Landing nearer the template centre means the warp direction is inverted "
        "-- this is the 11.5 L brain-volume bug."
    )


def test_apply_warp_output_shares_the_reference_geometry(tmp_path):
    import nibabel as nib

    from src.registration.fireants import apply_warp, register_to_template
    from src.registration.handoff import assert_geometry_match, geometry_of

    ants = pytest.importorskip("ants")

    shape = (16, 16, 16)
    affine = np.diag([1.1, 0.9, 1.3, 1.0])
    blob = np.zeros(shape, dtype=np.float32)
    blob[6:10, 6:10, 6:10] = 1.0

    paths = {}
    for name in ("subject", "template"):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(blob, affine), str(p))
        paths[name] = ants.image_read(str(p))

    warp = register_to_template(
        paths["subject"], paths["template"], temp_dir=tmp_path / "w", device="cpu"
    )
    moved = apply_warp(
        warp,
        paths["template"],
        paths["subject"],
        temp_dir=tmp_path / "w",
        interpolation="nearest",
    )
    assert_geometry_match(geometry_of(paths["subject"]), geometry_of(moved))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registration.py -k template -v`
Expected: FAIL — `ImportError: cannot import name 'register_to_template'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/registration/fireants.py`:

```python
class TemplateWarp:
    """A subject<->template registration, exposed in one direction only.

    Consumers pull template-space images into subject space -- the brain
    probability mask, and the DKT31 label atlas under --method quick. That is
    the only direction exposed. It replaces the previous
    `TemplateToSubjectTransforms`, a list of ANTs transform file paths whose
    orientation depended on whether the caller read fwdtransforms or
    invtransforms; reading the wrong one produced an 11.5 L brain volume
    (wrapper.py:839-851).

    Held as torch tensors. Never serialised to ANTs transform format, which is
    why the composition-order and inversion round trip through disk does not
    exist in this design.
    """

    def __init__(self, registration, subject_batch, template_batch, reference_itk, device):
        self._registration = registration
        self._subject_batch = subject_batch
        self._template_batch = template_batch
        self.reference_itk = reference_itk
        self.device = device

    def warp_parameters(self):
        """Coordinates that map template-space samples into subject space."""
        return self._registration.get_warp_parameters(
            self._subject_batch, self._template_batch
        )


def register_to_template(subject, template, *, temp_dir, device="cuda", logger=None):
    """Register `subject` to `template`, returning a template->subject warp.

    `subject` is the skull-stripped brain; `template` is
    T_template0_BrainCerebellum. Registering onto the whole-head T_template0
    biased the affine toward scaling the brain up to the head outline
    (wrapper.py:804-812) -- pass the brain-only template.
    """
    from fireants.io import BatchedImages, Image
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.syn import SyNRegistration

    log = logger or logging.getLogger("ants-nidm.registration")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    subject_path = write_nifti(subject, temp_dir / "tpl_subject.nii.gz")
    template_path = write_nifti(template, temp_dir / "tpl_template.nii.gz")

    # fixed = subject, moving = template. The registration is set up in the
    # direction we need to *apply*, so no inversion is ever required.
    fixed = Image.load_file(str(subject_path), device=device)
    moving = Image.load_file(str(template_path), device=device)
    fixed_batch, moving_batch = BatchedImages([fixed]), BatchedImages([moving])

    log.info("Template registration (affine)")
    affine = AffineRegistration(
        AFFINE_SCALES,
        AFFINE_ITERATIONS,
        fixed_batch,
        moving_batch,
        optimizer="Adam",
        optimizer_lr=3e-3,
        cc_kernel_size=CC_KERNEL_SIZE,
    )
    affine.optimize()

    log.info("Template registration (SyN)")
    syn = SyNRegistration(
        SYN_SCALES,
        SYN_ITERATIONS,
        fixed_batch,
        moving_batch,
        deformation_type="compositive",
        optimizer="Adam",
        optimizer_lr=SYN_LR,
        cc_kernel_size=CC_KERNEL_SIZE,
        smooth_grad_sigma=SMOOTH_GRAD_SIGMA,
        smooth_warp_sigma=SMOOTH_WARP_SIGMA,
        init_affine=affine.get_affine_matrix().detach(),
    )
    syn.optimize()

    return TemplateWarp(syn, fixed_batch, moving_batch, fixed.itk_image, device)


def apply_warp(
    warp,
    moving,
    reference,
    *,
    temp_dir,
    interpolation="genericlabel",
    label_map=None,
    logger=None,
):
    """Pull a template-space image into subject space.

    `interpolation='nearest'` warps raw intensities; 'genericlabel' treats the
    input as a label volume and requires `label_map`.
    """
    from fireants.interpolator import fireants_interpolator
    from fireants.io import BatchedImages, Image

    log = logger or logging.getLogger("ants-nidm.registration")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    reference_geom = geometry_of(reference)

    moving_path = write_nifti(moving, temp_dir / "apply_moving.nii.gz")
    coords = warp.warp_parameters()

    if interpolation == "genericlabel":
        if label_map is None:
            raise ValueError("interpolation='genericlabel' requires label_map")
        img = Image.load_file(
            str(moving_path),
            device=warp.device,
            is_segmentation=True,
            background_seg_label=-1,
            max_seg_label=label_map.max_index,
            seg_preprocessor=_seg_preprocessor(label_map),
        )
        result = fireants_interpolator(
            BatchedImages([img])(),
            **coords,
            mode=img.interpolation_mode or "bilinear",
            align_corners=True,
        )
        array = labels_from_warp_result(result, label_map).astype(np.float32)
    elif interpolation == "nearest":
        img = Image.load_file(str(moving_path), device=warp.device)
        result = fireants_interpolator(
            BatchedImages([img])(), **coords, mode="nearest", align_corners=True
        )
        array = result.detach().cpu().numpy()
    else:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; "
            "expected 'genericlabel' or 'nearest'"
        )

    out_path = _tensor_to_nifti(array, warp.reference_itk, temp_dir / "applied.nii.gz")
    log.info("Applied template->subject warp (%s)", interpolation)
    return read_nifti(out_path, expect=reference_geom, context="apply_warp result")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registration.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/registration/fireants.py tests/test_registration.py
git commit -m "Add TemplateWarp with a single, behaviourally tested direction

register_to_template sets fixed=subject and moving=template, so the
registration is already in the direction consumers apply and no inversion is
ever required. TemplateWarp exposes that one direction rather than a transform
list whose orientation depends on which attribute the caller reads -- reading
the wrong one produced the 11.5 L whole-FOV brain volume recorded at
wrapper.py:839-851.

The guard is behavioural: a marker placed in template space must land nearer
the subject box than the template box. An inverted warp fails it. The warp
stays a torch tensor and is never serialised to ANTs transform format."
```

---

### Task 6: Public API and the fusion loop

**Files:**
- Modify: `src/registration/__init__.py`
- Modify: `src/antspy/wrapper.py:23-30` (constructor), `405-439` (fusion loop)
- Test: `tests/test_wrapper.py` (append)

**Interfaces:**
- Consumes: `src.registration.fireants.{warp_atlases, register_to_template, apply_warp, TemplateWarp}`.
- Produces: package-level `warp_atlases`, `register_to_template`, `apply_warp`, `TemplateWarp`, `DeviceUnavailable`, `resolve_device`, imported lazily; `ANTsSegmentation(..., device='cuda', fireants_batch_size=0, label_interpolation='genericlabel')`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wrapper.py`:

```python
    def test_fusion_delegates_registration_to_warp_atlases(self):
        """The twenty-atlas ants.registration loop is gone; segment_image calls
        warp_atlases once and passes its output straight to joint_label_fusion.
        """
        from unittest.mock import patch

        warped_imgs = [mock_image_read() for _ in range(2)]
        warped_labs = [mock_image_read() for _ in range(2)]

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = (warped_imgs, warped_labs)
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        self.assertEqual(mock_warp.call_count, 1, "warp_atlases must be called once")
        self.assertIs(mock_jlf.call_args.kwargs["atlas_list"], warped_imgs)
        self.assertIs(mock_jlf.call_args.kwargs["label_list"], warped_labs)

    def test_fusion_passes_engine_options_through(self):
        from unittest.mock import patch

        self.segmenter.device = "cpu"
        self.segmenter.fireants_batch_size = 4
        self.segmenter.label_interpolation = "nearest"

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = ([], [])
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        kwargs = mock_warp.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["label_interpolation"], "nearest")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wrapper.py -k warp_atlases -v`
Expected: FAIL — `AttributeError: <module 'src.antspy.wrapper'> does not have the attribute 'warp_atlases'`

- [ ] **Step 3: Write the public API**

Replace `src/registration/__init__.py` with:

```python
"""Registration for the ants-nidm BIDS App, backed by FireANTs.

The FireANTs backend is imported lazily so that `handoff`, `labels`, and
`device` stay importable without torch or fireants -- CI runs their tests
unconditionally.
"""
from src.registration.device import DeviceUnavailable, resolve_device
from src.registration.handoff import (
    Geometry,
    GeometryMismatch,
    assert_geometry_match,
    geometry_of,
)
from src.registration.labels import LabelMap, build_label_map

__all__ = [
    "DeviceUnavailable",
    "Geometry",
    "GeometryMismatch",
    "LabelMap",
    "TemplateWarp",
    "apply_warp",
    "assert_geometry_match",
    "build_label_map",
    "geometry_of",
    "register_to_template",
    "resolve_device",
    "warp_atlases",
]


def __getattr__(name):
    if name in ("warp_atlases", "register_to_template", "apply_warp", "TemplateWarp"):
        from src.registration import fireants

        return getattr(fireants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 4: Wire the wrapper**

In `src/antspy/wrapper.py`, add to the imports at the top:

```python
from src.registration import apply_warp, register_to_template, warp_atlases
from src.registration.device import resolve_device
```

Add three parameters to `ANTsSegmentation.__init__` (after `num_threads=1,`):

```python
                 device='cuda',
                 fireants_batch_size=0,
                 label_interpolation='genericlabel',
```

and, next to the other assignments (after `self.num_threads = num_threads`):

```python
        # Registration runs on FireANTs. resolve_device raises rather than
        # silently falling back to CPU: FireANTs on CPU is slower than the ANTs
        # code it replaced, so a fallback under a BABS array yields hundreds of
        # many-hour jobs that all exit 0.
        self.device = resolve_device(device)
        self.fireants_batch_size = fireants_batch_size
        self.label_interpolation = label_interpolation
```

Replace `src/antspy/wrapper.py` lines 411-439 (from `# Register each atlas to the masked brain` through the end of the `for` loop) with:

```python
            # Registration is FireANTs; the fusion voting below stays on ANTs.
            # Images cross the boundary as NIfTI files under temp_dir and are
            # geometry-checked on the way back -- see src/registration/handoff.py.
            warped_images, warped_labels = warp_atlases(
                masked_brain,
                thickness_results['BrainExtractionMask'],
                T1s,
                labels,
                temp_dir=self.temp_dir,
                device=self.device,
                batch_size=self.fireants_batch_size,
                label_interpolation=self.label_interpolation,
                logger=self.logger,
            )
```

Delete the now-unused `atlas_images` / `atlas_labels` loading lines (currently 408-409) — `warp_atlases` reads the atlases from disk by path, so ANTsPy never loads them.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_wrapper.py -v`
Expected: all pass, including the two new tests

- [ ] **Step 6: Confirm no ANTs registration remains in the fusion path**

```bash
grep -n "ants.registration\|ants.apply_transforms" src/antspy/wrapper.py
```
Expected: only the template-registration sites (Task 7) remain; nothing between lines 400 and 460.

- [ ] **Step 7: Commit**

```bash
git add src/registration/__init__.py src/antspy/wrapper.py tests/test_wrapper.py
git commit -m "Route the fusion loop through FireANTs

The twenty ants.registration(SyNRA) calls and their apply_transforms label
warping collapse to one warp_atlases() call. ants.joint_label_fusion consumes
its output unchanged -- the voting is untouched.

ANTsPy no longer loads the atlas volumes at all: warp_atlases reads them from
disk by path, so 40 of the per-subject volumes never cross a library boundary.

The package's FireANTs backend is imported lazily so handoff, labels, and
device stay importable without torch, keeping their tests unconditional in CI."
```

---

### Task 7: Template registration in the wrapper

**Files:**
- Modify: `src/antspy/wrapper.py:355-378` (`quick` branch), `802-869` (`compute_cortical_thickness` tail)
- Modify: `tests/test_wrapper.py:197-214` (replace the structural direction test)

**Interfaces:**
- Consumes: `register_to_template`, `apply_warp`, `build_label_map` from Task 5/3.
- Produces: `compute_cortical_thickness` returns `TemplateWarp` under the key `'TemplateWarp'`. The key `'TemplateToSubjectTransforms'` is **removed**; nothing may read it afterwards.

- [ ] **Step 1: Replace the structural direction test**

In `tests/test_wrapper.py`, delete `test_cortical_thickness_keeps_invtransforms_for_template_to_subject` (lines 197-214) and its now-unused `_run_cortical_thickness_recording_calls` registration assertions, and add:

```python
    def test_cortical_thickness_returns_a_template_warp_not_transform_paths(self):
        """The 11.5 L regression guard, restated for the new representation.

        The old assertion compared TemplateToSubjectTransforms against
        invtransforms. That list no longer exists: the warp is a torch object
        that is never serialised, so there is no fwd/inv attribute to pick
        wrongly. What this test pins is that the old, footgun-shaped key is
        gone and the single-direction object is present. The behavioural
        direction check lives in
        tests/test_registration.py::test_template_warp_moves_template_space_into_subject_space.
        """
        from unittest.mock import patch

        sentinel = object()
        with patch("src.antspy.wrapper.register_to_template", return_value=sentinel), \
             patch("src.antspy.wrapper.ants.n4_bias_field_correction",
                   side_effect=mock_image_read), \
             patch("src.antspy.wrapper.shutil.which", return_value="/fake/abe.sh"), \
             patch("src.antspy.wrapper.subprocess.run") as mock_run, \
             patch("src.antspy.wrapper.Path.exists", return_value=True):
            mock_run.return_value.returncode = 0
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        self.assertIs(results["TemplateWarp"], sentinel)
        self.assertNotIn(
            "TemplateToSubjectTransforms", results,
            "the ANTs transform-path list must be gone, not merely unused: "
            "leaving it invites a caller to read the wrong direction again",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wrapper.py -k template_warp -v`
Expected: FAIL — `KeyError: 'TemplateWarp'`

- [ ] **Step 3: Replace the registration block**

In `src/antspy/wrapper.py`, replace lines 802-867 (from `reg_template = ants.image_read(...)` through `results['TemplateToSubjectTransforms'] = transforms`) with:

```python
        # Register to the brain-only template. brain_image is skull-stripped at
        # this point, so the fixed image must be skull-stripped too.
        # Registering onto the whole-head T_template0 biased the affine toward
        # scaling the brain up to the head outline, inflating every
        # template-space image warped back with these transforms.
        brain_template = ants.image_read(
            str(self.template_dir / 'T_template0_BrainCerebellum.nii.gz'))

        self.logger.info("Registering to template (FireANTs)")
        # register_to_template sets fixed=subject, moving=template, so the
        # returned warp already points template -> subject: the direction every
        # consumer needs. There is no fwd/inv pair to choose between, which is
        # what previously produced an 11.5 L brain volume on ABIDE sub-0051456.
        template_warp = register_to_template(
            brain_image,
            brain_template,
            temp_dir=self.temp_dir,
            device=self.device,
            logger=self.logger,
        )

        # BrainExtractionMask is the canonical script's product, untouched: its
        # Atropos refinement and morphology already finalized it.
        results = {
            'BrainSegmentationN4': n4_image,
            'BrainExtractionMask': final_mask,
            'TemplateWarp': template_warp,
        }

        return results
```

- [ ] **Step 4: Replace the `quick` branch**

In `src/antspy/wrapper.py`, replace lines 359-378 (from `template_labels = ants.image_read(...)` through the `return` of the `quick` branch) with:

```python
            template_warp = thickness_results.get('TemplateWarp')
            if template_warp is None:
                raise ValueError("No transforms available for template-to-subject mapping")

            label_map = build_label_map([self.template_labels_path])
            warped_labels = apply_warp(
                template_warp,
                ants.image_read(str(self.template_labels_path)),
                thickness_results['BrainSegmentationN4'],
                temp_dir=self.temp_dir,
                interpolation=self.label_interpolation,
                label_map=label_map,
                logger=self.logger,
            )

            return {
                'segmentation': warped_labels,
                'BrainSegmentationN4': thickness_results['BrainSegmentationN4'],
                'BrainExtractionMask': thickness_results['BrainExtractionMask']
            }
```

Add `build_label_map` to the import added in Task 6:

```python
from src.registration import apply_warp, build_label_map, register_to_template, warp_atlases
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_wrapper.py -v`
Expected: all pass

- [ ] **Step 6: Verify every ANTs registration site is gone**

```bash
grep -n "ants.registration\|ants.apply_transforms\|TemplateToSubjectTransforms" src/antspy/wrapper.py
```
Expected: **no output.** All five sites replaced.

- [ ] **Step 7: Commit**

```bash
git add src/antspy/wrapper.py tests/test_wrapper.py
git commit -m "Replace the template registration with TemplateWarp

The Affine+SyN pair and the quick-method apply_transforms both move to
FireANTs, completing the switch: grep for ants.registration or
ants.apply_transforms in wrapper.py now returns nothing.

TemplateToSubjectTransforms is removed rather than left unused. It was a list
of ANTs transform file paths whose meaning depended on whether the caller read
fwdtransforms or invtransforms, and reading the wrong one produced an 11.5 L
brain volume. TemplateWarp has one direction and cannot be read wrongly.

Because both the producer and the only consumer moved together, the warp is
never written to ANTs transform format and whichtoinvert inference disappears
with it."
```

---

### Task 8: CLI flags

**Files:**
- Modify: `src/run.py:198-248` (`parse_arguments`), `288-296` (`initialize`)
- Test: `tests/test_run.py` (append)

**Interfaces:**
- Consumes: `ANTsSegmentation(device=..., fireants_batch_size=..., label_interpolation=...)` from Task 6.
- Produces: `args.device`, `args.fireants_batch_size`, `args.label_interpolation`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run.py`, inside `class TestRun`:

```python
    def test_device_defaults_to_cuda(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant"]
        with patch.object(sys, "argv", argv):
            args = parse_arguments()
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.fireants_batch_size, 0)
        self.assertEqual(args.label_interpolation, "genericlabel")

    def test_device_and_engine_options_are_parsed(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = [
            "ants-nidm", "/bids", "/out", "participant",
            "--device", "cpu",
            "--fireants-batch-size", "5",
            "--label-interpolation", "nearest",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_arguments()
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.fireants_batch_size, 5)
        self.assertEqual(args.label_interpolation, "nearest")

    def test_invalid_device_is_rejected(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant", "--device", "mps"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_arguments()

    def test_negative_batch_size_is_rejected(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant",
                "--fireants-batch-size", "-2"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_arguments()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -k device -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'device'`

- [ ] **Step 3: Write minimal implementation**

In `src/run.py`, after the `--num-threads` argument, add:

```python
    # Registration runs on FireANTs (GPU). --device cuda with no visible GPU
    # fails before any processing rather than falling back: FireANTs on CPU is
    # slower than the ANTs code it replaced, so a silent fallback under a BABS
    # array yields hundreds of many-hour jobs that all exit 0.
    parser.add_argument('--device', help='Compute device for registration [default: cuda]',
                        choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--fireants-batch-size', dest='fireants_batch_size',
                        help='Atlases registered per batch; 0 registers all at once [default: 0]',
                        type=int, default=0)
    # Without built fused_ops, or on CPU, genericlabel degrades to a one-hot
    # return that warp_atlases argmaxes; see fireants/io/image.py:145-147.
    parser.add_argument('--label-interpolation', dest='label_interpolation',
                        help='Atlas label warping method [default: genericlabel]',
                        choices=['genericlabel', 'nearest'], default='genericlabel')
```

In the validation block just before `return args`, add:

```python
    if args.fireants_batch_size < 0:
        parser.error("--fireants-batch-size must be >= 0 (0 means all atlases at once)")
```

In `initialize`, extend the `ANTsSegmentation(...)` call with:

```python
            device=args.device,
            fireants_batch_size=args.fireants_batch_size,
            label_interpolation=args.label_interpolation,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/run.py tests/test_run.py
git commit -m "Add --device, --fireants-batch-size, --label-interpolation

No --engine flag: FireANTs is the only registration backend. --device cuda
with no visible GPU raises in the ANTsSegmentation constructor, before any
image is read."
```

---

### Task 9: Container, requirements, and build-file parity

**Files:**
- Modify: `requirements.txt`, `Dockerfile`, `Singularity`, `run_container.sh`
- Test: `tests/test_build_files.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: a container with CUDA torch, `fireants==1.5.0`, and a built `fused_ops`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_files.py`:

```python
"""Dockerfile and Singularity are near-duplicates and have drifted before.

This pins the versions that must match across both, and asserts the container
verifies its own fused_ops build -- without it FireANTs silently degrades to a
one-hot label path costing 97x the memory.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
SINGULARITY = (ROOT / "Singularity").read_text()
REQUIREMENTS = (ROOT / "requirements.txt").read_text()

PINNED = {
    "ants": r"ANTs/releases/download/v([\d.]+)/",
    "fireants": r"fireants==([\d.]+)",
}


@pytest.mark.parametrize("name,pattern", sorted(PINNED.items()))
def test_versions_match_across_build_files(name, pattern):
    d = re.search(pattern, DOCKERFILE)
    s = re.search(pattern, SINGULARITY)
    assert d, f"{name} version not found in Dockerfile"
    assert s, f"{name} version not found in Singularity"
    assert d.group(1) == s.group(1), (
        f"{name} differs: Dockerfile {d.group(1)} vs Singularity {s.group(1)}"
    )


def test_fireants_pin_matches_requirements():
    req = re.search(r"fireants==([\d.]+)", REQUIREMENTS)
    doc = re.search(r"fireants==([\d.]+)", DOCKERFILE)
    assert req and doc and req.group(1) == doc.group(1)


@pytest.mark.parametrize("text,name", [(DOCKERFILE, "Dockerfile"), (SINGULARITY, "Singularity")])
def test_build_verifies_fused_ops(text, name):
    """Without fused_ops the default label path silently degrades rather than
    failing, so the build must prove the extension imports."""
    assert "import fireants_fused_ops" in text, (
        f"{name} must assert `import fireants_fused_ops` succeeds"
    )


@pytest.mark.parametrize("text,name", [(DOCKERFILE, "Dockerfile"), (SINGULARITY, "Singularity")])
def test_build_installs_cuda_torch(text, name):
    assert "torch" in text, f"{name} must install torch"
    assert "cu12" in text or "cuda" in text.lower(), (
        f"{name} must install a CUDA torch build, not the CPU wheel"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_files.py -v`
Expected: FAIL — fireants version not found in Dockerfile

- [ ] **Step 3: Update requirements.txt**

Append to `requirements.txt`:

```
# Registration runs on FireANTs (GPU). torch is installed separately in the
# container with a CUDA index URL; the CPU wheel used by CI lives in
# requirements-test.txt.
fireants==1.5.0
SimpleITK>=2.2.1
```

- [ ] **Step 4: Update Dockerfile**

After the existing `pip3 install` block, insert:

```dockerfile
# CUDA torch + FireANTs. Registration is GPU-only; see
# docs/superpowers/specs/2026-09-01-fireants-engine-design.md.
RUN pip3 install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch>=2.3.0" && \
    pip3 install --no-cache-dir fireants==1.5.0 SimpleITK>=2.2.1

# The fused CUDA ops are NOT included by `pip install fireants` and must be
# built from source. Without them FireANTs does not fail -- it forces
# is_onehot=True (fireants/io/image.py:145-147) and silently returns a one-hot
# volume costing 97x the memory. Assert the extension imports so a broken build
# is caught here rather than in a BABS array.
RUN git clone --depth 1 --branch v1.5.0 https://github.com/rohitrango/FireANTs.git /tmp/fireants-src && \
    cd /tmp/fireants-src/fused_ops && \
    python3 setup.py build_ext && \
    python3 setup.py install && \
    cd / && rm -rf /tmp/fireants-src && \
    python3 -c "import fireants_fused_ops; print('fused ops OK')"
```

Apply the identical change to `Singularity` in its `%post` section, without the `RUN` prefixes and with `&&` continuations replaced by newlines, matching the file's existing style.

- [ ] **Step 5: Update run_container.sh**

After the line that builds the Apptainer argument list, add `--nv`:

```bash
# FireANTs registration needs the GPU visible inside the container.
APPTAINER_ARGS+=(--nv)
```

and document it in the header comment block:

```bash
#   - --nv so the GPU is visible inside the container (registration is
#     GPU-only; without it the app exits early with a device error)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_build_files.py -v`
Expected: 7 passed

- [ ] **Step 7: Build the container**

```bash
apptainer build --fakeroot ants-nidm_bidsapp.sif Singularity
```
Expected: build succeeds and prints `fused ops OK`. **If it does not print that, stop** — the default label path is unusable and every downstream tier is invalid.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-test.txt Dockerfile Singularity run_container.sh tests/test_build_files.py
git commit -m "Add CUDA torch, FireANTs, and a verified fused_ops build

pip install fireants does NOT include the fused CUDA ops, and their absence is
silent: fireants/io/image.py:145-147 forces is_onehot=True and the label warp
returns a one-hot volume costing 97x the memory instead of raising. Both build
files therefore compile fused_ops from source and assert the extension imports.

A parity test pins the ANTs and FireANTs versions across Dockerfile and
Singularity, which are near-duplicates that have drifted before."
```

---

### Task 10: Tier-2 GPU smoke test

**Files:**
- Create: `scripts/tier2_smoke.sbatch`

**Interfaces:**
- Consumes: the container from Task 9.
- Produces: a pass/fail smoke result for both `--method fusion` and `--method quick`.

- [ ] **Step 1: Write the script**

Create `scripts/tier2_smoke.sbatch`:

```bash
#!/bin/bash
# Tier-2 smoke: one real subject, both methods, on a GPU node.
#
# Proves the seam holds end to end -- FireANTs registers, ANTs fuses, outputs
# land in the per-subject layout with NIDM alongside. It does NOT check
# segmentation quality; that is tier 3.
#
# Usage:
#   sbatch scripts/tier2_smoke.sbatch <sif> <bids_dir> <out_dir> <participant>
#SBATCH --job-name=ants-fireants-tier2
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tier2_smoke_%j.out

set -euo pipefail

SIF="$1"; BIDS="$2"; OUT="$3"; SUB="$4"

nvidia-smi
mkdir -p "$OUT"

for METHOD in fusion quick; do
    echo "=== method=${METHOD} ==="
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/tier2-${METHOD}-XXXXXX")"
    apptainer run --nv \
        --bind "${BIDS}:/bids:ro" \
        --bind "${OUT}/${METHOD}:/out" \
        --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" \
        --method "$METHOD" \
        --device cuda \
        --num-threads 8 \
        --skip-nidm \
        --verbose
    rm -rf "$WORK"

    SUBDIR="${OUT}/${METHOD}/sub-${SUB}"
    for REQUIRED in "anat" "stats" "processing_summary.json"; do
        test -e "${SUBDIR}/${REQUIRED}" \
            || { echo "MISSING ${SUBDIR}/${REQUIRED}"; exit 1; }
    done
    test -s "${SUBDIR}/stats/sub-${SUB}_antsbrainvols.csv" \
        || { echo "empty brainvols for ${METHOD}"; exit 1; }
    echo "=== ${METHOD} OK ==="
done

echo "TIER 2 PASS"
```

- [ ] **Step 2: Make it executable and commit**

```bash
chmod +x scripts/tier2_smoke.sbatch
git add scripts/tier2_smoke.sbatch
git commit -m "Add tier-2 GPU smoke test for both segmentation methods

quick is covered as well as fusion because the template registration it
depends on also moved to FireANTs, and quick has no tier-3 Dice reference --
the delivered derivative was produced with fusion."
```

- [ ] **Step 3: Run it and record the result**

```bash
sbatch scripts/tier2_smoke.sbatch ants-nidm_bidsapp.sif \
  /orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech/sourcedata \
  "$PWD/tier2_out" 0051456
```
Expected: job exits 0 with `TIER 2 PASS`. **Do not proceed to Task 11 until it does** — tier 3 is meaningless if the pipeline does not complete.

---

### Task 11: Tier-3 agreement against the delivered derivative

**Files:**
- Create: `scripts/tier3_agreement.py`, `scripts/tier3_agreement.sbatch`

**Interfaces:**
- Consumes: tier-2 output; the reference derivative at `/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech/derivatives/babs-ants-nidm`.
- Produces: `tier3_report.csv` and a pass/fail exit code.

**Gate (spec section 8), derived from the 38-subject reference:**

| Tier | Labels | Criterion |
|---|---|---|
| Excluded | present in <90% of reference subjects (15, 2033) | not evaluated |
| Primary | median volume ≥ 2000 mm³ (87 labels) | median Dice ≥ 0.85, none below 0.75 |
| Relaxed | 250–2000 mm³ (50, 55, 30, 56, 49) | each ≥ 0.70 |
| Report-only | < 250 mm³ (69, 64, 63) | reported, not gated |

Plus: per-label \|Δvolume\| ≤ 2× that label's inter-subject SD in the reference; BVOL ∈ [1.29, 1.94] L; and the folding check from Task 4 (non-positive Jacobian determinants < 0.1%), which fails the run at registration time rather than here.

Surface Dice at 1 mm and 95th-percentile Hausdorff are written to the report for every gated label but are **not** gate criteria — no reference distribution exists for them yet. They catch boundary degradation that volumetric Dice hides on large structures, and establish the baseline for future runs.

- [ ] **Step 1: Write the comparison script**

Create `scripts/tier3_agreement.py`:

```python
#!/usr/bin/env python3
"""Tier-3 agreement: FireANTs output vs the delivered ANTs derivative.

The reference segmentations already exist on disk, so no ANTs registration run
is needed -- this reads delivered files. Thresholds are stratified by observed
label volume rather than flat, because labels 15 and 2033 appear in 2/38
reference subjects and label 69 has 99.8% inter-subject CV under ANTs alone.

Usage:
  tier3_agreement.py --new <dir> --reference <dir> --subjects <id> [<id> ...]
                     --out tier3_report.csv
"""
import argparse
import csv
import glob
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

PRIMARY_MIN_VOL = 2000.0
RELAXED_MIN_VOL = 250.0
PRIMARY_MEDIAN_DICE = 0.85
PRIMARY_FLOOR_DICE = 0.75
RELAXED_FLOOR_DICE = 0.70
PRESENCE_FRACTION = 0.90
BVOL_RANGE_L = (1.29, 1.94)


def reference_label_stats(reference):
    """Median volume, inter-subject SD, and presence fraction per label."""
    per_label = defaultdict(list)
    files = sorted(glob.glob(str(Path(reference) / "sub-*/stats/*antslabelstats.csv")))
    if not files:
        sys.exit(f"no reference label stats under {reference}")
    for path in files:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                per_label[int(row["Label"])].append(float(row["Volume_mm3"]))
    n = len(files)
    stats = {}
    for label, volumes in per_label.items():
        padded = volumes + [0.0] * (n - len(volumes))
        stats[label] = {
            "median": st.median(padded),
            "sd": st.pstdev(padded),
            "presence": len(volumes) / n,
        }
    return stats, n


def dice(a, b, label):
    x, y = a == label, b == label
    denom = x.sum() + y.sum()
    return float("nan") if denom == 0 else 2.0 * np.logical_and(x, y).sum() / denom


def surface_metrics(new, ref, label, voxel_mm3):
    """Surface Dice at 1 mm and 95th-percentile Hausdorff, via FireANTs' own
    evaluation harness.

    Reported, not gated: no reference distribution exists for them yet. They
    catch boundary degradation that volumetric Dice hides on large structures,
    and establish the baseline for future runs (spec section 8).
    """
    empty = {"surface_dice_1mm": float("nan"), "hausdorff95_mm": float("nan")}
    try:
        from fireants.scripts.evalutils import (
            compute_robust_hausdorff,
            compute_surface_dice_at_tolerance,
            compute_surface_distances,
        )
    except ImportError:
        return empty

    a, b = (new == label), (ref == label)
    if not a.any() or not b.any():
        return empty
    spacing = (voxel_mm3 ** (1.0 / 3.0),) * 3
    distances = compute_surface_distances(b, a, spacing_mm=spacing)
    return {
        "surface_dice_1mm": float(compute_surface_dice_at_tolerance(distances, 1.0)),
        "hausdorff95_mm": float(compute_robust_hausdorff(distances, 95.0, method="linear")),
    }


def find_segmentation(root, subject):
    hits = sorted(Path(root).glob(f"sub-{subject}/anat/*dseg.nii.gz"))
    if not hits:
        hits = sorted(Path(root).glob(f"sub-{subject}/anat/*BrainSegmentation.nii.gz"))
    if not hits:
        sys.exit(f"no segmentation for sub-{subject} under {root}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--out", default="tier3_report.csv")
    args = parser.parse_args()

    stats, n_ref = reference_label_stats(args.reference)
    print(f"reference: {n_ref} subjects, {len(stats)} labels")

    excluded = {l for l, s in stats.items() if s["presence"] < PRESENCE_FRACTION}
    primary = {l for l, s in stats.items()
               if l not in excluded and s["median"] >= PRIMARY_MIN_VOL}
    relaxed = {l for l, s in stats.items()
               if l not in excluded and RELAXED_MIN_VOL <= s["median"] < PRIMARY_MIN_VOL}
    report_only = {l for l, s in stats.items()
                   if l not in excluded and s["median"] < RELAXED_MIN_VOL}
    print(f"excluded={sorted(excluded)} primary={len(primary)} "
          f"relaxed={sorted(relaxed)} report_only={sorted(report_only)}")

    rows, failures = [], []
    for subject in args.subjects:
        new = np.asarray(nib.load(str(find_segmentation(args.new, subject))).dataobj)
        ref = np.asarray(nib.load(str(find_segmentation(args.reference, subject))).dataobj)
        if new.shape != ref.shape:
            failures.append(f"sub-{subject}: shape {new.shape} != reference {ref.shape}")
            continue

        voxel_mm3 = float(np.prod(nib.load(
            str(find_segmentation(args.new, subject))).header.get_zooms()[:3]))
        primary_dices = []
        for label in sorted(primary | relaxed | report_only):
            d = dice(new, ref, label)
            dv = (float((new == label).sum()) - float((ref == label).sum())) * voxel_mm3
            tier = ("primary" if label in primary
                    else "relaxed" if label in relaxed else "report_only")
            surface = surface_metrics(new, ref, label, voxel_mm3)
            rows.append({
                "subject": subject, "label": label, "tier": tier,
                "dice": d, "delta_volume_mm3": dv,
                "reference_sd_mm3": stats[label]["sd"],
                "surface_dice_1mm": surface["surface_dice_1mm"],
                "hausdorff95_mm": surface["hausdorff95_mm"],
            })
            if tier == "primary" and not np.isnan(d):
                primary_dices.append(d)
                if d < PRIMARY_FLOOR_DICE:
                    failures.append(f"sub-{subject} label {label}: dice {d:.3f} < {PRIMARY_FLOOR_DICE}")
            if tier == "relaxed" and not np.isnan(d) and d < RELAXED_FLOOR_DICE:
                failures.append(f"sub-{subject} label {label}: dice {d:.3f} < {RELAXED_FLOOR_DICE}")
            if tier != "report_only" and abs(dv) > 2.0 * stats[label]["sd"]:
                failures.append(
                    f"sub-{subject} label {label}: |dvol| {abs(dv):.0f} mm3 "
                    f"> 2x reference SD {stats[label]['sd']:.0f}")

        if primary_dices:
            median = st.median(primary_dices)
            print(f"sub-{subject}: median primary dice {median:.3f}")
            if median < PRIMARY_MEDIAN_DICE:
                failures.append(
                    f"sub-{subject}: median primary dice {median:.3f} < {PRIMARY_MEDIAN_DICE}")

        bvol_l = float((new > 0).sum()) * voxel_mm3 / 1e6
        print(f"sub-{subject}: BVOL {bvol_l:.2f} L")
        if not BVOL_RANGE_L[0] <= bvol_l <= BVOL_RANGE_L[1]:
            failures.append(f"sub-{subject}: BVOL {bvol_l:.2f} L outside {BVOL_RANGE_L}")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    if failures:
        print("\nTIER 3 FAIL")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nTIER 3 PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the SLURM wrapper**

Create `scripts/tier3_agreement.sbatch`:

```bash
#!/bin/bash
# Tier-3: run 3 Caltech subjects through the FireANTs container, then compare
# against the delivered ANTs derivative.
#SBATCH --job-name=ants-fireants-tier3
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tier3_%j.out

set -euo pipefail

SIF="${1:?usage: tier3_agreement.sbatch <sif> <out_dir>}"
OUT="${2:?usage: tier3_agreement.sbatch <sif> <out_dir>}"

DATASET=/orcd/data/satra/002/datasets/simple2_datalad/study-ABIDE/site-Caltech
BIDS="${DATASET}/sourcedata"
REFERENCE="${DATASET}/derivatives/babs-ants-nidm"
SUBJECTS=(0051456 0051457 0051458)

nvidia-smi
mkdir -p "$OUT"

for SUB in "${SUBJECTS[@]}"; do
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/tier3-${SUB}-XXXXXX")"
    apptainer run --nv \
        --bind "${BIDS}:/bids:ro" --bind "${OUT}:/out" --bind "${WORK}:/work" \
        "$SIF" /bids /out participant \
        --participant-label "$SUB" --method fusion --device cuda \
        --num-threads 8 --skip-nidm --verbose
    rm -rf "$WORK"
done

python3 scripts/tier3_agreement.py \
    --new "$OUT" --reference "$REFERENCE" \
    --subjects "${SUBJECTS[@]}" --out "${OUT}/tier3_report.csv"
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/tier3_agreement.py scripts/tier3_agreement.sbatch
git add scripts/tier3_agreement.py scripts/tier3_agreement.sbatch
git commit -m "Add tier-3 agreement harness with volume-stratified thresholds

Compares FireANTs output against the delivered 38-subject Caltech derivative.
Thresholds are derived from that cohort rather than chosen a priori: labels
present in under 90% of reference subjects are excluded (15 and 2033 appear in
2/38), labels under 250 mm3 are reported but not gated (69 has 99.8%
inter-subject CV under ANTs alone), and the 0.85 median applies to the 87
labels at or above 2000 mm3 where it is a real constraint.

Volume agreement is checked against 2x each label's own inter-subject SD, so
noisy structures are not held to the same bar as stable ones."
```

- [ ] **Step 4: Run it and record the result**

```bash
mkdir -p logs
sbatch scripts/tier3_agreement.sbatch ants-nidm_bidsapp.sif "$PWD/tier3_out"
```

Report the actual `tier3_report.csv` numbers. **A failure here blocks the merge** — there is no ANTs registration path left to fall back on. Before declaring failure, try the tuning levers in this order:

1. `--label-interpolation nearest`, which reproduces the reference's `nearestNeighbor` exactly and separates the interpolation change from the registration change.
2. SyN iterations and `optimizer_lr` in `src/registration/fireants.py`.
3. `cc_kernel_size`, `smooth_grad_sigma`, `smooth_warp_sigma`.

---
