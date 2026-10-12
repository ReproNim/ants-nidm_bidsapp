"""Contiguous remapping of sparse anatomical label ids.

fireants.utils.imageutils.integer_to_onehot allocates max_label + 1 channels.
DKT31 ids are sparse and reach 2035, so warping them directly would allocate
2036 channels -- roughly 136 GB at 256^3. The union across the twenty
OASIS-TRT-20 label volumes is 105 distinct ids including background (measured,
not estimated: build_label_map over data/OASIS-TRT-20_DKT31_CMA_labels_v2/),
so remapping them onto 0..104 makes the same operation cost 105 channels.

The count is derived from the data at run time, never hard-coded -- this
docstring is the only place the number appears, and it is here for orders of
magnitude, not as a contract.

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

    # There is deliberately no numpy `remap_array` here. Production remaps on
    # the torch tensor FireANTs hands to `seg_preprocessor`
    # (src/registration/fireants.py::_seg_preprocessor); a second numpy
    # implementation existed only to be tested, which left the live mapping
    # untested and the tested one dead. `restore_array` below *is* production
    # code -- labels_from_warp_result calls it on every warped label volume.

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
    antspyx. Each volume is read in full, one at a time.
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
