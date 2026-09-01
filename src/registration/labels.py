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
