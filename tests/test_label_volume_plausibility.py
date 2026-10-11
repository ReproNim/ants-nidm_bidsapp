"""A structure 70x too large is still under 1% of a 256^3 field of view.

That is why the existing pilot gate, whose checks are all FOV-relative, passed
every one of the 38 delivered Caltech subjects while each reported the third
ventricle at 50-130 thousand voxels against an atlas range of 321 to 1087.
These tests pin the check that catches it.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_label_volume_plausibility.py"

spec = importlib.util.spec_from_file_location("check_label_volume_plausibility", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture(scope="module")
def ranges():
    return gate.load_ranges()


def test_range_table_is_present_and_covers_the_atlas_vocabulary(ranges):
    table, n_atlases = ranges
    assert n_atlases == 20
    # 104 ids is the measured union across the twenty OASIS-TRT-20 atlases.
    assert len(table) == 104
    for label in (4, 35, 47, 59, 1002, 2035):
        assert label in table, f"atlas label {label} missing from the range table"
    assert table[4]["min"] <= table[4]["max"]


def test_the_delivered_defect_is_caught(ranges):
    """Label 4 at 77181 voxels is the real measurement from delivered
    sub-0051456. The gate exists to fail exactly this."""
    table, n = ranges
    findings = gate.check_one({4: 77181, 35: 18837}, table, n)
    assert len(findings) == 1
    f = findings[0]
    assert f["label"] == 4 and f["kind"] == "oversize"
    assert f["factor"] > 50


def test_a_plausible_subject_passes(ranges):
    """Volumes inside the atlas range must not be flagged, or the gate is
    noise and will be ignored."""
    table, n = ranges
    volumes = {int(k): (v["min"] + v["max"]) / 2 for k, v in table.items()}
    assert gate.check_one(volumes, table, n) == []


def test_genuinely_variable_structures_do_not_trip_it(ranges):
    """The lateral ventricles vary hugely with age and atrophy. Delivered
    sub-0051456 had 43170 voxels for label 51, which is only 1.5x the atlas
    maximum. Flagging that would make the gate useless."""
    table, n = ranges
    assert gate.check_one({51: 43170, 52: 43872}, table, n) == []


def test_undersize_is_only_judged_for_structures_every_atlas_has(ranges):
    """A label that is absent from some atlases may legitimately be absent
    from a subject, so absence there is not evidence of a defect."""
    table, n = ranges
    sometimes_absent = [int(k) for k, v in table.items() if v["n_atlases"] < n]
    assert sometimes_absent, "fixture assumes some labels are not in every atlas"
    label = sometimes_absent[0]
    assert gate.check_one({label: 1}, table, n) == []


def test_protocol_translation_changes_the_verdict(ranges):
    """quick emits FreeSurfer aseg ids. Checked as if they were atlas ids,
    aseg 15 (the 4th ventricle, ~2000 voxels) collides with atlas 15 (the 5th
    ventricle, 5-18 voxels) and looks 100x oversize. Told the right protocol,
    it is fine."""
    table, n = ranges
    wrong = gate.check_one({15: 1969}, table, n, protocol="neuromorphometrics")
    right = gate.check_one({15: 1969}, table, n, protocol="freesurfer")
    assert len(wrong) == 1 and wrong[0]["kind"] == "oversize"
    assert right == []


def test_a_structure_missing_entirely_is_flagged(ranges):
    """A structure every atlas has but the output lacks is the worst case of
    undersize. The stats file has no row for it, so a check that only reads
    the rows passed it while flagging a 1-voxel remnant. The fusion stats
    filter once dropped Right Accumbens (23) this way."""
    table, n = ranges
    always = [int(k) for k, v in table.items() if v["n_atlases"] == n]
    volumes = {int(k): (v["min"] + v["max"]) / 2 for k, v in table.items()}
    del volumes[always[0]]
    findings = gate.check_one(volumes, table, n, judge_missing=True)
    assert [(f["label"], f["kind"]) for f in findings] == [(always[0], "missing")]
    assert gate.check_one(volumes, table, n) == [], "partial dicts stay unjudged by default"


def test_a_missing_structure_is_named_in_the_output_protocol(ranges):
    """quick output is in aseg ids, so a missing structure is reported by its
    aseg id, the one a reader would look for in that stats file."""
    table, n = ranges
    always = [int(k) for k, v in table.items()
              if v["n_atlases"] == n and int(k) in gate.NEUROMORPHOMETRICS_TO_ASEG]
    nm = always[0]
    volumes = {gate.NEUROMORPHOMETRICS_TO_ASEG.get(int(k), int(k)): (v["min"] + v["max"]) / 2
               for k, v in table.items() if int(k) != nm}
    findings = gate.check_one(volumes, table, n, protocol="freesurfer", judge_missing=True)
    assert [(f["label"], f["atlas_label"], f["kind"]) for f in findings] == \
        [(gate.NEUROMORPHOMETRICS_TO_ASEG[nm], nm, "missing")]


def test_unknown_labels_are_ignored_not_guessed(ranges):
    """An id with no atlas evidence is outside this check's competence. It
    must not be flagged, and must not borrow another label's range."""
    table, n = ranges
    assert gate.check_one({999999: 10 ** 7}, table, n) == []


def test_reading_a_non_stats_csv_is_an_error(tmp_path):
    bad = tmp_path / "notstats.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="not an ANTs label stats file"):
        gate.read_label_stats(bad)


def test_reads_a_real_shaped_stats_file(tmp_path):
    f = tmp_path / "sub-X_antslabelstats.csv"
    f.write_text("Label,VolumeInVoxels,Volume_mm3\n4,77181,77181.0\n35,18837,18837.0\n")
    assert gate.read_label_stats(f) == {4: 77181.0, 35: 18837.0}


def test_cli_exits_nonzero_on_an_implausible_subject(tmp_path):
    import subprocess
    import sys

    stats = tmp_path / "sub-X" / "stats"
    stats.mkdir(parents=True)
    (stats / "sub-X_antslabelstats.csv").write_text(
        "Label,VolumeInVoxels,Volume_mm3\n4,77181,77181.0\n"
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "sub-X")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "label 4" in proc.stdout


def test_cli_exits_zero_on_a_plausible_subject(tmp_path):
    import subprocess
    import sys

    table, _ = gate.load_ranges()
    stats = tmp_path / "sub-Y" / "stats"
    stats.mkdir(parents=True)
    lines = ["Label,VolumeInVoxels,Volume_mm3"]
    for k, v in table.items():
        mid = (v["min"] + v["max"]) // 2
        lines.append(f"{k},{mid},{mid}.0")
    (stats / "sub-Y_antslabelstats.csv").write_text("\n".join(lines) + "\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "sub-Y")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_range_table_matches_the_atlases_when_they_are_available():
    """The table is generated from data/OASIS-TRT-20_DKT31_CMA_labels_v2, which
    is gitignored (downloaded at container build). Where it is present, the
    committed table must still agree with it."""
    atlas_dir = ROOT / "data" / "OASIS-TRT-20_DKT31_CMA_labels_v2"
    if not atlas_dir.exists():
        pytest.skip("atlas data not present in this checkout")
    nib = pytest.importorskip("nibabel")
    import numpy as np

    files = sorted(atlas_dir.glob("*.nii.gz"))
    assert files, "atlas directory exists but is empty"
    table, n_atlases = gate.load_ranges()
    assert n_atlases == len(files)
    # Spot-check one label end to end rather than re-reading all 20 volumes
    # for every id; label 4 is the one the whole gate was built for.
    vols = []
    for p in files:
        a = np.asarray(nib.load(str(p)).dataobj).astype(int)
        n = int((a == 4).sum())
        if n:
            vols.append(n)
    assert table[4]["min"] == min(vols)
    assert table[4]["max"] == max(vols)
    assert table[4]["n_atlases"] == len(vols)
