"""The example BABS config must describe a run that works with this app.

Two settings in it are easy to get wrong and fail silently: an app-name entry
in zip_foldernames instead of ${subid}, which makes every subject's NIDM land on
the same path so only the first survives the zip merge; and missing GPU flags,
which make every job in the array fail with DeviceUnavailable.
"""
from pathlib import Path

import yaml

CONFIG = Path(__file__).parents[1] / "examples" / "babs-ants-nidm-bids-study.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text())


def test_zips_each_subject_folder():
    assert _config()["zip_foldernames"] == {"${subid}": "0-2-0"}


def test_requests_a_gpu():
    cfg = _config()
    assert "--nv" in cfg["singularity_args"]
    assert "--gres=gpu:1" in cfg["cluster_resources"]["customized_text"]


def test_uses_the_bids_study_layout():
    cfg = _config()
    assert cfg["analysis_path"] == "."
    assert cfg["input_ria_path"] == ".babs/input_ria"
    assert cfg["output_ria_path"] == ".babs/output_ria"
    assert cfg["input_datasets"]["BIDS"]["path_in_babs"] == "sourcedata/BIDS"
    assert cfg["bids_app_args"]["$SUBJECT_SELECTION_FLAG"] == "--participant-label"


def test_has_no_site_paths():
    for name, ds in _config()["input_datasets"].items():
        assert ds["origin_url"].startswith("${"), f"{name} origin_url must be a placeholder"
