"""The app version is written in four places; they must agree.

The image's %help, the version stamped into processing_summary.json and the
zip suffix in the example BABS config are read by different people at
different times, and a release that bumps only some of them produces
derivatives whose zip name, summary and image disagree about what ran.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _one(pattern, path):
    found = re.findall(pattern, (ROOT / path).read_text(), re.MULTILINE)
    assert len(found) == 1, f"expected one match of {pattern!r} in {path}, got {found}"
    return found[0]


APP_VERSION = _one(r'^APP_VERSION = "([^"]+)"$', "src/run.py")


def test_setup_py_matches_app_version():
    assert _one(r'^    version="([^"]+)",$', "setup.py") == APP_VERSION


def test_singularity_help_matches_app_version():
    text = (ROOT / "Singularity").read_text()
    found = re.findall(r"ANTs BIDS App:? ([\d.]+)$", text, re.MULTILINE)
    assert found, "Singularity %help names no app version"
    assert set(found) == {APP_VERSION}, f"Singularity %help says {found}"


def test_example_babs_config_zips_under_app_version():
    zipped = _one(r'^    \$\{subid\}: "([^"]+)"$', "examples/babs-ants-nidm-bids-study.yaml")
    assert zipped == APP_VERSION.replace(".", "-")
