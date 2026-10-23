"""The app version is written in several places; they must agree.

APP_VERSION (stamped into processing_summary.json), setup.py, the image's
%help, its version label (Singularity %post and Dockerfile LABEL) and the zip
suffix in the example BABS config are read by different people at different
times, and a release that bumps only some of them produces derivatives whose
zip name, summary and image disagree about what ran.
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


def test_image_version_label_is_the_app_version():
    """Both image recipes label the image with the app version.

    The Singularity recipe writes the label in %post: without --force, a
    %labels entry does not replace a label the base image already sets
    (Apptainer keeps the base's org.opencontainers.image.version, e.g.
    Ubuntu's 22.04, and only warns).
    """
    singularity = (ROOT / "Singularity").read_text()
    labels = singularity.split("\n%labels\n", 1)[1].split("\n%", 1)[0]
    assert "org.opencontainers.image.version" not in labels
    # Only %post: the same line in another section would not label the image.
    post_section = singularity.split("\n%post\n", 1)[1].split("\n%", 1)[0]
    post = re.findall(
        r'^\s+echo "org\.opencontainers\.image\.version (\S+)" '
        r'>> "\$\{APPTAINER_LABELS:-\$SINGULARITY_LABELS\}"$',
        post_section, re.MULTILINE,
    )
    assert post == [APP_VERSION]
    docker = re.findall(
        r"org\.opencontainers\.image\.version=(\S+)", (ROOT / "Dockerfile").read_text()
    )
    assert docker == [APP_VERSION]
