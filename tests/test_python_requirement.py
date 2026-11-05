"""PyNIDM 5.0.0, which requirements.txt pins, requires Python 3.10 or newer.

Declaring an older floor makes `pip install` on 3.9 fail in dependency
resolution with a message about pynidm instead of the app's own Python
requirement, and a 3.9 CI leg fails at install and (fail-fast) cancels the
others.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_setup_py_requires_python_3_10():
    text = (ROOT / "setup.py").read_text()
    assert re.findall(r'python_requires="([^"]+)"', text) == [">=3.10"]
    versions = re.findall(r"Programming Language :: Python :: (3\.\d+)", text)
    assert versions and all(int(v.split(".")[1]) >= 10 for v in versions), versions


def test_ci_matrix_starts_at_python_3_10():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = re.findall(r"python-version: \[([^\]]+)\]", text)
    assert len(matrix) == 1, matrix
    versions = [v.strip().strip("\"'") for v in matrix[0].split(",")]
    assert versions and all(int(v.split(".")[1]) >= 10 for v in versions), versions
