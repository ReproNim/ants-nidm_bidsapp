import os
import sys

import pytest

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add the site-packages directory to the Python path
site_packages = os.path.join(os.path.dirname(sys.executable), 'lib', 'python3.12', 'site-packages')
if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)

# Print the Python path for debugging
print(f"Python path: {sys.path}")

# tests/test_run.py replaces sys.modules['nibabel'] with a MagicMock at module
# level (to intercept `import nibabel` inside src/antspy/wrapper.py during its
# own tests) and never restores it. Because pytest imports every test module
# during collection before running any test, that mock leaks into every test
# that later does its own `import nibabel`, regardless of which file it lives
# in. Modules that already bound `nib` to the real thing before collection
# reached test_run.py (e.g. src/registration/labels.py) are unaffected --
# their name is already resolved -- but a lazy `import nibabel` executed at
# test *run* time (e.g. tests/test_registration.py, which needs real NIfTI
# I/O) silently gets the mock instead, and calls like nib.save() become no-ops
# that raise nothing. Capturing the real module here, before any test file's
# collection-time code can replace it, and re-installing it before every test
# closes that leak without touching test_run.py's own mocking (which nothing
# else here depends on).
_real_nibabel = __import__("nibabel")


@pytest.fixture(autouse=True)
def _restore_real_nibabel():
    sys.modules['nibabel'] = _real_nibabel


# Same leak, same mechanism, different module: tests/test_run.py and
# tests/test_wrapper.py both do `sys.modules['ants'] = MagicMock()` at module
# level and never restore it. src/registration/handoff.py's write_nifti/
# read_nifti do a lazy `import ants` inside the function body, so once
# tests/test_registration.py started doing real ants.image_read/image_write
# I/O (Task 5's register_to_template/apply_warp tests), that lazy import
# silently resolved to the leaked mock: ants.image_write became a no-op, and
# the NIfTI file FireANTs expected to read next was never written.
#
# Unlike nibabel, this cannot be a blanket autouse fixture:
# tests/test_wrapper.py's own test_load_image legitimately needs the mock to
# stay installed during its own run, because src/antspy/wrapper.py's
# load_image() resolves `ants` lazily too, via
# `importlib.import_module("ants")` at call time -- restoring the real module
# unconditionally for every test breaks that test the same way the leak broke
# ours. So this is opt-in: captured here (before any test file's
# collection-time code can replace it, same as _real_nibabel above), restored
# only for the duration of a test that requests it, and put back afterward so
# later tests that depend on the mock are unaffected.
try:
    _real_ants = __import__("ants")
except ImportError:
    _real_ants = None


@pytest.fixture
def restore_real_ants():
    if _real_ants is None:
        yield
        return
    previous = sys.modules.get('ants')
    sys.modules['ants'] = _real_ants
    try:
        yield
    finally:
        if previous is not None:
            sys.modules['ants'] = previous
