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
