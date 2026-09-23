"""The joint-fusion vote has no background class, and the mask is bigger than
the atlases' labelled region.

That combination silently turned 89924 voxels of masked-but-unlabelled brain
into "third ventricle" in the delivered tier-2 output, because an argmax over
an all-near-zero probability stack returns the lowest label id.

``tests/test_wrapper.py`` pins the flag the wrapper passes. This pins the
behaviour that flag actually buys, against the real ANTsPy, so the choice
cannot be undone by an ANTsPy upgrade that redefines it.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

IN_CI = os.environ.get("CI", "false").lower() == "true"
ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "scripts" / "diag_jlf_background_fallback.py"


@unittest.skipIf(IN_CI, "needs a real ANTsPy, which CI does not install")
class TestJointFusionBackgroundFallback(unittest.TestCase):
    def test_max_lab_plus_one_is_what_keeps_the_mask_interior_off_label_4(self):
        """Run the reproduction against the real ants.joint_label_fusion.

        In a subprocess on purpose: tests/test_wrapper.py installs a MagicMock
        at sys.modules['ants'] when it is imported, so anything sharing this
        interpreter may be talking to the mock rather than to ANTsPy.
        """
        self.assertTrue(DIAG.exists(), f"missing diagnostic: {DIAG}")
        proc = subprocess.run(
            [sys.executable, str(DIAG)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=900,
        )
        self.assertEqual(
            proc.returncode, 0,
            "joint-fusion background behaviour changed:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        )
        self.assertIn("max_lab_plus_one=True recovers the true extent", proc.stdout)


if __name__ == "__main__":
    unittest.main()
