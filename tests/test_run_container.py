"""run_container.sh, run against fake apptainer/docker binaries that print
the command line they were given."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "run_container.sh"

FAKE = """#!/bin/bash
for a in "$@"; do printf '%s\\n' "$a"; done
"""


class TestRunContainer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name in ("apptainer", "docker"):
            fake = self.bin / name
            fake.write_text(FAKE)
            fake.chmod(0o755)
        self.bids = self.tmp / "bids"
        self.bids.mkdir()
        self.out = self.tmp / "out"
        self.nidm = self.tmp / "nidm"
        self.nidm.mkdir()
        self.sif = self.tmp / "app.sif"
        self.sif.write_text("")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_wrapper(self, image, *extra, env_extra=None):
        # lmod's BASH_ENV and exported functions would redefine commands in
        # the non-interactive bash; keep only what the wrapper needs.
        env = {k: v for k, v in os.environ.items()
               if k not in ("BASH_ENV", "CDPATH") and not k.startswith("BASH_FUNC_")}
        env["PATH"] = f"{self.bin}:/usr/bin:/bin"
        env["TMPDIR"] = str(self.tmp)
        env.update(env_extra or {})
        proc = subprocess.run(
            ["bash", str(SCRIPT), str(image), str(self.bids), str(self.out), "01", *extra],
            cwd=self.tmp, env=env, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.splitlines()

    def test_an_empty_bids_or_output_dir_is_rejected(self):
        """`cd ""` succeeds and stays put, so an empty BIDS_DIR (an unset
        variable) became the directory the wrapper was started in before
        the app could reject it; a blank OUTPUT_DIR was created there."""
        start = self.tmp / "start"
        env = {k: v for k, v in os.environ.items()
               if k not in ("BASH_ENV", "CDPATH") and not k.startswith("BASH_FUNC_")}
        env["PATH"] = f"{self.bin}:/usr/bin:/bin"
        env["TMPDIR"] = str(self.tmp)
        cases = [("bids_dir", empty, str(self.out)) for empty in ("", "   ")]
        cases += [("output_dir", str(self.bids), empty) for empty in ("", "   ")]
        for name, bids, out in cases:
            with self.subTest(name=name, bids=bids, out=out):
                shutil.rmtree(start, ignore_errors=True)
                start.mkdir()
                proc = subprocess.run(
                    ["bash", str(SCRIPT), str(self.sif), bids, out, "01"],
                    cwd=start, env=env, capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn(f"{name} must not be empty", proc.stderr)
                # The fake runner prints its arguments: no output, no run.
                self.assertEqual(proc.stdout, "")
                self.assertEqual(os.listdir(start), [])
                self.assertFalse(self.out.exists())

    def test_apptainer_mounts_the_nidm_dir_read_only(self):
        """The app rejects a --nidm-input-dir it cannot see, so the wrapper
        has to mount it; a relative path is made absolute first."""
        nidm = str(self.nidm)
        # argparse accepts unambiguous prefixes, so the wrapper must too.
        for flag in (["--nidm-input-dir", "nidm"], ["--nidm-input-dir=nidm"],
                     ["--nidm", "nidm"], ["--nidm-input=nidm"]):
            with self.subTest(flag=flag):
                argv = self.run_wrapper(self.sif, "--verbose", *flag, "--method", "quick")
                i = argv.index(f"{nidm}:{nidm}:ro")
                self.assertEqual(argv[i - 1], "-B")
                self.assertLess(i, argv.index(str(self.sif)), "bind must precede the image")
                app = argv[argv.index(str(self.sif)) + 1:]
                self.assertEqual(app[app.index("--nidm-input-dir") + 1], nidm)
                self.assertIn("--verbose", app)
                self.assertEqual(app[app.index("--method") + 1], "quick")

    def test_an_exported_cdpath_does_not_redirect_the_nidm_dir(self):
        """cd follows CDPATH for a relative name and prints what it chose,
        which would bind another directory under a two-line path."""
        elsewhere = self.tmp / "elsewhere"
        (elsewhere / "nidm").mkdir(parents=True)
        nidm = str(self.nidm)
        argv = self.run_wrapper(self.sif, "--nidm-input-dir", "nidm",
                                env_extra={"CDPATH": str(elsewhere)})
        self.assertIn(f"{nidm}:{nidm}:ro", argv)
        app = argv[argv.index(str(self.sif)) + 1:]
        self.assertEqual(app[app.index("--nidm-input-dir") + 1], nidm)

    def test_a_flag_that_only_starts_like_it_is_passed_through(self):
        argv = self.run_wrapper(self.sif, "--nidm-input-dirs", "nidm")
        self.assertFalse([a for a in argv if a.endswith(":ro") and a.startswith(str(self.nidm))])
        app = argv[argv.index(str(self.sif)) + 1:]
        self.assertEqual(app[app.index("--nidm-input-dirs") + 1], "nidm")

    def test_docker_mounts_the_nidm_dir_read_only(self):
        nidm = str(self.nidm)
        argv = self.run_wrapper("ants-nidm_bidsapp:latest", "--nidm-input-dir", nidm)
        i = argv.index(f"{nidm}:{nidm}:ro")
        self.assertEqual(argv[i - 1], "-v")
        self.assertLess(i, argv.index("ants-nidm_bidsapp:latest"))

    def test_a_missing_nidm_dir_is_passed_on_for_the_app_to_reject(self):
        argv = self.run_wrapper(self.sif, "--nidm-input-dir", "no_such_dir")
        self.assertFalse([a for a in argv if a.endswith(":ro") and "no_such_dir" in a])
        app = argv[argv.index(str(self.sif)) + 1:]
        self.assertEqual(app[app.index("--nidm-input-dir") + 1], "no_such_dir")

    def test_without_the_flag_only_the_standard_binds_are_added(self):
        argv = self.run_wrapper(self.sif, "--num-threads", "8")
        binds = [argv[i + 1] for i, a in enumerate(argv) if a == "-B"]
        self.assertEqual(len(binds), 3, binds)
        app = argv[argv.index(str(self.sif)) + 1:]
        self.assertNotIn("--nidm-input-dir", app)
        self.assertEqual(app.count("--num-threads"), 1)


if __name__ == "__main__":
    unittest.main()
