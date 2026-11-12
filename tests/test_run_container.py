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
        # Resolved: the wrapper's `cd ... && pwd` starts from bash's
        # physical working directory, so a symlink in TMPDIR (/home ->
        # /orcd/home on ORCD, /var -> /private/var on macOS) would make every
        # path it prints differ from the one given.
        self.tmp = Path(tempfile.mkdtemp()).resolve()
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

    def run_wrapper(self, image, *extra, env_extra=None, expect=0):
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
        self.assertEqual(proc.returncode, expect, proc.stderr)
        if expect:
            return proc
        return proc.stdout.splitlines()

    @staticmethod
    def mounts(argv, flag):
        return [argv[i + 1] for i, a in enumerate(argv) if a == flag]

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

    def test_every_prefix_argparse_accepts_is_mounted(self):
        """The wrapper must mount exactly the spellings the app reads as
        --nidm-input-dir. Which prefixes argparse accepts is taken from the
        app's own parser, not from a list kept here."""
        import contextlib
        import io
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        (self.nidm / "sub-01").mkdir()
        nidm = str(self.nidm)
        flag = "--nidm-input-dir"
        accepted = []
        for n in range(3, len(flag) + 1):
            prefix = flag[:n]
            argv = ["ants-nidm", str(self.bids), str(self.out), "participant", prefix, nidm]
            try:
                with patch.object(sys, "argv", argv), \
                     contextlib.redirect_stderr(io.StringIO()):
                    accepted.append(parse_arguments().nidm_input_dir == nidm)
            except SystemExit:
                accepted.append(False)
            with self.subTest(prefix=prefix):
                for form in ([prefix, "nidm"], [f"{prefix}=nidm"]):
                    argv = self.run_wrapper(self.sif, *form)
                    self.assertEqual(f"{nidm}:{nidm}:ro" in argv, accepted[-1], form)
        # A positive and a negative control: --ni resolves, --n is ambiguous.
        self.assertTrue(accepted[1])
        self.assertFalse(accepted[0])

    def test_a_nidm_dir_in_the_output_dir_is_refused(self):
        """A read-only mount of the NIDM dir over (part of) the output dir
        hides outputs that are only written after segmentation, so the GPU
        run would be lost; Docker refuses the output dir mounted twice."""
        self.out.mkdir()
        inside = self.out / "sub-01"
        inside.mkdir()
        alias = self.tmp / "alias"
        alias.symlink_to(inside)
        for image in (self.sif, "ants-nidm_bidsapp:latest"):
            for nidm in (self.out, inside, alias):
                with self.subTest(image=str(image), nidm=str(nidm)):
                    proc = self.run_wrapper(image, "--nidm-input-dir", str(nidm), expect=2)
                    self.assertIn("is the output directory or inside it", proc.stderr)
                    # Refused before the runner starts or scratch is made.
                    self.assertEqual(proc.stdout, "")
                    self.assertFalse(list(self.tmp.glob("ants-nidm-work.*")))

    def test_a_nidm_dir_outside_the_output_dir_is_mounted(self):
        """The overlap check compares whole path components: out-nidm next to
        out is not inside it, and an output dir inside the NIDM dir is
        harmless (the read-only parent mount is under the writable one)."""
        self.out.mkdir()
        sibling = self.tmp / "out-nidm"
        (sibling / "sub-01").mkdir(parents=True)
        argv = self.run_wrapper(self.sif, "--nidm-input-dir", str(sibling))
        self.assertIn(f"{sibling}:{sibling}:ro", argv)
        # The output dir inside the NIDM dir.
        (self.nidm / "sub-01").mkdir()
        self.out = self.nidm / "out"
        argv = self.run_wrapper(self.sif, "--nidm-input-dir", str(self.nidm))
        self.assertIn(f"{self.nidm}:{self.nidm}:ro", argv)
        self.assertIn(f"{self.out}:{self.out}", argv)

    def test_a_nidm_dir_in_the_bids_dir_gets_no_second_mount(self):
        """Inside <bids_dir> it is visible through the BIDS mount already (PyNIDM
        writes nidm.ttl at the BIDS root by default)."""
        bids = str(self.bids)
        (self.bids / "NIDM").mkdir()
        for image, flag in ((self.sif, "-B"), ("ants-nidm_bidsapp:latest", "-v")):
            for nidm in (bids, f"{bids}/NIDM"):
                with self.subTest(image=str(image), nidm=nidm):
                    argv = self.run_wrapper(image, "--nidm-input-dir", nidm)
                    self.assertEqual(self.mounts(argv, flag).count(f"{bids}:{bids}:ro"), 1)
                    self.assertFalse([m for m in self.mounts(argv, flag)
                                      if m.startswith(f"{bids}/")])
                    app = argv[argv.index(str(image)) + 1:]
                    self.assertEqual(app[app.index("--nidm-input-dir") + 1], nidm)

    def test_without_the_flag_only_the_standard_binds_are_added(self):
        argv = self.run_wrapper(self.sif, "--num-threads", "8")
        binds = [argv[i + 1] for i, a in enumerate(argv) if a == "-B"]
        self.assertEqual(len(binds), 3, binds)
        app = argv[argv.index(str(self.sif)) + 1:]
        self.assertNotIn("--nidm-input-dir", app)
        self.assertEqual(app.count("--num-threads"), 1)


if __name__ == "__main__":
    unittest.main()
