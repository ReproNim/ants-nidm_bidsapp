import os
import tempfile
import shutil
import unittest
import logging
from unittest.mock import patch, MagicMock, PropertyMock
import sys
from pathlib import Path
import json
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# Mock the external dependencies
mock_ants = MagicMock()

# Configure ANTs mock
def mock_image_read(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

def mock_n4_correction(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

def mock_registration(*args, **kwargs):
    return {
        'warpedmovout': mock_image_read(),
        'fwdtransforms': ['transform1', 'transform2'],
        'invtransforms': ['transform3', 'transform4']
    }

def mock_apply_transforms(*args, **kwargs):
    return mock_image_read()

def mock_get_mask(*args, **kwargs):
    return mock_image_read()

mock_ants.image_read = mock_image_read
mock_ants.n4_bias_field_correction = mock_n4_correction
mock_ants.registration = mock_registration
mock_ants.apply_transforms = mock_apply_transforms
mock_ants.get_mask = mock_get_mask
mock_ants.joint_label_fusion.return_value = {
    'segmentation': mock_image_read(),
    'intensity': mock_image_read(),
    'probabilityimages': [mock_image_read() for _ in range(3)],
    'segmentation_numbers': [1, 2, 3]
}

# Mock BIDSLayout
mock_layout = MagicMock()
mock_layout.get_subjects.return_value = ['01']
mock_layout.get_sessions.return_value = ['01']

# Mock bids module
mock_bids = MagicMock()
mock_bids.BIDSLayout = MagicMock(return_value=mock_layout)

# Configure the mocks
sys.modules['ants'] = mock_ants
sys.modules['numpy'] = np
sys.modules['nibabel'] = MagicMock()
sys.modules['bids'] = mock_bids

# Import after mocking
from src.antspy.inputs import AnatChoice
from src.run import (process_participant, process_session, main, nidm_conversion,
                     create_dataset_description, subject_output_dir,
                     find_nidm_input_file, save_processing_summary,
                     BIDS_VERSION, APP_VERSION, CONVERTER_BOOTSTRAP, CONVERTER_MODULE)

class TestRun(unittest.TestCase):
    """Test cases for the run module"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_dir = os.path.join(self.temp_dir, "bids")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create directories
        os.makedirs(self.bids_dir)
        os.makedirs(self.output_dir)
        
        # Create a simple BIDS structure
        self.create_bids_structure()
        
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        """Clean up test fixtures after each test"""
        shutil.rmtree(self.temp_dir)
    
    def create_bids_structure(self):
        """Create a minimal BIDS directory structure for testing"""
        # Create dataset_description.json
        os.makedirs(os.path.join(self.bids_dir, "sub-01", "anat"))
        with open(os.path.join(self.bids_dir, "dataset_description.json"), "w") as f:
            f.write('{"Name": "Test Dataset", "BIDSVersion": "1.7.0"}')
        
        # Create a dummy T1w file
        with open(os.path.join(self.bids_dir, "sub-01", "anat", 
                              "sub-01_T1w.nii.gz"), "w") as f:
            f.write("dummy data")

    def create_derivatives_structure(self):
        """Create a minimal derivatives structure for testing NIDM conversion.

        Per-subject layout: the derivative root *is* output_dir, and everything
        for a subject lives under output_dir/sub-<id>/[ses-<x>/].
        """
        derivatives_dir = self.output_dir
        subject_dir = os.path.join(derivatives_dir, "sub-01", "ses-01")
        anat_dir = os.path.join(subject_dir, "anat")
        stats_dir = os.path.join(subject_dir, "stats")

        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        # Create required files with subject/session prefix
        seg_file = os.path.join(anat_dir, "sub-01_ses-01_space-orig_dseg.nii.gz")
        labelstats_file = os.path.join(stats_dir, "sub-01_ses-01_antslabelstats.csv")
        brainvols_file = os.path.join(stats_dir, "sub-01_ses-01_antsbrainvols.csv")
        
        # Write dummy content
        with open(seg_file, 'wb') as f:
            f.write(b'dummy segmentation')  # Write as binary
        
        with open(labelstats_file, 'w') as f:
            f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
        
        with open(brainvols_file, 'w') as f:
            f.write("Name,Value\nBVOL,3000.0\nCSFVOL,1000.0\nGMVOL,1000.0\nWMVOL,1000.0\n")
        
        return derivatives_dir

    def test_nidm_conversion(self):
        """Test NIDM conversion with proper file structure"""
        # Create derivatives structure
        derivatives_dir = self.create_derivatives_structure()
        subject_dir = os.path.join(derivatives_dir, "sub-01", "ses-01")
        
        # Run NIDM conversion
        with patch('subprocess.run') as mock_run:
            # Configure mock to return success
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Success"
            mock_process.stderr = ""
            mock_run.return_value = mock_process
            
            # subprocess is mocked, so create the file the real converter would
            def write_nidm(*a, **kw):
                with open(os.path.join(subject_dir, "nidm.ttl"), "w") as f:
                    f.write("@prefix nidm: <http://purl.org/nidash/nidm#> .\n")
                return mock_process
            mock_run.side_effect = write_nidm

            result = nidm_conversion(
                self.logger,
                derivatives_dir,
                "01",
                bids_session="01",  # Pass session to match test file structure
                verbose=True
            )

            # Check results
            self.assertTrue(result)
            # NIDM lands in the subject's own directory, named nidm.ttl
            self.assertTrue(os.path.exists(os.path.join(subject_dir, "nidm.ttl")))
            
            # Verify subprocess call
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            
            # Verify all paths in command are strings
            self.assertTrue(all(isinstance(arg, str) for arg in cmd_args))
            
            # Verify required arguments are present
            self.assertIn("-f", cmd_args)
            self.assertIn("-subjid", cmd_args)
            # TTL format is default, so -j flag should NOT be present
            self.assertNotIn("-j", cmd_args)
            # ants_seg_to_nidm has no -session option; passing it would abort
            self.assertNotIn("-session", cmd_args)
            # Output is the per-subject nidm.ttl
            self.assertEqual(cmd_args[cmd_args.index("-o") + 1],
                             os.path.abspath(os.path.join(subject_dir, "nidm.ttl")))
            
            # Verify file paths exist
            paths_str = [arg for arg in cmd_args if ".csv" in arg or ".nii.gz" in arg][0]
            paths = paths_str.split(",")
            self.assertEqual(len(paths), 3)  # Should have 3 paths
            for path in paths:
                self.assertTrue(os.path.exists(path))

    def _run_conversion_capturing_cmd(self, method):
        """Run nidm_conversion with subprocess mocked and return the argv it built."""
        derivatives_dir = self.create_derivatives_structure()
        subject_dir = os.path.join(derivatives_dir, "sub-01", "ses-01")
        with patch('subprocess.run') as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Success"
            mock_process.stderr = ""

            def write_nidm(*a, **kw):
                with open(os.path.join(subject_dir, "nidm.ttl"), "w") as f:
                    f.write("@prefix nidm: <http://purl.org/nidash/nidm#> .\n")
                return mock_process
            mock_run.side_effect = write_nidm

            result = nidm_conversion(
                self.logger, derivatives_dir, "01",
                bids_session="01", verbose=False, method=method,
            )
        self.assertTrue(result)
        mock_run.assert_called_once()
        return mock_run.call_args[0][0]

    def test_fusion_is_named_with_the_neuromorphometrics_protocol(self):
        """fusion votes over the OASIS-TRT-20 DKT31_CMA atlases, whose
        subcortical ids are Neuromorphometrics/BrainCOLOR. Naming them with the
        FreeSurfer table is what put Left-Porg and Right-undetermined in
        delivered NIDM, and reported the right thalamus as
        Right-Substancia-Nigra."""
        cmd = self._run_conversion_capturing_cmd("fusion")
        self.assertIn("--label-protocol", cmd)
        self.assertEqual(cmd[cmd.index("--label-protocol") + 1], "neuromorphometrics")

    def test_quick_is_named_with_the_freesurfer_protocol(self):
        """quick warps a template map already numbered in FreeSurfer aseg ids."""
        cmd = self._run_conversion_capturing_cmd("quick")
        self.assertIn("--label-protocol", cmd)
        self.assertEqual(cmd[cmd.index("--label-protocol") + 1], "freesurfer")

    def test_unknown_method_falls_back_without_the_flag(self):
        """An unregistered method must not silently claim a protocol; it gets
        the converter's own FreeSurfer default and a warning."""
        cmd = self._run_conversion_capturing_cmd("some-future-method")
        self.assertNotIn("--label-protocol", cmd)

    def test_every_cli_method_has_a_protocol(self):
        """A new --method choice that nobody maps here would be named with the
        wrong table by default, silently."""
        from src.run import LABEL_PROTOCOL_BY_METHOD, parse_arguments
        import argparse as _argparse

        # Read the --method choices off the real parser rather than restating
        # them, so adding a method without registering its protocol fails here.
        holder = {}
        real_add = _argparse.ArgumentParser.add_argument

        def capture(parser_self, *a, **kw):
            if "--method" in a and "choices" in kw:
                holder["choices"] = set(kw["choices"])
            return real_add(parser_self, *a, **kw)

        argv = sys.argv
        sys.argv = ["ants-nidm", "bids", "out", "participant"]
        try:
            with patch.object(_argparse.ArgumentParser, "add_argument", capture):
                try:
                    parse_arguments()
                except SystemExit:
                    pass
        finally:
            sys.argv = argv

        self.assertTrue(holder.get("choices"), "could not read --method choices")
        self.assertEqual(set(LABEL_PROTOCOL_BY_METHOD), holder["choices"])

    def _run_conversion(self, returncode=0, stderr="", stdout="", **kwargs):
        """Run nidm_conversion with the converter mocked; return (result, argv, log)."""
        derivatives_dir = self.create_derivatives_structure()
        subject_dir = os.path.join(derivatives_dir, "sub-01", "ses-01")
        with patch('subprocess.run') as mock_run:
            mock_process = MagicMock()
            mock_process.returncode = returncode
            mock_process.stdout = stdout
            mock_process.stderr = stderr

            def convert(*a, **kw):
                if returncode == 0:
                    with open(os.path.join(subject_dir, "nidm.ttl"), "w") as f:
                        f.write("@prefix nidm: <http://purl.org/nidash/nidm#> .\n")
                return mock_process
            mock_run.side_effect = convert

            with self.assertLogs(self.logger, level="INFO") as log:
                result = nidm_conversion(self.logger, derivatives_dir, "01",
                                         bids_session="01", method="fusion", **kwargs)
        mock_run.assert_called_once()
        return result, mock_run.call_args[0][0], log.output

    def _input_nidm(self):
        path = os.path.join(self.temp_dir, "input_nidm.ttl")
        with open(path, "w") as f:
            f.write("@prefix nidm: <http://purl.org/nidash/nidm#> .\n")
        return path

    def test_converter_is_told_which_t1w_was_segmented(self):
        """With an input NIDM the converter links the stats to the acquisition
        whose file name is the T1w the app segmented; with several anatomical
        acquisitions and no name it links none."""
        t1w = Path(self.bids_dir) / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_run-2_T1w.nii.gz"
        result, cmd, _ = self._run_conversion(nidm_input_file=self._input_nidm(),
                                              input_file=t1w)
        self.assertTrue(result)
        self.assertIn("-acq_filename", cmd)
        self.assertEqual(cmd[cmd.index("-acq_filename") + 1], "sub-01_ses-01_run-2_T1w.nii.gz")
        self.assertIn("--nidm", cmd)

    def _odd_paths(self):
        """An input NIDM and a T1w whose paths hold spaces, quotes and '#'."""
        odd_dir = Path(self.temp_dir) / "in put's \"dir\" #2"
        odd_dir.mkdir()
        nidm = odd_dir / "nidm input.ttl"
        nidm.write_text("@prefix nidm: <http://purl.org/nidash/nidm#> .\n")
        t1w = odd_dir / "sub-01_ses-01_T1w (copy).nii.gz"
        return nidm, t1w

    def _logged_command(self, log):
        lines = [line.split("Running command: ", 1)[1] for line in log
                 if "Running command: " in line]
        self.assertEqual(len(lines), 1, log)
        return lines[0]

    def test_logged_command_splits_back_into_the_exact_argv(self):
        """The log line is the argv shell-quoted: the -c bootstrap and paths
        with spaces, quotes and '#' survive a shlex round trip."""
        import shlex
        nidm, t1w = self._odd_paths()
        result, cmd, log = self._run_conversion(nidm_input_file=str(nidm), input_file=t1w)
        self.assertTrue(result)
        self.assertEqual(shlex.split(self._logged_command(log)), cmd)

    def _replay_helper(self):
        import importlib.util
        path = Path(project_root) / "scripts" / "replay_nidm_from_log.py"
        spec = importlib.util.spec_from_file_location("replay_nidm_from_log", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_replay_helper_recovers_the_converter_argv_from_the_log(self):
        """scripts/replay_nidm_from_log.py reads the -c form the app logs:
        the converter module and exactly the arguments after the bootstrap."""
        replay = self._replay_helper()
        nidm, t1w = self._odd_paths()
        _, cmd, log = self._run_conversion(nidm_input_file=str(nidm), input_file=t1w)
        log_file = Path(self.temp_dir) / "ants.log"
        log_file.write_text("\n".join(log) + "\n")
        entries = replay._parse_log(log_file)
        self.assertEqual(len(entries), 1)
        module, args = replay._split_command(entries[0]["command"])
        self.assertEqual(module, CONVERTER_MODULE)
        self.assertEqual(args, cmd[3:])

    def test_replay_helper_refuses_c_code_that_is_not_the_converter(self):
        replay = self._replay_helper()
        with self.assertRaises(ValueError):
            replay._split_command("python3 -c 'print(1)' -f a,b,c")
        # Logs from before 0.2.3 used -m.
        self.assertEqual(
            replay._split_command("python -m ants_seg_to_nidm.ants_seg_to_nidm -f 'a b,c' -o x"),
            ("ants_seg_to_nidm.ants_seg_to_nidm", ["-f", "a b,c", "-o", "x"]))

    def test_no_acq_filename_without_an_input_nidm(self):
        """The converter rejects -acq_filename without -n: there is no
        acquisition to link."""
        t1w = Path(self.bids_dir) / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
        result, cmd, _ = self._run_conversion(input_file=t1w)
        self.assertTrue(result)
        self.assertNotIn("-acq_filename", cmd)

    def test_no_acq_filename_without_a_known_input(self):
        result, cmd, _ = self._run_conversion(nidm_input_file=self._input_nidm())
        self.assertTrue(result)
        self.assertNotIn("-acq_filename", cmd)

    def test_converter_failure_fails_the_conversion_with_its_stderr_logged(self):
        message = "ERROR: Subject ID sub-01 was not found in the input NIDM, which has subject IDs: 02"
        result, _, log = self._run_conversion(returncode=1, stderr=message,
                                              nidm_input_file=self._input_nidm())
        self.assertFalse(result)
        self.assertTrue(any(line.startswith("ERROR:") and message in line for line in log), log)

    def test_converter_stderr_on_success_is_logged_as_warnings(self):
        """A successful ants_seg_to_nidm run reports reduced provenance (stats
        not linked to an acquisition, a new subject agent) only on stderr."""
        stderr = ("/x/ants_seg_to_nidm.py:190: UserWarning: 0 anatomical AcquisitionObjects with "
                  "file name sub-01_T1w.nii.gz found for subject sub-01; the stats are not linked\n"
                  "  warnings.warn(\n"
                  "\n"
                  "The input NIDM has no subject agents; adding a new agent for sub-01 (--forcenidm)\n")
        result, _, log = self._run_conversion(stderr=stderr, nidm_input_file=self._input_nidm())
        self.assertTrue(result)
        warnings = [line for line in log if line.startswith("WARNING:")]
        self.assertEqual(warnings, [
            "WARNING:test_logger:ants_seg_to_nidm: /x/ants_seg_to_nidm.py:190: UserWarning: 0 anatomical "
            "AcquisitionObjects with file name sub-01_T1w.nii.gz found for subject sub-01; the stats are not linked",
            "WARNING:test_logger:ants_seg_to_nidm: warnings.warn(",
            "WARNING:test_logger:ants_seg_to_nidm: The input NIDM has no subject agents; adding a new "
            "agent for sub-01 (--forcenidm)",
        ])

    def test_converter_success_without_stderr_logs_no_warning(self):
        result, _, log = self._run_conversion(nidm_input_file=self._input_nidm())
        self.assertTrue(result)
        self.assertEqual([line for line in log if line.startswith("WARNING:")], [])

    def test_converter_failure_logs_its_stdout(self):
        result, _, log = self._run_conversion(returncode=1, stderr="error: boom",
                                              stdout="Reading stats files\n",
                                              nidm_input_file=self._input_nidm())
        self.assertFalse(result)
        self.assertTrue(any(line.startswith("ERROR:") and "Reading stats files" in line
                            for line in log), log)

    def _parse_with_checked_out_converter(self, **kwargs):
        """Build the converter argv exactly as nidm_conversion does, then hand
        it to the argument parser of the converter checked out at
        src/ants_seg_to_nidm. Return the converter's parsed namespace.

        Not mocked on the converter side: an option the pinned converter does
        not accept makes its own parser exit 2, which fails the test."""
        import importlib
        import argparse as _argparse

        checkout = Path(project_root) / "src" / "ants_seg_to_nidm"
        script = checkout / "ants_seg_to_nidm" / "ants_seg_to_nidm.py"
        if not script.is_file():
            if os.environ.get("CI"):
                self.fail(f"{script} missing: CI must check out the submodule")
            self.skipTest("ants_seg_to_nidm submodule not checked out")

        derivatives_dir = self.create_derivatives_structure()
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            nidm_conversion(self.logger, derivatives_dir, "01",
                            bids_session="01", method="fusion", **kwargs)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:3], [sys.executable, "-c", CONVERTER_BOOTSTRAP])

        saved_path = list(sys.path)
        saved_modules = {name: mod for name, mod in sys.modules.items()
                         if name == "ants_seg_to_nidm" or name.startswith("ants_seg_to_nidm.")}
        for name in saved_modules:
            del sys.modules[name]
        sys.path.insert(0, str(checkout))
        try:
            converter = importlib.import_module(CONVERTER_MODULE)
            self.assertTrue(
                Path(converter.__file__).resolve().is_relative_to(checkout.resolve()),
                f"imported {converter.__file__}, not the checked-out converter")

            parsed = {}
            real_parse_args = _argparse.ArgumentParser.parse_args

            def parse_args(parser_self, args=None, namespace=None):
                parsed["args"] = real_parse_args(parser_self, args, namespace)
                return parsed["args"]

            class ArgumentsAccepted(Exception):
                pass

            def stop(*a, **kw):
                # main() calls url_validator on -f right after its own checks
                # of the parsed arguments; stop there, before any conversion.
                raise ArgumentsAccepted()

            with patch.object(_argparse.ArgumentParser, "parse_args", parse_args), \
                 patch.object(converter, "url_validator", stop), \
                 patch.object(sys, "argv", ["ants_seg_to_nidm.py", *cmd[3:]]):
                try:
                    converter.main()
                except ArgumentsAccepted:
                    pass
                except SystemExit as exc:
                    self.fail(f"the checked-out converter rejected the app's argv "
                              f"(exit {exc.code}): {cmd[3:]}")
                else:
                    self.fail("converter main() returned without reaching its input checks")
            return parsed["args"]
        finally:
            sys.path[:] = saved_path
            for name in [n for n in sys.modules
                         if n == "ants_seg_to_nidm" or n.startswith("ants_seg_to_nidm.")]:
                del sys.modules[name]
            sys.modules.update(saved_modules)

    def test_checked_out_converter_accepts_the_argv_with_an_input_nidm(self):
        t1w = Path(self.bids_dir) / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
        args = self._parse_with_checked_out_converter(nidm_input_file=self._input_nidm(),
                                                      input_file=t1w)
        self.assertEqual(args.acq_filename, ["sub-01_ses-01_T1w.nii.gz"])
        self.assertTrue(args.forcenidm)
        self.assertEqual(args.label_protocol, "neuromorphometrics")

    def test_checked_out_converter_accepts_the_argv_without_an_input_nidm(self):
        args = self._parse_with_checked_out_converter()
        self.assertIsNone(args.nidm_file)
        self.assertEqual(args.subjid, "sub-01")

    def test_successful_conversion_logs_no_runtime_warning(self):
        """Unmocked: the checked-out converter converts the example stats it
        ships. Run as `python -m ants_seg_to_nidm.ants_seg_to_nidm`, runpy
        warned on every run that the module was already in sys.modules (the
        package __init__ imports it), and the app relayed that as two WARNING
        lines in every subject's log."""
        examples = Path(project_root) / "src" / "ants_seg_to_nidm" / "examples"
        if not (examples / "antslabelstats.csv").is_file():
            if os.environ.get("CI"):
                self.fail(f"{examples} missing: CI must check out the submodule")
            self.skipTest("ants_seg_to_nidm submodule not checked out")
        subject_dir = Path(self.output_dir) / "sub-01"
        (subject_dir / "anat").mkdir(parents=True)
        (subject_dir / "stats").mkdir()
        shutil.copy(examples / "antslabelstats.csv", subject_dir / "stats" / "sub-01_antslabelstats.csv")
        # BVOL and LABELVOL only: an id the CDE table lacks would be added to
        # the converter's tracked mapping_data.
        (subject_dir / "stats" / "sub-01_antsbrainvols.csv").write_text("BVOL,LABELVOL\n1666780,1200000\n")
        shutil.copy(examples / "antsBrainSegmentation.nii.gz",
                    subject_dir / "anat" / "sub-01_space-orig_dseg.nii.gz")

        with self.assertLogs(self.logger, level="INFO") as logs:
            ok = nidm_conversion(self.logger, self.output_dir, "01", method="quick")
        warnings = [r.getMessage() for r in logs.records if r.levelno >= logging.WARNING]
        self.assertTrue(ok, warnings)
        self.assertTrue((subject_dir / "nidm.ttl").is_file())
        self.assertEqual([w for w in warnings if "RuntimeWarning" in w or "sys.modules" in w], [])

    def test_nidm_conversion_missing_files(self):
        """Test NIDM conversion with missing required files"""
        derivatives_dir = self.output_dir

        result = nidm_conversion(
            self.logger,
            derivatives_dir,
            "01",
            verbose=True
        )
        
        self.assertFalse(result)

    def test_process_participant(self):
        """Test participant level processing with NIDM conversion"""
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "quick"
                self.skip_nidm = False
                self.skip_ants = False
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 0
                self.label_interpolation = "genericlabel"
        
        args = Args(self)
        
        # Create a real numpy array for testing
        test_array = np.zeros((10, 10, 10))
        test_array[5:8, 5:8, 5:8] = 1  # Create some non-zero values
        
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = test_array
        mock_seg.spacing = [1.0, 1.0, 1.0]
        
        mock_prob = MagicMock()
        mock_prob.numpy.return_value = test_array
        mock_prob.spacing = [1.0, 1.0, 1.0]
        
        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob, mock_prob, mock_prob],
            'BrainSegmentationN4': mock_seg,
            'BrainExtractionMask': mock_seg
        }
        
        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            
            # Configure BIDSLayout mock
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            
            # Create and configure the mock segmenter instance
            mock_instance = MockSegmenter.return_value
            
            # Configure the run_subject method
            def mock_run_subject(subject_id, session_label=None, method='fusion'):
                # Create BIDS-compliant output directories with sub-*/[ses-*]/ structure
                # subject_id comes in as "sub-01", extract the label
                bids_subject = subject_id.replace('sub-', '') if subject_id.startswith('sub-') else subject_id
                seg_base = f"sub-{bids_subject}"
                subject_dir = os.path.join(self.output_dir, f'sub-{bids_subject}')
                if session_label:
                    subject_dir = os.path.join(subject_dir, f'ses-{session_label}')
                    seg_base += f"_ses-{session_label}"

                anat_dir = os.path.join(subject_dir, 'anat')
                stats_dir = os.path.join(subject_dir, 'stats')
                os.makedirs(anat_dir, exist_ok=True)
                os.makedirs(stats_dir, exist_ok=True)

                # Create dummy stats files with subject prefix
                with open(os.path.join(stats_dir, f'{seg_base}_antslabelstats.csv'), 'w') as f:
                    f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
                with open(os.path.join(stats_dir, f'{seg_base}_antsbrainvols.csv'), 'w') as f:
                    f.write("Name,Value\nBVOL,3000.0\n")

                # Create dummy segmentation file
                with open(os.path.join(anat_dir, f"{seg_base}_space-orig_dseg.nii.gz"), 'w') as f:
                    f.write("dummy segmentation")

                return True
            
            mock_instance.run_subject.side_effect = mock_run_subject
            mock_instance.segment_image.return_value = segmentation_results
            
            # Configure NIDM mock
            mock_nidm.return_value = True
            
            # Run processing
            result = process_participant(args, self.logger)
            
            # Verify results
            self.assertEqual(result, 0)
            mock_instance.run_subject.assert_called_once()
            mock_nidm.assert_called_once()

    def test_participant_level_with_session_uses_session_pipeline(self):
        """BABS always passes the positional level `participant`, even for a
        session-wise project, and supplies the session as --session-label.
        Ignoring it would process a multi-session subject as one session."""
        args = MagicMock(session_label="ses-01")

        with patch("src.run.process_session", return_value=0) as mock_session:
            result = process_participant(args, self.logger)

        self.assertEqual(result, 0)
        mock_session.assert_called_once_with(args, self.logger)

    def test_process_participant_segmentation_failure_skips_nidm(self):
        """run_subject() returning False must not fall through to a NIDM
        conversion that would report success anyway -- exactly the
        exit-0-but-wrong shape this repo has been bitten by before (see
        memory: "a NIDM app can succeed and still be wrong"). Neither
        test_process_participant (happy path) nor
        test_processing_summary_records_failure (the summary writer in
        isolation, not this dispatch) previously covered this branch.
        """
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "fusion"
                self.skip_nidm = False
                self.skip_ants = False
                self.ants_input = None
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 5
                self.label_interpolation = "genericlabel"

        args = Args(self)

        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout

            mock_instance = MockSegmenter.return_value
            mock_instance.run_subject.return_value = False
            mock_nidm.return_value = True

            result = process_participant(args, self.logger)

            self.assertEqual(result, 1)
            mock_instance.run_subject.assert_called_once()
            mock_nidm.assert_not_called()

    def _unfetched_nidm_args(self, session_label):
        """Args whose NIDM input for sub-01 is an annex symlink with no content."""
        nidm_in = Path(self.temp_dir) / "NIDM"
        target = nidm_in / "sub-01" / (f"ses-{session_label}" if session_label else "") / "nidm.ttl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to("../.git/annex/objects/missing")

        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = session_label
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "fusion"
                self.skip_nidm = False
                self.skip_ants = False
                self.ants_input = None
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = str(nidm_in)
                self.device = "cuda"
                self.fireants_batch_size = 5
                self.label_interpolation = "genericlabel"

        return Args(self)

    def test_unfetched_nidm_input_stops_both_levels_before_segmentation(self):
        """The lookup runs first, so an unfetched input fails in seconds
        rather than after a GPU segmentation; main() turns the raise into
        exit 1."""
        from src.run import NIDMInputNotRetrieved

        for level, session in ((process_participant, None), (process_session, "01")):
            with self.subTest(level=level.__name__):
                args = self._unfetched_nidm_args(session)
                with patch('src.run.ANTsSegmentation') as MockSegmenter, \
                     patch('src.run.nidm_conversion') as mock_nidm, \
                     patch('bids.BIDSLayout') as MockBIDSLayout:
                    mock_layout = MagicMock()
                    mock_layout.get_subjects.return_value = ['01']
                    mock_layout.get_sessions.return_value = ['01']
                    MockBIDSLayout.return_value = mock_layout

                    with self.assertRaises(NIDMInputNotRetrieved):
                        level(args, self.logger)

                    MockSegmenter.return_value.run_subject.assert_not_called()
                    mock_nidm.assert_not_called()

    # An input NIDM whose only subject agent has id `subject_id` (or none).
    @staticmethod
    def _input_nidm_text(subject_id):
        text = (
            "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
            "@prefix ndar: <https://ndar.nih.gov/api/datadictionary/v2/dataelement/> .\n"
            "@prefix nidm: <http://purl.org/nidash/nidm#> .\n"
            "<http://iri.nidash.org/project> a nidm:Project, prov:Activity .\n"
        )
        if subject_id is not None:
            text += (f"<http://iri.nidash.org/person> a prov:Agent, prov:Person ;"
                     f' ndar:src_subject_id "{subject_id}" .\n')
        return text

    def _run_with_input_nidm(self, level, session, subject_id, text=None):
        """Run a level for sub-01 with an input NIDM naming `subject_id` (or
        holding `text`); return (exit code, segmenter mock, nidm_conversion
        mock, error log). Every log message is kept in self.last_messages."""
        shutil.rmtree(Path(self.temp_dir) / "NIDM", ignore_errors=True)
        args = self._unfetched_nidm_args(session)
        nidm_in = Path(args.nidm_input_dir)
        shutil.rmtree(nidm_in)
        target = nidm_in / "sub-01" / (f"ses-{session}" if session else "") / "nidm.ttl"
        target.parent.mkdir(parents=True)
        target.write_text(text if text is not None else self._input_nidm_text(subject_id))
        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion', return_value=True) as mock_nidm, \
             patch('src.run.resolve_anat_choice') as mock_choice, \
             patch('src.run.save_processing_summary'), \
             patch('bids.BIDSLayout') as MockBIDSLayout, \
             self.assertLogs(self.logger, level="INFO") as logs:
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            mock_layout.get_sessions.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            MockSegmenter.return_value.run_subject.return_value = True
            mock_choice.return_value.chosen = None
            code = level(args, self.logger)
        self.last_messages = [r.getMessage() for r in logs.records]
        errors = [r.getMessage() for r in logs.records if r.levelno >= logging.ERROR]
        return code, MockSegmenter.return_value, mock_nidm, errors

    @staticmethod
    def _two_person_nidm_text(id_a, id_b):
        """Two person nodes matching sub-01; only person-b owns an anatomical
        MRI acquisition."""
        return (
            "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
            "@prefix ndar: <https://ndar.nih.gov/api/datadictionary/v2/dataelement/> .\n"
            "@prefix nidm: <http://purl.org/nidash/nidm#> .\n"
            "<http://iri.nidash.org/project> a nidm:Project, prov:Activity .\n"
            f'<http://iri.nidash.org/person-a> a prov:Agent, prov:Person ; ndar:src_subject_id "{id_a}" .\n'
            f'<http://iri.nidash.org/person-b> a prov:Agent, prov:Person ; ndar:src_subject_id "{id_b}" .\n'
            "<http://iri.nidash.org/acq> a nidm:AcquisitionObject ;\n"
            "    nidm:hadAcquisitionModality nidm:MagneticResonanceImaging ;\n"
            "    nidm:hadImageUsageType nidm:Anatomical ;\n"
            "    prov:wasGeneratedBy <http://iri.nidash.org/acq-activity> .\n"
            "<http://iri.nidash.org/acq-activity> a prov:Activity ;\n"
            "    prov:qualifiedAssociation [ a prov:Association ;"
            " prov:agent <http://iri.nidash.org/person-b> ] .\n"
        )

    def test_input_check_logs_the_node_it_chose_among_duplicates(self):
        """Two nodes store the subject's id and one owns the scan: the check
        passes and the log names the node the stats will go to."""
        for level, session in ((process_participant, None), (process_session, "01")):
            with self.subTest(level=level.__name__):
                code, segmenter, mock_nidm, errors = self._run_with_input_nidm(
                    level, session, None, text=self._two_person_nidm_text("01", "01"))
                self.assertEqual(code, 0, errors)
                segmenter.run_subject.assert_called_once()
                chosen = [m for m in self.last_messages if m.startswith("NIDM input check: ")]
                self.assertEqual(len(chosen), 1, self.last_messages)
                self.assertIn("Subject ID sub-01 matches 2 agents in the input NIDM", chosen[0])
                self.assertIn("using http://iri.nidash.org/person-b, the only one that owns "
                              "an anatomical MRI acquisition", chosen[0])

    def test_input_nidm_with_differing_stored_ids_fails_before_segmentation(self):
        """Two nodes match sub-01 only after normalisation ("01" and "sub-01");
        they may be different people, so the job fails even though exactly one
        owns the scan."""
        code, segmenter, mock_nidm, errors = self._run_with_input_nidm(
            process_participant, None, None, text=self._two_person_nidm_text("01", "sub-01"))
        self.assertEqual(code, 1)
        segmenter.run_subject.assert_not_called()
        mock_nidm.assert_not_called()
        self.assertTrue(any("their stored subject IDs differ (01, sub-01)" in e
                            for e in errors), errors)

    def test_input_nidm_without_the_subject_fails_before_segmentation(self):
        """The converter would refuse an input NIDM that holds other subjects
        but not this one; the app runs the converter's own lookup on it
        before segmenting, so the job fails in seconds, with its message."""
        for level, session in ((process_participant, None), (process_session, "01")):
            with self.subTest(level=level.__name__):
                code, segmenter, mock_nidm, errors = self._run_with_input_nidm(
                    level, session, "02")
                self.assertEqual(code, 1)
                segmenter.run_subject.assert_not_called()
                mock_nidm.assert_not_called()
                self.assertTrue(any(
                    "Subject ID sub-01 was not found in the input NIDM, which has "
                    "subject IDs: 02" in e for e in errors), errors)

    def test_input_nidm_with_the_subject_proceeds_to_segmentation(self):
        for level, session in ((process_participant, None), (process_session, "01")):
            for subject_id in ("1", "sub-01", None):  # None: no subjects, --forcenidm adds one
                with self.subTest(level=level.__name__, subject_id=subject_id):
                    code, segmenter, mock_nidm, errors = self._run_with_input_nidm(
                        level, session, subject_id)
                    self.assertEqual(code, 0, errors)
                    segmenter.run_subject.assert_called_once()
                    mock_nidm.assert_called_once()

    def test_process_participant_subject_not_found(self):
        """A participant label absent from the BIDS layout must fail fast,
        before touching the segmenter or the filesystem."""
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "99"
                self.session_label = None
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "fusion"
                self.skip_nidm = False
                self.skip_ants = False
                self.ants_input = None
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 5
                self.label_interpolation = "genericlabel"

        args = Args(self)

        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']  # 99 is not in here
            MockBIDSLayout.return_value = mock_layout
            mock_instance = MockSegmenter.return_value

            result = process_participant(args, self.logger)

            self.assertEqual(result, 1)
            mock_instance.run_subject.assert_not_called()
            mock_nidm.assert_not_called()

    def test_process_participant_skip_ants_uses_existing_derivatives(self):
        """--skip-ants must not call run_subject, and with --ants-input set
        must redirect derivatives_dir there rather than the fresh output_dir
        (this is the BABS re-run-NIDM-only path)."""
        class Args:
            def __init__(self, test_instance, ants_input):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "fusion"
                self.skip_nidm = False
                self.skip_ants = True
                self.ants_input = ants_input
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 5
                self.label_interpolation = "genericlabel"

        existing_derivatives = os.path.join(self.temp_dir, "existing_derivatives")
        os.makedirs(existing_derivatives, exist_ok=True)
        args = Args(self, ants_input=existing_derivatives)

        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            mock_layout.get.return_value = []
            MockBIDSLayout.return_value = mock_layout
            mock_instance = MockSegmenter.return_value
            mock_nidm.return_value = True

            result = process_participant(args, self.logger)

            mock_instance.run_subject.assert_not_called()
            self.assertEqual(result, 0)
            # nidm_conversion must have been pointed at ants_input, not the
            # fresh (empty) output_dir -- called with derivatives_dir as a
            # positional/keyword arg either way, so inspect the call.
            _, kwargs = mock_nidm.call_args
            self.assertEqual(str(kwargs.get("derivatives_dir")), existing_derivatives)

    def test_process_participant_skip_nidm_leaves_segmentation_result_alone(self):
        """--skip-nidm must not call nidm_conversion, and a successful
        segmentation with NIDM skipped is still reported as success -- the
        skip is a deliberate no-op, not a failure."""
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "fusion"
                self.skip_nidm = True
                self.skip_ants = False
                self.ants_input = None
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 5
                self.label_interpolation = "genericlabel"

        args = Args(self)

        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            mock_instance = MockSegmenter.return_value
            mock_instance.run_subject.return_value = True

            result = process_participant(args, self.logger)

            self.assertEqual(result, 0)
            mock_instance.run_subject.assert_called_once()
            mock_nidm.assert_not_called()

    def test_process_session(self):
        """Test session level processing with NIDM conversion"""
        # Create session structure
        session_dir = os.path.join(self.bids_dir, "sub-01", "ses-01", "anat")
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, "sub-01_ses-01_T1w.nii.gz"), "w") as f:
            f.write("dummy data")
        
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = "01"
                self.modality = "T1w"
                self.max_voxel_size = 2.0
                self.method = "quick"
                self.skip_nidm = False
                self.skip_ants = False
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
                self.nidm_input_dir = None
                self.device = "cuda"
                self.fireants_batch_size = 0
                self.label_interpolation = "genericlabel"
        
        args = Args(self)
        
        # Create a real numpy array for testing
        test_array = np.zeros((10, 10, 10))
        test_array[5:8, 5:8, 5:8] = 1  # Create some non-zero values
        
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = test_array
        mock_seg.spacing = [1.0, 1.0, 1.0]
        
        mock_prob = MagicMock()
        mock_prob.numpy.return_value = test_array
        mock_prob.spacing = [1.0, 1.0, 1.0]
        
        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob, mock_prob, mock_prob],
            'BrainSegmentationN4': mock_seg,
            'BrainExtractionMask': mock_seg
        }
        
        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            
            # Configure BIDSLayout mock
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            mock_layout.get_sessions.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            
            # Create and configure the mock segmenter instance
            mock_instance = MockSegmenter.return_value
            
            # Configure the run_subject method
            def mock_run_subject(subject_id, session_label=None, method='fusion'):
                # Create BIDS-compliant output directories with sub-*/[ses-*]/ structure
                # subject_id comes in as "sub-01", session_label as "ses-01"
                bids_subject = subject_id.replace('sub-', '') if subject_id.startswith('sub-') else subject_id
                seg_base = f"sub-{bids_subject}"
                subject_dir = os.path.join(self.output_dir, f'sub-{bids_subject}')
                if session_label:
                    bids_session = session_label.replace('ses-', '') if session_label.startswith('ses-') else session_label
                    subject_dir = os.path.join(subject_dir, f'ses-{bids_session}')
                    seg_base += f"_ses-{bids_session}"

                anat_dir = os.path.join(subject_dir, 'anat')
                stats_dir = os.path.join(subject_dir, 'stats')
                os.makedirs(anat_dir, exist_ok=True)
                os.makedirs(stats_dir, exist_ok=True)

                # Create dummy stats files with subject prefix
                with open(os.path.join(stats_dir, f'{seg_base}_antslabelstats.csv'), 'w') as f:
                    f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
                with open(os.path.join(stats_dir, f'{seg_base}_antsbrainvols.csv'), 'w') as f:
                    f.write("Name,Value\nBVOL,3000.0\n")

                # Create dummy segmentation file
                with open(os.path.join(anat_dir, f"{seg_base}_space-orig_dseg.nii.gz"), 'w') as f:
                    f.write("dummy segmentation")

                return True

            mock_instance.run_subject.side_effect = mock_run_subject
            mock_instance.segment_image.return_value = segmentation_results

            # Configure NIDM mock
            mock_nidm.return_value = True

            # Run processing
            result = process_session(args, self.logger)

            # Verify results
            self.assertEqual(result, 0)
            mock_instance.run_subject.assert_called_once_with("sub-01", "ses-01", method="quick")
            mock_nidm.assert_called_once()

    def test_bids_version_is_a_spec_version(self):
        """BIDSVersion must be a BIDS *spec* version, not a library version.

        pybids' package version is 0.x; if BIDS_VERSION ever gets sourced from a
        library again this catches it.
        """
        major, minor = BIDS_VERSION.split(".")[:2]
        self.assertGreaterEqual(int(major), 1,
                                f"BIDS_VERSION {BIDS_VERSION!r} looks like a package version")
        self.assertTrue(minor.isdigit())

    def test_create_dataset_description(self):
        """Test creation of dataset_description.json"""
        test_dir = os.path.join(self.temp_dir, "test_desc")
        os.makedirs(test_dir)

        create_dataset_description(test_dir, "1.0.0")

        # Check if file exists
        desc_file = os.path.join(test_dir, "dataset_description.json")
        self.assertTrue(os.path.exists(desc_file))

        # Check content
        with open(desc_file) as f:
            desc = json.load(f)
            self.assertEqual(desc["BIDSVersion"], BIDS_VERSION)
            self.assertEqual(desc["GeneratedBy"][0]["Version"], "1.0.0")

    def test_processing_summary_lands_inside_subject_dir(self):
        """BABS zips only sub-<id>/, so the summary must be written inside it."""
        subject_dir = subject_output_dir(self.output_dir, "01")
        path = save_processing_summary(
            logger=self.logger,
            subject_dir=subject_dir,
            bids_subject="01",
            bids_session=None,
            app_version=APP_VERSION,
            succeeded=True,
            nidm_written=True,
        )
        self.assertEqual(path, subject_dir / "processing_summary.json")
        with open(path) as f:
            summary = json.load(f)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["failure"], 0)
        self.assertEqual(summary["success_list"], ["sub-01"])
        self.assertEqual(summary["version_info"]["ants-nidm"]["version"], APP_VERSION)
        self.assertIn("ants", summary["version_info"])

    def test_processing_summary_records_failure(self):
        """A failed subject is recorded as such, with the session in its label."""
        subject_dir = subject_output_dir(self.output_dir, "01", "baseline")
        path = save_processing_summary(
            logger=self.logger,
            subject_dir=subject_dir,
            bids_subject="01",
            bids_session="baseline",
            app_version=APP_VERSION,
            succeeded=False,
            nidm_written=False,
        )
        with open(path) as f:
            summary = json.load(f)
        self.assertEqual(summary["failure_list"], ["sub-01_ses-baseline"])
        self.assertFalse(summary["nidm_written"])

    def test_processing_summary_records_the_input_used(self):
        """The segmented image and any skipped alternatives are recorded,
        relative to the BIDS root."""
        bids = Path(self.temp_dir) / "bids_inputs"
        anat = bids / "sub-01" / "ses-1" / "anat"
        anat.mkdir(parents=True)
        used = anat / "sub-01_ses-1_run-1_T1w.nii.gz"
        other = anat / "sub-01_ses-1_acq-hires_run-1_T1w.nii.gz"
        path = save_processing_summary(
            logger=self.logger,
            subject_dir=subject_output_dir(self.output_dir, "01", "1"),
            bids_subject="01", bids_session="1", app_version=APP_VERSION,
            succeeded=True, nidm_written=True,
            bids_dir=bids, failure_reason=None,
            anat_choice=AnatChoice(used, [other], [(anat / "sub-01_ses-1_acq-lo_T1w.nii.gz", (1.5, 1.5, 4.0))]),
        )
        with open(path) as f:
            summary = json.load(f)
        self.assertEqual(summary["input_file"], "sub-01/ses-1/anat/sub-01_ses-1_run-1_T1w.nii.gz")
        self.assertEqual(summary["inputs_not_used"],
                         ["sub-01/ses-1/anat/sub-01_ses-1_acq-hires_run-1_T1w.nii.gz"])
        self.assertEqual(summary["inputs_too_coarse"],
                         {"sub-01/ses-1/anat/sub-01_ses-1_acq-lo_T1w.nii.gz": [1.5, 1.5, 4.0]})

    def test_processing_summary_records_failure_reason(self):
        path = save_processing_summary(
            logger=self.logger, subject_dir=subject_output_dir(self.output_dir, "01"),
            bids_subject="01", bids_session=None, app_version=APP_VERSION,
            succeeded=False, nidm_written=False,
            failure_reason="every T1w image is coarser than 2.0 mm",
        )
        with open(path) as f:
            self.assertEqual(json.load(f)["failure_reason"], "every T1w image is coarser than 2.0 mm")

    def test_subject_output_dir(self):
        """Per-subject output dir is the zip unit: no app wrapper, no nidm/."""
        self.assertEqual(str(subject_output_dir("/out", "01")), "/out/sub-01")
        self.assertEqual(str(subject_output_dir("/out", "01", "baseline")),
                         "/out/sub-01/ses-baseline")

    def test_no_cross_subject_nidm_collision(self):
        """Two subjects must never resolve to the same NIDM output path."""
        a = subject_output_dir(self.output_dir, "01") / "nidm.ttl"
        b = subject_output_dir(self.output_dir, "02") / "nidm.ttl"
        self.assertNotEqual(a, b)

    def test_find_nidm_input_file_per_subject_nidm_ttl(self):
        """sub-<id>/nidm.ttl is the layout the shared NIDM datasets use."""
        nidm_in = Path(self.temp_dir) / "NIDM"
        (nidm_in / "sub-01").mkdir(parents=True)
        target = nidm_in / "sub-01" / "nidm.ttl"
        target.write_text("")
        (nidm_in / "nidm.ttl").write_text("")  # dataset-level fallback

        # Per-subject file wins over the dataset-level fallback
        self.assertEqual(find_nidm_input_file(nidm_in, "01"), target)
        # Unknown subject falls back to the dataset-level file
        self.assertEqual(find_nidm_input_file(nidm_in, "99"), nidm_in / "nidm.ttl")
        # Session-nested form
        sess = nidm_in / "sub-02" / "ses-baseline"
        sess.mkdir(parents=True)
        (sess / "nidm.ttl").write_text("")
        self.assertEqual(find_nidm_input_file(nidm_in, "02", "baseline"),
                         sess / "nidm.ttl")

    def test_unfetched_annex_nidm_input_is_an_error_not_a_miss(self):
        """A dangling symlink is what datalad leaves before `datalad get`.
        Falling through to the dataset-level file, or to no input at all,
        would build nidm.ttl without the subject's demographics and exit 0."""
        from src.run import NIDMInputNotRetrieved

        nidm_in = Path(self.temp_dir) / "NIDM"
        (nidm_in / "sub-01").mkdir(parents=True)
        (nidm_in / "sub-01" / "nidm.ttl").symlink_to(
            "../.git/annex/objects/missing")
        (nidm_in / "nidm.ttl").write_text("")

        with self.assertRaises(NIDMInputNotRetrieved):
            find_nidm_input_file(nidm_in, "01")

        # Only the first candidate present decides: a fetched per-subject
        # file still wins over a dangling lower-priority one.
        (nidm_in / "sub-02").mkdir()
        (nidm_in / "sub-02" / "nidm.ttl").write_text("")
        (nidm_in / "sub-02.ttl").symlink_to("missing")
        self.assertEqual(find_nidm_input_file(nidm_in, "02"),
                         nidm_in / "sub-02" / "nidm.ttl")

    def test_nidm_input_miss_is_logged_as_a_warning(self):
        import logging

        nidm_in = Path(self.temp_dir) / "NIDM"
        nidm_in.mkdir()
        logger = logging.getLogger("ants-nidm-test")

        with self.assertLogs(logger, level="WARNING") as logs:
            self.assertIsNone(
                find_nidm_input_file(nidm_in, "01", "baseline", logger=logger))
        self.assertIn("sub-01 ses-baseline", logs.output[0])

        # No input directory at all is not a miss worth warning about.
        from unittest.mock import MagicMock
        quiet = MagicMock()
        self.assertIsNone(find_nidm_input_file(None, "01", logger=quiet))
        quiet.warning.assert_not_called()

    def test_device_defaults_to_cuda(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant"]
        with patch.object(sys, "argv", argv):
            args = parse_arguments()
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.fireants_batch_size, 5)
        self.assertEqual(args.label_interpolation, "genericlabel")

    def test_device_and_engine_options_are_parsed(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = [
            "ants-nidm", "/bids", "/out", "participant",
            "--device", "cpu",
            "--fireants-batch-size", "5",
            "--label-interpolation", "nearest",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_arguments()
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.fireants_batch_size, 5)
        self.assertEqual(args.label_interpolation, "nearest")

    def test_skip_ants_requires_the_method_that_made_the_input(self):
        """--skip-ants names the structures of --ants-input with the protocol
        of --method. Defaulting to fusion would name quick derivatives (aseg
        ids) with the Neuromorphometrics table: a valid, wrong NIDM file."""
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        base = ["ants-nidm", "/bids", "/out", "participant", "--skip-ants", "--ants-input", "/in"]
        with patch.object(sys, "argv", base), self.assertRaises(SystemExit):
            parse_arguments()
        with patch.object(sys, "argv", base + ["--method", "quick"]):
            self.assertEqual(parse_arguments().method, "quick")
        with patch.object(sys, "argv", ["ants-nidm", "/bids", "/out", "participant"]):
            self.assertEqual(parse_arguments().method, "fusion")

    def test_invalid_device_is_rejected(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant", "--device", "mps"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_arguments()

    def test_negative_batch_size_is_rejected(self):
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant",
                "--fireants-batch-size", "-2"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_arguments()

    def test_prob_threshold_is_rejected(self):
        """--prob-threshold was accepted but never read by either method, so a
        value passed to it changed nothing. It is now an unknown argument."""
        import sys
        from unittest.mock import patch

        from src.run import parse_arguments

        argv = ["ants-nidm", "/bids", "/out", "participant", "--prob-threshold", "0.3"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_arguments()

    def test_anat_choice_reuses_the_segmenters_choice(self):
        """The recorded input is the one segmented, not a second choice."""
        from types import SimpleNamespace

        from src.run import resolve_anat_choice

        made = AnatChoice(Path("/bids/sub-01/anat/sub-01_T1w.nii.gz"), [], [])
        segmenter = SimpleNamespace(last_choice=made)
        args = SimpleNamespace(skip_ants=False, bids_dir="/bids", modality="T1w",
                               max_voxel_size=2.0)
        with patch("src.run.choose_anat_input", side_effect=AssertionError("chose again")):
            got = resolve_anat_choice(args, segmenter, "/out", "01", None,
                                      logging.getLogger("t"))
        self.assertIs(got, made)

    def test_skip_ants_records_the_input_from_the_sidecar(self):
        from types import SimpleNamespace

        from src.run import resolve_anat_choice

        out = Path(self.temp_dir) / "derivs"
        anat = out / "sub-01" / "ses-1" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-01_ses-1_space-orig_dseg.json").write_text(json.dumps(
            {"Sources": ["bids:raw:sub-01/ses-1/anat/sub-01_ses-1_run-2_T1w.nii.gz"]}))
        args = SimpleNamespace(skip_ants=True, bids_dir="/bids", modality="T1w",
                               max_voxel_size=2.0)
        with patch("src.run.choose_anat_input", side_effect=AssertionError("chose again")):
            got = resolve_anat_choice(args, None, out, "01", "1", logging.getLogger("t"))
        self.assertEqual(got.chosen, Path("/bids/sub-01/ses-1/anat/sub-01_ses-1_run-2_T1w.nii.gz"))

    def test_skip_ants_without_a_recorded_input_chooses_again_and_warns(self):
        from types import SimpleNamespace

        from src.run import resolve_anat_choice

        fresh = AnatChoice(None, [], [])
        args = SimpleNamespace(skip_ants=True, bids_dir="/bids", modality="T1w",
                               max_voxel_size=2.0)
        with patch("src.run.choose_anat_input", return_value=fresh) as choose, \
                self.assertLogs("t", level="WARNING"):
            got = resolve_anat_choice(args, None, Path(self.temp_dir), "01", None,
                                      logging.getLogger("t"))
        self.assertIs(got, fresh)
        choose.assert_called_once()

if __name__ == "__main__":
    unittest.main()
