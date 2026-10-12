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

# Check if running in CI environment
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

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
                     BIDS_VERSION, APP_VERSION)

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

    @unittest.skipIf(IN_CI, "Skip in CI environment as it requires modifying template directory")
    def test_process_participant(self):
        """Test participant level processing with NIDM conversion"""
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.prob_threshold = 0.5
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
        
        # Create template directory structure in the project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        template_dir = data_dir / 'OASIS-30_Atropos_template'
        os.makedirs(template_dir, exist_ok=True)
        
        # Create dummy template files
        for fname in ['T_template0.nii.gz', 'T_template0_BrainCerebellum.nii.gz',
                     'T_template0_BrainCerebellumProbabilityMask.nii.gz',
                     'T_template0_BrainCerebellumExtractionMask.nii.gz']:
            with open(template_dir / fname, 'w') as f:
                f.write('dummy template')
        
        # Create Priors directory
        priors_dir = template_dir / 'Priors2'
        os.makedirs(priors_dir, exist_ok=True)
        for i in range(1, 7):
            with open(priors_dir / f'priors{i}.nii.gz', 'w') as f:
                f.write('dummy prior')

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
                self.prob_threshold = 0.5
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
                self.prob_threshold = 0.5
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
                self.prob_threshold = 0.5
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
                self.prob_threshold = 0.5
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

    @unittest.skipIf(IN_CI, "Skip in CI environment as it requires modifying template directory")
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
                self.prob_threshold = 0.5
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
        
        # Create template directory structure in the project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        template_dir = data_dir / 'OASIS-30_Atropos_template'
        os.makedirs(template_dir, exist_ok=True)
        
        # Create dummy template files
        for fname in ['T_template0.nii.gz', 'T_template0_BrainCerebellum.nii.gz',
                     'T_template0_BrainCerebellumProbabilityMask.nii.gz',
                     'T_template0_BrainCerebellumExtractionMask.nii.gz']:
            with open(template_dir / fname, 'w') as f:
                f.write('dummy template')
        
        # Create Priors directory
        priors_dir = template_dir / 'Priors2'
        os.makedirs(priors_dir, exist_ok=True)
        for i in range(1, 7):
            with open(priors_dir / f'priors{i}.nii.gz', 'w') as f:
                f.write('dummy prior')

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

if __name__ == "__main__":
    unittest.main()
