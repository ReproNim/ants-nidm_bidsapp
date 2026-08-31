import csv
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Check if running in CI environment
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Import numpy and nibabel for test image creation
import numpy as np
import nibabel as nib

# Mock ants module
mock_ants = MagicMock()
sys.modules['ants'] = mock_ants

# Configure ANTs mock
def mock_image_read(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

mock_ants.image_read = mock_image_read
mock_ants.n4_bias_field_correction = mock_image_read
mock_ants.get_mask = mock_image_read
mock_ants.mask_image = mock_image_read

# Import after mocking
from src.antspy.wrapper import ANTsSegmentation

@unittest.skipIf(IN_CI, "Skip in CI environment as it requires template directory")
class TestANTsSegmentation(unittest.TestCase):
    """Test cases for the ANTs segmentation wrapper"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.bids_dir = self.temp_dir / "bids"
        self.output_dir = self.temp_dir / "derivatives"
        self.temp_path = self.temp_dir / "tmp"
        
        # Create directories
        self.bids_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # Mock _load_templates method
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates') as mock_load:
            mock_load.return_value = {
                'T_template': mock_image_read(),
                'BrainCerebellum': mock_image_read(),
                'ProbabilityMask': mock_image_read(),
                'ExtractionMask': mock_image_read(),
                'priors': [mock_image_read() for _ in range(6)]
            }
            
            # Initialize segmenter
            self.segmenter = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                temp_dir=str(self.temp_path),
                modality='T1w',
                prob_threshold=0.5,
                num_threads=1,
                verbose=True
            )
    
    def tearDown(self):
        """Clean up test fixtures after each test"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of the ANTsSegmentation class"""
        # Check that the instance was created correctly
        self.assertEqual(str(self.segmenter.bids_dir), str(self.bids_dir))
        self.assertEqual(str(self.segmenter.output_dir), str(self.output_dir))
        self.assertEqual(str(self.segmenter.temp_dir), str(self.temp_path))
        self.assertEqual(self.segmenter.modality, 'T1w')
        self.assertEqual(self.segmenter.prob_threshold, 0.5)
        self.assertEqual(self.segmenter.num_threads, 1)
    
    def test_load_image(self):
        """Test loading an image"""
        # Create a test image file
        test_file = self.temp_dir / "test.nii.gz"
        test_file.write_text("dummy data")
            
        # Test loading the image
        result = self.segmenter.load_image(str(test_file))
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'numpy'))
        self.assertTrue(hasattr(result, 'spacing'))
    
    def test_preprocess_image(self):
        """Test preprocessing an image"""
        # Create a mock input image
        mock_image = mock_image_read()
        
        # Test preprocessing
        result = self.segmenter.preprocess_image(mock_image)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'numpy'))
        self.assertTrue(hasattr(result, 'spacing'))
    
    def test_organize_bids_output_single_session(self):
        """Test organizing outputs in BIDS format for single session"""
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 1, 2, 2, 2])  # 2 voxels of label 1, 3 voxels of label 2
        mock_seg.spacing = [1.0, 1.0, 1.0]  # 1mm isotropic

        # Create mock probability images
        mock_prob1 = MagicMock()
        mock_prob1.numpy.return_value = np.array([0.8, 0.2, 0.1, 0.1, 0.1, 0.1])
        mock_prob1.spacing = [1.0, 1.0, 1.0]

        mock_prob2 = MagicMock()
        mock_prob2.numpy.return_value = np.array([0.1, 0.7, 0.8, 0.1, 0.1, 0.1])
        mock_prob2.spacing = [1.0, 1.0, 1.0]

        mock_prob3 = MagicMock()
        mock_prob3.numpy.return_value = np.array([0.1, 0.1, 0.1, 0.8, 0.8, 0.8])
        mock_prob3.spacing = [1.0, 1.0, 1.0]

        # 'fusion' returns one posterior per anatomical label, plus the label
        # numbers those posteriors belong to.
        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob1, mock_prob2, mock_prob3],
            'segmentation_numbers': [1, 2, 24],
        }

        with patch('src.antspy.wrapper.ants.image_write') as mock_write:
            # Call the method
            result = self.segmenter._organize_bids_output(segmentation_results, "01")

            # Check directory structure
            subject_dir = self.output_dir / "sub-01"
            anat_dir = subject_dir / "anat"
            stats_dir = subject_dir / "stats"
            
            self.assertTrue(anat_dir.exists())
            self.assertTrue(stats_dir.exists())

            # Check stats files exist and have correct format (with subject prefix)
            labelstats_file = stats_dir / "sub-01_antslabelstats.csv"
            brainvols_file = stats_dir / "sub-01_antsbrainvols.csv"
            
            self.assertTrue(labelstats_file.exists())
            self.assertTrue(brainvols_file.exists())

            # Verify labelstats.csv content
            with open(labelstats_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["Label", "VolumeInVoxels", "Volume_mm3"])
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                volumes = {row["Label"]: int(float(row["VolumeInVoxels"])) for row in rows}
                self.assertEqual(volumes["1"], 2)
                self.assertEqual(volumes["2"], 3)

            # Verify brainvols.csv content. BVOL only: CSFVOL/GMVOL/WMVOL used to
            # be derived from probabilityimages[0..2] as if they were CSF/GM/WM
            # tissue posteriors, but under 'fusion' those are per-label
            # posteriors and under 'quick' they do not exist -- so the columns
            # were wrong in every path that produced them.
            with open(brainvols_file, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                self.assertEqual(fieldnames, {"BVOL"})
                self.assertFalse(fieldnames & {"CSFVOL", "GMVOL", "WMVOL"},
                                 "tissue volumes must not be faked from label posteriors")
                rows = list(reader)
                self.assertEqual(len(rows), 1)

            # Check that image_write was called for all images
            self.assertEqual(mock_write.call_count, 4)  # 1 segmentation + 3 probability maps

            # Probability maps are named by anatomical label, not list position
            written = [str(call.args[1]) for call in mock_write.call_args_list]
            for label_id in (1, 2, 24):
                self.assertTrue(
                    any(f"label-{label_id}_probseg" in w for w in written),
                    f"no probseg written for label {label_id}: {written}")

    def test_cortical_thickness_keeps_invtransforms_for_template_to_subject(self):
        """TemplateToSubjectTransforms must be invtransforms of subject->template.

        Regression test. compute_cortical_thickness registers subject -> template
        (fixed=brain_template, moving=brain_image), so fwdtransforms maps
        subject -> template. The 'quick' method pulls the template-space label
        atlas into subject space with these transforms; using fwdtransforms
        warped everything the wrong way (the original 11.5 L whole-FOV bug on
        ABIDE sub-0051456).
        """
        registrations, applied, subprocesses, results = \
            self._run_cortical_thickness_recording_calls()

        INV = ["/fake/0GenericAffine.mat", "/fake/1InverseWarp.nii.gz"]
        FWD = ["/fake/1Warp.nii.gz", "/fake/0GenericAffine.mat"]
        self.assertEqual(results["TemplateToSubjectTransforms"], INV,
                         "template->subject direction requires invtransforms")
        self.assertNotEqual(results["TemplateToSubjectTransforms"], FWD)

    def _run_cortical_thickness_recording_calls(self):
        """Run compute_cortical_thickness with mocks that tag every image read
        with its source path and record registration/apply_transforms calls
        and the antsBrainExtraction.sh subprocess invocation.

        Returns (registrations, applied, subprocesses, results).
        """
        def fake_image_read(path, *args, **kwargs):
            img = mock_image_read()
            img.src_path = str(path)
            return img

        registrations = []

        def fake_registration(*args, **kwargs):
            registrations.append(kwargs)
            return {
                "fwdtransforms": ["/fake/1Warp.nii.gz", "/fake/0GenericAffine.mat"],
                "invtransforms": ["/fake/0GenericAffine.mat", "/fake/1InverseWarp.nii.gz"],
                "warpedmovout": mock_image_read(),
            }

        applied = []

        def fake_apply_transforms(*args, **kwargs):
            applied.append(kwargs)
            out = mock_image_read()
            # Propagate the source tag so assertions can trace warped images
            moving = kwargs.get("moving")
            out.src_path = getattr(moving, "src_path", None)
            return out

        subprocesses = []

        def fake_subprocess_run(cmd, *args, **kwargs):
            cmd = [str(c) for c in cmd]
            subprocesses.append(cmd)
            # Honour the script's contract: create the output mask the wrapper
            # will check for, next to the -o prefix.
            if any("antsBrainExtraction" in c for c in cmd) and "-o" in cmd:
                prefix = cmd[cmd.index("-o") + 1]
                Path(prefix + "BrainExtractionMask.nii.gz").touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch('src.antspy.wrapper.ants.image_read', side_effect=fake_image_read), \
             patch('src.antspy.wrapper.ants.image_write', side_effect=lambda img, path: None), \
             patch('src.antspy.wrapper.ants.n4_bias_field_correction', side_effect=lambda img, **k: mock_image_read()), \
             patch('src.antspy.wrapper.ants.registration', side_effect=fake_registration), \
             patch('src.antspy.wrapper.ants.apply_transforms', side_effect=fake_apply_transforms), \
             patch('src.antspy.wrapper.ants.threshold_image', side_effect=lambda img, *a, **k: img), \
             patch('src.antspy.wrapper.ants.iMath', side_effect=lambda img, *a, **k: img), \
             patch('src.antspy.wrapper.subprocess.run', side_effect=fake_subprocess_run), \
             patch('src.antspy.wrapper.shutil.which', return_value='/opt/ants/bin/antsBrainExtraction.sh'):
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        return registrations, applied, subprocesses, results

    def test_brain_extraction_runs_canonical_ants_script(self):
        """Brain extraction must be antsBrainExtraction.sh, not a hand-rolled copy.

        Regression test. Two successive re-implementations of the extraction
        (fix 4f57540, then 3f51754) each fixed one failure mode and left
        another: the aug28 production run still delivered 11/33 subjects with
        2.05-3.87 L masks on hard anatomies, because the hand-rolled pipeline
        lacks the script's SyN stage and Atropos K=3 refinement. The canonical
        script is the published method the OASIS-30 template kit was built
        for, with each template file in its designed role:
          -e whole-head template, -m brain probability mask,
          -f extraction (registration-scope) mask.
        """
        registrations, applied, subprocesses, results = \
            self._run_cortical_thickness_recording_calls()

        abe = [c for c in subprocesses if any("antsBrainExtraction" in a for a in c)]
        self.assertEqual(len(abe), 1, f"expected one antsBrainExtraction.sh call, got {subprocesses}")
        cmd = abe[0]

        def arg_of(flag):
            self.assertIn(flag, cmd, f"missing {flag} in {cmd}")
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(arg_of("-d"), "3")
        self.assertTrue(arg_of("-e").endswith("T_template0.nii.gz"),
                        "-e must be the whole-head template")
        self.assertIn("ProbabilityMask", arg_of("-m"),
                      "-m must be the brain probability mask")
        self.assertIn("ExtractionMask", arg_of("-f"),
                      "-f must be the extraction registration mask")

        # The returned mask must be the script's output product
        self.assertIn("BrainExtractionMask.nii.gz",
                      results["BrainExtractionMask"].src_path,
                      "BrainExtractionMask must be read back from the script's output")

    def test_brain_extraction_raises_when_script_produces_no_mask(self):
        """A silent antsBrainExtraction.sh failure must become a loud error.

        Regression test. The script can fail and still exit 0 -- observed with
        a missing `bc` dependency: it printed "we cant find the bc program"
        to stdout, produced nothing, and returned success, so the failure only
        surfaced later as an unrelated file-not-found from image_read. The
        wrapper must verify the output mask exists and raise a RuntimeError
        carrying the script's own output.
        """
        def fake_subprocess_run(cmd, *args, **kwargs):
            # Exit 0 but produce no output file
            return MagicMock(returncode=0,
                             stdout="we cant find the bc program", stderr="")

        with patch('src.antspy.wrapper.ants.image_write', side_effect=lambda img, path: None), \
             patch('src.antspy.wrapper.ants.n4_bias_field_correction', side_effect=lambda img, **k: mock_image_read()), \
             patch('src.antspy.wrapper.subprocess.run', side_effect=fake_subprocess_run), \
             patch('src.antspy.wrapper.shutil.which', return_value='/opt/ants/bin/antsBrainExtraction.sh'):
            with self.assertRaises(RuntimeError) as ctx:
                self.segmenter.compute_cortical_thickness(mock_image_read())

        self.assertIn("bc program", str(ctx.exception),
                      "the script's own output must be in the error message")

    def test_refinement_registrations_target_brain_only_template(self):
        """Subject->template registrations must target the brain-only template.

        These run on the skull-stripped image (to produce the template-to-
        subject transforms the 'quick' method needs), so the fixed image must
        be T_template0_BrainCerebellum. Registering a brain-only image onto
        the whole-head T_template0 biases the affine to scale the brain
        toward the head outline.
        """
        registrations, applied, subprocesses, results = \
            self._run_cortical_thickness_recording_calls()

        self.assertGreaterEqual(len(registrations), 2)
        for i, reg in enumerate(registrations):
            fixed_src = reg["fixed"].src_path
            self.assertIn(
                "BrainCerebellum", fixed_src,
                f"registration #{i} must target the brain-only template, got {fixed_src}")

    def test_organize_bids_output_with_session(self):
        """Test organizing outputs in BIDS format with session"""
        # Create mock segmentation with minimal data
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': []
        }

        with patch('src.antspy.wrapper.ants.image_write') as mock_write:
            # Call the method with session
            result = self.segmenter._organize_bids_output(segmentation_results, "01", "01")

            # Check session directory structure
            session_dir = self.output_dir / "sub-01" / "ses-01"
            anat_dir = session_dir / "anat"
            stats_dir = session_dir / "stats"
            
            self.assertTrue(anat_dir.exists())
            self.assertTrue(stats_dir.exists())

            # Check segmentation file naming
            seg_file = anat_dir / "sub-01_ses-01_space-orig_dseg.nii.gz"
            self.assertTrue(mock_write.call_args_list[0][0][1].endswith(str(seg_file)))

            # Check stats files with subject/session prefix
            labelstats_file = stats_dir / "sub-01_ses-01_antslabelstats.csv"
            brainvols_file = stats_dir / "sub-01_ses-01_antsbrainvols.csv"

            self.assertTrue(labelstats_file.exists())
            self.assertTrue(brainvols_file.exists())

            with open(labelstats_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["Label", "VolumeInVoxels", "Volume_mm3"])

            with open(brainvols_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertIn("BVOL", reader.fieldnames)

    def test_save_results(self):
        """Test save_results method with output directory handling"""
        # Create mock segmentation
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': []
        }

        # Test with default output directory
        with patch('ants.image_write'):
            result = self.segmenter.save_results(segmentation_results, "01")
            self.assertIn('brain_volume', result)
            self.assertIn('label_stats', result)
            self.assertIn('brain_vols', result)
            self.assertIn('segmentation', result)

        # Test with custom output directory
        custom_output = self.temp_dir / "custom_output"
        custom_output.mkdir(parents=True, exist_ok=True)
        
        with patch('ants.image_write'):
            result = self.segmenter.save_results(segmentation_results, "01", output_dir=str(custom_output))
            self.assertTrue(all(str(custom_output) in str(path) for path in result.values() if isinstance(path, str)))

    def test_run_subject(self):
        """Test run_subject method with BIDS structure"""
        # Create BIDS directory structure
        subject_dir = self.bids_dir / "sub-01" / "anat"
        subject_dir.mkdir(parents=True, exist_ok=True)
        t1_file = subject_dir / "sub-01_T1w.nii.gz"
        t1_file.write_text("dummy data")

        # Mock the necessary methods
        with patch('ants.image_read') as mock_read, \
             patch.object(self.segmenter, 'segment_image') as mock_segment, \
             patch.object(self.segmenter, 'save_results') as mock_save:

            # Configure mocks
            mock_read.return_value = MagicMock()
            mock_segment.return_value = {
                'segmentation': MagicMock(),
                'probabilityimages': []
            }
            mock_save.return_value = {
                'brain_volume': 1000.0,
                'label_stats': 'path/to/stats.csv',
                'brain_vols': 'path/to/vols.csv',
                'segmentation': 'path/to/seg.nii.gz'
            }

            # Run the method
            result = self.segmenter.run_subject("sub-01")

            # Verify the results
            self.assertTrue(result)
            mock_read.assert_called_once()
            mock_segment.assert_called_once()
            mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main() 
