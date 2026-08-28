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

    def test_cortical_thickness_uses_invtransforms_for_template_to_subject(self):
        """The final brain mask must come template -> subject, i.e. invtransforms.

        Regression test. compute_cortical_thickness registers subject -> template
        (fixed=brain_template, moving=brain_image), so fwdtransforms maps
        subject -> template. Pulling the template-space brain mask into
        subject space with fwdtransforms produced a mask covering the whole FOV,
        so joint label fusion labelled air, skull and neck -- 100% of voxels
        labelled, no background, and an 11.5 L brain volume on ABIDE sub-0051456.
        """
        FWD = ["/fake/subj_to_tmpl_1Warp.nii.gz", "/fake/subj_to_tmpl_0GenericAffine.mat"]
        INV = ["/fake/subj_to_tmpl_0GenericAffine.mat", "/fake/subj_to_tmpl_1InverseWarp.nii.gz"]

        def fake_registration(*args, **kwargs):
            return {
                "fwdtransforms": list(FWD),
                "invtransforms": list(INV),
                "warpedmovout": mock_image_read(),
            }

        applied = []

        def fake_apply_transforms(*args, **kwargs):
            applied.append(kwargs)
            return mock_image_read()

        # Patch the name the module under test resolves, not this file's mock
        # object: tests/test_run.py also assigns sys.modules['ants'], so which
        # mock src.antspy.wrapper.ants points at depends on import order.
        with patch('src.antspy.wrapper.ants.registration', side_effect=fake_registration), \
             patch('src.antspy.wrapper.ants.apply_transforms', side_effect=fake_apply_transforms), \
             patch('src.antspy.wrapper.ants.threshold_image', side_effect=lambda *a, **k: mock_image_read()), \
             patch('src.antspy.wrapper.ants.iMath', side_effect=lambda img, *a, **k: img):
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        # Last apply_transforms call is the extraction mask -> subject space
        self.assertGreaterEqual(len(applied), 1)
        mask_call = applied[-1]
        self.assertEqual(
            mask_call["transformlist"], INV,
            "extraction mask must be warped with invtransforms (template -> subject)")
        self.assertNotEqual(
            mask_call["transformlist"], FWD,
            "using fwdtransforms warps the mask the wrong way and yields a full-FOV mask")
        # whichtoinvert must stay unset so ANTsPy infers (True, False) itself
        self.assertIsNone(mask_call.get("whichtoinvert"))

        # The stored template->subject transforms are the same ordered list
        self.assertEqual(results["TemplateToSubjectTransforms"], INV)

    def _run_cortical_thickness_recording_calls(self):
        """Run compute_cortical_thickness with mocks that tag every image read
        with its source path and record all registration/apply_transforms calls.

        Returns (registrations, applied, results).
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

        def fake_imath(img, *args, **kwargs):
            return img

        def fake_threshold(img, *args, **kwargs):
            out = mock_image_read()
            out.src_path = getattr(img, "src_path", None)
            return out

        with patch('src.antspy.wrapper.ants.image_read', side_effect=fake_image_read), \
             patch('src.antspy.wrapper.ants.n4_bias_field_correction', side_effect=lambda img, **k: mock_image_read()), \
             patch('src.antspy.wrapper.ants.registration', side_effect=fake_registration), \
             patch('src.antspy.wrapper.ants.apply_transforms', side_effect=fake_apply_transforms), \
             patch('src.antspy.wrapper.ants.threshold_image', side_effect=fake_threshold), \
             patch('src.antspy.wrapper.ants.iMath', side_effect=fake_imath):
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        return registrations, applied, results

    def test_final_brain_mask_derived_from_probability_mask(self):
        """The final brain mask must come from the brain probability mask.

        Regression test. T_template0_BrainCerebellumExtractionMask is the
        generous registration-scope mask of the OASIS-30 kit (5.9 L in template
        space -- antsBrainExtraction.sh's -f argument), not a brain mask. Using
        it as the final mask labelled ~5.6 L of head on ABIDE sub-0051456. The
        brain mask is the warped ProbabilityMask thresholded at 0.5 (1.34 L).
        """
        registrations, applied, results = self._run_cortical_thickness_recording_calls()

        self.assertGreaterEqual(len(applied), 2)
        final_mask_src = applied[-1]["moving"].src_path
        self.assertIn(
            "ProbabilityMask", final_mask_src,
            f"final brain mask must be warped from the ProbabilityMask, got {final_mask_src}")
        self.assertNotIn(
            "ExtractionMask", final_mask_src,
            "ExtractionMask is a registration-scope mask (5.9 L), not a brain mask")
        self.assertIn(
            "ProbabilityMask", results["BrainExtractionMask"].src_path,
            "returned BrainExtractionMask must trace back to the ProbabilityMask")

    def test_initial_registration_is_head_to_head_with_extraction_metric_mask(self):
        """The initial registration must match like content with like.

        Regression test. The old init registered the whole-head subject image
        onto the brain-only BrainCerebellum template with a rigid-only
        transform and no metric mask. That problem is bistable across CPU
        types: identical code and input produced a 1.70 L mask on node1406,
        2.84 L on node2906 and 3.25 L on node2000. Canonical
        antsBrainExtraction.sh semantics instead: register whole-head to the
        whole-head T_template0, restrict the metric with the ExtractionMask
        (that file's actual job), and use an affine so head size differences
        are absorbed by scaling, not by the mask.
        """
        registrations, applied, results = self._run_cortical_thickness_recording_calls()

        self.assertGreaterEqual(len(registrations), 2)
        init = registrations[0]
        fixed_src = init["fixed"].src_path
        self.assertTrue(
            fixed_src.endswith("T_template0.nii.gz"),
            f"init registration must target the whole-head template, got {fixed_src}")
        self.assertIn(
            "Affine", init.get("type_of_transform", ""),
            "init registration needs scaling (affine), rigid cannot absorb head size")
        mask = init.get("mask")
        self.assertIsNotNone(mask, "init registration must restrict its metric with a mask")
        self.assertIn(
            "ExtractionMask", mask.src_path,
            f"the metric mask must be the ExtractionMask, got {mask.src_path}")

    def test_refinement_registrations_target_brain_only_template(self):
        """Affine/SyN refinement must target the brain-only template.

        The subject image is skull-stripped before these registrations, so the
        fixed image must be T_template0_BrainCerebellum. Registering a
        brain-only image onto the whole-head T_template0 biases the affine to
        scale the brain toward the head outline, inflating the warped mask.
        """
        registrations, applied, results = self._run_cortical_thickness_recording_calls()

        self.assertGreaterEqual(len(registrations), 3)
        for i, reg in enumerate(registrations[1:], start=1):
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
