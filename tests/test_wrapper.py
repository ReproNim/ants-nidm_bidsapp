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
from src.registration.device import DeviceUnavailable

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
            # device='cpu' is deliberate and load-bearing, not a default worth
            # "helpfully" removing: these tests exercise segmentation logic on
            # mocks and never touch a GPU, while the production default is
            # 'cuda' (see ANTsSegmentation.__init__ and resolve_device). The
            # cuda-without-GPU fail-fast contract itself is covered directly by
            # test_device_cuda_without_gpu_raises_before_any_processing below
            # and by tests/test_device.py.
            self.segmenter = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                temp_dir=str(self.temp_path),
                modality='T1w',
                prob_threshold=0.5,
                num_threads=1,
                device='cpu',
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
        # The brain extraction mask covers all 6 voxels; only 5 carry a label.
        mock_mask = MagicMock()
        mock_mask.numpy.return_value = np.ones(6)
        mock_mask.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob1, mock_prob2, mock_prob3],
            'segmentation_numbers': [1, 2, 24],
            'BrainExtractionMask': mock_mask,
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

            # Verify brainvols.csv content. No CSFVOL/GMVOL/WMVOL: they used to
            # be derived from probabilityimages[0..2] as if they were CSF/GM/WM
            # tissue posteriors, but under 'fusion' those are per-label
            # posteriors and under 'quick' they do not exist -- so the columns
            # were wrong in every path that produced them.
            #
            # BVOL is the brain extraction mask volume, as in ANTs'
            # antsCorticalThickness.sh. LABELVOL is the sum of the labels, which
            # BVOL used to be: the label set has no cerebral white matter, so
            # that sum is about half a brain.
            with open(brainvols_file, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                self.assertEqual(fieldnames, {"BVOL", "LABELVOL"})
                self.assertFalse(fieldnames & {"CSFVOL", "GMVOL", "WMVOL"},
                                 "tissue volumes must not be faked from label posteriors")
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]["BVOL"]), 6.0)
                self.assertEqual(float(rows[0]["LABELVOL"]), 5.0)
            self.assertEqual(result['brain_volume'], 6.0)
            self.assertEqual(result['label_volume_sum'], 5.0)

            # Check that image_write was called for all images
            self.assertEqual(mock_write.call_count, 4)  # 1 segmentation + 3 probability maps

            # Probability maps are named by anatomical label, not list position
            written = [str(call.args[1]) for call in mock_write.call_args_list]
            for label_id in (1, 2, 24):
                self.assertTrue(
                    any(f"label-{label_id}_probseg" in w for w in written),
                    f"no probseg written for label {label_id}: {written}")

    def test_cortical_thickness_returns_a_template_warp_not_transform_paths(self):
        """The 11.5 L regression guard, restated for the new representation.

        The old assertion compared TemplateToSubjectTransforms against
        invtransforms. That list no longer exists: the warp is a torch object
        that is never serialised, so there is no fwd/inv attribute to pick
        wrongly. What this test pins is that the old, footgun-shaped key is
        gone and the single-direction object is present. The behavioural
        direction check lives in
        tests/test_registration.py::test_template_warp_moves_template_space_into_subject_space.
        """
        sentinel = object()

        def fake_subprocess_run(cmd, *args, **kwargs):
            cmd = [str(c) for c in cmd]
            # Honour antsBrainExtraction.sh's contract: create the output mask
            # next to the -o prefix, on disk, so the wrapper's own mask.exists()
            # check passes honestly rather than via a blanket Path.exists patch
            # (which would also paper over the template-file existence checks
            # elsewhere in this method).
            if any("antsBrainExtraction" in c for c in cmd) and "-o" in cmd:
                prefix = cmd[cmd.index("-o") + 1]
                Path(prefix + "BrainExtractionMask.nii.gz").touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.antspy.wrapper.register_to_template", return_value=sentinel), \
             patch("src.antspy.wrapper.ants.image_write", side_effect=lambda img, path: None), \
             patch("src.antspy.wrapper.ants.n4_bias_field_correction",
                   side_effect=lambda img, **k: mock_image_read()), \
             patch("src.antspy.wrapper.shutil.which", return_value="/fake/antsBrainExtraction.sh"), \
             patch("src.antspy.wrapper.subprocess.run", side_effect=fake_subprocess_run):
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        self.assertIs(results["TemplateWarp"], sentinel)
        self.assertNotIn(
            "TemplateToSubjectTransforms", results,
            "the ANTs transform-path list must be gone, not merely unused: "
            "leaving it invites a caller to read the wrong direction again",
        )

    def _run_cortical_thickness_recording_calls(self):
        """Run compute_cortical_thickness with mocks that tag every image read
        with its source path and record register_to_template calls and the
        antsBrainExtraction.sh subprocess invocation.

        Returns (registrations, subprocesses, results).
        """
        def fake_image_read(path, *args, **kwargs):
            img = mock_image_read()
            img.src_path = str(path)
            return img

        registrations = []

        def fake_register_to_template(subject, template, **kwargs):
            registrations.append({"subject": subject, "template": template, **kwargs})
            warp = MagicMock()
            # Propagate the source tag so assertions can trace which template
            # file the warp was built from.
            warp.src_path = getattr(template, "src_path", None)
            return warp

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
             patch('src.antspy.wrapper.register_to_template', side_effect=fake_register_to_template), \
             patch('src.antspy.wrapper.subprocess.run', side_effect=fake_subprocess_run), \
             patch('src.antspy.wrapper.shutil.which', return_value='/opt/ants/bin/antsBrainExtraction.sh'):
            results = self.segmenter.compute_cortical_thickness(mock_image_read())

        return registrations, subprocesses, results

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
        registrations, subprocesses, results = \
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
        """register_to_template must target the brain-only template.

        This registers the skull-stripped subject image (to produce the
        TemplateWarp the 'quick' method needs), so the template argument must
        be T_template0_BrainCerebellum. Registering a brain-only image onto
        the whole-head T_template0 biases the affine to scale the brain
        toward the head outline.
        """
        registrations, subprocesses, results = \
            self._run_cortical_thickness_recording_calls()

        self.assertEqual(len(registrations), 1)
        template_src = registrations[0]["template"].src_path
        self.assertIn(
            "BrainCerebellum", template_src,
            f"register_to_template must target the brain-only template, got {template_src}")

    def test_organize_bids_output_with_session(self):
        """Test organizing outputs in BIDS format with session"""
        # Create mock segmentation with minimal data
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [],
            'BrainExtractionMask': mock_seg,
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

    def test_organize_bids_output_requires_the_brain_mask(self):
        """BVOL comes from the brain extraction mask. Without it the only
        volume left is the label sum, which is about half a brain; writing
        that, or dropping BVOL silently, would both pass downstream unnoticed."""
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        with patch('src.antspy.wrapper.ants.image_write'):
            with self.assertRaises(ValueError) as ctx:
                self.segmenter._organize_bids_output(
                    {'segmentation': mock_seg, 'probabilityimages': []}, "01")
        self.assertIn("BrainExtractionMask", str(ctx.exception))

    def test_label_stats_keep_every_structure(self):
        """Only background is left out of the label stats. Ids 23, 75 and 76
        are Right Accumbens and the basal forebrain in the Neuromorphometrics
        protocol that --method fusion emits; an old filter dropped them as
        FreeSurfer reference lines and retired LUT entries."""
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 23, 30, 75, 76, 76])
        mock_seg.spacing = [1.0, 1.0, 1.0]
        mock_mask = MagicMock()
        mock_mask.numpy.return_value = np.ones(6)

        with patch('src.antspy.wrapper.ants.image_write'):
            self.segmenter._organize_bids_output(
                {'segmentation': mock_seg, 'probabilityimages': [],
                 'BrainExtractionMask': mock_mask}, "01")

        with open(self.output_dir / "sub-01" / "stats" / "sub-01_antslabelstats.csv",
                  newline='') as f:
            volumes = {r["Label"]: int(float(r["VolumeInVoxels"])) for r in csv.DictReader(f)}
        self.assertEqual(volumes, {"23": 1, "30": 1, "75": 1, "76": 2})

    def test_save_results(self):
        """Test save_results method with output directory handling"""
        # Create mock segmentation
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [],
            'BrainExtractionMask': mock_seg,
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

    def test_fusion_delegates_registration_to_warp_atlases(self):
        """The twenty-atlas ants.registration loop is gone; segment_image calls
        warp_atlases once and passes its output straight to joint_label_fusion.
        """
        from unittest.mock import patch

        warped_imgs = [mock_image_read() for _ in range(2)]
        warped_labs = [mock_image_read() for _ in range(2)]

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = (warped_imgs, warped_labs)
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        self.assertEqual(mock_warp.call_count, 1, "warp_atlases must be called once")
        self.assertIs(mock_jlf.call_args.kwargs["atlas_list"], warped_imgs)
        self.assertIs(mock_jlf.call_args.kwargs["label_list"], warped_labs)

    def test_only_quick_registers_to_the_template(self):
        """The subject-to-template warp is used by 'quick' alone. Under
        'fusion' it was one more GPU registration whose result stayed on the
        device through warp_atlases and could fail the subject."""
        from unittest.mock import patch

        for method, wanted in (("fusion", False), ("quick", True)):
            with patch.object(self.segmenter, "compute_cortical_thickness",
                              side_effect=RuntimeError("stop")) as mock_ct:
                with self.assertRaises(RuntimeError):
                    self.segmenter.segment_image(mock_image_read(), method=method)
            self.assertEqual(mock_ct.call_args.kwargs.get("register_template"), wanted, method)

    def test_preprocessing_without_template_registration(self):
        """register_template=False runs N4 and brain extraction only."""
        from unittest.mock import patch

        def fake_subprocess_run(cmd, *args, **kwargs):
            cmd = [str(c) for c in cmd]
            if any("antsBrainExtraction" in c for c in cmd) and "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1] + "BrainExtractionMask.nii.gz").touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.antspy.wrapper.register_to_template") as mock_reg, \
             patch("src.antspy.wrapper.ants.image_write", side_effect=lambda img, path: None), \
             patch("src.antspy.wrapper.ants.n4_bias_field_correction",
                   side_effect=lambda img, **k: mock_image_read()), \
             patch("src.antspy.wrapper.shutil.which", return_value="/fake/antsBrainExtraction.sh"), \
             patch("src.antspy.wrapper.subprocess.run", side_effect=fake_subprocess_run):
            results = self.segmenter.compute_cortical_thickness(
                mock_image_read(), register_template=False)

        mock_reg.assert_not_called()
        self.assertNotIn("TemplateWarp", results)
        self.assertIn("BrainExtractionMask", results)

    def test_fusion_asks_for_an_explicit_background_class(self):
        """ants.joint_label_fusion has no background class of its own. The
        brain mask is far larger than the union of the labelled structures, so
        without max_lab_plus_one every masked-but-unlabelled voxel is forced
        onto a real label, and an argmax over near-zero probabilities picks the
        lowest id -- 4, the third ventricle. The delivered tier-2 output for
        sub-0051456 called 89924 voxels third ventricle against ~700 in the
        atlases. See scripts/diag_jlf_background_fallback.py.
        """
        from unittest.mock import patch

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = ([], [])
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        kwargs = mock_jlf.call_args.kwargs
        self.assertTrue(
            kwargs["max_lab_plus_one"],
            "max_lab_plus_one must stay True or unlabelled mask interior lands on label 4",
        )
        # no_zeroes would also empty the band, but by constraining the solution
        # to voxels non-zero in every atlas, which erodes real structures.
        self.assertFalse(kwargs["no_zeroes"])

    def test_fusion_uses_scale_invariant_patch_similarity(self):
        """The N4 target (in-mask median ~388) and the atlas brains (~1100)
        are on different intensity scales. With mean squared difference the
        patch search moves each atlas's labels toward tissue of matching raw
        intensity, so WM-like structures shrink (brain stem 0.38x FreeSurfer,
        cerebellar WM 0.20-0.21x) and GM/CSF grow (lateral ventricles 3.2x).
        Correlation is scale-invariant. On 6 Caltech subjects usecor=True
        raised mean Dice with FreeSurfer from 0.36-0.40 to 0.69-0.76, above
        r_search=0 and target rescaling on every subject.
        See scripts/diag_jlf_validation_summary.py.
        """
        from unittest.mock import patch

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = ([], [])
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        kwargs = mock_jlf.call_args.kwargs
        self.assertTrue(kwargs["usecor"],
                        "mean squared difference across mismatched intensity scales mis-sizes labels by tissue")
        # The search is kept: with correlation it helps (ventricles, caudate)
        # instead of relocating labels by raw intensity.
        self.assertEqual(kwargs["r_search"], 3)

    def test_fusion_passes_engine_options_through(self):
        from unittest.mock import patch

        self.segmenter.device = "cpu"
        self.segmenter.fireants_batch_size = 4
        self.segmenter.label_interpolation = "nearest"

        with patch("src.antspy.wrapper.warp_atlases") as mock_warp, \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_warp.return_value = ([], [])
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        kwargs = mock_warp.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["label_interpolation"], "nearest")

    def test_quick_applies_warp_with_correct_argument_order(self):
        """segment_image(method='quick') must call apply_warp as
        apply_warp(warp, moving=template_label_image, reference=BrainSegmentationN4).

        Discriminating: apply_warp's signature is
        apply_warp(warp, moving, reference, *, ...). A call-count-only check
        would still pass if moving and reference were transposed, or if the
        wrong image (e.g. BrainExtractionMask, or the template itself) were
        threaded through instead of BrainSegmentationN4 -- exactly the class
        of bug (wrong direction / wrong image into a warp call) that produced
        the 11.5 L regression this task's brief is about. This test pins the
        identity of each positional argument, so a moving/reference swap or a
        wrong-image substitution fails it, and asserts moving is not
        reference so a "pass the same object twice" bug cannot slip through
        either.
        """
        fake_labels_path = self.temp_dir / "fake_template_labels.nii.gz"
        fake_labels_path.touch()
        self.segmenter.template_labels_path = fake_labels_path

        template_warp = object()
        template_label_image = mock_image_read()
        n4_image = mock_image_read()
        mask_image = mock_image_read()
        warped = mock_image_read()
        label_map = object()

        with patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.image_read", return_value=template_label_image), \
             patch("src.antspy.wrapper.build_label_map", return_value=label_map), \
             patch("src.antspy.wrapper.apply_warp", return_value=warped) as mock_apply_warp:
            mock_ct.return_value = {
                "BrainSegmentationN4": n4_image,
                "BrainExtractionMask": mask_image,
                "TemplateWarp": template_warp,
            }

            result = self.segmenter.segment_image(mock_image_read(), method="quick")

        self.assertEqual(mock_apply_warp.call_count, 1)
        args, kwargs = mock_apply_warp.call_args
        self.assertIs(args[0], template_warp,
                      "1st positional arg to apply_warp must be the TemplateWarp")
        self.assertIs(args[1], template_label_image,
                      "2nd positional arg (moving) must be the template-label "
                      "image, not the subject-space N4 image or mask")
        self.assertIs(args[2], n4_image,
                      "3rd positional arg (reference) must be "
                      "BrainSegmentationN4, not the template or the mask")
        self.assertIsNot(
            args[1], args[2],
            "moving and reference must be distinct images; a "
            "moving/reference swap that happened to pass the same object "
            "twice must still be caught")
        self.assertIs(kwargs.get("label_map"), label_map)
        self.assertIs(result["segmentation"], warped)

    def test_quick_builds_label_map_from_template_labels_path(self):
        """build_label_map must be called with the template labels path --
        the single pre-computed DKT31 labels-in-template-space file -- not
        the per-atlas labels_dir used by the 'fusion' method or any other
        path.

        Discriminating: passing the wrong path list would still produce *a*
        LabelMap and let segment_image return normally, so a test that only
        checks the method succeeds would not catch it. This pins the exact
        argument build_label_map receives.
        """
        fake_labels_path = self.temp_dir / "fake_template_labels.nii.gz"
        fake_labels_path.touch()
        self.segmenter.template_labels_path = fake_labels_path

        with patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.image_read", return_value=mock_image_read()), \
             patch("src.antspy.wrapper.build_label_map", return_value=object()) as mock_build_label_map, \
             patch("src.antspy.wrapper.apply_warp", return_value=mock_image_read()):
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }

            self.segmenter.segment_image(mock_image_read(), method="quick")

        mock_build_label_map.assert_called_once_with([fake_labels_path])

    def test_quick_raises_when_template_warp_missing(self):
        """A missing TemplateWarp must raise before apply_warp is ever
        called, rather than letting None (or some stale value) flow into a
        warp application.

        Discriminating: a guard that checked the wrong key, or that only
        logged a warning instead of raising, would let segment_image proceed
        to call apply_warp with a broken warp -- which is exactly how a
        wrongly-aligned parcellation would surface silently. This test
        requires both the ValueError and that apply_warp was never reached.
        """
        fake_labels_path = self.temp_dir / "fake_template_labels.nii.gz"
        fake_labels_path.touch()
        self.segmenter.template_labels_path = fake_labels_path

        with patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.apply_warp") as mock_apply_warp:
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                # TemplateWarp intentionally absent
            }

            with self.assertRaises(ValueError) as ctx:
                self.segmenter.segment_image(mock_image_read(), method="quick")

        self.assertIn("TemplateWarp", str(ctx.exception))
        mock_apply_warp.assert_not_called()

    def test_temp_dir_defaults_to_tmpdir_when_no_explicit_path_is_given(self):
        """The container sets TMPDIR=/work and run_container.sh plus both
        sbatch scripts bind a writable directory there. Nothing read it, so
        ~1-2 GB of per-subject registration scratch went to the output tree
        instead (spec section 4: every crossing goes through a file on
        TMPDIR)."""
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates'), \
             patch.dict(os.environ, {'TMPDIR': str(self.temp_dir / 'work')}):
            seg = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                modality='T1w',
                device='cpu',
            )
        assert str(seg.temp_dir).startswith(str(self.temp_dir / 'work')), (
            f"temp_dir {seg.temp_dir} is not under TMPDIR"
        )
        self.assertFalse(
            str(seg.temp_dir).startswith(str(self.output_dir)),
            "scratch must not land in the output tree, which BABS zips",
        )

    def test_tmpdir_without_room_falls_back_to_the_output_tree(self):
        """Under BABS's --writable-tmpfs, with nothing bound over /work,
        TMPDIR is a 64 MiB overlay on this cluster and registration scratch
        (1-2 GB) fails with ENOSPC partway through a subject."""
        small = shutil._ntuple_diskusage(total=64 << 20, used=0, free=64 << 20)
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates'), \
             patch('src.antspy.wrapper.shutil.disk_usage', return_value=small), \
             patch.dict(os.environ, {'TMPDIR': str(self.temp_dir / 'work')}):
            try:
                seg = ANTsSegmentation(bids_dir=str(self.bids_dir), output_dir=str(self.output_dir),
                                       modality='T1w', device='cpu')
                # ants.joint_label_fusion writes ~100 probability images with
                # mktemp, so the process temp root has to move as well.
                self.assertEqual(tempfile.gettempdir(), str(self.output_dir / 'tmp'))
            finally:
                tempfile.tempdir = None
        self.assertEqual(seg.temp_dir, self.output_dir / 'tmp')

    def test_unwritable_tmpdir_falls_back_to_the_output_tree(self):
        """Plain `apptainer run` with no /work bind leaves /work read-only."""
        ro = self.temp_dir / 'ro'
        ro.mkdir()
        ro.chmod(0o555)
        try:
            with patch('src.antspy.wrapper.ANTsSegmentation._load_templates'), \
                 patch.dict(os.environ, {'TMPDIR': str(ro)}):
                seg = ANTsSegmentation(bids_dir=str(self.bids_dir), output_dir=str(self.output_dir),
                                       modality='T1w', device='cpu')
                self.assertEqual(tempfile.gettempdir(), str(self.output_dir / 'tmp'))
        finally:
            ro.chmod(0o755)
            tempfile.tempdir = None
        self.assertEqual(seg.temp_dir, self.output_dir / 'tmp')

    def test_explicit_temp_dir_beats_tmpdir(self):
        """A caller who passes temp_dir means it."""
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates'), \
             patch.dict(os.environ, {'TMPDIR': str(self.temp_dir / 'work')}):
            seg = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                temp_dir=str(self.temp_path),
                modality='T1w',
                device='cpu',
            )
        self.assertEqual(str(seg.temp_dir), str(self.temp_path))

    def test_temp_dir_falls_back_to_the_output_tree_without_tmpdir(self):
        env = {k: v for k, v in os.environ.items() if k != 'TMPDIR'}
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates'), \
             patch.dict(os.environ, env, clear=True):
            seg = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                modality='T1w',
                device='cpu',
            )
        self.assertEqual(str(seg.temp_dir), str(self.output_dir / 'tmp'))

    def test_scratch_directory_is_removed_and_temp_dir_survives(self):
        """Cleanup must never reach a directory the caller supplied -- only the
        mkdtemp child this class created."""
        marker = self.temp_path / 'caller_owned.txt'
        marker.write_text('do not delete me')

        with self.segmenter._scratch('fireants_') as scratch:
            self.assertTrue(scratch.is_dir())
            self.assertEqual(scratch.parent, self.temp_path)
            (scratch / 'warped_atlas_00.nii.gz').write_text('x')
            captured = scratch

        self.assertFalse(captured.exists(), "per-run scratch was not removed")
        self.assertTrue(self.temp_path.is_dir(), "temp_dir itself must survive")
        self.assertTrue(marker.exists(), "caller-owned contents must survive")

    def test_scratch_is_removed_even_when_the_body_raises(self):
        with self.assertRaises(RuntimeError):
            with self.segmenter._scratch('fireants_') as scratch:
                captured = scratch
                raise RuntimeError("registration folded")
        self.assertFalse(captured.exists())

    def test_fusion_scratch_is_a_private_subdirectory_that_is_cleaned_up(self):
        """warp_atlases must be handed a directory of our own under temp_dir,
        and that directory must be gone once segment_image returns."""
        from unittest.mock import patch

        seen = {}

        def record(*args, **kwargs):
            seen['temp_dir'] = Path(kwargs['temp_dir'])
            seen['existed'] = seen['temp_dir'].is_dir()
            return ([], [])

        with patch("src.antspy.wrapper.warp_atlases", side_effect=record), \
             patch.object(self.segmenter, "compute_cortical_thickness") as mock_ct, \
             patch("src.antspy.wrapper.ants.joint_label_fusion") as mock_jlf:
            mock_ct.return_value = {
                "BrainSegmentationN4": mock_image_read(),
                "BrainExtractionMask": mock_image_read(),
                "TemplateWarp": object(),
            }
            mock_jlf.return_value = {
                "segmentation": mock_image_read(),
                "intensity": mock_image_read(),
                "probabilityimages": [],
                "segmentation_numbers": [1],
            }
            self.segmenter.segment_image(mock_image_read(), method="fusion")

        self.assertTrue(seen['existed'], "scratch must exist during the call")
        self.assertEqual(seen['temp_dir'].parent, self.temp_path)
        self.assertNotEqual(seen['temp_dir'], self.temp_path,
                            "must not hand out temp_dir itself")
        self.assertFalse(seen['temp_dir'].exists(),
                         "per-subject warped volumes (~1-2 GB) were not cleaned up")

    def test_device_cuda_without_gpu_raises_before_any_processing(self):
        """Constructing with device='cuda' on a GPU-less host must fail at
        construction time, before templates or images are touched.

        This is the fail-fast contract the whole FireANTs design depends on:
        a BABS array has to discover a missing GPU immediately, not after N4
        bias correction has already burned twenty minutes per job. The CUDA
        probe is patched (not relying on this host's actual GPU state) so the
        test is deterministic on GPU and GPU-less machines alike.
        """
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates') as mock_load, \
             patch('src.registration.device._torch_cuda_available', return_value=False):
            with self.assertRaises(DeviceUnavailable):
                ANTsSegmentation(
                    bids_dir=str(self.bids_dir),
                    output_dir=str(self.output_dir),
                    temp_dir=str(self.temp_path),
                    modality='T1w',
                    prob_threshold=0.5,
                    num_threads=1,
                    device='cuda',
                    verbose=True
                )

        # The raise must happen before any template loading -- not merely
        # somewhere during construction.
        mock_load.assert_not_called()

if __name__ == "__main__":
    unittest.main()
