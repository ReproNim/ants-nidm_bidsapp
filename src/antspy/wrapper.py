# src/ants/wrapper.py
import os
import shutil
import subprocess
import tempfile
import logging
import importlib
from contextlib import contextmanager
import ants
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from natsort import natsorted

from src.registration import apply_warp, build_label_map, register_to_template, warp_atlases
from src.registration.device import resolve_device

#: joint_label_fusion settings for --method fusion. Why max_lab_plus_one and
#: usecor are what they are is explained at the call in segment_image. Kept
#: here so validation harnesses fuse with exactly the production settings.
JLF_PARAMS = dict(
    beta=4.0,
    rad=2,
    rho=0.01,
    usecor=True,
    r_search=3,
    nonnegative=False,
    no_zeroes=False,
    max_lab_plus_one=True,
)

class ANTsSegmentation:
    """
    Wrapper class for ANTs segmentation functionality using ANTsPy.
    This class handles the preparation, execution, and post-processing
    of ANTs-based segmentation for neuroimaging data in BIDS format.
    """
    
    def __init__(self, 
                 bids_dir=None, 
                 output_dir=None,
                 temp_dir=None,
                 modality='T1w',
                 prob_threshold=0.5,
                 num_threads=1,
                 device='cuda',
                 fireants_batch_size=5,
                 label_interpolation='genericlabel',
                 verbose=False):
        """Initialize ANTs segmentation.
        
        Parameters
        ----------
        bids_dir : str or Path
            Path to BIDS directory
        output_dir : str or Path
            Path to output directory
        temp_dir : str or Path
            Path to temporary directory
        modality : str
            Modality to process (default: T1w)
        prob_threshold : float
            Probability threshold for binary mask creation
        num_threads : int
            Number of threads to use
        device : str
            Compute device for FireANTs registration ('cuda' or 'cpu')
        fireants_batch_size : int
            Atlases per FireANTs registration batch (0 = all at once). 20 at once
            exceeds 80 GB of GPU memory; 5 keeps peak use under ~20 GB
        label_interpolation : str
            Interpolation mode FireANTs uses when warping atlas labels
        verbose : bool
            Whether to print detailed logs
        """
        # Convert all path inputs to Path objects
        self.bids_dir = Path(bids_dir) if bids_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        # Scratch root. Order matters:
        #   1. an explicit temp_dir the caller passed -- never touched beyond
        #      creating per-run subdirectories inside it;
        #   2. $TMPDIR, which the container sets to the writable /work bind
        #      that run_container.sh and both sbatch scripts already mount.
        #      Nothing read it before, so ~1-2 GB of warped-atlas scratch per
        #      subject went to the output tree instead (spec section 4 says
        #      every crossing goes through a file on TMPDIR);
        #   3. output_dir/tmp, the previous behaviour, as a last resort.
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        elif os.environ.get('TMPDIR'):
            self.temp_dir = Path(os.environ['TMPDIR']) / 'ants-nidm'
        elif output_dir:
            self.temp_dir = Path(output_dir) / 'tmp'
        else:
            self.temp_dir = None
        
        # Set data paths - check both container and local paths
        # First check for container path, then local development path
        if Path('/opt/data').exists():
            base_data_dir = Path('/opt/data')
        else:
            # Check for local data directory relative to the package
            local_data_dir = Path(__file__).parent.parent.parent / 'data'
            if local_data_dir.exists():
                base_data_dir = local_data_dir
            else:
                base_data_dir = Path('/opt/data')  # Fallback to container path
        
        # OASIS-30 template might be in a subdirectory after extraction
        potential_template_dirs = [
            base_data_dir / 'OASIS-30_Atropos_template',
            base_data_dir / 'OASIS-30_Atropos_template_ants',
            base_data_dir
        ]
        self.template_dir = None
        for d in potential_template_dirs:
            if d.exists() and (d / 'T_template0.nii.gz').exists():
                self.template_dir = d
                break
        if not self.template_dir:
            self.template_dir = base_data_dir / 'OASIS-30_Atropos_template'
            
        # Atlas directories might be nested after extraction
        self.atlas_dir = base_data_dir / 'OASIS-TRT-20_brains'
        if not self.atlas_dir.exists():
            # Check if it's nested
            potential_atlas = base_data_dir / 'OASIS-TRT-20_brains' / 'OASIS-TRT-20_brains'
            if potential_atlas.exists():
                self.atlas_dir = potential_atlas
                
        self.labels_dir = base_data_dir / 'OASIS-TRT-20_DKT31_CMA_labels_v2' 
        if not self.labels_dir.exists():
            # Check if it's nested
            potential_labels = base_data_dir / 'OASIS-TRT-20_DKT31_CMA_labels_v2' / 'OASIS-TRT-20_DKT31_CMA_labels_v2'
            if potential_labels.exists():
                self.labels_dir = potential_labels
                
        self.template_labels_path = base_data_dir / 'OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz'
        
        self.modality = modality
        self.prob_threshold = prob_threshold
        self.num_threads = num_threads
        # Registration runs on FireANTs. resolve_device raises rather than
        # silently falling back to CPU: FireANTs on CPU is slower than the ANTs
        # code it replaced, so a fallback under a BABS array yields hundreds of
        # many-hour jobs that all exit 0.
        self.device = resolve_device(device)
        self.fireants_batch_size = fireants_batch_size
        self.label_interpolation = label_interpolation
        self.verbose = verbose

        # ANTsPy takes no threads argument: every ITK filter it calls reads
        # ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS from the environment. The
        # container pins that to 1 so an unconfigured run cannot oversubscribe a
        # shared node, which meant --num-threads was accepted and then ignored --
        # joint label fusion registers 20 atlases and is unusably slow at one
        # thread. Set it here so --num-threads actually takes effect.
        if self.num_threads and self.num_threads > 0:
            os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.num_threads)
        
        # Set up logging
        self.logger = logging.getLogger('ants-nidm.segmentation')
        
        # Validate required directories
        if not self.bids_dir or not self.bids_dir.exists():
            raise ValueError(f"BIDS directory not found: {self.bids_dir}")
        if not self.output_dir:
            raise ValueError("Output directory must be specified")
        if not self.template_dir.exists():
            self.logger.error(f"Template directory not found: {self.template_dir}")
            self.logger.error(f"Contents of /opt/data: {list(base_data_dir.iterdir()) if base_data_dir.exists() else 'Directory not found'}")
            raise ValueError(f"Template directory not found: {self.template_dir}")
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
        # Load templates
        self.templates = self._load_templates()
        
        # Validate atlas directories and log detailed information
        if not self.atlas_dir.exists():
            self.logger.warning(f"Atlas directory not found: {self.atlas_dir}")
            self.logger.warning("Only 'quick' segmentation method will be available.")
        else:
            atlas_files = list(self.atlas_dir.glob('*.nii.gz'))
            self.logger.info(f"Found {len(atlas_files)} atlas files in {self.atlas_dir}")
            
        if not self.labels_dir.exists():
            self.logger.warning(f"Labels directory not found: {self.labels_dir}")
            self.logger.warning("Only 'quick' segmentation method will be available.")
        else:
            label_files = list(self.labels_dir.glob('*.nii.gz'))
            self.logger.info(f"Found {len(label_files)} label files in {self.labels_dir}")
            
        if not self.template_labels_path.exists():
            self.logger.warning(f"Template labels file not found: {self.template_labels_path}")
            
    @contextmanager
    def _scratch(self, prefix):
        """A private scratch directory under temp_dir, removed on exit.

        Deliberately never yields temp_dir itself. temp_dir may be a path the
        caller handed us, or the container's /work bind shared with other
        jobs; deleting it is not ours to do. `mkdtemp` gives a directory we
        created and therefore may remove, so cleanup cannot reach anything
        passed in on purpose.

        This exists because nothing used to delete the per-subject
        registration scratch at all: warp_atlases writes target.nii.gz plus 40
        warped volumes (~1-2 GB) per subject and they accumulated for the life
        of the output tree.
        """
        base = str(self.temp_dir) if self.temp_dir else None
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _load_templates(self):
        """Load OASIS-30 Atropos templates.
        
        Returns
        -------
        dict
            Dictionary containing loaded template images
        """
        if not self.template_dir or not self.template_dir.exists():
            raise ValueError(f"Template directory not found: {self.template_dir}")
            
        templates = {}
        try:
            # Load main templates
            templates['T_template'] = ants.image_read(str(self.template_dir / 'T_template0.nii.gz'))
            templates['BrainCerebellum'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellum.nii.gz'))
            templates['ProbabilityMask'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumProbabilityMask.nii.gz'))
            templates['ExtractionMask'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumExtractionMask.nii.gz'))
            
            # Load tissue priors
            templates['priors'] = []
            priors_dir = self.template_dir / 'Priors2'
            for i in range(1, 7):
                prior = ants.image_read(str(priors_dir / f'priors{i}.nii.gz'))
                templates['priors'].append(prior)
                
            self.logger.info("Successfully loaded all templates")
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {str(e)}")
            raise
    
    def load_image(self, image_path):
        """Load an image using ANTs.
        
        Parameters
        ----------
        image_path : str or Path
            Path to the image file
            
        Returns
        -------
        ants.ANTsImage or None
            Loaded image or None if loading fails
        """
        try:
            self.logger.info(f"Loading image: {image_path}")
            ants_mod = importlib.import_module("ants")
            img = ants_mod.image_read(str(image_path))
            if img is None:
                self.logger.error(f"Failed to load image {image_path}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """
        Preprocess an ANTs image for segmentation.
        
        Parameters:
        -----------
        image : ants.ANTsImage
            Input image to preprocess
            
        Returns
        --------
        ants.ANTsImage
            Preprocessed image
        """
        self.logger.info("Preprocessing image")
        
        # N4 bias field correction
        self.logger.debug("Applying N4 bias field correction")
        corrected = ants.n4_bias_field_correction(image)
        
        # Brain extraction
        self.logger.debug("Performing brain extraction")
        brain_mask = ants.get_mask(corrected)
        brain_image = ants.mask_image(corrected, brain_mask)
        
        return brain_image
    
    def segment_image(self, image, method='fusion', **kwargs):
        """Perform ANTs segmentation on the given image.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input image to segment
        method : str, optional
            Segmentation method to use:
            - 'quick': Use pre-computed labels in template space
            - 'fusion': Use JointLabelFusion with atlas images (default)
        **kwargs : dict
            Additional arguments passed to specific segmentation method
            
        Returns
        -------
        dict
            Dictionary containing segmentation results
        """
        # Run cortical thickness to get N4 correction and brain mask
        self.logger.info("Running cortical thickness pipeline for preprocessing")
        thickness_results = self.compute_cortical_thickness(
            image, register_template=(method == 'quick'))
        
        # Create masked brain image (equivalent to MultiplyImages node)
        self.logger.info("Creating masked brain image")
        masked_brain = thickness_results['BrainSegmentationN4'] * thickness_results['BrainExtractionMask']
        
        if method == 'quick':
            # Use pre-computed labels in template space
            if not self.template_labels_path.exists():
                raise ValueError(f"Template labels not found: {self.template_labels_path}")
                
            template_warp = thickness_results.get('TemplateWarp')
            if template_warp is None:
                raise ValueError("No TemplateWarp available for template-to-subject mapping")

            label_map = build_label_map([self.template_labels_path])
            # The returned image is fully in memory, so the scratch NIfTIs the
            # warp round-trips through can go as soon as the call returns.
            with self._scratch('warp_') as scratch:
                warped_labels = apply_warp(
                    template_warp,
                    ants.image_read(str(self.template_labels_path)),
                    thickness_results['BrainSegmentationN4'],
                    temp_dir=scratch,
                    interpolation=self.label_interpolation,
                    label_map=label_map,
                    logger=self.logger,
                )

            return {
                'segmentation': warped_labels,
                'BrainSegmentationN4': thickness_results['BrainSegmentationN4'],
                'BrainExtractionMask': thickness_results['BrainExtractionMask']
            }
            
        elif method == 'fusion':
            if not self.atlas_dir.exists():
                self.logger.error(f"Atlas directory not found: {self.atlas_dir}")
                raise ValueError(f"Atlas directory not found: {self.atlas_dir}. Cannot use 'fusion' method.")
            if not self.labels_dir.exists():
                self.logger.error(f"Labels directory not found: {self.labels_dir}")
                raise ValueError(f"Labels directory not found: {self.labels_dir}. Cannot use 'fusion' method.")
                
            # Get all atlas T1s and labels
            T1s = natsorted(glob.glob(str(self.atlas_dir / '*.nii.gz')))
            labels = natsorted(glob.glob(str(self.labels_dir / '*.nii.gz')))
            
            if not T1s:
                self.logger.error(f"No atlas T1 images found in {self.atlas_dir}")
                self.logger.error(f"Directory contents: {list(self.atlas_dir.iterdir())[:5]}")
                raise ValueError(f"No atlas T1 images found in {self.atlas_dir}")
            if not labels:
                self.logger.error(f"No label images found in {self.labels_dir}")
                self.logger.error(f"Directory contents: {list(self.labels_dir.iterdir())[:5]}")
                raise ValueError(f"No label images found in {self.labels_dir}")
                
            self.logger.info(f"Found {len(T1s)} atlas T1 images and {len(labels)} label images")
            if len(T1s) != len(labels):
                self.logger.warning(f"Number of atlas images ({len(T1s)}) does not match number of labels ({len(labels)})")
                
            # Registration is FireANTs; the fusion voting below stays on ANTs.
            # Images cross the boundary as NIfTI files under temp_dir and are
            # geometry-checked on the way back -- see src/registration/handoff.py.
            # ~1-2 GB of target + 40 warped volumes, in a directory of our own
            # under TMPDIR. warp_atlases reads every one of them back into
            # memory as an ANTsImage before returning (and joint_label_fusion
            # below passes in-memory pointers), so the files are dead the
            # moment the call returns.
            with self._scratch('fireants_') as scratch:
                warped_images, warped_labels = warp_atlases(
                    masked_brain,
                    thickness_results['BrainExtractionMask'],
                    T1s,
                    labels,
                    temp_dir=scratch,
                    device=self.device,
                    batch_size=self.fireants_batch_size,
                    label_interpolation=self.label_interpolation,
                    logger=self.logger,
                )

            # Perform joint label fusion
            #
            # max_lab_plus_one=True is required, not a tuning knob. The
            # brain extraction mask is much larger than the union of the
            # labelled structures: the atlases leave CSF and everything
            # between structures at 0. The vote has no background class, so
            # with max_lab_plus_one=False every one of those masked-but-
            # unlabelled voxels still has to be assigned a real label, and an
            # argmax over a probability stack that is ~0 everywhere returns
            # the first entry -- the lowest label id, which for this atlas
            # vocabulary is 4, the third ventricle.
            #
            # That is not hypothetical. The delivered tier-2 output for
            # sub-0051456 called 89924 voxels "3rd ventricle" against ~700 in
            # the atlases, in 2725 disconnected components, 74% of them within
            # 3 voxels of the mask boundary, and 96% of them at a maximum
            # across-label probability below 0.05. A minimal reproduction
            # (scripts/diag_jlf_background_fallback.py) shows the same 54x
            # inflation and shows this flag removing it exactly.
            #
            # no_zeroes=True also empties the band, but it constrains the
            # solution to voxels non-zero in *every* atlas, which erodes real
            # structures at their boundaries -- in the reproduction it
            # undershoots the true structure by a third. max_lab_plus_one
            # recovers it exactly.
            #
            # usecor=True is also not a tuning knob. The target (N4-corrected
            # masked brain, in-mask median ~388) and the OASIS-TRT-20 atlas
            # brains (~1100) are on different intensity scales, and the
            # default mean-squared-difference similarity is not
            # scale-invariant: the r_search patch search moves each atlas's
            # labels toward wherever raw intensity matches. Subject WM then
            # matches darker atlas tissue, so WM-like structures shrink and
            # GM/CSF grow -- on 6 Caltech subjects vs FreeSurfer, brain stem
            # 0.38x, cerebellar WM 0.20-0.21x, lateral ventricles 3.2x, mean
            # Dice 0.36-0.40. Correlation is scale-invariant; with it the same
            # subjects reach 0.69-0.76, with higher Dice than r_search=0 or
            # rescaling the target in every structure group and on every
            # subject (scripts/diag_jlf_validation_summary.py). The bias predates
            # FireANTs: the ANTs-registered August outputs carry it too.
            #
            # Side effect documented by ANTsPy: this mutates the label images
            # passed in. warped_labels is not used again after this call.
            self.logger.info(f"Running joint label fusion with {len(warped_images)} atlases")
            fusion = ants.joint_label_fusion(
                target_image=masked_brain,
                target_image_mask=thickness_results['BrainExtractionMask'],
                atlas_list=warped_images,
                label_list=warped_labels,
                verbose=True,
                **JLF_PARAMS,
            )
            
            return {
                'segmentation': fusion['segmentation'],
                'intensity': fusion['intensity'],
                'probabilityimages': fusion['probabilityimages'],
                'segmentation_numbers': fusion['segmentation_numbers'],
                'BrainSegmentationN4': thickness_results['BrainSegmentationN4'],
                'BrainExtractionMask': thickness_results['BrainExtractionMask']
            }
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def _organize_bids_output(self, segmentation, bids_subject, bids_session=None):
        """Organize ANTs outputs in BIDS-compliant format.

        Parameters
        ----------
        segmentation : dict
            Segmentation results containing:
            - For Atropos: dictionary with 'segmentation' and 'probabilityimages'
            - For JointLabelFusion: dictionary with 'segmentation', 'intensity', 
              'probabilityimages', and 'segmentation_numbers'
        bids_subject : str
            Subject ID (without "sub-" prefix)
        bids_session : str, optional
            BIDS session label (without "ses-" prefix)
        
        Returns
        -------
        dict
            Dictionary containing tissue and label volumes
        """
        # Set up directories with BIDS-compliant structure
        # Create sub-*/ses-* subdirectories under output_dir
        session_part = f"_ses-{bids_session}" if bids_session else ""

        # Create subject/session directory structure
        subject_dir = self.output_dir / f"sub-{bids_subject}"
        if bids_session:
            subject_dir = subject_dir / f"ses-{bids_session}"

        anat_dir = subject_dir / "anat"
        stats_dir = subject_dir / "stats"
        anat_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Save labeled image
        label_filename = f"sub-{bids_subject}{session_part}_space-orig_dseg.nii.gz"
        label_path = anat_dir / label_filename
        ants.image_write(segmentation['segmentation'], str(label_path))
        
        # Calculate volumes and prepare stats
        labeled_image = segmentation['segmentation']
        labels = labeled_image.numpy()
        voxel_volume = float(np.prod(labeled_image.spacing))

        label_values, label_counts = np.unique(labels.astype(np.int64), return_counts=True)
        
        # Only background is left out. Ids 21-23 and 75-76 used to be dropped
        # as FreeSurfer reference lines and retired LUT entries, but --method
        # fusion emits Neuromorphometrics ids (label_vocabulary.py), where 23
        # is Right Accumbens and 75/76 the basal forebrain; quick's aseg map
        # contains none of them.
        valid_mask = label_values != 0
        label_values = label_values[valid_mask]
        label_counts = label_counts[valid_mask]
        label_volumes_mm3 = label_counts * voxel_volume

        label_volume_sum = float(label_volumes_mm3.sum()) if label_volumes_mm3.size else 0.0

        # BVOL is the brain extraction mask volume, as in ANTs'
        # antsCorticalThickness.sh, and its data element says it is about the
        # brain (UBERON_0000955). It used to be the label sum above, which is
        # not a brain volume: the DKT31_CMA label set has no cerebral white
        # matter label, so the sum is about half a brain (0.90-1.03 L against
        # a mask of 1.68-1.87 L on 6 Caltech subjects). The mask agrees with
        # FreeSurfer's BrainSegVol within 0.8-4.2% on the same subjects. It
        # includes the ventricles and some sulcal CSF, so it is not a
        # parenchymal volume. The label sum is still reported, as LABELVOL.
        mask_image = segmentation.get('BrainExtractionMask')
        if mask_image is None:
            raise ValueError(
                "segmentation has no BrainExtractionMask; BVOL is the brain mask "
                "volume and cannot be computed without it"
            )
        brain_volume = float(np.count_nonzero(mask_image.numpy())) * float(np.prod(mask_image.spacing))

        # Generate antslabelstats.csv with subject prefix
        seg_base = f"sub-{bids_subject}{session_part}"
        labelstats_file = stats_dir / f"{seg_base}_antslabelstats.csv"
        label_df = pd.DataFrame({
            "Label": label_values.astype(int),
            "VolumeInVoxels": label_counts.astype(int),
            "Volume_mm3": label_volumes_mm3.astype(float),
        })
        label_df.to_csv(labelstats_file, index=False)

        # Generate antsbrainvols.csv
        #
        # BVOL (brain mask) and LABELVOL (label sum), and no tissue volumes.
        # This used to also emit CSFVOL/GMVOL/WMVOL from
        # probabilityimages[0], [1] and [2], on the assumption that those are
        # CSF/GM/WM tissue posteriors. They are not, in either method:
        #   - 'fusion' returns one posterior per *anatomical label* (103 of them
        #     on ABIDE sub-0051456), so [0..2] were simply the first three DKT
        #     labels wearing tissue names -- delivered CSFVOL/GMVOL/WMVOL of
        #     4105/5426/10391 mm^3 against ~1.5e5/6.0e5/5.0e5 expected.
        #   - 'quick' returns no 'probabilityimages' key at all, so the branch
        #     never ran there.
        # That is every code path, so the columns were wrong whenever present.
        # Deriving real tissue volumes means grouping DKT labels into tissue
        # classes (or running Atropos for posteriors), which is a separate piece
        # of work; reporting nothing beats reporting numbers off by ~100x.
        brain_vol_data = {"BVOL": brain_volume, "LABELVOL": label_volume_sum}

        brainvols_file = stats_dir / f"{seg_base}_antsbrainvols.csv"
        pd.DataFrame([brain_vol_data]).to_csv(brainvols_file, index=False)

        # Save probability maps if available.
        # Name each map by the anatomical label it belongs to, taken from
        # segmentation_numbers, rather than by its position in the list. The
        # index-based names (label-1 ... label-104) did not correspond to any
        # label in the segmentation, so the maps could not be matched to the
        # structures they describe.
        prob_images = segmentation.get('probabilityimages') or []
        seg_numbers = segmentation.get('segmentation_numbers') or []
        if prob_images and len(seg_numbers) != len(prob_images):
            self.logger.warning(
                f"{len(prob_images)} probability maps but {len(seg_numbers)} label "
                "numbers; falling back to index-based probseg names"
            )
        for idx, prob_img in enumerate(prob_images):
            if len(seg_numbers) == len(prob_images):
                label_id = int(seg_numbers[idx])
            else:
                label_id = idx + 1
            prob_filename = f"sub-{bids_subject}{session_part}_space-orig_label-{label_id}_probseg.nii.gz"
            prob_path = anat_dir / prob_filename
            ants.image_write(prob_img, str(prob_path))
        
        self.logger.info(f"Saved segmentation results for subject {bids_subject}")
        volume_info = {
            'brain_volume': brain_volume,
            'label_volume_sum': label_volume_sum,
            'label_stats': str(labelstats_file),
            'brain_vols': str(brainvols_file),
            'segmentation': str(label_path),
            'label_volumes': {int(lbl): float(vol) for lbl, vol in zip(label_values, label_volumes_mm3)},
        }

        return volume_info

    def save_results(self, segmentation, bids_subject, bids_session=None, output_dir=None, input_t1w=None):
        """Save segmentation results in BIDS-compatible format and generate files for NIDM conversion.
        
        Parameters
        ----------
        segmentation : dict
            Dictionary containing segmentation results
        bids_subject : str
            Subject identifier
        bids_session : str, optional
            Session identifier
        output_dir : str or Path, optional
            Output directory (defaults to self.output_path)
        input_t1w : str or Path, optional
            Path to input T1w file used for segmentation
        """
        if output_dir:
            original_output = self.output_dir
            self.output_dir = Path(output_dir)

        try:
            # Organize outputs in BIDS format
            volumes = self._organize_bids_output(segmentation, bids_subject, bids_session)
            
            # Add input T1w file information if provided
            if input_t1w:
                volumes['input_t1w'] = str(input_t1w)
            
            # Add template and atlas information
            volumes['templates'] = {
                'template_dir': str(self.template_dir),
                'template_files': {
                    'T_template': 'T_template0.nii.gz',
                    'BrainCerebellum': 'T_template0_BrainCerebellum.nii.gz',
                    'ProbabilityMask': 'T_template0_BrainCerebellumProbabilityMask.nii.gz',
                    'ExtractionMask': 'T_template0_BrainCerebellumExtractionMask.nii.gz',
                    'priors_dir': 'Priors2'
                },
                'template_description': 'OASIS-30 Atropos template'
            }
            
            volumes['atlases'] = {
                'atlas_dir': str(self.atlas_dir),
                'labels_dir': str(self.labels_dir),
                'template_labels': str(self.template_labels_path),
                'atlas_description': 'OASIS-TRT-20 brains and DKT31 CMA labels'
            }
                
            self.logger.info("Segmentation results saved successfully")
            return volumes
        finally:
            if output_dir:
                self.output_dir = original_output
        
    def run_subject(self, subject_id, session_label=None, method='fusion'):
        """Run ANTs segmentation for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with "sub-" prefix)
        session_label : str, optional
            Session label (with "ses-" prefix)
        method : str, optional
            Segmentation method ('quick' or 'fusion')
            
        Returns
        -------
        bool
            True if processing succeeded, False otherwise
        """
        try:
            # Strip prefixes for BIDS queries
            bids_subject = subject_id[4:] if subject_id.startswith('sub-') else subject_id
            if session_label:
                bids_session = session_label[4:] if session_label.startswith('ses-') else session_label
            else:
                bids_session = None
                
            # Construct path to input image
            if bids_session:
                bids_path = self.bids_dir / f"sub-{bids_subject}" / f"ses-{bids_session}" / "anat"
                img_pattern = f"sub-{bids_subject}_ses-{bids_session}_{self.modality}.nii.gz"
            else:
                bids_path = self.bids_dir / f"sub-{bids_subject}" / "anat"
                img_pattern = f"sub-{bids_subject}_{self.modality}.nii.gz"
            
            # Find input image
            input_file = list(bids_path.glob(img_pattern))
            if not input_file:
                self.logger.error(f"No {self.modality} image found for subject {subject_id}")
                return False
            
            input_file = input_file[0]
            self.logger.info(f"Processing {input_file}")
            
            # Load image using helper to centralize error handling (and simplify testing)
            img = self.load_image(str(input_file))
            if img is None:
                return False
            
            # Run segmentation
            segmentation = self.segment_image(img, method=method)
            if not segmentation:
                return False
            
            # Save results and get volumes, passing the input T1w file path
            volumes = self.save_results(segmentation, bids_subject, bids_session, input_t1w=str(input_file))
            
            # Log volume information
            if volumes:
                self.logger.info("Segmentation volumes:")
                if 'brain_volume' in volumes:
                    self.logger.info(f"  BVOL (brain mask): {volumes['brain_volume']:.2f} mm³")
                if 'label_volume_sum' in volumes:
                    self.logger.info(f"  LABELVOL (sum of labels): {volumes['label_volume_sum']:.2f} mm³")
                if 'label_volumes' in volumes:
                    for label, volume in volumes['label_volumes'].items():
                        self.logger.info(f"  Label {label}: {volume:.2f} mm³")
            
            self.logger.info(f"Segmentation complete for subject {subject_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return False
        
    def compute_cortical_thickness(self, image, register_template=True):
        """Compute cortical thickness using ANTs.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input brain image
        register_template : bool, optional
            Also register the brain to the OASIS-30 template and return the
            warp as 'TemplateWarp'. Only --method quick uses it.
            
        Returns
        -------
        dict
            Dictionary containing segmentation results
        """
        self.logger.info("Starting cortical thickness computation")
        
        # N4 bias field correction
        self.logger.info("Performing N4 bias field correction")
        n4_image = ants.n4_bias_field_correction(
            image,
            shrink_factor=4,
            convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},
            spline_param=200
        )
        
        # Brain extraction: run the canonical antsBrainExtraction.sh, the
        # published method the OASIS-30 template kit was built for, with each
        # template file in its designed role (-e whole-head template, -m brain
        # probability mask, -f extraction registration-scope mask). Two
        # successive hand-rolled re-implementations each fixed one failure
        # mode and left another (whole-head masks, CPU-type bistability, and
        # finally 11/33 ABIDE Caltech subjects at 2.05-3.87 L on hard
        # anatomies): the script's SyN stage and Atropos K=3 refinement are
        # what adapt the mask to each subject, and they are not worth
        # re-deriving in ANTsPy piece by piece.
        self.logger.info("Performing brain extraction (antsBrainExtraction.sh)")
        abe_script = shutil.which('antsBrainExtraction.sh')
        if not abe_script:
            raise RuntimeError(
                "antsBrainExtraction.sh not found on PATH. The container must "
                "ship the ANTs binaries (see Singularity/Dockerfile); ANTsPy "
                "alone does not include them.")

        abe_dir = Path(tempfile.mkdtemp(prefix='abe_', dir=str(self.temp_dir) if self.temp_dir else None))
        try:
            abe_input = abe_dir / 'n4.nii.gz'
            ants.image_write(n4_image, str(abe_input))
            abe_prefix = str(abe_dir / 'abe_')
            cmd = [
                abe_script,
                '-d', '3',
                '-a', str(abe_input),
                '-e', str(self.template_dir / 'T_template0.nii.gz'),
                '-m', str(self.template_dir / 'T_template0_BrainCerebellumProbabilityMask.nii.gz'),
                '-f', str(self.template_dir / 'T_template0_BrainCerebellumExtractionMask.nii.gz'),
                '-o', abe_prefix,
            ]
            self.logger.info(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            mask_path = Path(abe_prefix + 'BrainExtractionMask.nii.gz')
            # The script can fail and still exit 0 (observed: missing `bc`
            # printed a complaint to stdout and returned success), so the
            # only trustworthy success signal is the output mask existing.
            if proc.returncode != 0 or not mask_path.exists():
                self.logger.error(f"antsBrainExtraction.sh stdout:\n{proc.stdout}")
                self.logger.error(f"antsBrainExtraction.sh stderr:\n{proc.stderr}")
                raise RuntimeError(
                    f"antsBrainExtraction.sh failed (exit code {proc.returncode}, "
                    f"mask present: {mask_path.exists()}). "
                    f"stdout: {proc.stdout[-2000:]} stderr: {proc.stderr[-2000:]}")

            final_mask = ants.image_read(str(mask_path))
        finally:
            # The extracted-brain intermediate is large; keep only what was
            # read back into memory.
            shutil.rmtree(abe_dir, ignore_errors=True)

        # BrainExtractionMask is the canonical script's product, untouched: its
        # Atropos refinement and morphology already finalized it.
        results = {
            'BrainSegmentationN4': n4_image,
            'BrainExtractionMask': final_mask,
        }
        if not register_template:
            return results

        brain_image = n4_image * final_mask

        # Register to the brain-only template. brain_image is skull-stripped at
        # this point, so the template argument passed to register_to_template
        # below must be skull-stripped too. Registering onto the whole-head
        # T_template0 biased the affine toward scaling the brain up to the
        # head outline, inflating every template-space image warped back
        # through this warp.
        brain_template = ants.image_read(
            str(self.template_dir / 'T_template0_BrainCerebellum.nii.gz'))

        self.logger.info("Registering to template (FireANTs)")
        # register_to_template sets fixed=subject, moving=template, so the
        # returned warp already points template -> subject: the direction every
        # consumer needs. There is no fwd/inv pair to choose between, which is
        # what previously produced an 11.5 L brain volume on ABIDE sub-0051456.
        # TemplateWarp holds torch tensors, not files: the two NIfTIs written
        # here are only needed while FireANTs loads them.
        with self._scratch('tplreg_') as scratch:
            template_warp = register_to_template(
                brain_image,
                brain_template,
                temp_dir=scratch,
                device=self.device,
                logger=self.logger,
            )

        results['TemplateWarp'] = template_warp
        return results
