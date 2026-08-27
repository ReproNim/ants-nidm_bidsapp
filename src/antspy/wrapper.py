# src/ants/wrapper.py
import os
import subprocess
import logging
import importlib
import ants
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from natsort import natsorted

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
        verbose : bool
            Whether to print detailed logs
        """
        # Convert all path inputs to Path objects
        self.bids_dir = Path(bids_dir) if bids_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.temp_dir = Path(temp_dir) if temp_dir else (Path(output_dir) / 'tmp' if output_dir else None)
        
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
    
    def segment_with_jointfusion(self, target_image, atlas_images, atlas_labels=None, search_radius=3, patch_radius=2, beta=4.0, rho=0.01, use_correlation=False, non_negative=False, no_zeroes=False, max_lab_plus_one=False):
        """Perform ANTs JointLabelFusion segmentation.
        
        Parameters
        ----------
        target_image : ants.ANTsImage
            Target image to segment
        atlas_images : list of ants.ANTsImage
            List of atlas images registered to target space
        atlas_labels : list of ants.ANTsImage, optional  
            List of atlas label images registered to target space.
            If not provided, will perform intensity fusion only.
        search_radius : int, optional
            Search radius for patch matching (default=3)
        patch_radius : int, optional
            Patch radius for local intensity similarity (default=2)
        beta : float, optional
            Weight sharpness parameter for patch similarity (default=4.0)
        rho : float, optional
            Gradient step size (default=0.01)
        use_correlation : bool, optional
            Use correlation metric (default=False)
        non_negative : bool, optional
            Use non-negative weights (default=False)
        no_zeroes : bool, optional
            Exclude zeroes from computations (default=False)
        max_lab_plus_one : bool, optional
            Add max label plus one to output (default=False)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'segmentation': ANTsImage of segmentation (None if no atlas_labels provided)
            - 'intensity': ANTsImage of intensity fusion result
            - 'probabilityimages': list of probability maps (empty if no atlas_labels provided)
            - 'segmentation_numbers': list of label numbers (empty if no atlas_labels provided)
        """
        self.logger.info("Starting JointLabelFusion segmentation")
        
        # Get mask for segmentation
        mask = ants.get_mask(target_image)
        
        # Run JointLabelFusion
        if atlas_labels is None:
            self.logger.info("No atlas labels provided, performing intensity fusion only")
            intensity = ants.joint_label_fusion(
                target_image=target_image,
                target_image_mask=mask,
                atlas_list=atlas_images,
                label_list=None,
                beta=beta,
                rad=patch_radius,
                r_search=search_radius,
                rho=rho,
                usecor=use_correlation,
                nonnegative=non_negative,
                no_zeroes=no_zeroes,
                max_lab_plus_one=max_lab_plus_one,
                verbose=self.verbose
            )[1]  # Only take intensity result
            return {
                'segmentation': None,
                'intensity': intensity,
                'probabilityimages': [],
                'segmentation_numbers': []
            }
        else:
            seg, intensity, probimgs, labels = ants.joint_label_fusion(
                target_image=target_image,
                target_image_mask=mask,
                atlas_list=atlas_images,
                label_list=atlas_labels,
                beta=beta,
                rad=patch_radius,
                r_search=search_radius,
                rho=rho,
                usecor=use_correlation,
                nonnegative=non_negative,
                no_zeroes=no_zeroes,
                max_lab_plus_one=max_lab_plus_one,
                verbose=self.verbose
            )
            
            return {
                'segmentation': seg,
                'intensity': intensity,
                'probabilityimages': probimgs,
                'segmentation_numbers': labels
            }

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
        thickness_results = self.compute_cortical_thickness(image)
        
        # Create masked brain image (equivalent to MultiplyImages node)
        self.logger.info("Creating masked brain image")
        masked_brain = thickness_results['BrainSegmentationN4'] * thickness_results['BrainExtractionMask']
        
        if method == 'quick':
            # Use pre-computed labels in template space
            if not self.template_labels_path.exists():
                raise ValueError(f"Template labels not found: {self.template_labels_path}")
                
            template_labels = ants.image_read(str(self.template_labels_path))
            
            # Apply template-to-subject transforms to labels
            # Build transform list based on what's available
            transformlist = []
            whichtoinvert = []
            
            if thickness_results.get('TemplateToSubject1GenericAffine'):
                transformlist.append(thickness_results['TemplateToSubject1GenericAffine'])
                whichtoinvert.append(False)
                
            if thickness_results.get('TemplateToSubject0Warp'):
                transformlist.append(thickness_results['TemplateToSubject0Warp'])
                whichtoinvert.append(False)
                
            if not transformlist:
                raise ValueError("No transforms available for template-to-subject mapping")
                
            warped_labels = ants.apply_transforms(
                fixed=thickness_results['BrainSegmentationN4'],
                moving=template_labels,
                transformlist=transformlist,
                interpolator='nearestNeighbor',
                whichtoinvert=whichtoinvert
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
                
            # Load atlas images and labels
            atlas_images = [ants.image_read(str(path)) for path in T1s]
            atlas_labels = [ants.image_read(str(path)) for path in labels]
            
            # Register each atlas to the masked brain
            warped_images = []
            warped_labels = []
            
            for atlas_img, atlas_lab in zip(atlas_images, atlas_labels):
                # Register atlas to target
                reg = ants.registration(
                    fixed=masked_brain,
                    moving=atlas_img,
                    type_of_transform='SyNRA',  # Combined rigid, affine, and SyN registration
                    reg_iterations=[200, 200, 100],
                    transform_parameters=(0.1, 3, 0),
                    flow_sigma=3,
                    total_sigma=0,
                    aff_metric='mattes',
                    syn_metric='mattes',
                    verbose=True
                )
                
                # Apply transforms to atlas labels
                warped_lab = ants.apply_transforms(
                    fixed=masked_brain,
                    moving=atlas_lab,
                    transformlist=reg['fwdtransforms'],
                    interpolator='nearestNeighbor',
                    whichtoinvert=[False] * len(reg['fwdtransforms'])
                )
                
                warped_images.append(reg['warpedmovout'])
                warped_labels.append(warped_lab)
            
            # Perform joint label fusion
            self.logger.info(f"Running joint label fusion with {len(warped_images)} atlases")
            fusion = ants.joint_label_fusion(
                target_image=masked_brain,
                target_image_mask=thickness_results['BrainExtractionMask'],
                atlas_list=warped_images,
                label_list=warped_labels,
                beta=4.0,
                rad=2,
                rho=0.01,
                usecor=False,
                r_search=3,
                nonnegative=False,
                no_zeroes=False,
                max_lab_plus_one=False,
                verbose=True
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
        
        # Filter out problematic labels:
        # - 0: background
        # - 21-23: FreeSurfer reference lines (spatial markers, not brain structures)
        # - 75-76: Removed from FreeSurferColorLUT as duplicates of 4/43
        # These are present in OASIS-TRT-20_DKT31_CMA_labels_v2 but should not be in stats
        invalid_labels = [0, 21, 22, 23, 75, 76]
        valid_mask = ~np.isin(label_values, invalid_labels)
        label_values = label_values[valid_mask]
        label_counts = label_counts[valid_mask]
        label_volumes_mm3 = label_counts * voxel_volume
        
        # Log info if reference line labels were filtered
        filtered_ref_lines = [lbl for lbl in [21, 22, 23] if lbl in np.unique(labels.astype(np.int64))]
        if filtered_ref_lines:
            self.logger.info(f"Filtered FreeSurfer reference line labels from stats output: {filtered_ref_lines}")

        brain_volume = float(label_volumes_mm3.sum()) if label_volumes_mm3.size else 0.0

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
        brain_vol_data = {"BVOL": brain_volume}
        tissue_volumes = {}

        probability_images = segmentation.get('probabilityimages') or []
        if len(probability_images) >= 3:
            csf_vol = float(np.sum(probability_images[0].numpy() > self.prob_threshold) * voxel_volume)
            gm_vol = float(np.sum(probability_images[1].numpy() > self.prob_threshold) * voxel_volume)
            wm_vol = float(np.sum(probability_images[2].numpy() > self.prob_threshold) * voxel_volume)

            tissue_volumes = {
                "CSFVOL": csf_vol,
                "GMVOL": gm_vol,
                "WMVOL": wm_vol,
            }
            brain_vol_data.update(tissue_volumes)

        brainvols_file = stats_dir / f"{seg_base}_antsbrainvols.csv"
        pd.DataFrame([brain_vol_data]).to_csv(brainvols_file, index=False)

        # Save probability maps if available
        if 'probabilityimages' in segmentation:
            for idx, prob_img in enumerate(segmentation['probabilityimages']):
                prob_filename = f"sub-{bids_subject}{session_part}_space-orig_label-{idx+1}_probseg.nii.gz"
                prob_path = anat_dir / prob_filename
                ants.image_write(prob_img, str(prob_path))
        
        self.logger.info(f"Saved segmentation results for subject {bids_subject}")
        volume_info = {
            'brain_volume': brain_volume,
            'label_stats': str(labelstats_file),
            'brain_vols': str(brainvols_file),
            'segmentation': str(label_path),
            'label_volumes': {int(lbl): float(vol) for lbl, vol in zip(label_values, label_volumes_mm3)},
        }

        if tissue_volumes:
            volume_info['tissue_volumes'] = tissue_volumes

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
                if 'tissue_volumes' in volumes:
                    for tissue, volume in volumes['tissue_volumes'].items():
                        self.logger.info(f"  {tissue}: {volume:.2f} mm³")
                if 'label_volumes' in volumes:
                    for label, volume in volumes['label_volumes'].items():
                        self.logger.info(f"  Label {label}: {volume:.2f} mm³")
            
            self.logger.info(f"Segmentation complete for subject {subject_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return False
        
    def compute_cortical_thickness(self, image):
        """Compute cortical thickness using ANTs.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input brain image
            
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
        
        # Initial brain extraction using registration template and probability mask
        self.logger.info("Performing initial brain extraction")
        reg_template = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellum.nii.gz'))
        prob_mask = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumProbabilityMask.nii.gz'))
        
        # Initial rigid registration
        init_reg = ants.registration(
            fixed=reg_template,
            moving=n4_image,
            type_of_transform='Rigid',
            aff_metric='mattes',
            aff_sampling=32,
            aff_random_sampling_rate=0.2,
            reg_iterations=[1000, 500, 250],
            verbose=True,
            random_seed=1
        )
        
        # Transform probability mask to subject space
        init_mask = ants.apply_transforms(
            fixed=n4_image,
            moving=prob_mask,
            transformlist=init_reg['invtransforms'],
            interpolator='lanczosWindowedSinc'
        )
        
        # Create brain mask and extract brain
        brain_mask = ants.threshold_image(init_mask, 0.5, 1.0)
        brain_mask = ants.iMath(brain_mask, "FillHoles")
        brain_mask = ants.iMath(brain_mask, "GetLargestComponent")
        brain_image = n4_image * brain_mask
        
        # Register to brain template
        self.logger.info("Registering to template")
        brain_template = ants.image_read(str(self.template_dir / 'T_template0.nii.gz'))
        
        try:
            # First try affine registration
            affine_reg = ants.registration(
                fixed=brain_template,
                moving=brain_image,
                type_of_transform='Affine',
                aff_metric='mattes',
                aff_sampling=32,
                reg_iterations=[100, 100, 100],
                verbose=True,
                random_seed=1
            )
            
            # Then try SyN registration starting from affine
            reg = ants.registration(
                fixed=brain_template,
                moving=brain_image,
                type_of_transform='SyN',
                initial_transform=affine_reg['fwdtransforms'][0],
                aff_metric='mattes',
                syn_metric='mattes',
                reg_iterations=[100, 70, 50, 20],
                verbose=True,
                random_seed=1
            )
            
            transforms = [
                reg['fwdtransforms'][0],  # Affine transform
                reg['fwdtransforms'][1]   # Warp transform
            ]
            
        except Exception as e:
            self.logger.warning(f"SyN registration failed: {str(e)}")
            self.logger.info("Using affine registration only")
            transforms = [affine_reg['fwdtransforms'][0]]
        
        # Apply final transforms to extraction mask
        ext_mask = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumExtractionMask.nii.gz'))
        final_mask = ants.apply_transforms(
            fixed=n4_image,
            moving=ext_mask,
            transformlist=transforms,
            interpolator='nearestNeighbor'
        )
        
        # Ensure final mask is binary
        final_mask = ants.threshold_image(final_mask, 0.5, 1.0)
        final_mask = ants.iMath(final_mask, "FillHoles")
        final_mask = ants.iMath(final_mask, "GetLargestComponent")
        
        # Prepare results dictionary
        results = {
            'BrainSegmentationN4': n4_image,
            'BrainExtractionMask': final_mask
        }
        
        # Handle transforms based on what's available
        if len(transforms) > 1:
            results['TemplateToSubject1GenericAffine'] = transforms[0]
            results['TemplateToSubject0Warp'] = transforms[1]
        elif len(transforms) == 1:
            # Only affine available
            results['TemplateToSubject1GenericAffine'] = transforms[0]
            results['TemplateToSubject0Warp'] = None
        else:
            # No transforms (shouldn't happen but handle gracefully)
            results['TemplateToSubject1GenericAffine'] = None
            results['TemplateToSubject0Warp'] = None
            
        return results
