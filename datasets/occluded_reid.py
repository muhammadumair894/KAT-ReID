 # encoding: utf-8
import glob
import re
import os.path as osp
import logging

from .bases import BaseImageDataset
from utils.iotools import mkdir_if_missing # Assuming this utility exists

logger = logging.getLogger("transreid.dataset")

class OccludedReID(BaseImageDataset):
    """
    Occluded-ReID Dataset

    Dataset statistics:
    # To be filled in after processing
    """
    # Define the dataset directory relative to the root path provided
    # to the constructor
    dataset_dir = 'occluded_REID' # This should match the folder name in TransReID/data/

    def __init__(self, root='/data_sata/ReID_Group/ReID_Group/KANTransfarmers/TransReID/data', verbose=True, pid_begin=0, **kwargs):
        super(OccludedReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        logger.info(f"Attempting to load OccludedReID dataset from: {self.dataset_dir}")

        # --- Expected Subdirectory Structure ---
        # Modify these if your occluded_REID dataset has different folder names
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        # ----------------------------------------

        self.pid_begin = pid_begin

        # --- Logging paths ---
        logger.info(f"Expected train directory: {self.train_dir}")
        logger.info(f"Expected query directory: {self.query_dir}")
        logger.info(f"Expected gallery directory: {self.gallery_dir}")
        # ---------------------

        self._check_before_run() # Check if directories exist

        # Process the directories
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            logger.info("=> Occluded-ReID dataset loaded")
            self.print_dataset_statistics(train, query, gallery) # Uses base class method

        self.train = train
        self.query = query
        self.gallery = gallery

        # Calculate dataset statistics using the base class method
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        logger.info(f"Train PIDs: {self.num_train_pids}, Images: {self.num_train_imgs}, Cams: {self.num_train_cams}")
        logger.info(f"Query PIDs: {self.num_query_pids}, Images: {self.num_query_imgs}, Cams: {self.num_query_cams}")
        logger.info(f"Gallery PIDs: {self.num_gallery_pids}, Images: {self.num_gallery_imgs}, Cams: {self.num_gallery_cams}")


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.isdir(self.dataset_dir):
            logger.error(f"Dataset directory not found: '{self.dataset_dir}'")
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.isdir(self.train_dir):
            logger.warning(f"Train directory not found: '{self.train_dir}'")
            # Decide if this is critical - maybe raise RuntimeError if needed
        if not osp.isdir(self.query_dir):
            logger.warning(f"Query directory not found: '{self.query_dir}'")
            # Decide if this is critical
        if not osp.isdir(self.gallery_dir):
            logger.warning(f"Gallery directory not found: '{self.gallery_dir}'")
            # Decide if this is critical

    def _process_dir(self, dir_path, relabel=False):
        # --- Image Extension ---
        # Modify if your images are not .jpg
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # -----------------------

        if not img_paths:
             logger.warning(f"No images found in {dir_path} using pattern '*.jpg'")
             return [] # Return empty list if no images found

        logger.info(f"Found {len(img_paths)} image files in {dir_path}.")
        if len(img_paths) > 0:
             logger.info(f"First 5 image paths: {img_paths[:5]}")

        # --- Filename Parsing Regex ---
        # This regex assumes filenames like: 0001_c1_f001.jpg or 0002_c3s1_f002.jpg etc.
        # It extracts the first number as PID and the number after 'c' as Camera ID.
        # Adjust this regex based on your actual filename format.
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # Example alternative for Market1501 format (0001_c1s1_000401_00.jpg):
        # pattern = re.compile(r'([-\d]+)_c(\d+)s')
        # ----------------------------

        pid_container = set()
        for img_path in img_paths:
            try:
                match = pattern.search(osp.basename(img_path))
                if match is None:
                    logger.warning(f"Could not parse PID/CamID from filename: {osp.basename(img_path)} in {dir_path}")
                    continue # Skip this file if pattern doesn't match

                pid, _ = map(int, match.groups())
                if pid == -1: continue # Ignore junk images like Market1501
                pid_container.add(pid)
            except Exception as e:
                logger.error(f"Error processing filename {osp.basename(img_path)}: {e}")
                continue # Skip file on error

        # Create mapping from original PID to a new label (0 to N-1) if relabel is True
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        cam_container = set()
        for img_path in img_paths:
            try:
                match = pattern.search(osp.basename(img_path))
                if match is None: continue # Skip again if pattern doesn't match

                pid, camid = map(int, match.groups())
                if pid == -1: continue # Ignore junk images

                # --- Camera ID Sanity Check ---
                # Adjust the range if your camera IDs are different
                # assert 1 <= camid <= 8 # Example for DukeMTMC
                # If no specific range, remove assert but maybe log unusual IDs
                # ----------------------------

                camid -= 1  # Index camera IDs from 0
                if relabel:
                    pid = pid2label[pid] # Use the new 0-based label

                # Add item to dataset: (image_path, pid, camid, 1)
                # The '1' is often a placeholder for tracklet ID, assuming 1 view per tracklet here
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
                cam_container.add(camid)
            except Exception as e:
                 logger.error(f"Error creating dataset entry for {osp.basename(img_path)}: {e}")

        logger.info(f"Processed {len(dataset)} images for {dir_path}.")
        logger.info(f"Found camera IDs in {dir_path}: {sorted(list(cam_container))}")

        return dataset
