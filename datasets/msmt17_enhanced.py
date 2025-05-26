from .msmt17 import MSMT17
from .bases import ImageDataset
import os.path as osp
import logging

class EnhancedMSMT17(MSMT17):
    """Enhanced MSMT17 dataset that emphasizes camera information"""
    
    def __init__(self, root='', **kwargs):
        super(EnhancedMSMT17, self).__init__(root, **kwargs)
        self.logger = logging.getLogger("enhanced_msmt17")
        self.logger.info("Using enhanced MSMT17 dataset with camera information")
        
        # Log camera statistics
        camera_stats = {}
        for _, pid, camid in self.train:
            if camid not in camera_stats:
                camera_stats[camid] = set()
            camera_stats[camid].add(pid)
        
        self.logger.info(f"Camera statistics: {len(camera_stats)} cameras in training set")
        for camid, pids in camera_stats.items():
            self.logger.info(f"Camera {camid}: {len(pids)} identities")