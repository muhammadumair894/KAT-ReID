import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import logging

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

# --- Import Dataset Classes ---
# Import all required dataset classes directly here
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17
# from .msmt17_enhanced import MSMT17Enhanced # Uncomment if used
from .veri import VeRi
from .vehicleid import VehicleID
from .occ_duke import OCC_DukeMTMCreID
from .occluded_reid import OccludedReID # <-- Import your new dataset class

# --- Re-introduce the Factory Dictionary ---
# This dictionary maps dataset names (used in config files) to their classes
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    # 'msmt17_enhanced': MSMT17Enhanced,
    'veri': VeRi,
    'VehicleID': VehicleID, # Note: Keep key consistent with how it's used in configs
    'occ_duke': OCC_DukeMTMCreID,
    'occluded_reid': OccludedReID, # <-- Add your dataset here
}
# -----------------------------------------

logger = logging.getLogger("transreid.dataset") # Initialize logger


def train_collate_fn(batch):
    # ... existing code ...
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    # ... existing code ...
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # --- Instantiate dataset using the local __factory ---
    dataset_name = cfg.DATASETS.NAMES[0] if isinstance(cfg.DATASETS.NAMES, (list, tuple)) else cfg.DATASETS.NAMES
    if dataset_name not in __factory:
        logger.error(f"Unknown dataset: {dataset_name}. Available datasets: {list(__factory.keys())}")
        raise KeyError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Initializing dataset '{dataset_name}' with root '{cfg.DATASETS.ROOT_DIR}'")
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)
    # ----------------------------------------------------

    # --- Logging for verification ---
    if hasattr(dataset, 'train') and dataset.train:
        logger.info(f"Loaded {len(dataset.train)} training images.")
    else:
        logger.warning("Training data not found or empty.") # More specific warning
    if hasattr(dataset, 'query') and dataset.query:
        logger.info(f"Loaded {len(dataset.query)} query images.")
    else:
        logger.warning("Query data not found or empty.")
    if hasattr(dataset, 'gallery') and dataset.gallery:
        logger.info(f"Loaded {len(dataset.gallery)} gallery images.")
    else:
         logger.warning("Gallery data not found or empty.")
    # ------------------------------------


    # Check if essential attributes exist before proceeding
    if not hasattr(dataset, 'num_train_pids') or \
       not hasattr(dataset, 'num_train_cams') or \
       not hasattr(dataset, 'num_train_vids'):
        logger.error("Dataset object missing required attributes (num_train_pids, num_train_cams, num_train_vids). Check dataset implementation.")
        # Assign default values or raise error, depending on downstream requirements
        num_classes = 0
        cam_num = 0
        view_num = 0
        # raise AttributeError("Dataset object missing required attributes.")
    else:
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids


    # Ensure train data exists before creating ImageDataset and samplers
    if not hasattr(dataset, 'train') or not dataset.train:
        logger.error("Training dataset is empty. Cannot proceed with creating training dataloaders.")
        # Handle this case gracefully - maybe return None for train loaders
        train_loader = None
        train_loader_normal = None
        # Or raise an error if training is mandatory
        # raise ValueError("Training dataset is empty.")
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)

        if 'triplet' in cfg.DATALOADER.SAMPLER:
            if cfg.MODEL.DIST_TRAIN:
                # ... distributed sampler logic ...
                print('DIST_TRAIN START')
                mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
            else:
                # ... standard triplet sampler logic ...
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn
                )
        elif cfg.DATALOADER.SAMPLER == 'softmax':
            # ... softmax sampler logic ...
            print('using softmax sampler')
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.DATALOADER.SAMPLER))
            train_loader = None # Handle unsupported sampler case

        # Create train_loader_normal only if train_set_normal makes sense
        train_loader_normal = DataLoader(
            train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )


    # Ensure query and gallery data exist before creating validation loader
    if (not hasattr(dataset, 'query') or not dataset.query) and \
       (not hasattr(dataset, 'gallery') or not dataset.gallery):
         logger.error("Both query and gallery datasets are empty. Cannot create validation set.")
         val_loader = None
         num_query = 0
    else:
        # Combine query and gallery, handling potential None or empty lists
        val_data = (dataset.query if hasattr(dataset, 'query') and dataset.query else []) + \
                   (dataset.gallery if hasattr(dataset, 'gallery') and dataset.gallery else [])
        if not val_data:
             logger.error("Validation data (query+gallery) is empty after combining.")
             val_loader = None
             num_query = 0
        else:
            val_set = ImageDataset(val_data, val_transforms)
            val_loader = DataLoader(
                val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=val_collate_fn
            )
        num_query = len(dataset.query) if hasattr(dataset, 'query') and dataset.query else 0


    final_num_query = num_query

    # Return values, ensuring train loaders might be None if train data was missing
    return train_loader, train_loader_normal, val_loader, final_num_query, num_classes, cam_num, view_num
