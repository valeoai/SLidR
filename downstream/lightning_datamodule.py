import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.transforms import make_transforms_clouds
from downstream.dataloader_kitti import SemanticKITTIDataset
from downstream.dataloader_nuscenes import NuScenesDataset, custom_collate_fn


class DownstreamDataModule(pl.LightningDataModule):
    """
    The equivalent of a DataLoader for pytorch lightning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # in multi-GPU the actual batch size is that
        self.batch_size = config["batch_size"] // config["num_gpus"]
        # the CPU workers are split across GPU
        self.num_workers = max(config["num_threads"] // config["num_gpus"], 1)

    def setup(self, stage):
        # setup the dataloader: this function is automatically called by lightning
        transforms = make_transforms_clouds(self.config)
        if self.config["dataset"].lower() == "nuscenes":
            Dataset = NuScenesDataset
        elif self.config["dataset"].lower() in ("kitti", "semantickitti"):
            Dataset = SemanticKITTIDataset
        else:
            raise Exception(f"Unknown dataset {self.config['dataset']}")
        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"
        self.train_dataset = Dataset(
            phase=phase_train, transforms=transforms, config=self.config
        )
        if Dataset == NuScenesDataset:
            self.val_dataset = Dataset(
                phase=phase_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = Dataset(phase=phase_val, config=self.config)

    def train_dataloader(self):
        # construct the training dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):
        # construct the validation dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
