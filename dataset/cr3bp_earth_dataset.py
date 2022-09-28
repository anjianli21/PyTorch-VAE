import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from utils.cr3bp_earth_dataset_setup import cr3bp_earth_dataset_setup


class CR3BPEarthDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            name: str,
            data_num: int,
            data_dir: list,
            data_output_file_name: str,
            data_min_max_file_name: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            # patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            data_dim: int = 64,
            data_distribution: str = "local_optimal",
            **kwargs,
    ):
        super().__init__()

        self.model_name = name
        self.data_num = int(data_num)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        # self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dim = data_dim
        self.data_distribution = data_distribution
        self.data_dir_list = data_dir
        self.data_output_file_name = data_output_file_name
        self.data_min_max_file_name = data_min_max_file_name

    def setup(self, stage: Optional[str] = None) -> None:

        train_size = int(self.data_num)
        val_size = int(self.data_num / 10)

        train_dataset, val_dataset = cr3bp_earth_dataset_setup(data_dir_list=self.data_dir_list,
                                                               data_distribution=self.data_distribution,
                                                               train_size=train_size, val_size=val_size,
                                                               data_output_file_name=self.data_output_file_name,
                                                               data_min_max_file_name=self.data_min_max_file_name)

        self.train_dataset = torch.from_numpy(train_dataset).float()
        self.val_dataset = torch.from_numpy(val_dataset).float()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
