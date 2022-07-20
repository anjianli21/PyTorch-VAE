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

class MyDataset(LightningDataModule):
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
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            # patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            data_dim: int = 64,
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

    def setup(self, stage: Optional[str] = None) -> None:

        rs = np.random.RandomState(seed=0)
        mean = np.zeros(self.data_dim)
        cov = np.identity(self.data_dim)

        train_size = int(self.data_num)
        val_size = int(self.data_num / 10)

        train_dataset = rs.multivariate_normal(mean, cov, size=train_size)
        plt.hist(train_dataset[:, 0])
        plt.show()
        if self.model_name == "VAEConv1d":
            train_dataset = np.expand_dims(train_dataset, axis=1)
        # train_dataset = np.tanh(train_dataset)

        val_dataset = rs.multivariate_normal(mean=mean, cov=cov, size=val_size)
        if self.model_name == "VAEConv1d":
            val_dataset = np.expand_dims(val_dataset, axis=1)
        # val_dataset = np.tanh(val_dataset)

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