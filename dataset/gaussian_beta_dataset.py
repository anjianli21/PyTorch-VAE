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


class GaussianBetaDataset(LightningDataModule):
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
            data_distribution: str = "beta",
            beta_parameters: list = [2, 5],
            gaussian_covariance_type: str = "independent",
            gaussian_var: float = 1.0,
            beta_type: str = "unified",
            gaussian_mixture_mean: list = [-2, 2],
            gaussian_mixture_var: list = [1, 1],
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
        self.beta_parameters = beta_parameters
        self.gaussian_covariance_type = gaussian_covariance_type
        self.gaussian_var = gaussian_var
        self.beta_type = beta_type
        self.gaussian_mixture_mean = gaussian_mixture_mean
        self.gaussian_mixture_var = gaussian_mixture_var

    def setup(self, stage: Optional[str] = None) -> None:

        train_size = int(self.data_num)
        val_size = int(self.data_num / 10)

        rs = np.random.RandomState(seed=0)

        if self.data_distribution == "gaussian":
            mean = np.zeros(self.data_dim)

            if self.gaussian_covariance_type == "independent":
                cov = np.identity(self.data_dim)
            elif self.gaussian_covariance_type == "half_dependent_half_independent":
                cov = np.identity(self.data_dim)
                for i in range(int(self.data_dim / 2)):
                    for j in range(int(self.data_dim / 2)):
                        if i != j:
                            cov[i, j] = 1 / 2
            elif self.gaussian_covariance_type == "large_var":
                cov = np.identity(self.data_dim) * self.gaussian_var
            elif self.gaussian_covariance_type == "mixed_var":
                cov = np.zeros((self.data_dim, self.data_dim))
                np.fill_diagonal(cov, np.linspace(1.0, self.gaussian_var, num=self.data_dim))
            else:
                raise SystemExit('Wrong data covariance type assigned')

            train_dataset = rs.multivariate_normal(mean, cov, size=train_size)
            val_dataset = rs.multivariate_normal(mean=mean, cov=cov, size=val_size)

        elif self.data_distribution == "beta":
            if self.beta_type == "unified":
                alpha, beta = self.beta_parameters

                train_dataset = rs.beta(alpha, beta, size=(train_size, self.data_dim))
                val_dataset = rs.beta(alpha, beta, size=(val_size, self.data_dim))
            elif self.beta_type == "mixed":
                # alpha from 0.5 to 2
                # beta from 0.5 to 5
                alpha = np.linspace(0.5, 2.0, num=self.data_dim)
                beta = np.linspace(0.5, 5.0, num=self.data_dim)
                train_dataset = rs.beta(alpha, beta, size=(train_size, self.data_dim))
                val_dataset = rs.beta(alpha, beta, size=(val_size, self.data_dim))

            # shift to have mean 0
            # train_dataset = train_dataset - np.mean(train_dataset, axis=0)
            # val_dataset = val_dataset - np.mean(val_dataset, axis=0)

        elif self.data_distribution == "gaussian_beta_distribution":
            "first half gaussian, mean=0, var=self.gaussian_var, second half beta, [alpha, beta] = self.beta_parameters"
            gaussian_dim = int(self.data_dim / 2)
            beta_dim = int(self.data_dim / 2)

            # Gaussian half
            mean = np.zeros(gaussian_dim)
            cov = np.identity(gaussian_dim) * self.gaussian_var
            train_dataset_gaussian = rs.multivariate_normal(mean, cov, size=train_size)
            val_dataset_gaussian = rs.multivariate_normal(mean, cov, size=val_size)

            # Beta half
            alpha, beta = self.beta_parameters
            train_dataset_beta = rs.beta(alpha, beta, size=(train_size, beta_dim))
            val_dataset_beta = rs.beta(alpha, beta, size=(val_size, beta_dim))

            # combine gaussian and beta data
            train_dataset = np.hstack((train_dataset_gaussian, train_dataset_beta))
            val_dataset = np.hstack((val_dataset_gaussian, val_dataset_beta))

        elif self.data_distribution == "gaussian_mixture":
            mean_1 = np.zeros(self.data_dim) + self.gaussian_mixture_mean[0]
            mean_2 = np.zeros(self.data_dim) + self.gaussian_mixture_mean[1]

            cov_1 = np.identity(self.data_dim) * self.gaussian_mixture_var[0]
            cov_2 = np.identity(self.data_dim) * self.gaussian_mixture_var[1]
            train_dataset = np.vstack((rs.multivariate_normal(mean_1, cov_1, size=int(train_size / 2)),
                                      rs.multivariate_normal(mean_2, cov_2, size=int(train_size / 2))))
            val_dataset = np.vstack((rs.multivariate_normal(mean_1, cov_1, size=int(val_size / 2)),
                                    rs.multivariate_normal(mean_2, cov_2, size=int(val_size / 2))))
            np.random.shuffle(train_dataset)
            np.random.shuffle(val_dataset)

        else:
            raise SystemExit('Wrong data distribution assigned')

        plt.clf()
        plt.hist(train_dataset[:, 0])
        plt.show()

        plt.clf()
        plt.hist(train_dataset[:, int(self.data_dim / 2)])
        plt.show()

        plt.clf()
        plt.hist(train_dataset[:, -1])
        plt.show()

        plt.clf()
        plt.matshow(np.cov(train_dataset, rowvar=False))
        plt.colorbar()
        plt.show()

        # If using convolutional network, should expand the data dim to add channel
        if self.model_name == "VAEConv1d":
            train_dataset = np.expand_dims(train_dataset, axis=1)
        # train_dataset = np.tanh(train_dataset)

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
