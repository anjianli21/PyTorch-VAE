import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils.utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy

from pytorch_lightning.core.saving import save_hparams_to_yaml
import yaml


class ExperimentVAE1d(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 data_params: dict) -> None:
        super(ExperimentVAE1d, self).__init__()

        self.model = vae_model
        self.params = params
        self.data_params = data_params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.save_hyperparameters()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        self.curr_device = batch.device

        results = self.forward(batch)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        # print("training loss", train_loss)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        # print(batch.size())
        self.curr_device = batch.device

        results = self.forward(batch)

        # print(len(results))
        # for result in results:
        #     print(result.size())

        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        # print("validation loss ", val_loss)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:  # Called at the end of validation.
        # self.sample_images()
        self.sample_point()
        pass

    def sample_point(self):

        try:
            samples = self.model.sample(100000,
                                        self.curr_device)
            samples = samples.cpu().data.numpy()

            Path(f"{self.logger.log_dir}/Samples/images/").mkdir(exist_ok=True, parents=True)
            Path(f"{self.logger.log_dir}/Samples/statistics/").mkdir(exist_ok=True, parents=True)

            if self.data_params["data_distribution"] == "gaussian":

                self.save_images(samples=samples)

                sample_mean = np.mean(samples, axis=0)
                sample_var = np.var(samples, axis=0)
                sample_cov = np.cov(samples, rowvar=False)

                plt.clf()
                plt.matshow(sample_cov)
                plt.colorbar()
                # plt.show()
                cov_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_covariance.png")
                plt.savefig(cov_file_name)

                sample_stat = {"sample_mean": sample_mean.tolist(),
                               "sample_var": sample_var.tolist(),
                               "sample_z_covariance": sample_cov.tolist()}

            elif self.data_params["data_distribution"] == "beta":

                # TODO: manually add mean
                # samples += 0.28578897

                # Clip samples to be within (0, 1)
                samples[samples >= 1] = 0.9999999
                samples[samples <= 0] = 0.0000001

                self.save_images(samples=samples)

                # fit param
                sample_alpha = []
                sample_beta = []
                for i in range(self.data_params["data_dim"]):
                    sample_alpha_curr, sample_beta_curr, _, _ = scipy.stats.beta.fit(samples[:, i], floc=0, fscale=1)
                    sample_alpha.append(sample_alpha_curr)
                    sample_beta.append(sample_beta_curr)
                sample_alpha = np.asarray(sample_alpha)
                sample_beta = np.asarray(sample_beta)

                sample_cov = np.cov(samples, rowvar=False)

                plt.clf()
                plt.matshow(sample_cov)
                plt.colorbar()
                # plt.show()

                cov_file_name = os.path.join(self.logger.log_dir,
                                             "Samples/images",
                                             f"{self.logger.name}_Epoch_{self.current_epoch}_covariance.png")
                plt.savefig(cov_file_name)

                sample_stat = {"sample_alpha": sample_alpha.tolist(),
                               "sample_beta": sample_beta.tolist(),
                               "sample_z_covariance": sample_cov.tolist()}

            elif self.data_params["data_distribution"] == "gaussian_beta_distribution":
                # first half gaussian, second half beta
                samples_gaussian = samples[:, :int(self.data_params["data_dim"] / 2)]
                samples_beta = samples[:, int(self.data_params["data_dim"] / 2):]
                # Clip beta_samples to be within (0, 1)
                samples_beta[samples_beta >= 1] = 0.9999999
                samples_beta[samples_beta <= 0] = 0.0000001

                # print(np.shape(samples_gaussian))
                # print(np.shape(samples_beta))

                sample_gaussian_mean = np.mean(samples_gaussian, axis=0)
                sample_gaussian_var = np.var(samples_gaussian, axis=0)
                sample_gaussian_cov = np.cov(samples_gaussian, rowvar=False)

                # show the first gaussian samples
                plt.clf()
                plt.hist(samples_gaussian[:, 0])
                # plt.show()
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_first_gaussian_samples.png")
                plt.savefig(image_file_name)

                # show the first beta samples
                plt.clf()
                plt.hist(samples_beta[:, 0])
                # plt.show()
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_first_beta_samples.png")
                plt.savefig(image_file_name)

                # Show gaussian covariance
                plt.clf()
                plt.matshow(sample_gaussian_cov)
                plt.colorbar()
                # plt.show()
                cov_file_name = os.path.join(self.logger.log_dir,
                                             "Samples/images",
                                             f"{self.logger.name}_Epoch_{self.current_epoch}_gaussian_covariance.png")
                plt.savefig(cov_file_name)

                sample_beta_alpha, sample_beta_beta, _, _ = scipy.stats.beta.fit(samples_beta, floc=0, fscale=1)

                sample_beta_cov = np.cov(samples_beta, rowvar=False)

                plt.clf()
                plt.matshow(sample_beta_cov)
                plt.colorbar()
                # plt.show()

                cov_file_name = os.path.join(self.logger.log_dir,
                                             "Samples/images",
                                             f"{self.logger.name}_Epoch_{self.current_epoch}_beta_covariance.png")
                plt.savefig(cov_file_name)

                sample_stat = {"sample_gaussian_mean": sample_gaussian_mean.tolist(),
                               "sample_gaussian_var": sample_gaussian_var.tolist(),
                               "sample_gaussian_z_covariance": sample_gaussian_cov.tolist(),
                               "sample_beta_alpha": sample_beta_alpha.tolist(),
                               "sample_beta_beta": sample_beta_beta.tolist(),
                               "sample_beta_z_covariance": sample_beta_cov.tolist()
                               }

            elif self.data_params["data_distribution"] == "gaussian_mixture":
                self.save_images(samples=samples)
                sample_mean = np.mean(samples, axis=0)
                sample_stat = {"sample_mean": sample_mean.tolist()}

            elif self.data_params["data_distribution"] == "cr3bp_earth_local_optimal":
                sample_mean = np.mean(samples, axis=0)
                sample_stat = {"sample_mean": sample_mean.tolist()}

            sample_stat_file_name = os.path.join(self.logger.log_dir,
                                           "Samples/statistics",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}_sample_stat.yml")

            with open(sample_stat_file_name, 'w') as outfile:
                yaml.dump(sample_stat, outfile, default_flow_style=False)

        except Warning:
            pass

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def save_images(self, samples):

        # show the first dimensional samples
        plt.clf()
        plt.hist(samples[:, 0])
        # plt.show()
        image_file_name = os.path.join(self.logger.log_dir,
                                       "Samples/images",
                                       f"{self.logger.name}_Epoch_{self.current_epoch}_first_samples.png")
        plt.savefig(image_file_name)

        # show the middle dimensional samples
        plt.clf()
        plt.hist(samples[:, int(self.data_params["data_dim"] / 2)])
        # plt.show()
        image_file_name = os.path.join(self.logger.log_dir,
                                       "Samples/images",
                                       f"{self.logger.name}_Epoch_{self.current_epoch}_middle_samples.png")
        plt.savefig(image_file_name)

        # show the last dimensional samples
        plt.clf()
        plt.hist(samples[:, -1])
        # plt.show()
        image_file_name = os.path.join(self.logger.log_dir,
                                       "Samples/images",
                                       f"{self.logger.name}_Epoch_{self.current_epoch}_last_samples.png")
        plt.savefig(image_file_name)