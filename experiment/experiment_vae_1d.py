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
import pickle
import pandas as pd
import seaborn as sns
import warnings

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
            Path(f"{self.logger.log_dir}/Samples/data/").mkdir(exist_ok=True, parents=True)

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

                # scale back samples to snopt solutions
                if self.data_params["data_distribution"] == "cr3bp_earth_local_optimal":
                    min_max_data_path = self.data_params["data_min_max_file_name"]

                with open(min_max_data_path, "rb") as f:  # load pickle
                    min_max_data = pickle.load(f)
                data_min = min_max_data["data_min"]
                data_max = min_max_data["data_max"]
                rescale_samples = samples * (data_max - data_min) + data_min

                # Save samples
                if self.current_epoch % 1 == 0:
                    file_path = os.path.join(self.logger.log_dir,
                                             "Samples/data",
                                             f"{self.logger.name}_Epoch_{self.current_epoch}_data.pkl")
                    with open(file_path, "wb") as fp:  # write pickle
                        pickle.dump(rescale_samples, fp)


                num_segments = 20
                shooting_time = []
                init_coast_time = []
                final_coast_time = []
                control_list = []
                all_samples = []
                for sample in rescale_samples:
                    shooting_time.append(sample[0])
                    init_coast_time.append(sample[1])
                    final_coast_time.append(sample[2])

                    current_control = sample[3: 11 * 3]
                    for i in range(10):
                        index = 20 - i
                        current_control = np.append(current_control, sample[index * 3: (index + 1) * 3])
                    control_list.append([current_control])

                    all_samples.append(sample)

                # Shooting time histogram
                fig, ax = plt.subplots()
                ax.hist(shooting_time)
                ax.set_xlim(0, 10)
                ax.set_title("histogram for shooting time")
                ax.set_xlabel("time")
                ax.set_ylabel("number")
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_shooting_time.png")
                fig.savefig(image_file_name)

                # init time histogram
                fig, ax = plt.subplots()
                ax.hist(init_coast_time)
                ax.set_xlim(0, 10)
                ax.set_title("histogram for initial coast time")
                ax.set_xlabel("time")
                ax.set_ylabel("number")
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_init_coast_time.png")
                fig.savefig(image_file_name)

                # final time histogram
                fig, ax = plt.subplots()
                ax.hist(final_coast_time)
                ax.set_xlim(0, 10)
                ax.set_title("histogram for final cost time")
                ax.set_xlabel("time")
                ax.set_ylabel("number")
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_final_coast_time.png")
                fig.savefig(image_file_name)

                # Save control confidence plot， verse the second half
                control = np.asarray(control_list).reshape(100000, num_segments, 3)
                radius = np.squeeze(control[:, :, -1])
                radius_df = pd.DataFrame(radius, columns=['{:d}'.format(i) for i in range(num_segments)])
                fig, ax = plt.subplots()
                df = pd.melt(frame=radius_df,
                             var_name='timestep',
                             value_name='control radius')
                sns.lineplot(ax=ax,
                             data=df,
                             x='timestep',
                             y='control radius',
                             sort=False).set(title='95% confidence intervel of control radius')
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_control_confidence.png")
                fig.savefig(image_file_name)

                # Covariance matrix for first 1millon variables
                # all_variables = normalize(all_variables, axis=0, norm='max')
                # all_variables = minmax_scale(all_variables, axis=0, feature_range=(-1, 1))
                df = pd.DataFrame(all_samples[:100000])
                f = plt.figure(figsize=(19, 15))
                plt.matshow(df.corr(), fignum=f.number)
                cb = plt.colorbar()
                cb.ax.tick_params(labelsize=14)
                image_file_name = os.path.join(self.logger.log_dir,
                                               "Samples/images",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}_covariance.png")
                plt.savefig(image_file_name)

                print(self.current_epoch)

                sample_stat = {}

            else:
                warnings.warn("incorrect data type")
                exit()

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
