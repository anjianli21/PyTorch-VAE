import sys
# sys.path.append('..')
sys.path.append('./')
print(sys.path)
import os
import yaml
import argparse
import warnings
import numpy as np
from pathlib import Path
from models import *
from experiment.experiment import VAEXperiment
from experiment.experiment_cat_vae_1d import ExperimentCATVAE1d
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset.gaussian_beta_dataset import GaussianBetaDataset
from dataset.cr3bp_earth_dataset import CR3BPEarthDataset

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.core.saving import save_hparams_to_yaml

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae_1d_gaussian_beta.yaml')
parser.add_argument('--dataset',
                    help='specify the training data',
                    default='GaussianBeta')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['logging_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'], **config['data_params'])
experiment = ExperimentCATVAE1d(model,
                             config['exp_params'], config["data_params"])

# use **config to represent a bunch of parameters instead of one parameter?
if args.dataset == "cr3bp_earth":
    data = CR3BPEarthDataset(**config["data_params"], **config['model_params'], pin_memory=len(config['trainer_params']['gpus']) != 0)
else:
    warnings.warn("incorrect data type")
    exit()

print("start setting up data")
data.setup()
print("data ready!")

# Run the training
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=-1,
                                     # the best k models according to the quantity monitored will be saved
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True)
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
# Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")

# Save yaml
if config["model_params"]["name"] == "VAE1d":
    if args.dataset == 'GaussianBeta':
        with open(r'/home/anjian/Desktop/project/PyTorch-VAE/configs/vae_1d_gaussian_beta.yaml') as file:
            vae_1d_config = yaml.load(file, Loader=yaml.FullLoader)

        path_file = tb_logger.log_dir + "/" + "vae_1d_gaussian_beta_hparam.yaml"
        with open(path_file, 'w') as file:
            documents = yaml.dump(vae_1d_config, file)
    if args.dataset == "cr3bp_earth":
        with open(r'/home/anjian/Desktop/project/PyTorch-VAE/configs/vae_1d_cr3bp_earth.yaml') as file:
            vae_1d_config = yaml.load(file, Loader=yaml.FullLoader)

        path_file = tb_logger.log_dir + "/" + "vae_1d_cr3bp_earth_hparam.yaml"
        with open(path_file, 'w') as file:
            documents = yaml.dump(vae_1d_config, file)

runner.fit(experiment, datamodule=data)
