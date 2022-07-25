import sys
# sys.path.append('..')
sys.path.append('./')
print(sys.path)
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment.experiment import VAEXperiment
from experiment.experiment_vae_1d import ExperimentVAE1d
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset.my_dataset import MyDataset

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.core.saving import save_hparams_to_yaml

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae_1d.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'], **config['data_params'])
experiment = ExperimentVAE1d(model,
                             config['exp_params'], config["data_params"])

# use **config to represent a bunch of parameters instead of one parameter?
data = MyDataset(**config["data_params"], **config['model_params'], pin_memory=len(config['trainer_params']['gpus']) != 0)
print("start setting up data")
data.setup()
print("data ready!")
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
    with open(r'/home/anjian/Desktop/project/PyTorch-VAE/configs/vae_1d.yaml') as file:
        vae_1d_config = yaml.load(file, Loader=yaml.FullLoader)

    path_file = tb_logger.log_dir + "/" + "vae_1d_hparam.yaml"
    with open(path_file, 'w') as file:
        documents = yaml.dump(vae_1d_config, file)
elif config["model_params"]["name"] == "VAEConv1d":
    with open(r'/home/anjian/Desktop/project/PyTorch-VAE/configs/vae_conv1d.yaml') as file:
        vae_1d_config = yaml.load(file, Loader=yaml.FullLoader)

    path_file = tb_logger.log_dir + "/" + "vae_conv1d_hparam.yaml"
    with open(path_file, 'w') as file:
        documents = yaml.dump(vae_1d_config, file)

runner.fit(experiment, datamodule=data)
