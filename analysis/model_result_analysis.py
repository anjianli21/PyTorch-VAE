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
from experiment.experiment_vae_1d import ExperimentVAE1d
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

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
                             config['exp_params'])

# ======================================================================================================================

PATH = "/home/anjian/Desktop/project/PyTorch-VAE/logs/VAE1d/version_147/checkpoints/epoch=59-step=93779.ckpt"
test_model = experiment.load_from_checkpoint(PATH)

samples = test_model.model.sample(100000, current_device="cpu")
samples = samples.cpu().data.numpy()
plt.hist(samples[:, 15])
plt.show()

print(f"sample mean is {np.mean(samples, axis=0)}")
print(f"varianace is {np.var(samples, axis=0)}")

sample_cov = np.cov(samples, rowvar=False)

plt.clf()
plt.matshow(sample_cov)
plt.colorbar()
plt.show()
