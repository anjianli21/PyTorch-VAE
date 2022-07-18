import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

import numpy as np
import copy


class VAEConv1d(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 data_dim: int,
                 hidden_channels: List = None,
                 **kwargs) -> None:
        super(VAEConv1d, self).__init__()

        self.hidden_channels = copy.copy(hidden_channels)
        self.latent_dim = copy.copy(latent_dim)
        self.init_in_channels = copy.copy(in_channels)

        self.len_in = data_dim
        self.len_out = None

        modules = []
        if hidden_channels is None:
            hidden_channels = [32, 16]

        len_in = copy.copy(self.len_in)

        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1

        # Build Encoder
        for h_channel in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_channel,  kernel_size=kernel_size, stride=stride, padding=padding), # in_channel: Number of channels in the input image,
                              # initial in_channel is 3, then is [32, 64, 128, 256, 512], between each is a conv2d
                    nn.BatchNorm1d(h_channel),
                    nn.LeakyReLU())
            )
            in_channels = h_channel

        # Count the length of node after the encoder
        len_out = None
        for i in range(len(hidden_channels)):
            len_out = np.floor((len_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            len_in = len_out

        self.len_out = int(len_out)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_channels[-1] * int(self.len_out), latent_dim)  # Use fully connected layers to get latent dim
        self.fc_var = nn.Linear(hidden_channels[-1] * int(self.len_out), latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_channels[-1] * int(len_out))

        hidden_channels.reverse()  # decoder is just the reverse of the encoder

        for i in range(len(hidden_channels) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_channels[i],
                                       hidden_channels[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(hidden_channels[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels[-1],
                               hidden_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # This module can be seen as the gradient of Conv2d with respect to its input.
            nn.BatchNorm1d(hidden_channels[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_channels[-1], out_channels=self.init_in_channels,
                      kernel_size=3, padding=1))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input.size())
        # print(self.encoder)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_channels[-1], self.len_out)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)  # exp( (1/2) * log (var) ) = (var)^(1/2)
        eps = torch.randn_like(std)  # normal distribution with mean 0 and variance 1.
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # print("\n")
        # print("mu is", mu.size(), mu)
        # print("log_var is", log_var.size(), log_var)
        # print("input is", input.size(), input)
        # print("reconstruction is", recons.size(), recons)


        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)

        samples = torch.squeeze(samples)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
