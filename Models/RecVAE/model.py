"""
Model Reference : https://github.com/ilya-shenbin/RecVAE
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_norm_pdf(x, mu, logvar):
    # torch.Parameter __add__ warning
    return -0.5 * (torch.add(logvar, np.log(2 * np.pi)) + (x - mu).pow(2) / torch.exp(logvar))


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleDict()

        # original code is too long
        for i in range(0, 5):
            if i == 0:
                self.encoder[f"fc{i+1}"] = nn.Linear(input_dim, hidden_dim)
                self.encoder[f"ln{i+1}"] = nn.LayerNorm(hidden_dim, eps=eps)
            else:
                self.encoder[f"fc{i+1}"] = nn.Linear(hidden_dim, hidden_dim)
                self.encoder[f"ln{i+1}"] = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_ratio):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_ratio, training=self.training)

        h1 = self.encoder['ln1'](self.swish(self.encoder['fc1'](x)))
        h2 = self.encoder['ln2'](self.swish(self.encoder['fc2'](h1) + h1))
        h3 = self.encoder['ln3'](self.swish(self.encoder['fc3'](h2) + h1 + h2))
        h4 = self.encoder['ln4'](self.swish(self.encoder['fc4'](h3) + h1 + h2 + h3))
        h5 = self.encoder['ln5'](self.swish(self.encoder['fc5'](h4) + h1 + h2 + h3 + h4))

        return self.fc_mu(h5), self.fc_logvar(h5)

    @staticmethod
    def swish(x):
        return x.mul(torch.sigmoid(x))


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=None):
        super(CompositePrior, self).__init__()

        if mixture_weights is None:
            mixture_weights = [3/20, 3/4, 1/10]

        self.mixture_weights = mixture_weights
        # self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        # self.mu_prior.data.fill_(0)
        #
        # self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        # self.logvar_prior.data.fill_(0)
        self.mu_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim, 0)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(RecVAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            return torch.mul(eps, std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, gamma=1, beta=None, dropout_ratio=0.5, cal_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_ratio)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        if cal_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
