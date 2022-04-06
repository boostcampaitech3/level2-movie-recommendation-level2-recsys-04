import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout, device, num_enc=7, num_dec=5):
        super(FLVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        enc_list = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_enc-2):
            enc_list.append(nn.Linear(hidden_dim, hidden_dim))
        enc_list.append(nn.Linear(hidden_dim, 2 * hidden_dim))
        self.encoder = nn.ModuleList(enc_list)

        dec_list = []
        for _ in range(num_dec-1):
            dec_list.append(nn.Linear(hidden_dim, hidden_dim))
        dec_list.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.ModuleList(dec_list)

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x):
        h = F.normalize(x)
        h = self.drop(x)

        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i != len(self.encoder) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.hidden_dim]
                logvar = h[:, self.hidden_dim:]
        return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i != len(self.decoder) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def init_weights(self):
        for layer in self.encoder:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.decoder:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class EASE(nn.Module):
    def __init__(self, input_dim, device):
        super(EASE, self).__init__()
        self.encoder = nn.Linear(input_dim, input_dim, bias=False)

        # constraint diagonal zero
        self.const_eye_zero = torch.ones((input_dim, input_dim), device=device)
        self.diag = torch.eye(input_dim, dtype=torch.bool, device=device)

    def forward(self, x):
        # setting diagonal weight to zero
        self._set_diag_zero()

        output = self.encoder(x)

        return output

    def _set_diag_zero(self):
        self.encoder.weight.data[self.diag] = 0.


class VASP(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout, device, num_enc=7, num_dec=5):
        super(VASP, self).__init__()
        self.ease = EASE(input_dim=input_dim, device=device)
        self.flvae = FLVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim,
                           dropout=dropout, device=device, num_enc=num_enc, num_dec=num_dec)

    def forward(self, x):
        ease_y = self.ease(x)
        ease_y = torch.sigmoid(ease_y)

        flvae_y, mu, logvar = self.flvae(x)
        flvae_y = torch.sigmoid(flvae_y)
        output = torch.mul(flvae_y, ease_y)

        return output, mu, logvar


def loss_function_vasp(recon_x, x, mu, logvar, alpha=0.25, gamma=2.0):
    # mll = (F.log_softmax(x_pred, dim=-1) * user_ratings)
    # F_loss = torch.pow(1 - torch.exp(mll), r) * mll
    mll = (F.log_softmax(recon_x, 1) * x)
    F_loss = alpha * torch.pow(1 - torch.exp(mll), gamma) * mll
    BCE = -F_loss.sum(dim=-1).mean()
    # BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + KLD, BCE, KLD

