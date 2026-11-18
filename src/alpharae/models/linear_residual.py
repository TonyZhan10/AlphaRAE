"""Frozen semantic RAE and a train-fitted linear reconstruction-residual code."""
import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import premultiply


def packed_rgba(rgba):
    return torch.cat(premultiply(rgba), dim=1)


class LinearResidualCodec(nn.Module):
    def __init__(self, base, detail_dim=256, patch_size=16, mean=None, basis=None):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.variant = "LINEAR_RESIDUAL"
        self.detail_dim, self.patch_size = detail_dim, patch_size
        self.num_tokens = base.num_tokens
        self.latent_dim = base.latent_dim + detail_dim
        self.register_buffer("residual_mean", torch.zeros(4 * patch_size ** 2))
        self.register_buffer("residual_basis", torch.zeros(4 * patch_size ** 2, detail_dim))
        self.register_buffer("basis_fitted", torch.tensor(False))
        if mean is not None and basis is not None:
            self.residual_mean.copy_(mean)
            self.residual_basis.copy_(basis)
            self.basis_fitted.fill_(True)

    @property
    def encoder(self):
        return self.base.encoder

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def observations(self, rgba):
        semantic = self.base.encode(rgba)
        return semantic, self.base.decode(semantic)

    @torch.no_grad()
    def encode(self, rgba, observations=None):
        if not self.basis_fitted:
            raise RuntimeError("Load an exported checkpoint with a fitted residual basis")
        semantic, prior = self.observations(rgba) if observations is None else observations
        patches = F.unfold(packed_rgba(rgba) - packed_rgba(prior), self.patch_size,
                           stride=self.patch_size).transpose(1, 2)
        coefficients = (patches - self.residual_mean) @ self.residual_basis
        return torch.cat((semantic, coefficients), dim=-1)

    def decode(self, latent):
        if not self.basis_fitted:
            raise RuntimeError("Load an exported checkpoint with a fitted residual basis")
        semantic, coefficients = latent.split((self.base.latent_dim, self.detail_dim), dim=-1)
        prior = self.base.decode(semantic)
        patches = coefficients @ self.residual_basis.T + self.residual_mean
        residual = F.fold(patches.transpose(1, 2), prior.shape[-2:], self.patch_size, stride=self.patch_size)
        output = packed_rgba(prior) + residual
        alpha = output[:, 3:].clamp(0, 1)
        p = torch.minimum(output[:, :3].clamp_min(0), alpha)
        rgb = torch.where(alpha > 0, p / alpha.clamp_min(1e-8), 0)
        return torch.cat((rgb, alpha), dim=1)

    def forward(self, rgba, observations=None):
        latent = self.encode(rgba, observations)
        return self.decode(latent), latent


def build_linear_residual(config):
    from alpharae.models.autoencoder import build_autoencoder
    return LinearResidualCodec(build_autoencoder(config["base_model_config"]),
                               config.get("detail_dim", 256), config.get("detail_patch_size", 16))
