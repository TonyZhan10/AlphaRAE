"""Frozen semantic RAE plus a learned low-resolution reconstruction-residual codec."""
import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import premultiply


class ResidualCodec(nn.Module):
    def __init__(self, base, detail_dim=256, residual_scale=.1):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.variant = "RESIDUAL_CODEC"
        self.detail_dim = detail_dim
        self.residual_scale = residual_scale
        self.num_tokens = base.num_tokens
        self.latent_dim = base.latent_dim + detail_dim
        channels = [4, 64, 128, 192, detail_dim]
        layers = []
        for source, target in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(source, target, 4, stride=2, padding=1))
            if target != detail_dim:
                layers.append(nn.SiLU())
        self.detail_encoder = nn.Sequential(*layers)
        self.detail_decoder = nn.ModuleList([
            nn.Conv2d(detail_dim, 192, 3, padding=1),
            nn.Conv2d(192, 128, 3, padding=1),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1),
        ])
        self.output = nn.Conv2d(32, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("stats_fitted", torch.tensor(False))

    @property
    def encoder(self):
        return self.base.encoder

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def observations(self, rgba):
        semantic = self.base.encode(rgba)
        return semantic, self.base.decode(semantic)

    @torch.no_grad()
    def fit_feature_stats(self, semantic, prior, std_floor=.01):
        self.stats_fitted.fill_(True)

    def encode(self, rgba, observations=None):
        semantic, prior = self.observations(rgba) if observations is None else observations
        p, alpha = premultiply(rgba)
        prior_p, prior_alpha = premultiply(prior)
        error = torch.cat((p - prior_p, alpha - prior_alpha), dim=1) / self.residual_scale
        detail = self.detail_encoder(error).flatten(2).transpose(1, 2)
        if detail.shape[1] != self.num_tokens:
            raise ValueError("Residual codec geometry must match the semantic latent grid")
        return torch.cat((semantic.detach(), detail), dim=-1)

    def decode(self, z):
        semantic, detail = z.split((self.base.latent_dim, self.detail_dim), dim=-1)
        with torch.no_grad():
            prior = self.base.decode(semantic)
        side = self.encoder.resolution // 16
        x = detail.transpose(1, 2).reshape(len(z), self.detail_dim, side, side)
        for layer in self.detail_decoder:
            x = F.silu(layer(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)))
        residual = self.output(x).float() * self.residual_scale
        prior_p, prior_alpha = premultiply(prior.float())
        alpha = (prior_alpha + residual[:, 3:]).clamp(0, 1)
        p = torch.minimum((prior_p + residual[:, :3]).clamp_min(0), alpha)
        rgb = torch.where(alpha > 0, p / alpha.clamp_min(1e-8), 0)
        return torch.cat((rgb, alpha), dim=1)

    def forward(self, rgba, observations=None):
        z = self.encode(rgba, observations)
        return self.decode(z), z


def build_residual_codec(config):
    from alpharae.models.autoencoder import build_autoencoder
    base = build_autoencoder(config["base_model_config"])
    return ResidualCodec(base, config.get("detail_dim", 256), config.get("residual_scale", .1))
