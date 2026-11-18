import math

import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import composite
from alpharae.models.frozen_encoder import FrozenEncoder


def position_encoding(n, width):
    position = torch.arange(n).float()[:, None]
    frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
    out = torch.zeros(n, width)
    out[:, 0::2] = torch.sin(position * frequency)
    out[:, 1::2] = torch.cos(position * frequency[:out[:, 1::2].shape[1]])
    return out[None]


class RGBADecoder(nn.Module):
    def __init__(self, resolution, patch_size, latent_dim, width=256, depth=4, heads=8):
        super().__init__()
        self.resolution, self.patch_size = resolution, patch_size
        n = (resolution // patch_size) ** 2
        self.input = nn.Linear(latent_dim, width)
        self.register_buffer("position", position_encoding(n, width))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, patch_size ** 2 * 4)

    def forward(self, z):
        x = self.input(z) + self.position
        for block in self.blocks:
            x = block(x)
        pixels = self.output(self.norm(x)).transpose(1, 2)
        return F.fold(pixels, (self.resolution, self.resolution), self.patch_size,
                      stride=self.patch_size).sigmoid()


class AssetAutoencoder(nn.Module):
    def __init__(self, encoder, latent_dim=None, variant="B1", decoder=None):
        super().__init__()
        if variant not in ("B0", "B1", "B2", "B3", "M1", "C1", "C2", "C3"):
            raise ValueError(f"Unknown variant {variant}")
        self.encoder, self.variant = encoder, variant
        d = encoder.dim
        self.latent_dim = latent_dim or d
        self.num_tokens = (encoder.resolution // encoder.patch_size) ** 2
        self.register_buffer("feature_mean", torch.zeros(1, 1, d))
        self.register_buffer("feature_std", torch.ones(1, 1, d))
        self.register_buffer("stats_fitted", torch.tensor(False))
        self.aggregator = nn.Sequential(nn.Linear(2 * d, self.latent_dim),
                                        nn.LayerNorm(self.latent_dim), nn.GELU(),
                                        nn.Linear(self.latent_dim, self.latent_dim))
        self.decoder = RGBADecoder(encoder.resolution, encoder.patch_size,
                                   self.latent_dim, **(decoder or {}))
        # Construct this after the shared modules to preserve B1/M1 initialization.
        self.alpha_branch = (nn.Conv2d(1, d, encoder.patch_size, stride=encoder.patch_size)
                             if variant == "B2" else None)

    @torch.no_grad()
    def observations(self, rgba):
        black = self.encoder(composite(rgba, torch.zeros_like(rgba[:, :3])))
        white = self.encoder(composite(rgba, torch.ones_like(rgba[:, :3])))
        return black, white

    @torch.no_grad()
    def fit_feature_stats(self, black, white, std_floor=.01):
        features = torch.cat((black, white), dim=0).float()
        self.feature_mean.copy_(features.mean((0, 1), keepdim=True))
        self.feature_std.copy_(features.std((0, 1), correction=0, keepdim=True).clamp_min(std_floor))
        self.stats_fitted.fill_(True)

    def encode(self, rgba, observations=None):
        if not self.stats_fitted:
            raise RuntimeError("Fit shared feature statistics on training assets first")
        black, white = self.observations(rgba) if observations is None else observations
        black = (black - self.feature_mean) / self.feature_std
        white = (white - self.feature_mean) / self.feature_std
        if self.variant == "B0":
            black = white
        elif self.variant == "B2":
            black = self.alpha_branch(rgba[:, 3:4]).flatten(2).transpose(1, 2)
        elif self.variant == "B3":
            extra = [self.encoder(composite(rgba, torch.full_like(rgba[:, :3], v)))
                     for v in (.25, .75)]
            average = (black + white + sum((x - self.feature_mean) / self.feature_std
                                           for x in extra)) / 4
            black = white = average
        return self.aggregator(torch.cat((black, white), dim=-1))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, rgba, observations=None):
        z = self.encode(rgba, observations)
        return self.decode(z), z


def build_autoencoder(config):
    if config.get("model_type") == "raev2_joint_full":
        from alpharae.models.raev2_joint_lora import build_joint_full
        return build_joint_full(config)
    if config.get("model_type") == "raev2_joint_lora":
        from alpharae.models.raev2_joint_lora import build_joint_lora
        return build_joint_lora(config)
    if config.get("model_type") == "raev2_rgba":
        from alpharae.models.raev2_rgba import build_raev2_rgba
        return build_raev2_rgba(config)
    if config.get("model_type") == "linear_residual":
        from alpharae.models.linear_residual import build_linear_residual
        return build_linear_residual(config)
    if config.get("model_type") == "residual_codec":
        from alpharae.models.residual_codec import build_residual_codec
        return build_residual_codec(config)
    if config.get("model_type") == "pretrained_rgba":
        from alpharae.models.pretrained_rgba import build_pretrained_rgba
        return build_pretrained_rgba(config)
    encoder = FrozenEncoder(**config["encoder"])
    return AssetAutoencoder(encoder, config.get("latent_dim"), config.get("variant", "B1"),
                            config.get("decoder"))
