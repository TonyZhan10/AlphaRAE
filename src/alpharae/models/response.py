import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import composite
from alpharae.models.autoencoder import position_encoding


class ResponseSupervisor(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_tokens, response_dim=128,
                 width=256, depth=2, heads=8, projection_seed=2026, std_floor=.05):
        super().__init__()
        generator = torch.Generator().manual_seed(projection_seed)
        self.register_buffer("projection", torch.randn(feature_dim, response_dim, generator=generator)
                             / response_dim ** .5)
        self.register_buffer("response_std", torch.ones(response_dim))
        self.register_buffer("stats_fitted", torch.tensor(False))
        self.std_floor = std_floor
        self.input = nn.Linear(latent_dim + 2 * response_dim, width)
        self.register_buffer("position", position_encoding(num_tokens, width))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, response_dim))

    @torch.no_grad()
    def targets(self, encoder, rgba, bi, bj):
        hi, hj = encoder(composite(rgba, bi)), encoder(composite(rgba, bj))
        gi, gj = encoder(bi) @ self.projection, encoder(bj) @ self.projection
        return (hj - hi) @ self.projection, gi, gj

    @torch.no_grad()
    def fit_response_stats(self, targets):
        self.response_std.copy_(targets.float().std((0, 1), correction=0).clamp_min(self.std_floor))
        self.stats_fitted.fill_(True)

    def forward(self, z, gi, gj):
        x = self.input(torch.cat((z, gi, gj), -1)) + self.position
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    def loss(self, z, target, gi, gj, rgba, patch_size, detach_latent=False, background_only=False):
        if not self.stats_fitted:
            raise RuntimeError("Fit response scaling on training assets first")
        source = torch.zeros_like(z) if background_only else (z.detach() if detach_latent else z)
        prediction = self(source, gi, gj)
        error = F.smooth_l1_loss(prediction / self.response_std,
                                 target.detach() / self.response_std, reduction="none").mean(-1)
        occupancy = F.avg_pool2d((rgba[:, 3:4] > .01).float(), patch_size).flatten(1)
        weights = .1 + .9 * occupancy
        return (error * weights).sum() / weights.sum()
