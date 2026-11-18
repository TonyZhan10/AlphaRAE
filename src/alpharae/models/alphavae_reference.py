"""Official AlphaVAE inference adapted to the project's linear RGBA interface."""

import torch
from torch import nn

from alpharae.data.rgba import linear_to_srgb, srgb_to_linear


class AlphaVAEReference(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, rgba):
        encoded_rgba = torch.cat((linear_to_srgb(rgba[:, :3]), rgba[:, 3:4]), dim=1)
        # Official AutoencoderKL.forward defaults to posterior mode for reconstruction.
        return self.vae.encode(encoded_rgba * 2 - 1).latent_dist.mode()

    @torch.no_grad()
    def decode(self, latent):
        encoded_rgba = (self.vae.decode(latent).sample.float() * .5 + .5).clamp(0, 1)
        return torch.cat((srgb_to_linear(encoded_rgba[:, :3]), encoded_rgba[:, 3:4]), dim=1)

    @torch.no_grad()
    def forward(self, rgba):
        latent = self.encode(rgba)
        return self.decode(latent), latent
