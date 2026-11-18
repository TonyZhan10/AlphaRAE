import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import linear_to_srgb, premultiply


def critic_input(rgba, input_space="linear_premult"):
    if input_space == "srgb_premult":
        return torch.cat((linear_to_srgb(rgba[:, :3]) * rgba[:, 3:], rgba[:, 3:]), dim=1) * 2 - 1
    premultiplied, alpha = premultiply(rgba)
    return torch.cat((linear_to_srgb(premultiplied), alpha), dim=1) * 2 - 1


class PatchAdversarialLoss(nn.Module):
    def __init__(self, weight=.05, start_step=500, ramp_steps=500,
                 discriminator_lr=1e-4, adaptive_max=10., base_channels=64,
                 input_space="linear_premult"):
        super().__init__()
        from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

        self.weight = weight
        self.start_step = start_step
        self.ramp_steps = ramp_steps
        self.discriminator_lr = discriminator_lr
        self.adaptive_max = adaptive_max
        if input_space not in ("linear_premult", "srgb_premult"):
            raise ValueError(f"Unknown discriminator input space: {input_space}")
        self.input_space = input_space
        self.discriminator = NLayerDiscriminator(input_nc=4, n_layers=3,
                                                  ndf=base_channels).apply(weights_init)

    def active(self, step):
        return self.weight > 0 and step >= self.start_step

    def generator_loss(self, pred, reconstruction_loss, last_layer, step):
        if not self.active(step):
            return {key: pred.new_zeros(()) for key in
                    ("gan_raw", "gan_adaptive_weight", "gan_weighted")}

        self.discriminator.requires_grad_(False).eval()
        raw = -self.discriminator(critic_input(pred, self.input_space)).mean()
        reconstruction_grad = torch.autograd.grad(reconstruction_loss, last_layer,
                                                   retain_graph=True)[0]
        adversarial_grad = torch.autograd.grad(raw, last_layer, retain_graph=True)[0]
        adaptive = (reconstruction_grad.norm() / (adversarial_grad.norm() + 1e-4))
        adaptive = adaptive.clamp(0, self.adaptive_max).detach()
        ramp = min(1., (step - self.start_step) / self.ramp_steps) if self.ramp_steps > 0 else 1.
        return {"gan_raw": raw, "gan_adaptive_weight": adaptive,
                "gan_weighted": raw * adaptive * self.weight * ramp}

    def discriminator_loss(self, real, pred):
        self.discriminator.requires_grad_(True).train()
        real_logits = self.discriminator(critic_input(real.detach(), self.input_space))
        fake_logits = self.discriminator(critic_input(pred.detach(), self.input_space))
        loss = .5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())
        return {"discriminator_loss": loss,
                "discriminator_real": real_logits.detach().mean(),
                "discriminator_fake": fake_logits.detach().mean()}
