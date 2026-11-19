"""Strict loaders for the class-conditioned RGB prior and RGB-to-RGBA adapter."""

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from alpharae.runtime import ROOT, activate_sources, load_codec

activate_sources()
from configs.stage2 import ConditioningArchConfig
from stage2.models.DDT import DiTwDDTHead, DiTwDDTHeadIG


class WideJointRefiner(nn.Module):
    def __init__(self, width=1536):
        super().__init__()
        self.input = nn.Conv2d(1088 + 1024 + 768, width, 1)
        self.norm = nn.GroupNorm(32, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.mix = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, 1088, 1)

    def forward(self, prediction, context, rgb):
        pixels = F.pixel_unshuffle(rgb.float() * 2 - 1, 16)
        value = torch.cat((prediction.float(), context.float(), pixels), 1)
        value = F.silu(self.norm(self.input(value)))
        return self.output(value + F.silu(self.mix(self.spatial(value))))


class ImageNetRGBAAdapter(DiTwDDTHead):
    """Existing image-only adapter, with the selected EMA loaded strictly."""

    def __init__(self, widths, depths, heads, refiner_width=1536):
        super().__init__(input_size=16, in_channels=1088, patch_size=[1, 1],
                         hidden_size=list(widths), depth=list(depths), num_heads=list(heads),
                         mlp_ratio=4., num_classes=1, condition_type='label',
                         cond_arch=ConditioningArchConfig(num_t_tokens=4, num_c_tokens=1))
        for embedder in (self.s_embedder, self.x_embedder):
            embedder.proj = nn.Conv2d(1088 + 1024 + 768, embedder.proj.out_channels, 1)
        self.refiner = WideJointRefiner(refiner_width)

    def forward(self, value, time, context, rgb):
        inputs = torch.cat((value, context, F.pixel_unshuffle(rgb.float() * 2 - 1, 16)), 1)
        prediction = super().forward(inputs, time,
                                     context=torch.zeros(len(value), device=value.device, dtype=torch.long))
        return prediction.float() + self.refiner(prediction, context, rgb).float()


class ResidualSpace:
    def __init__(self, stats, device):
        self.mean = stats['mean'].to(device)
        self.scale = (stats['var'].to(device) + stats['epsilon']).sqrt()
        self.detail_mean = stats['detail_mean'].to(device)
        self.detail_std = stats['detail_std'].to(device)

    def condition(self, raw):
        return (raw.float() - self.mean) / self.scale

    def compose(self, residual, raw):
        return torch.cat((raw.float() + residual[:, :1024].float() * self.scale,
                          residual[:, 1024:].float() * self.detail_std + self.detail_mean), 1).flatten(2).transpose(1, 2)


def load_pipeline(weights_dir, device):
    weights = Path(weights_dir)
    codec = load_codec(weights / 'codec_k7.pt', device)
    # Native class sampling provides its semantic latent directly.
    codec.encoder.cpu()
    rgb_decoder = copy.deepcopy(codec.decoder)
    initializer = torch.load(weights / 'training/rgb_k7_init.pt', map_location='cpu', weights_only=True, mmap=True)
    rgb_decoder.load_state_dict(initializer['ema_decoder'], strict=True)
    del initializer
    package = torch.load(weights / 'generation/imagenet_dit.pt', map_location='cpu', weights_only=True, mmap=True)
    architecture = dict(package['architecture'])
    if architecture['condition_type'] != 'label' or architecture['num_classes'] != 1000:
        raise ValueError('This entrypoint uses the ImageNet class-conditioned RGB prior')
    architecture['cond_arch'] = ConditioningArchConfig(**architecture['cond_arch'])
    with torch.device('meta'):
        prior = DiTwDDTHeadIG(**architecture)
    prior.load_state_dict(package['ema'], strict=True, assign=True)
    prior = prior.to(device).eval().requires_grad_(False)
    del package
    package = torch.load(weights / 'generation/rgba_adapter.pt', map_location='cpu', weights_only=True, mmap=True)
    adapter = ImageNetRGBAAdapter(**package['architecture'])
    adapter.load_state_dict(package['model'], strict=True)
    adapter = adapter.to(device).eval().requires_grad_(False)
    del package
    stats = torch.load(weights / 'generation/latent_stats.pt', map_location='cpu', weights_only=True)
    return codec, rgb_decoder, prior, adapter, ResidualSpace(stats, device)
