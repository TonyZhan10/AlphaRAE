"""Adapt the trained native1024 RAEv2 RGB decoder to the AlphaRAE interface."""

from functools import partial
from math import isqrt, log2
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from alpharae.data.rgba import composite, linear_to_srgb, srgb_to_linear


class SDPAAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.query, self.key, self.value = original.query, original.key, original.value
        self.dropout = original.dropout
        self.num_attention_heads = original.num_attention_heads
        self.attention_head_size = original.attention_head_size
        self.all_head_size = original.all_head_size

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        if head_mask is not None or output_attentions:
            raise ValueError("This reconstruction backend uses unmasked SDPA")
        shape = (*hidden_states.shape[:2], self.num_attention_heads, self.attention_head_size)
        q, k, v = (layer(hidden_states).view(shape).transpose(1, 2)
                   for layer in (self.query, self.key, self.value))
        value = F.scaled_dot_product_attention(q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0)
        return (value.transpose(1, 2).reshape(*hidden_states.shape[:2], self.all_head_size),)


def extend_rgba_head(decoder, alpha_init_std=0.):
    old = decoder.decoder_pred
    patch, width = decoder.config.patch_size, old.in_features
    head = nn.Linear(width, patch * patch * 4, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight.view(patch, patch, 4, width)[:, :, :3].copy_(old.weight.view(patch, patch, 3, width))
        head.bias.view(patch, patch, 4)[:, :, :3].copy_(old.bias.view(patch, patch, 3))
        if alpha_init_std:
            head.weight.view(patch, patch, 4, width)[:, :, 3].normal_(std=alpha_init_std)
    decoder.decoder_pred = head
    decoder.config.num_channels = 4
    return decoder


class LinearLightRAEv2Encoder(nn.Module):
    def __init__(self, backbone, resolution):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.resolution, self.patch_size, self.dim = resolution, backbone.patch_size, backbone.hidden_size
        self.num_tokens = (resolution // self.patch_size) ** 2
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, linear_rgb):
        return self.backbone(linear_to_srgb(linear_rgb).float() * 255.0).float()


class SpatialAlphaReadout(nn.Module):
    """Overlapping alpha-logit correction decoded only from the existing latent."""

    def __init__(self, latent_dim, patch_size, condition_coarse=False):
        super().__init__()
        levels = int(log2(patch_size))
        if 2 ** levels != patch_size:
            raise ValueError("Spatial readout requires a power-of-two patch size")
        channels = 128
        layers = [nn.Conv2d(latent_dim, channels, 1), nn.GELU()]
        for _ in range(levels):
            following = max(8, channels // 2)
            layers += [nn.Conv2d(channels, following * 4, 3, padding=1),
                       nn.PixelShuffle(2), nn.GELU()]
            channels = following
        self.features = nn.Sequential(*layers)
        self.condition_coarse = condition_coarse
        self.output = nn.Conv2d(channels + int(condition_coarse), 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, coarse_logits=None):
        grid = isqrt(latent.shape[1])
        image = latent.transpose(1, 2).reshape(latent.shape[0], latent.shape[2], grid, grid)
        features = self.features(image)
        if self.condition_coarse:
            features = torch.cat((features, coarse_logits.to(features.dtype)), dim=1)
        return self.output(features)


class CompressedAlphaEncoder(nn.Sequential):
    """Learned spatial alpha encoding;64 channels per16x16 patch is4:1 compression."""

    def __init__(self, channels, patch_size):
        layers=[]
        incoming=1
        levels=int(log2(patch_size))
        for level in range(levels):
            outgoing=channels if level == levels-1 else min(128,32*2**level)
            layers.append(nn.Conv2d(incoming,outgoing,3,stride=2,padding=1))
            if level < levels-1:
                layers.append(nn.GELU())
            incoming=outgoing
        super().__init__(*layers)


class RAEv2RGBAAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, adapter="learned_full", alpha_scale=0.1, alpha_head_init_std=0.,
                 spatial_alpha=False, alpha_codec_channels=0):
        super().__init__()
        self.encoder, self.decoder = encoder, extend_rgba_head(decoder, alpha_head_init_std)
        self.resolution, self.base_latent_dim = encoder.resolution, encoder.dim
        self.latent_dim = self.base_latent_dim + alpha_codec_channels
        self.num_tokens, self.patch_size = encoder.num_tokens, encoder.patch_size
        self.adapter, self.alpha_scale = adapter, alpha_scale
        self.train_spatial_alpha_only = False
        alpha_dim = self.patch_size ** 2
        if adapter == "learned_full":
            self.alpha_projection = nn.Linear(alpha_dim, self.base_latent_dim, bias=False)
            nn.init.zeros_(self.alpha_projection.weight)
        elif adapter == "fixed_isometric":
            if self.latent_dim < alpha_dim or alpha_scale <= 0:
                raise ValueError("Isometric opacity requires sufficient channels and positive scale")
            self.alpha_projection = None
        else:
            raise ValueError(f"Unknown alpha adapter: {adapter}")
        self.register_buffer("stats_fitted", torch.tensor(True))
        self.alpha_encoder = CompressedAlphaEncoder(alpha_codec_channels, self.patch_size) if alpha_codec_channels else None
        if alpha_codec_channels and not spatial_alpha:
            raise ValueError("The alpha codec requires its spatial decoder")
        self.alpha_refiner = (SpatialAlphaReadout(alpha_codec_channels or self.base_latent_dim, self.patch_size,
                                                 condition_coarse=bool(alpha_codec_channels)) if spatial_alpha else None)

    @torch.no_grad()
    def observations(self, rgba):
        return (self.encoder(composite(rgba, torch.zeros_like(rgba[:, :3]))),
                self.encoder(composite(rgba, torch.ones_like(rgba[:, :3]))))

    @torch.no_grad()
    def fit_feature_stats(self, black, white, std_floor=.01):
        self.stats_fitted.fill_(True)

    def encode(self, rgba, observations=None):
        if rgba.shape[1:] != (4, self.resolution, self.resolution):
            raise ValueError("RAEv2 RGBA input must match its configured four-channel resolution")
        white = (self.encoder(composite(rgba, torch.ones_like(rgba[:, :3])))
                 if observations is None else observations[1])
        opacity = F.pixel_unshuffle(1 - rgba[:, 3:4], self.patch_size).flatten(2).transpose(1, 2)
        if self.alpha_projection is not None:
            residual = self.alpha_projection(opacity)
        else:
            residual = F.pad(opacity, (0, self.base_latent_dim - opacity.shape[-1])) * self.alpha_scale
        latent = white.float() + residual.float()
        if self.alpha_encoder is not None:
            alpha_code = self.alpha_encoder(rgba[:,3:]).flatten(2).transpose(1,2)
            latent = torch.cat((latent,alpha_code.float()),dim=-1)
        return latent

    def decode(self, z, *, detach_base=False):
        if z.shape[1:] != (self.num_tokens, self.latent_dim):
            raise ValueError("RAEv2 latent shape does not match the decoder grid")
        base = z[:,:,:self.base_latent_dim]
        if detach_base:
            base = base.detach()
        pixels = self.decoder.unpatchify(self.decoder(base, drop_cls_token=False).logits).float()
        # The migrated decoder predicts sRGB directly; older DINOv2 calibration is not applicable.
        rgb = srgb_to_linear(pixels[:, :3].clamp(0, 1))
        alpha_logits = pixels[:, 3:4]
        if self.alpha_refiner is not None:
            detail = z[:,:,self.base_latent_dim:] if self.alpha_encoder is not None else base
            alpha_logits = alpha_logits + self.alpha_refiner(detail,alpha_logits).float()
        return torch.cat((rgb, alpha_logits.sigmoid()), dim=1)

    def forward(self, rgba, observations=None, decode_count=None):
        z = self.encode(rgba, observations)
        # Concatenating a trainable alpha code marks even the frozen RGB slice
        # as requiring grad; direct decode(z) retains gradients for latent fitting.
        return self.decode(z if decode_count is None else z[:decode_count],
                           detach_base=self.train_spatial_alpha_only), z


def load_trained_rgb(config):
    source = Path(config["raev2_source"]).resolve()
    sys.path.insert(0, str(source / "src"))
    os.environ["DINOV3_REPO_DIR"] = str(Path(config["dinov3_source"]).resolve())
    os.environ["DINOV3_CKPT_DIR"] = str(Path(config["encoder"]["weights"]).resolve())
    spec = importlib.util.spec_from_file_location("alpharae_raev2_source", source / "src/stage1/__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = (torch.load(config["pretrained_decoder"], map_location="cpu", weights_only=False, mmap=True)
             if config.get("load_pretrained_rgb", True) else None)
    resolution = config["encoder"]["resolution"]
    if state is not None and state["resolution"] != resolution:
        raise ValueError("Use the native-resolution RGB EMA for this migration")
    encoder_names = {
        "dinov3l_k23": "dinov3mls-vit-l16[layers=1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23]",
        "dinov3l_k7": "dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
    }
    kind = config["encoder"]["kind"]
    if state is not None and state.get("encoder_kind", kind) != kind:
        raise ValueError("RGB initializer and configured encoder must use the same layer aggregation")
    rgb = module.RAE(
        encoder_name=encoder_names[kind],
        resolution=resolution, decoder_config_path=str(source / "configs/decoder/ViTXL"),
        noise_tau=0.0, normalization_stat_path=None,
        pretrained_encoder=config.get("load_pretrained_encoder", True))
    if state is not None:
        rgb.decoder.load_state_dict(state["ema_decoder"], strict=True)
    del state
    for layer in rgb.decoder.decoder_layers:
        layer.attention.attention = SDPAAttention(layer.attention.attention)
    rgb.decoder.gradient_checkpointing = config.get("decoder_gradient_checkpointing", True)
    rgb.decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    rgb.decoder.requires_grad_(True)
    rgb.encoder.requires_grad_(False)
    return rgb


def build_raev2_rgba(config):
    rgb = load_trained_rgb(config)
    encoder = LinearLightRAEv2Encoder(rgb.encoder, rgb.resolution)
    model = RAEv2RGBAAutoencoder(encoder, rgb.decoder,
                              config.get("alpha_adapter", "learned_full"), config.get("alpha_scale", .1),
                              config.get("alpha_head_init_std", 0.), config.get("spatial_alpha", False),
                              config.get("alpha_codec_channels", 0))
    if config.get("train_spatial_alpha_only"):
        if model.alpha_refiner is None or model.alpha_projection is not None:
            raise ValueError("Spatial-only repair requires a spatial readout and fixed opacity encoder")
        model.decoder.requires_grad_(False)
        model.train_spatial_alpha_only = True
    return model


def load_rgba_initializer(model, state):
    state = dict(state)
    if model.alpha_refiner is not None:
        for key, value in model.alpha_refiner.state_dict().items():
            state.setdefault(f"alpha_refiner.{key}", value)
    if model.alpha_encoder is not None:
        for key,value in model.alpha_encoder.state_dict().items():
            state.setdefault(f"alpha_encoder.{key}",value)
    model.load_state_dict(state, strict=True)
