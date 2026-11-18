"""RGBA adaptation of the official RGB RAE, preserving its RGB initialization."""

from functools import partial
from math import isqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from alpharae.data.rgba import composite, linear_to_srgb, srgb_to_linear, premultiply
from references.RAE.src.stage1.decoders.decoder import GeneralDecoder, ViTMAESdpaSelfAttention
from references.RAE.src.stage1.decoders.utils import ViTMAEConfig


DECODER_CONFIG = (Path(__file__).resolve().parents[2] / "references" / "RAE" /
                  "configs" / "decoder" / "ViTB" / "config.json")


def resize_decoder_position_embeddings(position_embeddings, num_patches):
    """Resize the square patch grid in two dimensions, preserving CLS exactly."""
    source_patches = position_embeddings.shape[1] - 1
    if source_patches == num_patches:
        return position_embeddings
    source_grid, target_grid = isqrt(source_patches), isqrt(num_patches)
    if source_grid ** 2 != source_patches or target_grid ** 2 != num_patches:
        raise ValueError("Decoder position embeddings require square patch grids")
    patches = position_embeddings[:, 1:].reshape(1, source_grid, source_grid, -1)
    patches = F.interpolate(patches.permute(0, 3, 1, 2).float(),
                            size=(target_grid, target_grid), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
    return torch.cat((position_embeddings[:, :1], patches.to(position_embeddings.dtype)), dim=1)


def configure_decoder_execution(decoder, sdpa=False, gradient_checkpointing=False):
    """Use the reference decoder's published attention and checkpoint interfaces."""
    if sdpa:
        for layer in decoder.decoder_layers:
            previous = layer.attention.attention
            attention = ViTMAESdpaSelfAttention(decoder.decoder_config).to(
                device=previous.query.weight.device, dtype=previous.query.weight.dtype)
            attention.load_state_dict(previous.state_dict(), strict=True)
            attention.train(previous.training)
            layer.attention.attention = attention
    decoder.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    return decoder


def load_rgb_decoder(weights, config_path=DECODER_CONFIG, resolution=256,
                     decoder_sdpa=False, decoder_gradient_checkpointing=False,
                     pretrained_resolution=256):
    """Strictly load the configured source grid before adapting the output grid."""
    if resolution not in (256, 512, 1024) or pretrained_resolution not in (256, 512, 1024):
        raise ValueError("Pretrained RGBA source and output resolutions must be 256, 512 or 1024")
    config = ViTMAEConfig.from_json_file(str(config_path))
    config.hidden_size, config.patch_size, config.image_size = 768, 16, pretrained_resolution
    decoder = GeneralDecoder(config, num_patches=(pretrained_resolution // config.patch_size) ** 2)
    decoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    if resolution != pretrained_resolution:
        decoder.num_patches = (resolution // config.patch_size) ** 2
        decoder.config.image_size = resolution
        decoder.decoder_config.image_size = resolution
        decoder.decoder_pos_embed = nn.Parameter(
            resize_decoder_position_embeddings(decoder.decoder_pos_embed, decoder.num_patches),
            requires_grad=False)
    return configure_decoder_execution(decoder, decoder_sdpa, decoder_gradient_checkpointing)


def extend_rgb_decoder(decoder):
    """Preserve per-pixel RGB rows; introduce zero alpha logits."""
    old = decoder.decoder_pred
    p, width = decoder.config.patch_size, old.in_features
    head = nn.Linear(width, p * p * 4, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight.view(p, p, 4, width)[:, :, :3].copy_(old.weight.view(p, p, 3, width))
        head.bias.view(p, p, 4)[:, :, :3].copy_(old.bias.view(p, p, 3))
    decoder.decoder_pred = head
    decoder.config.num_channels = 4
    return decoder


class PretrainedRAEEncoder(nn.Module):
    """Asset interface with matching patch grids for RGBA and native DINO.

    ``resolution`` and ``patch_size`` describe the output/response grid, not
    the DINO patch embedding. The native dimensions are exposed separately.
    """

    def __init__(self, weights, resolution=256):
        super().__init__()
        from transformers import Dinov2WithRegistersModel

        if resolution not in (256, 512, 1024):
            raise ValueError("Pretrained RGBA resolution must be 256, 512 or 1024")
        self.backbone = Dinov2WithRegistersModel.from_pretrained(weights, local_files_only=True)
        self.encoder_input_size, self.encoder_patch_size = (resolution // 16) * 14, 14
        self.resolution, self.patch_size, self.dim = resolution, 16, 768
        self.num_tokens = (resolution // self.patch_size) ** 2
        config = self.backbone.config
        if (config.hidden_size, config.patch_size, config.num_register_tokens) != (768, 14, 4):
            raise ValueError("Pretrained RGB decoder requires DINOv2 with registers base")
        self.backbone.layernorm.elementwise_affine = False
        self.backbone.layernorm.weight = None
        self.backbone.layernorm.bias = None
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, linear_rgb):
        srgb = linear_to_srgb(linear_rgb).float()
        srgb = F.interpolate(srgb, (self.encoder_input_size, self.encoder_input_size),
                             mode="bicubic", align_corners=False)
        image = (srgb - self.rgb_mean) / self.rgb_std
        return self.backbone(pixel_values=image).last_hidden_state[:, 5:].float()


class LocalTokenReadout(nn.Sequential):
    """Decode local detail from the same latent, in the decoder's patch order."""

    def __init__(self, latent_dim, width, patch_size):
        super().__init__(nn.Linear(latent_dim, width), nn.GELU(),
                         nn.Linear(width, patch_size * patch_size * 4))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class PretrainedRGBAAutoencoder(nn.Module):
    def __init__(self, encoder, rgb_decoder, variant="B1", alpha_width=128, residual_input="alpha",
                 local_readout_width=0, joint_adapter_width=0):
        super().__init__()
        self.encoder, self.variant = encoder, variant
        self.resolution = getattr(encoder, "resolution", 256)
        self.latent_dim, self.num_tokens = 768, (self.resolution // 16) ** 2
        self.decoder = extend_rgb_decoder(rgb_decoder)
        if residual_input not in ("alpha", "premult_rgba", "rgb_projection"):
            raise ValueError(f"Unknown residual input: {residual_input}")
        self.residual_input = residual_input
        self.alpha_branch = nn.Sequential(
            nn.Conv2d(4 if residual_input == "premult_rgba" else 1, alpha_width, 16, stride=16), nn.GELU(),
            nn.Conv2d(alpha_width, self.latent_dim, 1),
        )
        nn.init.zeros_(self.alpha_branch[-1].weight)
        nn.init.zeros_(self.alpha_branch[-1].bias)
        self.rgb_projection = None
        if residual_input == "rgb_projection":
            self.rgb_projection = nn.Conv2d(3, self.latent_dim, 16, stride=16, bias=False)
            nn.init.zeros_(self.rgb_projection.weight)
        self.local_readout = (LocalTokenReadout(self.latent_dim, local_readout_width,
                                              self.decoder.config.patch_size)
                              if local_readout_width else None)
        self.residual_adapter = None
        if joint_adapter_width:
            self.residual_adapter = nn.Sequential(
                nn.Conv2d(4 * 16 * 16 + self.latent_dim, joint_adapter_width, 1), nn.GELU(),
                nn.Conv2d(joint_adapter_width, joint_adapter_width, 3, padding=1,
                          groups=joint_adapter_width), nn.GELU(),
                nn.Conv2d(joint_adapter_width, self.latent_dim, 1),
            )
            nn.init.zeros_(self.residual_adapter[-1].weight)
            nn.init.zeros_(self.residual_adapter[-1].bias)
        self.register_buffer("stats_fitted", torch.tensor(False))

    @torch.no_grad()
    def observations(self, rgba):
        black = self.encoder(composite(rgba, torch.zeros_like(rgba[:, :3])))
        white = self.encoder(composite(rgba, torch.ones_like(rgba[:, :3])))
        return black, white

    @torch.no_grad()
    def fit_feature_stats(self, black, white, std_floor=.01):
        # Native decoder expects DINO tokens, so dataset statistics are not applied.
        self.stats_fitted.fill_(True)

    def encode(self, rgba, observations=None):
        if rgba.ndim != 4 or rgba.shape[1] != 4 or rgba.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(f"Pretrained RGBA RAE requires Bx4x{self.resolution}x{self.resolution} assets")
        if observations is None:
            white = self.encoder(composite(rgba, torch.ones_like(rgba[:, :3])))
        else:
            _, white = observations
        if white.shape != (rgba.shape[0], self.num_tokens, self.latent_dim):
            raise ValueError(f"Encoder observations must have shape Bx{self.num_tokens}x{self.latent_dim}")
        if self.residual_input == "premult_rgba":
            p, alpha = premultiply(rgba)
            branch_input = torch.cat((p, alpha), dim=1)
        else:
            branch_input = rgba[:, 3:4]
        residual = self.alpha_branch(branch_input).flatten(2).transpose(1, 2)
        if self.rgb_projection is not None:
            residual = residual + self.rgb_projection(premultiply(rgba)[0]).flatten(2).transpose(1, 2)
        z = white + residual
        if self.residual_adapter is not None:
            p, alpha = premultiply(rgba)
            pixels = F.pixel_unshuffle(torch.cat((p, alpha), dim=1), 16)
            white_grid = white.transpose(1, 2).reshape(
                rgba.shape[0], self.latent_dim, self.resolution // 16, self.resolution // 16)
            joint = self.residual_adapter(torch.cat((pixels, white_grid), dim=1))
            z = z + joint.flatten(2).transpose(1, 2)
        return z

    def decode(self, z):
        if z.ndim != 3 or z.shape[1:] != (self.num_tokens, self.latent_dim):
            raise ValueError(f"Pretrained RGBA latent must have shape Bx{self.num_tokens}x{self.latent_dim}")
        output = self.decoder(z, drop_cls_token=False).logits
        if self.local_readout is not None:
            output = output + self.local_readout(z)
        pixels = self.decoder.unpatchify(output).float()
        srgb = pixels[:, :3] * self.encoder.rgb_std + self.encoder.rgb_mean
        rgb = srgb_to_linear(srgb.clamp(0, 1))
        return torch.cat((rgb, pixels[:, 3:4].sigmoid()), dim=1)

    def forward(self, rgba, observations=None, decode_count=None):
        z = self.encode(rgba, observations)
        return self.decode(z if decode_count is None else z[:decode_count]), z


def build_pretrained_rgba(config):
    encoder_config = config["encoder"]
    resolution = encoder_config.get("resolution", 256)
    encoder = PretrainedRAEEncoder(encoder_config["weights"], resolution=resolution)
    decoder = load_rgb_decoder(config["pretrained_decoder"],
                               config.get("decoder_config", DECODER_CONFIG), resolution=resolution,
                               decoder_sdpa=config.get("decoder_sdpa", False),
                               decoder_gradient_checkpointing=config.get("decoder_gradient_checkpointing", False),
                               pretrained_resolution=config.get("pretrained_decoder_resolution", 256))
    model = PretrainedRGBAAutoencoder(encoder, decoder, config.get("variant", "B1"),
                                     config.get("alpha_width", 128), config.get("residual_input", "alpha"),
                                     config.get("local_readout_width", 0), config.get("joint_adapter_width", 0))
    if config.get("freeze_decoder", False):
        model.decoder.requires_grad_(False)
        if model.local_readout is not None:
            model.local_readout.requires_grad_(False)
    return model


def adapt_alpha_state(model, state):
    """Adapt the spatial position grid and existing optional residual branches."""
    state = dict(state)
    pos_key = "decoder.decoder_pos_embed"
    if pos_key in state:
        state[pos_key] = resize_decoder_position_embeddings(state[pos_key], model.num_tokens)
    key = "alpha_branch.0.weight"
    source, target = state[key], model.state_dict()[key]
    if source.shape[1] == 1 and target.shape[1] == 4:
        expanded = source.new_zeros(target.shape)
        expanded[:, 3:4] = source
        state[key] = expanded
    if model.rgb_projection is not None and "rgb_projection.weight" not in state:
        state["rgb_projection.weight"] = source.new_zeros(model.rgb_projection.weight.shape)
    if model.local_readout is not None:
        for key, value in model.local_readout.state_dict().items():
            state.setdefault(f"local_readout.{key}", value)
    if model.residual_adapter is not None:
        for key, value in model.residual_adapter.state_dict().items():
            state.setdefault(f"residual_adapter.{key}", value)
    model.load_state_dict(state, strict=True)
