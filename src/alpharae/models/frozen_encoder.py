import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import linear_to_srgb


class FrozenEncoder(nn.Module):
    """Local Hugging Face DINOv2, or an explicitly non-semantic pixel test fixture."""

    def __init__(self, kind="dinov2", weights=None, resolution=224, patch_size=14,
                 layers=(-4, -3, -2, -1)):
        super().__init__()
        self.kind, self.resolution, self.patch_size = kind, resolution, patch_size
        self.layers = tuple(layers)
        if resolution % patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if kind == "pixel_fixture":
            self.backbone = nn.Identity()
            self.dim = 3 * patch_size * patch_size
        elif kind == "dinov2":
            from transformers import AutoModel
            if not weights:
                raise ValueError("Supply a local DINOv2 checkpoint; no implicit download")
            self.backbone = AutoModel.from_pretrained(weights, local_files_only=True)
            if self.backbone.config.model_type not in ("dinov2", "dinov2_with_registers"):
                raise ValueError("This adapter supports DINOv2 checkpoints only")
            if self.backbone.config.patch_size != patch_size:
                raise ValueError("Configured patch size differs from checkpoint")
            self.dim = self.backbone.config.hidden_size
        else:
            raise ValueError(f"Unknown encoder kind: {kind}")
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, linear_rgb):
        if linear_rgb.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError("Resize and pad the RGBA asset before encoding")
        image = (linear_to_srgb(linear_rgb).float() - self.rgb_mean) / self.rgb_std
        if self.kind == "pixel_fixture":
            return F.unfold(image, self.patch_size, stride=self.patch_size).transpose(1, 2)
        out = self.backbone(pixel_values=image, output_hidden_states=True)
        n = (self.resolution // self.patch_size) ** 2
        # Patch tokens follow CLS and optional register tokens.
        return torch.stack([out.hidden_states[i][:, -n:] for i in self.layers]).mean(0).float()
