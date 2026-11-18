"""Straight RGBA tensors use linear RGB; alpha is always linear coverage.

Input PNGs are 8-bit sRGB. ``premultiplied`` means RGB was multiplied by
alpha in encoded sRGB, so it is undone before the sRGB transfer function.
Resampling is always on linear premultiplied color and alpha together.
"""

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .manifest import read_manifest


INPUT_KINDS = ("rgba", "background_blend", "rgb_replay")
MIXING_SEED_OFFSET = 2000003


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0).pow(2.4))


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-10).pow(1 / 2.4) - 0.055)


def premultiply(rgba):
    alpha = rgba[..., 3:4, :, :]
    return rgba[..., :3, :, :] * alpha, alpha


def composite(rgba, background):
    p, alpha = premultiply(rgba)
    return p + (1 - alpha) * background


def inspect_png(path):
    with open(path, "rb") as handle:
        header = handle.read(26)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Expected PNG: {path}")
    if header[24] != 8:
        raise ValueError(f"Only 8-bit PNG is supported, found {header[24]}-bit: {path}")
    with Image.open(path) as image:
        if "A" not in image.getbands() and "transparency" not in image.info:
            raise ValueError(f"PNG has no alpha channel: {path}")
        if image.info.get("icc_profile"):
            raise ValueError(f"Convert embedded ICC profile to untagged sRGB before loading: {path}")
        return image.size


def load_rgba(path, resolution, alpha_mode="straight"):
    if alpha_mode not in ("straight", "premultiplied"):
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
    width, height = inspect_png(path)
    with Image.open(path) as image:
        rgba = torch.from_numpy(np.array(image.convert("RGBA"), copy=True)).permute(2, 0, 1).float() / 255
    rgb, alpha = rgba[:3], rgba[3:]
    if alpha_mode == "premultiplied":
        rgb = torch.where(alpha > 0, rgb / alpha.clamp_min(1 / 255), 0).clamp(0, 1)
    rgb = srgb_to_linear(rgb)
    packed = torch.cat((rgb * alpha, alpha), dim=0)
    target_h, target_w = (resolution, resolution) if isinstance(resolution, int) else resolution
    scale = min(target_h / height, target_w / width)
    out_h, out_w = max(1, round(height * scale)), max(1, round(width * scale))
    packed = F.interpolate(packed[None], size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True)[0]
    top, left = (target_h - out_h) // 2, (target_w - out_w) // 2
    packed = F.pad(packed, (left, target_w - out_w - left, top, target_h - out_h - top))
    alpha = packed[3:]
    rgb = torch.where(alpha > 0, packed[:3] / alpha.clamp_min(1e-8), 0)
    valid = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    valid[:, top:top + out_h, left:left + out_w] = 1
    return {"rgba": torch.cat((rgb, alpha)), "valid_mask": valid}


def save_rgba(rgba, path):
    rgba = rgba.detach().cpu().clamp(0, 1)
    if rgba.ndim != 3 or rgba.shape[0] != 4:
        raise ValueError("save_rgba expects a [4,H,W] tensor")
    alpha_byte = (rgba[3:] * 255).round().to(torch.uint8)
    rgb_byte = (linear_to_srgb(rgba[:3]) * 255).round().clamp(0, 255).to(torch.uint8)
    rgb_byte = torch.where(alpha_byte > 0, rgb_byte, 0)
    array = torch.cat((rgb_byte, alpha_byte)).permute(1, 2, 0).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def blend_opaque_srgb(rgba, background):
    """Apply the AlphaVAE color-cube composition in sRGB, keeping float targets."""
    srgb = linear_to_srgb(rgba[:3])
    alpha = rgba[3:]
    blended = srgb * alpha + background.to(rgba).reshape(3, 1, 1) * (1 - alpha)
    return torch.cat((srgb_to_linear(blended), torch.ones_like(alpha)))


def load_rgb_replay(row, resolution, generator):
    """Use random RGB square crops and flips with a sample-local RNG."""
    with Image.open(row["path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if min(width, height) < resolution:
        raise ValueError(f"Source image is smaller than training resolution: {row['path']}")
    crop = int(torch.randint(resolution, min(width, height) + 1, (), generator=generator))
    left = int(torch.randint(0, width - crop + 1, (), generator=generator))
    top = int(torch.randint(0, height - crop + 1, (), generator=generator))
    image = image.crop((left, top, left + crop, top + crop))
    if crop != resolution:
        image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    if row.get("allow_flip", True) and torch.rand((), generator=generator) < .5:
        image = ImageOps.mirror(image)
    srgb = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    alpha = torch.ones_like(srgb[:1])
    return {"rgba": torch.cat((srgb_to_linear(srgb), alpha)), "valid_mask": alpha.clone()}


class RGBADataset(Dataset):
    def __init__(self, manifest, resolution, split, shift_pixels=0, seed=0,
                 background_blend_probability=0., rgb_replay_probability=0.,
                 rgb_replay_manifest=None):
        self.resolution = resolution
        self.split = split
        self.shift_pixels=shift_pixels if split=="train" else 0
        self.seed,self.epoch=seed,0
        self.records = [row for row in read_manifest(manifest) if row["split"] == split]
        if not self.records:
            raise ValueError(f"No assets in split {split!r}: {manifest}")
        self.background_blend_probability = background_blend_probability if split == "train" else 0.
        self.rgb_replay_probability = rgb_replay_probability if split == "train" else 0.
        if (self.background_blend_probability < 0 or self.rgb_replay_probability < 0
                or self.background_blend_probability + self.rgb_replay_probability > 1):
            raise ValueError("Input mixing probabilities must be nonnegative and sum to at most one")
        self.rgb_replay_manifest = None
        self.rgb_records = []
        if self.rgb_replay_probability:
            if not rgb_replay_manifest:
                raise ValueError("rgb_replay_manifest is required for RGB replay")
            self.rgb_replay_manifest = Path(rgb_replay_manifest).resolve()
            with self.rgb_replay_manifest.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.rgb_records = [dict(row, path=os.path.abspath(self.rgb_replay_manifest.parent / row["path"]))
                                for row in rows if row.get("split", "train") == "train"]
            if not self.rgb_records:
                raise ValueError("RGB replay manifest has no training images")

    def input_mixing_metadata(self):
        return {
            "probabilities": {"rgba": 1 - self.background_blend_probability - self.rgb_replay_probability,
                              "background_blend": self.background_blend_probability,
                              "rgb_replay": self.rgb_replay_probability},
            "logical_slots": len(self.records),
            "pool_counts": {"rgba_assets": len(self.records), "rgb_replay_assets": len(self.rgb_records)},
            "rgb_replay_source_counts": dict(Counter(row.get("source", "unknown") for row in self.rgb_records)),
            "rgb_replay_manifest": str(self.rgb_replay_manifest) if self.rgb_replay_manifest else None,
            "rgb_manifest_unspecified_split": "train",
            "blend_space": "srgb_float",
            "blend_background_values": [0., .5, 1.],
            "sample_rng": "seed + 2000003 + epoch * logical_slots + index",
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        generator = None
        input_kind = "rgba"
        if self.background_blend_probability or self.rgb_replay_probability:
            generator = torch.Generator().manual_seed(
                self.seed + MIXING_SEED_OFFSET + self.epoch * len(self.records) + index)
            choice = float(torch.rand((), generator=generator))
            if choice < self.background_blend_probability:
                input_kind = "background_blend"
            elif choice < self.background_blend_probability + self.rgb_replay_probability:
                rgb_index = int(torch.randint(len(self.rgb_records), (), generator=generator))
                row = self.rgb_records[rgb_index]
                result = load_rgb_replay(row, self.resolution, generator)
                result.update(asset_id=f"rgb:{row['id']}", input_kind="rgb_replay")
                return result
        row = self.records[index]
        result = load_rgba(row["path"], self.resolution, row["alpha_mode"])
        if self.shift_pixels:
            shift_generator=torch.Generator().manual_seed(self.seed+self.epoch*len(self.records)+index)
            dy,dx=torch.randint(-self.shift_pixels,self.shift_pixels+1,(2,),generator=shift_generator).tolist()
            p=self.shift_pixels
            height,width=result["rgba"].shape[-2:]
            for key in ("rgba","valid_mask"):
                result[key]=F.pad(result[key],(p,p,p,p))[:,p-dy:p-dy+height,p-dx:p-dx+width]
        if input_kind == "background_blend":
            background = torch.randint(0, 3, (3,), generator=generator).float() / 2
            result["rgba"] = blend_opaque_srgb(result["rgba"], background)
            result["valid_mask"] = torch.ones_like(result["valid_mask"])
        result["asset_id"] = row["asset_id"]
        result["input_kind"] = input_kind
        return result
