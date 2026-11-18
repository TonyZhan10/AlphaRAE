"""Split-specific linear RGB backgrounds with a checkpointable local RNG.

Without a natural manifest, only color/gradient/texture are sampled, uniformly.
With a manifest, four families are sampled uniformly. Natural manifest paths
are relative to that manifest and its supplied split/group assignments apply.
"""

import warnings

import numpy as np
from PIL import Image, ImageOps
import torch

from .manifest import read_manifest
from .rgba import srgb_to_linear


class BackgroundSampler:
    RANGES = {
        "train": {"color": (0.05, 0.70), "frequency": (1.0, 4.0), "offset": 0},
        "val": {"color": (0.72, 0.84), "frequency": (4.5, 6.0), "offset": 1000003},
        "test": {"color": (0.86, 0.98), "frequency": (6.5, 8.0), "offset": 2000003},
    }

    def __init__(self, resolution, split="train", seed=0, natural_manifest=None):
        self.resolution = resolution
        self.split = split
        self.parameters = self.RANGES[split]
        self.generator = torch.Generator().manual_seed(seed + self.parameters["offset"])
        self.natural = []
        if natural_manifest is not None:
            self.natural = [row for row in read_manifest(natural_manifest) if row["split"] == split]
            if not self.natural:
                raise ValueError(f"Natural background manifest has no {split} rows")
        else:
            warnings.warn("Natural backgrounds unavailable: sampling color, gradient and texture only.", stacklevel=2)
        side = torch.linspace(0, 1, resolution)
        self.y, self.x = torch.meshgrid(side, side, indexing="ij")

    def state_dict(self):
        return {"generator_state": self.generator.get_state(), "split": self.split, "resolution": self.resolution}

    def load_state_dict(self, state):
        if state["split"] != self.split or state["resolution"] != self.resolution:
            raise ValueError("Background sampler split/resolution differs from checkpoint")
        self.generator.set_state(state["generator_state"].cpu())

    def sample(self, batch_size, device="cpu", family=None):
        families = {"color": 0, "gradient": 1, "texture": 2, "natural": 3}
        if family is not None and family not in families:
            raise ValueError(f"Unknown background family: {family}")
        if family == "natural" and not self.natural:
            raise ValueError("Natural background family requested without a manifest")
        outputs = []
        low, high = self.parameters["color"]
        for _ in range(batch_size):
            kind = families[family] if family else int(torch.randint(4 if self.natural else 3, (), generator=self.generator))
            colors = low + (high - low) * torch.rand((2, 3, 1, 1), generator=self.generator)
            if kind == 0:
                output = colors[0].expand(3, self.resolution, self.resolution)
            elif kind == 1:
                angle = torch.rand((), generator=self.generator) * (2 * torch.pi)
                weight = torch.cos(angle) * (self.x - 0.5) + torch.sin(angle) * (self.y - 0.5)
                weight = (weight / 1.41421356237 + 0.5).clamp(0, 1)
                output = colors[0] * weight + colors[1] * (1 - weight)
            elif kind == 2:
                freq_low, freq_high = self.parameters["frequency"]
                frequency = freq_low + (freq_high - freq_low) * torch.rand((), generator=self.generator)
                phase = torch.rand((), generator=self.generator) * (2 * torch.pi)
                weight = 0.5 + 0.5 * torch.sin(2 * torch.pi * frequency * self.x + phase) * torch.sin(2 * torch.pi * frequency * self.y + phase)
                output = colors[0] * weight + colors[1] * (1 - weight)
            else:
                index = int(torch.randint(len(self.natural), (), generator=self.generator))
                with Image.open(self.natural[index]["path"]) as image:
                    image = ImageOps.fit(image.convert("RGB"), (self.resolution, self.resolution), method=Image.Resampling.BILINEAR)
                    output = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
                output = srgb_to_linear(output)
            outputs.append(output)
        return torch.stack(outputs).to(device)
