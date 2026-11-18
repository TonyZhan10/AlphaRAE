"""RGBA input/output and reproducible background interventions."""

from .rgba import (
    RGBADataset, composite, linear_to_srgb, load_rgba, premultiply,
    save_rgba, srgb_to_linear,
)
from .backgrounds import BackgroundSampler

__all__ = [
    "RGBADataset", "BackgroundSampler", "composite", "linear_to_srgb",
    "load_rgba", "premultiply", "save_rgba", "srgb_to_linear",
]
