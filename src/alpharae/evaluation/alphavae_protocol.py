"""AlphaVAE's nine-background JPEG metric protocol on project RGBA inputs."""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from alpharae.data.rgba import linear_to_srgb


PROTOCOL_ID = "alphavae-nine-jpeg-v1"
BACKGROUNDS = {
    "gray": (128, 128, 128), "black": (0, 0, 0), "white": (255, 255, 255),
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
}
METRICS = ("psnr", "ssim", "lpips", "rfid", "laion_aes")
ROW_METRICS = ("psnr", "ssim", "lpips", "laion_aes", "target_laion_aes")


def rgba_to_bytes(rgba):
    """Linear RGBA -> sRGB bytes, with torchvision.save_image's +0.5 cast."""
    rgba = rgba.detach().float().cpu()
    encoded = torch.cat((linear_to_srgb(rgba[:3]), rgba[3:4]), dim=0)
    return encoded.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()


def blend_rgba_bytes(rgba, background):
    """Match official blend_rgba.py, including float32 and truncation."""
    rgba = np.asarray(rgba).astype(np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    bg = np.asarray(background, dtype=np.float32) / 255.0
    blended = alpha * rgb + (1 - alpha) * bg
    return Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8))


def save_composites(rgba, root, asset_id):
    encoded = rgba_to_bytes(rgba)
    for name, color in BACKGROUNDS.items():
        path = Path(root) / name / f"{asset_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Omitting quality reproduces PIL's default JPEG quality 75.
        blend_rgba_bytes(encoded, color).save(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def library_versions():
    names = ("pyiqa", "torch", "torchvision", "numpy", "scipy", "Pillow", "diffusers",
             "transformers", "openai-clip", "opencv-python-headless", "addict", "ftfy")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def aggregate_rows(rows, by_background):
    """Per-image means within a background, then equal means over nine colors."""
    for background in BACKGROUNDS:
        subset = [row for row in rows if row["background"] == background]
        scores = by_background.setdefault(background, {key: None for key in METRICS})
        scores["n_assets"] = len(subset)
        for key in ROW_METRICS:
            scores[key] = float(np.mean([float(row[key]) for row in subset])) if subset else None
    result = {}
    for key in (*METRICS, "target_laion_aes"):
        values = [by_background[name].get(key) for name in BACKGROUNDS]
        result[key] = float(np.mean(values)) if all(value is not None for value in values) else None
    return result


class OfficialMetrics:
    """pyiqa default metrics; no replacement metric implementations."""

    def __init__(self, device="cuda", cache_dir="weights/torch", on_loaded=None,
                 names=("psnr", "ssim", "lpips", "laion_aes", "fid")):
        from functools import partial

        cache = Path(cache_dir).resolve()
        torch.hub.set_dir(str(cache / "hub"))
        import pyiqa
        import clip
        import pyiqa.utils.download_util as downloads

        downloads.DEFAULT_CACHE_DIR = str(cache / "hub" / "pyiqa")
        self.device = device
        self.metrics = {}
        for name in names:
            original_load = clip.load
            try:
                # LAION's wrapper does not expose CLIP's download_root argument.
                clip.load = partial(original_load, download_root=str(cache / "clip"), device="cpu")
                self.metrics[name] = pyiqa.create_metric(name, device=device).float().eval()
            finally:
                clip.load = original_load
            if on_loaded:
                on_loaded(name)

    @torch.no_grad()
    def paired(self, prediction_paths, target_paths):
        from pyiqa.utils.img_util import imread2tensor

        prediction = torch.stack([imread2tensor(str(path), rgb=True) for path in prediction_paths]).to(self.device)
        target = torch.stack([imread2tensor(str(path), rgb=True) for path in target_paths]).to(self.device)
        scores = {name: self.metrics[name](prediction, target).flatten().cpu().tolist()
                  for name in ("psnr", "ssim", "lpips")}
        if "laion_aes" in self.metrics:
            scores["laion_aes"] = self.metrics["laion_aes"](prediction).flatten().cpu().tolist()
            scores["target_laion_aes"] = self.metrics["laion_aes"](target).flatten().cpu().tolist()
        return scores

    @torch.no_grad()
    def features(self, directory, cache_path, batch_size=8, workers=0):
        from pyiqa.archs.fid_arch import get_folder_features

        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)
        network = self.metrics["fid"].net
        features = get_folder_features(
            str(directory), model=network.model, num_workers=workers, batch_size=batch_size,
            test_img_size=network.test_img_size, device=self.device, mode="clean", verbose=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
        return features

    @staticmethod
    def rfid(prediction_features, target_features):
        from pyiqa.archs.fid_arch import frechet_distance

        if min(len(prediction_features), len(target_features)) < 2:
            raise ValueError("rFID requires at least two assets per background")
        # These are exactly the statistics passed by pyiqa.FID.forward.
        value = frechet_distance(
            np.mean(prediction_features, axis=0), np.cov(prediction_features, rowvar=False),
            np.mean(target_features, axis=0), np.cov(target_features, rowvar=False))
        return max(float(value), 0.0)
