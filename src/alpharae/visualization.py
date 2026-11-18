"""Write straight-alpha PNGs and views for checking foreground color and opacity."""

from pathlib import Path
from PIL import Image
import torch

from alpharae.data.rgba import linear_to_srgb, save_rgba


def image_file(value, path):
    Image.fromarray(value.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)


def save_views(rgba, folder, prefix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rgb = linear_to_srgb(rgba[:3]).clamp(0, 1)
    alpha = rgba[3:].clamp(0, 1)
    xy = torch.arange(rgba.shape[-1], device=rgba.device) // 8
    checker = ((xy[:, None] + xy[None, :]) % 2)[None].float() * .2 + .65
    values = {'rgb': rgb, 'alpha': alpha.expand(3, -1, -1),
              'checker': rgb * alpha + checker * (1 - alpha),
              'black': rgb * alpha, 'white': rgb * alpha + 1 - alpha}
    files = {key: f'{prefix}_{key}.png' for key in (*values, 'rgba')}
    for key, image in values.items():
        image_file(image, folder / files[key])
    save_rgba(rgba, folder / files['rgba'])
    return files
