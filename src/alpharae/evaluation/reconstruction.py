"""Asset-level RGBA metrics in linear RGB, composition metrics in sRGB.

Gradient error uses adjacent finite differences of alpha, not the standard
matting benchmark gradient metric. Edge regions are a 3x3 max-minus-min alpha
band (threshold 0.01). SSIM uses an 11x11 Gaussian, sigma=1.5, reflect padding;
images smaller than 11 use the largest fitting odd window. PSNR is capped at
120 dB by an MSE floor of 1e-12. ROI LPIPS uses the ground-truth alpha>0.01
bounding box intersected with valid pixels; crops are enlarged proportionally
only when their shorter side is below 32. Full LPIPS crops to the valid bbox.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from alpharae.data.rgba import composite, linear_to_srgb, premultiply


PROTOCOL = {
    "version": "rgba-v1",
    "color": "linear RGB for alpha/premultiplied errors; sRGB for composition",
    "gradient": "adjacent finite differences of alpha, not matting benchmark gradient",
    "foreground": "GT alpha > 0.01",
    "semitransparent": "0.05 < GT alpha < 0.95",
    "edge": "3x3 alpha maxpool-minpool > 0.01; intersect valid mask",
    "ssim": "11x11 Gaussian sigma=1.5 reflect padding; largest fitting odd window if smaller; prediction outside valid mask replaced by target",
    "psnr": "range 1; MSE floor 1e-12 (120 dB maximum)",
    "roi_lpips": "GT foreground bbox; scale shorter side to 32 only when necessary",
    "aggregation": "average backgrounds within each asset, then bootstrap assets",
}


def _masked_mean(values, mask):
    mask = mask.expand_as(values)
    return values[mask].mean().item() if mask.any() else None


def _ssim_map(x, y):
    size = min(11, x.shape[-2], x.shape[-1])
    size = size if size % 2 else size - 1
    coordinates = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, size, size)
    padding = size // 2

    def smooth(image):
        image = F.pad(image, (padding,) * 4, mode="reflect") if padding else image
        return F.conv2d(image, kernel, groups=3)

    mu_x, mu_y = smooth(x), smooth(y)
    var_x = (smooth(x.square()) - mu_x.square()).clamp_min(0)
    var_y = (smooth(y.square()) - mu_y.square()).clamp_min(0)
    covariance = smooth(x * y) - mu_x * mu_y
    return ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2) /
            ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)))


def _lpips_crop(model, pred, target, mask):
    points = mask[0].nonzero()
    if not len(points):
        return None
    top, left = points.amin(dim=0).tolist()
    bottom, right = (points.amax(dim=0) + 1).tolist()
    pred = pred[None, :, top:bottom, left:right]
    target = target[None, :, top:bottom, left:right]
    height, width = pred.shape[-2:]
    if min(height, width) < 32:
        scale = 32 / min(height, width)
        size = (math.ceil(height * scale), math.ceil(width * scale))
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
    return model(pred * 2 - 1, target * 2 - 1).mean().item()


def _background_queries(background, batch_size):
    if background.ndim == 3:
        return background[None, None].expand(1, batch_size, -1, -1, -1)
    if background.ndim == 4:
        return background[None].expand(1, batch_size, -1, -1, -1)
    if background.ndim == 5 and background.shape[1] == batch_size:
        return background
    raise ValueError("Backgrounds must have shape [3,H,W], [B,3,H,W], or [K,B,3,H,W]")


@torch.no_grad()
def evaluate_batch(pred, target, backgrounds, valid_mask=None, lpips_model=None):
    """Return one flat metrics dict per asset; K backgrounds never add assets."""
    if pred.ndim == 3:
        pred, target = pred[None], target[None]
    pred, target = pred.float(), target.float()
    p_pred, a_pred = premultiply(pred)
    p_target, a_target = premultiply(target)
    valid = torch.ones_like(a_target, dtype=torch.bool) if valid_mask is None else valid_mask.to(pred.device).bool()
    if valid.ndim == 3:
        valid = valid[None]
    valid = valid.expand_as(a_target)
    spread = F.max_pool2d(a_target, 3, 1, 1) + F.max_pool2d(-a_target, 3, 1, 1)
    regions = {
        "full": valid,
        "foreground": valid & (a_target > 0.01),
        "semitransparent": valid & (a_target > 0.05) & (a_target < 0.95),
        "edge": valid & (spread > 0.01),
    }
    rows = [{} for _ in range(len(pred))]
    errors = {"alpha_mae": (a_pred - a_target).abs(), "alpha_mse": (a_pred - a_target).square(),
              "premult_l1": (p_pred - p_target).abs(), "premult_mse": (p_pred - p_target).square()}
    dx = ((a_pred[..., 1:] - a_pred[..., :-1]) - (a_target[..., 1:] - a_target[..., :-1])).abs()
    dy = ((a_pred[..., 1:, :] - a_pred[..., :-1, :]) - (a_target[..., 1:, :] - a_target[..., :-1, :])).abs()
    for region, masks in regions.items():
        for index, row in enumerate(rows):
            mask = masks[index]
            row[f"{region}_count"] = int(mask.sum())
            for name, values in errors.items():
                row[f"{region}_{name}"] = _masked_mean(values[index], mask)
            mx, my = mask[..., 1:] & mask[..., :-1], mask[..., 1:, :] & mask[..., :-1, :]
            count = int(mx.sum() + my.sum())
            row[f"{region}_gradient_count"] = count
            row[f"{region}_gradient_l1"] = ((dx[index][mx].sum() + dy[index][my].sum()) / count).item() if count else None
    for row in rows:
        for name in (*errors, "gradient_l1"):
            row[name] = row[f"full_{name}"]
    if lpips_model is not None:
        lpips_model.eval()
    for family, background in backgrounds.items():
        per_asset = [{} for _ in rows]
        queries = _background_queries(background.to(device=pred.device, dtype=pred.dtype), len(pred))
        for query in queries:
            prediction = linear_to_srgb(composite(pred, query).clamp(0, 1))
            reference = linear_to_srgb(composite(target, query).clamp(0, 1))
            squared = (prediction - reference).square()
            ssim = _ssim_map(torch.where(valid, prediction, reference), reference)
            for index in range(len(pred)):
                for region, masks in regions.items():
                    mse = _masked_mean(squared[index], masks[index])
                    values = {f"{region}_psnr": -10 * math.log10(max(mse, 1e-12)) if mse is not None else None,
                              f"{region}_ssim": _masked_mean(ssim[index], masks[index])}
                    for name, value in values.items():
                        per_asset[index].setdefault(name, []).append(value)
                for label, mask in (("lpips", valid[index]), ("roi_lpips", regions["foreground"][index])):
                    value = _lpips_crop(lpips_model, prediction[index], reference[index], mask) if lpips_model is not None else None
                    per_asset[index].setdefault(label, []).append(value)
        for row, measurements in zip(rows, per_asset):
            for name, values in measurements.items():
                available = [value for value in values if value is not None]
                row[f"{family}_{name}"] = sum(available) / len(available) if available else None
            row[f"{family}_psnr"] = row[f"{family}_full_psnr"]
            row[f"{family}_ssim"] = row[f"{family}_full_ssim"]
    return rows


def summarize_rows(rows, bootstrap_samples=1000, seed=0):
    """Percentile 95% bootstrap CIs; each row must be one independent asset."""
    generator = np.random.default_rng(seed)
    numeric_keys = sorted({key for row in rows for key, value in row.items()
                           if isinstance(value, (int, float)) or value is None})
    summaries = {}
    for key in numeric_keys:
        values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=float)
        if not len(values):
            summaries[key] = {"mean": None, "ci_low": None, "ci_high": None, "n_assets": 0}
            continue
        samples = generator.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        summaries[key] = {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "n_assets": len(values)}
    return {"n_assets": len(rows), "bootstrap_samples": bootstrap_samples, "seed": seed, "protocol": PROTOCOL, "metrics": summaries}


def summarize_source_groups(rows, bootstrap_samples=1000, seed=0):
    """Give duplicate exports one source-group vote and bootstrap those groups."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    means = []
    for members in groups.values():
        keys = {key for member in members for key, value in member.items()
                if isinstance(value, (int, float)) or value is None}
        means.append({key: float(np.mean(values)) if (values := [member[key] for member in members
                                                               if member.get(key) is not None]) else None
                      for key in keys})
    summary = summarize_rows(means, bootstrap_samples, seed)
    summary["n_groups"] = summary.pop("n_assets")
    for value in summary["metrics"].values():
        value["n_groups"] = value.pop("n_assets")
    summary["resampling_unit"] = "source group; metrics first averaged over exports within each group"
    return summary


def _image(tensor):
    array = (tensor.detach().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def save_comparison(target, predictions, backgrounds, path, asset_ids=None, compositing="linear"):
    """Fixed-order grid: each method shows alpha, black, white and texture."""
    if target.ndim == 3:
        target = target[None]
        predictions = {name: value[None] for name, value in predictions.items()}
    batch, _, height, width = target.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    checker = (((x // 8 + y // 8) % 2).float() * 0.6 + 0.2).expand(3, -1, -1).to(target.device)
    defaults = {"black": torch.zeros_like(target[0, :3]), "white": torch.ones_like(target[0, :3]), "texture": checker}
    if "natural" in backgrounds:
        defaults["natural"] = backgrounds["natural"]
    selected = {name: _background_queries(backgrounds.get(name, value).to(target.device), batch)[0] for name, value in defaults.items()}
    methods = {"target": target, **predictions}
    label_height, gap = 22, 4
    tile_width = max(128, width)
    tile_height = round(height * tile_width / width)
    columns = 1 + len(selected)
    canvas = Image.new("RGB", (len(methods) * columns * (tile_width + gap), batch * (tile_height + 2 * label_height + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(batch):
        top = index * (tile_height + 2 * label_height + gap)
        draw.text((4, top + 3), str(asset_ids[index]) if asset_ids else f"asset {index}", fill="black")
        for method_index, (name, rgba) in enumerate(methods.items()):
            views = {"alpha": rgba[index, 3:].expand(3, -1, -1)}
            if compositing == "srgb":
                from alpharae.evaluation.alphavae_protocol import rgba_to_bytes, blend_rgba_bytes
                encoded = rgba_to_bytes(rgba[index])
                views.update({background: blend_rgba_bytes(encoded, value[index].permute(1,2,0).cpu().numpy()*255)
                              for background, value in selected.items()})
            elif compositing == "linear":
                views.update({background: linear_to_srgb(composite(rgba[index], value[index]).clamp(0, 1)) for background, value in selected.items()})
            else:
                raise ValueError("Choose linear or srgb compositing")
            for view_index, (view_name, tensor) in enumerate(views.items()):
                left = (method_index * columns + view_index) * (tile_width + gap)
                label = f"{name}: {view_name}"
                while draw.textlength(label) > tile_width - 4:
                    label = label[:-1]
                draw.text((left, top + label_height), label, fill="black")
                view = tensor if isinstance(tensor, Image.Image) else _image(tensor)
                canvas.paste(view.resize((tile_width, tile_height), Image.Resampling.NEAREST), (left, top + 2 * label_height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
