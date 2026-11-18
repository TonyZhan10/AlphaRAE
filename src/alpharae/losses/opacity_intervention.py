"""Known RGBA interventions with an unchanged white composite."""

import torch

from alpharae.data.rgba import premultiply
from alpharae.losses.reconstruction import masked_mean


def intervene_opacity(rgba, tau):
    if not 0 <= tau <= 1:
        raise ValueError("Opacity intervention requires tau in [0, 1]")
    foreground, alpha = rgba[:, :3], rgba[:, 3:]
    change = tau * (1 - alpha)
    delta = alpha * change
    other = torch.cat(((foreground + change) / (1 + change), alpha + delta), dim=1)
    return other, delta


def opacity_pair_loss(pred, other_pred, target, other_target, valid_mask=None, mode="response"):
    """Match region/channel weighting for response and augmentation controls."""
    mask = ((target[:, 3:] > 0) & (target[:, 3:] < 1)).to(target.dtype)
    if valid_mask is not None:
        mask = mask * valid_mask
    p0, a0 = premultiply(pred)
    p1, a1 = premultiply(other_pred)
    t0, b0 = premultiply(target)
    t1, b1 = premultiply(other_target)
    rgb_error, alpha_error = p1 - t1, a1 - b1
    if mode == "response":
        rgb_error = rgb_error - (p0 - t0)
        alpha_error = alpha_error - (a0 - b0)
    elif mode != "augmentation":
        raise ValueError("Choose response or augmentation")
    return masked_mean(rgb_error.abs(), mask) + masked_mean(alpha_error.abs(), mask)
