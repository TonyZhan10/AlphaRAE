"""Supervise actual matte detail, including opaque-to-clear boundaries."""

import torch
from torch.nn import functional as F

from alpharae.losses.reconstruction import masked_mean


def detail_region(alpha):
    high = F.max_pool2d(alpha, 7, stride=1, padding=3)
    low = -F.max_pool2d(-alpha, 7, stride=1, padding=3)
    return (high-low > .02).to(alpha.dtype)


def alpha_gradient_metrics(pred, target, valid_mask):
    region = detail_region(target) * valid_mask
    dx, dy = pred[...,1:]-pred[...,:-1], pred[...,1:,:]-pred[...,:-1,:]
    tx, ty = target[...,1:]-target[...,:-1], target[...,1:,:]-target[...,:-1,:]
    mx, my = region[...,1:]*region[...,:-1], region[...,1:,:]*region[...,:-1,:]
    numerator = (dx*tx*mx).sum((1,2,3)) + (dy*ty*my).sum((1,2,3))
    denominator = (tx.square()*mx).sum((1,2,3)) + (ty.square()*my).sum((1,2,3))
    return {"detail_gradient_l1":(masked_mean((dx-tx).abs(),mx)+masked_mean((dy-ty).abs(),my))/2,
            "gradient_gain":(numerator/denominator.clamp_min(1e-12)).mean()}


def alpha_detail_losses(pred, target, valid_mask=None):
    region = detail_region(target)
    if valid_mask is not None:
        region = region * valid_mask
    local_l1 = masked_mean((pred-target).abs(), region)
    # Fixed Gaussian Laplacian pyramid; compare residuals to GT, not to zero.
    kernel = pred.new_tensor([1,4,6,4,1])
    kernel = (kernel[:,None]*kernel[None,:]/256)[None,None]
    levels = []
    current_pred, current_target, mask = pred, target, valid_mask
    for _ in range(3):
        packed = torch.cat((current_pred,current_target),0)
        blurred = F.conv2d(F.pad(packed,(2,2,2,2),mode="reflect"),kernel)
        low = blurred[:,:,::2,::2]
        expanded = F.interpolate(low,size=packed.shape[-2:],mode="bilinear",align_corners=False)
        residual_pred,residual_target = (packed-expanded).chunk(2,0)
        levels.append(masked_mean((residual_pred-residual_target).abs(), mask))
        current_pred,current_target = low.chunk(2,0)
        if mask is not None:
            mask = F.avg_pool2d(mask,2)
    return {"alpha_detail_l1":local_l1,"alpha_laplacian":torch.stack(levels).sum()}
