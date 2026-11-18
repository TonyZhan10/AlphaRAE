import torch
from torch import nn

from alpharae.data.rgba import composite, linear_to_srgb, premultiply


def masked_mean(error, mask=None):
    if mask is None:
        return error.mean()
    per_asset = (error * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * error.shape[1]).clamp_min(1)
    return per_asset.mean()


def expected_uniform_composite_mse(pred, target, valid_mask=None):
    """Exact E_B[(delta_p - delta_alpha B)^2], independent B ~ U[0,1]."""
    pp, pa = premultiply(pred)
    tp, ta = premultiply(target)
    dp, da = pp - tp, pa - ta
    return masked_mean(dp.square() - dp * da + da.square() / 3, valid_mask)


def pixel_response_loss(pred, target, bi, bj, valid_mask=None):
    pred_response = composite(pred, bj) - composite(pred, bi)
    target_response = composite(target, bj) - composite(target, bi)
    return masked_mean((pred_response - target_response).square(), valid_mask)


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=1., composition=1., perceptual=.1, lpips_model=None,
                 composition_mode="sampled", alpha_detail=0.):
        super().__init__()
        self.alpha_weight, self.comp_weight, self.perc_weight = alpha, composition, perceptual
        self.alpha_detail_weight = alpha_detail
        self.extra_metric_keys = ("alpha_detail_l1", "alpha_laplacian") if alpha_detail else ()
        self.lpips_model = lpips_model
        if composition_mode not in ("sampled", "analytic_uniform"):
            raise ValueError(f"Unknown composition loss mode: {composition_mode}")
        self.composition_mode = composition_mode
        if perceptual and lpips_model is None:
            raise ValueError("LPIPS is requested but unavailable; explicitly set perceptual=0 for engineering runs")
        if self.lpips_model is not None:
            self.lpips_model.requires_grad_(False).eval()

    def forward(self, pred, target, backgrounds, valid_mask=None):
        pp, pa = premultiply(pred)
        tp, ta = premultiply(target)
        values = {"premult_l1": masked_mean((pp - tp).abs(), valid_mask),
                  "alpha_l1": masked_mean((pa - ta).abs(), valid_mask)}
        if self.composition_mode == "analytic_uniform":
            values["composition_mse"] = expected_uniform_composite_mse(pred, target, valid_mask)
        else:
            values["composition_mse"] = torch.stack([
                masked_mean((composite(pred, bg) - composite(target, bg)).square(), valid_mask)
                for bg in backgrounds
            ]).mean()
        values["lpips"] = pred.new_zeros(())
        if self.perc_weight:
            images = []
            for v in (0., 1.):
                bg = torch.full_like(target[:, :3], v)
                x = linear_to_srgb(composite(pred, bg)) * 2 - 1
                y = linear_to_srgb(composite(target, bg)) * 2 - 1
                images.append(self.lpips_model(x, y).mean())
            values["lpips"] = torch.stack(images).mean()
        values["total"] = (values["premult_l1"] + self.alpha_weight * values["alpha_l1"]
                           + self.comp_weight * values["composition_mse"]
                           + self.perc_weight * values["lpips"])
        if self.alpha_detail_weight:
            from alpharae.losses.alpha_detail import alpha_detail_losses
            values.update(alpha_detail_losses(pa, ta, valid_mask))
            values["total"] = values["total"] + self.alpha_detail_weight * (
                values["alpha_detail_l1"] + values["alpha_laplacian"])
        return values
