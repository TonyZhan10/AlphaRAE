"""Distortion-first RGBA supervision in the evaluation's sRGB coordinates."""

import torch
from torch import nn

from alpharae.data.rgba import linear_to_srgb
from alpharae.losses.alpha_detail import detail_region
from alpharae.losses.reconstruction import masked_mean


def nine_background_mse(pred, target, valid_mask=None, route_alpha=False):
    p, t = linear_to_srgb(pred[:,:3]), linear_to_srgb(target[:,:3])
    alpha=pred[:,3:].detach() if route_alpha else pred[:,3:]
    dp, da = p*alpha-t*target[:,3:], alpha-target[:,3:]
    gray = 128/255
    mean, second = (4+gray)/9, (4+gray*gray)/9
    value=masked_mean(dp.square()-2*mean*dp*da+second*da.square(),valid_mask)
    if route_alpha:
        alpha_error=masked_mean((pred[:,3:]-target[:,3:]).square(),valid_mask)
        # Preserve the scalar ABMSE exactly; only its opacity backward rule changes.
        value=value+(second-mean*mean)*(alpha_error-alpha_error.detach())
    return value


class RGBAFidelityLoss(nn.Module):
    metric_keys = ("srgb_nine_mse","alpha_mse","edge_alpha_mse","srgb_gradient_mse")

    def __init__(self, composition_weight=1000., alpha_weight=100., edge_weight=10., gradient_weight=10.,
                 perceptual_weight=0., lpips_model=None, route_alpha=False, gt_alpha_color=False):
        super().__init__()
        if route_alpha and gt_alpha_color:
            raise ValueError("Use either routed composition or GT-factor color supervision")
        self.weights=(composition_weight,alpha_weight,edge_weight,gradient_weight)
        self.perceptual_weight=perceptual_weight
        self.route_alpha=route_alpha
        self.gt_alpha_color=gt_alpha_color
        self.composition_key="gt_factor_mse" if gt_alpha_color else "srgb_nine_mse"
        self.gradient_key="gt_factor_gradient_mse" if gt_alpha_color else "srgb_gradient_mse"
        self.metric_keys=(self.composition_key,"alpha_mse","edge_alpha_mse",self.gradient_key)
        if gt_alpha_color:
            self.metric_keys=(*self.metric_keys,"srgb_nine_mse")
        self.lpips_model=lpips_model
        if perceptual_weight:
            if lpips_model is None:
                raise ValueError("Perceptual fidelity training requires a loaded LPIPS model")
            self.lpips_model.requires_grad_(False).eval()
            self.metric_keys=(*self.metric_keys,"fidelity_lpips")

    def forward(self,pred,target,valid_mask):
        alpha,target_alpha=pred[:,3:],target[:,3:]
        color_alpha=target_alpha if self.gt_alpha_color else alpha.detach() if self.route_alpha else alpha
        p=torch.cat((linear_to_srgb(pred[:,:3])*color_alpha,alpha),1)
        t=torch.cat((linear_to_srgb(target[:,:3])*target_alpha,target_alpha),1)
        error=p-t
        mx=valid_mask[...,1:]*valid_mask[...,:-1]
        my=valid_mask[...,1:,:]*valid_mask[...,:-1,:]
        gradient=(masked_mean((error[...,1:]-error[...,:-1]).square(),mx)
                  +masked_mean((error[...,1:,:]-error[...,:-1,:]).square(),my))/2
        alpha_mse=masked_mean((alpha-target_alpha).square(),valid_mask)
        if self.gt_alpha_color:
            gray=128/255
            mean,second=(4+gray)/9,(4+gray*gray)/9
            # Match the routed alpha branch while using GT visibility for color.
            composition=masked_mean(error[:,:3].square(),valid_mask)+(second-mean*mean)*alpha_mse
        else:
            composition=nine_background_mse(pred,target,valid_mask,self.route_alpha)
        values={self.composition_key:composition,
                "alpha_mse":alpha_mse,
                "edge_alpha_mse":masked_mean((alpha-target_alpha).square(),detail_region(target_alpha)*valid_mask),
                self.gradient_key:gradient}
        values["total"]=sum(weight*values[key] for key,weight in zip(self.metric_keys,self.weights))
        if self.gt_alpha_color:
            with torch.no_grad():
                values["srgb_nine_mse"]=nine_background_mse(pred,target,valid_mask)
        if self.perceptual_weight:
            distances=[]
            for background in (0.,1.):
                prediction=(p[:,:3]+(1-color_alpha)*background)*2-1
                truth=(t[:,:3]+(1-target_alpha)*background)*2-1
                distances.append(self.lpips_model(prediction,truth).mean())
            values["fidelity_lpips"]=torch.stack(distances).mean()
            values["total"]=values["total"]+self.perceptual_weight*values["fidelity_lpips"]
        return values
