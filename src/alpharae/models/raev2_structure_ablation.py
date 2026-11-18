"""Controlled RGBA representation variants; the default JointRGBA is unchanged."""

from math import isqrt

import torch
from torch import nn
from torch.nn import functional as F


class ZeroDetail(nn.Module):
    def forward(self, rgba):
        return rgba.new_zeros((len(rgba),16,rgba.shape[-2]//8,rgba.shape[-1]//8))


class AlphaOnlyDetail(nn.Sequential):
    def forward(self, value):
        # Retain a four-channel, equally sized encoder; all channels observe alpha.
        return super().forward(value[:,3:].expand(-1,4,-1,-1))


class IndependentReadoutBranch(nn.Module):
    def __init__(self,context_dim,output_channels):
        super().__init__()
        self.context=nn.Conv2d(context_dim,32,1)
        self.fuse=nn.Sequential(nn.Conv2d(48,95,3,padding=1),nn.GELU())
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU())
            for a,b in ((95,47),(47,23),(23,11))])
        self.interpolation_gains=nn.Parameter(torch.zeros(3))
        self.output=nn.Conv2d(11,output_channels,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,detail,tokens):
        grid=isqrt(tokens.shape[1])
        context=tokens.transpose(1,2).reshape(len(tokens),tokens.shape[-1],grid,grid)
        context=F.interpolate(self.context(context),size=detail.shape[-2:],mode='bilinear',align_corners=False)
        value=self.fuse(torch.cat((context,detail),1))
        for index,block in enumerate(self.blocks):
            smooth=F.interpolate(value,scale_factor=2,mode='bilinear',align_corners=False)
            nearest=F.interpolate(value,scale_factor=2,mode='nearest')
            value=block(smooth+self.interpolation_gains[index]*(nearest-smooth))
        return self.output(value).float()


class SeparateRGBAReadout(nn.Module):
    def __init__(self,context_dim):
        super().__init__()
        self.color=IndependentReadoutBranch(context_dim,3)
        self.alpha=IndependentReadoutBranch(context_dim,1)

    @property
    def interpolation_gains(self):
        return torch.cat((self.color.interpolation_gains,self.alpha.interpolation_gains))

    def forward(self,detail,tokens):
        return torch.cat((self.color(detail,tokens),self.alpha(detail,tokens)),1)


def apply_variant(model,config):
    arm=config['structure_ablation']
    assert model.joint_detail_rgb and model.alpha_readout.interpolation_gains is not None
    if arm=='no_detail':
        model.alpha_encoder=ZeroDetail()
    elif arm=='no_injection':
        model.alpha_scale=0.
    elif arm=='alpha_only_detail':
        model.alpha_encoder.__class__=AlphaOnlyDetail
    elif arm=='separate_readout':
        model.alpha_readout=SeparateRGBAReadout(model.decoder.config.decoder_hidden_size)
    elif arm!='full_seed2':
        raise ValueError(f'Unknown structure comparison: {arm}')
    return model
