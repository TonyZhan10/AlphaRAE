"""Joint RGBA decoding from direct1024 RGB weights with small LoRA updates."""

from math import isqrt

import torch
from torch import nn
from torch.nn import functional as F

from alpharae.data.rgba import composite, linear_to_srgb, srgb_to_linear
from alpharae.models.raev2_rgba import CompressedAlphaEncoder, LinearLightRAEv2Encoder, load_trained_rgb


def inject_decoder_lora(decoder, rank=16, alpha=16):
    from peft import LoraConfig, inject_adapter_in_model
    decoder.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias="none",
        target_modules=r"decoder_layers\.\d+\.attention\.(attention\.(query|key|value)|output\.dense)")
    return inject_adapter_in_model(config, decoder)


class JointAlphaReadout(nn.Module):
    """Decode alpha using both the RGB decoder's final features and alpha code."""

    def __init__(self, context_dim, alpha_channels=16, residual_interpolation=False):
        super().__init__()
        self.context = nn.Conv2d(context_dim, 64, 1)
        self.fuse = nn.Sequential(nn.Conv2d(64 + alpha_channels, 128, 3, padding=1), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU())
            for a,b in ((128,64),(64,32),(32,16))])
        self.output = nn.Conv2d(16,1,3,padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.interpolation_gains = nn.Parameter(torch.zeros(len(self.blocks))) if residual_interpolation else None

    def forward(self, alpha_code, rgb_features):
        grid=isqrt(rgb_features.shape[1])
        context=rgb_features.transpose(1,2).reshape(rgb_features.shape[0],rgb_features.shape[2],grid,grid)
        context=F.interpolate(self.context(context),size=alpha_code.shape[-2:],mode="bilinear",align_corners=False)
        features=self.fuse(torch.cat((context,alpha_code),dim=1))
        for index,block in enumerate(self.blocks):
            smooth=F.interpolate(features,scale_factor=2,mode="bilinear",align_corners=False)
            if self.interpolation_gains is not None:
                nearest=F.interpolate(features,scale_factor=2,mode="nearest")
                smooth=smooth+self.interpolation_gains[index]*(nearest-smooth)
            features=block(smooth)
        return self.output(features).float()


class JointRGBA(nn.Module):
    def __init__(self, encoder, decoder, alpha_scale=.1, joint_detail_rgb=False, rgb_residual_scale=1.,
                 freeze_rgb_in_forward=False, residual_interpolation=False):
        super().__init__()
        self.encoder,self.decoder=encoder,decoder
        self.resolution=encoder.resolution
        self.base_latent_dim=encoder.dim
        self.latent_dim=encoder.dim+64
        self.num_tokens=encoder.num_tokens
        self.patch_size=16
        self.alpha_scale=alpha_scale
        self.alpha_encoder=CompressedAlphaEncoder(16,8)
        self.alpha_readout=JointAlphaReadout(decoder.config.decoder_hidden_size,residual_interpolation=residual_interpolation)
        self.joint_detail_rgb = joint_detail_rgb
        self.rgb_residual_scale = rgb_residual_scale
        self.freeze_rgb_in_forward = freeze_rgb_in_forward
        if joint_detail_rgb:
            first = self.alpha_encoder[0]
            self.alpha_encoder[0] = nn.Conv2d(4, first.out_channels, 3, stride=2, padding=1)
            self.alpha_readout.output = nn.Conv2d(16, 4, 3, padding=1)
            nn.init.zeros_(self.alpha_readout.output.weight)
            nn.init.zeros_(self.alpha_readout.output.bias)
        self.register_buffer("stats_fitted",torch.tensor(True))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_rgb_in_forward:
            self.decoder.eval()
        return self

    def encode(self, rgba, observations=None):
        if rgba.shape[1:] != (4,self.resolution,self.resolution):
            raise ValueError("Joint RGBA input must match its native resolution")
        white=(self.encoder(composite(rgba,torch.ones_like(rgba[:,:3])))
               if observations is None else observations[1])
        opacity=F.pixel_unshuffle(1-rgba[:,3:],16).flatten(2).transpose(1,2)
        base=white.float()+F.pad(opacity,(0,self.base_latent_dim-256)).float()*self.alpha_scale
        # Packing changes storage layout only; decoding exactly reverses it.
        detail_input = (torch.cat((linear_to_srgb(rgba[:,:3])*rgba[:,3:],rgba[:,3:]),1)
                        if self.joint_detail_rgb else rgba[:,3:])
        code=F.pixel_unshuffle(self.alpha_encoder(detail_input),2).flatten(2).transpose(1,2)
        return torch.cat((base,code.float()),dim=-1)

    def decode(self, latent, *, detach_base=False):
        if latent.shape[1:] != (self.num_tokens,self.latent_dim):
            raise ValueError("Joint latent grid/channel count does not match decoder")
        base=latent[:,:,:self.base_latent_dim]
        decoded=self.decoder(base.detach() if detach_base else base,drop_cls_token=False,
                             output_last_hidden_state=True,return_dict=True)
        rgb_srgb=self.decoder.unpatchify(decoded.logits).float().clamp(0,1)
        grid=self.resolution//16
        code=latent[:,:,self.base_latent_dim:].transpose(1,2).reshape(latent.shape[0],64,grid,grid)
        alpha_code=F.pixel_shuffle(code,2)
        spatial=self.alpha_readout(alpha_code,decoded.last_hidden_state)
        if self.joint_detail_rgb:
            rgb_srgb=(rgb_srgb+self.rgb_residual_scale*spatial[:,:3]).clamp(0,1)
            alpha=spatial[:,3:].sigmoid()
        else:
            alpha=spatial.sigmoid()
        rgb=srgb_to_linear(rgb_srgb)
        return torch.cat((rgb,alpha),dim=1)

    def forward(self, rgba, observations=None, decode_count=None):
        latent=self.encode(rgba,observations)
        return self.decode(latent if decode_count is None else latent[:decode_count],
                           detach_base=self.freeze_rgb_in_forward),latent

    def optimizer_groups(self, config):
        groups = [{"params":[p for p in self.decoder.parameters() if p.requires_grad],
                 "lr":config["lr"],"name":"decoder_full" if config.get("decoder_train_mode")=="full" else "decoder_lora"},
                {"params":list(self.alpha_encoder.parameters())+list(self.alpha_readout.parameters()),
                 "lr":config["alpha_lr"],"name":"rgba_spatial"}]
        return [group for group in groups if group["params"]]


JointLoRARGBA = JointRGBA


def build_joint_full(config):
    """Start from pure RGB EMA weights with no LoRA construction or RGBA initializer."""
    rgb = load_trained_rgb(config)
    encoder = LinearLightRAEv2Encoder(rgb.encoder, rgb.resolution)
    model = JointRGBA(encoder, rgb.decoder, config.get("alpha_scale", .1),
                     config.get("joint_detail_rgb", True), config.get("rgb_residual_scale", .1),
                     False, config.get("residual_interpolation", False))
    if config.get("structure_ablation"):
        from alpharae.models.raev2_structure_ablation import apply_variant
        model = apply_variant(model, config)
    return model


def build_joint_lora(config):
    rgb=load_trained_rgb(config)
    decoder=inject_decoder_lora(rgb.decoder,config.get("lora_rank",16),config.get("lora_alpha",16))
    if config.get("decoder_train_mode", "lora") == "full":
        # Preserve the learned LoRA contribution while adapting the base decoder.
        for name, parameter in decoder.named_parameters():
            parameter.requires_grad_(".lora_A." not in name and ".lora_B." not in name)
    elif config.get("decoder_train_mode") == "frozen":
        decoder.requires_grad_(False)
    encoder=LinearLightRAEv2Encoder(rgb.encoder,rgb.resolution)
    return JointLoRARGBA(encoder,decoder,config.get("alpha_scale",.1),config.get("joint_detail_rgb",False),
                        config.get("rgb_residual_scale",1.),config.get("decoder_train_mode")=="frozen",
                        config.get("residual_interpolation",False))


def load_joint_initializer(model, state):
    if model.joint_detail_rgb and state["alpha_encoder.0.weight"].shape[1] == 1:
        state = dict(state)
        first = torch.zeros_like(model.alpha_encoder[0].weight, device="cpu")
        first[:,3:] = state["alpha_encoder.0.weight"].cpu()
        weight = torch.zeros_like(model.alpha_readout.output.weight, device="cpu")
        bias = torch.zeros_like(model.alpha_readout.output.bias, device="cpu")
        weight[3:] = state["alpha_readout.output.weight"].cpu()
        bias[3:] = state["alpha_readout.output.bias"].cpu()
        state.update({"alpha_encoder.0.weight":first,"alpha_readout.output.weight":weight,
                      "alpha_readout.output.bias":bias})
    if model.alpha_readout.interpolation_gains is not None and "alpha_readout.interpolation_gains" not in state:
        state = dict(state)
        state["alpha_readout.interpolation_gains"] = torch.zeros_like(model.alpha_readout.interpolation_gains,device="cpu")
    model.load_state_dict(state)
