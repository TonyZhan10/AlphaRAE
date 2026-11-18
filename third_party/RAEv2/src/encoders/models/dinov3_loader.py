"""Offline DINOv3-L/16 loader matching the released LVD1689M architecture."""
import os
from pathlib import Path
import torch
from torchvision import transforms


def make_dinov3_transform(resize_size=224):
    return transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])


def load_dinov3(model_name, pretrained=True):
    from dinov3.models.vision_transformer import DinoVisionTransformer
    if model_name != 'dinov3_vitl16':
        raise ValueError('This release uses the DINOv3-L/16 LVD1689M backbone')
    model = DinoVisionTransformer(
        img_size=224, patch_size=16, in_chans=3, pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords='separate', pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype='fp32', embed_dim=1024, depth=24, num_heads=16,
        ffn_ratio=4, qkv_bias=True, drop_path_rate=0., layerscale_init=1.e-5,
        norm_layer='layernormbf16', ffn_layer='mlp', ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True, untie_global_and_local_cls_norm=False)
    if pretrained:
        files = sorted(Path(os.environ['DINOV3_CKPT_DIR']).glob('dinov3_vitl16_pretrain_lvd1689m*.pth'))
        if len(files) != 1:
            raise FileNotFoundError('Place one DINOv3-L/16 LVD1689M checkpoint in weights/dinov3')
        state = torch.load(files[0], map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
    return model
