"""Generate 256-pixel RGBA assets from ImageNet class IDs and random seeds."""

import argparse
from pathlib import Path
import time

import torch
from torchvision.models import ResNet50_Weights

from alpharae.runtime import ROOT, write_json
from alpharae.generation.imagenet import load_pipeline
from alpharae.visualization import image_file, save_views
from configs.stage2 import CFGConfig, GuidanceConfig, IGConfig
from stage2.transport import Transport, Sampler
from utils.guidance_utils import get_model_forward_fn


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes', type=int, nargs='+', default=[985, 849])
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260926])
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/imagenet')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not all(0 <= label < 1000 for label in args.classes):
        parser.error('--classes must contain ImageNet class IDs between 0 and 999')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    codec, decoder, prior, adapter, space = load_pipeline(args.weights_dir, device)
    guidance = GuidanceConfig(cfg=CFGConfig(scale=1.), ig=IGConfig(scale=1.78, t_min=.1, t_max=1.))
    transport = Transport(prediction='x', time_dist_type='logit-normal_0_1', time_dist_shift=8., t_eps=.05)
    sample = Sampler(transport, guidance).sample_ode(num_steps=100)
    forward, kwargs = get_model_forward_fn(prior, guidance)
    categories = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
    zero = torch.zeros((1, 1088, 16, 16), device=device)
    one = torch.ones(1, device=device)
    records = []
    for label in args.classes:
        for seed in args.seeds:
            noise = torch.randn((1, 1024, 16, 16), device=device,
                                 generator=torch.Generator(device=device).manual_seed(seed))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                latent = sample(torch.cat((noise, noise)), forward,
                                context=torch.tensor([label, 1000], device=device), **kwargs)[-1][:1].float()
            raw = latent * space.scale + space.mean
            rgb = decoder.unpatchify(decoder(raw.flatten(2).transpose(1, 2), drop_cls_token=False).logits).float().clamp(0, 1)
            rgb = rgb.mul(255).round() / 255
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                residual = adapter(zero, one, context=space.condition(raw), rgb=rgb).float()
            rgba = codec.decode(space.compose(residual, raw))[0]
            if not torch.isfinite(rgba).all():
                raise RuntimeError('Nonfinite RGBA generation')
            prefix = f'class_{label:04d}_seed_{seed}'
            image_file(rgb[0], args.output / f'{prefix}_native_rgb.png')
            files = save_views(rgba, args.output, prefix)
            records.append({'class_id': label, 'class_name': categories[label], 'seed': seed,
                            'native_rgb': f'{prefix}_native_rgb.png', 'files': files})
            print(f'Generated {categories[label]} (class {label}), seed {seed}', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    write_json(args.output / 'report.json', {
        'status': 'complete', 'resolution': 256, 'input': 'ImageNet class ID and random seed only',
        'sampling': {'solver': 'Euler', 'steps': 100, 'cfg_scale': 1., 'ig_scale': 1.78, 'ig_interval': [.1, 1.]},
        'models_frozen': True, 'teacher_loaded': False, 'target_image_used': False,
        'seconds': time.time() - started, 'samples': records})


if __name__ == '__main__':
    main()
