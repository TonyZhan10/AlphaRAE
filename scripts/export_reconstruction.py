"""Export a trained codec without optimizer, RNG or training-machine paths."""

import argparse
from pathlib import Path

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = state['config']
    keys = ['model_type', 'variant', 'alpha_scale', 'joint_detail_rgb',
            'rgb_residual_scale', 'residual_interpolation', 'decoder_gradient_checkpointing']
    architecture = {key: config[key] for key in keys if key in config}
    architecture['encoder'] = {key: config['encoder'][key] for key in ['kind', 'resolution']}
    package = {'format': 'alpharae_reconstruction_v1', 'step': state['step'],
               'resolution': state['resolution'], 'config': architecture,
               'model': {key: value.detach().cpu() for key, value in state['model'].items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)
    print(f'Exported reconstruction checkpoint to {args.output}')
