"""Portable paths and strict loading for the released AlphaRAE models."""

from pathlib import Path
import json
import os
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def activate_sources():
    for path in (ROOT / 'src', ROOT / 'third_party/dinov3', ROOT / 'third_party/RAEv2/src'):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'weights/torch'))


activate_sources()

PATH_KEYS = {'manifest', 'rgb_replay_manifest', 'opaque_val_manifest', 'pretrained_decoder',
             'rae_checkpoint', 'torch_weights', 'output', 'init_checkpoint'}


def resolve_config(config):
    """Config paths are relative to the release root, independent of shell cwd."""
    config = dict(config)
    for key in PATH_KEYS:
        if config.get(key):
            path = Path(config[key]).expanduser()
            config[key] = str(path if path.is_absolute() else ROOT / path)
    config['raev2_source'] = str(ROOT / 'third_party/RAEv2')
    config['dinov3_source'] = str(ROOT / 'third_party/dinov3')
    if 'encoder' in config:
        config['encoder'] = dict(config['encoder'])
        path = Path(config['encoder'].get('weights', 'weights/dinov3')).expanduser()
        config['encoder']['weights'] = str(path if path.is_absolute() else ROOT / path)
    return config


def read_config(path):
    path = Path(path)
    text = path.read_text()
    config = json.loads(text) if path.suffix == '.json' else yaml.safe_load(text)
    return resolve_config(config)


def load_codec(config, device='cpu'):
    """Load the published model, ignoring stale training-machine asset paths."""
    from alpharae.training import build_system

    path = config['rae_checkpoint'] if isinstance(config, dict) else config
    package = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    cfg = package['config']
    cfg = {key: cfg[key] for key in ('model_type', 'variant', 'encoder', 'alpha_scale',
           'joint_detail_rgb', 'rgb_residual_scale', 'residual_interpolation',
           'decoder_gradient_checkpointing') if key in cfg}
    cfg = resolve_config(cfg)
    # A complete codec checkpoint already contains its trained RGB decoder.
    cfg['load_pretrained_rgb'] = False
    cfg['load_pretrained_encoder'] = False
    model, _ = build_system(cfg, device)
    model.load_state_dict(package['model'], strict=True)
    if isinstance(config, dict) and config.get('rae_step') is not None:
        if package['step'] != config['rae_step']:
            raise ValueError('RGBA codec step differs from the requested checkpoint')
    return model.eval().requires_grad_(False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
