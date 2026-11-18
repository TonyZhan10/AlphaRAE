"""Launch single-node K7/K23 codec training without editing configuration files."""

import argparse
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from alpharae.runtime import ROOT, read_config
from alpharae.data.manifest import read_manifest, write_manifest
from scripts.download_training_weights import INITIALIZERS, COMMON_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', choices=['k7', 'k23'], default='k7')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/rgba_manifest.jsonl')
    parser.add_argument('--rgb-replay-manifest', type=Path, default=ROOT / 'data/rgb_replay.jsonl')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Use a small real-data subset and two updates')
    parser.add_argument('--online-metrics', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.gpus < 1 or args.gpus > torch.cuda.device_count():
        parser.error('--gpus must be between 1 and the number of visible CUDA GPUs')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    weights = args.weights_dir.resolve()
    missing = [name for name in [INITIALIZERS[args.codec], *COMMON_MODELS] if not (weights / name).is_file()]
    if missing:
        parser.error(f'Missing training assets: {missing}. Run python -m scripts.download_training_weights --codec {args.codec}')
    for path in [args.manifest, args.rgb_replay_manifest]:
        if not path.is_file():
            parser.error(f'Dataset manifest does not exist: {path}')
    output = (args.output or ROOT / f'outputs/codec_{args.codec}').resolve()
    if (output / 'last.pt').exists() and args.resume is None:
        parser.error('Output already contains last.pt; use --resume or choose another --output')
    config = read_config(ROOT / f'configs/codec_{args.codec}.yaml')
    config.update(manifest=str(args.manifest.resolve()),
                  rgb_replay_manifest=str(args.rgb_replay_manifest.resolve()),
                  pretrained_decoder=str(weights / INITIALIZERS[args.codec]),
                  torch_weights=str(weights / 'training/torch'),
                  batch_size=args.batch_size, effective_batch_size=args.gpus * args.batch_size,
                  planned_world_size=args.gpus, live_official_validation=args.online_metrics)
    config['encoder']['weights'] = str(weights / 'training/dinov3')
    rows = read_manifest(args.manifest)
    if args.smoke:
        count = max(8, config['effective_batch_size'])
        rows = [row for split, limit in [('train', count), ('val', 2)]
                for row in [row for row in rows if row['split'] == split][:limit]]
        output.mkdir(parents=True, exist_ok=True)
        subset = output / 'smoke_manifest.jsonl'
        write_manifest(rows, subset)
        config.update(manifest=str(subset), steps=2, epochs=3, num_workers=0,
                      initial_previews=False, validation_samples=1, eval_every=2,
                      checkpoint_milestones=[], log_interval=1)
    if args.steps is not None:
        if args.steps < 1:
            parser.error('--steps must be positive')
        config['steps'] = args.steps
    train_count = sum(row['split'] == 'train' for row in rows)
    batches = train_count // config['effective_batch_size']
    if batches < 1:
        parser.error('Training data must contain at least one complete global batch')
    if not any(row['split'] == 'val' for row in rows):
        parser.error('The RGBA manifest must include validation assets')
    config['epochs'] = max(config['epochs'], (config['steps'] + batches - 1) // batches)
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / 'launch_config.yaml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={args.gpus}', '-m', 'scripts.train_codec',
               '--config', str(config_file), '--output', str(output)]
    if args.resume:
        command.extend(['--resume', str(args.resume.resolve())])
    print(f'{args.codec}: {args.gpus} GPU(s), global batch {config["effective_batch_size"]}, '
          f'{config["steps"]} updates, output {output}', flush=True)
    print('The trainer validates before the first update; progress is recorded in status.json.', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
