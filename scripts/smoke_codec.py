"""Run real codec optimizer updates, save/resume, and load the saved checkpoint."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from alpharae.runtime import ROOT, read_config, write_json
from alpharae.data.manifest import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_config(args.config)
    rows = read_manifest(cfg['manifest'])
    subset = [r for split, limit in [('train', 8), ('val', 2)]
              for r in [r for r in rows if r['split'] == split][:limit]]
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in subset))
    cfg.update(manifest=str(manifest), steps=2, epochs=3, planned_world_size=1,
               effective_batch_size=1, batch_size=1, num_workers=0,
               live_official_validation=False, opaque_val_manifest=None,
               validation_samples=1, initial_previews=False, eval_every=2,
               checkpoint_milestones=[], log_interval=1, live_status=True)
    config = output / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run = output / 'train'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=1', '-m', 'scripts.train_codec', '--config', str(config),
               '--output', str(run)]
    subprocess.run(command, cwd=ROOT, check=True)
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    before = checkpoint['model']['alpha_readout.output.weight'].clone()
    initial_steps = checkpoint['step']
    del checkpoint
    subprocess.run(command + ['--steps', '3', '--resume', str(run / 'last.pt')], cwd=ROOT, check=True)
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    delta = (checkpoint['model']['alpha_readout.output.weight'] - before).abs().max().item()
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    train_records = [row for row in records if 'train' in row]
    if initial_steps != 2 or checkpoint['step'] != 3 or delta <= 0:
        raise RuntimeError('Checkpoint resume did not perform the expected parameter update')
    if not all(row['gradient_norm'] > 0 for row in train_records):
        raise RuntimeError('The smoke run must have finite nonzero gradients')
    write_json(output / 'report.json', {
        'status': 'passed', 'scope': 'Training startup and resume; not a converged result',
        'resolution': checkpoint['resolution'], 'initial_steps': initial_steps,
        'resumed_steps': checkpoint['step'], 'readout_parameter_max_change_after_resume': delta,
        'optimizer_states': len(checkpoint['optimizer']['state']),
        'training_records': train_records, 'final_validation': checkpoint['last_validation'],
        'train_assets': sum(r['split'] == 'train' for r in subset),
        'val_assets': sum(r['split'] == 'val' for r in subset),
        'full_paper_loss_retained': cfg['fidelity_loss']})
    print(f'Codec smoke/resume passed: {output}', flush=True)


if __name__ == '__main__':
    main()
