"""Reconstruct straight-alpha RGBA images with the released K7/K23 codecs."""

import argparse
from pathlib import Path
import time

import torch

from alpharae.runtime import ROOT, load_codec, write_json
from alpharae.data.rgba import load_rgba
from alpharae.visualization import save_views


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reconstruct'], default='reconstruct')
    parser.add_argument('--codec', choices=['k7', 'k23'], default='k7')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.weights_dir / f'codec_{args.codec}.pt'
    codec = load_codec(checkpoint, device)
    rgba = load_rgba(args.input, codec.resolution)['rgba'][None].to(device)
    prediction, latent = codec(rgba)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite RGBA reconstruction')
    files = save_views(prediction[0], args.output, 'reconstruction')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    write_json(args.output / 'report.json', {
        'status': 'complete', 'mode': 'reconstruct', 'codec': args.codec,
        'checkpoint': str(checkpoint), 'resolution': codec.resolution,
        'device': str(device), 'seconds': time.time() - started,
        'input': str(args.input), 'latent_shape': list(latent.shape), 'files': files})
    print(f'Reconstruction written to {args.output}', flush=True)


if __name__ == '__main__':
    main()
