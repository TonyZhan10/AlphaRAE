"""Train the K7/K23 RGBA codec using the original distributed training loop."""

import argparse
from pathlib import Path
import torch
from alpharae.runtime import read_config
from scripts.train_autoencoder_distributed import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.steps is not None:
        config['steps'] = args.steps
    torch.hub.set_dir(str(Path(config.get('torch_weights', 'weights/torch')) / 'hub'))
    train(config, args.output, args.resume)


if __name__ == '__main__':
    main()
