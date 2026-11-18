"""Download the original RGB initialization and all codec-training model caches."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from scripts.download_weights import ROOT, REPO_ID, NOTICES

INITIALIZERS = {kind: f'training/rgb_{kind}_init.pt' for kind in ['k7', 'k23']}
COMMON_MODELS = [
    'training/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'training/torch/hub/checkpoints/alexnet-owt-7be5be79.pth',
    'training/torch/hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth',
]
TRAINING_NOTICES = ['training/manifest.json', 'training/README.md',
                    'training/LICENSE.Torchvision.txt', 'training/LICENSE.LPIPS.txt',
                    'training/LICENSE.PyIQA.txt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', choices=['all', 'k7', 'k23'], default='all')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    kinds = ['k7', 'k23'] if args.codec == 'all' else [args.codec]
    files = [INITIALIZERS[kind] for kind in kinds] + COMMON_MODELS + TRAINING_NOTICES + NOTICES
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        print(f'Downloading {filename}', flush=True)
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=args.weights_dir,
                        endpoint='https://huggingface.co', token=False)
    print(f'Training initialization ready in {args.weights_dir.resolve() / "training"}', flush=True)


if __name__ == '__main__':
    main()
