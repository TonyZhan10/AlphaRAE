"""Download the class-conditioned ImageNet pipeline and its shared codec assets."""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from scripts.download_weights import ROOT, REPO_ID as RECONSTRUCTION_REPO, NOTICES

GENERATION_REPO = 'TonyZhan/AlphaRAE-ImageNet'
GENERATION_FILES = ['imagenet_dit.pt', 'rgba_adapter.pt', 'latent_stats.pt',
                    'imagenet_manifest.json', 'README.md', 'LICENSE.md',
                    'LICENSE.RAEv2.txt', 'LICENSE.CODE.txt', 'LICENSE.DINOv3.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    for filename in ['codec_k7.pt', 'training/rgb_k7_init.pt', *NOTICES]:
        print(f'Downloading shared asset {filename}', flush=True)
        hf_hub_download(repo_id=RECONSTRUCTION_REPO, filename=filename,
                        local_dir=args.weights_dir, endpoint='https://huggingface.co', token=False)
    for filename in GENERATION_FILES:
        print(f'Downloading generation asset {filename}', flush=True)
        hf_hub_download(repo_id=GENERATION_REPO, filename=filename,
                        local_dir=args.weights_dir / 'generation', endpoint='https://huggingface.co', token=False)
    print(f'ImageNet generation weights ready in {args.weights_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
