"""Download the public AlphaRAE reconstruction models into weights/."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'TonyZhan/AlphaRAE-Reconstruction'
NOTICES = ['LICENSE.md', 'LICENSE.RAEv2.txt', 'LICENSE.DINOv3.md',
           'LICENSE.CODE.txt', 'reconstruction_manifest.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec', choices=['all', 'k7', 'k23'], default='all')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    codecs = ['k7', 'k23'] if args.codec == 'all' else [args.codec]
    files = [f'codec_{kind}.pt' for kind in codecs] + NOTICES
    for filename in files:
        destination = args.weights_dir / filename
        # Replace only a local link; its original checkpoint remains untouched.
        if destination.is_symlink():
            destination.unlink()
        print(f'Downloading {filename}', flush=True)
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=args.weights_dir,
                        endpoint='https://huggingface.co', token=False)
    print(f'Reconstruction weights ready in {args.weights_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
