"""Create a manifest from explicit train/val/test directories of RGBA PNGs."""

import argparse
import json
import os
from pathlib import Path

from alpharae.data.rgba import inspect_png
from alpharae.data.manifest import read_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/rgba_manifest.jsonl'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ['train', 'val', 'test']:
        folder = getattr(args, split)
        if folder is None:
            continue
        for path in sorted(folder.rglob('*.png')):
            width, height = inspect_png(path)
            asset_id = f'{split}/{path.relative_to(folder).as_posix()}'
            rows.append({'asset_id': asset_id, 'group_id': asset_id,
                         'path': os.path.relpath(path.resolve(), args.output.parent.resolve()),
                         'split': split, 'alpha_mode': 'straight', 'source': 'user_rgba',
                         'width': width, 'height': height})
    if not any(r['split'] == 'train' for r in rows) or not any(r['split'] == 'val' for r in rows):
        raise ValueError('Both train and val must contain RGBA PNGs')
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    read_manifest(args.output)
    print(f'Wrote {len(rows)} assets to {args.output}')
