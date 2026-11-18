"""JSONL manifests with explicit provenance and group-level split assignment."""

import json
from pathlib import Path
import random


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    group_splits = {}
    path_splits = {}
    for row in rows:
        row["path"] = str((path.parent / row["path"]).resolve())
        group = row.get("group_id", row.get("asset_id", row["path"]))
        split = row["split"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Asset group crosses splits: {group}")
        if row["path"] in path_splits and path_splits[row["path"]] != split:
            raise ValueError(f"Image path crosses splits: {row['path']}")
        group_splits[group] = split
        path_splits[row["path"]] = split
    return rows


def assign_group_splits(rows, seed=0):
    explicit = {}
    groups = sorted({row["group_id"] for row in rows})
    for row in rows:
        if row.get("split"):
            group, split = row["group_id"], row["split"]
            if split not in ("train", "val", "test"):
                raise ValueError(f"Unknown split {split!r}")
            if group in explicit and explicit[group] != split:
                raise ValueError(f"Conflicting official splits for group {group}")
            explicit[group] = split
    remaining = [group for group in groups if group not in explicit]
    random.Random(seed).shuffle(remaining)
    n = len(remaining)
    n_holdout = max(1, round(n * 0.1)) if n >= 3 else 0
    for i, group in enumerate(remaining):
        explicit[group] = "test" if i < n_holdout else "val" if i < 2 * n_holdout else "train"
    return [{**row, "split": explicit[row["group_id"]], "split_seed": seed} for row in rows]


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_manifest(root, output, source, license_note, seed=0, metadata=None):
    from .rgba import inspect_png

    root = Path(root).resolve()
    by_path = {}
    if metadata:
        with open(metadata, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    by_path[str((root / row["path"]).resolve())] = row
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".png":
            continue
        width, height = inspect_png(path)
        relative = path.relative_to(root).as_posix()
        supplied = by_path.get(str(path.resolve()), {})
        asset_id = supplied.get("asset_id", relative)
        rows.append({
            "asset_id": asset_id, "path": str(path.resolve()),
            "source": supplied.get("source", source), "width": width, "height": height,
            "alpha_mode": supplied.get("alpha_mode", "straight"),
            "split": supplied.get("split"), "caption": supplied.get("caption"),
            "caption_source": supplied.get("caption_source"),
            "group_id": supplied.get("group_id", asset_id),
            "license_note": supplied.get("license_note", license_note),
        })
    if not rows:
        raise ValueError(f"No PNG assets found under {root}")
    rows = assign_group_splits(rows, seed)
    write_manifest(rows, output)
    return rows
