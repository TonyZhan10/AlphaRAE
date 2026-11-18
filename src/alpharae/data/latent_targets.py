"""Training-only optimized latent labels, loaded one sample at a time."""

import json
from pathlib import Path

import numpy as np
import torch

from .rgba import RGBADataset


def train_asset_indices(dataset, asset_ids):
    """Validate a selection using train metadata without opening images."""
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("Duplicate asset IDs in latent target selection")
    by_id = {row["asset_id"]: index for index, row in enumerate(dataset.records)
             if row["split"] == "train"}
    missing = [asset_id for asset_id in asset_ids if asset_id not in by_id]
    if missing:
        raise ValueError(f"Latent targets require train assets; non-train or unknown IDs: {missing}")
    return [by_id[asset_id] for asset_id in asset_ids]


class LatentTargetDataset(RGBADataset):
    def __init__(self, manifest, index_path):
        self.index_path = Path(index_path)
        self.index = json.loads(self.index_path.read_text())
        self.metadata = self.index
        if self.index.get("split") != "train":
            raise ValueError("Latent targets require split='train'")
        super().__init__(manifest, self.index["resolution"], "train")
        self.target_records = self.index["records"]
        indices = train_asset_indices(self, [row["asset_id"] for row in self.target_records])
        self.records = [self.records[index] for index in indices]
        self.latent_shape = tuple(self.index["latent_shape"])
        self._shards = {}

    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = self.target_records[index]
        shard_name = row["shard"]
        if shard_name not in self._shards:
            shard = np.load(self.index_path.parent / shard_name, mmap_mode="r", allow_pickle=False)
            if shard.shape[1:] != self.latent_shape:
                raise ValueError(f"Unexpected latent shape in {shard_name}: {shard.shape}")
            self._shards[shard_name] = shard
        latent = self._shards[shard_name][row["index"]]
        result["target_latent"] = torch.from_numpy(np.array(latent, dtype=np.float32, copy=True))
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = {}
        return state
