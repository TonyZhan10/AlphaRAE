import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from alpharae.models.autoencoder import build_autoencoder
from alpharae.models.response import ResponseSupervisor


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_system(config, device):
    model = build_autoencoder(config).to(device)
    response = None
    if config.get("variant", "B1") in ("M1", "C1", "C3"):
        response = ResponseSupervisor(model.encoder.dim, model.latent_dim, model.num_tokens,
                                      **config.get("response", {})).to(device)
    return model, response


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class BatchStream:
    """Fixed-size batches spanning permutations, with an explicit cursor for resume."""
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor = torch.randperm(size, generator=self.generator), 0

    def next(self):
        indices = []
        while len(indices) < self.batch_size:
            if self.cursor == self.size:
                self.order, self.cursor = torch.randperm(self.size, generator=self.generator), 0
            end = min(self.size, self.cursor + self.batch_size - len(indices))
            indices.extend(self.order[self.cursor:end].tolist())
            self.cursor = end
        return indices

    def state_dict(self):
        return {"order": self.order, "cursor": self.cursor, "rng": self.generator.get_state(),
                "size": self.size, "batch_size": self.batch_size}

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Resume requires the same dataset size and batch size")
        self.order, self.cursor = state["order"].cpu(), state["cursor"]
        self.generator.set_state(state["rng"].cpu())


def load_run(checkpoint, device="cpu"):
    # Checkpoints are local experiment artifacts, including optimizer and RNG objects.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, response = build_system(state["config"], device)
    model.load_state_dict(state["model"])
    if response is not None:
        response.load_state_dict(state["response"])
    return model, response, state
