"""Torchrun/DDP reconstruction training with online frozen features."""

import argparse
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from alpharae.data.rgba import INPUT_KINDS, RGBADataset
from alpharae.losses.reconstruction import ReconstructionLoss
from alpharae.training import build_system, read_config, write_json
from scripts.train_autoencoder import collect_batch, save_validation_samples


LOSS_KEYS = ("premult_l1", "alpha_l1", "composition_mse", "lpips", "total")
TEACHER_SEED_OFFSET = 1000003


def latent_target_mse(latent, target):
    return F.mse_loss(latent.float(), target.detach().float(), reduction="mean")


def seed_rank(seed, device, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def rank_rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rank_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


class ResumedBatchSampler(BatchSampler):
    """Skip consumed indices before loading pixels or starting worker prefetch."""

    def __init__(self, sampler, batch_size, start_batch=0):
        super().__init__(sampler, batch_size, drop_last=True)
        self.start_batch = start_batch

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_batch, None)

    def __len__(self):
        return super().__len__() - self.start_batch


def atomic_checkpoint(state, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def validate_resume_configuration(config, state, world_size):
    allowed_changes = {"steps", "epochs", "gpu_hours", "device", "cpu_threads", "num_workers"}
    previous_tail, new_tail = state["config"].get("lr_tail"), config.get("lr_tail")
    if previous_tail is None and new_tail is not None:
        if new_tail["start_step"] != state["step"]:
            raise ValueError("A new LR tail must start at the resumed checkpoint step")
        allowed_changes.add("lr_tail")
    changes = [key for key in set(config) | set(state["config"])
               if key not in allowed_changes and config.get(key) != state["config"].get(key)]
    if changes or state["world_size"] != world_size:
        raise ValueError(f"Resume requires the same world size and training configuration: {changes}")


def build_lr_scheduler(optimizer, config, total_steps):
    name, tail = config.get("lr_scheduler"), config.get("lr_tail")
    warmup = config.get("lr_warmup_steps", 100)
    if tail is not None:
        start, end, minimum = tail["start_step"], tail["end_step"], tail["min_lr_ratio"]
        if name != "constant_with_warmup" or not (0 <= warmup <= start < end and 0 < minimum <= 1):
            raise ValueError("LR tail requires constant_with_warmup, warmup <= start < end, and 0 < min_lr_ratio <= 1")

        def multiplier(step):
            if step < warmup:
                return float(step) / max(1, warmup)
            if step <= start:
                return 1.
            fraction = min(1., (step - start) / (end - start))
            return minimum + (1. - minimum) * .5 * (1. + math.cos(math.pi * fraction))

        # Global update indices and fixed endpoints survive checkpoint resumption.
        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if name == "cosine":
        from transformers.optimization import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                             num_training_steps=total_steps)
    if name == "constant_with_warmup":
        from transformers.optimization import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    if name not in (None, "constant"):
        raise ValueError("lr_scheduler must be cosine, constant, or constant_with_warmup")
    return None


@torch.no_grad()
def validate_distributed(model, dataset, objective, batch_size, device, bf16, alpha_structure=False, official=None,
                         progress=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    was_training = model.training
    model.eval()
    loss_keys = LOSS_KEYS + objective.extra_metric_keys
    if alpha_structure:
        loss_keys += ("detail_gradient_l1", "gradient_gain")
    if official is not None:
        loss_keys += ("official_psnr", "official_ssim", "official_lpips")
    totals = torch.zeros(len(loss_keys) + 1, dtype=torch.float64, device=device)
    indices = list(range(rank, len(dataset), world_size))
    for start in range(0, len(indices), batch_size):
        rgba, mask = collect_batch(dataset, indices[start:start + batch_size], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            pred, _ = model(rgba)
        values = objective(pred.float(), rgba, [], mask)
        if alpha_structure:
            from alpharae.losses.alpha_detail import alpha_gradient_metrics
            values.update(alpha_gradient_metrics(pred[:,3:].float(),rgba[:,3:],mask))
        if official is not None:
            values.update(official(pred.float(),rgba))
        totals[:-1] += torch.stack([values[key].double() for key in loss_keys]) * len(rgba)
        totals[-1] += len(rgba)
        if progress is not None and rank == 0:
            progress(min(len(dataset), (start + len(rgba)) * world_size), len(dataset))
    dist.all_reduce(totals)
    model.train(was_training)
    result = {key: float(totals[index] / totals[-1]) for index, key in enumerate(loss_keys)}
    result["assets"] = int(totals[-1])
    return result


def train(config, output, resume=None):
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        return _train(config, Path(output), resume, device)
    finally:
        if owns_group:
            dist.destroy_process_group()


def _train(config, output, resume, device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if config.get("variant", "B1") in ("M1", "C1", "C2", "C3"):
        raise ValueError("Response losses are not enabled in this distributed trainer")
    if config.get("adversarial") and config.get("model_type") not in ("raev2_joint_lora", "raev2_joint_full"):
        raise ValueError("Distributed RGBA GAN is implemented for the joint RAE model")
    if config.get("feature_cache") or config.get("preload", False):
        raise ValueError("Distributed training loads pixels per batch and computes encoder features online")
    loss_config = {"perceptual": .5, "composition_mode": "analytic_uniform", **config.get("loss", {})}
    if loss_config["composition_mode"] != "analytic_uniform":
        raise ValueError("Distributed reconstruction requires composition_mode=analytic_uniform")
    batch_size = config.get("batch_size", 1)
    if config.get("effective_batch_size", batch_size * world_size) != batch_size * world_size:
        raise ValueError("effective_batch_size must equal batch_size * WORLD_SIZE; no accumulation is used")
    seed = config.get("seed", 0)
    seed_rank(seed, device, config.get("cpu_threads", 4))
    model, response = build_system(config, device)
    if response is not None:
        raise ValueError("Response supervision is not enabled in this reconstruction trainer")
    state = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        validate_resume_configuration(config, state, world_size)
        model.load_state_dict(state["model"])
    elif config.get("init_checkpoint"):
        initialization = torch.load(config["init_checkpoint"], map_location="cpu", weights_only=False)
        if config.get("model_type") == "pretrained_rgba":
            from alpharae.models.pretrained_rgba import adapt_alpha_state
            adapt_alpha_state(model, initialization["model"])
        elif config.get("model_type") == "raev2_rgba":
            from alpharae.models.raev2_rgba import load_rgba_initializer
            load_rgba_initializer(model, initialization["model"])
        elif config.get("model_type") in ("raev2_joint_lora", "raev2_joint_full"):
            from alpharae.models.raev2_joint_lora import load_joint_initializer
            load_joint_initializer(model, initialization["model"])
        else:
            model.load_state_dict(initialization["model"])
        del initialization
    if config.get("model_type") == "pretrained_rgba":
        model.stats_fitted.fill_(True)
    resolution = model.encoder.resolution
    train_data = RGBADataset(
        config["manifest"], resolution, "train", shift_pixels=config.get("train_shift_pixels", 0), seed=seed,
        background_blend_probability=config.get("background_blend_probability", 0.),
        rgb_replay_probability=config.get("rgb_replay_probability", 0.),
        rgb_replay_manifest=config.get("rgb_replay_manifest"))
    val_data = RGBADataset(config["manifest"], resolution, "val")
    opaque_data = (RGBADataset(config["opaque_val_manifest"], resolution, "val")
                   if config.get("opaque_val_manifest") else None)
    sampler = DistributedSampler(train_data, world_size, rank, shuffle=True, seed=seed, drop_last=True)
    batches_per_epoch = len(sampler) // batch_size
    if not batches_per_epoch:
        raise ValueError("The train split must contain at least one full global batch")
    teacher_config = config.get("latent_targets")
    teacher_data, teacher_sampler = None, None
    teacher_epoch, teacher_cursor = 0, 0
    train_loss_keys = LOSS_KEYS
    if teacher_config:
        from alpharae.data.latent_targets import LatentTargetDataset
        teacher_data = LatentTargetDataset(config["manifest"], teacher_config["index"])
        if (teacher_data.resolution != resolution
                or teacher_data.latent_shape != (model.num_tokens, model.latent_dim)):
            raise ValueError("Latent target resolution and shape must match the student")
        teacher_sampler = DistributedSampler(teacher_data, world_size, rank, shuffle=True,
                                             seed=seed + TEACHER_SEED_OFFSET, drop_last=True)
        teacher_batches = len(teacher_sampler) // batch_size
        if not teacher_batches:
            raise ValueError("Latent targets must contain at least one full global batch")
        train_loss_keys += ("reconstruction_total", "latent_target_mse", "latent_target_weighted")
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                  broadcast_buffers=False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups=model.optimizer_groups(config) if config.get("model_type") in ("raev2_joint_lora", "raev2_joint_full") else parameters
    optimizer = torch.optim.AdamW(groups, lr=config.get("lr", 1e-4),
                                  weight_decay=config.get("weight_decay", .01))
    adversarial, discriminator_optimizer = None, None
    if config.get("adversarial"):
        from alpharae.losses.adversarial import PatchAdversarialLoss
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 41)
            adversarial = PatchAdversarialLoss(**config["adversarial"]).to(device)
        discriminator_optimizer = torch.optim.AdamW(adversarial.discriminator.parameters(),
            lr=adversarial.discriminator_lr, betas=(.5,.9), weight_decay=0.)
        train_loss_keys += ("reconstruction_total", "gan_raw", "gan_adaptive_weight", "gan_weighted", "discriminator_loss")
    epochs = config.get("epochs", 20)
    scheduler = build_lr_scheduler(optimizer, config, epochs * batches_per_epoch)
    lpips_model = None
    if loss_config["perceptual"] or config.get("fidelity_loss", {}).get("perceptual_weight", 0):
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    objective = ReconstructionLoss(**loss_config, lpips_model=lpips_model).to(device)
    train_loss_keys += objective.extra_metric_keys
    fidelity = None
    if config.get("fidelity_loss"):
        from alpharae.losses.fidelity import RGBAFidelityLoss
        fidelity=RGBAFidelityLoss(**config["fidelity_loss"],lpips_model=lpips_model).to(device)
        train_loss_keys += fidelity.metric_keys
    official = None
    if config.get("live_official_validation"):
        from alpharae.evaluation.live_official import LiveOfficialMetrics
        official=LiveOfficialMetrics(device, config.get("torch_weights"))
    bf16 = device.type == "cuda" and config.get("bf16", True)
    validation_bf16 = bf16 and config.get("validation_precision", "bf16") != "fp32"
    alpha_structure = config.get("validate_alpha_structure", False)
    max_steps = min(config.get("steps", epochs * batches_per_epoch), epochs * batches_per_epoch)
    epoch, batch_cursor, step, images_seen, elapsed_before = 0, 0, 0, 0, 0.
    input_kind_counts_total = {kind: 0 for kind in INPUT_KINDS}
    if state is not None:
        if state["sampler"]["dataset_size"] != len(train_data):
            raise ValueError("Resume requires the same train split size")
        optimizer.load_state_dict(state["optimizer"])
        if adversarial is not None:
            adversarial.load_state_dict(state["adversarial"])
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        epoch, batch_cursor = state["sampler"]["epoch"], state["sampler"]["batch_cursor"]
        if teacher_data is not None:
            teacher_state = state["teacher_sampler"]
            if teacher_state["dataset_size"] != len(teacher_data):
                raise ValueError("Resume requires the same latent target dataset size")
            teacher_epoch, teacher_cursor = teacher_state["epoch"], teacher_state["batch_cursor"]
        step, images_seen, elapsed_before = state["step"], state["images_seen"], state["elapsed_seconds"]
        input_kind_counts_total.update(state.get("input_kind_counts_total", {}))
        restore_rank_rng(state["rng_by_rank"][rank], device)
    else:
        seed_rank(seed + rank, device, config.get("cpu_threads", 4))
    started = time.monotonic()

    def timing():
        elapsed = elapsed_before + time.monotonic() - started
        return {"elapsed_seconds": elapsed,
                "gpu_hours": world_size * elapsed / 3600 if device.type == "cuda" else 0.}

    metadata = {"resolution": resolution, "world_size": world_size, "batch_size_per_rank": batch_size,
                "global_batch_size": batch_size * world_size, "epochs": epochs,
                "steps_per_epoch": batches_per_epoch, "train_assets": len(train_data),
                "val_assets": len(val_data), "latent_tokens": model.num_tokens,
                "latent_dim": model.latent_dim, "bf16": bf16,
                "samples_per_epoch": batches_per_epoch * batch_size * world_size,
                "dropped_samples_per_epoch": len(train_data) - batches_per_epoch * batch_size * world_size,
                "input_mixing": train_data.input_mixing_metadata(),
                "checkpoint": str(output / "last.pt")}
    last_opaque = {}

    def publish(**values):
        if rank == 0 and config.get("live_status"):
            live = {**metadata, "status": "running", "step": step, "total_steps": max_steps,
                    "images_seen": images_seen, "updated": time.time(), **timing(), **values}
            temporary = output / "status.tmp"
            temporary.write_text(json.dumps(live, allow_nan=False) + "\n")
            temporary.replace(output / "status.json")

    def validation_progress(cohort):
        return lambda completed, total: publish(phase="validating", validation_progress={
            "cohort": cohort, "completed": completed, "total": total})

    def previews(data, folder):
        if rank == 0:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=validation_bf16):
                save_validation_samples(model, data, folder, device, step,
                                        config.get("validation_samples", 4))
    if teacher_data is not None:
        metadata["latent_targets"] = {
            "index": teacher_config["index"], "assets": len(teacher_data),
            "weight": teacher_config.get("weight", 5.0),
            "batch_size_per_rank": batch_size, "steps_per_epoch": teacher_batches,
            "provenance": {key: value for key, value in teacher_data.index.items() if key != "records"},
        }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "last.pt").exists() and resume is None:
            raise FileExistsError("Run directory already has a checkpoint; use --resume or a new output")
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(output / "asset_ids.json", {
            "train": [row["asset_id"] for row in train_data.records],
            "val": [row["asset_id"] for row in val_data.records],
            "rgb_replay_train": [f"rgb:{row['id']}" for row in train_data.rgb_records],
            "input_mixing": metadata["input_mixing"]})
        write_json(output / "summary.json", {**metadata, "status": "running", "step": step,
                                              "images_seen": images_seen,
                                              "input_kind_counts_total": input_kind_counts_total, **timing()})
    dist.barrier()
    initial_val = (state["initial_validation"] if state is not None else
                   validate_distributed(model, val_data, objective, batch_size, device, validation_bf16,
                                        alpha_structure, official, validation_progress("rgba")))
    last_val = state.get("last_validation", initial_val) if state is not None else initial_val
    if opaque_data is not None:
        last_opaque = (state.get("last_opaque_validation", {}) if state is not None else
                      validate_distributed(model, opaque_data, objective, batch_size, device,
                          validation_bf16, alpha_structure, official, validation_progress("opaque_rgb")))
    if rank == 0:
        write_json(output / "initial_validation.json", initial_val)
        if opaque_data is not None and state is None:
            write_json(output / "initial_opaque_validation.json", last_opaque)
    if state is None and config.get("initial_previews"):
        previews(val_data, output / "previews" / f"step_{step:07d}")
        if opaque_data is not None:
            previews(opaque_data, output / "opaque_previews" / f"step_{step:07d}")
        dist.barrier()
    del state

    def evaluate():
        nonlocal last_opaque
        values = validate_distributed(model, val_data, objective, batch_size, device, validation_bf16,
                                      alpha_structure, official, validation_progress("rgba"))
        if rank == 0:
            sample_output = (output / "previews" / f"step_{step:07d}"
                             if config.get("model_type") in ("raev2_rgba","raev2_joint_lora","raev2_joint_full") else output)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=validation_bf16):
                save_validation_samples(model, val_data, sample_output, device, step,
                                        config.get("validation_samples", 4))
        dist.barrier()
        if opaque_data is not None:
            last_opaque = validate_distributed(model, opaque_data, objective, batch_size, device,
                validation_bf16, alpha_structure, official, validation_progress("opaque_rgb"))
            previews(opaque_data, output / "opaque_previews" / f"step_{step:07d}")
            dist.barrier()
        return values

    def save(snapshot=False):
        rank_states = [None] * world_size
        dist.all_gather_object(rank_states, rank_rng_state(device))
        if rank == 0:
            checkpoint = {"config": config, "model": model.state_dict(), "response": None,
                          "optimizer": optimizer.state_dict(), "step": step,
                          "scheduler": scheduler.state_dict() if scheduler is not None else None,
                          "sampler": {"epoch": epoch, "batch_cursor": batch_cursor,
                                      "seed": seed, "batches_per_epoch": batches_per_epoch,
                                      "dataset_size": len(train_data)},
                          "rng_by_rank": rank_states, "images_seen": images_seen,
                          "input_kind_counts_total": dict(input_kind_counts_total),
                          "input_mixing": metadata["input_mixing"],
                          "initial_validation": initial_val, "last_validation": last_val,
                          "last_opaque_validation": last_opaque,
                          "resolution": resolution, "world_size": world_size, **timing()}
            if teacher_data is not None:
                checkpoint["teacher_sampler"] = {
                    "epoch": teacher_epoch, "batch_cursor": teacher_cursor,
                    "seed": seed + TEACHER_SEED_OFFSET, "batches_per_epoch": teacher_batches,
                    "dataset_size": len(teacher_data),
                }
                checkpoint["latent_targets"] = metadata["latent_targets"]
            if adversarial is not None:
                checkpoint["adversarial"] = adversarial.state_dict()
                checkpoint["discriminator_optimizer"] = discriminator_optimizer.state_dict()
            atomic_checkpoint(checkpoint, output / "last.pt")
            if snapshot:
                destination = output / f"step_{step:07d}.pt"
                temporary = destination.with_suffix(".pt.tmp")
                shutil.copyfile(output / "last.pt", temporary)
                temporary.replace(destination)
        dist.barrier()

    status = "complete"
    milestones = set(config.get("checkpoint_milestones", [100]))
    log = (output / "metrics.jsonl").open("a") if rank == 0 else None
    teacher_iterator = None
    if teacher_data is not None:
        teacher_batch_sampler = ResumedBatchSampler(teacher_sampler, batch_size, teacher_cursor)
        teacher_generator = torch.Generator()
        teacher_workers = config.get("num_workers", 2)
        teacher_loader = DataLoader(
            teacher_data, batch_sampler=teacher_batch_sampler,
            num_workers=teacher_workers, pin_memory=device.type == "cuda",
            multiprocessing_context="spawn" if teacher_workers else None,
            persistent_workers=teacher_workers > 0, generator=teacher_generator)
    try:
        model.train()
        while step < max_steps:
            sampler.set_epoch(epoch)
            train_data.epoch=epoch
            batch_sampler = ResumedBatchSampler(sampler, batch_size, batch_cursor)
            # A dedicated generator keeps DataLoader worker seeds out of the model RNG stream.
            loader = DataLoader(train_data, batch_sampler=batch_sampler,
                                num_workers=config.get("num_workers", 2), pin_memory=device.type == "cuda",
                                multiprocessing_context="spawn" if config.get("num_workers", 2) else None,
                                generator=torch.Generator().manual_seed(seed + epoch * world_size + rank))
            for batch in loader:
                update_started = time.monotonic()
                stop = torch.tensor((2 if rank == 0 and (output / "STOP").exists() else
                                    int(rank == 0 and timing()["gpu_hours"] >= config.get("gpu_hours", float("inf")))),
                                    device=device)
                dist.broadcast(stop, src=0)
                if stop.item():
                    status = "paused" if stop.item() == 2 else "budget_exhausted"
                    break
                rgba = batch["rgba"].to(device, non_blocking=True)
                mask = batch["valid_mask"].to(device, non_blocking=True)
                if teacher_data is not None:
                    if teacher_iterator is None:
                        teacher_sampler.set_epoch(teacher_epoch)
                        teacher_batch_sampler.start_batch = teacher_cursor
                        teacher_generator.manual_seed(
                            seed + TEACHER_SEED_OFFSET + teacher_epoch * world_size + rank)
                        teacher_iterator = iter(teacher_loader)
                    teacher_batch = next(teacher_iterator)
                    teacher_rgba = teacher_batch["rgba"].to(device, non_blocking=True)
                    teacher_target = teacher_batch["target_latent"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if adversarial is not None:
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    # Match DDP buffer semantics while the alternating critic uses explicit gradient reduction.
                    for buffer in adversarial.discriminator.buffers():
                        dist.broadcast(buffer, src=0)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                    if teacher_data is None:
                        prediction, latent = ddp(rgba)
                    else:
                        prediction, latent = ddp(torch.cat((rgba, teacher_rgba)), decode_count=len(rgba))
                if fidelity is None:
                    values = objective(prediction.float(), rgba, [], mask)
                else:
                    with torch.no_grad():
                        values = objective(prediction.detach().float(), rgba, [], mask)
                    values.update(fidelity(prediction.float(),rgba,mask))
                if teacher_data is not None:
                    values["reconstruction_total"] = values["total"]
                    values["latent_target_mse"] = latent_target_mse(latent[len(rgba):], teacher_target)
                    values["latent_target_weighted"] = (teacher_config.get("weight", 5.0)
                                                        * values["latent_target_mse"])
                    values["total"] = values["total"] + values["latent_target_weighted"]
                    latent = latent[:len(rgba)]
                if adversarial is not None:
                    values["reconstruction_total"] = values["total"]
                    values.update(adversarial.generator_loss(prediction.float(), values["total"],
                                  model.alpha_readout.output.weight, step + 1))
                    values["total"] = values["total"] + values["gan_weighted"]
                    values["discriminator_loss"] = values["total"].new_zeros(())
                finite = torch.isfinite(values["total"]).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f"Nonfinite objective at step {step + 1}")
                values["total"].backward()
                if adversarial is not None and adversarial.active(step + 1):
                    critic = adversarial.discriminator_loss(rgba, prediction.float())
                    critic["discriminator_loss"].backward()
                    for parameter in adversarial.discriminator.parameters():
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(world_size)
                    torch.nn.utils.clip_grad_norm_(adversarial.discriminator.parameters(),
                                                   config.get("grad_clip",1.), error_if_nonfinite=True)
                    discriminator_optimizer.step()
                    values["discriminator_loss"] = critic["discriminator_loss"]
                gradient = torch.nn.utils.clip_grad_norm_(parameters, config.get("grad_clip", 1.),
                                                         error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                images_seen += len(rgba) * world_size
                batch_cursor += 1
                if teacher_data is not None:
                    teacher_cursor += 1
                    if teacher_cursor == teacher_batches:
                        teacher_epoch, teacher_cursor = teacher_epoch + 1, 0
                        teacher_iterator = None
                if batch_cursor == batches_per_epoch:
                    epoch, batch_cursor = epoch + 1, 0
                totals = torch.stack([values[key].detach() for key in train_loss_keys])
                local_kind_counts = totals.new_tensor([batch["input_kind"].count(kind) for kind in INPUT_KINDS])
                totals = torch.cat((totals, local_kind_counts))
                dist.all_reduce(totals)
                input_kind_counts = {kind: int(value) for kind, value in zip(INPUT_KINDS, totals[-len(INPUT_KINDS):].tolist())}
                for kind, count in input_kind_counts.items():
                    input_kind_counts_total[kind] += count
                latent_rms = latent.detach().float().square().mean().sqrt()
                dist.all_reduce(latent_rms)
                record = {"step": step, "train": {key: float(totals[i] / world_size)
                                                  for i, key in enumerate(train_loss_keys)},
                          "gradient_norm": float(gradient), "epoch": epoch, "batch_cursor": batch_cursor,
                          "learning_rate": optimizer.param_groups[0]["lr"],
                          "learning_rates":{g.get("name",str(i)):g["lr"] for i,g in enumerate(optimizer.param_groups)},
                          "resolution": resolution, "world_size": world_size, "images_seen": images_seen,
                          "input_kind_counts": input_kind_counts,
                          "input_kind_counts_total": dict(input_kind_counts_total),
                          "latent_rms": float(latent_rms / world_size)}
                record["update_seconds"] = time.monotonic() - update_started
                if rank == 0 and config.get("residual_interpolation"):
                    record["interpolation_gains"] = model.alpha_readout.interpolation_gains.detach().float().cpu().tolist()
                if teacher_data is not None:
                    record["teacher_sampler"] = {"epoch": teacher_epoch, "batch_cursor": teacher_cursor}
                if step % config.get("eval_every", 1000) == 0 or step in milestones or step == max_steps:
                    last_val = evaluate()
                    record["validation"] = last_val
                    if opaque_data is not None:
                        record["opaque_validation"] = last_opaque
                    save(snapshot=True)
                record.update(timing())
                if rank == 0:
                    if step % config.get("log_interval", 1) == 0 or "validation" in record or step == 1:
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        print(json.dumps(record), flush=True)
                    publish(phase="sft", train=record["train"], gradient_norm=record["gradient_norm"],
                            learning_rates=record["learning_rates"], update_seconds=record["update_seconds"],
                            last_validation=last_val, last_opaque_validation=last_opaque,
                            peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2**30)
                if step == max_steps:
                    break
            if status in ("budget_exhausted", "paused"):
                break
    finally:
        if log is not None:
            log.close()
    discriminator_sync_error = None
    if adversarial is not None:
        difference = torch.zeros((), device=device)
        for parameter in adversarial.discriminator.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            difference = torch.maximum(difference, (parameter-reference).abs().max())
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        discriminator_sync_error = float(difference)
        if discriminator_sync_error != 0:
            raise RuntimeError(f"Discriminator parameters diverged across ranks: {discriminator_sync_error}")
    save()
    summary = {**metadata, "status": status, "step": step, "epoch": epoch,
               "batch_cursor": batch_cursor, "images_seen": images_seen,
               "input_kind_counts_total": input_kind_counts_total,
               "initial_validation": initial_val, "last_validation": last_val,
               "last_opaque_validation": last_opaque,
               "trainable_main_parameters": sum(parameter.numel() for parameter in parameters),
               "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
               "auxiliary_parameters": 0, "perceptual_enabled": lpips_model is not None,
               "validation_role": ("official JPEG PSNR/SSIM/LPIPS plus optimization diagnostics; rFID/AES require separate evaluation"
                                   if official is not None else "optimization diagnostic; official metrics require separate evaluation"),
               "validation_precision": "bf16" if validation_bf16 else "fp32",
               "adversarial_parameters": sum(p.numel() for p in adversarial.parameters()) if adversarial is not None else 0,
               "discriminator_parameter_sync_max_error": discriminator_sync_error,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
               **timing()}
    if teacher_data is not None:
        summary["teacher_sampler"] = {"epoch": teacher_epoch, "batch_cursor": teacher_cursor}
    if rank == 0:
        write_json(output / "summary.json", summary)
        publish(status=status, phase="finished", last_validation=last_val,
                last_opaque_validation=last_opaque, checkpoint_step=step)
    dist.barrier()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(read_config(args.config), args.output, args.resume)


if __name__ == "__main__":
    main()
