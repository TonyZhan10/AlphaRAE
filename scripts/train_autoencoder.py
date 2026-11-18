import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from alpharae.data.backgrounds import BackgroundSampler
from alpharae.data.rgba import RGBADataset, composite, linear_to_srgb
from alpharae.evaluation.reconstruction import _lpips_crop, save_comparison
from alpharae.losses.reconstruction import ReconstructionLoss, pixel_response_loss
from alpharae.training import (BatchStream, build_system, read_config, restore_rng, rng_state,
                               seed_everything, write_json)


def collect_batch(dataset, indices, device):
    items = [dataset[i] for i in indices]
    return (torch.stack([x["rgba"] for x in items]).to(device),
            torch.stack([x["valid_mask"] for x in items]).to(device))


@torch.no_grad()
def prepare_observations(model, dataset, batch_size, device):
    black, white = [], []
    for start in range(0, len(dataset), batch_size):
        rgba, _ = collect_batch(dataset, range(start, min(start + batch_size, len(dataset))), device)
        b, w = model.observations(rgba)
        black.append(b.cpu())
        white.append(w.cpu())
    return torch.cat(black), torch.cat(white)


def training_observations(model, dataset, batch_size, device, config):
    cache_path = config.get("feature_cache")
    identity = {"encoder": config["encoder"], "model_type": config.get("model_type"),
                "asset_ids": [dataset[i]["asset_id"] for i in range(len(dataset))],
                "manifest": str(Path(config["manifest"]).resolve()), "split": "train",
                "preprocessing": "linear premultiplied resize/pad -> sRGB RGB encoder"}
    if config.get("base_checkpoint"):
        identity["base_checkpoint"] = str(Path(config["base_checkpoint"]).resolve())
    if cache_path and Path(cache_path).exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["identity"] != identity:
            raise ValueError("Training feature cache differs from the configured encoder, manifest or asset order")
        return cached["black"], cached["white"]
    observations = prepare_observations(model, dataset, batch_size, device)
    if cache_path:
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": identity, "black": observations[0], "white": observations[1]}, destination)
    return observations


def selection_policy(config):
    selection = config.get("selection", "proxy")
    if selection == "proxy":
        return {"name": selection, "score": "val alpha_l1 + premult_l1 + composition_mse; screening only",
                "alpha_max": None}
    if selection != "roi_lpips":
        raise ValueError(f"Unknown checkpoint selection: {selection}")
    if config.get("selection_alpha_max") is None:
        raise ValueError("roi_lpips selection requires a fixed selection_alpha_max")
    alpha_max = float(config["selection_alpha_max"])
    if not math.isfinite(alpha_max) or alpha_max < 0:
        raise ValueError("selection_alpha_max must be finite and nonnegative")
    return {"name": selection, "score": "val ROI LPIPS; mean of six backgrounds per asset, then mean assets",
            "alpha_max": alpha_max, "alpha_rule": "val alpha_l1 <= alpha_max",
            "backgrounds": ["black", "white", "color", "gradient", "texture", "natural"],
            "background_split": "val", "background_seed": 12345,
            "roi": "GT alpha > 0.01 intersect valid mask; existing evaluation bbox crop"}


def checkpoint_selection(validation, policy, best):
    if policy["name"] == "proxy":
        score = validation["alpha_l1"] + validation["premult_l1"] + validation["composition_mse"]
        eligible = True
    else:
        score = validation["roi_lpips"]
        eligible = validation["alpha_l1"] <= policy["alpha_max"]
    return score, eligible and math.isfinite(score) and score < best


@torch.no_grad()
def save_validation_samples(model, dataset, output, device, step, count):
    was_training = model.training
    model.eval()
    for index in range(min(count, len(dataset))):
        rgba, _ = collect_batch(dataset, [index], device)
        prediction, _ = model(rgba)
        save_comparison(rgba, {f"recon step {step}": prediction}, {},
                        Path(output) / "evaluation" / f"sample_{index:03d}.png",
                        [dataset[index]["asset_id"]])
    model.train(was_training)


@torch.no_grad()
def validate(model, dataset, loss, batch_size, device, seed, natural_manifest=None, selection="proxy"):
    if selection == "roi_lpips" and (loss.lpips_model is None or natural_manifest is None):
        raise ValueError("roi_lpips selection requires LPIPS and natural validation backgrounds")
    was_training = model.training
    model.eval()
    sampler = BackgroundSampler(model.encoder.resolution, "val", seed, natural_manifest)
    roi_sampler = (BackgroundSampler(model.encoder.resolution, "val", seed, natural_manifest)
                   if selection == "roi_lpips" else None)
    totals, count = {}, 0
    roi_values = []
    for start in range(0, len(dataset), batch_size):
        rgba, mask = collect_batch(dataset, range(start, min(start + batch_size, len(dataset))), device)
        pred, _ = model(rgba)
        values = loss(pred, rgba, [sampler.sample(len(rgba), device)], mask)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + float(value) * len(rgba)
        if roi_sampler is not None:
            # Asset-major sampling keeps the fixed validation queries independent
            # of batch size. Only validation-split natural backgrounds are loaded.
            for index in range(len(rgba)):
                roi = (rgba[index, 3:] > .01) & mask[index].bool()
                backgrounds = [torch.zeros_like(rgba[index, :3]), torch.ones_like(rgba[index, :3])]
                backgrounds.extend(roi_sampler.sample(1, device, family=family)[0]
                                   for family in ("color", "gradient", "texture", "natural"))
                asset_values = []
                for background in backgrounds:
                    prediction = linear_to_srgb(composite(pred[index], background).clamp(0, 1))
                    reference = linear_to_srgb(composite(rgba[index], background).clamp(0, 1))
                    value = _lpips_crop(loss.lpips_model, prediction, reference, roi)
                    if value is not None:
                        asset_values.append(value)
                if asset_values:
                    roi_values.append(sum(asset_values) / len(asset_values))
        count += len(rgba)
    model.train(was_training)
    result = {key: value / count for key, value in totals.items()}
    if roi_sampler is not None:
        if not roi_values:
            raise ValueError("Validation assets have no foreground ROI for checkpoint selection")
        result.update(roi_lpips=sum(roi_values) / len(roi_values), roi_lpips_assets=len(roi_values))
    return result


def train(config, output, resume=None, dry_run=False):
    policy = selection_policy(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Run directory already has a checkpoint; use --resume or a new output")
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this interpreter; use the verified Ray environment")
    seed_everything(config.get("seed", 0), config.get("cpu_threads", 4))
    model, response = build_system(config, device)
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        changes = [key for key in set(config) | set(state["config"])
                   if key not in ("steps", "gpu_hours", "device", "cpu_threads")
                   and config.get(key) != state["config"].get(key)]
        if changes:
            raise ValueError(f"Resume configuration changed: {changes}")
        model.load_state_dict(state["model"])
        if response is not None:
            response.load_state_dict(state["response"])
    if config.get("base_checkpoint") and not resume:
        initialization = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=False)
        model.base.load_state_dict(initialization["model"], strict=True)
        del initialization
    if config.get("init_checkpoint") and not resume:
        initialization = torch.load(config["init_checkpoint"], map_location="cpu", weights_only=False)
        if config.get("model_type") == "pretrained_rgba":
            from alpharae.models.pretrained_rgba import adapt_alpha_state
            adapt_alpha_state(model, initialization["model"])
        else:
            model.load_state_dict(initialization["model"], strict=True)
        del initialization
    resolution = model.encoder.resolution
    train_data = RGBADataset(config["manifest"], resolution, "train")
    val_data = RGBADataset(config["manifest"], resolution, "val")
    if not len(train_data) or not len(val_data):
        raise ValueError("Both train and validation splits must contain assets")
    if config.get("preload", True):
        train_data, val_data = list(train_data), list(val_data)
    batch_size = config.get("batch_size", 16)
    effective_batch = config.get("effective_batch_size", batch_size)
    if effective_batch % batch_size:
        raise ValueError("effective_batch_size must be a multiple of batch_size")
    accumulation = effective_batch // batch_size
    stream = BatchStream(len(train_data), batch_size, config.get("seed", 0))
    backgrounds = BackgroundSampler(resolution, "train", config.get("seed", 0) + 100,
                                    config.get("background_manifest"))
    lpips_model = None
    if config.get("loss", {}).get("perceptual", .1) or policy["name"] == "roi_lpips":
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    objective = ReconstructionLoss(**config.get("loss", {}), lpips_model=lpips_model).to(device)
    main_parameters = [p for p in model.parameters() if p.requires_grad]
    auxiliary_parameters = list(response.parameters()) if response is not None else []
    parameters = main_parameters + auxiliary_parameters
    optimizer = torch.optim.AdamW(parameters, lr=config.get("lr", 1e-4),
                                  weight_decay=config.get("weight_decay", .01))
    adversarial, discriminator_optimizer, last_layer = None, None, None
    if config.get("adversarial"):
        from alpharae.losses.adversarial import PatchAdversarialLoss
        # Critic initialization must not change the paired generator RNG stream.
        with torch.random.fork_rng(devices=[]):
            adversarial = PatchAdversarialLoss(**config["adversarial"]).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            adversarial.discriminator.parameters(),
            lr=config["adversarial"].get("discriminator_lr", 1e-4), betas=(.5, .9), weight_decay=0.)
        decoder_output = (model.decoder.decoder_pred if config.get("model_type") == "pretrained_rgba"
                          else model.decoder.output)
        last_layer = decoder_output.weight
    start_step, elapsed_before, best = 0, 0., float("inf")
    best_eligible, best_validation = False, None
    started = time.monotonic()
    cached_observations = training_observations(model, train_data, batch_size, device, config)
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        if adversarial is not None:
            adversarial.load_state_dict(state["adversarial"])
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        stream.load_state_dict(state["sampler"])
        backgrounds.load_state_dict(state["backgrounds"])
        start_step, elapsed_before, best = state["step"], state["elapsed_seconds"], state["best"]
        best_eligible = state.get("best_eligible", math.isfinite(best))
        best_validation = state.get("best_validation")
        restore_rng(state["rng"])
    else:
        model.fit_feature_stats(*cached_observations)
        if response is not None:
            targets = []
            stats_sampler = BackgroundSampler(resolution, "train", config.get("seed", 0) + 900,
                                               config.get("background_manifest"))
            for start in range(0, min(len(train_data), config.get("stats_assets", 256)), batch_size):
                rgba, _ = collect_batch(train_data, range(start, min(start + batch_size, len(train_data))), device)
                bi, bj = stats_sampler.sample(len(rgba), device), stats_sampler.sample(len(rgba), device)
                targets.append(response.targets(model.encoder, rgba, bi, bj)[0])
            response.fit_response_stats(torch.cat(targets))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "asset_ids.json").write_text(json.dumps({
        "train": [train_data[i]["asset_id"] for i in range(len(train_data))],
        "val": [val_data[i]["asset_id"] for i in range(len(val_data))]}))
    initial_val = validate(model, val_data, objective, batch_size, device, 12345,
                           config.get("background_manifest"), selection=policy["name"])
    save_validation_samples(model, val_data, output, device, start_step, config.get("validation_samples", 0))
    resume_validation = dict(initial_val) if resume else None
    if not resume:
        write_json(output / "initial_validation.json", initial_val)
    else:
        if "initial_validation" in state:
            initial_val = state["initial_validation"]
        else:
            initial_val = json.loads((Path(resume).parent / "initial_validation.json").read_text())
        if not (output / "initial_validation.json").exists():
            write_json(output / "initial_validation.json", initial_val)
    last_val = resume_validation if resume else initial_val
    max_steps = min(config.get("steps", 5000), start_step + 1) if dry_run else config.get("steps", 5000)
    model.train()
    step = start_step
    status = "complete"

    def save(path):
        elapsed = elapsed_before + time.monotonic() - started
        state = {"config": config, "model": model.state_dict(),
                 "response": response.state_dict() if response is not None else None,
                 "optimizer": optimizer.state_dict(), "step": step, "best": best,
                 "best_eligible": best_eligible, "best_validation": best_validation,
                 "initial_validation": initial_val,
                 "sampler": stream.state_dict(), "backgrounds": backgrounds.state_dict(),
                 "rng": rng_state(), "elapsed_seconds": elapsed,
                 "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.}
        if adversarial is not None:
            state["adversarial"] = adversarial.state_dict()
            state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        torch.save(state, path)

    if not resume and policy["name"] == "roi_lpips":
        score, improves = checkpoint_selection(initial_val, policy, best)
        if improves:
            best, best_eligible, best_validation = score, True, dict(initial_val)
            save(output / "best.pt")

    with (output / "metrics.jsonl").open("a") as log:
        for step in range(start_step + 1, max_steps + 1):
            if device.type == "cuda" and (elapsed_before + time.monotonic() - started) / 3600 >= config.get("gpu_hours", 8):
                step -= 1
                status = "budget_exhausted"
                break
            optimizer.zero_grad(set_to_none=True)
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            totals = {}
            for _ in range(accumulation):
                indices = stream.next()
                rgba, valid_mask = collect_batch(train_data, indices, device)
                obs = tuple(x[indices].to(device) for x in cached_observations)
                bi, bj = backgrounds.sample(len(rgba), device), backgrounds.sample(len(rgba), device)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and config.get("bf16", True) and step > 100):
                    pred, z = model(rgba, obs)
                values = objective(pred.float(), rgba, [bi, bj], valid_mask)
                if response is not None:
                    target, gi, gj = response.targets(model.encoder, rgba, bi, bj)
                    values["response"] = response.loss(z.float(), target, gi, gj, rgba,
                                                        model.encoder.patch_size,
                                                        detach_latent=config["variant"] == "C1",
                                                        background_only=config["variant"] == "C3")
                    values["total"] = values["total"] + config.get("lambda_response", .1) * values["response"]
                if config["variant"] == "C2":
                    values["pixel_response"] = pixel_response_loss(pred.float(), rgba, bi, bj, valid_mask)
                    values["total"] = values["total"] + config.get("lambda_response", .1) * values["pixel_response"]
                if adversarial is not None:
                    values["reconstruction_total"] = values["total"]
                    values.update(adversarial.generator_loss(pred.float(), values["total"], last_layer, step))
                    values["total"] = values["total"] + values["gan_weighted"]
                if not torch.isfinite(values["total"]):
                    raise FloatingPointError(f"Nonfinite objective at step {step}")
                (values["total"] / accumulation).backward()
                if adversarial is not None and adversarial.active(step):
                    discriminator_values = adversarial.discriminator_loss(rgba, pred.float())
                    if not torch.isfinite(discriminator_values["discriminator_loss"]):
                        raise FloatingPointError(f"Nonfinite discriminator objective at step {step}")
                    (discriminator_values["discriminator_loss"] / accumulation).backward()
                    values.update(discriminator_values)
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) / accumulation
            gradient = torch.nn.utils.clip_grad_norm_(main_parameters, config.get("grad_clip", 1.), error_if_nonfinite=True)
            if auxiliary_parameters:
                torch.nn.utils.clip_grad_norm_(auxiliary_parameters, config.get("grad_clip", 1.), error_if_nonfinite=True)
            optimizer.step()
            discriminator_gradient = None
            if adversarial is not None and adversarial.active(step):
                discriminator_gradient = torch.nn.utils.clip_grad_norm_(
                    adversarial.discriminator.parameters(), config.get("grad_clip", 1.), error_if_nonfinite=True)
                discriminator_optimizer.step()
            record = {"step": step, "train": totals, "gradient_norm": float(gradient),
                      "elapsed_seconds": elapsed_before + time.monotonic() - started}
            if discriminator_gradient is not None:
                record["discriminator_gradient_norm"] = float(discriminator_gradient)
            if step % config.get("eval_every", 100) == 0 or step == max_steps:
                last_val = validate(model, val_data, objective, batch_size, device, 12345,
                                    config.get("background_manifest"), selection=policy["name"])
                save_validation_samples(model, val_data, output, device, step, config.get("validation_samples", 0))
                record["validation"] = last_val
                score, improves = checkpoint_selection(last_val, policy, best)
                if improves:
                    best, best_eligible, best_validation = score, True, dict(last_val)
                    save(output / "best.pt")
                save(output / "last.pt")
                print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    save(output / "last.pt")
    elapsed = elapsed_before + time.monotonic() - started
    summary = {"status": status, "step": step, "initial_validation": initial_val,
               "resume_validation": resume_validation,
               "last_validation": last_val, "elapsed_seconds": elapsed,
               "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
               "trainable_main_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
               "auxiliary_parameters": sum(p.numel() for p in response.parameters()) if response is not None else 0,
               "train_assets": len(train_data), "val_assets": len(val_data),
               "backbone": config["encoder"], "latent_tokens": model.num_tokens,
               "latent_dim": model.latent_dim, "perceptual_enabled": lpips_model is not None,
               "selection_score": policy["score"], "selection": policy["name"],
               "selection_alpha_max": policy["alpha_max"], "selection_protocol": policy,
               "best_eligible": best_eligible, "best_validation": best_validation,
               "best_score": best if best_eligible else None,
               "checkpoint": str(output / "last.pt")}
    if adversarial is not None:
        summary["adversarial_parameters"] = sum(p.numel() for p in adversarial.parameters())
        summary["adversarial_config"] = config["adversarial"]
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(read_config(args.config), args.output, args.resume, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
