#!/usr/bin/env python3
"""Train one controlled alpha experiment exactly as specified by the guide."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    atomic_json,
    environment_report,
    load_config,
    resolve_paper_path,
    seed_everything,
    seed_worker,
    sha256_file,
)
from src.dataset import DistillationDataset
from src.losses import compute_distillation_loss
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.student import LiteDenoiseNet


def alpha_name(alpha: float) -> str:
    return f"alpha_{alpha:.1f}".replace(".", "p")


def average_sums(sums: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / count for name, value in sums.items()}


def train_epoch(
    model: LiteDenoiseNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    alpha: float,
    clip_norm: float,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    finite_gradient_batches = 0
    skipped_optimizer_steps = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(noisy)
            terms = compute_distillation_loss(output, teacher, clean, alpha=alpha)
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before_step:
            skipped_optimizer_steps += 1

        batch_size = noisy.shape[0]
        samples += batch_size
        sums["loss"] += float(terms.total.detach()) * batch_size
        sums["gt_mse"] += float(terms.gt_mse.detach()) * batch_size
        sums["kd_mse"] += float(terms.kd_mse.detach()) * batch_size
        sums["gt_l1"] += float(terms.gt_l1.detach()) * batch_size
        if math.isfinite(float(gradient_norm)):
            sums["gradient_norm_before_clip"] += float(gradient_norm)
            finite_gradient_batches += 1
    if samples == 0:
        raise RuntimeError("Training epoch processed no samples")
    result = average_sums(sums, samples)
    result["gradient_norm_before_clip"] = (
        sums["gradient_norm_before_clip"] / finite_gradient_batches
        if finite_gradient_batches
        else 0.0
    )
    result["finite_gradient_batches"] = finite_gradient_batches
    result["skipped_optimizer_steps"] = skipped_optimizer_steps
    result["samples"] = samples
    result["seconds"] = time.perf_counter() - start
    return result


def validate(
    model: LiteDenoiseNet,
    loader: DataLoader,
    device: torch.device,
    border: int,
    window_size: int,
    sigma: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = defaultdict(list)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="validation", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            output = model(noisy)
            tensors = {
                "student_psnr": psnr_per_image(output, clean, border=border),
                "student_ssim": gaussian_ssim_per_image(
                    output, clean, border=border, window_size=window_size, sigma=sigma
                ),
                "noisy_psnr": psnr_per_image(noisy, clean, border=border),
                "noisy_ssim": gaussian_ssim_per_image(
                    noisy, clean, border=border, window_size=window_size, sigma=sigma
                ),
                "teacher_psnr": psnr_per_image(teacher, clean, border=border),
                "teacher_ssim": gaussian_ssim_per_image(
                    teacher, clean, border=border, window_size=window_size, sigma=sigma
                ),
            }
            for name, tensor in tensors.items():
                values[name].extend(tensor.float().cpu().tolist())
    if not values:
        raise RuntimeError("Validation processed no samples")
    result = {name: float(np.mean(series)) for name, series in values.items()}
    result["samples"] = len(values["student_psnr"])
    result["seconds"] = time.perf_counter() - start
    return result


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_fingerprint(
    config: dict[str, Any],
    manifest_path: Path,
    alpha: float,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> tuple[str, dict[str, Any]]:
    source_root = Path(__file__).parents[1] / "src"
    scripts_root = Path(__file__).parent
    payload = {
        "alpha": alpha,
        "seed": int(config["project"]["seed"]),
        "preprocessing_version": config["project"]["preprocessing_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "model": config["model"],
        "training": {**config["training"], "epochs": epochs},
        "metrics": config["metrics"],
        "datasets": config["data"]["datasets"],
        "diagnostic_limits": {
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
        },
        "source_sha256": {
            name: sha256_file(source_root / name)
            for name in ("student.py", "losses.py", "dataset.py", "metrics.py")
        },
        "training_script_sha256": {
            name: sha256_file(scripts_root / name) for name in ("train.py", "common.py")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument("--alpha", type=float, required=True, choices=(0.0, 0.7, 0.9))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Diagnostic override; full runs use config value.")
    parser.add_argument("--max-train-batches", type=int, help="Diagnostic smoke-test limit.")
    parser.add_argument("--max-val-batches", type=int, help="Diagnostic smoke-test limit.")
    args = parser.parse_args()

    config = load_config(args.config)
    alpha = float(args.alpha)
    if alpha not in {float(value) for value in config["training"]["alphas"]}:
        raise ValueError(f"alpha {alpha} is not in the controlled experiment matrix")
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    runs_root = resolve_paper_path(config["outputs"]["runs"])
    run_dir = (args.output_dir.resolve() if args.output_dir else runs_root / alpha_name(alpha))
    if args.resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Fresh run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    fingerprint, fingerprint_payload = run_fingerprint(
        config,
        manifest_path,
        alpha,
        epochs,
        args.max_train_batches,
        args.max_val_batches,
    )
    batch_size = int(training["batch_size"])
    workers = int(training["workers"])
    train_dataset = DistillationDataset(
        manifest_path,
        root=cache_root,
        split="train",
        augment=True,
        augmentation_seed=seed,
    )
    validation_dataset = DistillationDataset(
        manifest_path,
        root=cache_root,
        split="validation",
        augment=False,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteDenoiseNet().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["model"]["expected_parameters"]):
        raise RuntimeError(f"Unexpected model parameter count: {parameter_count}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training["minimum_learning_rate"]),
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 1
    best_psnr = float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        if float(checkpoint["alpha"]) != alpha:
            raise ValueError("Resume checkpoint alpha does not match this run")
        if checkpoint.get("run_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint does not match this dataset, code, or run configuration")
        if int(checkpoint["epoch"]) >= epochs:
            raise ValueError("Resume checkpoint already reached the configured epoch budget")
        resume_history_path = run_dir / "history.jsonl"
        if resume_history_path.is_file():
            history_records = [
                json.loads(line)
                for line in resume_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            completed_epochs = [int(record["epoch"]) for record in history_records]
            expected_epochs = list(range(1, len(completed_epochs) + 1))
            if completed_epochs != expected_epochs:
                raise ValueError("Resume history must be a unique contiguous prefix starting at epoch 1")
            checkpoint_epoch = int(checkpoint["epoch"])
            if completed_epochs and completed_epochs[-1] < checkpoint_epoch:
                raise ValueError("Resume checkpoint is ahead of history.jsonl")
            if completed_epochs and completed_epochs[-1] > checkpoint_epoch:
                retained = [record for record in history_records if int(record["epoch"]) <= checkpoint_epoch]
                temporary_history = resume_history_path.with_suffix(".jsonl.tmp")
                temporary_history.write_text(
                    "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
                    encoding="utf-8",
                )
                temporary_history.replace(resume_history_path)
        elif int(checkpoint["epoch"]) > 0:
            raise FileNotFoundError("Resume history.jsonl is missing")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint["best_psnr"])

    resolved = {
        "config": config,
        "alpha": alpha,
        "loss_coefficients": {
            "clean_mse": 1000.0 * (1.0 - alpha),
            "teacher_mse": 1000.0 * alpha,
            "clean_l1": 50.0,
        },
        "epochs_effective": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "diagnostic_limits": {
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "environment": environment_report(config, manifest_path),
        "run_fingerprint": fingerprint,
        "run_fingerprint_payload": fingerprint_payload,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    run_path = run_dir / "run.json"
    if args.resume is None:
        atomic_json(run_path, resolved)
    elif not run_path.is_file():
        raise FileNotFoundError(f"Resume run metadata is missing: {run_path}")
    else:
        existing_run = json.loads(run_path.read_text(encoding="utf-8"))
        if existing_run.get("run_fingerprint") != fingerprint:
            raise ValueError("Resume run metadata does not match the checkpoint fingerprint")
    history_path = run_dir / "history.jsonl"
    status_path = run_dir / "status.json"
    atomic_json(status_path, {"state": "running", "epoch": start_epoch - 1, "epochs": epochs, "alpha": alpha})

    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            train_dataset.set_epoch(epoch)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            epoch_start = time.perf_counter()
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                alpha,
                float(training["gradient_clip_norm"]),
                amp_enabled,
                args.max_train_batches,
            )
            validation_metrics = validate(
                model,
                validation_loader,
                device,
                border,
                window_size,
                sigma,
                args.max_val_batches,
            )
            scheduler.step()
            improved = validation_metrics["student_psnr"] > best_psnr
            if improved:
                best_psnr = validation_metrics["student_psnr"]
            record = {
                "epoch": epoch,
                "epochs": epochs,
                "alpha": alpha,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_psnr": best_psnr,
                "improved": improved,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, sort_keys=True) + "\n")
            state = {
                "epoch": epoch,
                "alpha": alpha,
                "best_psnr": best_psnr,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(loader_generator),
                "run_fingerprint": fingerprint,
                "config": config,
                "metrics": record,
            }
            atomic_checkpoint(run_dir / "last.pt", state)
            if improved:
                atomic_checkpoint(run_dir / "best.pt", state)
            if epoch % int(training["checkpoint_interval"]) == 0:
                atomic_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", state)
            atomic_json(status_path, {
                "state": "running" if epoch < epochs else "complete",
                "epoch": epoch,
                "epochs": epochs,
                "alpha": alpha,
                "best_psnr": best_psnr,
                "last_validation": validation_metrics,
                "updated_at": record["timestamp"],
            })
            print(
                f"epoch {epoch:03d}/{epochs} alpha={alpha:.1f} "
                f"loss={train_metrics['loss']:.5f} "
                f"val_psnr={validation_metrics['student_psnr']:.4f} "
                f"val_ssim={validation_metrics['student_ssim']:.6f} "
                f"best={best_psnr:.4f} seconds={record['epoch_seconds']:.1f}",
                flush=True,
            )
    except BaseException as error:
        atomic_json(status_path, {
            "state": "failed",
            "epoch": max(current_epoch, 0),
            "epochs": epochs,
            "alpha": alpha,
            "error": repr(error),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
