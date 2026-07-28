#!/usr/bin/env python3
"""Train the controlled NIND teacher-only/full-reference ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
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
from src.ablation_losses import compute_masked_distillation_loss
from src.dataset import DistillationDataset
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.student import LiteDenoiseNet
from train import atomic_checkpoint, restore_rng, rng_state


def supervision_name(nind_gt_weight: float) -> str:
    if math.isclose(nind_gt_weight, 0.0, abs_tol=1e-12):
        return "nind_teacher_only"
    if math.isclose(nind_gt_weight, 1.0, abs_tol=1e-12):
        return "nind_full_reference"
    raise ValueError(f"Controlled ablation requires NIND GT weight 0 or 1, got {nind_gt_weight}")


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def gt_weights(
    datasets: list[str] | tuple[str, ...],
    nind_gt_weight: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [nind_gt_weight if str(dataset) == "nind" else 1.0 for dataset in datasets],
        dtype=torch.float32,
        device=device,
    )


def train_epoch(
    model: LiteDenoiseNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    alpha: float,
    nind_gt_weight: float,
    clip_norm: float,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    finite_gradient_batches = 0
    skipped_optimizer_steps = 0
    optimizer_steps = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        weight = gt_weights(batch["dataset"], nind_gt_weight, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(noisy)
            terms = compute_masked_distillation_loss(
                output,
                teacher,
                clean,
                weight,
                alpha=alpha,
            )
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before_step:
            skipped_optimizer_steps += 1
        else:
            optimizer_steps += 1

        batch_size = noisy.shape[0]
        samples += batch_size
        for name in ("total", "gt_mse", "kd_mse", "gt_l1", "gt_weight_mean"):
            key = "loss" if name == "total" else name
            sums[key] += float(getattr(terms, name).detach()) * batch_size
        if math.isfinite(float(gradient_norm)):
            sums["gradient_norm_before_clip"] += float(gradient_norm)
            finite_gradient_batches += 1
    if samples == 0:
        raise RuntimeError("Training epoch processed no samples")
    result = {
        name: value / samples
        for name, value in sums.items()
        if name != "gradient_norm_before_clip"
    }
    result["gradient_norm_before_clip"] = (
        sums["gradient_norm_before_clip"] / finite_gradient_batches
        if finite_gradient_batches
        else 0.0
    )
    result["finite_gradient_batches"] = finite_gradient_batches
    result["skipped_optimizer_steps"] = skipped_optimizer_steps
    result["optimizer_steps"] = optimizer_steps
    result["samples"] = samples
    result["seconds"] = time.perf_counter() - start
    return result


def validation_tensors(
    output: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    teacher: torch.Tensor,
    border: int,
    window_size: int,
    sigma: float,
) -> dict[str, torch.Tensor]:
    return {
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
        "student_teacher_psnr": psnr_per_image(output, teacher, border=border),
        "student_teacher_ssim": gaussian_ssim_per_image(
            output, teacher, border=border, window_size=window_size, sigma=sigma
        ),
    }


def validate(
    model: LiteDenoiseNet,
    loader: DataLoader,
    device: torch.device,
    border: int,
    window_size: int,
    sigma: float,
    selection_datasets: set[str],
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    all_values: dict[str, list[float]] = defaultdict(list)
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    selection_values: dict[str, list[float]] = defaultdict(list)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="validation", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            output = model(noisy)
            tensors = validation_tensors(
                output, noisy, clean, teacher, border, window_size, sigma
            )
            datasets = [str(value) for value in batch["dataset"]]
            for name, tensor in tensors.items():
                values = tensor.float().cpu().tolist()
                all_values[name].extend(values)
                for dataset, value in zip(datasets, values, strict=True):
                    by_dataset[dataset][name].append(float(value))
                    if dataset in selection_datasets:
                        selection_values[name].append(float(value))
    if not all_values or not selection_values:
        raise RuntimeError("Validation or checkpoint-selection subset is empty")
    result: dict[str, Any] = {
        name: float(np.mean(values)) for name, values in all_values.items()
    }
    result["selection_student_psnr"] = float(np.mean(selection_values["student_psnr"]))
    result["selection_student_ssim"] = float(np.mean(selection_values["student_ssim"]))
    result["selection_samples"] = len(selection_values["student_psnr"])
    result["selection_datasets"] = sorted(selection_datasets)
    result["by_dataset"] = {
        dataset: {
            **{name: float(np.mean(values)) for name, values in metrics.items()},
            "samples": len(metrics["student_psnr"]),
        }
        for dataset, metrics in sorted(by_dataset.items())
    }
    result["samples"] = len(all_values["student_psnr"])
    result["seconds"] = time.perf_counter() - start
    return result


def run_fingerprint(
    config: dict[str, Any],
    manifest_path: Path,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> tuple[str, dict[str, Any]]:
    paper_root = Path(__file__).parents[1]
    payload = {
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
            relative: sha256_file(paper_root / relative)
            for relative in (
                "src/student.py",
                "src/ablation_losses.py",
                "src/dataset.py",
                "src/metrics.py",
                "scripts/train_ablation.py",
                "scripts/common.py",
                "scripts/train.py",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    alpha = float(training["alpha"])
    nind_gt_weight = float(training["nind_gt_weight"])
    mode = supervision_name(nind_gt_weight)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    runs_root = resolve_paper_path(config["outputs"]["runs"])
    run_dir = args.output_dir.resolve() if args.output_dir else runs_root / mode
    if args.resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Fresh run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    fingerprint, fingerprint_payload = run_fingerprint(
        config, manifest_path, epochs, args.max_train_batches, args.max_val_batches
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
        manifest_path, root=cache_root, split="validation", augment=False
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
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        config["model"]["expected_parameters"]
    ):
        raise RuntimeError("Unexpected model parameter count")
    initial_model_sha256 = model_digest(model)
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
        if checkpoint.get("mode") != mode or checkpoint.get("run_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint does not match this ablation run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint["best_psnr"])

    dataset_counts = Counter(
        (record.split, record.dataset)
        for record in train_dataset.records + validation_dataset.records
    )
    selection_datasets = {str(value) for value in training["selection_datasets"]}
    resolved = {
        "config": config,
        "mode": mode,
        "alpha": alpha,
        "nind_gt_weight": nind_gt_weight,
        "loss_coefficients": {
            "clean_mse": 1000.0 * (1.0 - alpha),
            "teacher_mse": 1000.0 * alpha,
            "clean_l1": 50.0,
        },
        "selection_datasets": sorted(selection_datasets),
        "epochs_effective": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "dataset_counts": {
            f"{split}/{dataset}": count
            for (split, dataset), count in sorted(dataset_counts.items())
        },
        "initial_model_sha256": initial_model_sha256,
        "environment": environment_report(config, manifest_path),
        "run_fingerprint": fingerprint,
        "run_fingerprint_payload": fingerprint_payload,
        "diagnostic_limits": {
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    run_path = run_dir / "run.json"
    if args.resume is None:
        atomic_json(run_path, resolved)
    elif not run_path.is_file() or json.loads(run_path.read_text()).get(
        "run_fingerprint"
    ) != fingerprint:
        raise ValueError("Resume run metadata is missing or incompatible")

    history_path = run_dir / "history.jsonl"
    status_path = run_dir / "status.json"
    atomic_json(
        status_path,
        {"state": "running", "mode": mode, "epoch": start_epoch - 1, "epochs": epochs},
    )
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
                nind_gt_weight,
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
                selection_datasets,
                args.max_val_batches,
            )
            if train_metrics["optimizer_steps"] > 0:
                scheduler.step()
            selection_psnr = float(validation_metrics["selection_student_psnr"])
            improved = selection_psnr > best_psnr
            if improved:
                best_psnr = selection_psnr
            record = {
                "epoch": epoch,
                "epochs": epochs,
                "mode": mode,
                "alpha": alpha,
                "nind_gt_weight": nind_gt_weight,
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
                "mode": mode,
                "alpha": alpha,
                "nind_gt_weight": nind_gt_weight,
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
            atomic_json(
                status_path,
                {
                    "state": "running" if epoch < epochs else "complete",
                    "mode": mode,
                    "epoch": epoch,
                    "epochs": epochs,
                    "best_selection_psnr": best_psnr,
                    "last_validation": validation_metrics,
                    "updated_at": record["timestamp"],
                },
            )
            nind_metrics = validation_metrics["by_dataset"].get("nind", {})
            print(
                f"epoch {epoch:03d}/{epochs} mode={mode} "
                f"loss={train_metrics['loss']:.5f} "
                f"select_psnr={selection_psnr:.4f} "
                f"nind_teacher_psnr={nind_metrics.get('student_teacher_psnr', math.nan):.4f} "
                f"best={best_psnr:.4f} seconds={record['epoch_seconds']:.1f}",
                flush=True,
            )
    except BaseException as error:
        atomic_json(
            status_path,
            {
                "state": "failed",
                "mode": mode,
                "epoch": current_epoch,
                "epochs": epochs,
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
