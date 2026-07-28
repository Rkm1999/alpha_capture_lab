#!/usr/bin/env python3
"""Train the NPU-native LiteDenoise student from scratch."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from student_litedenoise import LiteDenoiseStudent


class MultiscaleResidualHeads(nn.Module):
    """Training-only heads that make each decoder scale explain noise."""

    def __init__(self, base_width: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            "body": nn.Conv2d(base_width * 16, 3, 1),
            "decoder3": nn.Conv2d(base_width * 8, 3, 1),
            "decoder2": nn.Conv2d(base_width * 4, 3, 1),
            "decoder1": nn.Conv2d(base_width * 2, 3, 1),
        })

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: head(features[name]) for name, head in self.heads.items()}


class PatchDataset(Dataset):
    def __init__(self, cache: Path, split: str, augment: bool,
                 manifest_name: str = "manifest.json") -> None:
        records = json.loads((cache / manifest_name).read_text(encoding="utf-8"))
        self.records = [x for x in records if x["split"] == split]
        self.cache, self.augment = cache, augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        values = [torch.from_numpy(np.load(self.cache / record[key]).astype(np.float32)).permute(2, 0, 1)
                  for key in ("input", "teacher", "clean")]
        if self.augment:
            turns = random.randrange(4)
            values = [torch.rot90(value, turns, (1, 2)) for value in values]
            if random.random() < 0.5:
                values = [value.flip(2) for value in values]
        return values[0], values[1], values[2]


def sampling_weights(dataset: PatchDataset, settings: dict) -> tuple[torch.Tensor, dict]:
    counts = Counter(record["dataset"] for record in dataset.records)
    power = float(settings.get("correction_sampling_power", 0.0))
    if not power:
        values = [1.0 / counts[record["dataset"]] for record in dataset.records]
        return torch.tensor(values, dtype=torch.double), {"mode": "dataset-balanced"}
    magnitudes = []
    for record in tqdm(dataset.records, desc="Measuring teacher corrections"):
        source = np.load(dataset.cache / record["input"]).astype(np.float32)
        teacher = np.load(dataset.cache / record["teacher"]).astype(np.float32)
        magnitudes.append(float(np.sqrt(np.mean((teacher - source) ** 2))))
    dataset_means = {
        name: float(np.mean([value for value, record in zip(magnitudes, dataset.records)
                            if record["dataset"] == name]))
        for name in counts
    }
    floor = float(settings.get("correction_sampling_floor", 0.25))
    cap = float(settings.get("correction_sampling_cap", 6.0))
    values = []
    for magnitude, record in zip(magnitudes, dataset.records):
        relative = magnitude / max(dataset_means[record["dataset"]], 1e-8)
        correction_weight = min(cap, max(floor, relative) ** power)
        values.append(correction_weight / counts[record["dataset"]])
    return torch.tensor(values, dtype=torch.double), {
        "mode": "dataset-balanced correction-weighted", "power": power,
        "minimum_correction": min(magnitudes), "mean_correction": float(np.mean(magnitudes)),
        "maximum_correction": max(magnitudes), "dataset_means": dataset_means,
    }


def core(value: torch.Tensor, border: int) -> torch.Tensor:
    return value[:, :, border:-border, border:-border] if border else value


def residual_bands(value: torch.Tensor) -> dict[str, torch.Tensor]:
    low_3 = F.avg_pool2d(value, 3, stride=1, padding=1, count_include_pad=False)
    low_9 = F.avg_pool2d(value, 9, stride=1, padding=4, count_include_pad=False)
    low_33 = F.avg_pool2d(value, 33, stride=1, padding=16, count_include_pad=False)
    return {
        "fine": value - low_3,
        "medium": low_3 - low_9,
        "coarse": low_9 - low_33,
        "very_coarse": low_33,
    }


def vector_similarity(dot: float, student_energy: float, teacher_energy: float) -> float:
    student_norm = math.sqrt(max(student_energy, 1e-12))
    teacher_norm = math.sqrt(max(teacher_energy, 1e-12))
    cosine = dot / max(student_norm * teacher_norm, 1e-12)
    magnitude_ratio = student_norm / max(teacher_norm, 1e-12)
    magnitude_agreement = min(magnitude_ratio, 1.0 / max(magnitude_ratio, 1e-12))
    return cosine * magnitude_agreement


def luminance(value: torch.Tensor) -> torch.Tensor:
    coefficients = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (value * coefficients).sum(dim=1, keepdim=True)


def masked_detail_losses(prediction: torch.Tensor, source: torch.Tensor,
                         teacher: torch.Tensor, clean: torch.Tensor,
                         settings: dict) -> tuple[torch.Tensor, torch.Tensor]:
    correction = (teacher - source).abs().mean(dim=1, keepdim=True).detach()
    scale = float(settings.get("detail_correction_scale", 0.04))
    threshold = float(settings.get("detail_edge_threshold", 0.01))
    slope = float(settings.get("detail_edge_slope", 100.0))

    def weighted(error: torch.Tensor, edge: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mask = (torch.sigmoid((edge - threshold) * slope) * torch.exp(-noise / scale)).detach()
        return (error * mask).sum() / (mask.sum() * error.shape[1]).clamp_min(1e-6)

    pred_x, clean_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1], clean[:, :, :, 1:] - clean[:, :, :, :-1]
    pred_y, clean_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :], clean[:, :, 1:, :] - clean[:, :, :-1, :]
    edge_x, edge_y = clean_x.abs().mean(dim=1, keepdim=True), clean_y.abs().mean(dim=1, keepdim=True)
    noise_x = (correction[:, :, :, 1:] + correction[:, :, :, :-1]) * 0.5
    noise_y = (correction[:, :, 1:, :] + correction[:, :, :-1, :]) * 0.5
    gradient = 0.5 * (weighted((pred_x - clean_x).abs(), edge_x, noise_x)
                      + weighted((pred_y - clean_y).abs(), edge_y, noise_y))

    kernel = prediction.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(prediction.shape[1], 1, 1, 1)
    pred_lap = F.conv2d(prediction, kernel, padding=1, groups=prediction.shape[1])
    clean_lap = F.conv2d(clean, kernel, padding=1, groups=clean.shape[1])
    lap_edge = clean_lap.abs().mean(dim=1, keepdim=True)
    laplacian = weighted((pred_lap - clean_lap).abs(), lap_edge, correction)
    return gradient, laplacian


def loss_for(prediction: torch.Tensor, source: torch.Tensor, teacher: torch.Tensor,
             clean: torch.Tensor, settings: dict, border: int, warmup: bool,
             detail_factor: float = 1.0) -> torch.Tensor:
    prediction, source, teacher, clean = (core(value, border) for value in (prediction, source, teacher, clean))
    teacher_weight = 0.0 if warmup else float(settings["teacher_mse_weight"])
    teacher_error = (prediction - teacher).square()
    shadow_strength = float(settings.get("shadow_kd_strength", 0.0))
    if shadow_strength:
        threshold = float(settings.get("shadow_kd_threshold", 0.22))
        slope = float(settings.get("shadow_kd_slope", 24.0))
        emphasis = torch.sigmoid((threshold - luminance(source).detach()) * slope)
        shadow_weights = 1.0 + shadow_strength * emphasis
        shadow_weights = shadow_weights / shadow_weights.mean().detach().clamp_min(1e-6)
        teacher_loss = (teacher_error * shadow_weights).mean()
    else:
        teacher_loss = teacher_error.mean()
    loss = (float(settings["clean_mse_weight"]) * F.mse_loss(prediction, clean)
            + teacher_weight * teacher_loss
            + float(settings["clean_l1_weight"]) * F.l1_loss(prediction, clean))
    gradient_weight = float(settings.get("clean_gradient_weight", 0.0)) * detail_factor
    laplacian_weight = float(settings.get("clean_laplacian_weight", 0.0)) * detail_factor
    if gradient_weight or laplacian_weight:
        gradient, laplacian = masked_detail_losses(prediction, source, teacher, clean, settings)
        loss = loss + gradient_weight * gradient + laplacian_weight * laplacian
    projection_weight = float(settings.get("residual_projection_weight", 0.0))
    if projection_weight and not warmup:
        target_residual = (teacher - source).float()
        predicted_residual = (prediction - source).float()
        target_energy = target_residual.square().flatten(1).mean(1)
        projection = ((predicted_residual * target_residual).flatten(1).mean(1)
                      / target_energy.clamp_min(1e-8))
        minimum_rms = float(settings.get("residual_projection_minimum_rms", 0.01))
        strong_correction = target_energy.sqrt() >= minimum_rms
        if strong_correction.any():
            projection_loss = (projection[strong_correction] - 1.0).square().mean()
            loss = loss + projection_weight * projection_loss
    multiscale_weight = float(settings.get("residual_multiscale_weight", 0.0))
    if multiscale_weight and not warmup:
        predicted_residual = prediction.float() - source.float()
        target_residual = teacher.float() - source.float()
        scale_losses = []
        energy_floor = float(settings.get("residual_multiscale_energy_floor", 1e-5))
        for scale in settings.get("residual_multiscale_scales", (4, 16, 64)):
            predicted_low = F.avg_pool2d(predicted_residual, int(scale), int(scale))
            target_low = F.avg_pool2d(target_residual, int(scale), int(scale))
            target_energy = target_low.square().mean().detach().clamp_min(energy_floor)
            scale_losses.append(F.mse_loss(predicted_low, target_low) / target_energy)
        loss = loss + multiscale_weight * torch.stack(scale_losses).mean()
    bandpass_weight = float(settings.get("residual_bandpass_weight", 0.0))
    if bandpass_weight and not warmup:
        predicted_bands = residual_bands(prediction.float() - source.float())
        target_bands = residual_bands(teacher.float() - source.float())
        configured_weights = settings.get("residual_band_weights", {})
        energy_floor = float(settings.get("residual_band_energy_floor", 1e-6))
        losses, weights = [], []
        for name, target_band in target_bands.items():
            weight = float(configured_weights.get(name, 1.0))
            target_energy = target_band.square().mean().detach().clamp_min(energy_floor)
            losses.append(F.mse_loss(predicted_bands[name], target_band) / target_energy)
            weights.append(weight)
        normalized = sum(value * weight for value, weight in zip(losses, weights)) / sum(weights)
        loss = loss + bandpass_weight * normalized
    return loss


def decoder_multiscale_loss(
    predictions: dict[str, torch.Tensor], source: torch.Tensor,
    teacher: torch.Tensor, settings: dict,
) -> torch.Tensor:
    target_residual = (teacher - source).float()
    configured_weights = settings.get("decoder_multiscale_level_weights", {})
    energy_floor = float(settings.get("decoder_multiscale_energy_floor", 1e-6))
    losses, weights = [], []
    for name, prediction in predictions.items():
        target = F.adaptive_avg_pool2d(target_residual, prediction.shape[-2:])
        target_energy = target.square().mean().detach().clamp_min(energy_floor)
        weight = float(configured_weights.get(name, 1.0))
        losses.append(F.mse_loss(prediction.float(), target) / target_energy)
        weights.append(weight)
    return sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float, updates: int) -> None:
    effective_decay = min(decay, (1.0 + updates) / (10.0 + updates))
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - effective_decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             amp: bool, settings: dict, border: int) -> dict[str, float]:
    model.eval()
    totals = Counter()
    for source, teacher, clean in loader:
        source, teacher, clean = source.to(device), teacher.to(device), clean.to(device)
        with torch.autocast(device.type, enabled=amp):
            prediction = model(source)
        p, t, c = core(prediction.float(), border), core(teacher, border), core(clean, border)
        s = core(source, border)
        clean_mse, teacher_mse = F.mse_loss(p, c), F.mse_loss(p, t)
        count = source.shape[0]
        edge = 0.5 * (F.l1_loss(p[:, :, :, 1:] - p[:, :, :, :-1], c[:, :, :, 1:] - c[:, :, :, :-1])
                      + F.l1_loss(p[:, :, 1:, :] - p[:, :, :-1, :], c[:, :, 1:, :] - c[:, :, :-1, :]))
        totals.update(samples=count, clean_mse=clean_mse.item() * count,
                      teacher_mse=teacher_mse.item() * count, clean_l1=F.l1_loss(p, c).item() * count,
                      clean_edge_l1=edge.item() * count,
                      residual_dot=((p - s) * (t - s)).sum().item(),
                      residual_student_energy=(p - s).square().sum().item(),
                      residual_energy=(t - s).square().sum().item())
        predicted_bands = residual_bands(p - s)
        target_bands = residual_bands(t - s)
        source_luminance = luminance(s)
        region_masks = {
            "shadow": source_luminance < 0.20,
            "midtone": (source_luminance >= 0.20) & (source_luminance < 0.60),
            "highlight": source_luminance >= 0.60,
        }
        for name, target_band in target_bands.items():
            predicted_band = predicted_bands[name]
            totals[f"{name}_dot"] += (predicted_band * target_band).sum().item()
            totals[f"{name}_student_energy"] += predicted_band.square().sum().item()
            totals[f"{name}_teacher_energy"] += target_band.square().sum().item()
            for region, mask in region_masks.items():
                region_mask = mask.to(predicted_band.dtype)
                totals[f"{region}_{name}_dot"] += (predicted_band * target_band * region_mask).sum().item()
                totals[f"{region}_{name}_student_energy"] += (predicted_band.square() * region_mask).sum().item()
                totals[f"{region}_{name}_teacher_energy"] += (target_band.square() * region_mask).sum().item()
        predicted_residual, target_residual = p - s, t - s
        for region, mask in region_masks.items():
            region_mask = mask.to(predicted_residual.dtype)
            totals[f"{region}_dot"] += (predicted_residual * target_residual * region_mask).sum().item()
            totals[f"{region}_student_energy"] += (predicted_residual.square() * region_mask).sum().item()
            totals[f"{region}_teacher_energy"] += (target_residual.square() * region_mask).sum().item()
    count = totals["samples"]
    clean_mse, teacher_mse = totals["clean_mse"] / count, totals["teacher_mse"] / count
    band_similarities = {
        name: vector_similarity(
            totals[f"{name}_dot"],
            totals[f"{name}_student_energy"],
            totals[f"{name}_teacher_energy"],
        )
        for name in ("fine", "medium", "coarse", "very_coarse")
    }
    regional_similarities = {
        region: vector_similarity(
            totals[f"{region}_dot"], totals[f"{region}_student_energy"],
            totals[f"{region}_teacher_energy"],
        )
        for region in ("shadow", "midtone", "highlight")
    }
    shadow_band_similarities = {
        name: vector_similarity(
            totals[f"shadow_{name}_dot"], totals[f"shadow_{name}_student_energy"],
            totals[f"shadow_{name}_teacher_energy"],
        )
        for name in ("fine", "medium", "coarse", "very_coarse")
    }
    configured_weights = settings.get("validation_band_weights", {})
    weights = {name: float(configured_weights.get(name, 1.0)) for name in band_similarities}
    harmonic = sum(weights.values()) / sum(
        weights[name] / max(min(value, 1.0), 1e-4)
        for name, value in band_similarities.items()
    )
    critical = min(band_similarities["medium"], band_similarities["coarse"])
    critical_weight = float(settings.get("validation_critical_band_weight", 0.5))
    weakness_score = (1.0 - critical_weight) * harmonic + critical_weight * critical
    shadow_harmonic = sum(weights.values()) / sum(
        weights[name] / max(min(value, 1.0), 1e-4)
        for name, value in shadow_band_similarities.items()
    )
    shadow_critical = min(shadow_band_similarities["medium"], shadow_band_similarities["coarse"])
    shadow_score = (1.0 - critical_weight) * shadow_harmonic + critical_weight * shadow_critical
    shadow_selection_score = 3.0 / (
        2.0 / max(shadow_score, 1e-4) + 1.0 / max(weakness_score, 1e-4)
    )
    result = {
        "clean_psnr": -10 * math.log10(max(clean_mse, 1e-12)),
        "teacher_psnr": -10 * math.log10(max(teacher_mse, 1e-12)),
        "clean_l1": totals["clean_l1"] / count,
        "clean_edge_l1": totals["clean_edge_l1"] / count,
        "correction_projection": totals["residual_dot"] / max(totals["residual_energy"], 1e-12),
        "residual_similarity": vector_similarity(
            totals["residual_dot"],
            totals["residual_student_energy"],
            totals["residual_energy"],
        ),
        "band_harmonic_similarity": harmonic,
        "weakness_score": weakness_score,
        "shadow_score": shadow_score,
        "shadow_selection_score": shadow_selection_score,
    }
    result.update({f"{name}_similarity": value for name, value in band_similarities.items()})
    result.update({f"{name}_residual_similarity": value for name, value in regional_similarities.items()})
    result.update({f"shadow_{name}_similarity": value for name, value in shadow_band_similarities.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_litedenoise.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--run-name", default="litedenoise_w16")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base, settings = config_path.parent, config["training"]
    run = base / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(settings["amp"] and device.type == "cuda")
    cache = (base / config["data"]["cache_root"]).resolve()
    manifest_name = config["data"].get("manifest", "manifest.json")
    train_data = PatchDataset(cache, "train", True, manifest_name)
    validation_data = PatchDataset(cache, "validation", False, manifest_name)
    counts = Counter(x["dataset"] for x in train_data.records)
    weights, sampling_report = sampling_weights(train_data, settings)
    samples_per_epoch = int(settings.get("samples_per_epoch", len(train_data)))
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(config["seed"]))
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], sampler=sampler,
                              num_workers=settings["num_workers"], pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False,
                                   num_workers=settings["num_workers"], pin_memory=device.type == "cuda")
    model = LiteDenoiseStudent(**config["student"], clamp_output=False).to(device)
    decoder_weight = float(settings.get("decoder_multiscale_weight", 0.0))
    auxiliary = (MultiscaleResidualHeads(int(config["student"]["base_width"])).to(device)
                 if decoder_weight else None)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = float(settings.get("ema_decay", 0.0))
    ema_updates = 0
    if args.resume and args.initial_checkpoint:
        parser.error("--resume and --initial-checkpoint are mutually exclusive")
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(initial.get("ema_model", initial["model"]), strict=False)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing = [key for key in incompatible.missing_keys if "_extra." not in key]
        if unexpected or allowed_missing:
            raise RuntimeError(f"Incompatible initial checkpoint: missing={allowed_missing} unexpected={unexpected}")
        if auxiliary and initial.get("auxiliary") is not None:
            auxiliary.load_state_dict(initial["auxiliary"])
        ema_model.load_state_dict(model.state_dict())
    trainable = list(model.parameters()) + (list(auxiliary.parameters()) if auxiliary else [])
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    epochs = args.epochs or int(settings["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=settings["minimum_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    warmup_epochs = int(settings["warmup_epochs"])
    monitored_metric = settings.get("monitor_metric", "teacher_psnr")
    start_epoch, history, best_phase_metric, stale = 0, [], -math.inf, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        if auxiliary and checkpoint.get("auxiliary") is not None:
            auxiliary.load_state_dict(checkpoint["auxiliary"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        ema_updates = int(checkpoint.get("ema_updates", 0))
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, history = checkpoint["epoch"], checkpoint.get("history", [])
        phase_history = [x for x in history if x["epoch"] > warmup_epochs]
        if phase_history:
            best_phase_metric = max(x[monitored_metric] for x in phase_history)
            best_index = max(range(len(phase_history)), key=lambda i: phase_history[i][monitored_metric])
            stale = len(phase_history) - best_index - 1
        elif history:
            best_phase_metric = max(x["clean_psnr"] for x in history)
        if not (run / "best_distilled.pt").exists():
            torch.save(checkpoint, run / "best_distilled.pt")
    accumulation, border = int(settings["gradient_accumulation"]), int(config["loss_border"])
    auxiliary_parameters = sum(p.numel() for p in auxiliary.parameters()) if auxiliary else 0
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters())} "
          f"training_only_parameters={auxiliary_parameters} train={len(train_data)} "
          f"validation={len(validation_data)} datasets={dict(counts)} sampling={sampling_report}")
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"LiteDenoise {epoch}/{epochs}")
        for step, (source, teacher, clean) in enumerate(progress, 1):
            source, teacher, clean = source.to(device), teacher.to(device), clean.to(device)
            with torch.autocast(device.type, enabled=amp):
                ramp_epochs = max(1, int(settings.get("detail_ramp_epochs", 1)))
                detail_factor = min(1.0, epoch / ramp_epochs)
                if auxiliary:
                    prediction, features = model.forward_with_features(source)
                    auxiliary_predictions = auxiliary(features)
                else:
                    prediction, auxiliary_predictions = model(source), None
                loss = loss_for(prediction, source, teacher, clean, settings, border,
                                epoch <= warmup_epochs, detail_factor)
                if auxiliary_predictions is not None and epoch > warmup_epochs:
                    loss = loss + decoder_weight * decoder_multiscale_loss(
                        auxiliary_predictions, source, teacher, settings
                    )
                loss = loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                gradient_clip = float(settings.get("gradient_clip_norm", 0.0))
                if gradient_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, gradient_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema_updates += 1
                if ema_decay:
                    update_ema(ema_model, model, ema_decay, ema_updates)
                else:
                    ema_model.load_state_dict(model.state_dict())
            progress.set_postfix(loss=f"{loss.item() * accumulation:.4f}")
        metrics = evaluate(ema_model, validation_loader, device, amp, settings, border)
        scheduler.step()
        history.append({"epoch": epoch, **metrics, "learning_rate": scheduler.get_last_lr()[0]})
        checkpoint = {"model": model.state_dict(), "ema_model": ema_model.state_dict(),
                      "ema_updates": ema_updates,
                      "auxiliary": auxiliary.state_dict() if auxiliary else None,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch, "history": history, "config": config}
        torch.save(checkpoint, run / "latest.pt")
        interval = int(settings.get("checkpoint_interval", 0))
        if interval and epoch % interval == 0:
            torch.save(checkpoint, run / f"epoch_{epoch:03d}.pt")
        if epoch == warmup_epochs + 1:
            best_phase_metric, stale = -math.inf, 0
        phase_metric = metrics["clean_psnr"] if epoch <= warmup_epochs else metrics[monitored_metric]
        improvement = phase_metric - best_phase_metric
        if improvement > 0:
            torch.save(checkpoint, run / ("best_warmup.pt" if epoch <= warmup_epochs else "best_distilled.pt"))
        if improvement >= float(settings["early_stopping_min_delta_db"]):
            best_phase_metric, stale = phase_metric, 0
        else:
            best_phase_metric, stale = max(best_phase_metric, phase_metric), stale + 1
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        phase_name = "clean_psnr" if epoch <= warmup_epochs else monitored_metric
        print(f"validation={metrics} monitored={phase_name} best={best_phase_metric:.3f} stale={stale}")
        if epoch > warmup_epochs and stale >= int(settings["early_stopping_patience"]):
            print("early stopping: distillation teacher PSNR no longer improving")
            break


if __name__ == "__main__":
    main()
