#!/usr/bin/env python3
"""Build deterministic per-patch hard-example sampling weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import atomic_json, resolve_paper_path, seed_worker
from src.mixed_dataset import MixedDistillationDataset, MixedManifestSnapshot
from src.noise_conditioning import model_input_from_config
from src.student import student_from_checkpoint
from train_mixed import chroma_projection, gaussian_blur


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Score current student errors against SCUNet."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shadow-threshold", type=float, default=0.25)
    parser.add_argument("--fine-sigma", type=float, default=1.0)
    parser.add_argument("--medium-sigma", type=float, default=4.0)
    parser.add_argument("--coarse-sigma", type=float, default=12.0)
    parser.add_argument("--very-coarse-sigma", type=float, default=24.0)
    parser.add_argument("--flat-gradient-threshold", type=float, default=0.03)
    parser.add_argument(
        "--bin-weights", type=float, nargs=4, default=(0.55, 0.85, 1.25, 1.85)
    )
    args = parser.parse_args()
    manifest_path = resolve_paper_path(args.manifest)
    cache_root = resolve_paper_path(args.cache_root)
    snapshot = MixedManifestSnapshot.load(manifest_path)
    dataset = MixedDistillationDataset(
        snapshot, root=cache_root, split="train", augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = None
    checkpoint_metadata = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        student = student_from_checkpoint(checkpoint).to(device).eval()
        checkpoint_metadata = {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "epoch": checkpoint.get("epoch"),
        }
    measured: dict[int, tuple[float, ...]] = {}
    luma_weights = torch.tensor((0.2126, 0.7152, 0.0722), device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="difficulty"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            shadow = ((noisy * luma_weights).sum(1, keepdim=True) < args.shadow_threshold).float()
            correction = (teacher - noisy).float()
            student_error = (
                student(
                    model_input_from_config(
                        noisy,
                        checkpoint["config"]["model"],
                    )
                ).clamp(0.0, 1.0).float() - teacher.float()
                if student is not None
                else correction
            )
            shadow_pixels = shadow.flatten(1).sum(1)
            shadow_fraction = shadow_pixels / (noisy.shape[-2] * noisy.shape[-1])
            shadow_magnitude = (
                correction.abs() * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            fine = gaussian_blur(correction, args.fine_sigma)
            medium = gaussian_blur(correction, args.medium_sigma)
            coarse = gaussian_blur(correction, args.coarse_sigma)
            multiscale = (
                (fine - medium).abs().flatten(1).mean(1)
                + (medium - coarse).abs().flatten(1).mean(1)
            )
            shadow_error = (
                student_error.abs() * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            error_fine = gaussian_blur(student_error, args.fine_sigma)
            error_medium = gaussian_blur(student_error, args.medium_sigma)
            error_coarse = gaussian_blur(student_error, args.coarse_sigma)
            multiscale_error = (
                (error_fine - error_medium).abs().flatten(1).mean(1)
                + (error_medium - error_coarse).abs().flatten(1).mean(1)
            )
            teacher_chroma = chroma_projection(correction)
            error_chroma = chroma_projection(student_error)
            shadow_chroma = (
                teacher_chroma.abs() * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            shadow_chroma_error = (
                error_chroma.abs() * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            teacher_chroma_blurs = [
                gaussian_blur(teacher_chroma, sigma)
                for sigma in (
                    args.fine_sigma,
                    args.medium_sigma,
                    args.coarse_sigma,
                    args.very_coarse_sigma,
                )
            ]
            error_chroma_blurs = [
                gaussian_blur(error_chroma, sigma)
                for sigma in (
                    args.fine_sigma,
                    args.medium_sigma,
                    args.coarse_sigma,
                    args.very_coarse_sigma,
                )
            ]
            chroma_multiscale = sum(
                (left - right).abs()
                for left, right in zip(
                    teacher_chroma_blurs[:-1],
                    teacher_chroma_blurs[1:],
                    strict=True,
                )
            )
            chroma_multiscale_error = sum(
                (left - right).abs()
                for left, right in zip(
                    error_chroma_blurs[:-1],
                    error_chroma_blurs[1:],
                    strict=True,
                )
            )
            very_coarse_chroma = (
                teacher_chroma_blurs[-2] - teacher_chroma_blurs[-1]
            ).abs().flatten(1).mean(1)
            very_coarse_chroma_error = (
                error_chroma_blurs[-2] - error_chroma_blurs[-1]
            ).abs().flatten(1).mean(1)
            row_column_chroma = 0.5 * (
                teacher_chroma.mean(dim=3).abs().flatten(1).mean(1)
                + teacher_chroma.mean(dim=2).abs().flatten(1).mean(1)
            )
            row_column_chroma_error = 0.5 * (
                error_chroma.mean(dim=3).abs().flatten(1).mean(1)
                + error_chroma.mean(dim=2).abs().flatten(1).mean(1)
            )
            shadow_chroma_multiscale = (
                chroma_multiscale * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            shadow_chroma_multiscale_error = (
                chroma_multiscale_error * shadow
            ).flatten(1).sum(1) / (shadow_pixels * noisy.shape[1]).clamp_min(1.0)
            teacher_luma = (teacher * luma_weights).sum(1, keepdim=True)
            horizontal_gradient = torch.nn.functional.pad(
                (teacher_luma[..., 1:] - teacher_luma[..., :-1]).abs(),
                (0, 1, 0, 0),
                mode="replicate",
            )
            vertical_gradient = torch.nn.functional.pad(
                (teacher_luma[..., 1:, :] - teacher_luma[..., :-1, :]).abs(),
                (0, 0, 0, 1),
                mode="replicate",
            )
            flat_shadow = (
                shadow
                * (
                    0.5 * (horizontal_gradient + vertical_gradient)
                    < args.flat_gradient_threshold
                ).to(shadow)
            )
            flat_pixels = flat_shadow.flatten(1).sum(1)
            flat_denominator = (flat_pixels * noisy.shape[1]).clamp_min(1.0)
            flat_shadow_fraction = flat_pixels / (
                noisy.shape[-2] * noisy.shape[-1]
            )
            flat_shadow_chroma = (
                teacher_chroma.abs() * flat_shadow
            ).flatten(1).sum(1) / flat_denominator
            flat_shadow_chroma_error = (
                error_chroma.abs() * flat_shadow
            ).flatten(1).sum(1) / flat_denominator
            for values in zip(
                batch["index"].tolist(),
                shadow_fraction.cpu().tolist(),
                shadow_magnitude.cpu().tolist(),
                multiscale.cpu().tolist(),
                shadow_error.cpu().tolist(),
                multiscale_error.cpu().tolist(),
                shadow_chroma.cpu().tolist(),
                shadow_chroma_error.cpu().tolist(),
                shadow_chroma_multiscale.cpu().tolist(),
                shadow_chroma_multiscale_error.cpu().tolist(),
                flat_shadow_fraction.cpu().tolist(),
                flat_shadow_chroma.cpu().tolist(),
                flat_shadow_chroma_error.cpu().tolist(),
                very_coarse_chroma.cpu().tolist(),
                very_coarse_chroma_error.cpu().tolist(),
                row_column_chroma.cpu().tolist(),
                row_column_chroma_error.cpu().tolist(),
                strict=True,
            ):
                measured[int(values[0])] = tuple(map(float, values[1:]))
    if len(measured) != len(dataset):
        raise RuntimeError("Difficulty pass did not measure every training record")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[record.dataset].append(index)
    sampling_weights: dict[str, float] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    dataset_bin_counts: dict[str, dict[str, int]] = {}
    for dataset_name, indices in sorted(grouped.items()):
        values = np.asarray([measured[index] for index in indices], dtype=np.float64)
        if student is None:
            combined = (
                0.15 * percentile_ranks(values[:, 0])
                + 0.45 * percentile_ranks(values[:, 1])
                + 0.40 * percentile_ranks(values[:, 2])
            )
        else:
            combined = (
                0.02 * percentile_ranks(values[:, 0])
                + 0.04 * percentile_ranks(values[:, 1])
                + 0.04 * percentile_ranks(values[:, 2])
                + 0.07 * percentile_ranks(values[:, 3])
                + 0.05 * percentile_ranks(values[:, 4])
                + 0.04 * percentile_ranks(values[:, 5])
                + 0.10 * percentile_ranks(values[:, 6])
                + 0.03 * percentile_ranks(values[:, 7])
                + 0.08 * percentile_ranks(values[:, 8])
                + 0.01 * percentile_ranks(values[:, 9])
                + 0.04 * percentile_ranks(values[:, 10])
                + 0.05 * percentile_ranks(values[:, 11])
                + 0.03 * percentile_ranks(values[:, 12])
                + 0.20 * percentile_ranks(values[:, 13])
                + 0.03 * percentile_ranks(values[:, 14])
                + 0.17 * percentile_ranks(values[:, 15])
            )
        combined_ranks = percentile_ranks(combined)
        bins = np.minimum((combined_ranks * 4).astype(np.int64), 3)
        counts = Counter(map(int, bins))
        dataset_bin_counts[dataset_name] = {
            str(key): counts.get(key, 0) for key in range(4)
        }
        for local_index, record_index in enumerate(indices):
            record = dataset.records[record_index]
            sampling_weights[record.input] = float(args.bin_weights[bins[local_index]])
            (
                fraction,
                shadow_value,
                scale_value,
                shadow_gap,
                scale_gap,
                shadow_chroma_value,
                shadow_chroma_gap,
                chroma_scale_value,
                chroma_scale_gap,
                flat_fraction,
                flat_chroma_value,
                flat_chroma_gap,
                very_coarse_chroma_value,
                very_coarse_chroma_gap,
                row_column_chroma_value,
                row_column_chroma_gap,
            ) = measured[record_index]
            metrics[record.input] = {
                "shadow_fraction": fraction,
                "shadow_teacher_correction": shadow_value,
                "medium_coarse_teacher_correction": scale_value,
                "student_shadow_error": shadow_gap,
                "student_medium_coarse_error": scale_gap,
                "shadow_chroma_teacher_correction": shadow_chroma_value,
                "student_shadow_chroma_error": shadow_chroma_gap,
                "shadow_multiscale_chroma_teacher_correction": chroma_scale_value,
                "student_shadow_multiscale_chroma_error": chroma_scale_gap,
                "flat_shadow_fraction": flat_fraction,
                "flat_shadow_chroma_teacher_correction": flat_chroma_value,
                "student_flat_shadow_chroma_error": flat_chroma_gap,
                "very_coarse_chroma_teacher_correction": very_coarse_chroma_value,
                "student_very_coarse_chroma_error": very_coarse_chroma_gap,
                "row_column_chroma_teacher_correction": row_column_chroma_value,
                "student_row_column_chroma_error": row_column_chroma_gap,
                "difficulty_percentile": float(combined_ranks[local_index]),
                "difficulty_bin": int(bins[local_index]),
            }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output,
        {
            "version": 1,
            "manifest": str(manifest_path),
            "manifest_sha256": snapshot.sha256,
            "records": len(dataset),
            "scoring": (
                "student_error with shadow, multiscale, shadow-chroma, "
                "flat-shadow, very-coarse, and row-column chroma residual ranking"
                if student is not None
                else "0.15 shadow_fraction + 0.45 shadow_correction + 0.40 medium_coarse"
            ),
            "student_checkpoint": checkpoint_metadata,
            "bin_weights": list(args.bin_weights),
            "dataset_bin_counts": dataset_bin_counts,
            "sampling_weights": sampling_weights,
            "metrics": metrics,
        },
    )
    print(f"wrote {output} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
