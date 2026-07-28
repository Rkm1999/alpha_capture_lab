#!/usr/bin/env python3
"""Run the guide's structural, alignment, teacher, and cache-parity gates."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import atomic_json, load_config, resolve_paper_path, seed_everything
from src.dataset import DistillationDataset
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.scunet_teacher import load_scunet_teacher


def phase_shift(first: np.ndarray, second: np.ndarray) -> tuple[int, int] | None:
    first_gray = first.mean(axis=2).astype(np.float64)
    second_gray = second.mean(axis=2).astype(np.float64)
    first_gray -= first_gray.mean()
    second_gray -= second_gray.mean()
    if first_gray.std() < 1e-4 or second_gray.std() < 1e-4:
        return None
    cross = np.fft.fft2(first_gray) * np.conj(np.fft.fft2(second_gray))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross))
    row, column = np.unravel_index(np.argmax(correlation), correlation.shape)
    if row > first_gray.shape[0] // 2:
        row -= first_gray.shape[0]
    if column > first_gray.shape[1] // 2:
        column -= first_gray.shape[1]
    return int(row), int(column)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def contact_sheet(
    dataset: DistillationDataset,
    destination: Path,
    count: int,
    seed: int,
) -> None:
    generator = random.Random(seed)
    indices = generator.sample(range(len(dataset)), min(count, len(dataset)))
    rows = []
    for index in indices:
        sample = dataset[index]
        panels = []
        for key in ("noisy", "teacher", "clean"):
            array = sample[key].permute(1, 2, 0).numpy()
            panels.append(Image.fromarray(np.rint(array * 255).clip(0, 255).astype(np.uint8)))
        row = Image.new("RGB", (192 * 3, 216), "white")
        for panel_index, panel in enumerate(panels):
            row.paste(panel, (panel_index * 192, 24))
        draw = ImageDraw.Draw(row)
        draw.text((4, 4), f"{sample['dataset']} / {sample['scene']}    noisy | SCUNet | clean", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (192 * 3, 216 * len(rows)), "white")
    for row_index, row in enumerate(rows):
        sheet.paste(row, (0, row_index * 216))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--contact-count", type=int, default=12)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document["records"]

    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    shifts: collections.Counter[str] = collections.Counter()
    dtypes: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    for record in tqdm(records, desc="Validating arrays"):
        scene_splits[(record["dataset"], record["scene"])].add(record["split"])
        counts[(record["split"], record["dataset"])] += 1
        arrays = {}
        for field in ("input", "clean", "teacher"):
            path = cache_root / record[field]
            if not path.is_file():
                problems.append(f"missing:{path}")
                continue
            array = np.load(path, allow_pickle=False)
            arrays[field] = array
            dtypes[f"{field}/{array.dtype}"] += 1
            if array.shape != (192, 192, 3):
                problems.append(f"shape:{path}:{array.shape}")
            elif not np.isfinite(array).all():
                problems.append(f"nonfinite:{path}")
            elif float(array.min()) < 0.0 or float(array.max()) > 1.0:
                problems.append(f"range:{path}:{float(array.min())}:{float(array.max())}")
        if "input" in arrays and "clean" in arrays:
            shift = phase_shift(arrays["input"], arrays["clean"])
            shifts["low_texture" if shift is None else f"{shift[0]},{shift[1]}"] += 1

    leakage = [key for key, split_set in scene_splits.items() if len(split_set) > 1]
    if leakage:
        problems.append(f"scene_leakage:{leakage[0]}")
    nonzero_shifts = {
        shift: count for shift, count in shifts.items() if shift not in {"0,0", "low_texture"}
    }
    if nonzero_shifts:
        first_shift = next(iter(nonzero_shifts))
        problems.append(f"integer_registration_shift:{first_shift}:{nonzero_shifts[first_shift]}")
    if problems:
        raise RuntimeError(f"Dataset gate failed with {len(problems)} problems; first: {problems[0]}")

    validation = DistillationDataset(manifest_path, root=cache_root, split="validation")
    loader = DataLoader(
        validation,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    border = int(config["metrics"]["border_crop"])
    window = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    by_dataset: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Qualifying SCUNet"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            metrics = {
                "noisy_psnr": psnr_per_image(noisy, clean, border=border),
                "teacher_psnr": psnr_per_image(teacher, clean, border=border),
                "noisy_ssim": gaussian_ssim_per_image(noisy, clean, border=border, window_size=window, sigma=sigma),
                "teacher_ssim": gaussian_ssim_per_image(teacher, clean, border=border, window_size=window, sigma=sigma),
            }
            for row, dataset in enumerate(batch["dataset"]):
                for name, value in metrics.items():
                    by_dataset[str(dataset)][name].append(float(value[row].cpu()))

    baseline = {}
    combined: dict[str, list[float]] = collections.defaultdict(list)
    for dataset, metrics in sorted(by_dataset.items()):
        baseline[dataset] = {name: summarize(values) for name, values in metrics.items()}
        for name, values in metrics.items():
            combined[name].extend(values)
    baseline["all"] = {name: summarize(values) for name, values in combined.items()}
    baseline["all"]["teacher_psnr_gain"] = {
        "mean": baseline["all"]["teacher_psnr"]["mean"] - baseline["all"]["noisy_psnr"]["mean"]
    }
    target_dataset_names = set(config["data"].get("target_datasets", []))
    present_target_names = target_dataset_names.intersection(by_dataset)
    target_teacher_gate: bool | None = None
    if present_target_names:
        target_noisy = [
            value for name in present_target_names for value in by_dataset[name]["noisy_psnr"]
        ]
        target_teacher = [
            value for name in present_target_names for value in by_dataset[name]["teacher_psnr"]
        ]
        target_teacher_gate = float(np.mean(target_teacher)) > float(np.mean(target_noisy))

    parity = {"skipped": True}
    if not args.skip_parity:
        parity_count = min(int(config["teacher"]["parity_samples"]), len(validation))
        indices = random.Random(seed).sample(range(len(validation)), parity_count)
        values = torch.stack([validation[index]["noisy"] for index in indices]).to(device)
        cached = torch.stack([validation[index]["teacher"] for index in indices])
        teacher_model = load_scunet_teacher(
            resolve_paper_path(config["teacher"]["repository"]),
            resolve_paper_path(config["teacher"]["checkpoint"]),
            device,
        )
        fresh_batches = []
        parity_batch_size = int(config["teacher"]["cache_batch_size"])
        with torch.inference_mode():
            for offset in range(0, len(values), parity_batch_size):
                fresh_batches.append(
                    teacher_model(values[offset : offset + parity_batch_size]).clamp(0.0, 1.0).cpu()
                )
        fresh = torch.cat(fresh_batches)
        fresh_float16 = fresh.to(torch.float16)
        difference = (cached - fresh).abs()
        parity = {
            "skipped": False,
            "samples": parity_count,
            "float16_bit_exact": bool(torch.equal(cached.to(torch.float16), fresh_float16)),
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }

    sheet_path = Path(__file__).parents[1] / "data_gate_contact_sheet.jpg"
    contact_sheet(validation, sheet_path, args.contact_count, seed)
    signoff_path = Path(__file__).parents[1] / "data/visual_alignment_signoff.json"
    if not signoff_path.is_file():
        raise RuntimeError(f"Visual alignment signoff is missing: {signoff_path}")
    visual_signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    if not visual_signoff.get("approved"):
        raise RuntimeError("Visual alignment signoff is not approved")
    zero_shift = shifts.get("0,0", 0)
    measured_shifts = sum(shifts.values()) - shifts.get("low_texture", 0)
    report = {
        "records": len(records),
        "counts": {f"{split}/{dataset}": count for (split, dataset), count in sorted(counts.items())},
        "scene_groups": len(scene_splits),
        "scene_leakage": len(leakage),
        "array_dtypes": dict(sorted(dtypes.items())),
        "phase_shift_counts": dict(shifts.most_common()),
        "zero_shift_fraction": zero_shift / measured_shifts if measured_shifts else math.nan,
        "teacher_baseline": baseline,
        "teacher_cache_parity": parity,
        "teacher_cache_parity_tolerance": float(config["teacher"]["parity_maximum_absolute_error"]),
        "public_pretraining_teacher_gate_passed": baseline["all"]["teacher_psnr"]["mean"] > baseline["all"]["noisy_psnr"]["mean"],
        "target_camera_data_present": bool(present_target_names),
        "target_camera_teacher_gate_passed": target_teacher_gate,
        "final_product_teacher_gate_passed": False,
        "contact_sheet": str(sheet_path),
        "visual_alignment_signoff": {"path": str(signoff_path), **visual_signoff},
    }
    destination = Path(__file__).parents[1] / "data_gate_report.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2))
    if not report["public_pretraining_teacher_gate_passed"]:
        raise RuntimeError("SCUNet did not improve mean validation PSNR over the noisy input")
    parity_tolerance = float(config["teacher"]["parity_maximum_absolute_error"])
    if not args.skip_parity and parity["maximum_absolute_error"] > parity_tolerance:
        raise RuntimeError(
            "Teacher cache parity exceeded the configured float16 tolerance: "
            f"{parity['maximum_absolute_error']} > {parity_tolerance}"
        )


if __name__ == "__main__":
    main()
