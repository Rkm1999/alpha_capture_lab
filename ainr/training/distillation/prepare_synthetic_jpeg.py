#!/usr/bin/env python3
"""Add severe synthetic JPEG-noise patches and SCUNet targets to a cache manifest."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from compare_full_images import load_scunet


def fine_corrupt(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    read_noise = rng.uniform(0.025, 0.075)
    shot_noise = rng.uniform(0.002, 0.018)
    sigma = np.sqrt(read_noise * read_noise + shot_noise * clean)
    noisy = clean + rng.normal(0.0, sigma, clean.shape)
    # Real JPEG noise is partially correlated between color channels.
    luma = rng.normal(0.0, read_noise * 0.35, clean.shape[:2] + (1,))
    noisy = np.clip(noisy + luma, 0.0, 1.0)
    encoded = io.BytesIO()
    Image.fromarray(np.rint(noisy * 255).astype(np.uint8), "RGB").save(
        encoded, format="JPEG", quality=int(rng.integers(82, 98)), subsampling=0
    )
    encoded.seek(0)
    return np.asarray(Image.open(encoded).convert("RGB"), dtype=np.float32) / 255.0


def normalized_field(shape: tuple[int, int, int], sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
    field = gaussian_filter(
        rng.normal(0.0, 1.0, shape).astype(np.float32),
        sigma=(sigma, sigma, 0.0),
        mode="reflect",
    )
    field -= field.mean(axis=(0, 1), keepdims=True)
    rms = np.sqrt(np.mean(field * field, axis=(0, 1), keepdims=True))
    return field / np.maximum(rms, 1e-6)


def correlated_noise(clean: np.ndarray, sigma: float, luma_rms: float,
                     chroma_rms: float, rng: np.random.Generator) -> np.ndarray:
    shape = clean.shape
    luma = normalized_field(shape[:2] + (1,), sigma, rng)
    chroma = normalized_field(shape, sigma, rng)
    chroma -= chroma.mean(axis=2, keepdims=True)
    chroma /= np.maximum(np.sqrt(np.mean(chroma * chroma)), 1e-6)
    brightness = clean.mean(axis=2, keepdims=True)
    shadow_gain = 0.6 + 1.4 * np.power(np.clip(1.0 - brightness, 0.0, 1.0), 1.5)
    return shadow_gain * (luma * luma_rms + chroma * chroma_rms)


def multiscale_corrupt(clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    read_rms = float(rng.uniform(0.004, 0.024))
    shot_variance = float(rng.uniform(0.00004, 0.0012))
    fine_sigma = np.sqrt(read_rms * read_rms + shot_variance * np.clip(clean, 0.0, 1.0))
    fine = rng.normal(0.0, 1.0, clean.shape).astype(np.float32) * fine_sigma
    fine += rng.normal(0.0, read_rms * 0.3, clean.shape[:2] + (1,)).astype(np.float32)

    medium_sigma = float(rng.uniform(1.2, 4.0))
    coarse_sigma = float(rng.uniform(5.0, 16.0))
    medium_luma = float(rng.uniform(0.003, 0.018))
    medium_chroma = float(rng.uniform(0.003, 0.022))
    coarse_luma = float(rng.uniform(0.0015, 0.010))
    coarse_chroma = float(rng.uniform(0.002, 0.015))
    medium = correlated_noise(clean, medium_sigma, medium_luma, medium_chroma, rng)
    coarse = correlated_noise(clean, coarse_sigma, coarse_luma, coarse_chroma, rng)

    noisy = np.clip(clean + fine + medium + coarse, 0.0, 1.0)
    quality = int(rng.integers(88, 99))
    subsampling = int(rng.choice((0, 1), p=(0.35, 0.65)))
    encoded = io.BytesIO()
    Image.fromarray(np.rint(noisy * 255).astype(np.uint8), "RGB").save(
        encoded, format="JPEG", quality=quality, subsampling=subsampling
    )
    encoded.seek(0)
    source = np.asarray(Image.open(encoded).convert("RGB"), dtype=np.float32) / 255.0
    return source, {
        "read_rms": read_rms,
        "shot_variance": shot_variance,
        "medium_sigma": medium_sigma,
        "medium_luma_rms": medium_luma,
        "medium_chroma_rms": medium_chroma,
        "coarse_sigma": coarse_sigma,
        "coarse_luma_rms": coarse_luma,
        "coarse_chroma_rms": coarse_chroma,
        "jpeg_quality": quality,
        "jpeg_subsampling": subsampling,
    }


def amplify_real_noise(clean: np.ndarray, noisy: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(1.6, 4.0)
    amplified = np.clip(clean + (noisy - clean) * scale, 0.0, 1.0)
    encoded = io.BytesIO()
    Image.fromarray(np.rint(amplified * 255).astype(np.uint8), "RGB").save(
        encoded, format="JPEG", quality=int(rng.integers(88, 99)), subsampling=0
    )
    encoded.seek(0)
    return np.asarray(Image.open(encoded).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--cache", type=Path, default=root / "data/litedenoise_cache")
    parser.add_argument("--base-manifest", default="manifest.json")
    parser.add_argument("--source-manifest", default="manifest_teacher_192.json")
    parser.add_argument("--train-count", type=int, default=2400)
    parser.add_argument("--validation-count", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9182)
    parser.add_argument("--output-manifest", default="manifest_synthetic_jpeg.json")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "multiscale", "amplified-real"),
        default="multiscale",
    )
    parser.add_argument("--directory", default="synthetic_jpeg")
    args = parser.parse_args()
    cache = args.cache.resolve()
    original = json.loads((cache / args.base_manifest).read_text(encoding="utf-8"))
    source_records = json.loads((cache / args.source_manifest).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected = []
    for split, count in (("train", args.train_count), ("validation", args.validation_count)):
        candidates = [record for record in source_records if record["split"] == split]
        selected.extend((split, record) for record in rng.sample(candidates, min(count, len(candidates))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to prepare SCUNet targets")
    teacher = load_scunet(root / "data/sources/scunet",
                          root / "data/sources/scunet/model_zoo/scunet_color_real_psnr.pth", device)
    generated = []
    destination = cache / args.directory
    destination.mkdir(parents=True, exist_ok=True)
    for offset in tqdm(range(0, len(selected), args.batch_size), desc="Synthetic JPEG teacher cache"):
        batch_records = selected[offset:offset + args.batch_size]
        sources, cleans, profiles = [], [], []
        for index, (_, record) in enumerate(batch_records, offset):
            clean = np.load(cache / record["clean"]).astype(np.float32)
            generator = np.random.default_rng(args.seed + index)
            if args.mode == "amplified-real":
                noisy = np.load(cache / record["input"]).astype(np.float32)
                sources.append(amplify_real_noise(clean, noisy, generator))
                profiles.append({"mode": args.mode})
            elif args.mode == "multiscale":
                source, profile = multiscale_corrupt(clean, generator)
                sources.append(source)
                profiles.append({"mode": args.mode, **profile})
            else:
                sources.append(fine_corrupt(clean, generator))
                profiles.append({"mode": args.mode})
            cleans.append(clean)
        value = torch.from_numpy(np.stack(sources)).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            predictions = teacher(value).clamp(0, 1).float().cpu().permute(0, 2, 3, 1).numpy()
        for index, ((split, record), source, clean, prediction, profile) in enumerate(
                zip(batch_records, sources, cleans, predictions, profiles), offset):
            stem = f"{split}_{index:05d}"
            paths = {name: f"{args.directory}/{stem}_{name}.npy" for name in ("input", "teacher", "clean")}
            for name, array in (("input", source), ("teacher", prediction), ("clean", clean)):
                np.save(cache / paths[name], array.astype(np.float16))
            generated.append({
                "dataset": args.directory,
                "scene": stem,
                "split": split,
                "noise_profile": profile,
                **paths,
            })
    (cache / args.output_manifest).write_text(json.dumps(original + generated, indent=2), encoding="utf-8")
    print(json.dumps({"original": len(original), "generated": len(generated),
                      "manifest": str(cache / args.output_manifest)}, indent=2))


if __name__ == "__main__":
    main()
