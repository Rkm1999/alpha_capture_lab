#!/usr/bin/env python3
"""Append SCUNet-labeled patches from independent A6300 high-ISO JPEGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from compare_full_images import load_scunet


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--cache", type=Path, default=root / "data/litedenoise_cache")
    parser.add_argument("--source", type=Path, default=root / "data/downloads/a6300_high_iso")
    parser.add_argument("--base-manifest", default="manifest_teacher_192.json")
    parser.add_argument("--output-manifest", default="manifest_teacher_192_a6300.json")
    parser.add_argument("--patches-per-image", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6300512)
    args = parser.parse_args()
    cache = args.cache.resolve()
    original = json.loads((cache / args.base_manifest).read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    patches = []
    source_images = sorted(
        path for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    for path in source_images:
        image = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        for patch_index in range(args.patches_per_image):
            x = int(rng.integers(0, width - 192 + 1))
            y = int(rng.integers(0, height - 192 + 1))
            patches.append((path.stem, patch_index, image[y:y + 192, x:x + 192].copy()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_scunet(root / "data/sources/scunet",
                          root / "data/sources/scunet/model_zoo/scunet_color_real_psnr.pth", device)
    destination = cache / "a6300_unpaired"
    destination.mkdir(parents=True, exist_ok=True)
    generated = []
    for offset in tqdm(range(0, len(patches), args.batch_size), desc="A6300 SCUNet cache"):
        batch = patches[offset:offset + args.batch_size]
        sources = np.stack([item[2].astype(np.float32) / 255.0 for item in batch])
        value = torch.from_numpy(sources).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            predictions = teacher(value).clamp(0, 1).float().cpu().permute(0, 2, 3, 1).numpy()
        for index, ((scene, patch_index, _), source, prediction) in enumerate(
                zip(batch, sources, predictions), offset):
            stem = f"{scene}_{patch_index:04d}"
            paths = {name: f"a6300_unpaired/{stem}_{name}.npy" for name in ("input", "teacher", "clean")}
            np.save(cache / paths["input"], source.astype(np.float16))
            np.save(cache / paths["teacher"], prediction.astype(np.float16))
            np.save(cache / paths["clean"], prediction.astype(np.float16))
            generated.append({"dataset": "a6300_high_iso", "scene": scene,
                              "split": "train", **paths})
    (cache / args.output_manifest).write_text(json.dumps(original + generated, indent=2), encoding="utf-8")
    print(json.dumps({"base": len(original), "generated": len(generated),
                      "manifest": str(cache / args.output_manifest)}, indent=2))


if __name__ == "__main__":
    main()
