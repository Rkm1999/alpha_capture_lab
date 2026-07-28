#!/usr/bin/env python3
"""Append PolyU real-JPEG pairs to the existing LiteDenoise cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from compare_full_images import load_scunet
from prepare_litedenoise import contextual_crop, crop_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--cache", type=Path, default=root / "data/litedenoise_cache")
    parser.add_argument("--source", type=Path, default=root / "data/sources/polyu/CroppedImages")
    parser.add_argument("--patches-per-pair", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7731)
    parser.add_argument("--output-manifest", default="manifest_polyu.json")
    args = parser.parse_args()
    cache, source = args.cache.resolve(), args.source.resolve()
    originals = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    files = {path.stem.lower(): path for path in source.glob("*.JPG")}
    pairs = []
    for key, noisy in sorted(files.items()):
        if not key.endswith("_real"):
            continue
        base = key[:-5]
        clean = files.get(base + "_mean")
        if clean:
            scene = re.sub(r"_\d+$", "", base)
            split_value = int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16) / 0xffffffff
            pairs.append(("validation" if split_value < 0.15 else "train", scene, noisy, clean))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_scunet(root / "data/sources/scunet",
                          root / "data/sources/scunet/model_zoo/scunet_color_real_psnr.pth", device)
    destination = cache / "polyu"
    destination.mkdir(parents=True, exist_ok=True)
    work = []
    for pair_index, (split, scene, noisy_path, clean_path) in enumerate(pairs):
        noisy, clean = Image.open(noisy_path).convert("RGB"), Image.open(clean_path).convert("RGB")
        if noisy.size != clean.size or min(noisy.size) < 320:
            continue
        positions = crop_positions(noisy.width, noisy.height, 192, args.patches_per_pair,
                                   args.seed ^ pair_index)
        for patch_index, (x, y) in enumerate(positions):
            work.append((split, scene, noisy_path.name, patch_index,
                         contextual_crop(noisy, x, y, 192, 320),
                         np.asarray(clean.crop((x, y, x + 192, y + 192)), np.float32) / 255.0))

    generated = []
    margin = 64
    for offset in tqdm(range(0, len(work), args.batch_size), desc="PolyU SCUNet cache"):
        batch = work[offset:offset + args.batch_size]
        value = torch.from_numpy(np.stack([item[4] for item in batch])).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            predictions = teacher(value).clamp(0, 1)[:, :, margin:margin + 192, margin:margin + 192]
        predictions = predictions.float().cpu().permute(0, 2, 3, 1).numpy()
        for index, (item, prediction) in enumerate(zip(batch, predictions), offset):
            split, scene, filename, patch_index, noisy_context, clean = item
            source_patch = noisy_context[margin:margin + 192, margin:margin + 192]
            stem = hashlib.sha256(f"{filename}:{patch_index}".encode()).hexdigest()[:16]
            paths = {name: f"polyu/{stem}_{name}.npy" for name in ("input", "teacher", "clean")}
            for name, array in (("input", source_patch), ("teacher", prediction), ("clean", clean)):
                np.save(cache / paths[name], array.astype(np.float16))
            generated.append({"dataset": "polyu", "scene": scene, "split": split, **paths})
    (cache / args.output_manifest).write_text(json.dumps(originals + generated, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "patches": len(generated),
                      "train": sum(x["split"] == "train" for x in generated),
                      "validation": sum(x["split"] == "validation" for x in generated)}, indent=2))


if __name__ == "__main__":
    main()
