#!/usr/bin/env python3
"""Regenerate SCUNet labels using the exact 192 px deployment input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from compare_full_images import load_scunet


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--cache", type=Path, default=root / "data/litedenoise_cache")
    parser.add_argument("--source-manifest", default="manifest.json")
    parser.add_argument("--output-manifest", default="manifest_teacher_192.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    cache = args.cache.resolve()
    records = json.loads((cache / args.source_manifest).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to regenerate SCUNet labels")
    teacher = load_scunet(root / "data/sources/scunet",
                          root / "data/sources/scunet/model_zoo/scunet_color_real_psnr.pth", device)
    destination = cache / "teacher_192"
    destination.mkdir(parents=True, exist_ok=True)
    updated = []
    for offset in tqdm(range(0, len(records), args.batch_size), desc="SCUNet 192 labels"):
        batch_records = records[offset:offset + args.batch_size]
        sources = np.stack([
            np.load(cache / record["input"]).astype(np.float32)
            for record in batch_records
        ])
        value = torch.from_numpy(sources).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            predictions = teacher(value).clamp(0, 1).float().cpu().permute(0, 2, 3, 1).numpy()
        for index, (record, prediction) in enumerate(zip(batch_records, predictions), offset):
            path = f"teacher_192/{index:06d}.npy"
            np.save(cache / path, prediction.astype(np.float16))
            updated.append({**record, "teacher": path})
    (cache / args.output_manifest).write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(updated), "manifest": str(cache / args.output_manifest)}, indent=2))


if __name__ == "__main__":
    main()
