#!/usr/bin/env python3
"""Run the distilled student and SCUNet teacher on full-resolution JPEGs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps

from student import MobileUNetStudent
from student_litedenoise import LiteDenoiseStudent


def load_student(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"]["student"])
    if "base_width" in config:
        model = LiteDenoiseStudent(**config, clamp_output=False)
    else:
        config["encoder_blocks"] = tuple(config["encoder_blocks"])
        model = MobileUNetStudent(**config)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]), strict=True)
    return model.eval().to(device)


def load_scunet(repo: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    module_path = repo / "models" / "network_scunet.py"
    spec = importlib.util.spec_from_file_location("comparison_network_scunet", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SCUNet architecture from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


def iso_for(image: Image.Image, fallback: str) -> str:
    exif = image.getexif()
    iso_keys = [key for key, name in ExifTags.TAGS.items() if name in ("PhotographicSensitivity", "ISOSpeedRatings")]
    containers = [exif]
    if hasattr(ExifTags, "IFD"):
        try:
            containers.append(exif.get_ifd(ExifTags.IFD.Exif))
        except KeyError:
            pass
    for container in containers:
        for key in iso_keys:
            if key in container:
                return str(container[key])
    return fallback


def positions(length: int, core: int) -> list[int]:
    result = list(range(0, max(length - core, 0) + 1, core))
    final = max(length - core, 0)
    if not result or result[-1] != final:
        result.append(final)
    return result


def infer_tiled(
    model: torch.nn.Module,
    pixels: np.ndarray,
    device: torch.device,
    tile: int,
    padding: int,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    height, width = pixels.shape[:2]
    core = tile - padding * 2
    if core <= 0:
        raise ValueError("Tile padding must leave a positive core")
    padded = np.pad(pixels, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
    output = np.zeros_like(pixels, dtype=np.float32)
    coordinates = [(x, y) for y in positions(height, core) for x in positions(width, core)]
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[offset:offset + batch_size]
            patches = np.stack([
                padded[y:y + tile, x:x + tile] for x, y in batch_coordinates
            ])
            tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).to(device)
            prediction = model(tensor).clamp_(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            for patch, (x, y) in zip(prediction, batch_coordinates):
                copy_width = min(core, width - x)
                copy_height = min(core, height - y)
                output[y:y + copy_height, x:x + copy_width] = patch[
                    padding:padding + copy_height,
                    padding:padding + copy_width,
                ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - start, len(coordinates)


def save_rgb(pixels: np.ndarray, path: Path) -> None:
    Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8), "RGB").save(
        path, quality=96, subsampling=0
    )


def comparison_sheet(original: Image.Image, warmup: np.ndarray, student: np.ndarray, teacher: np.ndarray, iso: str, path: Path) -> None:
    width = 1600
    panel_width = width // 4
    panel_height = round(original.height * panel_width / original.width)
    header = 58
    canvas = Image.new("RGB", (width, panel_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    panels = [
        (original, "Original"),
        (Image.fromarray(np.rint(np.clip(warmup, 0, 1) * 255).astype(np.uint8)), "Warmup · Clean only"),
        (Image.fromarray(np.rint(np.clip(student, 0, 1) * 255).astype(np.uint8)), "Distilled · Current"),
        (Image.fromarray(np.rint(np.clip(teacher, 0, 1) * 255).astype(np.uint8)), "Quality · SCUNet"),
    ]
    for index, (image, label) in enumerate(panels):
        resized = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (index * panel_width, header))
        draw.text((index * panel_width + 10, 16), label, fill="black", font=font)
    draw.text((width - 120, 16), f"ISO {iso}", fill="black", font=font)
    canvas.save(path, quality=94, subsampling=0)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student", type=Path, default=root / "runs/mobile_unet_w24_detail_plateau/best.pt")
    parser.add_argument("--warmup", type=Path)
    parser.add_argument("--scunet-repo", type=Path, default=root / "data/sources/scunet")
    parser.add_argument("--scunet", type=Path, default=root / "data/sources/scunet/model_zoo/scunet_color_real_psnr.pth")
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--scunet-batch", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-resolution comparison")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = sorted(path for path in args.input.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    student = load_student(args.student, device)
    warmup = load_student(args.warmup, device) if args.warmup else student
    teacher = load_scunet(args.scunet_repo, args.scunet, device)
    report = []
    for index, path in enumerate(inputs, 1):
        encoded = Image.open(path)
        iso = iso_for(encoded, path.stem)
        original = ImageOps.exif_transpose(encoded).convert("RGB")
        pixels = np.asarray(original, dtype=np.float32) / 255.0
        print(f"[{index}/{len(inputs)}] ISO {iso}: warmup", flush=True)
        warmup_output, warmup_seconds, _ = infer_tiled(
            warmup, pixels, device, args.tile, args.padding, args.student_batch
        )
        save_rgb(warmup_output, args.output / f"ISO{iso}_warmup.jpg")
        print(f"[{index}/{len(inputs)}] ISO {iso}: distilled", flush=True)
        student_output, student_seconds, tiles = infer_tiled(
            student, pixels, device, args.tile, args.padding, args.student_batch
        )
        save_rgb(student_output, args.output / f"ISO{iso}_distilled.jpg")
        print(f"[{index}/{len(inputs)}] ISO {iso}: SCUNet", flush=True)
        teacher_output, teacher_seconds, _ = infer_tiled(
            teacher, pixels, device, args.tile, args.padding, args.scunet_batch
        )
        save_rgb(teacher_output, args.output / f"ISO{iso}_scunet.jpg")
        comparison_sheet(original, warmup_output, student_output, teacher_output, iso, args.output / f"ISO{iso}_comparison.jpg")
        report.append({
            "source": str(path), "iso": iso, "width": original.width, "height": original.height,
            "tiles": tiles, "warmup_seconds": round(warmup_seconds, 3), "distilled_seconds": round(student_seconds, 3),
            "scunet_seconds": round(teacher_seconds, 3),
        })
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(inputs)}] ISO {iso}: {student_seconds:.1f}s distilled, {teacher_seconds:.1f}s SCUNet", flush=True)


if __name__ == "__main__":
    main()
