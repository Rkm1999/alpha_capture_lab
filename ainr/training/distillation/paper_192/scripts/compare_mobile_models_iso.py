#!/usr/bin/env python3
"""Compare deployed INT8 QAT, float students, and SCUNet on the ISO set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "scripts"))

from common import atomic_json, resolve_paper_path  # noqa: E402
from src.noise_conditioning import model_input_from_config  # noqa: E402
from src.scunet_teacher import load_scunet_teacher  # noqa: E402
from src.student import student_from_checkpoint  # noqa: E402
from validate_iso_set import (  # noqa: E402
    REGIONS,
    blend_axis,
    correction_bands,
    crop_box,
    fonts,
    full_teacher_metrics,
    infer_tiled,
    iso_for,
    reflected_indices,
    save_png,
    sha256_file,
    uint8_image,
)


LABELS = {
    "original": "Original JPEG",
    "w24": "Original performance W24",
    "w32_fp16": "W32 FP16",
    "w32_qat": "W32 W8A8 QAT",
    "scunet": "Full SCUNet FP16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--w24-checkpoint", type=Path, required=True)
    parser.add_argument("--w32-checkpoint", type=Path, required=True)
    parser.add_argument("--qat-checkpoint", type=Path, required=True)
    parser.add_argument("--qat-model", type=Path, required=True)
    parser.add_argument("--student-batch", type=int, default=12)
    parser.add_argument("--teacher-batch", type=int, default=4)
    parser.add_argument("--qat-workers", type=int, default=2)
    parser.add_argument("--qat-threads", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=768)
    return parser.parse_args()


def load_images(input_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(path) as encoded:
            records.append(
                {
                    "path": path,
                    "iso": iso_for(encoded, path.stem),
                    "original": np.array(
                        ImageOps.exif_transpose(encoded).convert("RGB"), copy=True
                    ),
                }
            )
    if not records:
        raise FileNotFoundError(f"No images found in {input_dir}")
    records.sort(key=lambda item: int(item["iso"]))
    return records


def tiled_geometry(source: np.ndarray) -> dict[str, Any]:
    tile = 192
    padding = 8
    core = tile - 2 * padding
    height, width = source.shape[:2]
    columns = math.ceil(width / core)
    rows = math.ceil(height / core)
    return {
        "tile": tile,
        "padding": padding,
        "core": core,
        "height": height,
        "width": width,
        "columns": columns,
        "rows": rows,
        "coordinates": [
            (column, row) for row in range(rows) for column in range(columns)
        ],
        "x_indices": [
            reflected_indices(column * core - padding, tile, width)
            for column in range(columns)
        ],
        "y_indices": [
            reflected_indices(row * core - padding, tile, height)
            for row in range(rows)
        ],
        "x_weights": [
            blend_axis(column, columns, tile, 2 * padding)
            for column in range(columns)
        ],
        "y_weights": [
            blend_axis(row, rows, tile, 2 * padding)
            for row in range(rows)
        ],
    }


def prepare_qat_inputs(
    source: np.ndarray,
    geometry: dict[str, Any],
    config: dict[str, Any],
    input_detail: dict[str, Any],
    batch_size: int = 16,
) -> np.ndarray:
    values = []
    coordinates = geometry["coordinates"]
    for offset in range(0, len(coordinates), batch_size):
        selection = coordinates[offset : offset + batch_size]
        patches = np.stack(
            [
                source[
                    np.ix_(
                        geometry["y_indices"][row],
                        geometry["x_indices"][column],
                    )
                ]
                for column, row in selection
            ]
        )
        rgb = (
            torch.from_numpy(patches)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        conditioned = model_input_from_config(rgb, config)
        values.append(
            conditioned.permute(0, 2, 3, 1).contiguous().numpy()
        )
    value = np.concatenate(values, axis=0)
    scale, zero_point = input_detail["quantization"]
    return np.clip(
        np.rint(value / float(scale) + float(zero_point)),
        -128,
        127,
    ).astype(np.int8)


def infer_qat_tiled(
    model_path: Path,
    source: np.ndarray,
    config: dict[str, Any],
    workers: int,
    threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    geometry = tiled_geometry(source)
    resolver = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES

    def create_interpreter() -> Any:
        interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=threads,
            experimental_op_resolver_type=resolver,
        )
        interpreter.allocate_tensors()
        return interpreter

    probe = create_interpreter()
    input_detail = probe.get_input_details()[0]
    output_detail = probe.get_output_details()[0]
    inputs = prepare_qat_inputs(
        source, geometry, config, input_detail
    )
    output_scale, output_zero_point = output_detail["quantization"]
    local = threading.local()

    def run(index: int) -> tuple[int, np.ndarray]:
        if not hasattr(local, "interpreter"):
            local.interpreter = create_interpreter()
            local.input_detail = local.interpreter.get_input_details()[0]
            local.output_detail = local.interpreter.get_output_details()[0]
        local.interpreter.set_tensor(local.input_detail["index"], inputs[index : index + 1])
        local.interpreter.invoke()
        quantized = local.interpreter.get_tensor(local.output_detail["index"])[0]
        output = (
            quantized.astype(np.float32) - float(output_zero_point)
        ) * float(output_scale)
        return index, np.clip(output, 0.0, 1.0)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        predictions = list(executor.map(run, range(len(inputs))))
    accumulation = np.zeros(
        (geometry["height"], geometry["width"], 3), dtype=np.float32
    )
    weight_sum = np.zeros(
        (geometry["height"], geometry["width"]), dtype=np.float32
    )
    for index, prediction in predictions:
        column, row = geometry["coordinates"][index]
        start_x = column * geometry["core"] - geometry["padding"]
        start_y = row * geometry["core"] - geometry["padding"]
        local_x0 = max(0, -start_x)
        local_y0 = max(0, -start_y)
        local_x1 = min(geometry["tile"], geometry["width"] - start_x)
        local_y1 = min(geometry["tile"], geometry["height"] - start_y)
        image_x0 = start_x + local_x0
        image_y0 = start_y + local_y0
        image_x1 = start_x + local_x1
        image_y1 = start_y + local_y1
        weight = np.outer(
            geometry["y_weights"][row], geometry["x_weights"][column]
        )[local_y0:local_y1, local_x0:local_x1]
        accumulation[image_y0:image_y1, image_x0:image_x1] += (
            prediction[local_y0:local_y1, local_x0:local_x1] * weight[..., None]
        )
        weight_sum[image_y0:image_y1, image_x0:image_x1] += weight
    accumulation /= weight_sum[..., None]
    return accumulation, {
        "seconds": round(time.perf_counter() - started, 3),
        "tiles": len(inputs),
        "workers": workers,
        "threads_per_worker": threads,
        "input_scale": float(input_detail["quantization"][0]),
        "input_zero_point": int(input_detail["quantization"][1]),
        "output_scale": float(output_detail["quantization"][0]),
        "output_zero_point": int(output_detail["quantization"][1]),
    }


def pair_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = first.astype(np.float32) - second.astype(np.float32)
    mae = float(np.abs(difference).mean()) / 255.0
    mse = float(np.square(difference).mean()) / (255.0 * 255.0)
    return {
        "mae": mae,
        "psnr": float("inf") if mse == 0.0 else -10.0 * math.log10(mse),
    }


def make_iso_sheet(
    outputs: dict[str, np.ndarray],
    iso: int | str,
    output: Path,
    crop_size: int,
) -> None:
    title_font, body_font = fonts()
    columns = list(LABELS)
    panel = 320
    label_height = 54
    regions = [REGIONS[index] for index in (0, 1, 4, 6, 7, 10)]
    canvas = Image.new(
        "RGB",
        (panel * len(columns), label_height + panel * len(regions)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, key in enumerate(columns):
        draw.text(
            (column * panel + 8, 12),
            LABELS[key],
            fill="black",
            font=body_font,
        )
    draw.text((canvas.width - 150, 12), f"ISO {iso}", fill="black", font=title_font)
    height, width = outputs["original"].shape[:2]
    for row, region in enumerate(regions):
        box = crop_box(region, width, height, crop_size)
        for column, key in enumerate(columns):
            crop = Image.fromarray(outputs[key][box[1] : box[3], box[0] : box[2]])
            crop = crop.resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(crop, (column * panel, label_height + row * panel))
        draw.rectangle(
            (0, label_height + row * panel, 235, label_height + row * panel + 30),
            fill="white",
        )
        draw.text(
            (6, label_height + row * panel + 4),
            region.label,
            fill="black",
            font=body_font,
        )
    canvas.save(output, quality=96, subsampling=0)


def make_cross_iso_sheet(
    records: list[dict[str, Any]],
    region_index: int,
    output: Path,
    crop_size: int,
) -> None:
    title_font, body_font = fonts()
    region = REGIONS[region_index]
    keys = list(LABELS)
    panel = 280
    row_label = 210
    header = 74
    canvas = Image.new(
        "RGB",
        (
            row_label + panel * len(records),
            header + panel * len(keys),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), region.label, fill="black", font=title_font)
    for column, record in enumerate(records):
        draw.text(
            (row_label + column * panel + 10, 42),
            f"ISO {record['iso']}",
            fill="black",
            font=body_font,
        )
        height, width = record["outputs"]["original"].shape[:2]
        box = crop_box(region, width, height, crop_size)
        for row, key in enumerate(keys):
            pixels = record["outputs"][key][box[1] : box[3], box[0] : box[2]]
            crop = Image.fromarray(pixels).resize(
                (panel, panel), Image.Resampling.LANCZOS
            )
            canvas.paste(crop, (row_label + column * panel, header + row * panel))
            if column == 0:
                draw.text(
                    (8, header + row * panel + 12),
                    LABELS[key],
                    fill="black",
                    font=body_font,
                )
    canvas.save(output, quality=96, subsampling=0)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_images(args.input.expanduser().resolve())
    checkpoint_paths = {
        "w24": args.w24_checkpoint.expanduser().resolve(),
        "w32_fp16": args.w32_checkpoint.expanduser().resolve(),
        "w32_qat": args.qat_checkpoint.expanduser().resolve(),
    }
    checkpoints = {
        key: torch.load(path, map_location="cpu", weights_only=False)
        for key, path in checkpoint_paths.items()
    }
    device = torch.device("cuda")

    for record in records:
        iso_dir = output_dir / f"ISO{record['iso']}_{record['path'].stem}"
        iso_dir.mkdir(parents=True, exist_ok=True)
        record["folder"] = iso_dir.name
        record["outputs"] = {"original": record["original"]}
        record["runtimes"] = {}
        save_png(record["original"], iso_dir / "original.png")

    for key in ("w24", "w32_fp16"):
        model = student_from_checkpoint(checkpoints[key]).to(device).eval()
        config = checkpoints[key]["config"]["model"]
        for record in records:
            print(f"ISO {record['iso']}: {LABELS[key]}", flush=True)
            value, runtime = infer_tiled(
                model,
                record["original"],
                device,
                args.student_batch,
                LABELS[key],
                config,
            )
            output = uint8_image(value)
            record["outputs"][key] = output
            record["runtimes"][key] = runtime
            save_png(
                output,
                output_dir / record["folder"] / f"{key}.png",
            )
        del model
        torch.cuda.empty_cache()

    qat_model = args.qat_model.expanduser().resolve()
    qat_config = checkpoints["w32_qat"]["config"]["model"]
    for record in records:
        print(f"ISO {record['iso']}: {LABELS['w32_qat']}", flush=True)
        value, runtime = infer_qat_tiled(
            qat_model,
            record["original"],
            qat_config,
            args.qat_workers,
            args.qat_threads,
        )
        output = uint8_image(value)
        record["outputs"]["w32_qat"] = output
        record["runtimes"]["w32_qat"] = runtime
        save_png(
            output,
            output_dir / record["folder"] / "w32_qat.png",
        )

    teacher_config = checkpoints["w32_fp16"]["config"]["teacher"]
    teacher = load_scunet_teacher(
        resolve_paper_path(teacher_config["repository"]),
        resolve_paper_path(teacher_config["checkpoint"]),
        device,
    )
    for record in records:
        print(f"ISO {record['iso']}: {LABELS['scunet']}", flush=True)
        value, runtime = infer_tiled(
            teacher,
            record["original"],
            device,
            args.teacher_batch,
            LABELS["scunet"],
        )
        output = uint8_image(value)
        record["outputs"]["scunet"] = output
        record["runtimes"]["scunet"] = runtime
        save_png(
            output,
            output_dir / record["folder"] / "scunet.png",
        )
    del teacher
    torch.cuda.empty_cache()

    report_records = []
    for record in records:
        outputs = record["outputs"]
        teacher_output = outputs["scunet"]
        metrics = {}
        for key in ("w24", "w32_fp16", "w32_qat"):
            metrics[key] = {
                "agreement_to_scunet": full_teacher_metrics(
                    outputs[key], teacher_output
                ),
                "correction_by_input_luma": correction_bands(
                    outputs["original"], outputs[key], teacher_output
                ),
            }
        metrics["qat_to_fp16"] = pair_metrics(
            outputs["w32_qat"], outputs["w32_fp16"]
        )
        make_iso_sheet(
            outputs,
            record["iso"],
            output_dir / record["folder"] / "detail_comparison.jpg",
            args.crop_size,
        )
        report_records.append(
            {
                "source": str(record["path"]),
                "source_sha256": sha256_file(record["path"]),
                "iso": record["iso"],
                "width": int(outputs["original"].shape[1]),
                "height": int(outputs["original"].shape[0]),
                "folder": record["folder"],
                "runtimes": record["runtimes"],
                "metrics": metrics,
            }
        )

    region_dir = output_dir / "by_region"
    region_dir.mkdir(exist_ok=True)
    for index, region in enumerate(REGIONS):
        make_cross_iso_sheet(
            records,
            index,
            region_dir / f"{index + 1:02d}_{region.slug}.jpg",
            args.crop_size,
        )

    summary = {}
    for key in ("w24", "w32_fp16", "w32_qat"):
        summary[key] = {
            "mean_psnr_to_scunet": mean(
                [
                    float(record["metrics"][key]["agreement_to_scunet"]["psnr"])
                    for record in report_records
                ]
            ),
            "mean_preview_ssim_to_scunet": mean(
                [
                    float(
                        record["metrics"][key]["agreement_to_scunet"][
                            "ssim_preview"
                        ]
                    )
                    for record in report_records
                ]
            ),
        }
    summary["qat_to_fp16"] = {
        "mean_psnr": mean(
            [
                float(record["metrics"]["qat_to_fp16"]["psnr"])
                for record in report_records
            ]
        ),
        "mean_mae": mean(
            [
                float(record["metrics"]["qat_to_fp16"]["mae"])
                for record in report_records
            ]
        ),
    }
    report = {
        "description": (
            "No-reference ISO comparison. PSNR/SSIM measure agreement with "
            "SCUNet, not restoration accuracy."
        ),
        "models": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in checkpoint_paths.items()
        }
        | {
            "w32_qat_litert": {
                "path": str(qat_model),
                "sha256": sha256_file(qat_model),
            }
        },
        "summary": summary,
        "images": report_records,
    }
    atomic_json(output_dir / "report.json", report)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# S25 model quality comparison",
                "",
                report["description"],
                "",
                "Open each ISO folder's `detail_comparison.jpg` or the "
                "`by_region` sheets for matched 100% crops.",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
