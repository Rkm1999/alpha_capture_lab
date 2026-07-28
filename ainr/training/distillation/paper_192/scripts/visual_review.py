#!/usr/bin/env python3
"""Build deterministic original-pixel review sheets for restoration failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import load_config, resolve_paper_path, sha256_file
from evaluate import (
    checkpoint_identity,
    evaluation_identifier,
    ordered_checkpoint_specs,
    parse_checkpoint,
    validation_content_identity,
)
from src.dataset import DistillationDataset
from src.student import LiteDenoiseNet


def image(value: torch.Tensor) -> Image.Image:
    array = value.detach().float().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255).clip(0, 255).astype(np.uint8))


def texture_score(value: torch.Tensor) -> float:
    gray = value.mean(dim=0)
    horizontal = (gray[:, 1:] - gray[:, :-1]).abs().mean()
    vertical = (gray[1:, :] - gray[:-1, :]).abs().mean()
    return float(horizontal + vertical)


def render_sheet(
    name: str,
    indices: list[int],
    dataset: DistillationDataset,
    models: dict[str, LiteDenoiseNet],
    rows_by_index: dict[int, dict],
    destination: Path,
    device: torch.device,
) -> None:
    headings = ["noisy", *models, "SCUNet", "clean"]
    row_height = 240
    sheet = Image.new("RGB", (192 * len(headings), row_height * len(indices)), "white")
    draw = ImageDraw.Draw(sheet)
    with torch.inference_mode():
        for row_number, index in enumerate(indices):
            sample = dataset[index]
            noisy = sample["noisy"].unsqueeze(0).to(device)
            panels = [image(sample["noisy"])]
            panels.extend(image(model(noisy)[0]) for model in models.values())
            panels.extend((image(sample["teacher"]), image(sample["clean"])))
            top = row_number * row_height
            metrics = rows_by_index[index]["metrics"]
            draw.text(
                (4, top + 4),
                f"{name} | {sample['dataset']} / {sample['scene']} | noisy {metrics['noisy']['psnr']:.2f} dB",
                fill="black",
            )
            for column, (heading, panel) in enumerate(zip(headings, panels, strict=True)):
                left = column * 192
                sheet.paste(panel, (left, top + 24))
                metric_key = "teacher" if heading == "SCUNet" else heading
                metric = metrics.get(metric_key)
                suffix = f" {metric['psnr']:.2f} dB" if metric else ""
                draw.text((left + 4, top + 218), heading + suffix, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def load_bound_rows(summary_path: Path, per_image_path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation_id = summary.get("evaluation_id")
    provenance = summary.get("provenance")
    if not isinstance(evaluation_id, str) or not isinstance(provenance, dict):
        raise ValueError("Summary lacks evaluation provenance; rerun evaluate.py")
    if evaluation_identifier(provenance) != evaluation_id:
        raise ValueError("Summary evaluation_id does not match its provenance")
    if summary.get("checkpoints") != provenance.get("checkpoints"):
        raise ValueError("Summary checkpoint metadata disagrees with its provenance")

    per_image = summary.get("per_image")
    if not isinstance(per_image, dict):
        raise ValueError("Summary lacks bound per-image metadata; rerun evaluate.py")
    if per_image.get("evaluation_id") != evaluation_id:
        raise ValueError("Summary and per-image metadata have different evaluation IDs")
    recorded_path = Path(str(per_image.get("path", ""))).expanduser().resolve()
    if per_image_path != recorded_path:
        raise ValueError(
            f"Per-image path differs from summary: {per_image_path} != {recorded_path}"
        )
    if sha256_file(per_image_path) != per_image.get("sha256"):
        raise ValueError("Per-image metrics hash differs from summary")

    rows = [
        json.loads(line)
        for line in per_image_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(per_image.get("rows", -1)):
        raise ValueError("Per-image row count differs from summary")
    if any(row.get("evaluation_id") != evaluation_id for row in rows):
        raise ValueError("A per-image row is not bound to the summary evaluation ID")
    return summary, rows


def validate_rows_against_dataset(
    rows: list[dict], dataset: DistillationDataset, checkpoint_labels: set[str]
) -> dict[int, dict]:
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Per-image metrics contain duplicate validation indices")
    expected_indices = set(range(len(dataset)))
    if set(indices) != expected_indices:
        raise ValueError("Per-image metrics do not cover the current validation selection exactly")

    expected_metric_labels = {"noisy", "teacher", *checkpoint_labels}
    rows_by_index = {int(row["index"]): row for row in rows}
    for index, row in rows_by_index.items():
        record = dataset.records[index]
        expected_record = {
            "dataset": record.dataset,
            "scene": record.scene,
            "input": record.input,
            "clean": record.clean,
            "teacher": record.teacher,
        }
        if any(row.get(key) != value for key, value in expected_record.items()):
            raise ValueError(f"Per-image record {index} does not match the current manifest")
        if set(row.get("metrics", {})) != expected_metric_labels:
            raise ValueError(f"Per-image metric labels are incomplete at validation index {index}")
    return rows_by_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    config = load_config(args.config)
    manifest = resolve_paper_path(config["data"]["manifest"])
    summary, rows = load_bound_rows(args.summary.resolve(), args.per_image.resolve())
    provenance = summary["provenance"]
    manifest_identity = provenance.get("manifest", {})
    if manifest != Path(str(manifest_identity.get("path", ""))).expanduser().resolve():
        raise ValueError("Current manifest path differs from the evaluation summary")
    if sha256_file(manifest) != manifest_identity.get("sha256"):
        raise ValueError("Current manifest hash differs from the evaluation summary")
    if str(config["project"]["preprocessing_version"]) != provenance.get(
        "preprocessing_version"
    ):
        raise ValueError("Current preprocessing version differs from the evaluation summary")
    dataset = DistillationDataset(
        manifest,
        root=resolve_paper_path(config["data"]["cache_root"]),
        split="validation",
    )
    validation_identity = provenance.get("validation", {})
    if validation_identity.get("split") != "validation" or int(
        validation_identity.get("samples", -1)
    ) != len(dataset):
        raise ValueError("Current validation selection differs from the evaluation summary")
    if validation_content_identity(dataset) != provenance.get("validation_content"):
        raise ValueError("Current validation cache content differs from the evaluation summary")
    expected_checkpoints = provenance.get("checkpoints")
    if not isinstance(expected_checkpoints, dict):
        raise ValueError("Evaluation summary has no checkpoint identities")
    rows_by_index = validate_rows_against_dataset(rows, dataset, set(expected_checkpoints))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    checkpoint_specs = ordered_checkpoint_specs(args.checkpoint)
    if {label for label, _ in checkpoint_specs} != set(expected_checkpoints):
        raise ValueError("Visual review requires every checkpoint from the evaluation summary")
    for label, path in checkpoint_specs:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actual_identity = checkpoint_identity(label, path, checkpoint)
        if actual_identity != expected_checkpoints[label]:
            raise ValueError(f"Checkpoint identity differs from evaluation summary: {label}")
        model = LiteDenoiseNet().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        models[label] = model.to(device)

    content_scores = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        content_scores[index] = {
            "darkness": float(sample["noisy"].mean()),
            "texture": texture_score(sample["clean"]),
        }
    selections = {
        "highest_noise": sorted(
            rows_by_index, key=lambda i: rows_by_index[i]["metrics"]["noisy"]["psnr"]
        ),
        "darkest": sorted(content_scores, key=lambda i: content_scores[i]["darkness"]),
        "strongest_texture": sorted(
            content_scores,
            key=lambda i: content_scores[i]["texture"],
            reverse=True,
        ),
    }
    for label in models:
        def difference(index: int, current: str = label) -> float:
            return (
                rows_by_index[index]["metrics"][current]["psnr"]
                - rows_by_index[index]["metrics"]["teacher"]["psnr"]
            )

        selections[f"student_vs_teacher_worst_{label}"] = sorted(
            rows_by_index, key=difference
        )
        selections[f"student_vs_teacher_best_{label}"] = sorted(
            rows_by_index, key=difference, reverse=True
        )
    output_dir = args.output_dir.resolve()
    for name, indices in selections.items():
        render_sheet(
            name,
            indices[: args.count],
            dataset,
            models,
            rows_by_index,
            output_dir / f"{name}.png",
            device,
        )
    print(
        json.dumps(
            {
                "evaluation_id": summary["evaluation_id"],
                "sheets": {name: str(output_dir / f"{name}.png") for name in selections},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
