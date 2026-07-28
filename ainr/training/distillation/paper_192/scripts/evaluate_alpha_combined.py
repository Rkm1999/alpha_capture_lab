#!/usr/bin/env python3
"""Evaluate the legacy alpha=0.7 checkpoint on the 850-sample combined set.

The controlled NIND comparison already records the fixed noisy/reference/teacher
metrics. This evaluator validates and reuses those rows, then computes only the
legacy alpha=0.7 student outputs with the same metric implementation.

Run from ``paper_192`` with the training environment used for the source runs:

    /home/ryu/.cache/scunet-int8-venv/bin/python \
      scripts/evaluate_alpha_combined.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate_ablation as ablation
from common import atomic_json, load_config, resolve_paper_path, seed_everything, sha256_file
from src.dataset import DistillationDataset
from src.student import LiteDenoiseNet
from train import run_fingerprint


PAPER_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("noisy", "teacher", "alpha_0p7")
SHARED_METHODS = ("noisy", "teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs/high_iso_ablation_teacher_only.yaml",
        help="Configuration selecting the combined validation manifest and metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_ROOT / "runs/alpha_0p7/best.pt",
    )
    parser.add_argument(
        "--shared-comparison",
        type=Path,
        default=PAPER_ROOT / "evaluation/high_iso_ablation/comparison/summary.json",
        help="Audited two-arm report supplying exact noisy/teacher metric rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PAPER_ROOT
            / "evaluation/high_iso_ablation/alpha_0p7_combined_validation"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_artifact_path(value: str, report_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (report_path.parent / path).resolve()


def validate_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[LiteDenoiseNet, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.name != "best.pt" or not checkpoint_path.is_file():
        raise ValueError(f"Expected the final alpha=0.7 best.pt: {checkpoint_path}")

    run_path = checkpoint_path.parent / "run.json"
    status_path = checkpoint_path.parent / "status.json"
    history_path = checkpoint_path.parent / "history.jsonl"
    for path in (run_path, status_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint provenance: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run = read_json(run_path)
    status = read_json(status_path)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config != run.get("config"):
        raise ValueError("Checkpoint and run.json configurations differ")
    if checkpoint.get("mode") is not None:
        raise ValueError("Expected a legacy alpha checkpoint without an ablation mode")
    require_close(checkpoint.get("alpha", math.nan), 0.7, "checkpoint alpha")
    require_close(run.get("alpha", math.nan), 0.7, "run alpha")

    expected_epochs = int(run.get("epochs_effective", -1))
    if (
        status.get("state") != "complete"
        or int(status.get("epoch", -1)) != expected_epochs
        or int(status.get("epochs", -1)) != expected_epochs
        or run.get("diagnostic_limits")
        != {"max_train_batches": None, "max_val_batches": None}
    ):
        raise ValueError("Checkpoint does not belong to a complete full-length run")

    training_manifest = resolve_paper_path(config["data"]["manifest"])
    expected_fingerprint, fingerprint_payload = run_fingerprint(
        config,
        training_manifest,
        0.7,
        expected_epochs,
        None,
        None,
    )
    if (
        checkpoint.get("run_fingerprint") != expected_fingerprint
        or run.get("run_fingerprint") != expected_fingerprint
        or run.get("run_fingerprint_payload") != fingerprint_payload
    ):
        raise ValueError("Legacy alpha checkpoint fingerprint is stale or altered")

    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(history) != expected_epochs or any(
        int(record.get("epoch", -1)) != index
        for index, record in enumerate(history, start=1)
    ):
        raise ValueError("Legacy alpha history is incomplete or out of sequence")
    best_record = max(history, key=lambda record: record["validation"]["student_psnr"])
    saved_metrics = checkpoint.get("metrics", {})
    saved_psnr = saved_metrics.get("validation", {}).get("student_psnr", math.nan)
    best_psnr = checkpoint.get("best_psnr", math.nan)
    if (
        int(checkpoint.get("epoch", -1)) != int(best_record["epoch"])
        or saved_metrics.get("improved") is not True
    ):
        raise ValueError("best.pt does not identify the best history record")
    for label, value in (
        ("saved metric", saved_psnr),
        ("history best", best_record["validation"]["student_psnr"]),
        ("status best", status.get("best_psnr", math.nan)),
    ):
        require_close(value, best_psnr, label)

    model = LiteDenoiseNet().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 1_963_411:
        raise ValueError("Legacy checkpoint has the wrong student parameter count")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Legacy checkpoint contains nonfinite model parameters")
    model.to(device)

    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "alpha": float(checkpoint["alpha"]),
        "best_validation_psnr_at_save": float(best_psnr),
        "run_fingerprint": expected_fingerprint,
        "run_status": str(status["state"]),
        "status_epoch": int(status["epoch"]),
        "expected_epochs": expected_epochs,
        "training_manifest": {
            "path": str(training_manifest),
            "sha256": sha256_file(training_manifest),
        },
        "training_validation_samples": int(run["validation_samples"]),
        "training_datasets": list(config["data"]["datasets"]),
        "training_source_sha256": fingerprint_payload["source_sha256"],
        "training_script_sha256": fingerprint_payload["training_script_sha256"],
    }
    return model, metadata


def method_subset(summary: dict[str, Any], methods: tuple[str, ...]) -> dict[str, Any]:
    return {
        group: {method: value["methods"][method] for method in methods}
        for group, value in summary.items()
    }


def validate_shared_comparison(
    report_path: Path,
    manifest_path: Path,
    validation_content: dict[str, Any],
    dataset: DistillationDataset,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    report_path = report_path.expanduser().resolve()
    report = read_json(report_path)
    provenance = report.get("provenance", {})
    if (
        int(report.get("schema_version", -1)) != 2
        or provenance.get("complete_validation") is not True
        or int(provenance.get("samples", -1)) != len(dataset)
    ):
        raise ValueError("Shared comparison is not a complete schema-v2 evaluation")
    manifest = provenance.get("manifest", {})
    if (
        manifest.get("sha256") != sha256_file(manifest_path)
        or provenance.get("validation_content") != validation_content
    ):
        raise ValueError("Shared comparison uses different validation content")

    for relative, expected_hash in provenance.get("evaluator_sources", {}).items():
        source_path = PAPER_ROOT / relative
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(f"Shared comparison evaluator source changed: {relative}")

    per_image_path = resolve_artifact_path(
        report.get("artifacts", {}).get("per_image_jsonl", ""), report_path
    )
    if not per_image_path.is_file():
        raise FileNotFoundError(f"Missing shared per-image rows: {per_image_path}")
    rows = [
        json.loads(line)
        for line in per_image_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(dataset):
        raise ValueError("Shared per-image row count does not match validation data")
    for index, (row, record) in enumerate(zip(rows, dataset.records, strict=True)):
        expected = (index, record.dataset, record.scene, record.input, record.clean, record.teacher)
        actual = (
            int(row.get("index", -1)),
            row.get("dataset"),
            row.get("scene"),
            row.get("input"),
            row.get("reference"),
            row.get("teacher"),
        )
        if actual != expected:
            raise ValueError(f"Shared per-image identity mismatch at row {index}")
        if any(method not in row.get("metrics", {}) for method in SHARED_METHODS):
            raise ValueError(f"Shared fixed metrics are missing at row {index}")

    ablation.METHODS = SHARED_METHODS
    rebuilt_summary = ablation.build_summary(rows)
    rebuilt_levels = ablation.build_nind_noise_summary(rows)
    if method_subset(rebuilt_summary, SHARED_METHODS) != method_subset(
        report["summary"], SHARED_METHODS
    ):
        raise ValueError("Shared per-image rows do not reproduce the dataset summary")
    if method_subset(rebuilt_levels, SHARED_METHODS) != method_subset(
        report["nind_by_noise_level"], SHARED_METHODS
    ):
        raise ValueError("Shared per-image rows do not reproduce the NIND summary")
    return report, per_image_path, rows


def same_shadow_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in (
        "pixels",
        "fraction",
        "teacher_correction_mse",
        "correction_capture_eligible",
    ):
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, float) or isinstance(right_value, float):
            if not math.isclose(
                float(left_value), float(right_value), rel_tol=0.0, abs_tol=1e-12
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def atomic_generated_file(path: Path, writer: Callable[[Path], None]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]))
    device = ablation.choose_device(args.device)
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    dataset = DistillationDataset(
        manifest_path,
        root=cache_root,
        split="validation",
        augment=False,
    )
    raw_metadata = ablation.load_manifest_metadata(manifest_path, dataset)
    validation_content = ablation.validation_content_identity(dataset)
    shared_report, shared_rows_path, shared_rows = validate_shared_comparison(
        args.shared_comparison,
        manifest_path,
        validation_content,
        dataset,
    )
    model, checkpoint_metadata = validate_checkpoint(args.checkpoint, device)

    workers = int(config["training"]["workers"] if args.workers is None else args.workers)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    shared_metrics = shared_report["provenance"]["metrics"]
    for label, actual, expected in (
        ("border crop", border, shared_metrics["border_crop"]),
        ("SSIM window", window_size, shared_metrics["ssim_window_size"]),
        ("SSIM sigma", sigma, shared_metrics["ssim_sigma"]),
    ):
        require_close(actual, expected, label)

    rows: list[dict[str, Any]] = []
    cursor = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating alpha_0p7 on combined validation"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            reference = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            prediction = model(noisy)
            reference_pair = ablation.metric_pair(
                prediction,
                reference,
                border=border,
                window_size=window_size,
                sigma=sigma,
            )
            teacher_pair = ablation.metric_pair(
                prediction,
                teacher,
                border=border,
                window_size=window_size,
                sigma=sigma,
            )
            alpha_shadows = ablation.shadow_metrics(
                {"alpha_0p7": prediction}, noisy, teacher, border
            )
            for local_index, selection_index in enumerate(batch["index"].tolist()):
                selection_index = int(selection_index)
                if selection_index != cursor:
                    raise ValueError("Validation loader order changed")
                source = shared_rows[selection_index]
                record = dataset.records[selection_index]
                raw = raw_metadata[selection_index]
                alpha_shadow = alpha_shadows[local_index]
                if not same_shadow_identity(source["shadow"], alpha_shadow):
                    raise ValueError(
                        f"Shadow-mask identity changed at row {selection_index}"
                    )
                shadow = {
                    key: value for key, value in source["shadow"].items() if key != "methods"
                }
                shadow["methods"] = {
                    method: source["shadow"]["methods"][method]
                    for method in SHARED_METHODS
                }
                shadow["methods"]["alpha_0p7"] = alpha_shadow["methods"]["alpha_0p7"]
                rows.append(
                    {
                        "index": selection_index,
                        "id": str(raw.get("id", selection_index)),
                        "dataset": record.dataset,
                        "scene": record.scene,
                        "noise_level": str(raw.get("noise_level", "unknown")),
                        "iso": raw.get("iso"),
                        "supervision": str(raw.get("supervision", "unknown")),
                        "input": record.input,
                        "reference": record.clean,
                        "teacher": record.teacher,
                        "metrics": {
                            "noisy": source["metrics"]["noisy"],
                            "teacher": source["metrics"]["teacher"],
                            "alpha_0p7": {
                                "reference": {
                                    "psnr": ablation.finite_float(
                                        reference_pair[0][local_index]
                                    ),
                                    "ssim": ablation.finite_float(
                                        reference_pair[1][local_index]
                                    ),
                                },
                                "teacher": {
                                    "psnr": ablation.finite_float(
                                        teacher_pair[0][local_index]
                                    ),
                                    "ssim": ablation.finite_float(
                                        teacher_pair[1][local_index]
                                    ),
                                },
                            },
                        },
                        "shadow": shadow,
                    }
                )
                cursor += 1
    if cursor != len(dataset):
        raise RuntimeError(f"Processed {cursor} of {len(dataset)} validation samples")

    ablation.METHODS = METHODS
    summary = ablation.build_summary(rows)
    nind_by_noise_level = ablation.build_nind_noise_summary(rows)
    if method_subset(summary, SHARED_METHODS) != method_subset(
        shared_report["summary"], SHARED_METHODS
    ):
        raise ValueError("Generated report changed the shared dataset metrics")
    if method_subset(nind_by_noise_level, SHARED_METHODS) != method_subset(
        shared_report["nind_by_noise_level"], SHARED_METHODS
    ):
        raise ValueError("Generated report changed the shared NIND metrics")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_jsonl = output_dir / "per_image.jsonl"

    def write_jsonl(path: Path) -> None:
        with path.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    atomic_generated_file(per_image_jsonl, write_jsonl)
    atomic_generated_file(
        output_dir / "per_image.csv",
        lambda path: ablation.write_per_image_csv(path, rows),
    )
    atomic_generated_file(
        output_dir / "summary.csv",
        lambda path: ablation.write_summary_csv(path, summary),
    )
    atomic_generated_file(
        output_dir / "nind_by_noise_level.csv",
        lambda path: ablation.write_summary_csv(
            path, nind_by_noise_level, group_field="noise_level"
        ),
    )

    evaluator_sources = {
        relative: sha256_file(PAPER_ROOT / relative)
        for relative in (
            "scripts/evaluate_alpha_combined.py",
            "scripts/evaluate_ablation.py",
            "src/dataset.py",
            "src/metrics.py",
            "src/student.py",
        )
    }
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    shared_report_path = args.shared_comparison.expanduser().resolve()
    provenance = {
        "schema_version": 2,
        "purpose": (
            "cross-manifest evaluation of legacy alpha_0p7 baseline on the exact "
            "high-ISO-ablation combined validation set"
        ),
        "evaluation_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "validation_content": validation_content,
        "samples": len(rows),
        "complete_validation": True,
        "device": str(device),
        "runtime": runtime,
        "checkpoint": checkpoint_metadata,
        "evaluator_sources": evaluator_sources,
        "metrics": shared_metrics,
        "preprocessing_version": config["project"]["preprocessing_version"],
        "reference_semantics": shared_report["provenance"]["reference_semantics"],
        "comparability_note": (
            "Checkpoint was trained on the legacy MIDD+SIDD manifest; all reported "
            "metrics use the exact 850-sample high-ISO-ablation validation manifest."
        ),
        "shared_metric_source": {
            "summary_path": str(shared_report_path),
            "summary_sha256": sha256_file(shared_report_path),
            "per_image_path": str(shared_rows_path),
            "per_image_sha256": sha256_file(shared_rows_path),
            "evaluation_id": shared_report["evaluation_id"],
            "purpose": (
                "reuse exact noisy/teacher metrics and shadow-mask identities from "
                "the controlled comparison"
            ),
        },
    }
    evaluation_payload = {
        "checkpoint_sha256": checkpoint_metadata["sha256"],
        "evaluation_manifest_sha256": provenance["evaluation_manifest"]["sha256"],
        "validation_content_sha256": validation_content["sha256"],
        "shared_summary_sha256": provenance["shared_metric_source"]["summary_sha256"],
        "shared_per_image_sha256": provenance["shared_metric_source"][
            "per_image_sha256"
        ],
        "evaluator_sources": evaluator_sources,
        "metrics": shared_metrics,
        "samples": len(rows),
        "device": str(device),
        "runtime": runtime,
    }
    evaluation_id = hashlib.sha256(
        json.dumps(
            evaluation_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "provenance": provenance,
        "summary": summary,
        "nind_by_noise_level": nind_by_noise_level,
        "artifacts": {
            "per_image_jsonl": str(per_image_jsonl),
            "per_image_csv": str(output_dir / "per_image.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "nind_by_noise_level_csv": str(output_dir / "nind_by_noise_level.csv"),
        },
    }
    atomic_json(output_dir / "summary.json", report)
    print(
        json.dumps(
            {
                "evaluation_id": evaluation_id,
                "samples": len(rows),
                "device": str(device),
                "checkpoint": checkpoint_metadata,
                "all": summary["all"]["methods"]["alpha_0p7"],
                "nind": summary["nind"]["methods"]["alpha_0p7"],
                "output_dir": str(output_dir),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
