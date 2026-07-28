#!/usr/bin/env python3
"""Finalize Core ML W8A8 QAT and export an audited iOS 17 ML Program."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PAPER_ROOT / "scripts"))

from common import atomic_json, resolve_paper_path  # noqa: E402
from src.mixed_dataset import MixedDistillationDataset  # noqa: E402
from src.noise_conditioning import model_input_from_config  # noqa: E402
from src.student import LiteDenoiseNet, checkpoint_model_kwargs  # noqa: E402
from train_mixed import prepare_coreml_int8_qat  # noqa: E402


def package_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative)
        digest.update(content)
        total_bytes += len(content)
    return digest.hexdigest(), total_bytes


def operation_counts(model: ct.models.MLModel) -> dict[str, int]:
    counts: dict[str, int] = {}
    program = model.get_spec().mlProgram
    for function in program.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                counts[operation.type] = counts.get(operation.type, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--maximum-mae", type=float, default=0.02)
    args = parser.parse_args()
    if args.parity_samples < 1:
        parser.error("--parity-samples must be positive")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    qat_state = checkpoint.get("qat_model")
    if not isinstance(config, dict) or not isinstance(qat_state, dict):
        raise ValueError("Checkpoint does not contain Core ML QAT state")
    qat_config = config["training"].get("quantization_aware_training", {})
    if qat_config.get("backend") != "coreml_fx":
        raise ValueError("Checkpoint is not a Core ML QAT run")

    kwargs = checkpoint_model_kwargs(checkpoint)
    model = LiteDenoiseNet(**kwargs, clamp_output=False).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    prepared, quantizer = prepare_coreml_int8_qat(
        model,
        input_channels=int(kwargs["input_channels"]),
        observer_freeze_step=1,
    )
    prepared.load_state_dict(qat_state, strict=True)
    prepared.eval()

    dataset = MixedDistillationDataset(
        resolve_paper_path(config["data"]["manifest"]),
        root=resolve_paper_path(config["data"]["cache_root"]),
        split="validation",
        augment=False,
    )
    values = []
    references = []
    with torch.inference_mode():
        for index in range(min(args.parity_samples, len(dataset))):
            noisy = dataset[index]["noisy"].unsqueeze(0)
            value = model_input_from_config(noisy, config["model"])
            values.append(value)
            references.append(prepared(value).numpy())

    finalized = quantizer.finalize(inplace=False).eval()
    absolute_errors = []
    with torch.inference_mode():
        for value, reference in zip(values, references, strict=True):
            converted = finalized(value).numpy()
            absolute_errors.append(
                float(np.mean(np.abs(reference - converted)))
            )
    maximum_mae = max(absolute_errors)
    if maximum_mae > args.maximum_mae:
        raise RuntimeError(
            f"Core ML QAT parity failed: {maximum_mae:.6f} exceeds "
            f"{args.maximum_mae:.6f}"
        )

    sample = values[0]
    traced = torch.jit.trace(finalized, sample)
    converted_model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(
                name="input",
                shape=sample.shape,
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
    )
    counts = operation_counts(converted_model)
    quantize_count = counts.get("quantize", 0)
    dequantize_count = counts.get("dequantize", 0)
    compressed_weight_count = sum(
        count
        for name, count in counts.items()
        if name.startswith("constexpr_") and "dequantize" in name
    )
    if not quantize_count or not dequantize_count or not compressed_weight_count:
        raise RuntimeError(
            "Core ML export is not W8A8: "
            f"quantize={quantize_count}, dequantize={dequantize_count}, "
            f"compressed_weights={compressed_weight_count}"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted_model.save(str(output))
    digest, total_bytes = package_digest(output)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "output": str(output),
        "bytes": total_bytes,
        "sha256": digest,
        "minimum_deployment_target": "iOS17",
        "quantization": "coreml_static_per_tensor_a8_per_channel_w8_qat",
        "weight_dtype": "int8",
        "activation_dtype": "uint8",
        "quantize_operation_count": quantize_count,
        "dequantize_operation_count": dequantize_count,
        "compressed_weight_operation_count": compressed_weight_count,
        "parity_samples": len(references),
        "parity_mean_absolute_errors": absolute_errors,
        "parity_maximum_mean_absolute_error": maximum_mae,
        "operator_counts": counts,
    }
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
