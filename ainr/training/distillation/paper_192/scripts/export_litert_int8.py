#!/usr/bin/env python3
"""Export and strictly audit a fully integer LiteRT W8/A8 denoiser."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, resolve_paper_path
from src.mixed_dataset import MixedDistillationDataset
from src.noise_conditioning import model_input_from_config
from src.student import LiteDenoiseNet, checkpoint_model_kwargs


def dequantize(value: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale <= 0.0:
        raise RuntimeError(f"Tensor has invalid quantization parameters: {detail}")
    return (value.astype(np.float32) - float(zero_point)) * float(scale)


def quantize(value: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale <= 0.0:
        raise RuntimeError(f"Tensor has invalid quantization parameters: {detail}")
    return np.clip(
        np.rint(value / float(scale) + float(zero_point)),
        -128,
        127,
    ).astype(np.int8)


def representative_indices(dataset: MixedDistillationDataset, count: int) -> list[int]:
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_dataset[str(record.dataset)].append(index)
    selected: list[int] = []
    offsets = {name: 0 for name in by_dataset}
    names = sorted(by_dataset)
    while len(selected) < min(count, len(dataset)):
        progressed = False
        for name in names:
            indices = by_dataset[name]
            offset = offsets[name]
            if offset >= len(indices):
                continue
            selected.append(indices[offset])
            offsets[name] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--parity-samples", type=int, default=16)
    parser.add_argument("--maximum-mae", type=float, default=0.02)
    args = parser.parse_args()
    if args.calibration_samples < 1 or args.parity_samples < 1:
        parser.error("calibration and parity sample counts must be positive")

    try:
        import litert_torch
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        from ai_edge_quantizer import calibrator, quantizer, recipe
        from ai_edge_quantizer.utils import qsv_utils
    except ImportError as error:
        raise RuntimeError(
            "INT8 export requires litert_torch, ai_edge_litert, and "
            "ai_edge_quantizer"
        ) from error

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain its training configuration")
    model_kwargs = checkpoint_model_kwargs(checkpoint)
    if not model_kwargs.get("precomputed_noise_gate", False):
        raise ValueError(
            "Fully integer export requires a precomputed noise gate checkpoint"
        )
    # Keep clamping in the app compositor. An in-graph clamp causes LiteRT to
    # expose a float output boundary; the anchored calibration below preserves
    # the valid RGB range while retaining fully INT8 model I/O.
    model = LiteDenoiseNet(**model_kwargs, clamp_output=False).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    deployed = litert_torch.to_channel_last_io(
        model, args=[0], outputs=[0]).eval()

    manifest = resolve_paper_path(config["data"]["manifest"])
    dataset = MixedDistillationDataset(
        manifest,
        root=resolve_paper_path(config["data"]["cache_root"]),
        split="validation",
        augment=False,
    )
    indices = representative_indices(dataset, args.calibration_samples)
    calibration_values: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for position, index in enumerate(indices):
        noisy = dataset[index]["noisy"].unsqueeze(0)
        conditioned = model_input_from_config(noisy, config["model"])
        value = conditioned.permute(0, 2, 3, 1).contiguous()
        calibration_values.append(value.numpy())
        if position < args.parity_samples:
            with torch.inference_mode():
                references.append(deployed(value).numpy())

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ainr-int8-") as temporary:
        float_path = Path(temporary) / "float.tflite"
        sample = torch.from_numpy(calibration_values[0])
        edge_model = litert_torch.convert(deployed, (sample,))
        edge_model.export(str(float_path))

        # Real validation patches may not contain exact black and white. Anchor
        # both ends so integer I/O never clips valid RGB or conditioning maps.
        calibration_inputs = [
            np.zeros_like(calibration_values[0]),
            np.ones_like(calibration_values[0]),
            *calibration_values,
        ]
        calibration = {
            "serving_default": [
                {"args_0": value} for value in calibration_inputs
            ]
        }
        edge_quantizer = quantizer.Quantizer(
            float_path,
            recipe.static_wi8_ai8(),
        )
        calibration_runner = calibrator.Calibrator(
            str(float_path),
            qsv_update_func=qsv_utils.min_max_update,
        )
        calibration_runner.calibrate(
            calibration,
            edge_quantizer._recipe_manager,
        )
        calibration_result = calibration_runner.get_model_qsvs()
        result = edge_quantizer.quantize(calibration_result)
        result.export_model(output, overwrite=True)

    # XNNPACK cannot currently prepare this fully integer graph on desktop.
    # The builtin kernels provide deterministic export parity validation while
    # mobile delegate coverage is verified separately on the target device.
    interpreter = Interpreter(
        model_path=str(output),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"INT8 export produced non-INT8 I/O: "
            f"{input_detail['dtype']}, {output_detail['dtype']}"
        )

    tensor_details = interpreter.get_tensor_details()
    tensor_by_index = {detail["index"]: detail for detail in tensor_details}
    float_tensors = [
        detail["name"]
        for detail in tensor_details
        if detail["dtype"] in (np.float16, np.float32, np.float64)
    ]
    operations = interpreter._get_ops_details()
    dequantize_count = sum(
        operation["op_name"] == "DEQUANTIZE" for operation in operations
    )
    convolution_count = 0
    invalid_convolutions = []
    for operation in operations:
        if operation["op_name"] not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            continue
        convolution_count += 1
        input_dtypes = [
            tensor_by_index[index]["dtype"]
            for index in operation["inputs"]
            if index >= 0
        ]
        output_dtypes = [
            tensor_by_index[index]["dtype"]
            for index in operation["outputs"]
            if index >= 0
        ]
        if (
            len(input_dtypes) < 3
            or input_dtypes[0] != np.int8
            or input_dtypes[1] != np.int8
            or input_dtypes[2] != np.int32
            or output_dtypes != [np.int8]
        ):
            invalid_convolutions.append(
                {
                    "operation": operation["op_name"],
                    "inputs": [str(value) for value in input_dtypes],
                    "outputs": [str(value) for value in output_dtypes],
                }
            )
    if float_tensors or dequantize_count or invalid_convolutions:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "Export is not fully integer: "
            f"float_tensors={float_tensors[:5]}, "
            f"dequantize_ops={dequantize_count}, "
            f"invalid_convolutions={invalid_convolutions[:3]}"
        )

    absolute_errors = []
    for value, reference in zip(
        calibration_values[: len(references)],
        references,
        strict=True,
    ):
        interpreter.set_tensor(
            input_detail["index"],
            quantize(value, input_detail),
        )
        interpreter.invoke()
        converted = dequantize(
            interpreter.get_tensor(output_detail["index"]),
            output_detail,
        )
        absolute_errors.append(float(np.mean(np.abs(reference - converted))))
    maximum_mae = max(absolute_errors)
    if maximum_mae > args.maximum_mae:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"INT8 parity failed: maximum MAE {maximum_mae:.6f} exceeds "
            f"{args.maximum_mae:.6f}"
        )

    operator_counts: dict[str, int] = defaultdict(int)
    for operation in operations:
        operator_counts[operation["op_name"]] += 1
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": digest,
        "quantization": "static_int8_activations_per_channel_int8_weights",
        "input_dtype": "int8",
        "output_dtype": "int8",
        "input_shape": [int(value) for value in input_detail["shape"]],
        "output_shape": [int(value) for value in output_detail["shape"]],
        "input_scale": float(input_detail["quantization"][0]),
        "input_zero_point": int(input_detail["quantization"][1]),
        "output_scale": float(output_detail["quantization"][0]),
        "output_zero_point": int(output_detail["quantization"][1]),
        "float_tensor_count": 0,
        "dequantize_operation_count": 0,
        "convolution_count": convolution_count,
        "all_convolutions_integer": True,
        "calibration_samples": len(calibration_inputs),
        "parity_samples": len(references),
        "parity_mean_absolute_errors": absolute_errors,
        "parity_maximum_mean_absolute_error": maximum_mae,
        "operator_counts": dict(sorted(operator_counts.items())),
    }
    atomic_json(output.with_suffix(output.suffix + ".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
