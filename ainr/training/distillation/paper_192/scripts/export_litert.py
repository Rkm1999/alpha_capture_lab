#!/usr/bin/env python3
"""Export fixed-layout LiteRT and enforce random plus real-input parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, load_config, resolve_paper_path
from src.dataset import DistillationDataset
from src.mixed_dataset import MixedDistillationDataset
from src.noise_conditioning import model_input_from_config
from src.student import LiteDenoiseNet, checkpoint_model_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "configs/default.yaml"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--real-samples", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--layout", choices=("nhwc", "nchw"), default="nhwc")
    parser.add_argument(
        "--omit-output-clamp",
        action="store_true",
        help="Leave output clipping to the mobile compositor for delegate compatibility",
    )
    parser.add_argument(
        "--precomputed-noise-gate",
        action="store_true",
        help="Export conditioned models with a fifth, delegate-friendly gate plane",
    )
    parser.add_argument("--parity-threshold", type=float)
    args = parser.parse_args()

    try:
        import litert_torch
    except ImportError as error:
        raise RuntimeError(
            "litert_torch is required; run this script in the separate export environment"
        ) from error

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    config = (
        checkpoint_config
        if isinstance(checkpoint_config, dict)
        else load_config(args.config)
    )
    output = (
        args.output.resolve()
        if args.output
        else resolve_paper_path(config["outputs"]["export"]) / "scunet_student_192_float.tflite"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = checkpoint_model_kwargs(checkpoint)
    if args.precomputed_noise_gate:
        if not model_kwargs["noise_adapter_channels"]:
            raise ValueError("precomputed noise gate requires a noise-adapter checkpoint")
        model_kwargs["input_channels"] = 5
        model_kwargs["precomputed_noise_gate"] = True
    model = LiteDenoiseNet(
        **model_kwargs,
        clamp_output=not args.omit_output_clamp,
    ).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    deployed_model = (
        litert_torch.to_channel_last_io(model, args=[0], outputs=[0]).eval()
        if args.layout == "nhwc"
        else model
    )
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    input_channels = int(model_kwargs["input_channels"])
    expected_input_shape = (
        (1, 192, 192, input_channels)
        if args.layout == "nhwc"
        else (1, input_channels, 192, 192)
    )
    expected_output_shape = (
        (1, 192, 192, 3) if args.layout == "nhwc" else (1, 3, 192, 192)
    )
    sample = torch.rand(expected_input_shape, generator=generator)
    with torch.inference_mode():
        reference = deployed_model(sample).numpy()
    if reference.shape != expected_output_shape:
        raise RuntimeError(f"Unexpected PyTorch output shape: {reference.shape}")

    quant_config = None
    if args.precision == "fp16":
        from litert_torch.generative.quantize.quant_attrs import (
            Algorithm,
            Dtype,
            Granularity,
            Mode,
        )
        from litert_torch.generative.quantize.quant_recipe import (
            GenerativeQuantRecipe,
            LayerQuantRecipe,
        )
        from litert_torch.quantize.quant_config import QuantConfig

        fp16_weights = LayerQuantRecipe(
            activation_dtype=Dtype.FP32,
            weight_dtype=Dtype.FP16,
            mode=Mode.WEIGHT_ONLY,
            algorithm=Algorithm.FLOAT_CAST,
            granularity=Granularity.NONE,
        )
        quant_config = QuantConfig(
            generative_recipe=GenerativeQuantRecipe(default=fp16_weights)
        )

    edge_model = litert_torch.convert(deployed_model, (sample,), quant_config=quant_config)
    edge_model.export(str(output))

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    actual_input_shape = tuple(int(value) for value in input_detail["shape"])
    actual_input_signature = tuple(int(value) for value in input_detail["shape_signature"])
    actual_output_shape = tuple(int(value) for value in output_detail["shape"])
    actual_output_signature = tuple(int(value) for value in output_detail["shape_signature"])
    if (
        actual_input_shape != expected_input_shape
        or actual_input_signature != expected_input_shape
    ):
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unexpected LiteRT input shape/signature: "
            f"{actual_input_shape}, {actual_input_signature}"
        )
    if (
        actual_output_shape != expected_output_shape
        or actual_output_signature != expected_output_shape
    ):
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unexpected LiteRT output shape/signature: "
            f"{actual_output_shape}, {actual_output_signature}"
        )
    if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unexpected LiteRT I/O dtypes: {input_detail['dtype']}, {output_detail['dtype']}"
        )

    def run_litert(value: torch.Tensor) -> np.ndarray:
        interpreter.set_tensor(input_detail["index"], value.detach().cpu().numpy())
        interpreter.invoke()
        return np.asarray(interpreter.get_tensor(output_detail["index"]))

    converted = run_litert(sample)
    if converted.shape != expected_output_shape:
        raise RuntimeError(f"Unexpected LiteRT output shape: {converted.shape}")
    random_maximum_error = float(np.max(np.abs(reference - converted)))
    parity_threshold = (
        float(args.parity_threshold)
        if args.parity_threshold is not None
        else (1e-4 if args.precision == "fp32" else 1e-3)
    )
    if random_maximum_error >= parity_threshold:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Random-input LiteRT parity failed: {random_maximum_error}")

    manifest = resolve_paper_path(config["data"]["manifest"])
    dataset_class = (
        MixedDistillationDataset
        if config["data"].get("datasets")
        else DistillationDataset
    )
    validation = dataset_class(
        manifest,
        root=resolve_paper_path(config["data"]["cache_root"]),
        split="validation",
        augment=False,
    )
    real_errors = []
    real_sample_count = min(args.real_samples, len(validation))
    real_indices = np.linspace(
        0, len(validation) - 1, num=real_sample_count, dtype=np.int64
    )
    for index in np.unique(real_indices).tolist():
        noisy = validation[index]["noisy"].unsqueeze(0)
        value = model_input_from_config(noisy, config.get("model", {}))
        if args.precomputed_noise_gate:
            strength = value[:, 3:4]
            start = float(model_kwargs["noise_gate_start"])
            end = float(model_kwargs["noise_gate_end"])
            position = ((strength - start) / (end - start)).clamp(0.0, 1.0)
            gate = position.square() * (3.0 - 2.0 * position)
            value = torch.cat((value, gate), dim=1)
        if args.layout == "nhwc":
            value = value.permute(0, 2, 3, 1).contiguous()
        with torch.inference_mode():
            expected = deployed_model(value).numpy()
        actual = run_litert(value)
        if (
            expected.shape != expected_output_shape
            or actual.shape != expected_output_shape
        ):
            raise RuntimeError(
                f"Unexpected real-input output shapes: {expected.shape}, {actual.shape}"
            )
        real_errors.append(float(np.max(np.abs(expected - actual))))
    maximum_error = max([random_maximum_error, *real_errors])
    if maximum_error >= parity_threshold:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Real-input LiteRT parity failed: {maximum_error}")
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_alpha": float(checkpoint["alpha"]),
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": output_sha256,
        "precision": args.precision,
        "weight_storage": "float32" if args.precision == "fp32" else "float16",
        "compute_dtype": "float32",
        "output_clamp": not args.omit_output_clamp,
        "precomputed_noise_gate": args.precomputed_noise_gate,
        "input_shape": list(actual_input_shape),
        "input_shape_signature": list(actual_input_signature),
        "input_layout": args.layout.upper(),
        "output_shape": list(actual_output_shape),
        "output_shape_signature": list(actual_output_signature),
        "output_layout": args.layout.upper(),
        "random_maximum_absolute_error": random_maximum_error,
        "real_maximum_absolute_errors": real_errors,
        "maximum_absolute_error": maximum_error,
        "parity_threshold": parity_threshold,
    }
    atomic_json(output.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
