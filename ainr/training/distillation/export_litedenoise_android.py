#!/usr/bin/env python3
"""Export and numerically validate the fixed-192 LiteDenoise student."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import litert_torch as edge_torch
except ImportError:
    import ai_edge_torch as edge_torch

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

from student_litedenoise import LiteDenoiseStudent


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--iso-conditioned", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "export/litedenoise_random.tflite")
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        student_config = dict(checkpoint["config"]["student"])
        if args.iso_conditioned:
            student_config["iso_conditioned"] = True
        model = LiteDenoiseStudent(**student_config, clamp_output=False).eval()
        model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]), strict=True)
    else:
        torch.manual_seed(1337)
        model = LiteDenoiseStudent(
            base_width=args.base_width, iso_conditioned=args.iso_conditioned, clamp_output=False
        ).eval()

    channels = 4 if model.iso_conditioned else 3
    sample = torch.rand(1, channels, 192, 192, generator=torch.Generator().manual_seed(1337))
    converted = edge_torch.convert(model, (sample,))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.export(str(args.output))

    interpreter = Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    value = np.random.default_rng(1337).random((1, channels, 192, 192), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(value)).numpy()
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    actual = interpreter.get_tensor(output_detail["index"])
    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error > 1e-5:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"LiteRT validation failed: maximum error {maximum_error}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Exported {args.output} ({args.output.stat().st_size} bytes, "
        f"{parameters} parameters, max error {maximum_error:.3g})"
    )


if __name__ == "__main__":
    main()
