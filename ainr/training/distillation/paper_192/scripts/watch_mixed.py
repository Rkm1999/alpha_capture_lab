#!/usr/bin/env python3
"""Print compact progress for one mixed-domain training run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import resolve_paper_path


def metric_text(status: dict) -> str:
    validation = status.get("last_validation", {})
    datasets = validation.get("by_dataset", {})
    parts = []
    for name in validation.get("selection_datasets", []):
        value = datasets.get(name, {}).get("student_psnr")
        parts.append(f"{name}={value:.3f}" if isinstance(value, (int, float)) else f"{name}=n/a")
    return " ".join(parts)


def display(run_dir: Path) -> None:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        print(f"{run_dir.name}: waiting for status.json")
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    state = status.get("state", "unknown")
    epoch = status.get("epoch", 0)
    epochs = status.get("epochs", "?")
    line = f"{status.get('run_name', run_dir.name)}: {state} epoch {epoch}/{epochs}"
    best = status.get("best_selection_psnr")
    if isinstance(best, (int, float)):
        line += f" best={best:.4f} dB"
    print(line)
    metrics = metric_text(status)
    if metrics:
        print(f"validation: {metrics}")
    if status.get("error"):
        print(f"error: {status['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=resolve_paper_path("runs/domain_expansion/uhd_snic_alpha_0p7"),
    )
    parser.add_argument("--interval", type=float)
    args = parser.parse_args()
    while True:
        display(args.run_dir.resolve())
        if args.interval is None:
            return
        print()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
