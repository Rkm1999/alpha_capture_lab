#!/usr/bin/env python3
"""Print compact status for the three controlled runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def display(root: Path) -> None:
    for name in ("alpha_0p0", "alpha_0p7", "alpha_0p9"):
        path = root / name / "status.json"
        if not path.is_file():
            print(f"{name}: pending")
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        text = f"{name}: {status['state']} epoch {status.get('epoch', 0)}/{status.get('epochs', '?')}"
        if "best_psnr" in status:
            text += f" best PSNR {status['best_psnr']:.4f} dB"
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, help="Refresh continuously at this interval in seconds.")
    args = parser.parse_args()
    root = Path(__file__).parents[1] / "runs"
    while True:
        display(root)
        if args.interval is None:
            return
        print()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
