#!/usr/bin/env python3
"""Preserve the best raw target-score checkpoint alongside guarded selection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch


def target_score(path: Path) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {})
    validation = metrics.get("validation", {})
    target = validation.get("target_validation", {})
    return int(checkpoint["epoch"]), float(target["score"])


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def saved_candidates(run_dir: Path, *, initialize: bool) -> list[Path]:
    paths = sorted(run_dir.glob("epoch_*.pt")) if initialize else []
    last = run_dir / "last.pt"
    if last.is_file():
        paths.append(last)
    return paths


def update(run_dir: Path, best_score: float) -> float:
    candidates: list[tuple[float, int, Path]] = []
    for path in saved_candidates(run_dir, initialize=best_score == float("-inf")):
        epoch, score = target_score(path)
        candidates.append((score, epoch, path))
    if not candidates:
        return best_score
    score, epoch, source = max(candidates)
    if score <= best_score:
        return best_score
    destination = run_dir / "target-score-best.pt"
    atomic_copy(source, destination)
    atomic_json(
        run_dir / "target-score-best.json",
        {
            "checkpoint": str(destination),
            "epoch": epoch,
            "guardrails_enforced": False,
            "selection": "maximum raw target_validation.score",
            "source": str(source),
            "target_score": score,
            "updated_at": time.time(),
        },
    )
    print(
        f"target-score-best epoch={epoch} score={score:.9f} source={source.name}",
        flush=True,
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")

    metadata_path = run_dir / "target-score-best.json"
    best_score = float("-inf")
    if metadata_path.is_file():
        best_score = float(json.loads(metadata_path.read_text())["target_score"])
    while True:
        best_score = update(run_dir, best_score)
        status_path = run_dir / "status.json"
        state = (
            json.loads(status_path.read_text()).get("state")
            if status_path.is_file()
            else None
        )
        if args.once or state in {"complete", "failed", "interrupted"}:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
