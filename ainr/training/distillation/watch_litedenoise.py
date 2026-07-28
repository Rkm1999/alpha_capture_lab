#!/usr/bin/env python3
"""Display live LiteDenoise training metrics and ETA."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


def duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path(__file__).with_name("runs") / "litedenoise_high_noise_v2")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--epoch-seconds", type=float,
                        help="Measured epoch duration for a reliable ETA")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    started = args.run.stat().st_mtime if args.run.exists() else time.time()
    while True:
        history_path = args.run / "history.json"
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        now = time.time()
        if history:
            current = history[-1]
            warmup = [item for item in history if item["epoch"] <= args.warmup_epochs]
            distilled = [item for item in history if item["epoch"] > args.warmup_epochs]
            best_clean = max(warmup or history, key=lambda item: item["clean_psnr"])
            monitored_metric = "weakness_score" if "weakness_score" in current else "teacher_psnr"
            best_distilled = max(distilled or history, key=lambda item: item[monitored_metric])
            residual_bands = {
                name: current[f"{name}_similarity"]
                for name in ("fine", "medium", "coarse", "very_coarse")
                if f"{name}_similarity" in current
            }
            weakest_name, weakest_value = min(residual_bands.items(), key=lambda item: item[1]) \
                if residual_bands else ("n/a", 0.0)
            completed = int(current["epoch"])
            elapsed = now - started
            # A resumed run changes the run directory timestamp. Use the measured
            # workstation rate instead of presenting a misleadingly short ETA.
            per_epoch = args.epoch_seconds or elapsed / completed
            eta = per_epoch * (args.epochs - completed)
            lines = [
                "LiteDenoise training",
                f"Progress       {completed}/{args.epochs} epochs ({completed / args.epochs:.1%})",
                f"Clean PSNR     {current['clean_psnr']:.3f} dB",
                f"Teacher PSNR   {current['teacher_psnr']:.3f} dB",
                f"Clean L1       {current['clean_l1']:.6f}",
                *( [
                    f"Selection score {current['weakness_score']:.2%}",
                    f"Weakest band    {weakest_name} ({weakest_value:.2%})",
                    f"Fine residual  {current['fine_similarity']:.2%}",
                    f"Medium residual {current['medium_similarity']:.2%}",
                    f"Coarse residual {current['coarse_similarity']:.2%}",
                    f"V.coarse resid. {current['very_coarse_similarity']:.2%}",
                ] if monitored_metric == "weakness_score" else [] ),
                f"Learning rate  {current['learning_rate']:.8f}",
                *( [f"Best warmup    {best_clean['clean_psnr']:.3f} dB clean at epoch {best_clean['epoch']}"]
                   if warmup else [] ),
                (f"Best distilled {best_distilled['weakness_score']:.2%} selection at epoch "
                 f"{best_distilled['epoch']}" if monitored_metric == "weakness_score" else
                 f"Best distilled {best_distilled['teacher_psnr']:.3f} dB teacher at epoch "
                 f"{best_distilled['epoch']}"),
                f"Epoch time     about {duration(per_epoch)}",
                f"ETA            {duration(eta)} ({datetime.fromtimestamp(now + eta):%Y-%m-%d %H:%M:%S})",
            ]
        else:
            lines = ["LiteDenoise training", "Waiting for the first completed epoch..."]
        if not args.once:
            print("\033[2J\033[H", end="")
        print("\n".join(lines), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
