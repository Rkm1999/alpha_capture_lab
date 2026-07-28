#!/usr/bin/env python3
"""Select high-noise dark MIDD crops as a teacher-only tuning domain."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import atomic_json


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return ranks


def crop_score(root: Path, record: dict[str, Any], threshold: float) -> tuple[float, float, float]:
    noisy = np.load(root / record["input"], allow_pickle=False).astype(np.float32)
    clean = np.load(root / record["clean"], allow_pickle=False).astype(np.float32)
    teacher = np.load(root / record["teacher"], allow_pickle=False).astype(np.float32)
    luma = noisy @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    shadow = luma < threshold
    shadow_fraction = float(shadow.mean())
    if not shadow.any():
        return shadow_fraction, 0.0, 0.0
    residual = noisy - clean
    local = 0.25 * (
        np.roll(residual, 1, axis=0)
        + np.roll(residual, -1, axis=0)
        + np.roll(residual, 1, axis=1)
        + np.roll(residual, -1, axis=1)
    )
    high_frequency_noise = float(np.abs(residual - local)[shadow].mean())
    teacher_correction = float(np.abs(teacher - noisy)[shadow].mean())
    return shadow_fraction, high_frequency_noise, teacher_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--shadow-threshold", type=float, default=0.25)
    parser.add_argument("--minimum-shadow-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0,1]")

    manifest_path = args.manifest.expanduser().resolve()
    root = args.cache_root.expanduser().resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document["records"]
    darkest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record["dataset"] != "midd" or record["supervision"] != "paired":
            continue
        key = (record["split"], record["scene"])
        current = darkest.get(key)
        if current is None or float(record["crop_mean_luminance"]) < float(
            current["crop_mean_luminance"]
        ):
            darkest[key] = record

    candidates: list[dict[str, Any]] = []
    for record in darkest.values():
        shadow_fraction, noise, correction = crop_score(
            root, record, args.shadow_threshold
        )
        if shadow_fraction < args.minimum_shadow_fraction:
            continue
        candidates.append(
            {
                "record": record,
                "shadow_fraction": shadow_fraction,
                "shadow_high_frequency_noise": noise,
                "shadow_teacher_correction": correction,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for candidate in candidates:
        record = candidate["record"]
        grouped[(record["split"], record["camera"])].append(candidate)

    selected: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = collections.Counter()
    for (split, camera), values in sorted(grouped.items()):
        noise = np.asarray(
            [value["shadow_high_frequency_noise"] for value in values], dtype=np.float64
        )
        correction = np.asarray(
            [value["shadow_teacher_correction"] for value in values], dtype=np.float64
        )
        shadow = np.asarray(
            [value["shadow_fraction"] for value in values], dtype=np.float64
        )
        score = (
            0.55 * percentile_ranks(noise)
            + 0.35 * percentile_ranks(correction)
            + 0.10 * percentile_ranks(shadow)
        )
        count = max(1, math.ceil(len(values) * args.fraction))
        chosen = np.argsort(score, kind="stable")[-count:]
        camera_weight = 1.0 / len(
            {key_camera for key_split, key_camera in grouped if key_split == split}
        )
        per_record_weight = camera_weight / count
        for index in chosen:
            value = values[int(index)]
            source = value.pop("record")
            duplicate = {
                **source,
                "id": f"shadow_{source['id']}",
                "dataset": "midd_shadow",
                "supervision": "teacher_only_midd_shadow_proxy",
                "gt_weight": 0.0,
                "kd_weight": 1.0,
                "sample_weight": per_record_weight,
                "shadow_selection": {
                    **value,
                    "score": float(score[int(index)]),
                    "camera_ranked": True,
                },
            }
            selected.append(duplicate)
            selection_counts[f"{split}/{camera}"] += 1

    combined = [*records, *selected]
    combined.sort(
        key=lambda row: (
            row["split"], row["dataset"], row.get("camera", ""), row["id"]
        )
    )
    output_document = {
        **document,
        "records": combined,
        "shadow_domain": {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "selection": (
                "darkest paired crop per MIDD scene; top per-camera fraction by "
                "55% shadow high-frequency noisy/clean residual, 35% SCUNet shadow "
                "correction, 10% shadow coverage"
            ),
            "fraction": args.fraction,
            "minimum_shadow_fraction": args.minimum_shadow_fraction,
            "teacher_only": True,
            "records": len(selected),
            "counts": dict(sorted(selection_counts.items())),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, output_document)
    counts = collections.Counter(
        (record["split"], record["dataset"]) for record in combined
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": len(selected),
                "counts": {
                    f"{split}/{dataset}": count
                    for (split, dataset), count in sorted(counts.items())
                },
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
