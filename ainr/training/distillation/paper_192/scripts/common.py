"""Shared configuration, provenance, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PAPER_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def resolve_paper_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PAPER_ROOT / path).resolve()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment_report(
    config: dict[str, Any],
    manifest: Path,
    *,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    teacher_checkpoint = resolve_paper_path(config["teacher"]["checkpoint"])
    teacher_repo = resolve_paper_path(config["teacher"]["repository"])
    project_repo = PAPER_ROOT.parents[3]
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": sys.platform,
        "pid": os.getpid(),
        "project_commit": git_value(project_repo, "rev-parse", "HEAD"),
        "project_dirty": bool(git_value(project_repo, "status", "--porcelain")),
        "teacher_repo_commit": git_value(teacher_repo, "rev-parse", "HEAD"),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "dataset_manifest_sha256": manifest_sha256 or sha256_file(manifest),
        "preprocessing_version": config["project"]["preprocessing_version"],
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
