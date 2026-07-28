"""Manifest-backed exact-192 paired distillation dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestRecord:
    """One aligned noisy, clean, and cached-teacher sample."""

    dataset: str
    scene: str
    split: str
    input: str
    clean: str
    teacher: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ManifestRecord":
        required = ("dataset", "scene", "split", "input", "clean", "teacher")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"manifest record is missing required keys: {missing}")
        return cls(**{key: str(value[key]) for key in required})


class Sample(TypedDict):
    """Tensor sample plus identifiers retained for per-image evaluation."""

    noisy: torch.Tensor
    clean: torch.Tensor
    teacher: torch.Tensor
    index: int
    dataset: str
    scene: str
    split: str


class DistillationDataset(Dataset[Sample]):
    """Load exact 192x192 HWC NumPy triples from a JSON manifest.

    When augmentation is enabled, a deterministic transform is derived from
    ``augmentation_seed``, the current epoch, and the sample index. Call
    :meth:`set_epoch` before each training epoch. Keep DataLoader workers
    non-persistent so epoch changes are visible in worker dataset copies.
    """

    SHAPE = (192, 192, 3)

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root: str | Path | None = None,
        split: str | None = None,
        datasets: set[str] | None = None,
        augment: bool = False,
        augmentation_seed: int = 1337,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else self.manifest_path.parent
        )
        self.augment = augment
        self.augmentation_seed = augmentation_seed
        self._epoch = 0

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        raw_records = document.get("records") if isinstance(document, dict) else document
        if not isinstance(raw_records, list):
            raise ValueError("manifest must be a JSON list or an object containing a 'records' list")

        records = [ManifestRecord.from_mapping(cast(dict[str, Any], row)) for row in raw_records]
        if split is not None:
            records = [record for record in records if record.split == split]
        if datasets is not None:
            records = [record for record in records if record.dataset in datasets]
        if not records:
            raise ValueError(
                f"manifest selection is empty (split={split!r}, datasets={datasets!r})"
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic augmentation sequence for an epoch."""

        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = epoch

    def _load_array(self, relative_path: str, *, label: str) -> torch.Tensor:
        path = self.root / relative_path
        array = np.load(path, allow_pickle=False)
        if array.shape != self.SHAPE:
            raise ValueError(f"{label} array {path} has shape {array.shape}, expected {self.SHAPE}")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{label} array {path} must be floating point, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} array {path} contains non-finite values")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"{label} array {path} is outside [0, 1]: min={minimum}, max={maximum}"
            )

        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        return torch.from_numpy(contiguous).permute(2, 0, 1)

    def _augmentation(self, index: int) -> tuple[int, bool, bool]:
        generator = torch.Generator()
        generator.manual_seed(
            self.augmentation_seed + self._epoch * len(self.records) + index
        )
        rotation = int(torch.randint(0, 4, (), generator=generator).item())
        horizontal_flip = bool(torch.randint(0, 2, (), generator=generator).item())
        vertical_flip = bool(torch.randint(0, 2, (), generator=generator).item())
        return rotation, horizontal_flip, vertical_flip

    @staticmethod
    def _transform(
        value: torch.Tensor,
        *,
        rotation: int,
        horizontal_flip: bool,
        vertical_flip: bool,
    ) -> torch.Tensor:
        if rotation:
            value = torch.rot90(value, rotation, dims=(-2, -1))
        if horizontal_flip:
            value = torch.flip(value, dims=(-1,))
        if vertical_flip:
            value = torch.flip(value, dims=(-2,))
        return value.contiguous()

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        noisy = self._load_array(record.input, label="input")
        clean = self._load_array(record.clean, label="clean")
        teacher = self._load_array(record.teacher, label="teacher")

        if self.augment:
            rotation, horizontal_flip, vertical_flip = self._augmentation(index)
            transform = {
                "rotation": rotation,
                "horizontal_flip": horizontal_flip,
                "vertical_flip": vertical_flip,
            }
            noisy = self._transform(noisy, **transform)
            clean = self._transform(clean, **transform)
            teacher = self._transform(teacher, **transform)

        return {
            "noisy": noisy,
            "clean": clean,
            "teacher": teacher,
            "index": index,
            "dataset": record.dataset,
            "scene": record.scene,
            "split": record.split,
        }
