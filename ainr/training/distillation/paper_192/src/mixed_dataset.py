"""Manifest dataset with explicit per-record clean and teacher loss weights."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch.utils.data import Dataset


SYNTHETIC_DATASET = "synthetic_camera_jpeg"


@dataclass(frozen=True)
class MixedManifestSnapshot:
    """One immutable manifest payload shared by every training-startup consumer."""

    path: Path
    payload: bytes
    sha256: str
    document: Any

    @classmethod
    def load(cls, path: str | Path) -> "MixedManifestSnapshot":
        resolved = Path(path).expanduser().resolve()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(resolved, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"Mixed manifest is not a regular file: {resolved}")
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise RuntimeError(
                        f"Mixed manifest was truncated while being snapshotted: {resolved}"
                    )
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                raise RuntimeError(
                    f"Mixed manifest grew while being snapshotted: {resolved}"
                )
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
                raise RuntimeError(
                    f"Mixed manifest changed while being snapshotted: {resolved}"
                )
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Mixed manifest is not valid UTF-8 JSON: {resolved}") from error
        return cls(
            path=resolved,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            document=document,
        )


def mixed_manifest_snapshot(
    value: str | Path | MixedManifestSnapshot,
) -> MixedManifestSnapshot:
    return value if isinstance(value, MixedManifestSnapshot) else MixedManifestSnapshot.load(value)


@dataclass(frozen=True)
class VerifiedArrayPin:
    descriptor: int
    sha256: str
    size: int


class VerifiedSyntheticArrayStore:
    """Process-owned pins for accepted synthetic arrays.

    Workers open the owning process's descriptor through procfs. This remains
    bound to the verified inode even if the mixed-cache path is atomically
    replaced. The exact bytes passed to NumPy are hashed again on every load,
    so an in-place write to that inode fails instead of reaching training.
    """

    READ_CHUNK_BYTES = 1024 * 1024

    def __init__(self, pins: dict[str, VerifiedArrayPin], *, owner_pid: int) -> None:
        self._pins = dict(pins)
        self.owner_pid = int(owner_pid)
        self._closed = False

    def __len__(self) -> int:
        return len(self._pins)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.getpid() != self.owner_pid:
            return
        for pin in self._pins.values():
            try:
                os.close(pin.descriptor)
            except OSError:
                pass

    def __del__(self) -> None:
        self.close()

    def _open_pin(self, pin: VerifiedArrayPin) -> int:
        if self._closed:
            raise RuntimeError("Verified synthetic array store is closed")
        if os.getpid() == self.owner_pid:
            return os.dup(pin.descriptor)
        proc_path = f"/proc/{self.owner_pid}/fd/{pin.descriptor}"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        try:
            return os.open(proc_path, flags)
        except OSError as error:
            raise RuntimeError(
                "Cannot duplicate the parent process's verified synthetic array "
                f"descriptor: {proc_path}"
            ) from error

    def load_bytes(self, relative: str, label: str) -> bytes:
        pin = self._pins.get(relative)
        if pin is None:
            raise RuntimeError(
                f"Synthetic {label} is not bound to a verified descriptor: {relative}"
            )
        descriptor = self._open_pin(pin)
        try:
            current_size = os.fstat(descriptor).st_size
            if current_size != pin.size:
                raise RuntimeError(
                    f"Pinned synthetic {label} size changed: {relative} "
                    f"({current_size} != {pin.size})"
                )
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            offset = 0
            while offset < pin.size:
                chunk = os.pread(
                    descriptor,
                    min(self.READ_CHUNK_BYTES, pin.size - offset),
                    offset,
                )
                if not chunk:
                    raise RuntimeError(
                        f"Pinned synthetic {label} was truncated while reading: {relative}"
                    )
                chunks.append(chunk)
                digest.update(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, pin.size):
                raise RuntimeError(
                    f"Pinned synthetic {label} grew while reading: {relative}"
                )
            actual_sha = digest.hexdigest()
            if actual_sha != pin.sha256:
                raise RuntimeError(
                    f"Pinned synthetic {label} changed after preflight: {relative}"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def load_array(self, relative: str, label: str) -> np.ndarray:
        payload = self.load_bytes(relative, label)
        return np.load(io.BytesIO(payload), allow_pickle=False)


@dataclass(frozen=True)
class MixedManifestRecord:
    dataset: str
    scene: str
    split: str
    input: str
    clean: str
    teacher: str
    supervision: str
    gt_weight: float
    kd_weight: float
    sample_weight: float = 1.0
    iso: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MixedManifestRecord":
        required = (
            "dataset",
            "scene",
            "split",
            "input",
            "clean",
            "teacher",
            "supervision",
            "gt_weight",
            "kd_weight",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"mixed manifest record is missing fields: {missing}")
        record = cls(
            **{key: str(value[key]) for key in required[:7]},
            gt_weight=float(value["gt_weight"]),
            kd_weight=float(value["kd_weight"]),
            sample_weight=float(value.get("sample_weight", 1.0)),
            iso=(int(value["iso"]) if value.get("iso") is not None else None),
        )
        if not 0.0 <= record.gt_weight <= 1.0:
            raise ValueError(f"gt_weight must be in [0,1], got {record.gt_weight}")
        if not 0.0 <= record.kd_weight <= 1.0:
            raise ValueError(f"kd_weight must be in [0,1], got {record.kd_weight}")
        if record.gt_weight == 0.0 and record.kd_weight == 0.0:
            raise ValueError("mixed manifest record has no clean or teacher supervision")
        if not math.isfinite(record.sample_weight) or record.sample_weight <= 0.0:
            raise ValueError(
                f"sample_weight must be finite and positive, got {record.sample_weight}"
            )
        return record


class MixedSample(TypedDict):
    noisy: torch.Tensor
    clean: torch.Tensor
    teacher: torch.Tensor
    gt_weight: float
    kd_weight: float
    sample_weight: float
    iso: int
    index: int
    dataset: str
    scene: str
    split: str
    supervision: str


class MixedDistillationDataset(Dataset[MixedSample]):
    SHAPE = (192, 192, 3)

    def __init__(
        self,
        manifest_path: str | Path | MixedManifestSnapshot,
        *,
        root: str | Path | None = None,
        split: str | None = None,
        datasets: set[str] | None = None,
        augment: bool = False,
        augmentation_seed: int = 1337,
        verified_synthetic_arrays: VerifiedSyntheticArrayStore | None = None,
    ) -> None:
        snapshot = mixed_manifest_snapshot(manifest_path)
        self.manifest_snapshot = snapshot
        self.manifest_path = snapshot.path
        self.manifest_sha256 = snapshot.sha256
        self.root = Path(root).expanduser().resolve() if root else self.manifest_path.parent
        self.augment = augment
        self.augmentation_seed = augmentation_seed
        self.verified_synthetic_arrays = verified_synthetic_arrays
        self._epoch = 0
        document = snapshot.document
        raw_records = document.get("records") if isinstance(document, dict) else document
        if not isinstance(raw_records, list):
            raise ValueError("manifest must be a list or contain records")
        records = [
            MixedManifestRecord.from_mapping(cast(dict[str, Any], row))
            for row in raw_records
        ]
        if split is not None:
            records = [record for record in records if record.split == split]
        if datasets is not None:
            records = [record for record in records if record.dataset in datasets]
        if not records:
            raise ValueError(f"mixed manifest selection is empty: split={split}, datasets={datasets}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = epoch

    def _load(self, relative: str, label: str, *, synthetic: bool) -> torch.Tensor:
        path = self.root / relative
        if synthetic and self.verified_synthetic_arrays is not None:
            value = self.verified_synthetic_arrays.load_array(relative, label)
        else:
            value = np.load(path, allow_pickle=False)
        if value.shape != self.SHAPE or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"Invalid {label} tensor {value.shape}/{value.dtype}: {path}")
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise ValueError(f"Invalid {label} range: {path}")
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)

    def _augmentation(self, index: int) -> tuple[int, bool, bool]:
        generator = torch.Generator().manual_seed(
            self.augmentation_seed + self._epoch * len(self.records) + index
        )
        return (
            int(torch.randint(0, 4, (), generator=generator).item()),
            bool(torch.randint(0, 2, (), generator=generator).item()),
            bool(torch.randint(0, 2, (), generator=generator).item()),
        )

    @staticmethod
    def _transform(value: torch.Tensor, rotation: int, horizontal: bool, vertical: bool) -> torch.Tensor:
        if rotation:
            value = torch.rot90(value, rotation, dims=(-2, -1))
        if horizontal:
            value = torch.flip(value, dims=(-1,))
        if vertical:
            value = torch.flip(value, dims=(-2,))
        return value.contiguous()

    def __getitem__(self, index: int) -> MixedSample:
        record = self.records[index]
        synthetic = record.dataset == SYNTHETIC_DATASET
        noisy = self._load(record.input, "input", synthetic=synthetic)
        clean = self._load(record.clean, "clean", synthetic=synthetic)
        teacher = self._load(record.teacher, "teacher", synthetic=synthetic)
        if self.augment:
            transform = self._augmentation(index)
            noisy = self._transform(noisy, *transform)
            clean = self._transform(clean, *transform)
            teacher = self._transform(teacher, *transform)
        return {
            "noisy": noisy,
            "clean": clean,
            "teacher": teacher,
            "gt_weight": record.gt_weight,
            "kd_weight": record.kd_weight,
            "sample_weight": record.sample_weight,
            "iso": record.iso if record.iso is not None else -1,
            "index": index,
            "dataset": record.dataset,
            "scene": record.scene,
            "split": record.split,
            "supervision": record.supervision,
        }
