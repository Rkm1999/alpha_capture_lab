#!/usr/bin/env python3
"""Train a balanced mix of paired, photometric-paired, and teacher-only data."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import random
import stat
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import resource
except ImportError:  # pragma: no cover - synthetic pinning is Linux-only.
    resource = None  # type: ignore[assignment]

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from common import (
    atomic_json,
    environment_report,
    load_config,
    resolve_paper_path,
    seed_everything,
    seed_worker,
    sha256_file,
)
from src.mixed_dataset import (
    MixedDistillationDataset,
    MixedManifestSnapshot,
    VerifiedArrayPin,
    VerifiedSyntheticArrayStore,
    mixed_manifest_snapshot,
)
from src.metrics import gaussian_ssim_per_image, psnr_per_image
from src.noise_conditioning import (
    conditioned_input,
    estimate_noise_strength,
    training_noise_strength,
)
from src.student import LiteDenoiseNet
from src.weighted_losses import compute_weighted_distillation_loss
from src.width_expansion import expansion_gradient_masks
from train import atomic_checkpoint, restore_rng, rng_state


MIXED_ARRAY_FIELDS = ("input", "clean", "teacher")
SYNTHETIC_DATASET = "synthetic_camera_jpeg"


def state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_digest(model: torch.nn.Module) -> str:
    return state_dict_digest(model.state_dict())


def prepare_litert_int8_qat(
    model: LiteDenoiseNet,
    *,
    input_channels: int,
    maximum_batch_size: int,
) -> torch.fx.GraphModule:
    """Prepare LiteRT-compatible per-channel W8/A8 QAT for a convolutional model."""

    try:
        from litert_torch.quantize.pt2e_quantizer import (
            PT2EQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e import allow_exported_model_train_eval
        from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
    except ImportError as error:
        raise RuntimeError(
            "LiteRT INT8 QAT requires the litert_torch and torchao packages"
        ) from error

    class ConvolutionalQATQuantizer(PT2EQuantizer):
        # LiteDenoiseNet has no batch normalization. Disabling this obsolete
        # matcher avoids a LiteRT 0.9.1/PyTorch 2.12 API incompatibility.
        STATIC_QAT_ONLY_OPS: list[str] = []

    model = model.cpu()
    sample = torch.zeros(
        (
            maximum_batch_size,
            input_channels,
            LiteDenoiseNet.INPUT_SIZE,
            LiteDenoiseNet.INPUT_SIZE,
        )
    )
    exported = torch.export.export(
        model.eval(),
        (sample,),
        dynamic_shapes=(
            {
                0: torch.export.Dim(
                    "batch",
                    min=1,
                    max=maximum_batch_size,
                )
            },
        ),
    ).module()
    quantizer = ConvolutionalQATQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_qat=True,
        )
    )
    prepared = prepare_qat_pt2e(exported, quantizer)
    allow_exported_model_train_eval(prepared)
    return prepared


def prepare_coreml_int8_qat(
    model: LiteDenoiseNet,
    *,
    input_channels: int,
    observer_freeze_step: int,
) -> tuple[torch.fx.GraphModule, Any]:
    """Prepare Core ML optimized per-channel W8/per-tensor A8 QAT."""

    try:
        from coremltools.optimize.torch.quantization import (
            LinearQuantizer,
            LinearQuantizerConfig,
            ModuleLinearQuantizerConfig,
            QuantizationScheme,
        )
    except ImportError as error:
        raise RuntimeError(
            "Core ML INT8 QAT requires coremltools.optimize.torch"
        ) from error

    if observer_freeze_step < 1:
        raise ValueError("Core ML observer freeze step must be positive")
    configuration = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            weight_dtype=torch.qint8,
            activation_dtype=torch.quint8,
            weight_per_channel=True,
            quantization_scheme=QuantizationScheme.symmetric,
            milestones=[0, 0, observer_freeze_step, 0],
        )
    )
    model = model.cpu()
    example = torch.zeros(
        (1, input_channels, LiteDenoiseNet.INPUT_SIZE, LiteDenoiseNet.INPUT_SIZE)
    )
    quantizer = LinearQuantizer(model, configuration)
    prepared = quantizer.prepare(example_inputs=example, inplace=False)
    return prepared, quantizer


def float_model_state(
    model: torch.nn.Module,
    float_state_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Extract deployable float parameters from a QAT graph checkpoint."""

    state = model.state_dict()
    missing = [name for name in float_state_keys if name not in state]
    if missing:
        raise RuntimeError(f"QAT graph lost model parameters: {missing[:5]}")
    return {name: state[name] for name in float_state_keys}


def load_model_only_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load model weights and record the state intentionally not restored."""

    resolved = path.expanduser().resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Initialization checkpoint has no model state: {resolved}")
    model_state = checkpoint["model"]
    if not model_state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in model_state.items()
    ):
        raise ValueError(f"Initialization checkpoint has an invalid model state: {resolved}")
    source_epoch = checkpoint.get("epoch")
    if source_epoch is not None:
        source_epoch = int(source_epoch)
    checkpoint_config = checkpoint.get("config")
    checkpoint_model_config = (
        checkpoint_config.get("model")
        if isinstance(checkpoint_config, dict)
        else None
    )
    source_input_channels = (
        int(checkpoint_model_config["input_channels"])
        if isinstance(checkpoint_model_config, dict)
        and "input_channels" in checkpoint_model_config
        else int(model_state["input_conv.weight"].shape[1])
    )
    provenance = {
        "mode": "model_only",
        "checkpoint": str(resolved),
        "checkpoint_sha256": sha256_file(resolved),
        "source_epoch": source_epoch,
        "source_run_name": checkpoint.get("run_name"),
        "source_model_sha256": state_dict_digest(model_state),
        "source_input_channels": source_input_channels,
        "optimizer": "reset",
        "scheduler": "reset",
        "scaler": "reset",
        "rng": "reset_from_config_seed",
    }
    width_expansion = checkpoint.get("width_expansion")
    if width_expansion is None:
        prior_initialization = checkpoint.get("initialization")
        if isinstance(prior_initialization, dict):
            width_expansion = prior_initialization.get("width_expansion")
    if width_expansion is not None:
        if not isinstance(width_expansion, dict):
            raise ValueError("Initialization checkpoint has invalid width-expansion metadata")
        provenance["width_expansion"] = dict(width_expansion)
    return model_state, provenance


def load_compatible_model_state(
    model: LiteDenoiseNet,
    source: dict[str, torch.Tensor],
) -> str:
    """Load legacy RGB weights into an otherwise identical conditioned model."""

    target = model.state_dict()
    missing = sorted(set(target) - set(source))
    extra = sorted(set(source) - set(target))
    shape_mismatches = {
        name: (tuple(source[name].shape), tuple(value.shape))
        for name, value in target.items()
        if name in source and source[name].shape != value.shape
    }
    profile_input_key = "chroma_profile_head.input_projection.0.weight"
    if (
        not missing
        and not extra
        and set(shape_mismatches) == {profile_input_key}
        and target[profile_input_key].shape[0] == source[profile_input_key].shape[0]
        and target[profile_input_key].shape[1]
        == source[profile_input_key].shape[1] + 6
        and target[profile_input_key].shape[2:]
        == source[profile_input_key].shape[2:]
    ):
        expanded = dict(source)
        profile_weight = target[profile_input_key].zero_()
        profile_weight[:, : source[profile_input_key].shape[1]].copy_(
            source[profile_input_key]
        )
        expanded[profile_input_key] = profile_weight
        model.load_state_dict(expanded, strict=True)
        return "profile_head_restoration_input_zero_initialized"
    if (
        missing
        and all(
            name.startswith(
                (
                    "noise_adapter.",
                    "multiscale_adapters.",
                    "chroma_head.",
                    "global_chroma_head.",
                    "chroma_unet_head.",
                    "chroma_profile_head.",
                    "chroma_refinement_head.",
                )
            )
            for name in missing
        )
        and not shape_mismatches
        and not extra
    ):
        initialized = dict(target)
        initialized.update(source)
        model.load_state_dict(initialized, strict=True)
        if all(name.startswith("multiscale_adapters.") for name in missing):
            return "residual_adapter_to_multiscale_zero_initialized"
        if all(name.startswith("chroma_head.") for name in missing):
            return "multiscale_to_chroma_head_zero_initialized"
        if all(name.startswith("global_chroma_head.") for name in missing):
            return "chroma_head_to_global_chroma_head_zero_initialized"
        if all(name.startswith("chroma_unet_head.") for name in missing):
            return "global_to_chroma_unet_head_zero_initialized"
        if all(name.startswith("chroma_profile_head.") for name in missing):
            if all(
                name.startswith("chroma_profile_head.refinement.")
                for name in missing
            ):
                return "profile_head_refinement_zero_initialized"
            return "chroma_unet_to_profile_head_zero_initialized"
        if all(name.startswith("chroma_refinement_head.") for name in missing):
            return "chroma_unet_to_refinement_head_zero_initialized"
        return "backbone_to_residual_adapter_zero_initialized"
    expected = {
        "input_conv.weight": (
            (model.base_width, 3, 3, 3),
            (model.base_width, 4, 3, 3),
        )
    }
    if missing or shape_mismatches != expected or extra:
        model.load_state_dict(source, strict=True)
        return "exact"
    expanded = dict(source)
    input_weight = target["input_conv.weight"].zero_()
    input_weight[:, :3].copy_(source["input_conv.weight"])
    expanded["input_conv.weight"] = input_weight
    model.load_state_dict(expanded, strict=True)
    return "rgb_to_noise_conditioned_zero_initialized"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def contained_mixed_array(cache_root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Mixed {label} path must stay inside the cache: {relative}")
    root = cache_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Mixed {label} path escapes the cache: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("Array was truncated during integrity verification")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise RuntimeError("Array grew during integrity verification")
    return digest.hexdigest()


def _verify_mixed_array_integrity(
    manifest_path: Path | MixedManifestSnapshot,
    cache_root: Path,
    *,
    retain_synthetic_descriptors: bool,
) -> tuple[dict[str, Any], VerifiedSyntheticArrayStore | None]:
    """Verify accepted tensors and optionally retain their exact open inodes."""

    snapshot = mixed_manifest_snapshot(manifest_path)
    document = snapshot.document
    records = document.get("records") if isinstance(document, dict) else None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Mixed manifest must contain object records for integrity preflight")
    synthetic_by_dataset = [
        row for row in records if str(row.get("dataset")) == SYNTHETIC_DATASET
    ]
    synthetic_by_source = [
        row for row in records if str(row.get("mixed_source")) == "synthetic"
    ]
    if {id(row) for row in synthetic_by_dataset} != {id(row) for row in synthetic_by_source}:
        raise ValueError(
            "Synthetic dataset records must map exactly to mixed_source=synthetic"
        )
    if not synthetic_by_source:
        return (
            {
                "status": "not_applicable",
                "manifest_sha256": snapshot.sha256,
                "synthetic_records": 0,
                "synthetic_files_verified": 0,
                "legacy_domain_hash_guarantee": False,
            },
            None,
        )

    sources = document.get("sources")
    synthetic_source = sources.get("synthetic") if isinstance(sources, dict) else None
    acceptance = (
        synthetic_source.get("acceptance")
        if isinstance(synthetic_source, dict)
        else None
    )
    if not isinstance(acceptance, dict) or acceptance.get("status") != "accepted":
        raise RuntimeError("Mixed synthetic source has no accepted gate provenance")
    if int(synthetic_source.get("records", -1)) != len(synthetic_by_source):
        raise RuntimeError("Mixed synthetic record count differs from accepted source provenance")

    identifiers = [str(row.get("id", "")) for row in synthetic_by_source]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("Mixed synthetic records require unique non-empty IDs")

    digest = hashlib.sha256()
    bytes_total = 0
    files = 0
    mixed_paths: set[str] = set()
    retained: dict[str, VerifiedArrayPin] = {}

    def close_retained() -> None:
        for pin in retained.values():
            try:
                os.close(pin.descriptor)
            except OSError:
                pass

    if retain_synthetic_descriptors:
        if resource is None:
            raise RuntimeError(
                "Synthetic array descriptor pinning requires Linux resource limits"
            )
        proc_fd = Path(f"/proc/{os.getpid()}/fd")
        if not proc_fd.is_dir():
            raise RuntimeError(
                "Synthetic array descriptor pinning requires Linux procfs at /proc"
            )
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        current_descriptors = len(list(proc_fd.iterdir()))
        required = len(synthetic_by_source) * len(MIXED_ARRAY_FIELDS)
        reserve = 256
        if soft_limit != resource.RLIM_INFINITY and (
            current_descriptors + required + reserve > soft_limit
        ):
            raise RuntimeError(
                "Not enough file descriptors to pin accepted synthetic arrays: "
                f"open={current_descriptors}, required={required}, reserve={reserve}, "
                f"limit={soft_limit}"
            )
    try:
        for record in sorted(synthetic_by_source, key=lambda row: str(row["id"])):
            declared_hashes = record.get("array_sha256")
            accepted_paths = record.get("accepted_source_array_paths")
            if not isinstance(declared_hashes, dict) or set(declared_hashes) != set(
                MIXED_ARRAY_FIELDS
            ):
                raise RuntimeError(
                    f"Synthetic record {record['id']} lacks the complete array_sha256 contract"
                )
            if not isinstance(accepted_paths, dict) or set(accepted_paths) != set(
                MIXED_ARRAY_FIELDS
            ):
                raise RuntimeError(
                    f"Synthetic record {record['id']} lacks accepted source array paths"
                )
            for field in MIXED_ARRAY_FIELDS:
                source_relative = Path(str(accepted_paths[field]))
                if source_relative.is_absolute() or ".." in source_relative.parts:
                    raise ValueError(
                        f"Accepted synthetic {field} path is unsafe: {source_relative}"
                    )
                expected_mixed_relative = Path("synthetic") / source_relative
                mixed_relative = Path(str(record.get(field)))
                if mixed_relative != expected_mixed_relative:
                    raise RuntimeError(
                        f"Mixed synthetic {field} path lost its accepted source binding: "
                        f"{mixed_relative} != {expected_mixed_relative}"
                    )
                mixed_path_text = str(mixed_relative)
                if mixed_path_text in mixed_paths:
                    raise RuntimeError(
                        f"Duplicate mixed synthetic array path: {mixed_relative}"
                    )
                mixed_paths.add(mixed_path_text)
                path = contained_mixed_array(cache_root, mixed_relative, field)
                declared_sha = declared_hashes[field]
                if not isinstance(declared_sha, str) or len(declared_sha) != 64:
                    raise RuntimeError(
                        f"Synthetic record {record['id']} has invalid {field} SHA-256"
                    )
                try:
                    int(declared_sha, 16)
                except ValueError as error:
                    raise RuntimeError(
                        f"Synthetic record {record['id']} has non-hex {field} SHA-256"
                    ) from error
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                    os, "O_NOFOLLOW", 0
                )
                descriptor = os.open(path, flags)
                keep_descriptor = False
                try:
                    file_stat = os.fstat(descriptor)
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise RuntimeError(f"Synthetic array is not a regular file: {path}")
                    actual_sha = sha256_descriptor(descriptor, file_stat.st_size)
                    if actual_sha != declared_sha.lower():
                        raise RuntimeError(
                            "Mixed synthetic array changed after acceptance: "
                            f"{mixed_relative}"
                        )
                    if retain_synthetic_descriptors:
                        retained[mixed_path_text] = VerifiedArrayPin(
                            descriptor=descriptor,
                            sha256=actual_sha,
                            size=file_stat.st_size,
                        )
                        keep_descriptor = True
                    digest.update(
                        canonical_json(
                            [
                                str(record["id"]),
                                field,
                                str(accepted_paths[field]),
                                actual_sha,
                            ]
                        )
                    )
                    bytes_total += file_stat.st_size
                    files += 1
                finally:
                    if not keep_descriptor:
                        os.close(descriptor)
    except BaseException:
        close_retained()
        raise

    actual_identity = {
        "files": files,
        "bytes": bytes_total,
        "sha256": digest.hexdigest(),
    }
    if acceptance.get("cache_content") != actual_identity:
        close_retained()
        raise RuntimeError(
            "Mixed synthetic payload no longer matches its accepted cache identity: "
            f"accepted={acceptance.get('cache_content')!r}, actual={actual_identity!r}"
        )
    if files != len(synthetic_by_source) * len(MIXED_ARRAY_FIELDS):
        close_retained()
        raise RuntimeError("Synthetic integrity preflight did not cover every record array")
    report = {
        "status": "verified",
        "manifest_sha256": snapshot.sha256,
        "synthetic_records": len(synthetic_by_source),
        "synthetic_files_verified": files,
        "accepted_cache_content": actual_identity,
        "pinning": {
            "status": "active" if retain_synthetic_descriptors else "not_requested",
            "strategy": "parent_proc_fd_exact_payload_sha256_v1",
            "synthetic_files_pinned": len(retained),
            "exact_payload_rehash_on_load": retain_synthetic_descriptors,
        },
        "legacy_domain_hash_guarantee": False,
    }
    store = None
    if retain_synthetic_descriptors:
        store = VerifiedSyntheticArrayStore(retained, owner_pid=os.getpid())
    return report, store


def validate_mixed_array_integrity(
    manifest_path: Path | MixedManifestSnapshot, cache_root: Path
) -> dict[str, Any]:
    """Verify accepted synthetic tensors without retaining descriptor pins."""

    report, store = _verify_mixed_array_integrity(
        manifest_path,
        cache_root,
        retain_synthetic_descriptors=False,
    )
    assert store is None
    return report


def pin_verified_synthetic_arrays(
    manifest_path: Path | MixedManifestSnapshot, cache_root: Path
) -> tuple[dict[str, Any], VerifiedSyntheticArrayStore | None]:
    """Verify and pin accepted synthetic payloads before workers are created."""

    return _verify_mixed_array_integrity(
        manifest_path,
        cache_root,
        retain_synthetic_descriptors=True,
    )


def validate_training_contract(
    config: dict[str, Any],
    manifest_path: Path | MixedManifestSnapshot,
    train_records: list,
    validation_records: list,
) -> dict[str, Any]:
    """Fail before training when cache provenance or loss weights are inconsistent."""

    snapshot = mixed_manifest_snapshot(manifest_path)
    document = snapshot.document
    if not isinstance(document, dict):
        raise ValueError("mixed training manifest must include provenance metadata")
    if document.get("schema_version") != 2:
        raise ValueError(
            f"mixed training manifest must use schema_version 2, got "
            f"{document.get('schema_version')!r}"
        )
    configured_preprocessing = str(config["project"]["preprocessing_version"])
    if document.get("preprocessing") != configured_preprocessing:
        raise ValueError(
            "mixed cache preprocessing does not match the training configuration: "
            f"cache={document.get('preprocessing')!r}, configured={configured_preprocessing!r}"
        )
    expected_source_preprocessing = {
        str(namespace): str(version)
        for namespace, version in config["data"]["source_preprocessing"].items()
    }
    cached_source_preprocessing = document.get("source_preprocessing")
    if cached_source_preprocessing != expected_source_preprocessing:
        raise ValueError(
            "mixed cache source preprocessing does not match the training configuration: "
            f"cache={cached_source_preprocessing!r}, "
            f"configured={expected_source_preprocessing!r}"
        )
    cached_sources = document.get("sources")
    if not isinstance(cached_sources, dict):
        raise ValueError("mixed training manifest has no source provenance map")
    provenance_preprocessing = {
        str(namespace): metadata.get("preprocessing")
        for namespace, metadata in cached_sources.items()
        if isinstance(metadata, dict)
    }
    if provenance_preprocessing != expected_source_preprocessing:
        raise ValueError(
            "mixed cache source provenance has inconsistent preprocessing: "
            f"sources={provenance_preprocessing!r}, "
            f"configured={expected_source_preprocessing!r}"
        )
    cached_teacher_hash = document.get("teacher_checkpoint_sha256")
    if not isinstance(cached_teacher_hash, str) or len(cached_teacher_hash) != 64:
        raise ValueError("mixed training manifest has no valid teacher checkpoint hash")
    configured_teacher_hash = sha256_file(
        resolve_paper_path(config["teacher"]["checkpoint"])
    )
    if cached_teacher_hash.lower() != configured_teacher_hash.lower():
        raise ValueError(
            "cached teacher outputs do not match the configured teacher checkpoint: "
            f"cache={cached_teacher_hash}, configured={configured_teacher_hash}"
        )

    configured_list = [str(value) for value in config["data"]["datasets"]]
    if len(configured_list) != len(set(configured_list)):
        raise ValueError("data.datasets contains duplicate dataset names")
    configured_datasets = set(configured_list)
    train_datasets = {str(record.dataset) for record in train_records}
    validation_datasets = {str(record.dataset) for record in validation_records}
    for split, actual in (("train", train_datasets), ("validation", validation_datasets)):
        if actual != configured_datasets:
            raise ValueError(
                f"{split} datasets do not match data.datasets: "
                f"records={sorted(actual)}, configured={sorted(configured_datasets)}"
            )

    sampling_datasets = {
        str(value) for value in config["training"]["dataset_sampling_weights"]
    }
    if sampling_datasets != configured_datasets:
        raise ValueError(
            "dataset_sampling_weights must exactly match data.datasets: "
            f"sampling={sorted(sampling_datasets)}, configured={sorted(configured_datasets)}"
        )
    selection_list = [str(value) for value in config["training"]["selection_datasets"]]
    if not selection_list or len(selection_list) != len(set(selection_list)):
        raise ValueError("selection_datasets must be non-empty and contain no duplicates")
    selection_datasets = set(selection_list)
    if not selection_datasets <= configured_datasets:
        raise ValueError(
            f"selection_datasets contains unconfigured datasets: "
            f"{sorted(selection_datasets - configured_datasets)}"
        )

    alpha = float(config["training"]["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"training alpha must be in [0,1], got {alpha}")
    for name, default in (("mse_scale", 1000.0), ("clean_l1_lambda", 50.0)):
        value = float(config["training"].get(name, default))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"training {name} must be finite and non-negative")
    width_expansion_warmup_epochs = int(
        config["training"].get("width_expansion_warmup_epochs", 0)
    )
    if width_expansion_warmup_epochs < 0:
        raise ValueError("training width_expansion_warmup_epochs must be non-negative")
    correction_config = config["training"].get("correction_loss")
    if correction_config and bool(correction_config.get("enabled", False)):
        required_correction_fields = {
            "shadow_luminance_threshold",
            "fine_sigma",
            "medium_sigma",
            "coarse_sigma",
            "shadow_lambda",
            "medium_coarse_lambda",
        }
        missing = sorted(required_correction_fields - set(correction_config))
        if missing:
            raise ValueError(f"correction_loss is missing fields: {missing}")
        threshold = float(correction_config["shadow_luminance_threshold"])
        sigmas = tuple(
            float(correction_config[name])
            for name in ("fine_sigma", "medium_sigma", "coarse_sigma")
        )
        coefficients = tuple(
            float(correction_config[name])
            for name in ("shadow_lambda", "medium_coarse_lambda")
        )
        optional_coefficients = tuple(
            float(correction_config.get(name, 0.0))
            for name in (
                "normalized_shadow_lambda",
                "normalized_shadow_chroma_lambda",
                "shadow_medium_coarse_chroma_lambda",
                "flat_shadow_chroma_lambda",
                "very_coarse_chroma_lambda",
                "row_column_chroma_lambda",
                "normalized_very_coarse_chroma_lambda",
                "weighted_very_coarse_chroma_lambda",
                "normalized_row_column_chroma_lambda",
                "normalized_very_coarse_chroma_mse_lambda",
                "normalized_row_column_chroma_mse_lambda",
                "pyramid_chroma_lambda",
                "paired_detail_lambda",
            )
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError("correction_loss shadow threshold must be in (0,1)")
        if not 0.0 < sigmas[0] < sigmas[1] < sigmas[2]:
            raise ValueError("correction_loss sigmas must satisfy fine < medium < coarse")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in coefficients + optional_coefficients
        ):
            raise ValueError("correction_loss coefficients must be finite and non-negative")
        if "very_coarse_sigma" in correction_config and float(
            correction_config["very_coarse_sigma"]
        ) <= sigmas[-1]:
            raise ValueError(
                "correction_loss very_coarse_sigma must exceed coarse_sigma"
            )
        warmup_epochs = int(correction_config.get("warmup_epochs", 0))
        ramp_epochs = int(correction_config.get("ramp_epochs", 0))
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError(
                "correction_loss warmup_epochs and ramp_epochs must be non-negative"
            )
    paired_kd_override_value = config["training"].get("paired_kd_weight_override")
    paired_kd_override = (
        float(paired_kd_override_value)
        if paired_kd_override_value is not None
        else None
    )
    if paired_kd_override is not None and (
        not 0.0 <= paired_kd_override <= 1.0
        or not math.isclose(paired_kd_override, alpha, rel_tol=0.0, abs_tol=1e-7)
    ):
        raise ValueError(
            "paired_kd_weight_override must be in [0,1] and equal training alpha"
        )
    cached_paired_kd_weights = sorted(
        {
            float(record.kd_weight)
            for record in train_records + validation_records
            if record.gt_weight > 0.0
        }
    )
    for record in train_records + validation_records:
        expected_kd_weight = (
            record.kd_weight
            if paired_kd_override is not None and record.gt_weight > 0.0
            else (alpha if record.gt_weight > 0.0 else 1.0)
        )
        if not math.isclose(record.kd_weight, expected_kd_weight, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                "manifest loss weight conflicts with training alpha: "
                f"dataset={record.dataset}, split={record.split}, "
                f"gt_weight={record.gt_weight}, kd_weight={record.kd_weight}, "
                f"expected_kd_weight={expected_kd_weight}"
            )

    for dataset in selection_datasets:
        if not any(
            record.dataset == dataset and record.gt_weight > 0.0
            for record in validation_records
        ):
            raise ValueError(
                f"selection dataset has no clean-supervised validation records: {dataset}"
            )
    target_config = config.get("target_validation")
    if target_config and bool(target_config.get("enabled", False)):
        target_datasets = target_config.get("datasets")
        if not isinstance(target_datasets, dict) or not target_datasets:
            raise ValueError("target_validation.datasets must be a non-empty mapping")
        if "synthetic_camera_jpeg" in target_datasets:
            raise ValueError("Synthetic data cannot participate in target validation")
        if not set(map(str, target_datasets)) <= configured_datasets:
            raise ValueError("target_validation contains an unconfigured dataset")
        for dataset in target_datasets:
            if not any(
                target_record_selected(record.dataset, record.iso or -1, target_config)
                for record in validation_records
                if record.dataset == dataset
            ):
                raise ValueError(
                    f"Target validation dataset has no matching real records: {dataset}"
                )
    return {
        "manifest_sha256": snapshot.sha256,
        "cached_teacher_checkpoint_sha256": cached_teacher_hash.lower(),
        "configured_teacher_checkpoint_sha256": configured_teacher_hash.lower(),
        "datasets": sorted(configured_datasets),
        "selection_datasets": sorted(selection_datasets),
        "preprocessing": configured_preprocessing,
        "source_preprocessing": expected_source_preprocessing,
        "loss_weight_contract": (
            "paired_kd_runtime_override_teacher_only_kd_equals_1"
            if paired_kd_override is not None
            else "paired_kd_equals_alpha_teacher_only_kd_equals_1"
        ),
        "cached_paired_kd_weights": cached_paired_kd_weights,
        "effective_paired_kd_weight": (
            paired_kd_override if paired_kd_override is not None else alpha
        ),
        "target_validation_datasets": (
            sorted(target_config["datasets"])
            if target_config and bool(target_config.get("enabled", False))
            else []
        ),
    }


CORRECTION_LOSS_COEFFICIENTS = (
    "shadow_lambda",
    "medium_coarse_lambda",
    "normalized_shadow_lambda",
    "normalized_shadow_chroma_lambda",
    "shadow_medium_coarse_chroma_lambda",
    "flat_shadow_chroma_lambda",
    "very_coarse_chroma_lambda",
    "row_column_chroma_lambda",
    "normalized_very_coarse_chroma_lambda",
    "weighted_very_coarse_chroma_lambda",
    "normalized_row_column_chroma_lambda",
    "normalized_very_coarse_chroma_mse_lambda",
    "normalized_row_column_chroma_mse_lambda",
    "pyramid_chroma_lambda",
    "paired_detail_lambda",
)


def correction_loss_config_for_epoch(
    config: dict[str, Any] | None,
    epoch: int,
) -> dict[str, Any] | None:
    """Ramp specialized correction terms after base distillation stabilizes."""

    if not config or not bool(config.get("enabled", False)):
        return config
    if epoch < 1:
        raise ValueError("correction-loss scheduling requires a positive epoch")
    warmup_epochs = int(config.get("warmup_epochs", 0))
    ramp_epochs = int(config.get("ramp_epochs", 0))
    initial_scale = float(config.get("initial_scale", 0.0))
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError(
            "correction_loss warmup_epochs and ramp_epochs must be non-negative"
        )
    if not 0.0 <= initial_scale <= 1.0:
        raise ValueError("correction_loss initial_scale must be in [0,1]")
    if epoch <= warmup_epochs:
        scale = initial_scale
    elif ramp_epochs == 0:
        scale = 1.0
    else:
        progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
        scale = initial_scale + (1.0 - initial_scale) * progress
    scheduled = dict(config)
    for name in CORRECTION_LOSS_COEFFICIENTS:
        if name in scheduled:
            scheduled[name] = float(scheduled[name]) * scale
    scheduled["schedule_scale"] = scale
    return scheduled


def apply_paired_kd_weight_override(
    gt_weight: torch.Tensor,
    kd_weight: torch.Tensor,
    override: float | None,
) -> torch.Tensor:
    """Apply an explicit runtime KD weight to clean-paired records only."""

    if override is None:
        return kd_weight
    if not 0.0 <= override <= 1.0:
        raise ValueError("paired KD override must be in [0,1]")
    return torch.where(
        gt_weight > 0.0,
        torch.full_like(kd_weight, override),
        kd_weight,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    alpha: float,
    mse_scale: float,
    clean_l1_lambda: float,
    paired_kd_weight_override: float | None,
    clip_norm: float,
    amp_enabled: bool,
    correction_config: dict[str, Any] | None,
    conditioning_config: dict[str, Any] | None,
    condition_channel_only: bool,
    adapter_only: bool,
    trainable_prefixes: tuple[str, ...],
    gradient_masks: dict[str, torch.Tensor] | None,
    max_batches: int | None,
    quantizer_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    finite_gradient_batches = 0
    skipped_optimizer_steps = 0
    optimizer_steps = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        gt_weight = batch["gt_weight"].to(device, non_blocking=True, dtype=torch.float32)
        kd_weight = batch["kd_weight"].to(device, non_blocking=True, dtype=torch.float32)
        kd_weight = apply_paired_kd_weight_override(
            gt_weight,
            kd_weight,
            paired_kd_weight_override,
        )
        noise_strength: torch.Tensor | None = None
        model_input = noisy
        if conditioning_config and bool(conditioning_config.get("enabled", False)):
            noise_strength, conditioned_strength = training_noise_strength(
                noisy, conditioning_config
            )
            model_input = conditioned_input(
                noisy,
                conditioned_strength,
                conditioning_config,
            )
        if quantizer_step is not None:
            quantizer_step()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(model_input)
            terms = compute_weighted_distillation_loss(
                output,
                teacher,
                clean,
                gt_weight,
                kd_weight,
                alpha=alpha,
                mse_scale=mse_scale,
                lambda_l1=clean_l1_lambda,
                noisy=noisy,
                correction_config=correction_config,
                noise_strength=noise_strength,
            )
        scaler.scale(terms.total).backward()
        if adapter_only:
            adapter_prefixes = (
                ("multiscale_adapters.",)
                if model.multiscale_adapters is not None
                else ("noise_adapter.",)
            )
            for name, parameter in model.named_parameters():
                if not name.startswith(adapter_prefixes):
                    parameter.grad = None
        elif trainable_prefixes:
            for name, parameter in model.named_parameters():
                if not name.startswith(trainable_prefixes):
                    parameter.grad = None
        elif condition_channel_only:
            for name, parameter in model.named_parameters():
                if name != "input_conv.weight":
                    parameter.grad = None
            input_gradient = model.input_conv.weight.grad
            if input_gradient is None or input_gradient.shape[1] != 4:
                raise RuntimeError("condition-channel warmup requires a four-channel input")
            input_gradient[:, :3].zero_()
        if gradient_masks is not None:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_masks[name])
        scaler.unscale_(optimizer)
        gradient_norm = clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before_step:
            skipped_optimizer_steps += 1
        else:
            optimizer_steps += 1

        batch_size = noisy.shape[0]
        samples += batch_size
        if noise_strength is not None:
            sums["noise_strength_mean"] += float(noise_strength.mean()) * batch_size
        for name in (
            "total",
            "gt_mse",
            "kd_mse",
            "gt_l1",
            "shadow_kd_l1",
            "medium_coarse_kd_l1",
            "normalized_shadow_kd_l1",
            "normalized_shadow_chroma_kd_l1",
            "shadow_medium_coarse_chroma_kd_l1",
            "flat_shadow_chroma_kd_l1",
            "very_coarse_chroma_kd_l1",
            "row_column_chroma_kd_l1",
            "normalized_very_coarse_chroma_kd_l1",
            "weighted_very_coarse_chroma_kd_l1",
            "normalized_row_column_chroma_kd_l1",
            "normalized_very_coarse_chroma_kd_mse",
            "normalized_row_column_chroma_kd_mse",
            "pyramid_chroma_kd_l1",
            "paired_detail_l1",
            "gt_weight_mean",
            "kd_weight_mean",
        ):
            key = "loss" if name == "total" else name
            sums[key] += float(getattr(terms, name).detach()) * batch_size
        if math.isfinite(float(gradient_norm)):
            sums["gradient_norm_before_clip"] += float(gradient_norm)
            finite_gradient_batches += 1
    if samples == 0:
        raise RuntimeError("Training epoch processed no samples")
    result = {
        name: value / samples
        for name, value in sums.items()
        if name != "gradient_norm_before_clip"
    }
    result["gradient_norm_before_clip"] = (
        sums["gradient_norm_before_clip"] / finite_gradient_batches
        if finite_gradient_batches
        else 0.0
    )
    result["finite_gradient_batches"] = finite_gradient_batches
    result["skipped_optimizer_steps"] = skipped_optimizer_steps
    result["optimizer_steps"] = optimizer_steps
    result["samples"] = samples
    result["condition_channel_only"] = condition_channel_only
    result["adapter_only"] = adapter_only
    result["width_expansion_only"] = gradient_masks is not None
    result["correction_loss_scale"] = (
        float(correction_config.get("schedule_scale", 1.0))
        if correction_config and bool(correction_config.get("enabled", False))
        else 0.0
    )
    result["seconds"] = time.perf_counter() - start
    return result


def validation_tensors(
    output: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    teacher: torch.Tensor,
    border: int,
    window_size: int,
    sigma: float,
) -> dict[str, torch.Tensor]:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=output.device).view(1, 3, 1, 1)
    output_luma = (output * weights).sum(dim=1).flatten(1).mean(1)
    noisy_luma = (noisy * weights).sum(dim=1).flatten(1).mean(1)
    return {
        "student_psnr": psnr_per_image(output, clean, border=border),
        "student_ssim": gaussian_ssim_per_image(
            output, clean, border=border, window_size=window_size, sigma=sigma
        ),
        "noisy_psnr": psnr_per_image(noisy, clean, border=border),
        "noisy_ssim": gaussian_ssim_per_image(
            noisy, clean, border=border, window_size=window_size, sigma=sigma
        ),
        "teacher_psnr": psnr_per_image(teacher, clean, border=border),
        "teacher_ssim": gaussian_ssim_per_image(
            teacher, clean, border=border, window_size=window_size, sigma=sigma
        ),
        "student_teacher_psnr": psnr_per_image(output, teacher, border=border),
        "student_teacher_ssim": gaussian_ssim_per_image(
            output, teacher, border=border, window_size=window_size, sigma=sigma
        ),
        "output_luminance_drift": (output_luma - noisy_luma).abs(),
    }


def gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be finite and positive, got {sigma}")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    positions = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    filtered = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(filtered, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )


def chroma_projection(value: torch.Tensor) -> torch.Tensor:
    luma = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    projected_luma = (value * luma).sum(1, keepdim=True) * (
        luma / luma.square().sum()
    )
    return value - projected_luma


def masked_mean_per_image(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or mask.shape != (value.shape[0], 1, value.shape[2], value.shape[3]):
        raise ValueError("Target metric value/mask shapes are incompatible")
    denominator = mask.flatten(1).sum(1) * value.shape[1]
    numerator = (value * mask).flatten(1).sum(1)
    return numerator / denominator.clamp_min(1.0)


def target_correction_tensors(
    output: torch.Tensor,
    noisy: torch.Tensor,
    teacher: torch.Tensor,
    *,
    shadow_luminance_threshold: float,
    fine_sigma: float,
    medium_sigma: float,
    coarse_sigma: float,
    very_coarse_sigma: float | None = None,
) -> dict[str, torch.Tensor]:
    """Measure how much real high-ISO teacher correction the student captures."""

    if not 0.0 < shadow_luminance_threshold < 1.0:
        raise ValueError("shadow_luminance_threshold must be in (0,1)")
    if not 0.0 < fine_sigma < medium_sigma < coarse_sigma:
        raise ValueError("Target band sigmas must satisfy 0 < fine < medium < coarse")
    luma_weights = torch.tensor(
        [0.2126, 0.7152, 0.0722], device=noisy.device, dtype=noisy.dtype
    ).view(1, 3, 1, 1)
    shadow = ((noisy * luma_weights).sum(1, keepdim=True) < shadow_luminance_threshold).to(
        noisy.dtype
    )
    shadow_sample_valid = shadow.flatten(1).any(1)
    teacher_delta = teacher - noisy
    student_delta = output - noisy
    shadow_teacher_magnitude = masked_mean_per_image(teacher_delta.abs(), shadow)
    shadow_student_error = masked_mean_per_image((student_delta - teacher_delta).abs(), shadow)
    shadow_capture = 1.0 - shadow_student_error / shadow_teacher_magnitude.clamp_min(1e-6)
    shadow_capture = torch.where(
        shadow_sample_valid,
        shadow_capture,
        torch.full_like(shadow_capture, float("nan")),
    )
    shadow_teacher_magnitude = torch.where(
        shadow_sample_valid,
        shadow_teacher_magnitude,
        torch.full_like(shadow_teacher_magnitude, float("nan")),
    )
    teacher_chroma = chroma_projection(teacher_delta)
    student_chroma = chroma_projection(student_delta)
    shadow_chroma_teacher_magnitude = masked_mean_per_image(
        teacher_chroma.abs(), shadow
    )
    shadow_chroma_student_error = masked_mean_per_image(
        (student_chroma - teacher_chroma).abs(), shadow
    )
    shadow_chroma_capture = 1.0 - shadow_chroma_student_error / (
        shadow_chroma_teacher_magnitude.clamp_min(1e-6)
    )
    shadow_chroma_capture = torch.where(
        shadow_sample_valid,
        shadow_chroma_capture,
        torch.full_like(shadow_chroma_capture, float("nan")),
    )
    shadow_chroma_teacher_magnitude = torch.where(
        shadow_sample_valid,
        shadow_chroma_teacher_magnitude,
        torch.full_like(shadow_chroma_teacher_magnitude, float("nan")),
    )

    teacher_g1 = gaussian_blur(teacher_delta, fine_sigma)
    teacher_g4 = gaussian_blur(teacher_delta, medium_sigma)
    teacher_g12 = gaussian_blur(teacher_delta, coarse_sigma)
    student_g1 = gaussian_blur(student_delta, fine_sigma)
    student_g4 = gaussian_blur(student_delta, medium_sigma)
    student_g12 = gaussian_blur(student_delta, coarse_sigma)
    teacher_medium = teacher_g1 - teacher_g4
    teacher_coarse = teacher_g4 - teacher_g12
    student_medium = student_g1 - student_g4
    student_coarse = student_g4 - student_g12
    medium_coarse_teacher_magnitude = (
        teacher_medium.abs().flatten(1).mean(1)
        + teacher_coarse.abs().flatten(1).mean(1)
    )
    medium_coarse_student_error = (
        (student_medium - teacher_medium).abs().flatten(1).mean(1)
        + (student_coarse - teacher_coarse).abs().flatten(1).mean(1)
    )
    medium_coarse_capture = 1.0 - medium_coarse_student_error / (
        medium_coarse_teacher_magnitude.clamp_min(1e-6)
    )
    teacher_chroma_g1 = gaussian_blur(teacher_chroma, fine_sigma)
    teacher_chroma_g4 = gaussian_blur(teacher_chroma, medium_sigma)
    teacher_chroma_g12 = gaussian_blur(teacher_chroma, coarse_sigma)
    student_chroma_g1 = gaussian_blur(student_chroma, fine_sigma)
    student_chroma_g4 = gaussian_blur(student_chroma, medium_sigma)
    student_chroma_g12 = gaussian_blur(student_chroma, coarse_sigma)
    teacher_chroma_medium = teacher_chroma_g1 - teacher_chroma_g4
    teacher_chroma_coarse = teacher_chroma_g4 - teacher_chroma_g12
    student_chroma_medium = student_chroma_g1 - student_chroma_g4
    student_chroma_coarse = student_chroma_g4 - student_chroma_g12
    medium_coarse_chroma_teacher_magnitude = (
        teacher_chroma_medium.abs().flatten(1).mean(1)
        + teacher_chroma_coarse.abs().flatten(1).mean(1)
    )
    medium_coarse_chroma_student_error = (
        (student_chroma_medium - teacher_chroma_medium).abs().flatten(1).mean(1)
        + (student_chroma_coarse - teacher_chroma_coarse)
        .abs()
        .flatten(1)
        .mean(1)
    )
    medium_coarse_chroma_capture = 1.0 - medium_coarse_chroma_student_error / (
        medium_coarse_chroma_teacher_magnitude.clamp_min(1e-6)
    )
    extra: dict[str, torch.Tensor] = {}
    if very_coarse_sigma is not None:
        if very_coarse_sigma <= coarse_sigma:
            raise ValueError("very_coarse_sigma must exceed coarse_sigma")
        teacher_chroma_g24 = gaussian_blur(teacher_chroma, very_coarse_sigma)
        student_chroma_g24 = gaussian_blur(student_chroma, very_coarse_sigma)
        teacher_very_coarse = teacher_chroma_g12 - teacher_chroma_g24
        student_very_coarse = student_chroma_g12 - student_chroma_g24
        very_coarse_teacher_magnitude = (
            teacher_very_coarse.abs().flatten(1).mean(1)
        )
        very_coarse_student_error = (
            student_very_coarse - teacher_very_coarse
        ).abs().flatten(1).mean(1)
        teacher_row = teacher_chroma.mean(dim=3)
        teacher_column = teacher_chroma.mean(dim=2)
        row_error = (student_chroma - teacher_chroma).mean(dim=3)
        column_error = (student_chroma - teacher_chroma).mean(dim=2)
        row_column_teacher_magnitude = 0.5 * (
            teacher_row.abs().flatten(1).mean(1)
            + teacher_column.abs().flatten(1).mean(1)
        )
        row_column_student_error = 0.5 * (
            row_error.abs().flatten(1).mean(1)
            + column_error.abs().flatten(1).mean(1)
        )
        extra = {
            "very_coarse_chroma_teacher_correction_capture": (
                1.0
                - very_coarse_student_error
                / very_coarse_teacher_magnitude.clamp_min(1e-6)
            ),
            "very_coarse_chroma_teacher_correction_magnitude": (
                very_coarse_teacher_magnitude
            ),
            "row_column_chroma_teacher_correction_capture": (
                1.0
                - row_column_student_error
                / row_column_teacher_magnitude.clamp_min(1e-6)
            ),
            "row_column_chroma_teacher_correction_magnitude": (
                row_column_teacher_magnitude
            ),
        }
    return {
        "shadow_teacher_correction_capture": shadow_capture,
        "shadow_teacher_correction_magnitude": shadow_teacher_magnitude,
        "medium_coarse_teacher_correction_capture": medium_coarse_capture,
        "medium_coarse_teacher_correction_magnitude": medium_coarse_teacher_magnitude,
        "shadow_chroma_teacher_correction_capture": shadow_chroma_capture,
        "shadow_chroma_teacher_correction_magnitude": (
            shadow_chroma_teacher_magnitude
        ),
        "medium_coarse_chroma_teacher_correction_capture": (
            medium_coarse_chroma_capture
        ),
        "medium_coarse_chroma_teacher_correction_magnitude": (
            medium_coarse_chroma_teacher_magnitude
        ),
        "shadow_fraction": shadow.flatten(1).mean(1),
        "shadow_sample_valid": shadow_sample_valid,
        **extra,
    }


def target_record_selected(dataset: str, iso: int, config: dict[str, Any] | None) -> bool:
    if not config or not bool(config.get("enabled", False)):
        return False
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict) or dataset not in datasets:
        return False
    specification = datasets[dataset]
    if not isinstance(specification, dict):
        raise ValueError("target_validation.datasets entries must be mappings")
    configured_isos = specification.get("isos")
    return configured_isos is None or iso in {int(value) for value in configured_isos}


def validate(
    model: LiteDenoiseNet,
    loader: DataLoader,
    device: torch.device,
    border: int,
    window_size: int,
    sigma: float,
    selection_datasets: set[str],
    target_config: dict[str, Any] | None,
    max_batches: int | None,
    conditioning_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    all_values: dict[str, list[float]] = defaultdict(list)
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    target_values: dict[str, list[float]] = defaultdict(list)
    target_by_dataset: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_noise_bucket: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="validation", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            noise_strength: torch.Tensor | None = None
            model_input = noisy
            if conditioning_config and bool(conditioning_config.get("enabled", False)):
                noise_strength = estimate_noise_strength(noisy, conditioning_config)
                model_input = conditioned_input(
                    noisy,
                    noise_strength,
                    conditioning_config,
                )
            output = model(model_input)
            tensors = validation_tensors(
                output, noisy, clean, teacher, border, window_size, sigma
            )
            datasets = [str(value) for value in batch["dataset"]]
            isos = [int(value) for value in batch["iso"]]
            gt_valid = [float(value) > 0.0 for value in batch["gt_weight"]]
            for name, tensor in tensors.items():
                values = tensor.float().cpu().tolist()
                clean_metric = name in {
                    "student_psnr",
                    "student_ssim",
                    "noisy_psnr",
                    "noisy_ssim",
                    "teacher_psnr",
                    "teacher_ssim",
                }
                for dataset, value, valid in zip(datasets, values, gt_valid, strict=True):
                    if clean_metric and not valid:
                        continue
                    all_values[name].append(float(value))
                    by_dataset[dataset][name].append(float(value))
                if noise_strength is not None:
                    bucket_edges = conditioning_config.get(
                        "validation_bucket_edges", [0.4, 0.7]
                    )
                    if len(bucket_edges) != 2:
                        raise ValueError("validation_bucket_edges must contain two values")
                    low, high = map(float, bucket_edges)
                    buckets = [
                        "normal" if value < low else ("high" if value < high else "extreme")
                        for value in noise_strength.cpu().tolist()
                    ]
                    for bucket, value, valid in zip(
                        buckets, values, gt_valid, strict=True
                    ):
                        if clean_metric and not valid:
                            continue
                        by_noise_bucket[bucket][name].append(float(value))
            if target_config and bool(target_config.get("enabled", False)):
                selected_target = [
                    target_record_selected(dataset, iso, target_config)
                    for dataset, iso in zip(datasets, isos, strict=True)
                ]
                target_indices = [
                    index for index, selected_row in enumerate(selected_target) if selected_row
                ]
                if target_indices:
                    index_tensor = torch.tensor(target_indices, device=device, dtype=torch.long)
                    target_tensors = target_correction_tensors(
                        output.index_select(0, index_tensor),
                        noisy.index_select(0, index_tensor),
                        teacher.index_select(0, index_tensor),
                        shadow_luminance_threshold=float(
                            target_config["shadow_luminance_threshold"]
                        ),
                        fine_sigma=float(target_config["fine_sigma"]),
                        medium_sigma=float(target_config["medium_sigma"]),
                        coarse_sigma=float(target_config["coarse_sigma"]),
                        very_coarse_sigma=(
                            float(target_config["very_coarse_sigma"])
                            if "very_coarse_sigma" in target_config
                            else None
                        ),
                    )
                    selected_datasets = [datasets[index] for index in target_indices]
                else:
                    target_tensors = {}
                    selected_datasets = []
                shadow_sample_valid = target_tensors.pop(
                    "shadow_sample_valid", torch.empty(0, device=device, dtype=torch.bool)
                ).bool().cpu().tolist()
                shadow_metrics = {
                    "shadow_teacher_correction_capture",
                    "shadow_teacher_correction_magnitude",
                    "shadow_chroma_teacher_correction_capture",
                    "shadow_chroma_teacher_correction_magnitude",
                }
                for name, tensor in target_tensors.items():
                    values = tensor.float().cpu().tolist()
                    for dataset, value, shadow_valid in zip(
                        selected_datasets, values, shadow_sample_valid, strict=True
                    ):
                        if name in shadow_metrics and not shadow_valid:
                            continue
                        target_values[name].append(float(value))
                        target_by_dataset[dataset][name].append(float(value))
    if not all_values:
        raise RuntimeError("Validation or checkpoint-selection subset is empty")
    result: dict[str, Any] = {
        name: float(np.mean(values)) for name, values in all_values.items()
    }
    result["by_dataset"] = {
        dataset: {
            **{name: float(np.mean(values)) for name, values in metrics.items()},
            "samples": len(metrics["student_teacher_psnr"]),
            "clean_samples": len(metrics.get("student_psnr", [])),
        }
        for dataset, metrics in sorted(by_dataset.items())
    }
    if by_noise_bucket:
        result["by_noise_bucket"] = {
            bucket: {
                **{name: float(np.mean(values)) for name, values in metrics.items()},
                "samples": len(metrics["student_teacher_psnr"]),
                "clean_samples": len(metrics.get("student_psnr", [])),
            }
            for bucket, metrics in sorted(by_noise_bucket.items())
        }
    selected = [
        dataset
        for dataset in sorted(selection_datasets)
        if by_dataset[dataset].get("student_psnr")
    ]
    if set(selected) != selection_datasets:
        missing = sorted(selection_datasets - set(selected))
        raise RuntimeError(f"Selection datasets have no clean-valid validation rows: {missing}")
    # Each selected dataset has equal checkpoint influence regardless of its
    # number of source images or cached crops.
    result["selection_student_psnr"] = float(
        np.mean([np.mean(by_dataset[dataset]["student_psnr"]) for dataset in selected])
    )
    result["selection_student_ssim"] = float(
        np.mean([np.mean(by_dataset[dataset]["student_ssim"]) for dataset in selected])
    )
    result["selection_samples"] = sum(
        len(by_dataset[dataset]["student_psnr"]) for dataset in selected
    )
    result["selection_datasets"] = selected
    result["selection_aggregation"] = "equal_mean_of_dataset_means"
    if target_config and bool(target_config.get("enabled", False)):
        required = tuple(
            target_config.get(
                "score_metrics",
                (
                    "shadow_teacher_correction_capture",
                    "medium_coarse_teacher_correction_capture",
                ),
            )
        )
        if not required or len(required) != len(set(required)):
            raise ValueError(
                "target_validation.score_metrics must be non-empty and unique"
            )
        missing_required = [name for name in required if not target_values[name]]
        if missing_required:
            raise RuntimeError(
                "Target validation has no contributing rows for score metrics: "
                f"{missing_required}"
            )
        configured_target_datasets = {
            str(dataset) for dataset in target_config.get("datasets", {})
        }
        missing_target_datasets = configured_target_datasets - set(target_by_dataset)
        if missing_target_datasets:
            raise RuntimeError(
                "Target validation datasets have no selected records: "
                f"{sorted(missing_target_datasets)}"
            )
        unsupported_target_metrics = {
            dataset: [name for name in required if not metrics.get(name)]
            for dataset, metrics in target_by_dataset.items()
            if any(not metrics.get(name) for name in required)
        }
        if unsupported_target_metrics:
            raise RuntimeError(
                "Target validation datasets lack contributing rows for required metrics: "
                f"{unsupported_target_metrics}"
            )
        target_means = {
            name: float(np.mean(values))
            for name, values in target_values.items()
            if values
        }
        target_dataset_summaries = {
            dataset: {
                **{
                    name: float(np.mean(values))
                    for name, values in metrics.items()
                    if values
                },
                "samples": max(len(values) for values in metrics.values()),
                "shadow_contributing_samples": len(
                    metrics.get("shadow_teacher_correction_capture", [])
                ),
                "metric_samples": {
                    name: len(values) for name, values in metrics.items() if values
                },
            }
            for dataset, metrics in sorted(target_by_dataset.items())
        }
        metric_dataset_means = {
            name: float(
                np.mean(
                    [
                        np.mean(metrics[name])
                        for metrics in target_by_dataset.values()
                    ]
                )
            )
            for name in required
        }
        score_values = np.asarray(
            [metric_dataset_means[name] for name in required], dtype=np.float64
        )
        score_aggregation = str(
            target_config.get("score_aggregation", "mean")
        )
        if score_aggregation == "mean":
            target_score = float(score_values.mean())
            score_description = "equal_mean_of_per_metric_equal_dataset_means"
        elif score_aggregation == "mean_min":
            minimum_weight = float(target_config.get("minimum_metric_weight", 0.5))
            if not 0.0 <= minimum_weight <= 1.0:
                raise ValueError("minimum_metric_weight must be in [0,1]")
            target_score = float(
                (1.0 - minimum_weight) * score_values.mean()
                + minimum_weight * score_values.min()
            )
            score_description = (
                "blend_of_equal_mean_and_weakest_per_metric_equal_dataset_mean"
            )
        else:
            raise ValueError(
                "target_validation.score_aggregation must be mean or mean_min"
            )
        result["target_validation"] = {
            "score": target_score,
            "score_aggregation": score_description,
            "score_metric_minimum": float(score_values.min()),
            "score_metrics": list(required),
            "metrics": target_means,
            "metric_dataset_means": metric_dataset_means,
            "metric_samples": {
                name: len(values) for name, values in target_values.items() if values
            },
            "by_dataset": target_dataset_summaries,
            "samples": max(len(values) for values in target_values.values() if values),
            "shadow_contributing_samples": len(
                target_values["shadow_teacher_correction_capture"]
            ),
            "datasets": sorted(target_by_dataset),
            "real_only": True,
            "synthetic_samples": 0,
        }
    result["samples"] = len(all_values["student_teacher_psnr"])
    result["seconds"] = time.perf_counter() - start
    return result


def run_fingerprint(
    config: dict[str, Any],
    manifest_path: Path | MixedManifestSnapshot,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    initialization: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    paper_root = Path(__file__).parents[1]
    snapshot = mixed_manifest_snapshot(manifest_path)
    payload = {
        "seed": int(config["project"]["seed"]),
        "preprocessing_version": config["project"]["preprocessing_version"],
        "source_preprocessing": config["data"]["source_preprocessing"],
        "manifest_sha256": snapshot.sha256,
        "model": config["model"],
        "training": {**config["training"], "epochs": epochs},
        "metrics": config["metrics"],
        "target_validation": config.get("target_validation"),
        "datasets": config["data"]["datasets"],
        "initialization": (
            None
            if initialization is None
            else {
                key: value
                for key, value in initialization.items()
                if key != "checkpoint"
            }
        ),
        "diagnostic_limits": {
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
        },
        "source_sha256": {
            relative: sha256_file(paper_root / relative)
            for relative in (
                "src/student.py",
                "src/weighted_losses.py",
                "src/mixed_dataset.py",
                "src/metrics.py",
                "scripts/train_mixed.py",
                "scripts/common.py",
                "scripts/train.py",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def source_only_resume_matches(
    expected_fingerprint: str,
    current_payload: dict[str, Any],
    run_metadata_path: Path,
) -> bool:
    """Accept an explicit resume only when source hashes are the sole difference."""

    if not run_metadata_path.is_file():
        return False
    metadata = json.loads(run_metadata_path.read_text())
    stored_payload = metadata.get("run_fingerprint_payload")
    if (
        metadata.get("run_fingerprint") != expected_fingerprint
        or not isinstance(stored_payload, dict)
        or not isinstance(stored_payload.get("source_sha256"), dict)
    ):
        return False
    compatible_payload = json.loads(json.dumps(current_payload))
    compatible_payload.pop("source_sha256", None)
    stored_compatible_payload = dict(stored_payload)
    stored_compatible_payload.pop("source_sha256", None)
    return compatible_payload == stored_compatible_payload


def balanced_sample_weights(
    records: list,
    requested: dict[str, float],
    difficulty_weights: dict[str, float] | None = None,
    scene_balanced_datasets: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    counts = Counter(str(record.dataset) for record in records)
    configured = {str(name): float(weight) for name, weight in requested.items()}
    if set(counts) != set(configured):
        raise ValueError(
            f"Sampling datasets do not match training records: "
            f"records={sorted(counts)}, configured={sorted(configured)}"
        )
    if any(weight <= 0.0 for weight in configured.values()):
        raise ValueError("Every dataset sampling weight must be positive")
    total = sum(configured.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Dataset sampling weights must sum to 1, got {total}")
    scene_balanced = set(scene_balanced_datasets or ())
    unknown_scene_balanced = scene_balanced - set(configured)
    if unknown_scene_balanced:
        raise ValueError(
            "Scene-balanced datasets are not present in sampling weights: "
            f"{sorted(unknown_scene_balanced)}"
        )
    record_totals: dict[str, float] = defaultdict(float)
    scene_totals: dict[tuple[str, str], float] = defaultdict(float)
    dataset_scenes: dict[str, set[str]] = defaultdict(set)
    record_weights: list[float] = []
    for record in records:
        value = float(getattr(record, "sample_weight", 1.0))
        if difficulty_weights is not None:
            if record.input not in difficulty_weights:
                raise ValueError(f"Difficulty index is missing record: {record.input}")
            value *= float(difficulty_weights[record.input])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Every record sample_weight must be finite and positive, got {value}"
            )
        dataset = str(record.dataset)
        record_totals[dataset] += value
        if dataset in scene_balanced:
            scene = str(record.scene)
            scene_totals[(dataset, scene)] += value
            dataset_scenes[dataset].add(scene)
        record_weights.append(value)
    weights = torch.tensor(
        [
            (
                configured[str(record.dataset)]
                * value
                / scene_totals[(str(record.dataset), str(record.scene))]
                / len(dataset_scenes[str(record.dataset)])
                if str(record.dataset) in scene_balanced
                else configured[str(record.dataset)]
                * value
                / record_totals[str(record.dataset)]
            )
            for record, value in zip(records, record_weights, strict=True)
        ],
        dtype=torch.double,
    )
    return weights, dict(sorted(counts.items()))


def load_difficulty_weights(
    config: dict[str, Any], manifest_sha256: str
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    difficulty = config["training"].get("difficulty_sampling")
    if not difficulty or not bool(difficulty.get("enabled", False)):
        return None, None
    path = resolve_paper_path(difficulty["index"])
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected_sha256 = str(difficulty["sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise ValueError("Difficulty index SHA-256 does not match configuration")
    document = json.loads(payload)
    if document.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Difficulty index was built from a different manifest")
    weights = {
        str(key): float(value)
        for key, value in document.get("sampling_weights", {}).items()
    }
    if not weights or any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("Difficulty index contains invalid sampling weights")
    return weights, {
        "path": str(path),
        "sha256": actual_sha256,
        "records": len(weights),
        "bin_weights": document.get("bin_weights"),
        "dataset_bin_counts": document.get("dataset_bin_counts"),
    }


def difficulty_strength_for_epoch(config: dict[str, Any], epoch: int) -> float:
    """Ramp hard-example sampling after an ordinary-data warm-up."""

    difficulty = config["training"].get("difficulty_sampling")
    if not difficulty or not bool(difficulty.get("enabled", False)):
        return 0.0
    warmup = int(difficulty.get("warmup_epochs", 0))
    ramp = int(difficulty.get("ramp_epochs", 0))
    final = float(difficulty.get("final_strength", 1.0))
    if warmup < 0 or ramp < 0 or not 0.0 <= final <= 1.0:
        raise ValueError("Invalid difficulty-sampling curriculum")
    if epoch <= warmup:
        return 0.0
    if ramp == 0:
        return final
    progress = min(1.0, (epoch - warmup) / ramp)
    return final * progress


def blend_sample_weights(
    ordinary: torch.Tensor, hard: torch.Tensor, strength: float
) -> torch.Tensor:
    if ordinary.shape != hard.shape:
        raise ValueError("ordinary and hard sample weights must have the same shape")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("difficulty strength must be in [0,1]")
    return ordinary * (1.0 - strength) + hard * strength


def update_early_stopping(
    score: float,
    best: float,
    epochs_without_improvement: int,
    minimum_delta: float,
) -> tuple[float, int, bool]:
    if not math.isfinite(score):
        raise ValueError(f"Early-stopping score must be finite, got {score}")
    if not math.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise ValueError("Early-stopping minimum delta must be finite and non-negative")
    improved = score > best + minimum_delta
    if improved:
        return score, 0, True
    return best, epochs_without_improvement + 1, False


def rank_validation(
    validation_metrics: dict[str, Any],
    *,
    best_psnr: float,
    best_target_score: float | None,
    target_enabled: bool,
    general_psnr_guardrail: float,
    general_ssim_guardrail: float,
    early_stopping_metric: str,
    early_stopping_best: float,
    epochs_without_improvement: int,
    early_stopping_min_delta: float,
    target_component_minimums: dict[str, float] | None = None,
    count_guardrail_failure: bool = True,
) -> dict[str, Any]:
    """Rank one validation result against prior checkpoint and stop state."""

    selection_psnr = float(validation_metrics["selection_student_psnr"])
    selection_ssim = float(validation_metrics.get("selection_student_ssim", 1.0))
    general_improved = selection_psnr > best_psnr
    if general_improved:
        best_psnr = selection_psnr
    target_score = (
        float(validation_metrics["target_validation"]["score"])
        if target_enabled
        else None
    )
    general_guardrail_passed = (
        selection_psnr >= general_psnr_guardrail
        and selection_ssim >= general_ssim_guardrail
    )
    component_values = (
        validation_metrics.get("target_validation", {}).get(
            "metric_dataset_means", {}
        )
        if target_enabled
        else {}
    )
    target_component_minimums = target_component_minimums or {}
    target_component_guardrail_passed = all(
        float(component_values.get(name, float("-inf"))) >= minimum
        for name, minimum in target_component_minimums.items()
    )
    target_improved = bool(
        target_enabled
        and general_guardrail_passed
        and target_component_guardrail_passed
        and target_score is not None
        and (best_target_score is None or target_score > best_target_score)
    )
    if target_improved:
        best_target_score = target_score
    early_stopping_score = (
        target_score if early_stopping_metric == "target" else selection_psnr
    )
    if early_stopping_score is None:
        raise RuntimeError("Configured early-stopping metric is unavailable")
    if early_stopping_metric == "target" and not (
        general_guardrail_passed and target_component_guardrail_passed
    ):
        if count_guardrail_failure:
            epochs_without_improvement += 1
        early_stopping_improved = False
    else:
        (
            early_stopping_best,
            epochs_without_improvement,
            early_stopping_improved,
        ) = update_early_stopping(
            early_stopping_score,
            early_stopping_best,
            epochs_without_improvement,
            early_stopping_min_delta,
        )
    return {
        "selection_psnr": selection_psnr,
        "selection_ssim": selection_ssim,
        "best_psnr": best_psnr,
        "general_improved": general_improved,
        "target_score": target_score,
        "best_target_score": best_target_score,
        "target_improved": target_improved,
        "general_guardrail_passed": general_guardrail_passed,
        "target_component_values": component_values,
        "target_component_minimums": target_component_minimums,
        "target_component_guardrail_passed": target_component_guardrail_passed,
        "early_stopping_score": early_stopping_score,
        "early_stopping_best": early_stopping_best,
        "epochs_without_improvement": epochs_without_improvement,
        "early_stopping_improved": early_stopping_improved,
    }


def save_ranked_checkpoints(
    run_dir: Path,
    state: dict[str, Any],
    *,
    general_improved: bool,
    target_improved: bool,
) -> None:
    atomic_checkpoint(run_dir / "last.pt", state)
    if general_improved:
        atomic_checkpoint(run_dir / "best.pt", state)
        atomic_checkpoint(run_dir / "general-best.pt", state)
    if target_improved:
        atomic_checkpoint(run_dir / "target-best.pt", state)


def stratified_validation_indices(
    records: list,
    maximum_samples: int,
    clean_required: set[str],
    target_config: dict[str, Any] | None = None,
) -> list[int]:
    """Bound diagnostics while retaining every dataset and clean-selection domain."""

    if maximum_samples < 1:
        raise ValueError("maximum validation samples must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record.dataset)].append(index)
    if maximum_samples < len(grouped):
        raise ValueError(
            "--max-val-batches budget is too small to cover every validation dataset: "
            f"samples={maximum_samples}, datasets={len(grouped)}"
        )

    selected: list[int] = []
    used: set[int] = set()
    for dataset in sorted(grouped):
        candidates = grouped[dataset]
        if (
            target_config
            and bool(target_config.get("enabled", False))
            and dataset in target_config.get("datasets", {})
        ):
            candidates = [
                index
                for index in candidates
                if target_record_selected(
                    records[index].dataset, records[index].iso or -1, target_config
                )
            ]
            if not candidates:
                raise ValueError(
                    f"Target validation dataset has no matching diagnostic row: {dataset}"
                )
        if dataset in clean_required:
            candidates = [index for index in candidates if records[index].gt_weight > 0.0]
            if not candidates:
                raise ValueError(
                    f"Selection dataset has no clean-valid diagnostic row: {dataset}"
                )
        index = candidates[0]
        selected.append(index)
        used.add(index)

    queues = {
        dataset: [index for index in indices if index not in used]
        for dataset, indices in sorted(grouped.items())
    }
    while len(selected) < min(maximum_samples, len(records)):
        progressed = False
        for dataset in sorted(queues):
            if queues[dataset] and len(selected) < maximum_samples:
                selected.append(queues[dataset].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize only model weights; reset optimizer, scheduler, scaler, and RNG.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run through the configured epoch count while retaining ranked checkpoints",
    )
    parser.add_argument(
        "--allow-source-only-resume",
        action="store_true",
        help="Allow resume when only fingerprinted source files changed",
    )
    args = parser.parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")

    config = load_config(args.config)
    training = config["training"]
    alpha = float(training["alpha"])
    mse_scale = float(training.get("mse_scale", 1000.0))
    clean_l1_lambda = float(training.get("clean_l1_lambda", 50.0))
    run_name = str(training["run_name"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    initial_model_state: dict[str, torch.Tensor] | None = None
    initialization: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        initial_model_state, initialization = load_model_only_checkpoint(
            args.init_checkpoint
        )
        conditioning = config["model"].get("noise_conditioning")
        target_input_channels = (
            int(config["model"].get("input_channels", 4))
            if conditioning and bool(conditioning.get("enabled", False))
            else int(config["model"].get("input_channels", 3))
        )
        source_input_channels = int(initialization["source_input_channels"])
        adapter_channels = int(config["model"].get("noise_adapter_channels", 0))
        multiscale_adapter_channels = int(
            config["model"].get("multiscale_adapter_channels", 0)
        )
        chroma_head_channels = int(
            config["model"].get("chroma_head_channels", 0)
        )
        global_chroma_head_channels = int(
            config["model"].get("global_chroma_head_channels", 0)
        )
        chroma_unet_head_channels = int(
            config["model"].get("chroma_unet_head_channels", 0)
        )
        chroma_profile_head_channels = int(
            config["model"].get("chroma_profile_head_channels", 0)
        )
        chroma_profile_refinement_blocks = int(
            config["model"].get("chroma_profile_refinement_blocks", 0)
        )
        chroma_refinement_head_channels = int(
            config["model"].get("chroma_refinement_head_channels", 0)
        )
        source_has_multiscale_adapter = any(
            name.startswith("multiscale_adapters.")
            for name in initial_model_state
        )
        source_has_noise_adapter = any(
            name.startswith("noise_adapter.")
            for name in initial_model_state
        )
        source_has_chroma_head = any(
            name.startswith("chroma_head.")
            for name in initial_model_state
        )
        source_has_global_chroma_head = any(
            name.startswith("global_chroma_head.")
            for name in initial_model_state
        )
        source_has_chroma_unet_head = any(
            name.startswith("chroma_unet_head.")
            for name in initial_model_state
        )
        source_has_chroma_profile_head = any(
            name.startswith("chroma_profile_head.")
            for name in initial_model_state
        )
        source_has_chroma_profile_refinement = any(
            name.startswith("chroma_profile_head.refinement.")
            for name in initial_model_state
        )
        profile_input_key = "chroma_profile_head.input_projection.0.weight"
        target_uses_restored_profile = bool(
            config["model"].get("chroma_profile_use_restored", False)
        )
        source_profile_input_channels = (
            int(initial_model_state[profile_input_key].shape[1])
            if profile_input_key in initial_model_state
            else 0
        )
        target_profile_input_channels = (
            target_input_channels
            + int(config["model"]["base_width"]) * 16
            + (6 if target_uses_restored_profile else 0)
        )
        source_has_chroma_refinement_head = any(
            name.startswith("chroma_refinement_head.")
            for name in initial_model_state
        )
        if multiscale_adapter_channels and not source_has_multiscale_adapter:
            initialization["model_load"] = (
                "residual_adapter_to_multiscale_zero_initialized"
            )
        elif adapter_channels and not source_has_noise_adapter:
            initialization["model_load"] = (
                "backbone_to_residual_adapter_zero_initialized"
            )
        elif chroma_head_channels and not source_has_chroma_head:
            initialization["model_load"] = (
                "multiscale_to_chroma_head_zero_initialized"
            )
        elif global_chroma_head_channels and not source_has_global_chroma_head:
            initialization["model_load"] = (
                "chroma_head_to_global_chroma_head_zero_initialized"
            )
        elif chroma_unet_head_channels and not source_has_chroma_unet_head:
            initialization["model_load"] = (
                "global_to_chroma_unet_head_zero_initialized"
            )
        elif chroma_profile_head_channels and not source_has_chroma_profile_head:
            initialization["model_load"] = (
                "chroma_unet_to_profile_head_zero_initialized"
            )
        elif (
            chroma_profile_head_channels
            and target_uses_restored_profile
            and source_profile_input_channels + 6 == target_profile_input_channels
        ):
            initialization["model_load"] = (
                "profile_head_restoration_input_zero_initialized"
            )
        elif (
            chroma_profile_refinement_blocks
            and not source_has_chroma_profile_refinement
        ):
            initialization["model_load"] = (
                "profile_head_refinement_zero_initialized"
            )
        elif (
            chroma_refinement_head_channels
            and not source_has_chroma_refinement_head
        ):
            initialization["model_load"] = (
                "chroma_unet_to_refinement_head_zero_initialized"
            )
        else:
            initialization["model_load"] = (
                "rgb_to_noise_conditioned_zero_initialized"
                if source_input_channels == 3 and target_input_channels == 4
                else "exact"
            )
    elif args.resume is not None:
        resume_metadata = torch.load(
            args.resume.expanduser().resolve(), map_location="cpu", weights_only=False
        )
        if not isinstance(resume_metadata, dict):
            raise ValueError("Resume checkpoint is not a checkpoint mapping")
        stored_initialization = resume_metadata.get("initialization")
        if stored_initialization is not None and not isinstance(stored_initialization, dict):
            raise ValueError("Resume checkpoint has invalid initialization provenance")
        initialization = stored_initialization
    manifest_path = resolve_paper_path(config["data"]["manifest"])
    manifest_snapshot = MixedManifestSnapshot.load(manifest_path)
    cache_root = resolve_paper_path(config["data"]["cache_root"])
    runs_root = resolve_paper_path(config["outputs"]["runs"])
    array_integrity, verified_synthetic_arrays = pin_verified_synthetic_arrays(
        manifest_snapshot, cache_root
    )
    if verified_synthetic_arrays is not None:
        atexit.register(verified_synthetic_arrays.close)
    run_dir = args.output_dir.resolve() if args.output_dir else runs_root / run_name
    if args.resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Fresh run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    fingerprint, fingerprint_payload = run_fingerprint(
        config,
        manifest_snapshot,
        epochs,
        args.max_train_batches,
        args.max_val_batches,
        initialization,
    )
    batch_size = int(training["batch_size"])
    workers = int(training["workers"])
    train_dataset = MixedDistillationDataset(
        manifest_snapshot,
        root=cache_root,
        split="train",
        augment=True,
        augmentation_seed=seed,
        verified_synthetic_arrays=verified_synthetic_arrays,
    )
    validation_dataset = MixedDistillationDataset(
        manifest_snapshot,
        root=cache_root,
        split="validation",
        augment=False,
        verified_synthetic_arrays=verified_synthetic_arrays,
    )
    training_contract = validate_training_contract(
        config,
        manifest_snapshot,
        train_dataset.records,
        validation_dataset.records,
    )
    selection_datasets = {str(value) for value in training["selection_datasets"]}
    loader_generator = torch.Generator().manual_seed(seed)
    difficulty_weights, difficulty_metadata = load_difficulty_weights(
        config, manifest_snapshot.sha256
    )
    ordinary_sample_weights, train_dataset_counts = balanced_sample_weights(
        train_dataset.records,
        training["dataset_sampling_weights"],
        None,
        set(map(str, training.get("scene_balanced_datasets", ()))),
    )
    hard_sample_weights, _ = balanced_sample_weights(
        train_dataset.records,
        training["dataset_sampling_weights"],
        difficulty_weights,
        set(map(str, training.get("scene_balanced_datasets", ()))),
    )
    initial_difficulty_strength = difficulty_strength_for_epoch(config, 1)
    sample_weights = blend_sample_weights(
        ordinary_sample_weights,
        hard_sample_weights,
        initial_difficulty_strength,
    )
    samples_per_epoch = int(training["samples_per_epoch"])
    if samples_per_epoch < batch_size:
        raise ValueError("samples_per_epoch must be at least one batch")
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=loader_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    validation_loader_dataset = validation_dataset
    validation_batch_limit = args.max_val_batches
    if args.max_val_batches is not None:
        if args.max_val_batches < 1:
            parser.error("--max-val-batches must be positive")
        validation_indices = stratified_validation_indices(
            validation_dataset.records,
            args.max_val_batches * batch_size,
            selection_datasets,
            config.get("target_validation"),
        )
        validation_loader_dataset = Subset(validation_dataset, validation_indices)
        validation_batch_limit = None
    validation_loader = DataLoader(
        validation_loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditioning_config = config["model"].get("noise_conditioning")
    conditioned = bool(
        conditioning_config and conditioning_config.get("enabled", False)
    )
    qat_config = training.get("quantization_aware_training")
    qat_enabled = bool(qat_config and qat_config.get("enabled", False))
    qat_backend = (
        str(qat_config.get("backend", "litert_pt2e"))
        if qat_enabled
        else "none"
    )
    if qat_backend not in ("litert_pt2e", "coreml_fx", "none"):
        raise ValueError(f"Unsupported QAT backend: {qat_backend}")
    coreml_quantizer: Any | None = None
    model = LiteDenoiseNet(
        base_width=int(config["model"].get("base_width", 16)),
        input_channels=int(config["model"].get("input_channels", 4 if conditioned else 3)),
        noise_adapter_channels=int(config["model"].get("noise_adapter_channels", 0)),
        multiscale_adapter_channels=int(
            config["model"].get("multiscale_adapter_channels", 0)
        ),
        multiscale_spatial_gate=bool(
            config["model"].get("multiscale_spatial_gate", False)
        ),
        multiscale_chroma_floor=float(
            config["model"].get("multiscale_chroma_floor", 0.15)
        ),
        chroma_head_channels=int(
            config["model"].get("chroma_head_channels", 0)
        ),
        chroma_head_spatial_floor=float(
            config["model"].get("chroma_head_spatial_floor", 0.15)
        ),
        chroma_head_noise_floor=float(
            config["model"].get("chroma_head_noise_floor", 0.0)
        ),
        chroma_head_use_rgb=bool(
            config["model"].get("chroma_head_use_rgb", False)
        ),
        chroma_head_dilations=tuple(
            int(value)
            for value in config["model"].get("chroma_head_dilations", (2,))
        ),
        global_chroma_head_channels=int(
            config["model"].get("global_chroma_head_channels", 0)
        ),
        global_chroma_head_blocks=int(
            config["model"].get("global_chroma_head_blocks", 4)
        ),
        global_chroma_head_use_bottleneck=bool(
            config["model"].get("global_chroma_head_use_bottleneck", False)
        ),
        global_chroma_head_bilinear_upsample=bool(
            config["model"].get("global_chroma_head_bilinear_upsample", False)
        ),
        chroma_unet_head_channels=int(
            config["model"].get("chroma_unet_head_channels", 0)
        ),
        chroma_profile_head_channels=int(
            config["model"].get("chroma_profile_head_channels", 0)
        ),
        chroma_profile_use_restored=bool(
            config["model"].get("chroma_profile_use_restored", False)
        ),
        chroma_profile_refinement_blocks=int(
            config["model"].get("chroma_profile_refinement_blocks", 0)
        ),
        chroma_refinement_head_channels=int(
            config["model"].get("chroma_refinement_head_channels", 0)
        ),
        chroma_refinement_use_restored=bool(
            config["model"].get("chroma_refinement_use_restored", False)
        ),
        noise_gate_start=float(config["model"].get("noise_gate_start", 0.35)),
        noise_gate_end=float(config["model"].get("noise_gate_end", 0.75)),
        precomputed_noise_gate=bool(
            config["model"].get("precomputed_noise_gate", False)
        ),
        clamp_output=bool(config["model"].get("clamp_output", True)),
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        config["model"]["expected_parameters"]
    ):
        raise RuntimeError("Unexpected model parameter count")
    if initial_model_state is not None:
        initialization_mode = load_compatible_model_state(model, initial_model_state)
        if initialization is None or initialization["model_load"] != initialization_mode:
            raise RuntimeError("Initialization provenance disagrees with model loading")
        # A model-only continuation must not inherit the source run's random
        # stream or the RNG consumed by constructing the temporary model.
        seed_everything(seed)
        loader_generator.manual_seed(seed)
    float_state_keys = tuple(model.state_dict())
    if qat_enabled:
        if trainable_prefixes := training.get("trainable_prefixes", []):
            raise ValueError(
                "INT8 QAT currently requires all model parameters to be trainable; "
                f"got prefixes {trainable_prefixes}"
            )
        if qat_backend == "litert_pt2e":
            model = prepare_litert_int8_qat(
                model,
                input_channels=int(config["model"]["input_channels"]),
                maximum_batch_size=batch_size,
            ).to(device)
        else:
            observer_freeze_epoch = int(
                qat_config.get("observer_freeze_epoch", epochs)
            )
            observer_freeze_step = max(
                1, (observer_freeze_epoch - 1) * len(train_loader)
            )
            model, coreml_quantizer = prepare_coreml_int8_qat(
                model,
                input_channels=int(config["model"]["input_channels"]),
                observer_freeze_step=observer_freeze_step,
            )
            model = model.to(device)
    initial_model_sha256 = model_digest(model)
    trainable_prefixes = tuple(
        str(value) for value in training.get("trainable_prefixes", [])
    )
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training["minimum_learning_rate"]),
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda" and not qat_enabled
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 1
    best_psnr = float("-inf")
    best_target_score: float | None = None
    target_config = config.get("target_validation")
    target_enabled = bool(target_config and target_config.get("enabled", False))
    general_psnr_guardrail = (
        float(target_config["minimum_general_selection_psnr"])
        if target_enabled
        else float("-inf")
    )
    general_ssim_guardrail = (
        float(target_config.get("minimum_general_selection_ssim", float("-inf")))
        if target_enabled
        else float("-inf")
    )
    target_component_minimums = (
        {
            "shadow_teacher_correction_capture": float(
                target_config["minimum_shadow_teacher_correction_capture"]
            ),
            "medium_coarse_teacher_correction_capture": float(
                target_config["minimum_medium_coarse_teacher_correction_capture"]
            ),
        }
        if target_enabled
        and "minimum_shadow_teacher_correction_capture" in target_config
        and "minimum_medium_coarse_teacher_correction_capture" in target_config
        else {}
    )
    configured_component_minimums = (
        target_config.get("minimum_component_capture", {})
        if target_enabled
        else {}
    )
    if not isinstance(configured_component_minimums, dict):
        raise ValueError("minimum_component_capture must be a mapping")
    target_component_minimums.update(
        {
            str(name): float(value)
            for name, value in configured_component_minimums.items()
        }
    )
    if target_enabled and (
        ("minimum_shadow_teacher_correction_capture" in target_config)
        != ("minimum_medium_coarse_teacher_correction_capture" in target_config)
    ):
        raise ValueError(
            "Target component guardrails require both shadow and medium/coarse minimums"
        )
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 2.0
        for value in target_component_minimums.values()
    ):
        raise ValueError("Target component guardrails must be finite and in [0,2]")
    if target_enabled and not math.isfinite(general_psnr_guardrail):
        raise ValueError("minimum_general_selection_psnr must be finite")
    if target_enabled and "minimum_general_selection_ssim" in target_config and not (
        math.isfinite(general_ssim_guardrail) and 0.0 <= general_ssim_guardrail <= 1.0
    ):
        raise ValueError("minimum_general_selection_ssim must be finite and in [0,1]")
    early_stopping_metric = str(training.get("early_stopping_metric", "general"))
    if early_stopping_metric not in {"general", "target"}:
        raise ValueError("early_stopping_metric must be general or target")
    if early_stopping_metric == "target" and not target_enabled:
        raise ValueError("Target early stopping requires target_validation.enabled")
    early_stopping_patience_value = training.get("early_stopping_patience")
    early_stopping_patience = (
        None
        if early_stopping_patience_value is None
        else int(early_stopping_patience_value)
    )
    if args.disable_early_stopping:
        early_stopping_patience = None
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when configured")
    early_stopping_min_delta = float(
        training.get(
            "early_stopping_min_delta",
            training.get("early_stopping_min_delta_db", 0.0),
        )
    )
    if not math.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta_db must be finite and non-negative")
    early_stopping_best = float("-inf")
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_fingerprint = checkpoint.get("run_fingerprint")
        source_only_resume = bool(
            args.allow_source_only_resume
            and isinstance(checkpoint_fingerprint, str)
            and source_only_resume_matches(
                checkpoint_fingerprint,
                fingerprint_payload,
                run_dir / "run.json",
            )
        )
        if checkpoint.get("run_name") != run_name or (
            checkpoint_fingerprint != fingerprint and not source_only_resume
        ):
            raise ValueError("Resume checkpoint does not match this mixed run")
        resume_model_state = (
            checkpoint.get("qat_model") if qat_enabled else checkpoint.get("model")
        )
        if not isinstance(resume_model_state, dict):
            raise ValueError("Resume checkpoint has no compatible QAT model state")
        model.load_state_dict(resume_model_state, strict=True)
        if coreml_quantizer is not None:
            coreml_quantizer._step_count = int(
                checkpoint.get(
                    "qat_step_count",
                    int(checkpoint["epoch"]) * len(train_loader),
                )
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint["best_psnr"])
        stored_target_score = checkpoint.get("best_target_score")
        best_target_score = (
            float(stored_target_score) if stored_target_score is not None else None
        )
        stored_early_stopping_best = checkpoint.get("early_stopping_best")
        if stored_early_stopping_best is None:
            stored_early_stopping_best = (
                best_target_score
                if early_stopping_metric == "target" and best_target_score is not None
                else (best_psnr if early_stopping_metric == "general" else float("-inf"))
            )
        early_stopping_best = float(stored_early_stopping_best)
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    expansion_warmup_epochs = int(
        training.get("width_expansion_warmup_epochs", 0)
    )
    expansion_masks: dict[str, torch.Tensor] | None = None
    if expansion_warmup_epochs:
        metadata = (
            initialization.get("width_expansion")
            if isinstance(initialization, dict)
            else None
        )
        if not isinstance(metadata, dict):
            raise ValueError(
                "width-expansion warmup requires an expanded initialization checkpoint"
            )
        source_path = Path(str(metadata["source_checkpoint"])).expanduser().resolve()
        if sha256_file(source_path) != str(metadata["source_sha256"]):
            raise RuntimeError("Width-expansion source checkpoint digest changed")
        source_checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
        source_state = source_checkpoint.get("model")
        if not isinstance(source_state, dict):
            raise ValueError("Width-expansion source checkpoint has no model state")
        source_width = int(metadata["source_width"])
        target_width = int(metadata["target_width"])
        if target_width != model.base_width:
            raise ValueError("Width-expansion metadata disagrees with target model")
        expansion_masks = {
            name: mask.to(device)
            for name, mask in expansion_gradient_masks(
                source_state,
                model.state_dict(),
                source_width,
                target_width,
            ).items()
        }

    dataset_counts = Counter(
        (record.split, record.dataset)
        for record in train_dataset.records + validation_dataset.records
    )
    resolved = {
        "config": config,
        "run_name": run_name,
        "alpha": alpha,
        "loss_coefficients": {
            "clean_mse": mse_scale * (1.0 - alpha),
            "paired_teacher_mse": mse_scale * alpha,
            "teacher_only_mse": mse_scale,
            "clean_l1": clean_l1_lambda,
        },
        "dataset_sampling_weights": training["dataset_sampling_weights"],
        "difficulty_sampling": difficulty_metadata,
        "difficulty_curriculum": {
            "warmup_epochs": int(
                training.get("difficulty_sampling", {}).get("warmup_epochs", 0)
            ),
            "ramp_epochs": int(
                training.get("difficulty_sampling", {}).get("ramp_epochs", 0)
            ),
            "final_strength": float(
                training.get("difficulty_sampling", {}).get("final_strength", 1.0)
            ),
        },
        "train_dataset_counts": train_dataset_counts,
        "samples_per_epoch": samples_per_epoch,
        "selection_datasets": sorted(selection_datasets),
        "checkpoint_selection": {
            "general_metric": "selection_student_psnr",
            "target_metric": (
                "target_validation.score" if target_enabled else None
            ),
            "target_general_psnr_guardrail": (
                general_psnr_guardrail if target_enabled else None
            ),
            "target_general_ssim_guardrail": (
                general_ssim_guardrail if target_enabled and math.isfinite(general_ssim_guardrail) else None
            ),
            "target_component_minimums": target_component_minimums,
            "early_stopping_metric": early_stopping_metric,
        },
        "training_contract": training_contract,
        "quantization_aware_training": (
            {
                "enabled": True,
                "backend": qat_backend,
                "scheme": "static_per_tensor_a8_per_channel_w8",
                "real_export": (
                    "coremltools_linear_quantizer_w8a8"
                    if qat_backend == "coreml_fx"
                    else "ai_edge_quantizer_static_wi8_ai8"
                ),
                "observer_freeze_epoch": int(
                    qat_config.get("observer_freeze_epoch", epochs)
                ),
            }
            if qat_enabled
            else {"enabled": False}
        ),
        "array_integrity": array_integrity,
        "epochs_effective": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "dataset_counts": {
            f"{split}/{dataset}": count
            for (split, dataset), count in sorted(dataset_counts.items())
        },
        "initial_model_sha256": initial_model_sha256,
        "initialization": initialization,
        "environment": environment_report(
            config,
            manifest_path,
            manifest_sha256=manifest_snapshot.sha256,
        ),
        "run_fingerprint": fingerprint,
        "run_fingerprint_payload": fingerprint_payload,
        "diagnostic_limits": {
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    run_path = run_dir / "run.json"
    if args.resume is None:
        atomic_json(run_path, resolved)
    elif not run_path.is_file():
        raise ValueError("Resume run metadata is missing or incompatible")
    else:
        stored_run_fingerprint = json.loads(run_path.read_text()).get(
            "run_fingerprint"
        )
        if stored_run_fingerprint != fingerprint and not (
            args.allow_source_only_resume
            and isinstance(stored_run_fingerprint, str)
            and source_only_resume_matches(
                stored_run_fingerprint,
                fingerprint_payload,
                run_path,
            )
        ):
            raise ValueError("Resume run metadata is missing or incompatible")

    history_path = run_dir / "history.jsonl"
    status_path = run_dir / "status.json"
    atomic_json(
        status_path,
        {"state": "running", "run_name": run_name, "epoch": start_epoch - 1, "epochs": epochs},
    )
    border = int(config["metrics"]["border_crop"])
    window_size = int(config["metrics"]["ssim_window_size"])
    sigma = float(config["metrics"]["ssim_sigma"])
    current_epoch = start_epoch - 1
    try:
        if args.init_checkpoint is not None:
            initial_validation_start = time.perf_counter()
            initial_validation_rng = rng_state(loader_generator)
            try:
                validation_metrics = validate(
                    model,
                    validation_loader,
                    device,
                    border,
                    window_size,
                    sigma,
                    selection_datasets,
                    config.get("target_validation"),
                    validation_batch_limit,
                    conditioning_config,
                )
            finally:
                # Ranking the starting weights must not perturb the fresh run's
                # deterministic training stream.
                restore_rng(initial_validation_rng, loader_generator)
            ranking = rank_validation(
                validation_metrics,
                best_psnr=best_psnr,
                best_target_score=best_target_score,
                target_enabled=target_enabled,
                general_psnr_guardrail=general_psnr_guardrail,
                general_ssim_guardrail=general_ssim_guardrail,
                early_stopping_metric=early_stopping_metric,
                early_stopping_best=early_stopping_best,
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_min_delta=early_stopping_min_delta,
                target_component_minimums=target_component_minimums,
                count_guardrail_failure=False,
            )
            selection_psnr = ranking["selection_psnr"]
            best_psnr = ranking["best_psnr"]
            general_improved = ranking["general_improved"]
            target_score = ranking["target_score"]
            best_target_score = ranking["best_target_score"]
            target_improved = ranking["target_improved"]
            general_guardrail_passed = ranking["general_guardrail_passed"]
            target_component_guardrail_passed = ranking[
                "target_component_guardrail_passed"
            ]
            early_stopping_score = ranking["early_stopping_score"]
            early_stopping_best = ranking["early_stopping_best"]
            epochs_without_improvement = ranking["epochs_without_improvement"]
            early_stopping_improved = ranking["early_stopping_improved"]
            timestamp = datetime.now(timezone.utc).isoformat()
            record = {
                "epoch": 0,
                "epochs": epochs,
                "run_name": run_name,
                "alpha": alpha,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": None,
                "validation": validation_metrics,
                "initial_validation": True,
                "best_psnr": best_psnr,
                "best_target_score": best_target_score,
                "general_improved": general_improved,
                "target_improved": target_improved,
                "general_psnr_guardrail": {
                    "minimum": general_psnr_guardrail if target_enabled else None,
                    "passed": general_guardrail_passed if target_enabled else None,
                },
                "general_ssim_guardrail": {
                    "minimum": general_ssim_guardrail if target_enabled and math.isfinite(general_ssim_guardrail) else None,
                    "passed": general_guardrail_passed if target_enabled else None,
                },
                "target_component_guardrail": {
                    "minimums": target_component_minimums,
                    "values": ranking["target_component_values"],
                    "passed": target_component_guardrail_passed,
                },
                "early_stopping": {
                    "enabled": early_stopping_patience is not None,
                    "patience": early_stopping_patience,
                    "metric": early_stopping_metric,
                    "score": early_stopping_score,
                    "minimum_delta": early_stopping_min_delta,
                    "best_score": (
                        early_stopping_best if math.isfinite(early_stopping_best) else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "improved": early_stopping_improved,
                    "stopped": False,
                },
                "epoch_seconds": time.perf_counter() - initial_validation_start,
                "timestamp": timestamp,
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, sort_keys=True) + "\n")
            deployable_model_state = (
                float_model_state(model, float_state_keys)
                if qat_enabled
                else model.state_dict()
            )
            state = {
                "epoch": 0,
                "run_name": run_name,
                "alpha": alpha,
                "best_psnr": best_psnr,
                "best_target_score": best_target_score,
                "early_stopping_best": early_stopping_best,
                "epochs_without_improvement": epochs_without_improvement,
                "initialization": initialization,
                "model": deployable_model_state,
                "qat_model": model.state_dict() if qat_enabled else None,
                "qat_step_count": (
                    int(coreml_quantizer._step_count)
                    if coreml_quantizer is not None
                    else None
                ),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(loader_generator),
                "run_fingerprint": fingerprint,
                "config": config,
                "metrics": record,
            }
            save_ranked_checkpoints(
                run_dir,
                state,
                general_improved=general_improved,
                target_improved=target_improved,
            )
            atomic_json(
                status_path,
                {
                    "state": "running" if epochs > 0 else "complete",
                    "run_name": run_name,
                    "epoch": 0,
                    "epochs": epochs,
                    "best_selection_psnr": best_psnr,
                    "best_target_score": best_target_score,
                    "general_psnr_guardrail": {
                        "minimum": general_psnr_guardrail if target_enabled else None,
                        "last_passed": (
                            general_guardrail_passed if target_enabled else None
                        ),
                    },
                    "general_ssim_guardrail": {
                        "minimum": general_ssim_guardrail if target_enabled and math.isfinite(general_ssim_guardrail) else None,
                        "last_passed": general_guardrail_passed if target_enabled else None,
                    },
                    "target_component_guardrail": {
                        "minimums": target_component_minimums,
                        "last_values": ranking["target_component_values"],
                        "last_passed": target_component_guardrail_passed,
                    },
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_best_score": (
                        early_stopping_best if math.isfinite(early_stopping_best) else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "last_validation": validation_metrics,
                    "updated_at": timestamp,
                },
            )
            print(
                f"epoch 000/{epochs} run={run_name} initialized "
                f"select_psnr={selection_psnr:.4f} "
                f"target={target_score if target_score is not None else math.nan:.4f} "
                f"seconds={record['epoch_seconds']:.1f}",
                flush=True,
            )
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            if qat_enabled and qat_backend == "litert_pt2e":
                from torchao.quantization.pt2e import (
                    disable_observer,
                    enable_fake_quant,
                    enable_observer,
                )

                model.apply(enable_fake_quant)
                observer_freeze_epoch = int(
                    qat_config.get("observer_freeze_epoch", epochs)
                )
                model.apply(
                    disable_observer
                    if epoch >= observer_freeze_epoch
                    else enable_observer
                )
            train_dataset.set_epoch(epoch)
            difficulty_strength = difficulty_strength_for_epoch(config, epoch)
            sampler.weights = blend_sample_weights(
                ordinary_sample_weights,
                hard_sample_weights,
                difficulty_strength,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            epoch_start = time.perf_counter()
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                alpha,
                mse_scale,
                clean_l1_lambda,
                (
                    float(training["paired_kd_weight_override"])
                    if "paired_kd_weight_override" in training
                    else None
                ),
                float(training["gradient_clip_norm"]),
                amp_enabled,
                correction_loss_config_for_epoch(
                    training.get("correction_loss"),
                    epoch,
                ),
                conditioning_config,
                epoch
                <= int(training.get("conditioning_warmup_epochs", 0)),
                epoch <= int(training.get("adapter_only_epochs", 0)),
                trainable_prefixes,
                (
                    expansion_masks
                    if epoch <= expansion_warmup_epochs
                    else None
                ),
                args.max_train_batches,
                (
                    coreml_quantizer.step
                    if coreml_quantizer is not None
                    else None
                ),
            )
            validation_metrics = validate(
                model,
                validation_loader,
                device,
                border,
                window_size,
                sigma,
                selection_datasets,
                config.get("target_validation"),
                validation_batch_limit,
                conditioning_config,
            )
            if train_metrics["optimizer_steps"] > 0:
                scheduler.step()
            ranking = rank_validation(
                validation_metrics,
                best_psnr=best_psnr,
                best_target_score=best_target_score,
                target_enabled=target_enabled,
                general_psnr_guardrail=general_psnr_guardrail,
                general_ssim_guardrail=general_ssim_guardrail,
                early_stopping_metric=early_stopping_metric,
                early_stopping_best=early_stopping_best,
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_min_delta=early_stopping_min_delta,
                target_component_minimums=target_component_minimums,
            )
            selection_psnr = ranking["selection_psnr"]
            best_psnr = ranking["best_psnr"]
            general_improved = ranking["general_improved"]
            target_score = ranking["target_score"]
            best_target_score = ranking["best_target_score"]
            target_improved = ranking["target_improved"]
            general_guardrail_passed = ranking["general_guardrail_passed"]
            target_component_guardrail_passed = ranking[
                "target_component_guardrail_passed"
            ]
            early_stopping_score = ranking["early_stopping_score"]
            early_stopping_best = ranking["early_stopping_best"]
            epochs_without_improvement = ranking["epochs_without_improvement"]
            early_stopping_improved = ranking["early_stopping_improved"]
            early_stopped = (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            )
            record = {
                "epoch": epoch,
                "epochs": epochs,
                "run_name": run_name,
                "alpha": alpha,
                "learning_rate": learning_rate,
                "difficulty_strength": difficulty_strength,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_psnr": best_psnr,
                "best_target_score": best_target_score,
                "general_improved": general_improved,
                "target_improved": target_improved,
                "general_psnr_guardrail": {
                    "minimum": general_psnr_guardrail if target_enabled else None,
                    "passed": general_guardrail_passed if target_enabled else None,
                },
                "general_ssim_guardrail": {
                    "minimum": general_ssim_guardrail if target_enabled and math.isfinite(general_ssim_guardrail) else None,
                    "passed": general_guardrail_passed if target_enabled else None,
                },
                "target_component_guardrail": {
                    "minimums": target_component_minimums,
                    "values": ranking["target_component_values"],
                    "passed": target_component_guardrail_passed,
                },
                "early_stopping": {
                    "enabled": early_stopping_patience is not None,
                    "patience": early_stopping_patience,
                    "metric": early_stopping_metric,
                    "score": early_stopping_score,
                    "minimum_delta": early_stopping_min_delta,
                    "best_score": (
                        early_stopping_best if math.isfinite(early_stopping_best) else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "improved": early_stopping_improved,
                    "stopped": early_stopped,
                },
                "epoch_seconds": time.perf_counter() - epoch_start,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record, sort_keys=True) + "\n")
            deployable_model_state = (
                float_model_state(model, float_state_keys)
                if qat_enabled
                else model.state_dict()
            )
            state = {
                "epoch": epoch,
                "run_name": run_name,
                "alpha": alpha,
                "best_psnr": best_psnr,
                "best_target_score": best_target_score,
                "early_stopping_best": early_stopping_best,
                "epochs_without_improvement": epochs_without_improvement,
                "initialization": initialization,
                "model": deployable_model_state,
                "qat_model": model.state_dict() if qat_enabled else None,
                "qat_step_count": (
                    int(coreml_quantizer._step_count)
                    if coreml_quantizer is not None
                    else None
                ),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(loader_generator),
                "run_fingerprint": fingerprint,
                "config": config,
                "metrics": record,
            }
            save_ranked_checkpoints(
                run_dir,
                state,
                general_improved=general_improved,
                target_improved=target_improved,
            )
            if epoch % int(training["checkpoint_interval"]) == 0:
                atomic_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", state)
            atomic_json(
                status_path,
                {
                    "state": (
                        "early_stopped"
                        if early_stopped
                        else ("running" if epoch < epochs else "complete")
                    ),
                    "run_name": run_name,
                    "epoch": epoch,
                    "epochs": epochs,
                    "best_selection_psnr": best_psnr,
                    "best_target_score": best_target_score,
                    "general_psnr_guardrail": {
                        "minimum": general_psnr_guardrail if target_enabled else None,
                        "last_passed": general_guardrail_passed if target_enabled else None,
                    },
                    "general_ssim_guardrail": {
                        "minimum": general_ssim_guardrail if target_enabled and math.isfinite(general_ssim_guardrail) else None,
                        "last_passed": general_guardrail_passed if target_enabled else None,
                    },
                    "target_component_guardrail": {
                        "minimums": target_component_minimums,
                        "last_values": ranking["target_component_values"],
                        "last_passed": target_component_guardrail_passed,
                    },
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_best_score": (
                        early_stopping_best if math.isfinite(early_stopping_best) else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "last_validation": validation_metrics,
                    "updated_at": record["timestamp"],
                },
            )
            uhd_metrics = validation_metrics["by_dataset"].get("uhd_ll", {})
            snic_metrics = validation_metrics["by_dataset"].get("snic_sony", {})
            print(
                f"epoch {epoch:03d}/{epochs} run={run_name} "
                f"loss={train_metrics['loss']:.5f} "
                f"select_psnr={selection_psnr:.4f} "
                f"uhd_psnr={uhd_metrics.get('student_psnr', math.nan):.4f} "
                f"snic_psnr={snic_metrics.get('student_psnr', math.nan):.4f} "
                f"target={target_score if target_score is not None else math.nan:.4f} "
                f"general_best={best_psnr:.4f} "
                f"target_best={best_target_score if best_target_score is not None else math.nan:.4f} "
                f"seconds={record['epoch_seconds']:.1f}",
                flush=True,
            )
            if early_stopped:
                print(
                    f"early stopping run={run_name} epoch={epoch} "
                    f"patience={early_stopping_patience} "
                    f"metric={early_stopping_metric} "
                    f"minimum_delta={early_stopping_min_delta}",
                    flush=True,
                )
                break
    except BaseException as error:
        atomic_json(
            status_path,
            {
                "state": "failed",
                "run_name": run_name,
                "epoch": current_epoch,
                "epochs": epochs,
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
