#!/usr/bin/env python3
"""Combine the audited legacy/high-ISO cache with UHD-LL/SNIC domain data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

from common import atomic_json, load_config, resolve_paper_path, sha256_file
from validate_synthetic_camera_jpeg import (
    ANALYSIS_SCHEMA_VERSION,
    active_validator_identity,
    report_contract_errors,
)


ARRAY_FIELDS = ("input", "clean", "teacher")
VISUAL_ACCEPTANCE_KEYS = {
    "schema_version",
    "decision",
    "reviewer",
    "accepted_at",
    "manifest_sha256",
    "cache_content_sha256",
    "contact_png_sha256",
    "contact_jpg_sha256",
    "analysis_sha256",
    "reviewed_report_sha256",
}


def load_records(path: Path) -> tuple[dict, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("records") if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError(f"Manifest has no records: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Manifest records must be JSON objects: {path}")
    return document if isinstance(document, dict) else {}, records


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Mixed-cache destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def require_teacher_hash(document: dict, manifest_path: Path) -> str:
    value = document.get("teacher_checkpoint_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"Manifest has no valid teacher checkpoint hash: {manifest_path}")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(
            f"Manifest has a non-hex teacher checkpoint hash: {manifest_path}"
        ) from error
    return value.lower()


def require_preprocessing(document: dict, manifest_path: Path) -> str:
    value = document.get("preprocessing")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Manifest has no preprocessing version: {manifest_path}")
    return value


def safe_relative_path(value: object, *, manifest_path: Path, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Manifest {field} path must remain inside its cache: {manifest_path}: {path}"
        )
    return path


def configured_sample_weight(record: dict, config: dict) -> float:
    """Return a positive train-only record multiplier.

    Dataset totals are normalized later by ``balanced_sample_weights``. This
    multiplier only changes sampling within one dataset, and validation rows
    deliberately remain uniform.
    """

    if str(record.get("split")) != "train":
        return 1.0
    weight = float(record.get("sample_weight", 1.0))
    record_sampling = config.get("training", {}).get("record_sampling", {})
    dataset_config = record_sampling.get(str(record.get("dataset")))
    if dataset_config is not None:
        if not isinstance(dataset_config, dict):
            raise ValueError("training.record_sampling entries must be mappings")
        field = str(dataset_config.get("field", ""))
        values = dataset_config.get("values", {})
        if not field or not isinstance(values, dict):
            raise ValueError(
                "record_sampling entries require a field and values mapping"
            )
        default = float(dataset_config.get("default", 1.0))
        configured = {str(key): float(value) for key, value in values.items()}
        weight *= configured.get(str(record.get(field)), default)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            f"Record sample weight must be finite and positive: "
            f"dataset={record.get('dataset')}, scene={record.get('scene')}, weight={weight}"
        )
    return weight


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def cache_content_identity(root: Path, records: list[dict]) -> dict[str, int | str]:
    """Recompute the validator's identity over every cached training tensor."""

    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    identifiers = [str(record.get("id", "")) for record in records]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("Synthetic cache records require unique non-empty IDs")
    for record in sorted(records, key=lambda value: str(value.get("id"))):
        for field in ARRAY_FIELDS:
            relative = safe_relative_path(
                record.get(field), manifest_path=root / "manifest.json", field=field
            )
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            entry = [str(record.get("id")), field, str(record[field]), sha256_file(path)]
            digest.update(canonical_json(entry))
            files += 1
            bytes_total += path.stat().st_size
    return {"files": files, "bytes": bytes_total, "sha256": digest.hexdigest()}


def load_json_mapping(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def report_artifact_path(report_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Synthetic gate report has no {label} path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (report_path.parent / path).resolve()


def validate_synthetic_acceptance(
    cache_root: Path,
    manifest_path: Path,
    records: list[dict],
    report_path: Path,
    signoff_path: Path,
) -> dict:
    """Reject synthetic data unless its exact payload has an accepted release gate."""

    report_path = report_path.expanduser().resolve()
    signoff_path = signoff_path.expanduser().resolve()
    report = load_json_mapping(report_path, "synthetic gate report")
    signoff = load_json_mapping(signoff_path, "synthetic visual acceptance")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "accepted"
        or report.get("release_gate_passed") is not True
    ):
        raise RuntimeError(
            "Synthetic cache is not accepted by its release gate: "
            f"schema={report.get('schema_version')!r}, status={report.get('status')!r}, "
            f"passed={report.get('release_gate_passed')!r}"
        )
    smoke = report.get("smoke")
    if not isinstance(smoke, dict) or smoke.get("generation") is not False or smoke.get(
        "calibration"
    ) is not False:
        raise RuntimeError(f"Synthetic gate report is smoke/provisional: {report_path}")

    contract_errors = report_contract_errors(report)
    if contract_errors:
        raise RuntimeError(
            f"Synthetic gate report is schema/semantically inconsistent: {contract_errors}"
        )
    analysis = report["analysis"]
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise RuntimeError("Synthetic gate report uses an inactive analysis schema")
    if analysis.get("validator") != active_validator_identity():
        raise RuntimeError("Synthetic gate report was not produced by the active validator")
    findings = analysis.get("findings")
    if findings.get("errors", {}).get("total") != 0:
        raise RuntimeError("Accepted synthetic gate report contains hard errors")
    if analysis.get("smoke") != {"generation": False, "calibration": False}:
        raise RuntimeError("Accepted synthetic gate analysis is smoke/provisional")
    if (
        analysis.get("records_loaded") != len(records)
        or analysis.get("records_measured") != len(records)
    ):
        raise RuntimeError(
            "Accepted synthetic gate report did not load and measure every manifest record"
        )

    manifest_identity = report.get("manifest")
    actual_manifest_sha = sha256_file(manifest_path)
    if not isinstance(manifest_identity, dict) or manifest_identity.get(
        "sha256"
    ) != actual_manifest_sha:
        raise RuntimeError("Synthetic gate report does not bind the active manifest SHA")
    reported_manifest_path = report_artifact_path(
        report_path, manifest_identity.get("path"), "manifest"
    )
    if reported_manifest_path != manifest_path.resolve():
        raise RuntimeError(
            "Synthetic gate report references a different manifest: "
            f"{reported_manifest_path} != {manifest_path.resolve()}"
        )

    actual_cache = cache_content_identity(cache_root, records)
    if report.get("cache_content") != actual_cache:
        raise RuntimeError(
            "Synthetic cache content differs from the accepted gate report: "
            f"report={report.get('cache_content')!r}, actual={actual_cache!r}"
        )
    if actual_cache["files"] != len(records) * len(ARRAY_FIELDS):
        raise RuntimeError("Synthetic cache content identity did not cover every record array")

    contact = report.get("contact_sheet")
    if not isinstance(contact, dict):
        raise RuntimeError("Accepted synthetic gate report has no contact sheet")
    contact_hashes: dict[str, str] = {}
    for extension in ("png", "jpg"):
        path = report_artifact_path(report_path, contact.get(extension), f"contact {extension}")
        actual = sha256_file(path)
        if contact.get(f"{extension}_sha256") != actual:
            raise RuntimeError(f"Synthetic contact-sheet {extension} hash mismatch: {path}")
        contact_hashes[extension] = actual

    analysis_sha = report.get("analysis_sha256")
    if not isinstance(analysis_sha, str) or len(analysis_sha) != 64:
        raise RuntimeError("Accepted synthetic gate report has no analysis SHA")
    try:
        int(analysis_sha, 16)
    except ValueError as error:
        raise RuntimeError("Synthetic gate analysis SHA is not hexadecimal") from error
    if not isinstance(analysis, dict) or hashlib.sha256(canonical_json(analysis)).hexdigest() != (
        analysis_sha
    ):
        raise RuntimeError("Synthetic gate report analysis payload does not match its SHA")
    if set(signoff) != VISUAL_ACCEPTANCE_KEYS:
        raise RuntimeError(
            "Synthetic visual acceptance has an invalid schema: "
            f"missing={sorted(VISUAL_ACCEPTANCE_KEYS - set(signoff))}, "
            f"unexpected={sorted(set(signoff) - VISUAL_ACCEPTANCE_KEYS)}"
        )
    if signoff.get("schema_version") != 2 or signoff.get("decision") != "accepted":
        raise RuntimeError(f"Synthetic visual acceptance is not accepted: {signoff_path}")
    reviewed_report_sha = report.get("reviewed_report_sha256")
    if (
        not isinstance(reviewed_report_sha, str)
        or len(reviewed_report_sha) != 64
    ):
        raise RuntimeError("Accepted synthetic report has no reviewed-report SHA")
    try:
        int(reviewed_report_sha, 16)
    except ValueError as error:
        raise RuntimeError("Synthetic reviewed-report SHA is not hexadecimal") from error
    if reviewed_report_sha != reviewed_report_sha.lower():
        raise RuntimeError("Synthetic reviewed-report SHA must be lowercase")
    reviewer = signoff.get("reviewer")
    accepted_at = signoff.get("accepted_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise RuntimeError("Synthetic visual acceptance has no reviewer or acceptance time")
    try:
        accepted_time = datetime.fromisoformat(str(accepted_at).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("Synthetic visual acceptance time is invalid") from error
    if accepted_time.tzinfo is None or accepted_time.utcoffset() is None:
        raise RuntimeError("Synthetic visual acceptance time must include a timezone")
    expected_signoff = {
        "manifest_sha256": actual_manifest_sha,
        "cache_content_sha256": actual_cache["sha256"],
        "contact_png_sha256": contact_hashes["png"],
        "contact_jpg_sha256": contact_hashes["jpg"],
        "analysis_sha256": analysis_sha,
        "reviewed_report_sha256": reviewed_report_sha,
    }
    mismatches = {
        key: (signoff.get(key), value)
        for key, value in expected_signoff.items()
        if signoff.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Synthetic visual acceptance binding mismatch: {mismatches}")
    embedded_acceptance = report.get("visual_acceptance")
    signoff_sha = sha256_file(signoff_path)
    if (
        not isinstance(embedded_acceptance, dict)
        or embedded_acceptance.get("status") != "accepted"
        or embedded_acceptance.get("reviewer") != reviewer
        or embedded_acceptance.get("accepted_at") != accepted_at
        or embedded_acceptance.get("reviewed_report_sha256")
        != signoff.get("reviewed_report_sha256")
        or embedded_acceptance.get("sha256") != signoff_sha
    ):
        raise RuntimeError("Synthetic gate report does not embed the current accepted signoff")
    return {
        "status": "accepted",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "visual_acceptance": str(signoff_path),
        "visual_acceptance_sha256": signoff_sha,
        "manifest_sha256": actual_manifest_sha,
        "cache_content": actual_cache,
        "contact_sheet_sha256": contact_hashes,
        "analysis_sha256": analysis_sha,
        "reviewed_report_sha256": signoff["reviewed_report_sha256"],
        "reviewer": signoff["reviewer"],
        "accepted_at": signoff["accepted_at"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=resolve_paper_path("data/high_iso_ablation/cache"),
    )
    parser.add_argument(
        "--domain-root",
        type=Path,
        default=resolve_paper_path("data/uhd_snic_gate/cache"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=resolve_paper_path("data/uhd_snic_mixed/cache"),
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        help="Optional audited synthetic-camera-JPEG cache to add as a third namespace.",
    )
    parser.add_argument(
        "--synthetic-gate-report",
        type=Path,
        help="Accepted gate report; defaults beside the synthetic cache directory.",
    )
    parser.add_argument(
        "--synthetic-visual-acceptance",
        type=Path,
        help="Hash-bound visual signoff; defaults beside the synthetic cache directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/uhd_snic_alpha_0p7.yaml",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = args.output_root.resolve()
    build_root = output_root.with_name(f".{output_root.name}.building")
    backup_root = output_root.with_name(f".{output_root.name}.previous")
    if backup_root.exists() and not output_root.exists():
        backup_root.rename(output_root)
    expected_preprocessing = {
        str(namespace): str(version)
        for namespace, version in config["data"]["source_preprocessing"].items()
    }
    sources = [
        ("legacy", args.legacy_root.resolve()),
        ("domain", args.domain_root.resolve()),
    ]
    if args.synthetic_root is not None:
        sources.append(("synthetic", args.synthetic_root.resolve()))
    actual_namespaces = {namespace for namespace, _ in sources}
    configured_namespaces = set(expected_preprocessing)
    if actual_namespaces != configured_namespaces:
        raise RuntimeError(
            "Mixed-cache source namespaces do not match the training configuration: "
            f"sources={sorted(actual_namespaces)}, configured={sorted(configured_namespaces)}"
        )
    loaded_sources = []
    teacher_hashes = set()
    source_preprocessing = {}
    source_acceptance: dict[str, dict] = {}
    for namespace, root in sources:
        manifest_path = root / "manifest.json"
        document, records = load_records(manifest_path)
        teacher_hashes.add(require_teacher_hash(document, manifest_path))
        source_preprocessing[namespace] = require_preprocessing(document, manifest_path)
        for source_record in records:
            for field in ARRAY_FIELDS:
                if field not in source_record:
                    raise ValueError(f"Manifest record is missing {field}: {manifest_path}")
                source_relative = safe_relative_path(
                    source_record[field], manifest_path=manifest_path, field=field
                )
                source_path = root / source_relative
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
        if namespace == "synthetic":
            report_path = (
                args.synthetic_gate_report
                if args.synthetic_gate_report is not None
                else root.parent / "synthetic_camera_jpeg_gate_report.json"
            )
            signoff_path = (
                args.synthetic_visual_acceptance
                if args.synthetic_visual_acceptance is not None
                else root.parent / "visual_acceptance.json"
            )
            source_acceptance[namespace] = validate_synthetic_acceptance(
                root,
                manifest_path,
                records,
                report_path,
                signoff_path,
            )
        loaded_sources.append((namespace, root, manifest_path, records))
    if len(teacher_hashes) != 1:
        raise RuntimeError(f"Combined caches use different teacher checkpoints: {teacher_hashes}")
    if source_preprocessing != expected_preprocessing:
        raise RuntimeError(
            "Combined cache preprocessing does not match the training configuration: "
            f"sources={source_preprocessing}, configured={expected_preprocessing}"
        )

    # Keep the published cache intact until a complete replacement is ready.
    if output_root.exists():
        if not args.replace:
            raise FileExistsError(f"Output cache exists: {output_root}; pass --replace")
    if build_root.exists():
        if not args.replace:
            raise FileExistsError(f"Incomplete mixed cache exists: {build_root}; pass --replace")
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    output = []
    link_modes: collections.Counter[str] = collections.Counter()
    provenance = {}
    materialized_sources: dict[Path, Path] = {}
    for namespace, root, manifest_path, records in loaded_sources:
        provenance[namespace] = {
            "root": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "records": len(records),
            "preprocessing": source_preprocessing[namespace],
            **(
                {"acceptance": source_acceptance[namespace]}
                if namespace in source_acceptance
                else {}
            ),
        }
        for index, source_record in enumerate(records):
            record = dict(source_record)
            if namespace == "synthetic":
                # Keep the exact pre-prefix paths used by the accepted cache
                # digest. Training can then reconstruct that identity over the
                # hardlinked mixed-cache payload instead of trusting metadata.
                record["accepted_source_array_paths"] = {
                    field: str(source_record[field]) for field in ARRAY_FIELDS
                }
            if namespace == "legacy":
                # Carry forward the selected NIND full-reference arm. All
                # legacy records therefore retain the paper's 0.7 KD ratio.
                record["original_supervision"] = record.get("supervision", "paired")
                record["supervision"] = (
                    "reference_paired" if record.get("dataset") == "nind" else "paired"
                )
                record["gt_weight"] = 1.0
                record["kd_weight"] = 0.7
            record["sample_weight"] = configured_sample_weight(record, config)
            for field in ARRAY_FIELDS:
                source_relative = safe_relative_path(
                    source_record[field], manifest_path=manifest_path, field=field
                )
                source_path = root / source_relative
                relative = Path(namespace) / source_relative
                destination = build_root / relative
                previous = materialized_sources.get(destination)
                if previous is None:
                    link_modes[materialize(source_path, destination)] += 1
                    materialized_sources[destination] = source_path
                elif previous == source_path:
                    link_modes["reuse"] += 1
                else:
                    raise FileExistsError(
                        "Different source arrays map to the same mixed-cache path: "
                        f"{destination}"
                    )
                record[field] = str(relative)
            record["mixed_source"] = namespace
            record["mixed_source_index"] = index
            output.append(record)
    output.sort(key=lambda row: (row["split"], row["dataset"], row["scene"], row["input"]))
    counts = collections.Counter(
        (row["split"], row["dataset"], row["supervision"]) for row in output
    )
    sample_weight_sums: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in output:
        sample_weight_sums[(row["split"], row["dataset"])] += float(row["sample_weight"])
    scene_splits: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in output:
        scene_splits[(row["dataset"], row["scene"])].add(row["split"])
    leakage = [key for key, splits in scene_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Scene leakage in mixed cache: {leakage[0]}")
    published_manifest = output_root / "manifest.json"
    build_manifest = build_root / "manifest.json"
    atomic_json(
        build_manifest,
        {
            "schema_version": 2,
            "purpose": "Balanced UHD-LL/SNIC Sony domain-expansion training",
            "preprocessing": str(config["project"]["preprocessing_version"]),
            "source_preprocessing": source_preprocessing,
            "teacher_checkpoint_sha256": next(iter(teacher_hashes)),
            "sources": provenance,
            "records": output,
        },
    )
    report = {
        "manifest": str(published_manifest),
        "manifest_sha256": sha256_file(build_manifest),
        "preprocessing": str(config["project"]["preprocessing_version"]),
        "source_preprocessing": source_preprocessing,
        "records": len(output),
        "scene_groups": len(scene_splits),
        "scene_leakage": 0,
        "counts": {
            f"{split}/{dataset}/{supervision}": count
            for (split, dataset, supervision), count in sorted(counts.items())
        },
        "sample_weight_sums": {
            f"{split}/{dataset}": weight
            for (split, dataset), weight in sorted(sample_weight_sums.items())
        },
        "array_materialization": dict(sorted(link_modes.items())),
        "sources": provenance,
    }
    atomic_json(build_root / "manifest.report.json", report)

    if backup_root.exists():
        shutil.rmtree(backup_root)
    had_previous = output_root.exists()
    if had_previous:
        output_root.rename(backup_root)
    try:
        build_root.rename(output_root)
    except BaseException:
        if had_previous and backup_root.exists() and not output_root.exists():
            backup_root.rename(output_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
