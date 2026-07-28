#!/usr/bin/env python3
"""Download selected TIFF pairs from the large SNIC Dataverse archives.

The official SNIC archives contain both RAW and 16-bit sRGB TIFF files.  This
script uses HTTP range requests so a mobile-RGB experiment can fetch only the
TIFF entries and ISO levels it needs instead of downloading every RAW file.
Run it with ``uv run --with remotezip python scripts/download_snic_subset.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from remotezip import RemoteZip
from tqdm import tqdm

DATAVERSE_FILE_API = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVES = {
    13237248: "sony_a7r_iii_indoor1.zip",
    13237336: "sony_a7r_iii_indoor2.zip",
    13242871: "sony_a7r_iii_outdoor1.zip",
    13243019: "sony_a7r_iii_outdoor2.zip",
    13237440: "sony_a7r_iii_outdoor3.zip",
}
TIFF_NAME = re.compile(
    r"^(?P<prefix>.+)_(?P<iso>\d{5})_(?P<kind>clean|noisy)_01\.tiff$",
    re.IGNORECASE,
)


def resolve_paper_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PAPER_ROOT / path).resolve()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def signed_download_url(file_id: int) -> str:
    request = Request(
        DATAVERSE_FILE_API.format(file_id=file_id),
        method="GET",
        headers={"User-Agent": "Mozilla/5.0 SNIC-dataset-downloader/1.0"},
    )
    try:
        build_opener(NoRedirect).open(request)
    except HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise
        location = error.headers.get("Location")
        if location:
            return location
    raise RuntimeError(f"Dataverse did not return a signed URL for file {file_id}")


def selected_entries(remote: RemoteZip, noisy_isos: set[int]) -> list:
    infos = {info.filename: info for info in remote.infolist() if not info.is_dir()}
    selected = {}
    found_isos: set[int] = set()
    noisy_counts: Counter[int] = Counter()
    for name, info in infos.items():
        member = Path(name)
        if member.is_absolute() or any(part == ".." for part in member.parts):
            raise ValueError(f"Unsafe path in SNIC archive: {name!r}")
        match = TIFF_NAME.match(name)
        if not match or match.group("kind").lower() != "noisy":
            continue
        iso = int(match.group("iso"))
        if iso not in noisy_isos:
            continue
        clean_name = f"{match.group('prefix')}_00100_clean_01.tiff"
        clean = infos.get(clean_name)
        if clean is None:
            raise FileNotFoundError(f"SNIC clean counterpart is missing: {clean_name}")
        selected[name] = info
        selected[clean_name] = clean
        found_isos.add(iso)
        noisy_counts[iso] += 1
    if not selected:
        raise RuntimeError("No requested SNIC TIFF pairs were found in the archive")
    missing_isos = sorted(noisy_isos - found_isos)
    if missing_isos:
        raise RuntimeError(f"SNIC archive is missing requested noisy ISO levels: {missing_isos}")
    if len(set(noisy_counts.values())) != 1:
        raise RuntimeError(f"SNIC archive has inconsistent scene counts by ISO: {dict(noisy_counts)}")
    return [selected[name] for name in sorted(selected)]


def extract_entry(remote: RemoteZip, info, root: Path) -> str:  # noqa: ANN001
    destination = root / info.filename
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and destination.stat().st_size == info.file_size:
        crc = 0
        with destination.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                crc = zlib.crc32(block, crc)
        if crc & 0xFFFFFFFF == info.CRC:
            temporary.unlink(missing_ok=True)
            return "existing"
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    crc = 0
    size = 0
    with remote.open(info) as source, temporary.open("wb") as output:
        while block := source.read(8 * 1024 * 1024):
            output.write(block)
            size += len(block)
            crc = zlib.crc32(block, crc)
    if size != info.file_size:
        raise IOError(
            f"Incomplete SNIC extraction for {info.filename}: "
            f"{size} != {info.file_size}"
        )
    if crc & 0xFFFFFFFF != info.CRC:
        raise IOError(
            f"CRC mismatch for {info.filename}: {crc & 0xFFFFFFFF:08x} != {info.CRC:08x}"
        )
    temporary.replace(destination)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-id",
        type=int,
        action="append",
        dest="file_ids",
        help="Dataverse data-file ID; repeat for multiple archives.",
    )
    parser.add_argument(
        "--iso",
        type=int,
        action="append",
        dest="isos",
        help="Noisy ISO to retain; defaults to 1600, 3200, 6400, and 12800.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_paper_path("../data/sources/snic/sony_a7r_iii"),
    )
    parser.add_argument(
        "--manifest-name",
        default="download_manifest.json",
        help="Manifest filename inside --output; use distinct names for parallel archives.",
    )
    args = parser.parse_args()
    file_ids = args.file_ids or list(DEFAULT_ARCHIVES)
    noisy_isos = set(args.isos or (1600, 3200, 6400, 12800))
    if len(file_ids) != len(set(file_ids)):
        parser.error("--file-id values must be unique")
    if not noisy_isos or any(iso <= 0 or iso > 1_000_000 for iso in noisy_isos):
        parser.error("--iso values must be positive and plausible")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / args.manifest_name
    if Path(args.manifest_name).name != args.manifest_name:
        parser.error("--manifest-name must be a filename, not a path")

    report = {
        "schema_version": 1,
        "dataset": "SNIC",
        "doi": "10.7910/DVN/SGHDCP",
        "license": "MIT",
        "camera": "Sony A7R III",
        "integrity": "size_and_zip_crc32_verified",
        "noisy_isos": sorted(noisy_isos),
        "complete": False,
        "archives": [],
    }
    atomic_json(manifest_path, report)
    for file_id in file_ids:
        url = signed_download_url(file_id)
        archive_name = DEFAULT_ARCHIVES.get(file_id, f"dataverse_{file_id}.zip")
        archive_root = output / Path(archive_name).stem
        with RemoteZip(url) as remote:
            entries = selected_entries(remote, noisy_isos)
            states = {"downloaded": 0, "existing": 0}
            for info in tqdm(entries, desc=archive_name, unit="file"):
                states[extract_entry(remote, info, archive_root)] += 1
            report["archives"].append(
                {
                    "datafile_id": file_id,
                    "name": archive_name,
                    "entries": len(entries),
                    "uncompressed_bytes": sum(info.file_size for info in entries),
                    "compressed_bytes": sum(info.compress_size for info in entries),
                    "states": states,
                    "zip_entries": [
                        {
                            "name": info.filename,
                            "crc32": f"{info.CRC:08x}",
                            "size": info.file_size,
                            "compressed_size": info.compress_size,
                        }
                        for info in entries
                    ],
                }
            )
        atomic_json(manifest_path, report)
    report["complete"] = True
    atomic_json(manifest_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
