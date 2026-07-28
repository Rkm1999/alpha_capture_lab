#!/usr/bin/env python3
"""Download a gdown JSON listing with resumable parallel Google Drive GETs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


def drive_id(url: str) -> str:
    identifier = parse_qs(urlparse(url).query).get("id", [None])[0]
    if not identifier:
        raise ValueError(f"Google Drive URL has no id: {url}")
    return identifier


def jpeg_has_complete_markers(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            return False
        source.seek(-2, os.SEEK_END)
        return source.read(2) == b"\xff\xd9"


def valid_existing(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return True
    return jpeg_has_complete_markers(path)


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe output path in listing: {value!r}")
    return path


def download(row: dict[str, str], output: Path, retries: int) -> tuple[str, str]:
    relative = safe_relative_path(row["path"])
    destination = output / relative
    if valid_existing(destination):
        return "existing", row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    identifier = drive_id(row["url"])
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={identifier}&export=download&confirm=t"
    )
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 UHD-LL-downloader/1.0"})
            with urlopen(request, timeout=120) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            if temporary.stat().st_size == 0:
                raise IOError(f"Downloaded content is empty: {row['path']}")
            if destination.suffix.lower() in {".jpg", ".jpeg"} and not jpeg_has_complete_markers(
                temporary
            ):
                raise IOError(f"Downloaded content is not a complete JPEG: {row['path']}")
            os.replace(temporary, destination)
            return "downloaded", row["path"]
        except Exception as caught:  # network failures are retried and reported
            error = caught
            temporary.unlink(missing_ok=True)
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"Failed after {retries} attempts: {row['path']}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    rows = json.loads(args.listing.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Listing must be a non-empty JSON array")
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be positive")
    normalized_paths = []
    identifiers = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(
            row.get("url"), str
        ):
            raise ValueError(f"Invalid listing row {index}: expected string path and url")
        normalized_paths.append(safe_relative_path(row["path"]).as_posix())
        identifiers.append(drive_id(row["url"]))
    duplicate_paths = [path for path, count in Counter(normalized_paths).items() if count > 1]
    if duplicate_paths:
        raise ValueError(f"Listing contains duplicate output path: {duplicate_paths[0]}")
    duplicate_ids = [identifier for identifier, count in Counter(identifiers).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Listing contains duplicate Drive ID: {duplicate_ids[0]}")
    output = args.output.resolve()
    counts = {"downloaded": 0, "existing": 0}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, row, output, args.retries): row for row in rows
        }
        total = len(futures)
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = futures[future]
            try:
                state, _ = future.result()
                counts[state] += 1
            except Exception as error:
                failures.append({"path": row["path"], "error": str(error)})
            if completed % 100 == 0 or completed == total:
                print(
                    f"{completed}/{total} downloaded={counts['downloaded']} "
                    f"existing={counts['existing']} failed={len(failures)}",
                    flush=True,
                )
    report = {"files": len(rows), "counts": counts, "failures": failures}
    report_path = output / "download_report.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    if failures:
        raise RuntimeError(f"{len(failures)} UHD-LL files failed; see {report_path}")


if __name__ == "__main__":
    main()
