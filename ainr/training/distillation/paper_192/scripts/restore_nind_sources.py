#!/usr/bin/env python3
"""Restore only the NIND files referenced by an audited source manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from common import atomic_json, resolve_paper_path


USER_AGENT = (
    "AlphaCaptureLab-AINR/0.1 "
    "(https://github.com/Rkm1999/alpha_capture_lab; dataset preparation)"
)
API = "https://commons.wikimedia.org/w/api.php"


def canonical_upload_url(filename: str) -> str:
    parameters = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": f"File:{filename}",
            "iiprop": "url",
        }
    )
    request = urllib.request.Request(
        f"{API}?{parameters}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        document = json.load(response)
    page = next(iter(document["query"]["pages"].values()))
    information = page.get("imageinfo", [None])[0]
    if information is None:
        raise FileNotFoundError(f"Commons has no image record for {filename}")
    return str(information["url"])


def download(relative: str, root: Path, retries: int, delay: float) -> dict:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    filename = Path(relative).name
    url = canonical_upload_url(filename)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            offset = partial.stat().st_size if partial.exists() else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", 200))
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if partial.stat().st_size < 1024:
                raise RuntimeError(f"Downloaded payload is too small: {partial}")
            os.replace(partial, destination)
            return {
                "file": relative,
                "bytes": destination.stat().st_size,
                "url": url,
                "status": "downloaded",
            }
        except Exception as error:
            if attempt + 1 == retries:
                return {"file": relative, "url": url, "status": "error", "error": str(error)}
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parents[1] / "data/high_iso_source_manifest.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).parents[2] / "data/sources",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    manifest = resolve_paper_path(args.manifest)
    source_root = resolve_paper_path(args.source_root)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    records = document.get("records", document)
    paths = sorted(
        {
            str(record[field])
            for record in records
            if record.get("dataset") == "nind"
            for field in ("input", "clean")
        }
    )
    existing = [
        {
            "file": relative,
            "bytes": (source_root / relative).stat().st_size,
            "status": "existing",
        }
        for relative in paths
        if (source_root / relative).is_file()
    ]
    missing = [relative for relative in paths if not (source_root / relative).is_file()]
    results = list(existing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, relative, source_root, args.retries, args.delay): relative
            for relative in missing
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(missing)}] {result['status']}: {result['file']}",
                flush=True,
            )
            if args.workers == 1 and args.delay > 0:
                time.sleep(args.delay)

    results.sort(key=lambda row: row["file"])
    errors = [row for row in results if row["status"] == "error"]
    report = {
        "source_manifest": str(manifest),
        "source_root": str(source_root),
        "required_files": len(paths),
        "restored_files": sum(row["status"] == "downloaded" for row in results),
        "existing_files": len(existing),
        "errors": len(errors),
        "bytes": sum(int(row.get("bytes", 0)) for row in results),
        "files": results,
    }
    atomic_json(source_root / "nind" / "RESTORE_REPORT.json", report)
    if errors:
        raise RuntimeError(f"Failed to restore {len(errors)} NIND files")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
