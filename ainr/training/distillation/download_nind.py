#!/usr/bin/env python3
"""Download the current NIND XT1 8-bit set and retain attribution metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "AlphaCaptureLab-AINR/0.1 "
    "(https://github.com/Rkm1999/alpha_capture_lab; dataset preparation)"
)


def open_with_retry(request: urllib.request.Request):
    for delay in (0, 5, 15, 30, 60):
        if delay:
            time.sleep(delay)
        try:
            return urllib.request.urlopen(request, timeout=120)
        except HTTPError as error:
            if error.code != 429 or delay == 60:
                raise
    raise RuntimeError("unreachable")


def request_json(parameters: dict[str, str]) -> dict:
    url = API + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_with_retry(request) as response:
        return json.load(response)


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    catalog = runpy.run_path(str(args.catalog))["imageslist"]["XT1_8bit"]
    args.target.mkdir(parents=True, exist_ok=True)
    metadata = []
    errors = 0
    for entry in catalog["images"]:
        scene, *isos = entry.split(",")
        destination = args.target / scene
        destination.mkdir(parents=True, exist_ok=True)
        for iso in isos:
            name = f"NIND_{scene}_ISO{iso}.{catalog['ext']}"
            result = request_json({
                "action": "query", "format": "json", "prop": "imageinfo",
                "titles": "File:" + name.replace("_", " "),
                "iiprop": "timestamp|url|sha1|extmetadata",
                "iiurlwidth": "2048",
            })
            page = next(iter(result["query"]["pages"].values()))
            info = page.get("imageinfo", [None])[0]
            if info is None:
                print(f"Missing: {name}", file=sys.stderr)
                errors += 1
                continue
            path = destination / name
            source_url = info.get("thumburl", info["url"])
            expected_sha1 = info["sha1"] if source_url == info["url"] else None
            if not path.exists() or (expected_sha1 is not None and sha1(path) != expected_sha1):
                print(f"Downloading {name}")
                request = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
                with open_with_retry(request) as response, path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if expected_sha1 is not None and sha1(path) != expected_sha1:
                    path.unlink(missing_ok=True)
                    raise RuntimeError(f"SHA-1 verification failed for {name}")
            fields = info.get("extmetadata", {})
            metadata.append({
                "file": str(path.relative_to(args.target)),
                "source": info["descriptionurl"],
                "sha1": info["sha1"],
                "download_url": source_url,
                "license": fields.get("LicenseShortName", {}).get("value"),
                "license_url": fields.get("LicenseUrl", {}).get("value"),
                "artist": fields.get("Artist", {}).get("value"),
            })
            time.sleep(2.0)
    (args.target / "ATTRIBUTION.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(metadata)} NIND images with {errors} missing catalog entries.")


if __name__ == "__main__":
    main()
