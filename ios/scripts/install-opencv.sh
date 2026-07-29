#!/usr/bin/env bash
set -euo pipefail

readonly VERSION="4.13.0"
readonly SHA256="7ac1a77d21aa9556422e08d8b7ffcc30dfa9ebc0351a0ff32216395e8b14bede"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DESTINATION="${ROOT}/Vendor/opencv2.framework"

if [[ -f "${DESTINATION}/opencv2" ]]; then
    exit 0
fi

archive="$(mktemp)"
expanded="$(mktemp -d)"
trap 'rm -f "${archive}"; rm -rf "${expanded}"' EXIT

curl -L --fail --retry 3 \
    "https://github.com/opencv/opencv/releases/download/${VERSION}/opencv-${VERSION}-ios-framework.zip" \
    -o "${archive}"
echo "${SHA256}  ${archive}" | sha256sum --check -
unzip -q "${archive}" -d "${expanded}"

mkdir -p "${ROOT}/Vendor"
cp -a "${expanded}/opencv2.framework" "${DESTINATION}"
llvm-lipo "${DESTINATION}/Versions/A/opencv2" \
    -thin arm64 \
    -output "${DESTINATION}/Versions/A/opencv2.arm64"
mv "${DESTINATION}/Versions/A/opencv2.arm64" \
    "${DESTINATION}/Versions/A/opencv2"
