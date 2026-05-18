#!/usr/bin/env bash
# Fetch the Tanks-and-Temples "Truck" source Gaussian-splat checkpoint
# and verify its SHA256 against the value pinned in source_manifest.json.
# The convert pipeline refuses to proceed on a SHA mismatch.
#
# Source: Tanks-and-Temples training set, "Truck" scene.
# License: CC-BY 4.0 (Knapitsch et al., SIGGRAPH 2017).
#
# Implementation note: this script's URL points at a pretrained
# Gaussian-splat checkpoint if one is available in a public model zoo
# (e.g., the gsplat or Nerfstudio release). If no pretrained checkpoint
# is available at run time, swap to a "fetch raw images + ns-train
# splatfacto" path — see the comment block below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${HERE}/source_manifest.json"
OUTPUT_PATH="${HERE}/source.ply"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "error: ${MANIFEST_PATH} not found." >&2
  exit 2
fi

URL="$(python3 -c "import json,sys; print(json.load(open('${MANIFEST_PATH}'))['source_url'])")"
EXPECTED_SHA="$(python3 -c "import json,sys; print(json.load(open('${MANIFEST_PATH}'))['expected_sha256'])")"

echo "Fetching ${URL} -> ${OUTPUT_PATH}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

# Prefer curl; fall back to wget.
if command -v curl >/dev/null 2>&1; then
  curl --location --fail --silent --show-error --output "${OUTPUT_PATH}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --quiet --output-document="${OUTPUT_PATH}" "${URL}"
else
  echo "error: neither curl nor wget is on PATH; cannot fetch source." >&2
  exit 2
fi

ACTUAL_SHA="$(python3 -c "
import hashlib, sys
h = hashlib.sha256()
with open('${OUTPUT_PATH}', 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())
")"

if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "error: SHA256 mismatch for ${OUTPUT_PATH}" >&2
  echo "  expected: ${EXPECTED_SHA}" >&2
  echo "  got:      ${ACTUAL_SHA}" >&2
  exit 3
fi

echo "OK: ${OUTPUT_PATH} (sha256=${ACTUAL_SHA})"

# -----------------------------------------------------------------------------
# Fallback path: train a Gaussian splat from the raw TNT Truck images.
#
# If the URL in source_manifest.json no longer resolves to a pretrained
# checkpoint, replace it with the URL of the Tanks-and-Temples Truck
# training-image archive and uncomment the following lines. The convert
# pipeline records which path ran in convert_log.json.
#
#   ns-train splatfacto \
#     --data    "${HERE}/raw_images/" \
#     --output-dir "${HERE}/ns_output/" \
#     --max-num-iterations 30000
#   cp "${HERE}/ns_output/.../splat.ply" "${OUTPUT_PATH}"
# -----------------------------------------------------------------------------
