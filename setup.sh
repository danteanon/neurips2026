#!/usr/bin/env bash
# One-shot reviewer setup. Idempotent: re-running only does the steps
# that haven't completed yet.
#
# Steps:
#   1. pip install -r requirements.txt  (skipped if SETUP_NO_PIP=1)
#   2. Download the B9 checkpoint into weights/
#   3. Download the minimal eval data (DFC18+DFC19) into data/
#   4. Clone the three competitor repos into benchmarking/repos/
#
# Override granularity:
#   SETUP_NO_PIP=1     skip pip install
#   SETUP_FULL_DATA=1  download SynRS3D + Open-Canopy as well (~620 GB)
#   SETUP_NO_REPOS=1   skip cloning competitor repos

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PACKAGE_ROOT"

echo "==> setup.sh starting in $PACKAGE_ROOT"

if [[ "${SETUP_NO_PIP:-0}" != "1" ]]; then
    echo ""
    echo "==> [1/4] pip install -r requirements.txt"
    pip install -r requirements.txt
else
    echo "==> [1/4] (skipped, SETUP_NO_PIP=1)"
fi

echo ""
echo "==> [2/4] Downloading B9 checkpoint"
python scripts/download_weights.py

echo ""
echo "==> [3/4] Downloading eval data"
if [[ "${SETUP_FULL_DATA:-0}" == "1" ]]; then
    bash scripts/download_data.sh --full
else
    bash scripts/download_data.sh --minimal
fi

if [[ "${SETUP_NO_REPOS:-0}" != "1" ]]; then
    echo ""
    echo "==> [4/4] Cloning competitor repos"
    bash scripts/clone_competitor_repos.sh
else
    echo "==> [4/4] (skipped, SETUP_NO_REPOS=1)"
fi

echo ""
echo "==> setup.sh complete"
echo "    Now try: bash benchmarking/run_all_benchmarks.sh"
