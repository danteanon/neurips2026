#!/usr/bin/env bash
# One-shot reviewer setup. Idempotent: re-running only does the steps
# that haven't completed yet.
#
# Steps:
#   1. pip install -r requirements.txt          (skipped if SETUP_NO_PIP=1)
#   2. Download checkpoints into weights/       (CATT P1 + Hp7)
#   3. Download eval data into data/            (minimal DFC by default)
#   4. Clone competitor repos into benchmarking/repos/
#
# Override granularity:
#   SETUP_NO_PIP=1        skip pip install
#   SETUP_NO_HP7=1        skip the Hp7 checkpoint download (CATT only)
#   SETUP_NO_CATT=1       skip the CATT checkpoint download (Hp7 only)
#   SETUP_ARKIT=1         also download ARKitScenes Validation (~80 GB)
#                         needed for benchmarking/run_hp7_arkit_benchmark.sh
#   SETUP_FULL_DATA=1     download everything: DFC + ARKit + SynRS3D +
#                         Open-Canopy (~700 GB)
#   SETUP_NO_REPOS=1      skip cloning competitor repos

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
echo "==> [2/4] Downloading checkpoints"
if [[ "${SETUP_NO_CATT:-0}" != "1" ]]; then
    python scripts/download_weights.py --variant catt
else
    echo "    (skipped CATT, SETUP_NO_CATT=1)"
fi
if [[ "${SETUP_NO_HP7:-0}" != "1" ]]; then
    python scripts/download_weights.py --variant hp7
else
    echo "    (skipped Hp7, SETUP_NO_HP7=1)"
fi

echo ""
echo "==> [3/4] Downloading eval data"
if [[ "${SETUP_FULL_DATA:-0}" == "1" ]]; then
    bash scripts/download_data.sh --full
else
    bash scripts/download_data.sh --minimal
    if [[ "${SETUP_ARKIT:-0}" == "1" ]]; then
        bash scripts/download_data.sh --arkitscenes
    fi
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
echo "    Aerial benchmark : bash benchmarking/run_all_benchmarks.sh"
echo "    Indoor benchmark : bash benchmarking/run_hp7_arkit_benchmark.sh  (needs SETUP_ARKIT=1)"
