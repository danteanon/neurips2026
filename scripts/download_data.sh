#!/usr/bin/env bash
# Download the datasets used in this submission.
#
# Modes:
#   --minimal  (default)  DFC18 + DFC19 only (~22 GB; reproduces eval).
#   --full                Also pulls SynRS3D + Open-Canopy (~620 GB total;
#                         required for retraining).
#
# All downloads land under ../data/synrs3d/ inside this reviewer
# package. See ../DATA.md for licences and citations.

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PACKAGE_ROOT/data/synrs3d"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

MODE="${1:-${MODE:---minimal}}"
case "$MODE" in
    --minimal|minimal) FULL=0 ;;
    --full|full)       FULL=1 ;;
    *)
        echo "Usage: $0 [--minimal|--full]" >&2
        exit 2
        ;;
esac

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' not on PATH. Run 'pip install -r requirements.txt' first." >&2
        exit 1
    fi
}

# -------------------------------------------------------------------------
# DFC18 + DFC19 (always)  — Google Drive zips, gdown
# -------------------------------------------------------------------------
need gdown
mkdir -p dfc
cd dfc

if [[ ! -d DFC18 ]]; then
    echo "==> Downloading DFC18 (~10 GB)"
    gdown 1rq8w7YT25y2kxxRhuIpI68QeZmX1GZ0F -O DFC18.zip
    unzip -q DFC18.zip
    rm -f DFC18.zip
else
    echo "==> DFC18 already present, skipping."
fi

if [[ ! -d DFC19 && ! -d DFC19_JAX ]]; then
    echo "==> Downloading DFC19 (~12 GB)"
    gdown 1eoF16sxIHOQ5928SrboMqbi686sfKFLF -O DFC19.zip
    mkdir -p DFC19
    unzip -q DFC19.zip -d DFC19
    rm -f DFC19.zip

    # DFC19 zip extracts flat (no DFC19/ parent in some mirrors).
    # Create JAX/OMA aliases by filename-prefix splitting (same trick the
    # SynRS3D fetch script uses upstream).
    echo "==> Building DFC19_JAX / DFC19_OMA convenience symlinks"
    for city in JAX OMA; do
        target_dir="DFC19_${city}"
        mkdir -p "$target_dir"
        find DFC19 -type f -name "${city}_*.tif" -exec ln -sf "$(realpath {})" "$target_dir/" \;
    done
else
    echo "==> DFC19 already present, skipping."
fi

cd "$DATA_DIR"

if [[ "$FULL" == "0" ]]; then
    echo ""
    echo "==> Minimal data download complete. ($DATA_DIR/dfc/)"
    echo "    Run with --full to also pull SynRS3D + Open-Canopy."
    exit 0
fi

# -------------------------------------------------------------------------
# SynRS3D synthetic — Hugging Face dataset
# -------------------------------------------------------------------------
need hf
if [[ ! -d SynRS3D ]] || [[ -z "$(ls -A SynRS3D 2>/dev/null)" ]]; then
    echo "==> Downloading SynRS3D (~206 GB) — this will take a while."
    HF_HUB_ENABLE_HF_TRANSFER=1 \
        hf download JTRNEO/SynRS3D --repo-type dataset --local-dir SynRS3D
    echo "==> Unpacking SynRS3D zips in place"
    pushd SynRS3D >/dev/null
    for z in *.zip; do
        [[ -f "$z" ]] || continue
        echo "    unzip $z"
        unzip -nq "$z"
    done
    popd >/dev/null
else
    echo "==> SynRS3D already present, skipping."
fi

# -------------------------------------------------------------------------
# Open-Canopy — Hugging Face dataset
# -------------------------------------------------------------------------
if [[ ! -d open_canopy ]] || [[ -z "$(ls -A open_canopy 2>/dev/null)" ]]; then
    echo "==> Downloading Open-Canopy (~393 GB)"
    HF_HUB_ENABLE_HF_TRANSFER=1 \
        hf download AI4Forest/Open-Canopy --repo-type dataset \
            --local-dir open_canopy
else
    echo "==> Open-Canopy already present, skipping."
fi

echo ""
echo "==> Full data download complete. ($DATA_DIR)"
