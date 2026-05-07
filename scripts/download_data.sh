#!/usr/bin/env bash
# Download the datasets used in this submission.
#
# Modes:
#   --minimal  (default)  DFC18 + DFC19 only (~22 GB; reproduces the
#                         CATT corruption sweep on aerial CHM).
#   --arkitscenes         ARKitScenes upsampling split (~120 GB; needed
#                         for the Hp7 vs PromptDA prompt-corruption
#                         sweep on indoor metric depth).
#   --full                Everything: minimal + arkitscenes + SynRS3D +
#                         Open-Canopy (~740 GB total; required only
#                         for retraining).
#
# Aerial datasets land under ../data/synrs3d/, ARKitScenes lands under
# ../data/arkitscenes/. See ../DATA.md for licences and citations.

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PACKAGE_ROOT/data/synrs3d"
ARKIT_DIR="$PACKAGE_ROOT/data/arkitscenes"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

MODE="${1:-${MODE:---minimal}}"
case "$MODE" in
    --minimal|minimal)         FULL=0; DO_ARKIT=0 ;;
    --arkitscenes|arkitscenes) FULL=0; DO_ARKIT=1; SKIP_AERIAL=1 ;;
    --full|full)               FULL=1; DO_ARKIT=1 ;;
    *)
        echo "Usage: $0 [--minimal|--arkitscenes|--full]" >&2
        exit 2
        ;;
esac
SKIP_AERIAL="${SKIP_AERIAL:-0}"

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' not on PATH. Run 'pip install -r requirements.txt' first." >&2
        exit 1
    fi
}

if [[ "$SKIP_AERIAL" == "1" ]]; then
    echo "==> Skipping aerial-CHM datasets (mode=$MODE)."
else

# -------------------------------------------------------------------------
# DFC18 + DFC19 (always for --minimal/--full)  — Google Drive zips, gdown
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

fi  # SKIP_AERIAL guard

# -------------------------------------------------------------------------
# ARKitScenes (upsampling split) — pulled by the official downloader
# from https://github.com/apple/ARKitScenes
# -------------------------------------------------------------------------
if [[ "$DO_ARKIT" == "1" ]]; then
    echo ""
    echo "==> ARKitScenes upsampling split (~120 GB across train + val)"
    REPOS_DIR="$PACKAGE_ROOT/benchmarking/repos"
    ARKIT_REPO="$REPOS_DIR/ARKitScenes"
    if [[ ! -d "$ARKIT_REPO" ]]; then
        mkdir -p "$REPOS_DIR"
        echo "    cloning ARKitScenes downloader → $ARKIT_REPO"
        git clone --depth 1 https://github.com/apple/ARKitScenes "$ARKIT_REPO"
    fi
    CSV_FILE="$ARKIT_REPO/depth_upsampling/upsampling_train_val_splits.csv"
    if [[ ! -f "$CSV_FILE" ]]; then
        echo "ERROR: $CSV_FILE missing — clone of ARKitScenes is incomplete." >&2
        exit 1
    fi
    mkdir -p "$ARKIT_DIR"

    # Validation only is sufficient for the Hp7 vs PromptDA benchmark.
    # Pass ARKIT_FULL=1 to also pull the 1970-video Training split (only
    # needed if you want to retrain or evaluate on Train).
    SPLITS=("Validation")
    if [[ "${ARKIT_FULL:-0}" == "1" ]]; then
        SPLITS=("Training" "Validation")
    fi
    for split in "${SPLITS[@]}"; do
        echo "    [→] downloading $split split"
        ( cd "$ARKIT_REPO" && \
          python3 download_data.py upsampling \
              --video_id_csv "$CSV_FILE" \
              --split "$split" \
              --download_dir "$ARKIT_DIR/upsampling" )
    done
fi

if [[ "$FULL" == "0" ]]; then
    echo ""
    if [[ "$DO_ARKIT" == "1" ]]; then
        echo "==> Download complete. (ARKitScenes only)"
    else
        echo "==> Minimal data download complete. ($DATA_DIR/dfc/)"
        echo "    Run with --arkitscenes for the indoor metric-depth eval,"
        echo "    or --full to also pull SynRS3D + Open-Canopy."
    fi
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
