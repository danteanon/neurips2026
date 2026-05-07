#!/usr/bin/env bash
# Clone the competitor repositories used by the benchmarking adapters,
# into ../benchmarking/repos/.
#
# Idempotent: re-running pulls latest changes for repos that already
# exist (no `git pull` is forced — we just skip them).
#
# Repos:
#   - ResDepth   : aerial CHM baseline (DFC corruption sweep)
#   - PromptDA   : indoor metric-depth baseline (ARKitScenes sweep)
#   - Open-Canopy: aerial CHM baseline (DFC + Open-Canopy sweep)
#   - ARKitScenes: official downloader for the upsampling Validation
#                  split used by run_hp7_arkit_benchmark.sh
#
# This is the same set of clones documented in
# benchmarking/README.md §Phase 1b.

set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPOS_DIR="$PACKAGE_ROOT/benchmarking/repos"
mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

clone_if_absent() {
    local url="$1"
    local name="$2"
    if [[ -d "$name/.git" ]]; then
        echo "==> $name already cloned, skipping."
    else
        echo "==> Cloning $name from $url"
        git clone --depth 1 "$url" "$name"
    fi
}

clone_if_absent https://github.com/prs-eth/ResDepth.git           ResDepth
clone_if_absent https://github.com/DepthAnything/PromptDA.git     PromptDA
clone_if_absent https://github.com/fajwel/Open-Canopy.git         Open-Canopy
clone_if_absent https://github.com/apple/ARKitScenes.git          ARKitScenes

# ResDepth ships its pretrained weights via the same repo. Fetch them
# now so run_all_benchmarks.sh can find them.
RESDEPTH_CKPT_DIR="$REPOS_DIR/ResDepth/logs/pretrained_models"
RESDEPTH_CKPT="$RESDEPTH_CKPT_DIR/ResDepth-stereo_trained_on_BER_ZUR/checkpoints/Model_best.pth"
if [[ ! -f "$RESDEPTH_CKPT" ]]; then
    echo "==> Downloading ResDepth pretrained weights"
    mkdir -p "$RESDEPTH_CKPT_DIR"
    # Upstream provides this via a git LFS sidecar; check the repo
    # README for the most current download URL if this redirects.
    pushd "$REPOS_DIR/ResDepth" >/dev/null
    git lfs pull 2>/dev/null \
        || echo "    (git lfs not configured; weights may need manual download)"
    popd >/dev/null
fi

echo ""
echo "==> All competitor repos cloned into $REPOS_DIR"
