#!/usr/bin/env bash
# Hp7 vs PromptDA — paired prompt-corruption sweep on the ARKitScenes
# upsampling validation set.
#
# Reproduces benchmarking/results/arkitscenes/results_hp7_vs_promptda_arkit_summary.md
# Hp7 (our DINOv2 + DA-V2 cross-attn model) is compared head-to-head
# against the published PromptDA-ViTL baseline under five regimes:
#   - clean
#   - shift_24, shift_48
#   - cutout_25pct, cutout_50pct
#
# Prerequisites (run once via setup.sh):
#   - pip install -r requirements.txt
#   - python scripts/download_weights.py --variant hp7   # → weights/Hp7_hypersim_last.ckpt
#   - bash   scripts/download_data.sh --arkitscenes      # → data/arkitscenes/upsampling/
#   - bash   scripts/clone_competitor_repos.sh           # → benchmarking/repos/PromptDA
#   - PromptDA pulls its own checkpoint on first call (cached under ~/.cache/huggingface)
#
# Optional env-var overrides:
#   HP7_CKPT, ARKIT_DIR, DEVICE, MAX_FRAMES, OUT_DIR
#
# Default budget is 2000 round-robin-sampled frames (≈ 35 min on an
# RTX A6000; 20 000 forward passes total).

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_ROOT="$(cd "$BENCH_DIR/.." && pwd)"

HP7_CKPT="${HP7_CKPT:-$PACKAGE_ROOT/weights/Hp7_hypersim_last.ckpt}"
ARKIT_DIR="${ARKIT_DIR:-$PACKAGE_ROOT/data/arkitscenes/upsampling}"
DEVICE="${DEVICE:-cuda:0}"
MAX_FRAMES="${MAX_FRAMES:-2000}"
OUT_DIR="${OUT_DIR:-$BENCH_DIR/results/arkitscenes}"
SCRIPT="$BENCH_DIR/eval/hp7_vs_promptda_arkit.py"

if [[ ! -f "$HP7_CKPT" ]]; then
    echo "ERROR: Hp7 checkpoint not found at $HP7_CKPT" >&2
    echo "Hint: run 'python scripts/download_weights.py --variant hp7' first." >&2
    exit 1
fi

if [[ ! -d "$ARKIT_DIR/Validation" ]]; then
    echo "ERROR: ARKitScenes validation dir not found at $ARKIT_DIR/Validation" >&2
    echo "Hint: run 'bash scripts/download_data.sh --arkitscenes' first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  Hp7 vs PromptDA — ARKitScenes sweep — $(date -Is)"
echo "  Hp7 ckpt:    $HP7_CKPT"
echo "  ARKit data:  $ARKIT_DIR"
echo "  Max frames:  $MAX_FRAMES"
echo "  Output:      $OUT_DIR"
echo "  Device:      $DEVICE"
echo "========================================"

python3 "$SCRIPT" \
    --hp7_ckpt "$HP7_CKPT" \
    --data_dir "$ARKIT_DIR" \
    --max_total_frames "$MAX_FRAMES" \
    --device "$DEVICE" \
    --out_json     "$OUT_DIR/results_hp7_vs_promptda_arkit_full.json" \
    --out_summary  "$OUT_DIR/results_hp7_vs_promptda_arkit_summary.json" \
    --out_markdown "$OUT_DIR/results_hp7_vs_promptda_arkit_summary.md"

echo ""
echo "Sweep complete. Aggregated tables:"
echo "  Per-frame rows : $OUT_DIR/results_hp7_vs_promptda_arkit_full.json"
echo "  Aggregate JSON : $OUT_DIR/results_hp7_vs_promptda_arkit_summary.json"
echo "  Markdown table : $OUT_DIR/results_hp7_vs_promptda_arkit_summary.md"
