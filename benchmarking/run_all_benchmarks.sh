#!/usr/bin/env bash
# Reproducer entry point for the NeurIPS reviewer package.
#
# Runs the full corruption sweep (12 regimes: clean, shifted, blurred,
# masked, degraded, zero, shift_{0,4,8,16,24,48}, cutout_{10,30,50}pct,
# zero_chm, zero_image) for our CATT model on the DFC val split.
#
# Prerequisites (run once via setup.sh):
#   - pip install -r requirements.txt
#   - python scripts/download_weights.py     # → weights/B9_last.ckpt
#   - bash scripts/download_data.sh --minimal  # → data/synrs3d/dfc/
#
# Optional: set CATT_CKPT, DATA_DIR, DEVICE before invoking to override
# the bundled defaults.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_ROOT="$(cd "$BENCH_DIR/.." && pwd)"

CATT_CKPT="${CATT_CKPT:-$PACKAGE_ROOT/weights/P1_catt_epoch17.ckpt}"
DATA_DIR="${DATA_DIR:-$PACKAGE_ROOT/data/synrs3d/dfc}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-$BENCH_DIR/results}"
SWEEP="$BENCH_DIR/eval/corruption_sweep.py"

if [[ ! -f "$CATT_CKPT" ]]; then
    echo "ERROR: CATT checkpoint not found at $CATT_CKPT" >&2
    echo "Hint: run 'python scripts/download_weights.py' first." >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found at $DATA_DIR" >&2
    echo "Hint: run 'bash scripts/download_data.sh --minimal' first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  CATT corruption sweep — $(date -Is)"
echo "  Checkpoint: $CATT_CKPT"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Device:     $DEVICE"
echo "========================================"

python3 "$SWEEP" \
    --adapter ours \
    --checkpoint "$CATT_CKPT" \
    --dataset dfc_val \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --save_images_frac 0.02

echo ""
echo "Sweep complete. Aggregated tables:"
python3 "$BENCH_DIR/eval/generate_tables.py" \
    --results_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    || echo "(generate_tables.py raised; result JSONs are still in $OUTPUT_DIR)"
