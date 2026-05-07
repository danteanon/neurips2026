"""Read result JSONs and produce markdown tables + matplotlib figures.

Usage
-----
    python generate_tables.py --results_dir results/ --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

MODEL_ORDER = ["ours_catt", "promptda", "resdepth", "opencanopy_pvtv2"]
MODEL_LABELS = {
    "ours_catt": "Ours (CATT)",
    "promptda": "PromptDA",
    "resdepth": "ResDepth",
    "opencanopy_pvtv2": "Open-Canopy PVTv2",
}
MODEL_COLORS = {
    "ours_catt": "#2196F3",
    "promptda": "#FF5722",
    "resdepth": "#4CAF50",
    "opencanopy_pvtv2": "#9C27B0",
}


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files into {(model, dataset, regime): metrics}."""
    results = {}
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        key = (data["model"], data["dataset"], data["regime"])
        results[key] = data
    return results


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_separator(n: int) -> str:
    return "| " + " | ".join(["---"] * n) + " |"


def table_shift_sweep(results: dict, dataset: str) -> str:
    shifts = ["shift_0", "shift_4", "shift_8", "shift_16", "shift_24", "shift_48"]
    header = ["Model"] + [s.replace("shift_", "shift=") for s in shifts]
    lines = [_md_row(header), _md_separator(len(header))]

    for model in MODEL_ORDER:
        label = MODEL_LABELS.get(model, model)
        row = [label]
        for regime in shifts:
            key = (model, dataset, regime)
            if key in results:
                row.append(f"{results[key]['mae']:.3f}")
            else:
                row.append("—")
        lines.append(_md_row(row))

    return "\n".join(lines)


def table_dependency_probes(results: dict, dataset: str) -> str:
    probes = ["shift_0", "zero_chm", "zero_image"]
    probe_labels = ["Full input", "Zero CHM", "Zero Image"]
    header = ["Model"] + probe_labels
    lines = [_md_row(header), _md_separator(len(header))]

    for model in MODEL_ORDER:
        label = MODEL_LABELS.get(model, model)
        row = [label]
        for regime in probes:
            key = (model, dataset, regime)
            if key in results:
                row.append(f"{results[key]['mae']:.3f}")
            else:
                row.append("—")
        lines.append(_md_row(row))

    return "\n".join(lines)


def table_degradation_regimes(results: dict, dataset: str) -> str:
    regimes = [
        "regime_clean", "regime_shifted", "regime_masked",
        "regime_degraded", "regime_blurred", "regime_zero",
    ]
    regime_labels = ["clean", "shifted", "masked", "degraded", "blurred", "zero"]
    header = ["Model"] + regime_labels
    lines = [_md_row(header), _md_separator(len(header))]

    for model in MODEL_ORDER:
        label = MODEL_LABELS.get(model, model)
        row = [label]
        for regime in regimes:
            key = (model, dataset, regime)
            if key in results:
                row.append(f"{results[key]['mae']:.3f}")
            else:
                row.append("—")
        lines.append(_md_row(row))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def figure_mae_vs_shift(results: dict, dataset: str, output_path: Path):
    shifts = [0, 4, 8, 16, 24, 48]
    shift_keys = [f"shift_{s}" for s in shifts]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODEL_ORDER:
        maes = []
        valid_shifts = []
        for s, sk in zip(shifts, shift_keys):
            key = (model, dataset, sk)
            if key in results:
                maes.append(results[key]["mae"])
                valid_shifts.append(s)

        if valid_shifts:
            ax.plot(
                valid_shifts, maes,
                marker="o", linewidth=2, markersize=6,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, None),
            )

    ax.set_xlabel("Spatial shift (pixels)", fontsize=12)
    ax.set_ylabel("MAE (m)", fontsize=12)
    ax.set_title("Misalignment Robustness", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def figure_dependency_probes(results: dict, dataset: str, output_path: Path):
    probes = ["shift_0", "zero_chm", "zero_image"]
    probe_labels = ["Full", "Zero CHM", "Zero Image"]

    models_with_data = []
    for model in MODEL_ORDER:
        if any((model, dataset, p) in results for p in probes):
            models_with_data.append(model)

    if not models_with_data:
        return

    x = np.arange(len(probes))
    width = 0.8 / len(models_with_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(models_with_data):
        maes = []
        for p in probes:
            key = (model, dataset, p)
            maes.append(results[key]["mae"] if key in results else 0)

        offset = (i - len(models_with_data) / 2 + 0.5) * width
        ax.bar(
            x + offset, maes, width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(probe_labels)
    ax.set_ylabel("MAE (m)", fontsize=12)
    ax.set_title("Dependency Probes", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", default="figures/")
    parser.add_argument("--dataset", default="synrs3d_val")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        logger.warning("No result JSONs found in %s", results_dir)
        return

    # Detect datasets present
    datasets = {k[1] for k in results}
    logger.info("Datasets found: %s", datasets)

    for ds in datasets:
        # Tables
        t1 = table_shift_sweep(results, ds)
        t2 = table_dependency_probes(results, ds)
        t3 = table_degradation_regimes(results, ds)

        table_path = results_dir / f"tables_{ds}.md"
        with open(table_path, "w") as f:
            f.write(f"# Benchmark Results — {ds}\n\n")
            f.write("## Table 1: Misalignment Robustness — MAE (m)\n\n")
            f.write(t1 + "\n\n")
            f.write("## Table 2: Dependency Probes — MAE (m)\n\n")
            f.write(t2 + "\n\n")
            f.write("## Table 3: Degradation Regimes — MAE (m)\n\n")
            f.write(t3 + "\n")

        logger.info("Wrote %s", table_path)

        # Figures
        figure_mae_vs_shift(results, ds, output_dir / f"mae_vs_shift_{ds}.png")
        figure_dependency_probes(results, ds, output_dir / f"dependency_probes_{ds}.png")


if __name__ == "__main__":
    main()
