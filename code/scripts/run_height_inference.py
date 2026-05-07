#!/usr/bin/env python
"""
Run height inference on prepared eval samples with different CHM variants.

Loads a checkpoint, runs the model on each sample × each CHM variant,
and produces comparison figures.

Usage:
    python scripts/run_height_inference.py \
        --checkpoint checkpoints/tree-height/experiment_20260416_191454/last.ckpt \
        --config configs/height/train_config-height-synrs3d.yaml \
        --samples_dir data/tg/tree_height/synrs3d/output_analysis \
        --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from model.get_model import get_model
from lightning_modules.height_module import HeightEstimationModule
from utils.normalisations import Normalization


def height_to_colormap(height, vmin=0, vmax=None):
    if vmax is None:
        vmax = max(height.max(), 1.0)
    normed = np.clip((height - vmin) / (vmax - vmin), 0, 1)
    colored = (cm.viridis(normed) * 255).astype(np.uint8)
    return colored[:, :, :3]


def run_inference(model, image_np, chm_np, device, image_size=512):
    """Run model on a single (image, chm) pair."""
    normalize = Normalization(method="8bit")
    img = normalize.apply(image_np)

    # Resize to model input size
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
    img_pil = img_pil.resize((image_size, image_size), PILImage.BILINEAR)
    img_t = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float() / 255.0

    chm_pil = PILImage.fromarray(chm_np.astype(np.float32), mode="F")
    chm_pil = chm_pil.resize((image_size, image_size), PILImage.BILINEAR)
    chm_t = torch.from_numpy(np.array(chm_pil)).float().unsqueeze(0)

    img_t = img_t.unsqueeze(0).to(device)
    chm_t = chm_t.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t, chm_t)

    return pred.squeeze().cpu().numpy()


def make_comparison_figure(sample_id, rgb, gt, predictions, vmax):
    """Create a comparison figure: RGB | GT | pred per variant."""
    n_variants = len(predictions)
    fig, axes = plt.subplots(2, max(n_variants, 3), figsize=(6 * max(n_variants, 3), 12))

    # Top row: RGB, GT, then per-variant predictions
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB Input", fontsize=13, fontweight="bold")

    axes[0, 1].imshow(gt, cmap="viridis", vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"GT Height (max={gt.max():.1f}m)", fontsize=13, fontweight="bold")

    # Hide unused top-row cells
    for j in range(2, axes.shape[1]):
        axes[0, j].axis("off")

    # Bottom row: predictions for each CHM variant
    variant_labels = {
        "chm_clean": "Clean CHM",
        "chm_shifted": "Shifted CHM",
        "chm_masked": "Partial CHM (~30% missing)",
        "chm_degraded": "Degraded CHM",
    }

    for j, (variant, pred) in enumerate(predictions.items()):
        ax = axes[1, j]
        ax.imshow(pred, cmap="viridis", vmin=0, vmax=vmax)
        label = variant_labels.get(variant, variant)
        mae = np.abs(pred - gt)[gt > 0].mean() if (gt > 0).any() else 0
        ax.set_title(f"Pred: {label}\nMAE={mae:.2f}m", fontsize=12)

    for j in range(len(predictions), axes.shape[1]):
        axes[1, j].axis("off")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"Sample: {sample_id}", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_error_figure(sample_id, gt, predictions, vmax):
    """Create error maps for each variant."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    variant_labels = {
        "chm_clean": "Clean CHM",
        "chm_shifted": "Shifted CHM",
        "chm_masked": "Partial CHM",
        "chm_degraded": "Degraded CHM",
    }

    error_vmax = vmax * 0.3

    for j, (variant, pred) in enumerate(predictions.items()):
        error = np.abs(pred - gt)
        mask = gt > 0
        mae = error[mask].mean() if mask.any() else 0
        rmse = np.sqrt((error[mask] ** 2).mean()) if mask.any() else 0

        axes[j].imshow(error, cmap="hot", vmin=0, vmax=error_vmax)
        label = variant_labels.get(variant, variant)
        axes[j].set_title(f"{label}\nMAE={mae:.2f}m  RMSE={rmse:.2f}m", fontsize=11)
        axes[j].axis("off")

    fig.suptitle(f"Absolute Error — {sample_id}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/height/train_config-height-synrs3d.yaml")
    parser.add_argument("--samples_dir", default="data/tg/tree_height/synrs3d/output_analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config and model
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading model...")
    model = get_model(config)
    module = HeightEstimationModule.load_from_checkpoint(
        args.checkpoint, model=model, config=config, map_location=device
    )
    module.eval()
    module.to(device)
    net = module.model

    # Load manifest
    manifest_path = os.path.join(args.samples_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    global_vmax = manifest["global_vmax"]
    variants = manifest["chm_variants"]

    output_dir = os.path.join(args.samples_dir, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    summary = []

    for sample_info in manifest["samples"]:
        sid = sample_info["id"]
        sample_dir = os.path.join(args.samples_dir, sid)
        print(f"\nProcessing {sid} (max_height={sample_info['max_height']:.1f}m)...")

        # Load data
        rgb = np.array(Image.open(os.path.join(sample_dir, "rgb.png")))
        gt = np.load(os.path.join(sample_dir, "gt_height.npy"))

        # Read the raw TIF for proper normalization
        try:
            import rasterio
            subset_dir = os.path.join(
                config["train_data_loader"]["data_dir"],
                manifest["subset"],
            )
            with rasterio.open(os.path.join(subset_dir, "opt", f"{sid}.tif")) as src:
                image_raw = src.read()[:3]
        except Exception:
            image_raw = np.array(Image.open(os.path.join(sample_dir, "rgb.png"))).transpose(2, 0, 1)

        predictions = {}
        sample_summary = {"id": sid}

        for variant in variants:
            chm = np.load(os.path.join(sample_dir, f"{variant}.npy"))
            pred = run_inference(net, image_raw, chm, device, args.image_size)

            # Resize pred back to GT size for metrics
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray(pred.astype(np.float32), mode="F")
            pred_resized = np.array(pred_pil.resize((gt.shape[1], gt.shape[0]), PILImage.BILINEAR))
            predictions[variant] = pred_resized

            # Compute metrics
            mask = gt > 0
            if mask.any():
                mae = np.abs(pred_resized - gt)[mask].mean()
                rmse = np.sqrt(((pred_resized - gt)[mask] ** 2).mean())
            else:
                mae, rmse = 0, 0

            sample_summary[f"{variant}_mae"] = float(mae)
            sample_summary[f"{variant}_rmse"] = float(rmse)
            print(f"  {variant:20s}  MAE={mae:.2f}m  RMSE={rmse:.2f}m")

            # Save predicted height as npy and vis
            np.save(os.path.join(output_dir, f"{sid}_{variant}_pred.npy"), pred_resized)
            Image.fromarray(height_to_colormap(pred_resized, vmax=global_vmax)).save(
                os.path.join(output_dir, f"{sid}_{variant}_pred_vis.png")
            )

        summary.append(sample_summary)

        # Comparison figure
        fig = make_comparison_figure(sid, rgb, gt, predictions, global_vmax)
        fig.savefig(os.path.join(output_dir, f"{sid}_comparison.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Error figure
        fig = make_error_figure(sid, gt, predictions, global_vmax)
        fig.savefig(os.path.join(output_dir, f"{sid}_errors.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Summary table
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "samples": summary}, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':20s}  {'Mean MAE':>10s}  {'Mean RMSE':>10s}")
    print("-" * 45)
    for variant in variants:
        maes = [s[f"{variant}_mae"] for s in summary]
        rmses = [s[f"{variant}_rmse"] for s in summary]
        print(f"{variant:20s}  {np.mean(maes):10.2f}m  {np.mean(rmses):10.2f}m")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
