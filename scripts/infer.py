#!/usr/bin/env python
"""Single-image inference for the CATT (P1) model.

Usage
-----
    python scripts/infer.py \
        --image  path/to/rgb.tif \
        --chm    path/to/chm_prompt.tif \
        --output path/to/predicted_height.tif

* ``--image`` and ``--chm`` may be GeoTIFFs (recommended; geotransform
  and CRS are preserved in the output) or any format Pillow can read.
* If ``--chm`` is omitted, an all-zeros prompt is used (matches the
  ``zero_chm`` corruption regime — a useful sanity check that the model
  can degrade gracefully when no prior is available).

The image is processed in non-overlapping 512 × 512 tiles per the
benchmarking convention (see ``benchmarking/README.md`` Phase 2).
Tiles smaller than 512 × 512 are zero-padded; predictions are cropped
back to the original size before stitching.

Outputs
-------
* ``--output`` : float32 GeoTIFF (or PNG if extension differs) with
  per-pixel predicted canopy height in metres.
* ``<output>.png`` : viridis colour preview alongside the height map.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from model.get_model import get_model  # noqa: E402
from lightning_modules.height_module import HeightEstimationModule  # noqa: E402
from utils.normalisations import Normalization  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE = 512  # non-overlapping tile size (matches benchmarking harness)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _read_raster(path: Path):
    """Read a raster as float32. Returns (array, geo_meta).

    Tries rasterio first (preserves CRS / transform). Falls back to
    Pillow for plain PNG/JPEG inputs.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore
        except ImportError:  # pragma: no cover
            log.warning("rasterio not available; falling back to PIL for %s",
                        path)
        else:
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
                meta = src.meta.copy()
                if arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    arr = arr.transpose(1, 2, 0)
                return arr, meta
    img = np.asarray(Image.open(path)).astype(np.float32)
    return img, None


def _write_geotiff(path: Path, arr: np.ndarray, meta: dict | None):
    """Write a single-band float32 GeoTIFF if rasterio + geo_meta are
    available; otherwise fall back to a numpy ``.npy`` dump and a PNG
    preview only."""
    try:
        import rasterio  # type: ignore
    except ImportError:
        rasterio = None  # type: ignore

    if path.suffix.lower() in (".tif", ".tiff") and rasterio and meta:
        meta = meta.copy()
        meta.update({
            "count": 1,
            "dtype": "float32",
            "height": arr.shape[0],
            "width": arr.shape[1],
        })
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr.astype(np.float32), 1)
        log.info("wrote GeoTIFF: %s", path)
    else:
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, arr.astype(np.float32))
        log.info("wrote numpy: %s (no GeoTIFF metadata available)", npy_path)


def _write_preview(path: Path, height: np.ndarray):
    """viridis-colored PNG preview."""
    import matplotlib.cm as cm
    vmax = float(np.nanpercentile(height, 99))
    vmax = max(vmax, 1.0)
    normed = np.clip(height / vmax, 0, 1)
    rgba = (cm.viridis(normed) * 255).astype(np.uint8)
    Image.fromarray(rgba[..., :3]).save(path)
    log.info("wrote preview: %s (vmax=%.2f m)", path, vmax)


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------

def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """uint8/float RGB → ImageNet-normalised float32."""
    img = img.astype(np.float32)
    if img.max() > 1.5:  # plausibly 0–255 range
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def _pad_to_tile(arr: np.ndarray, tile: int = TILE):
    """Zero-pad H, W to multiples of *tile*. Returns padded + original
    (h, w)."""
    h, w = arr.shape[:2]
    new_h = int(np.ceil(h / tile) * tile)
    new_w = int(np.ceil(w / tile) * tile)
    if new_h == h and new_w == w:
        return arr, h, w
    pad_h = new_h - h
    pad_w = new_w - w
    if arr.ndim == 3:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
    else:
        padded = np.pad(arr, ((0, pad_h), (0, pad_w)))
    return padded, h, w


def predict(model, image: np.ndarray, chm: np.ndarray,
            device: torch.device, tile: int = TILE) -> np.ndarray:
    """Run the model on (image, chm) in non-overlapping tiles."""
    img_norm = _preprocess_image(image)
    img_pad, h0, w0 = _pad_to_tile(img_norm, tile)
    chm_pad, _, _ = _pad_to_tile(chm.astype(np.float32), tile)

    H, W = img_pad.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            img_t = img_pad[y:y + tile, x:x + tile]
            chm_t = chm_pad[y:y + tile, x:x + tile]
            img_b = (
                torch.from_numpy(img_t)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )
            chm_b = (
                torch.from_numpy(chm_t)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )
            with torch.no_grad():
                pred = model(img_b, chm_b)
            out[y:y + tile, x:x + tile] = (
                pred.squeeze().cpu().numpy()
            )

    return out[:h0, :w0]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CATT (B9) single-image inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Input RGB image (GeoTIFF or any PIL format).")
    parser.add_argument("--chm", type=Path, default=None,
                        help="Optional CHM prompt (single-band float32). "
                             "If omitted, a zero-prompt is used.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output predicted height map (GeoTIFF "
                             "preferred; .png/.npy fallback supported).")
    parser.add_argument("--checkpoint", type=Path,
                        default=PACKAGE_ROOT / "weights" / "P1_catt_epoch17.ckpt",
                        help="Path to a Lightning checkpoint .ckpt.")
    parser.add_argument("--config", type=Path,
                        default=CODE_ROOT / "configs" / "P1_v1_catt_synoc_dfcval.yaml",
                        help="Training YAML the checkpoint was produced "
                             "with. The config stored inside the .ckpt is "
                             "preferred when present; this YAML is only a "
                             "fallback for older checkpoints.")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device id (e.g. cuda:0, cuda:1). "
                             "A CUDA-capable GPU with >= 24 GB VRAM is required; "
                             "see README.md for hardware details.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(
            f"checkpoint not found: {args.checkpoint}\n"
            f"  Hint: run 'python scripts/download_weights.py' first."
        )

    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: a CUDA-capable GPU is required.\n"
            "  This package does not support CPU inference; the P1 checkpoint "
            "and the DINOv3 backbone weights are stored on CUDA. Run on a host "
            "with a 24 GB-class GPU (RTX 3090 / 4090 / A5000 / L4 / A6000)."
        )

    device = torch.device(args.device)

    log.info("loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("hyper_parameters", {}).get("config")
    if ckpt_config is not None:
        build_config = ckpt_config
        log.info("using config embedded in checkpoint")
    else:
        log.info("checkpoint has no embedded config; loading %s", args.config)
        with open(args.config) as fh:
            build_config = yaml.safe_load(fh)

    net = get_model(build_config)
    module = HeightEstimationModule(model=net, config=build_config)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval().to(device)
    model = module.model

    log.info("loading image: %s", args.image)
    img, geo = _read_raster(args.image)

    if args.chm is not None and args.chm.exists():
        log.info("loading CHM prompt: %s", args.chm)
        chm, _ = _read_raster(args.chm)
        if chm.ndim > 2:
            chm = chm.squeeze()
    else:
        log.warning("no CHM prompt supplied — using zero prompt")
        h = img.shape[0] if img.ndim >= 2 else 0
        w = img.shape[1] if img.ndim >= 2 else 0
        chm = np.zeros((h, w), dtype=np.float32)

    if chm.shape[:2] != img.shape[:2]:
        sys.exit(
            f"image / CHM shape mismatch: image {img.shape[:2]} vs "
            f"chm {chm.shape}"
        )

    log.info("running tiled inference on %s tiles of %d × %d",
             tuple(np.ceil(np.array(img.shape[:2]) / TILE).astype(int)),
             TILE, TILE)
    height = predict(model, img, chm, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_geotiff(args.output, height, geo)
    _write_preview(args.output.with_suffix(".png"), height)

    log.info("done. predicted height range: [%.2f, %.2f] m, mean %.2f m",
             float(height.min()), float(height.max()), float(height.mean()))


if __name__ == "__main__":
    main()
