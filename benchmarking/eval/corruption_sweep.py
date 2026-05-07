"""Run a model through every corruption regime and write result JSONs.

Usage
-----
    python corruption_sweep.py \
        --adapter ours \
        --checkpoint /path/to/best.ckpt \
        --dataset synrs3d_val \
        --data_dir /path/to/SynRS3D \
        --output_dir results/ \
        --device cuda:0

The ``--adapter`` flag selects which model wrapper to load.  Every adapter
must expose the ``ModelAdapter`` interface (see below).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

# Reviewer-package layout:
#   reviewer_package/
#     ├── benchmarking/eval/corruption_sweep.py  (this file)
#     ├── code/                                  (bundled training/eval subset)
#     └── data/                                  (downloaded by scripts/download_data.sh)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PACKAGE_ROOT / "code"
REPO_ROOT = PACKAGE_ROOT  # kept as alias so DEFAULT_DATA_DIRS below resolves under data/
sys.path.insert(0, str(CODE_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from data_loaders.synrs3d_dataset import CHMCorruptor  # noqa: E402

from benchmark_eval import evaluate  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model adapter interface
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Thin wrapper that every competitor model must implement."""

    @abstractmethod
    def load(self, checkpoint_path: str, device: str) -> None:
        """Load weights."""

    @abstractmethod
    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        """Run inference.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB
        chm   : (H, W) float32 metres.  May be all-zeros for ``zero_chm``.

        Returns
        -------
        (H, W) float32 predicted height in metres.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model identifier for file naming."""

    @property
    def accepts_chm(self) -> bool:
        """Override to False for monocular models (Open-Canopy PVTv2)."""
        return True


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

def _make_shift_only_corruptor(max_shift: int) -> CHMCorruptor:
    return CHMCorruptor(
        cutout_prob=0.0,
        max_shift=max_shift,
        resolution_factor=1.0,
        height_noise_sigma=0.0,
        full_dropout_prob=0.0,
        always_on_noise_sigma=0.0,
    )


def _make_cutout_only_corruptor(area: float) -> CHMCorruptor:
    return CHMCorruptor(
        cutout_prob=1.0,
        cutout_area_range=[area, area],
        cutout_one_large_prob=1.0,
        max_shift=0,
        resolution_factor=1.0,
        height_noise_sigma=0.0,
        full_dropout_prob=0.0,
        always_on_noise_sigma=0.0,
    )


REGIMES: dict[str, str | CHMCorruptor] = {
    # --- Misalignment sweep ---
    "shift_0": _make_shift_only_corruptor(0),
    "shift_4": _make_shift_only_corruptor(4),
    "shift_8": _make_shift_only_corruptor(8),
    "shift_16": _make_shift_only_corruptor(16),
    "shift_24": _make_shift_only_corruptor(24),
    "shift_48": _make_shift_only_corruptor(48),
    # --- Cutout sweep ---
    "cutout_10pct": _make_cutout_only_corruptor(0.10),
    "cutout_30pct": _make_cutout_only_corruptor(0.30),
    "cutout_50pct": _make_cutout_only_corruptor(0.50),
    # --- Dependency probes ---
    "zero_chm": "ZERO_CHM",
    "zero_image": "ZERO_IMAGE",
    # --- Standard 6 regimes (from CHMCorruptor.for_regime) ---
    "regime_clean": CHMCorruptor.for_regime("clean"),
    "regime_shifted": CHMCorruptor.for_regime("shifted"),
    "regime_masked": CHMCorruptor.for_regime("masked"),
    "regime_degraded": CHMCorruptor.for_regime("degraded"),
    "regime_blurred": CHMCorruptor.for_regime("blurred"),
    "regime_zero": CHMCorruptor.for_regime("zero"),
}


# ---------------------------------------------------------------------------
# Dataset loading helpers
#
# All loaders yield ``(image, chm_clean, gt_height)`` triplets:
#   image      — ``(H, W, 3)`` uint8 RGB
#   chm_clean  — ``(H, W)`` float32 metres (used as prompt before corruption)
#   gt_height  — ``(H, W)`` float32 metres
# ---------------------------------------------------------------------------

def _read_tif(path: str) -> np.ndarray:
    """Read a GeoTIFF, return ``[C, H, W]`` numpy array."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read()
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))
        if img.ndim == 2:
            return img[np.newaxis]
        return img.transpose(2, 0, 1)


def _discover_synrs3d_samples(data_dir: str, subsets: list[str]):
    """Return list of ``{"opt": ..., "ndsm": ..., "ss_mask": ... | None}``."""
    import os
    samples = []
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        opt_dir = os.path.join(subset_dir, "opt")
        ndsm_dir = os.path.join(subset_dir, "gt_nDSM")
        mask_dir = os.path.join(subset_dir, "gt_ss_mask")
        has_masks = os.path.isdir(mask_dir)
        if not os.path.isdir(opt_dir):
            continue
        ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(opt_dir)
            if f.endswith(".tif")
        )
        for sid in ids:
            opt_path = os.path.join(opt_dir, f"{sid}.tif")
            ndsm_path = os.path.join(ndsm_dir, f"{sid}.tif")
            mask_path = os.path.join(mask_dir, f"{sid}.tif") if has_masks else None
            if mask_path and not os.path.exists(mask_path):
                mask_path = None
            if os.path.exists(ndsm_path):
                samples.append({
                    "opt": opt_path, "ndsm": ndsm_path, "ss_mask": mask_path,
                })
    return samples


def load_synrs3d_val(data_dir: str, subsets: list[str] | None = None):
    """Yield (image, chm_clean, gt_height, class_mask|None) from SynRS3D val."""
    if subsets is None:
        subsets = ["grid_g005_mid_v2", "grid_g05_mid_v2", "terrain_g005_mid_v1"]

    samples = _discover_synrs3d_samples(data_dir, subsets)
    logger.info("SynRS3D val: %d samples from %s", len(samples), subsets)

    for s in samples:
        rgb_chw = _read_tif(s["opt"])[:3]
        ndsm = _read_tif(s["ndsm"])[0]

        image = rgb_chw.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        gt_height = ndsm.astype(np.float32)
        chm_clean = gt_height.copy()
        ss_mask = None
        if s.get("ss_mask"):
            ss_mask = _read_tif(s["ss_mask"])[0].astype(np.uint8)
        yield image, chm_clean, gt_height, ss_mask


def load_dfc_val(data_dir: str, subsets: list[str] | None = None):
    """Yield (image, chm_clean, gt_height, class_mask) from DFC18/DFC19.

    DFC19 includes per-tile semantic class masks under ``gt_ss_mask/`` (ASPRS
    LiDAR scheme: 2=ground, 5=high vegetation, 6=building, etc.). DFC18 has
    no class masks so ``class_mask`` is ``None`` for those tiles.
    """
    if subsets is None:
        subsets = ["DFC18", "DFC19"]

    samples = _discover_synrs3d_samples(data_dir, subsets)
    n_with_mask = sum(1 for s in samples if s.get("ss_mask"))
    logger.info("DFC val: %d samples from %s (%d with class masks)",
                len(samples), subsets, n_with_mask)

    for s in samples:
        rgb_chw = _read_tif(s["opt"])[:3]
        ndsm = _read_tif(s["ndsm"])[0]

        image = rgb_chw.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        gt_height = ndsm.astype(np.float32)
        chm_clean = gt_height.copy()
        ss_mask = None
        if s.get("ss_mask"):
            ss_mask = _read_tif(s["ss_mask"])[0].astype(np.uint8)
        yield image, chm_clean, gt_height, ss_mask


def load_open_canopy(data_dir: str, **kwargs):
    """Yield raw triplets from Open-Canopy stale-pair windows.

    Reads the manifest to find ``(image_2023, chm_2021, chm_2023)`` tiles.
    ``chm_2021`` = stale prompt (``chm_clean``), ``chm_2023`` = GT.

    Uses rasterio windowed reads — the source SPOT images are ~40k x 40k
    pixels so reading them in full per 256x256 crop is prohibitive.
    """
    import json as _json
    import rasterio
    from rasterio.windows import Window

    manifest_path = kwargs.get(
        "manifest_path",
        str(Path(data_dir) / "manifests" / "train_w256_s256.jsonl"),
    )

    SPOT_RGB_BANDS = (3, 2, 1)  # rasterio 1-indexed R, G, B from BGRNIR stack
    SPOT_NIR_BAND = 4           # band 4 = NIR in BGRNIR stack
    LIDAR_DM_TO_M = 0.1

    n_records = 0
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            if "image_path" not in rec:
                continue
            n_records += 1
            if n_records == 1:
                logger.info("Open-Canopy: streaming from %s", manifest_path)

            try:
                iw = rec["image_window"]  # [row_off, col_off, height, width]
                tw = rec["target_window"]
                win_img = Window(col_off=iw[1], row_off=iw[0],
                                 width=iw[3], height=iw[2])
                win_tgt = Window(col_off=tw[1], row_off=tw[0],
                                 width=tw[3], height=tw[2])

                with rasterio.open(str(Path(data_dir) / rec["image_path"])) as src_img:
                    img_bands = src_img.read(SPOT_RGB_BANDS, window=win_img,
                                             boundless=True, fill_value=0)
                    nir_band = src_img.read(SPOT_NIR_BAND, window=win_img,
                                            boundless=True, fill_value=0)
                    geo_bounds = rasterio.windows.bounds(win_img, src_img.transform)

                with rasterio.open(str(Path(data_dir) / rec["prompt_path"])) as src_p:
                    # Compute prompt window from geo-coordinates (rasters
                    # have different origins so pixel windows don't match)
                    prompt_win = rasterio.windows.from_bounds(
                        *geo_bounds, transform=src_p.transform,
                    ).round_offsets().round_lengths()
                    chm_prompt = src_p.read(
                        1, window=prompt_win,
                        boundless=True, fill_value=0,
                    ).astype(np.float32) * LIDAR_DM_TO_M

                with rasterio.open(str(Path(data_dir) / rec["target_path"])) as src_t:
                    chm_target = src_t.read(
                        1, window=win_tgt,
                        boundless=True, fill_value=0,
                    ).astype(np.float32) * LIDAR_DM_TO_M
                    target_geo = rasterio.windows.bounds(win_tgt, src_t.transform)

                # Read class mask aligned to the target's geographic extent
                ss_mask = None
                cls_path = rec.get(
                    "target_class_path",
                    rec["target_path"].replace("/lidar/", "/lidar_classification/")
                                       .replace("compressed_lidar_",
                                                "compressed_lidar_classification_"),
                )
                cls_full = Path(data_dir) / cls_path
                if cls_full.exists():
                    with rasterio.open(str(cls_full)) as src_c:
                        cls_win = rasterio.windows.from_bounds(
                            *target_geo, transform=src_c.transform,
                        ).round_offsets().round_lengths()
                        ss_mask = src_c.read(
                            1, window=cls_win,
                            boundless=True, fill_value=0,
                        ).astype(np.uint8)

                # Ensure prompt and mask match target spatial dimensions
                th, tw_ = chm_target.shape

                def _fit(arr, h, w):
                    if arr is None:
                        return None
                    ah, aw = arr.shape
                    if (ah, aw) == (h, w):
                        return arr
                    if ah >= h and aw >= w:
                        return arr[:h, :w]
                    return np.pad(arr,
                                  ((0, max(0, h-ah)), (0, max(0, w-aw))),
                                  mode='constant')[:h, :w]

                chm_prompt = _fit(chm_prompt, th, tw_)
                ss_mask = _fit(ss_mask, th, tw_)
            except Exception as e:
                logger.warning("Skipping OC record: %s", e)
                continue

            image = img_bands.transpose(1, 2, 0)  # [H, W, 3] RGB
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            # Attach real NIR so the adapter can build a proper 4-channel input
            image = MSIImageWrapper(image, None, None, nir=nir_band)

            yield image, chm_prompt, chm_target, ss_mask

    logger.info("Open-Canopy: yielded %d records", n_records)


def load_arkitscenes_val(data_dir: str, **kwargs):
    """Yield raw triplets from ARKitScenes depth upsampling validation.

    Image = RGB (1920x1440), prompt = low-res LiDAR upsampled to image
    size, GT = high-res Faro depth.  Both depths stored as uint16 mm PNGs.
    """
    import cv2
    val_dir = Path(data_dir) / "Validation"
    if not val_dir.exists():
        raise FileNotFoundError(f"ARKitScenes Validation dir not found: {val_dir}")

    max_depth_m = kwargs.get("max_depth_m", 10.0)

    video_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    logger.info("ARKitScenes val: %d video dirs in %s", len(video_dirs), val_dir)

    for vdir in video_dirs:
        color_dir = vdir / "color"
        lowres_dir = vdir / "lowres_depth"
        highres_dir = vdir / "highres_depth"
        if not all(d.is_dir() for d in [color_dir, lowres_dir, highres_dir]):
            continue
        for cfile in sorted(color_dir.glob("*.png")):
            lr_file = lowres_dir / cfile.name
            hr_file = highres_dir / cfile.name
            if not (lr_file.exists() and hr_file.exists()):
                continue

            image = cv2.imread(str(cfile), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

            hr_raw = cv2.imread(str(hr_file), cv2.IMREAD_UNCHANGED)
            gt_depth = np.clip(hr_raw.astype(np.float32) / 1000.0, 0, max_depth_m)

            lr_raw = cv2.imread(str(lr_file), cv2.IMREAD_UNCHANGED)
            lr_depth = np.clip(lr_raw.astype(np.float32) / 1000.0, 0, max_depth_m)
            prompt = cv2.resize(
                lr_depth, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            yield image, prompt, gt_depth


def load_dfc_track2_rgb(data_dir: str, **kwargs):
    """Yield triplets from DFC2019 US3D Track2 RGB stereo + AGL ground truth.

    Each Track2 tile is 1024x1024 with uint8 LEFT/RIGHT RGB stereo pairs
    and float32 LEFT_AGL (above-ground-level height).  Each 1024 tile is
    split into four non-overlapping 512x512 subtiles.

    Yields per 512x512 subtile:
        image    — (512, 512, 3) uint8  RGB (from LEFT_RGB)
        chm      — (512, 512) float32   AGL height (used as clean prompt)
        gt       — (512, 512) float32   AGL height

    A ``StereoRGBWrapper`` carries the RIGHT_RGB crop so that the ResDepth
    adapter can access the stereo pair for its grayscale channels.
    """
    import os
    import tifffile

    rgb_dir = str(Path(data_dir) / "Track2-RGB-2")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Track2 RGB dir not found: {rgb_dir}")

    left_files = sorted(
        f for f in os.listdir(rgb_dir) if f.endswith("_LEFT_RGB.tif")
    )
    logger.info("DFC Track2 RGB: %d LEFT files in %s", len(left_files), rgb_dir)

    subtile_offsets = [(0, 0), (0, 512), (512, 0), (512, 512)]

    for lf in left_files:
        tile_id = lf.replace("_LEFT_RGB.tif", "")       # e.g. JAX_314_001_002
        right_path = os.path.join(rgb_dir, tile_id + "_RIGHT_RGB.tif")
        agl_path = os.path.join(rgb_dir, tile_id + "_LEFT_AGL.tif")

        if not os.path.exists(right_path) or not os.path.exists(agl_path):
            continue

        left_rgb = tifffile.imread(os.path.join(rgb_dir, lf))    # (1024,1024,3) uint8
        right_rgb = tifffile.imread(right_path)                    # (1024,1024,3) uint8
        agl = tifffile.imread(agl_path).astype(np.float32)        # (1024,1024) float32

        for row_off, col_off in subtile_offsets:
            img_crop = left_rgb[row_off:row_off+512, col_off:col_off+512]
            right_crop = right_rgb[row_off:row_off+512, col_off:col_off+512]
            gt_crop = agl[row_off:row_off+512, col_off:col_off+512]

            img_with_stereo = StereoRGBWrapper(img_crop, right_crop)
            yield img_with_stereo, gt_crop.copy(), gt_crop


class StereoRGBWrapper(np.ndarray):
    """uint8 RGB array that also carries the RIGHT stereo view."""

    def __new__(cls, left_rgb, right_rgb):
        obj = np.asarray(left_rgb).view(cls)
        obj._right_rgb = right_rgb
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._right_rgb = getattr(obj, '_right_rgb', None)


def load_dfc_track2_msi(data_dir: str, **kwargs):
    """Yield triplets from DFC2019 US3D Track2 MSI stereo + DFC19 GT nDSM.

    Each Track2 tile is 1024x1024 with 8-band 16-bit MSI (LEFT + RIGHT).
    The 3-digit tile ID (e.g. JAX_004_013) maps to four 512x512 DFC19 GT
    subtiles (_0_0, _0_512, _512_0, _512_512).

    RGB is synthesised from WV-3 bands: R=B5(idx4), G=B3(idx2), B=B2(idx1)
    with per-channel 2nd-98th percentile stretch.
    """
    import os
    import tifffile

    msi_dir = kwargs.get("msi_dir", str(Path(data_dir) / "Track2_MSI1" / "Track2-MSI-1"))
    gt_dir = kwargs.get("gt_dir", str(
        REPO_ROOT / "data" / "tg" / "tree_height" / "synrs3d" / "dfc" / "DFC19" / "gt_nDSM"
    ))

    if not os.path.isdir(msi_dir):
        raise FileNotFoundError(f"Track2 MSI dir not found: {msi_dir}")

    left_files = sorted(
        f for f in os.listdir(msi_dir) if f.endswith("_LEFT_MSI.tif")
    )
    logger.info("DFC Track2 MSI: %d LEFT files in %s", len(left_files), msi_dir)

    RGB_BANDS = (4, 2, 1)  # WV-3: R=B5, G=B3, B=B2 (0-indexed)
    NIR_BAND = 6           # WV-3: B7 = NIR1 (770-895nm)
    subtile_offsets = [(0, 0), (0, 512), (512, 0), (512, 512)]

    for lf in left_files:
        tile_id = lf.replace("_LEFT_MSI.tif", "")
        base_3 = "_".join(tile_id.split("_")[:3])
        rf = tile_id + "_RIGHT_MSI.tif"

        right_path = os.path.join(msi_dir, rf)
        if not os.path.exists(right_path):
            continue

        left_msi = tifffile.imread(os.path.join(msi_dir, lf))
        right_msi = tifffile.imread(right_path)

        # Per-channel percentile stretch for RGB
        rgb_f = np.stack([left_msi[:, :, b] for b in RGB_BANDS], axis=-1).astype(np.float32)
        rgb_8 = np.zeros_like(rgb_f, dtype=np.uint8)
        for c in range(3):
            ch = rgb_f[:, :, c]
            lo = np.percentile(ch, 2)
            hi = np.percentile(ch, 98)
            rgb_8[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

        # NIR band stretched to 8-bit
        nir_f = left_msi[:, :, NIR_BAND].astype(np.float32)
        nir_lo = np.percentile(nir_f, 2)
        nir_hi = np.percentile(nir_f, 98)
        nir_8 = np.clip((nir_f - nir_lo) / max(nir_hi - nir_lo, 1) * 255, 0, 255).astype(np.uint8)

        for row_off, col_off in subtile_offsets:
            gt_name = f"{base_3}_{row_off}_{col_off}.tif"
            gt_path = os.path.join(gt_dir, gt_name)
            if not os.path.exists(gt_path):
                continue

            gt_data = _read_tif(gt_path)[0].astype(np.float32)
            img_crop = rgb_8[row_off:row_off+512, col_off:col_off+512]
            nir_crop = nir_8[row_off:row_off+512, col_off:col_off+512]
            gt_crop = gt_data

            left_crop = left_msi[row_off:row_off+512, col_off:col_off+512]
            right_crop = right_msi[row_off:row_off+512, col_off:col_off+512]
            img_with_meta = MSIImageWrapper(img_crop, left_crop, right_crop, nir=nir_crop)

            yield img_with_meta, gt_crop.copy(), gt_crop


class MSIImageWrapper(np.ndarray):
    """uint8 RGB array that also carries raw 16-bit MSI stereo crops and NIR."""

    def __new__(cls, rgb, left_msi, right_msi, nir=None):
        obj = np.asarray(rgb).view(cls)
        obj._left_msi = left_msi
        obj._right_msi = right_msi
        obj._nir = nir
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._left_msi = getattr(obj, '_left_msi', None)
        self._right_msi = getattr(obj, '_right_msi', None)
        self._nir = getattr(obj, '_nir', None)


DATASET_LOADERS = {
    "synrs3d_val": load_synrs3d_val,
    "dfc_val": load_dfc_val,
    "open_canopy": load_open_canopy,
    "arkitscenes_val": load_arkitscenes_val,
    "dfc_track2_rgb": load_dfc_track2_rgb,
    "dfc_track2_msi": load_dfc_track2_msi,
}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _save_visual(
    image: np.ndarray,
    chm_input: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: Path,
):
    """Save a side-by-side visualization: RGB | CHM prompt | Pred | GT | Error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB")

    vmin = min(float(np.nanmin(gt)), 0)
    vmax = max(float(np.nanmax(gt)), 1)

    axes[1].imshow(chm_input, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("CHM/Depth Prompt")

    axes[2].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("Prediction")

    axes[3].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[3].set_title("Ground Truth")

    err = np.abs(pred - gt)
    err_vmax = max(float(np.nanpercentile(err, 95)), 0.1)
    im = axes[4].imshow(err, cmap="hot", vmin=0, vmax=err_vmax)
    axes[4].set_title(f"Abs Error (MAE={np.nanmean(err):.3f})")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_sweep(
    adapter: ModelAdapter,
    dataset_iter,
    regimes: dict[str, str | CHMCorruptor],
    output_dir: Path,
    dataset_name: str,
    max_samples: int | None = None,
    save_images_frac: float = 0.1,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out regimes that this adapter will skip, so we know upfront
    # how many passes we actually need.
    active_regimes = {}
    for regime_name, regime_spec in regimes.items():
        if not adapter.accepts_chm:
            if regime_name.startswith("shift_") or regime_name.startswith("cutout_"):
                logger.info("Skipping %s (monocular model)", regime_name)
                continue
        active_regimes[regime_name] = regime_spec

    if not active_regimes:
        logger.warning("No active regimes after filtering – nothing to do.")
        return

    # Streaming single-pass: iterate the generator once, apply all active
    # regimes to each sample.  Avoids materialising the full dataset in RAM.
    from benchmark_eval import evaluate_batch

    from benchmark_eval import StreamingMetrics

    per_regime: dict[str, dict] = {
        rn: {
            "agg": StreamingMetrics(),
            "img_dir": output_dir / "images" / f"{adapter.name}_{dataset_name}" / rn,
        }
        for rn in active_regimes
    }

    def _flush_results(elapsed_s: float, n_samples_done: int) -> None:
        """Write per-regime JSON files. Safe to call mid-run for partial results."""
        for regime_name in active_regimes:
            agg = per_regime[regime_name]["agg"]
            metrics = agg.finalise()
            metrics["regime"] = regime_name
            metrics["model"] = adapter.name
            metrics["dataset"] = dataset_name
            metrics["elapsed_s"] = round(elapsed_s, 2)
            metrics["n_samples"] = n_samples_done
            out_path = output_dir / f"{adapter.name}_{dataset_name}_{regime_name}.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

    logger.info("Streaming dataset (max_samples=%s) …", max_samples)
    t0 = time.time()
    n_samples = 0

    for idx, sample in enumerate(dataset_iter):
        if max_samples is not None and idx >= max_samples:
            break
        n_samples = idx + 1

        # Loaders may yield (image, chm, gt) or (image, chm, gt, class_mask)
        if len(sample) == 4:
            image, chm_clean, gt, class_mask = sample
        else:
            image, chm_clean, gt = sample
            class_mask = None

        should_save = (save_images_frac > 0) and (idx % max(1, int(1 / save_images_frac)) == 0)

        for regime_name, regime_spec in active_regimes.items():
            img_r = image
            if regime_spec == "ZERO_CHM":
                chm_input = np.zeros_like(chm_clean)
            elif regime_spec == "ZERO_IMAGE":
                chm_input = chm_clean.copy()
                img_r = np.full_like(image, 128)
            elif isinstance(regime_spec, CHMCorruptor):
                chm_input = regime_spec(chm_clean.copy())
            else:
                chm_input = chm_clean.copy()

            pred = adapter.predict(img_r, chm_input)
            per_regime[regime_name]["agg"].update(pred, gt, class_mask)

            if should_save:
                _save_visual(
                    img_r, chm_input, pred, gt,
                    per_regime[regime_name]["img_dir"] / f"sample_{idx:04d}.png",
                )

        if n_samples % 2000 == 0:
            elapsed = time.time() - t0
            logger.info("  … %d samples processed (%.1fs)", n_samples, elapsed)
            _flush_results(elapsed, n_samples)

    total_elapsed = time.time() - t0
    logger.info("Streaming done: %d samples in %.1fs", n_samples, total_elapsed)

    _flush_results(total_elapsed, n_samples)

    for regime_name in active_regimes:
        agg = per_regime[regime_name]["agg"]
        metrics = agg.finalise()
        logger.info(
            "  %s | tree_MAE=%.4f | all_MAE=%.4f | RMSE=%.4f | %.1fs | %d samples",
            regime_name,
            metrics.get("mae_tree_only", float("nan")),
            metrics.get("mae_all_pixels", float("nan")),
            metrics.get("rmse_all_pixels", float("nan")),
            total_elapsed,
            n_samples,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BENCHMARKING_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"

DEFAULT_DATA_DIRS = {
    "synrs3d_val": str(DATA_ROOT / "synrs3d" / "SynRS3D"),
    "dfc_val": str(DATA_ROOT / "synrs3d" / "dfc"),
    "open_canopy": str(DATA_ROOT / "synrs3d" / "open_canopy"),
    "arkitscenes_val": str(DATA_ROOT / "benchmarking" / "arkitscenes"),
    "dfc_track2_rgb": str(DATA_ROOT / "benchmarking" / "dfc2019_us3d" / "Train-Track2-RGB-2"),
    "dfc_track2_msi": str(DATA_ROOT / "benchmarking" / "dfc2019_us3d"),
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Corruption sweep benchmark")
    parser.add_argument("--adapter", required=True,
                        help="Adapter name: ours | promptda | resdepth | opencanopy")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", default="synrs3d_val",
                        choices=list(DATASET_LOADERS.keys()),
                        help="Dataset key (default: synrs3d_val)")
    parser.add_argument("--data_dir", default=None,
                        help="Path to dataset root (auto-detected if omitted)")
    parser.add_argument("--output_dir", default=str(BENCHMARKING_ROOT / "results"),
                        help="Output directory for result JSONs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples (for quick smoke tests)")
    parser.add_argument("--regimes", nargs="*", default=None,
                        help="Subset of regime names to run (default: all)")
    parser.add_argument("--save_images_frac", type=float, default=0.1,
                        help="Fraction of samples to save as visual outputs (default: 0.1)")
    args = parser.parse_args()

    # Resolve data directory
    data_dir = args.data_dir or DEFAULT_DATA_DIRS.get(args.dataset)
    if data_dir is None:
        raise ValueError(
            f"No default data_dir for dataset {args.dataset!r}. "
            f"Pass --data_dir explicitly."
        )

    # Add adapter directory to path so importlib can find it
    sys.path.insert(0, str(BENCHMARKING_ROOT))

    adapter_module = importlib.import_module(f"adapters.{args.adapter}_adapter")
    adapter_cls = adapter_module.Adapter
    adapter = adapter_cls()
    adapter.load(args.checkpoint, args.device)

    # Select regimes
    selected = REGIMES
    if args.regimes:
        selected = {k: v for k, v in REGIMES.items() if k in args.regimes}

    # Load dataset
    loader_fn = DATASET_LOADERS.get(args.dataset)
    if loader_fn is None:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Available: {list(DATASET_LOADERS)}"
        )

    dataset_iter = loader_fn(data_dir)

    run_sweep(
        adapter=adapter,
        dataset_iter=dataset_iter,
        regimes=selected,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        save_images_frac=args.save_images_frac,
    )


if __name__ == "__main__":
    main()
