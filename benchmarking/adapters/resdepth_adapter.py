"""Adapter for ResDepth (U-Net DSM refinement via residual learning).

ResDepth input: [B, C, tile, tile] where C=2 for geom-mono (DSM + image).
  Channel 0 = initial DSM (normalised)
  Channel 1 = panchromatic / grayscale ortho image (normalised)

Output: [B, 1, tile, tile] normalised DSM → denormalise with dsm_mean/std.

For benchmarking we operate in "geom-mono" mode (DSM + single grayscale
image), which is the simplest way to feed (CHM, RGB→gray) without needing
stereo pair metadata.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RESDEPTH_ROOT = PACKAGE_ROOT / "benchmarking" / "repos" / "ResDepth"
sys.path.insert(0, str(RESDEPTH_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "benchmarking"))

from eval.corruption_sweep import ModelAdapter  # noqa: E402


class Adapter(ModelAdapter):
    # UNet (depth=5) is fully convolutional but requires input H,W divisible
    # by 2**depth = 32. When the input fits, we run the WHOLE image in a
    # single forward pass — this matches the model's eval-mode BatchNorm
    # statistics best and avoids any blending/seam concerns. We fall back to
    # tiled linear-blend inference (port of ResDepth.lib.evaluation) only when
    # dims are non-conforming or the image is too large to fit in GPU memory.

    DOWNSAMPLE_FACTOR: int = 32          # 2 ** UNet.depth
    MAX_WHOLE_IMAGE_PIXELS: int = 2048 * 2048  # ~12GB peak activations

    def __init__(self, tile_size: int = 256, tile_stride: int | None = None):
        """``tile_stride`` defaults to ``tile_size // 2`` (50% overlap),
        matching the canonical ResDepth ``predict_linear_blend`` test
        pipeline. Tiling is only used when whole-image inference is not
        possible.
        """
        self._tile_size = tile_size
        self._tile_stride = tile_stride if tile_stride is not None else tile_size // 2
        self._model = None
        self._device = None
        self._n_input_channels: int = 2
        self._use_local_dsm_mean: bool = False
        self._dsm_mean: float = 0.0
        self._dsm_std: float = 1.0
        self._img_mean: float = 0.0
        self._img_std: float = 1.0

    def load(self, checkpoint_path: str, device: str) -> None:
        self._device = torch.device(device)
        ckpt_dir = Path(checkpoint_path).parent

        from lib.UNet import UNet

        # model_config.json may be in ckpt_dir or one level up (checkpoints/)
        config_path = ckpt_dir / "model_config.json"
        if not config_path.exists():
            config_path = ckpt_dir.parent / "model_config.json"

        if config_path.exists():
            with open(config_path) as f:
                mcfg = json.load(f)
            settings = mcfg.get("settings", mcfg)
            self._n_input_channels = settings.get("n_input_channels", 2)
            self._model = UNet(**settings)
        else:
            self._n_input_channels = 2
            self._model = UNet(
                n_input_channels=2,
                start_kernel=64,
                depth=4,
                act_fn_encoder="relu",
                act_fn_decoder="relu",
                act_fn_bottleneck="relu",
                up_mode="transpose",
                outer_skip=True,
            )

        state = torch.load(checkpoint_path, map_location="cpu")
        self._model.load_state_dict(state["model_state_dict"])
        self._model.to(self._device).eval()

        # Normalization params may be next to config (one level above checkpoints/)
        norm_dir = config_path.parent

        dsm_norm_path = norm_dir / "DSM_normalization_parameters.p"
        if dsm_norm_path.exists():
            with open(dsm_norm_path, "rb") as f:
                dsm_p = pickle.load(f)
            self._use_local_dsm_mean = bool(dsm_p.get("use_local_mean", False))
            dsm_mean = dsm_p.get("mean")
            self._dsm_mean = float(dsm_mean) if dsm_mean is not None else 0.0
            self._dsm_std = float(dsm_p.get("std", 1.0))

        img_norm_path = norm_dir / "Image_normalization_parameters.p"
        if img_norm_path.exists():
            with open(img_norm_path, "rb") as f:
                img_p = pickle.load(f)
            img_mean = img_p.get("mean")
            self._img_mean = float(img_mean) if img_mean is not None else 0.0
            self._img_std = float(img_p.get("std", 1.0))

    def _prepare_image_channels(self, image: np.ndarray, chm: np.ndarray):
        """Return (img_ch1, img_ch2) normalised image channel(s).

        Handles three cases:
        1. MSIImageWrapper  — raw 16-bit MSI stereo → simulated PAN with
           original 16-bit normalization.
        2. StereoRGBWrapper — uint8 LEFT+RIGHT RGB → grayscale with per-image
           z-norm for each view.
        3. Plain ndarray    — grayscale with per-image z-norm, duplicated for
           both channels.
        """
        # Case 1: raw 16-bit MSI stereo
        left_msi = getattr(image, '_left_msi', None)
        right_msi = getattr(image, '_right_msi', None)
        if left_msi is not None and right_msi is not None:
            left_pan = np.mean(
                left_msi[:, :, [1, 2, 4]].astype(np.float32), axis=2
            )
            right_pan = np.mean(
                right_msi[:, :, [1, 2, 4]].astype(np.float32), axis=2
            )
            ch1 = (left_pan - self._img_mean) / max(self._img_std, 1e-8)
            ch2 = (right_pan - self._img_mean) / max(self._img_std, 1e-8)
            return ch1, ch2

        # Case 2: stereo RGB (uint8 LEFT + RIGHT) → luminance grayscale
        # ITU-R BT.601 weights — the spectral response that best approximates
        # panchromatic on RGB-only data.
        right_rgb = getattr(image, '_right_rgb', None)
        if right_rgb is not None:
            left_gray = self._rgb_to_luminance(image)
            right_gray = self._rgb_to_luminance(right_rgb)
            lm, ls = float(np.mean(left_gray)), float(np.std(left_gray))
            rm, rs = float(np.mean(right_gray)), float(np.std(right_gray))
            ch1 = (left_gray - lm) / max(ls, 1e-8)
            ch2 = (right_gray - rm) / max(rs, 1e-8)
            return ch1, ch2

        # Case 3: plain RGB → luminance grayscale, duplicated
        gray = self._rgb_to_luminance(image)
        g_mean = float(np.mean(gray)) if gray.size else 0.0
        g_std = float(np.std(gray)) if gray.size else 1.0
        ch = (gray - g_mean) / max(g_std, 1e-8)
        return ch, ch

    @staticmethod
    def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
        rgb_f = rgb.astype(np.float32)
        return (0.299 * rgb_f[..., 0]
                + 0.587 * rgb_f[..., 1]
                + 0.114 * rgb_f[..., 2])

    @staticmethod
    def _axis_grid(extent: int, tile_size: int, stride: int):
        """Return list of (origin, border_lo, border_hi) for one axis.

        Mirrors ResDepth's ``rasterutils.create_regular_grid``: tiles step at
        ``stride``; if the next tile would overflow the image, the LAST tile
        is shifted backward to fit, and ``border_lo`` is bumped by the shift
        amount so blending still works correctly.

        ``border_lo``/``border_hi`` are the indices (within the tile's own
        coordinate frame) of the region that does NOT overlap any neighbour;
        ``_blend_weights`` uses them to position the linear ramps.
        """
        if extent <= tile_size:
            return [(0, 0, tile_size - 1)]

        overlap = tile_size - stride
        positions = []
        pos = 0
        border_lo = 0
        while True:
            end = pos + tile_size - 1
            if end >= extent - 1:
                shift = end - (extent - 1)
                pos_new = max(0, extent - tile_size)
                positions.append((pos_new, border_lo + shift, tile_size - 1))
                break
            positions.append((pos, border_lo, stride - 1))
            pos += stride
            border_lo = overlap
        return positions

    @staticmethod
    def _blend_weights(ts: int, stride: int, ulx: int, uly: int,
                       lrx: int, lry: int) -> np.ndarray:
        """Linear blending weights for one tile (port of
        ``ResDepth.lib.evaluation._get_blend_weights``).
        Pixels in the unique region get weight 1; overlap edges get a linear
        ramp from 0 to 1 (or 1 to 0). Across all tiles covering a pixel the
        weights sum to exactly 1.
        """
        weights = np.ones((ts, ts), dtype=np.float32)
        overlap = ts - stride
        if overlap <= 0:
            return weights
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)

        if ulx > 0:
            if ulx == overlap:
                weights[:, 0:ulx] *= np.tile(ramp, (ts, 1))
            else:
                weights[:, ulx - overlap:ulx] *= np.tile(ramp, (ts, 1))
                weights[:, 0:ulx - overlap] = 0.0
        if lrx < ts - 1:
            weights[:, lrx + 1:] *= np.tile(ramp[::-1], (ts, 1))
        if uly > 0:
            if uly == overlap:
                weights[0:uly, :] *= np.tile(ramp.reshape(uly, 1), (1, ts))
            else:
                weights[uly - overlap:uly, :] *= np.tile(
                    ramp.reshape(overlap, 1), (1, ts)
                )
                weights[0:uly - overlap, :] = 0.0
        if lry < ts - 1:
            n = ts - lry - 1
            weights[lry + 1:, :] *= np.tile(ramp[::-1].reshape(n, 1), (1, ts))
        return weights

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        h, w = chm.shape
        chm_f = chm.astype(np.float32)
        img_ch1, img_ch2 = self._prepare_image_channels(image, chm)

        # ------------------------------------------------------------------
        # Fast path: whole-image inference.
        # The UNet is fully convolutional and BN runs in eval() mode using
        # the saved running stats, so a single forward pass on the entire
        # image is cleaner than any tiling — it produces NO seams or
        # per-tile mean discontinuities and matches the model's training
        # distribution best.
        # ------------------------------------------------------------------
        df = self.DOWNSAMPLE_FACTOR
        if (h % df == 0 and w % df == 0
                and h * w <= self.MAX_WHOLE_IMAGE_PIXELS):
            return self._predict_whole_image(chm_f, img_ch1, img_ch2)

        # Fallback: padded whole-image (pad to nearest 32, predict, crop).
        if h * w <= self.MAX_WHOLE_IMAGE_PIXELS:
            pad_h = (df - h % df) % df
            pad_w = (df - w % df) % df
            chm_p = np.pad(chm_f, ((0, pad_h), (0, pad_w)))
            c1_p = np.pad(img_ch1, ((0, pad_h), (0, pad_w)))
            c2_p = np.pad(img_ch2, ((0, pad_h), (0, pad_w)))
            return self._predict_whole_image(chm_p, c1_p, c2_p)[:h, :w]

        # Last resort: tiled linear-blend inference.
        return self._predict_tiled(chm_f, img_ch1, img_ch2, h, w)

    def _predict_whole_image(self, dsm: np.ndarray, c1: np.ndarray,
                             c2: np.ndarray) -> np.ndarray:
        """Single forward pass on the whole image (no tiling)."""
        if self._use_local_dsm_mean:
            dsm_mean = float(np.mean(dsm))
        else:
            dsm_mean = self._dsm_mean
        dsm_norm = (dsm - dsm_mean) / max(self._dsm_std, 1e-8)

        if self._n_input_channels == 3:
            inp = np.stack([dsm_norm, c1, c2], axis=0)[None]
        elif self._n_input_channels == 1:
            inp = dsm_norm[None, None]
        else:
            inp = np.stack([dsm_norm, c1], axis=0)[None]

        inp_t = torch.from_numpy(inp).float().to(self._device)
        with torch.no_grad():
            out = self._model(inp_t).squeeze().cpu().numpy()
        return out * self._dsm_std + dsm_mean

    def _predict_tiled(self, chm_f: np.ndarray, img_ch1: np.ndarray,
                       img_ch2: np.ndarray, h: int, w: int) -> np.ndarray:
        ts = self._tile_size
        stride = self._tile_stride
        ys = self._axis_grid(h, ts, stride)
        xs = self._axis_grid(w, ts, stride)

        out = np.zeros((h, w), dtype=np.float32)

        for (uly, b_uly, b_lry) in ys:
            for (ulx, b_ulx, b_lrx) in xs:
                dsm_tile = chm_f[uly:uly + ts, ulx:ulx + ts].copy()
                c1_tile = img_ch1[uly:uly + ts, ulx:ulx + ts]
                c2_tile = img_ch2[uly:uly + ts, ulx:ulx + ts]

                # Pad if the image is smaller than one tile (rare).
                if dsm_tile.shape != (ts, ts):
                    pad_h = ts - dsm_tile.shape[0]
                    pad_w = ts - dsm_tile.shape[1]
                    dsm_tile = np.pad(dsm_tile, ((0, pad_h), (0, pad_w)))
                    c1_tile = np.pad(c1_tile, ((0, pad_h), (0, pad_w)))
                    c2_tile = np.pad(c2_tile, ((0, pad_h), (0, pad_w)))

                if self._use_local_dsm_mean:
                    tile_mean = float(np.mean(dsm_tile))
                else:
                    tile_mean = self._dsm_mean
                dsm_norm = (dsm_tile - tile_mean) / max(self._dsm_std, 1e-8)

                if self._n_input_channels == 3:
                    inp = np.stack([dsm_norm, c1_tile, c2_tile], axis=0)[None]
                elif self._n_input_channels == 1:
                    inp = dsm_norm[None, None]
                else:
                    inp = np.stack([dsm_norm, c1_tile], axis=0)[None]

                inp_t = torch.from_numpy(inp).float().to(self._device)
                with torch.no_grad():
                    pred_t = self._model(inp_t)
                pred_np = pred_t.squeeze().cpu().numpy()
                pred_metres = pred_np * self._dsm_std + tile_mean

                weights = self._blend_weights(
                    ts, stride, b_ulx, b_uly, b_lrx, b_lry,
                )

                # Crop back to original tile region in case we padded.
                tile_h = min(ts, h - uly)
                tile_w = min(ts, w - ulx)
                out[uly:uly + tile_h, ulx:ulx + tile_w] += (
                    pred_metres[:tile_h, :tile_w]
                    * weights[:tile_h, :tile_w]
                )

        return out

    @property
    def name(self) -> str:
        return "resdepth"
