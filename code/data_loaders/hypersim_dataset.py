"""Hypersim depth dataset for the Dinov3HeightModel CATT stack.

Hypersim (Apple, ICCV 2021) is a photorealistic synthetic indoor dataset
with ~74.6 k images across 461 scenes and pixel-perfect distance-from-
camera depth maps stored as HDF5. We use it as a clean-supervision
source for the prompted-height estimation pipeline -- the target is
metric depth, the "prompt" is the same depth map after :class:`CHMCorruptor`
runs on it (= an old / lo-quality CHM, exactly the SynRS3D pattern).

PromptDA (the model that motivated this loader) uses Hypersim with a
*sparse-anchor RGB-KNN LiDAR simulator* to mimic the iPhone ARKit
LiDAR's block-coherent noise. We deliberately skip that step: our
windowed cross-attention CHM head is supposed to learn to ignore
prompt noise, and we want to see if a simpler corruption pipeline
plus per-sample min-max normalisation (PromptDA §3.4) is enough.

What this loader does
---------------------
1. Walks the canonical Hypersim layout
   ``ai_VVV_NNN/images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5``
   pairing each frame with its tone-mapped preview JPG at
   ``ai_VVV_NNN/images/scene_cam_XX_final_preview/frame.IIII.tonemap.jpg``.
2. Reads depth from HDF5 (Apple's standard format, single ``"dataset"``
   key carrying a ``[H, W]`` float array of *Euclidean distance to
   camera centre* in metres).
3. Masks invalid pixels (``inf``, ``NaN``, ``> max_depth_m``) to 0,
   matching the SynRS3D / ARKitScenes "0 = no measurement" convention
   so existing ``ignore_zero=True`` losses Just Work.
4. (Optional) applies per-sample min-max normalisation BEFORE the
   :class:`CHMCorruptor` runs -- so the corruption sees normalised
   values and zeros from cutout don't contaminate the scale factor.
5. Returns ``(image, chm, gt)`` -- or ``(image, chm, gt, meta)`` when
   min-max normalisation is enabled, where ``meta = [shift, scale]``
   gets carried through to the lightning module so val metrics can be
   recovered in metres.

Depth convention -- distance vs Z
---------------------------------
Hypersim's ``depth_meters.hdf5`` contains the *Euclidean distance from
each pixel's surface to the camera's optical centre*, NOT the
camera-space Z coordinate. The two differ by a per-pixel factor of
``1 / sqrt(1 + (x/fx)^2 + (y/fy)^2)`` -- about 1 % at the image
centre, up to ~10 % at the corners with HyperSim's typical FOV.

We default to keeping the raw distance (``depth_convention="distance"``)
because:

* It's the format Apple ships -- no per-scene intrinsic lookup
  required.
* For training a depth-prediction model, internal consistency
  (prompt and target use the same convention) is what matters; the
  numerical units can be either distance-to-centre or planar Z.
* For mixed-source runs (Hypersim + ARKitScenes, where ARKit is
  Z-depth), set ``depth_convention="z"`` and pass ``fov_y_deg`` (a
  good default is 60 -- HyperSim's nominal vertical FoV). We then do
  the Niklaus-style on-the-fly conversion using a single canonical
  intrinsic. Fully per-scene intrinsics are a future improvement.

Train/val carve
---------------
Two carve strategies are supported:

* ``role="train" / "val"`` with ``held_out_pct``: deterministic
  fraction-of-scenes carve, sorted by scene_id. Mirrors the existing
  ARKitScenes loader's pattern. Recommended when you've downloaded a
  partial subset of HyperSim that doesn't match Apple's official
  split CSV.
* ``role="all"``: use every scene under ``data_dir``.

Apple's official ``metadata_images_split_scene_v1.csv`` train/val/test
split (in the ``ml-hypersim`` repo) is *not* automatically applied:
that CSV references all 461 scenes and would be misleading when only
a subset is on disk. Wire it in via a future ``split_csv`` argument
if needed.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.synrs3d_dataset import CHMContrastiveCorruptor, CHMCorruptor
from utils.normalisations import Normalization
from utils.prompt_normalisation import minmax_normalise_pair
from utils.transformations import Transformations

logger = logging.getLogger(__name__)


# Apple's canonical Hypersim layout. All paths are relative to a single
# scene directory ``ai_VVV_NNN``. The ``XX`` and ``IIII`` placeholders
# refer to the camera-trajectory index and the frame index respectively.
#
# Note on RGB filename: the publicly distributed Hypersim subsets
# (downloaded via Thomas Germer's contributed subset script that the
# Apple README references for partial-scene downloads) name the
# tone-mapped JPG ``frame.IIII.color.jpg``. Some Apple-internal
# pipelines also emit ``frame.IIII.tonemap.jpg`` / ``frame.IIII.tonemap.png``
# from the same renderer. We try them in order so the same loader
# works against either layout.
_GEOMETRY_DIR_GLOB = "images/scene_cam_*_geometry_hdf5"
_PREVIEW_DIR_TEMPLATE = "images/scene_cam_{cam}_final_preview"
_DEPTH_FILE_SUFFIX = ".depth_meters.hdf5"
_RGB_FILE_SUFFIXES = (".color.jpg", ".tonemap.jpg", ".tonemap.png")


class HypersimDepthDataset(Dataset):
    """Hypersim dataset returning ``(image, prompt, gt[, meta])`` tuples.

    Constructor signature mirrors :class:`ARKitScenesDepthDataset` so
    the same :class:`SegmentationDataModule` config surface works
    without changes (``data_dir`` positional + everything else as
    kwargs).

    Args:
        data_dir:                       Root containing one or more
                                        ``ai_VVV_NNN/`` scene folders.
        held_out_pct:                   Fraction of scenes (sorted by
                                        scene_id) reserved for the
                                        ``"val"`` role.  ``0.0``
                                        disables the carve.
        role:                           ``"train"`` keeps
                                        ``floor((1 - held_out_pct) * N)``
                                        scenes from the front; ``"val"``
                                        keeps the rest from the back;
                                        ``"all"`` ignores the carve.
        normalisation:                  Image normalisation method
                                        (``"8bit"`` for the 8-bit
                                        tone-mapped JPGs HyperSim
                                        ships).
        max_depth_m:                    Clip depths to this value in
                                        metres. Hypersim renders sky /
                                        background as ``inf``, plus
                                        through-window views can hit
                                        100 m+; for indoor-consistent
                                        training, 20 m is a sane cap.
        depth_convention:               ``"distance"`` (default) keeps
                                        the raw distance-to-camera-
                                        centre values from HyperSim's
                                        HDF5 files. ``"z"`` performs
                                        Niklaus-style conversion to
                                        planar Z-depth using a fixed
                                        ``fov_y_deg``.
        fov_y_deg:                      Vertical FoV in degrees, only
                                        used when
                                        ``depth_convention="z"``.
                                        HyperSim's nominal value is
                                        ~60 deg; per-scene intrinsics
                                        vary slightly (tilt-shift
                                        photography) but the residual
                                        error is small for downstream
                                        training.
        minmax_normalise:               If True, apply per-sample
                                        affine renormalisation
                                        ``(prompt, gt) -> (prompt - h_min) / scale``
                                        BEFORE :class:`CHMCorruptor`
                                        runs, and emit a 4-tuple with
                                        a ``[2]``-tensor ``meta`` so
                                        the lightning module can
                                        recover metric values for
                                        validation metrics.
        chm_corruption:                 Dict of kwargs for
                                        :class:`CHMCorruptor`.
                                        **Important**: when
                                        ``minmax_normalise=True``, the
                                        corruptor sees values in
                                        ~``[0, 1]``, so any params
                                        with metric units (e.g.
                                        ``height_noise_sigma``) must
                                        be set in normalised units in
                                        the YAML.
        chm_contrastive_corruption:     Dict of kwargs for
                                        :class:`CHMContrastiveCorruptor`.
                                        Same scale-unit caveat as
                                        ``chm_corruption``. When
                                        provided, ``__getitem__``
                                        returns a 5-tuple (or 6-tuple
                                        with ``meta`` if min-max norm
                                        is on).
        transforms_config:              Dict of kwargs for
                                        :class:`Transformations`.
        channels:                       Number of image channels to
                                        read (always 3 for HyperSim
                                        previews).
    """

    def __init__(
        self,
        data_dir: str,
        held_out_pct: float = 0.0,
        role: str = "all",
        normalisation: str = "8bit",
        max_depth_m: float = 1000.0,
        depth_convention: str = "distance",
        fov_y_deg: float = 60.0,
        minmax_normalise: bool = False,
        minmax_min_scale: float = 1.0e-2,
        chm_corruption: dict | None = None,
        chm_contrastive_corruption: dict | None = None,
        transforms_config: dict | None = None,
        channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        if not 0.0 <= held_out_pct < 1.0:
            raise ValueError(f"held_out_pct must be in [0, 1), got {held_out_pct}")
        if role not in ("train", "val", "all"):
            raise ValueError(f"role must be 'train', 'val' or 'all', got {role!r}")
        if max_depth_m <= 0:
            raise ValueError(f"max_depth_m must be > 0, got {max_depth_m}")
        if depth_convention not in ("distance", "z"):
            raise ValueError(
                f"depth_convention must be 'distance' or 'z', got {depth_convention!r}"
            )

        self.data_dir = Path(data_dir)
        self.held_out_pct = float(held_out_pct)
        self.role = role
        self.channels = int(channels)
        self.max_depth_m = float(max_depth_m)
        self.depth_convention = depth_convention
        self.fov_y_deg = float(fov_y_deg)
        self.minmax_normalise = bool(minmax_normalise)
        self.minmax_min_scale = float(minmax_min_scale)

        self.normalize = Normalization(method=normalisation)
        self.corruptor = CHMCorruptor(**(chm_corruption or {}))
        self.contrastive_corruptor: CHMContrastiveCorruptor | None = None
        if chm_contrastive_corruption is not None:
            self.contrastive_corruptor = CHMContrastiveCorruptor(
                **dict(chm_contrastive_corruption)
            )
        self.transformations: Transformations | None = None
        if transforms_config is not None:
            self.transformations = Transformations(**dict(transforms_config))

        # Caches for the on-the-fly distance->Z conversion. Filled lazily
        # the first time a frame at a given (H, W) is read, since the
        # conversion grid only depends on shape + intrinsic and is
        # identical for every frame at that resolution.
        self._distance_to_z_cache: dict[tuple[int, int], np.ndarray] = {}

        self.samples: list[dict] = []
        self._discover_samples()
        logger.info(
            "HypersimDepthDataset: %d samples (role=%s, held_out_pct=%.2f, "
            "minmax_normalise=%s, depth_convention=%s)",
            len(self.samples), self.role, self.held_out_pct,
            self.minmax_normalise, self.depth_convention,
        )
        if not self.samples:
            raise RuntimeError(
                f"HypersimDepthDataset: no usable frames found under {self.data_dir}. "
                f"Expected layout: ai_VVV_NNN/images/scene_cam_XX_geometry_hdf5/"
                f"frame.IIII.depth_meters.hdf5 paired with "
                f"scene_cam_XX_final_preview/frame.IIII.tonemap.jpg"
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_samples(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"HypersimDepthDataset: data_dir does not exist: {self.data_dir}"
            )

        scenes_all = sorted(
            d for d in os.listdir(self.data_dir)
            if (self.data_dir / d).is_dir() and d.startswith("ai_")
        )
        scenes = self._apply_role_carve(scenes_all)

        for scene_id in scenes:
            scene_dir = self.data_dir / scene_id
            for geom_dir in sorted(scene_dir.glob(_GEOMETRY_DIR_GLOB)):
                # geom_dir name: "scene_cam_XX_geometry_hdf5"
                cam = geom_dir.name.split("_")[2]
                preview_dir = scene_dir / _PREVIEW_DIR_TEMPLATE.format(cam=cam)
                if not preview_dir.is_dir():
                    continue
                for depth_file in sorted(geom_dir.glob(f"*{_DEPTH_FILE_SUFFIX}")):
                    frame_stem = depth_file.name[: -len(_DEPTH_FILE_SUFFIX)]
                    rgb_file = self._resolve_rgb(preview_dir, frame_stem)
                    if rgb_file is None:
                        continue
                    self.samples.append({
                        "image": str(rgb_file),
                        "depth": str(depth_file),
                        "scene_id": scene_id,
                        "cam": cam,
                        "frame": frame_stem,
                    })

    @staticmethod
    def _resolve_rgb(preview_dir: Path, frame_stem: str) -> Path | None:
        """Find the matching RGB file for ``frame_stem`` under ``preview_dir``.

        Tries the candidate suffixes from :data:`_RGB_FILE_SUFFIXES` in
        order; returns the first hit or ``None`` if no JPG/PNG is
        available (which we treat as an incomplete frame and skip).
        """
        for suffix in _RGB_FILE_SUFFIXES:
            cand = preview_dir / f"{frame_stem}{suffix}"
            if cand.exists():
                return cand
        return None

    def _apply_role_carve(self, scenes_all: list[str]) -> list[str]:
        """Deterministic, sorted-by-scene_id train/val carve."""
        if self.role == "all" or self.held_out_pct == 0.0:
            return scenes_all
        n = len(scenes_all)
        n_val = max(1, int(round(self.held_out_pct * n)))
        n_train = n - n_val
        if self.role == "train":
            return scenes_all[:n_train]
        return scenes_all[n_train:]

    # ------------------------------------------------------------------
    # Depth I/O
    # ------------------------------------------------------------------
    def _read_depth(self, path: str) -> np.ndarray:
        """Read a Hypersim depth HDF5 and return float32 metres, masked.

        Returns a float32 ``[H, W]`` array where invalid pixels (``inf``,
        ``NaN``, or ``> max_depth_m``) are set to 0. The ``> max_depth_m``
        clip is applied **after** invalidation so the sentinel-zero
        convention is preserved for losses that use ``ignore_zero=True``.
        """
        with h5py.File(path, "r") as f:
            # Apple's HDF5 files store the array under a single dataset
            # named "dataset" -- documented in the ml-hypersim README and
            # consistent across all geometry channels.
            d = np.asarray(f["dataset"], dtype=np.float32)

        if d.ndim != 2:
            raise RuntimeError(
                f"Expected 2D depth array, got shape {d.shape} for {path}"
            )

        invalid = ~np.isfinite(d) | (d <= 0.0)
        if self.depth_convention == "z":
            d = self._distance_to_z(d)
        d = np.where(invalid, 0.0, d).astype(np.float32)
        # Cap at max_depth_m -- any pixel beyond that becomes 0 (sentinel
        # for "no signal") so far-distance through-window pixels don't
        # warp the loss in indoor scenes.
        far = d > self.max_depth_m
        if far.any():
            d = d.copy()
            d[far] = 0.0
        return d

    def _distance_to_z(self, distance: np.ndarray) -> np.ndarray:
        """Convert HyperSim distance-to-camera-centre to planar Z-depth.

        Uses a fixed-FoV intrinsic (no per-scene tilt-shift correction).
        Sufficient for training; the per-scene intrinsic variation in
        HyperSim is small enough to be absorbed by the data augmentation.
        """
        H, W = distance.shape
        cache_key = (H, W)
        factor = self._distance_to_z_cache.get(cache_key)
        if factor is None:
            fy = 0.5 * H / math.tan(0.5 * math.radians(self.fov_y_deg))
            fx = fy  # square pixels (HyperSim default)
            cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
            # ``indexing="xy"`` returns ``(xx, yy)`` where ``xx`` ranges
            # over column indices [0..W-1] and ``yy`` over rows
            # [0..H-1], both shaped ``(H, W)``. Swapping these axes was
            # the original bug -- it left the centre pixel with a non-
            # zero ``npc`` and inflated the centre conversion factor.
            xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
            npc_x = (xx - cx) / fx
            npc_y = (yy - cy) / fy
            factor = 1.0 / np.sqrt(npc_x * npc_x + npc_y * npc_y + 1.0)
            self._distance_to_z_cache[cache_key] = factor.astype(np.float32)
        return distance * factor

    # ------------------------------------------------------------------
    # PyTorch Dataset
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # --- read RGB ---
        bgr = cv2.imread(s["image"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread returned None for {s['image']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Channels-first to match the SynRS3D / ARKitScenes convention;
        # the joint augment decorator below flips back as needed.
        image = rgb.transpose(2, 0, 1)[: self.channels]  # [3, H, W] uint8

        # --- read clean depth (substrate for both prompt and gt) ---
        gt_depth = self._read_depth(s["depth"])  # [H, W] float32 metres
        # Resize depth to image resolution if HyperSim renderer used a
        # different geometry resolution than the preview JPG. In practice
        # they always match, but defensive resizing is cheap.
        if gt_depth.shape != image.shape[1:]:
            gt_depth = cv2.resize(
                gt_depth, (image.shape[2], image.shape[1]),
                interpolation=cv2.INTER_LINEAR,
            )

        # --- per-sample min-max normalisation (BEFORE corruption) ---
        # User-explicit ordering: compute (shift, scale) on the clean
        # signal so cutout-induced zeros don't pull the min down to 0.
        if self.minmax_normalise:
            valid_mask = gt_depth > 0
            prompt_clean, gt_norm, shift, scale = minmax_normalise_pair(
                gt_depth, gt_depth,
                valid_mask=valid_mask,
                min_scale=self.minmax_min_scale,
            )
        else:
            prompt_clean = gt_depth
            gt_norm = gt_depth
            shift, scale = 0.0, 1.0

        # --- corrupt the prompt (operates on whichever scale we picked) ---
        chm = self.corruptor(prompt_clean.copy())

        # --- normalise image ([0, 1] float32) ---
        image = self.normalize.apply(image, channels=self.channels)

        # --- contrastive views from clean (already-normalised) prompt ---
        chm_v1: np.ndarray | None = None
        chm_v2: np.ndarray | None = None
        if self.contrastive_corruptor is not None:
            chm_v1 = self.contrastive_corruptor(prompt_clean)
            chm_v2 = self.contrastive_corruptor(prompt_clean)

        # --- joint spatial augmentation (same op on every CHM) ---
        if self.transformations is not None:
            mask_list = [chm, gt_norm]
            if chm_v1 is not None:
                mask_list.extend([chm_v1, chm_v2])
            image, transformed = self.transformations.apply(image, mask_list)
            chm, gt_norm = transformed[0], transformed[1]
            if chm_v1 is not None:
                chm_v1, chm_v2 = transformed[2], transformed[3]

        # --- to tensors ---
        image_t = torch.as_tensor(image, dtype=torch.float32)
        chm_t = torch.as_tensor(chm, dtype=torch.float32)
        gt_t = torch.as_tensor(gt_norm, dtype=torch.float32)
        if chm_t.ndim == 2:
            chm_t = chm_t.unsqueeze(0)
        if gt_t.ndim == 2:
            gt_t = gt_t.unsqueeze(0)

        meta_t = torch.as_tensor([shift, scale], dtype=torch.float32)

        if chm_v1 is None:
            return image_t, chm_t, gt_t, meta_t

        v1_t = torch.as_tensor(chm_v1, dtype=torch.float32)
        v2_t = torch.as_tensor(chm_v2, dtype=torch.float32)
        if v1_t.ndim == 2:
            v1_t = v1_t.unsqueeze(0)
        if v2_t.ndim == 2:
            v2_t = v2_t.unsqueeze(0)
        return image_t, chm_t, v1_t, v2_t, gt_t, meta_t
