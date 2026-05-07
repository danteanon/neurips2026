"""ARKitScenes depth-upsampling dataset for the Dinov3HeightModel CATT stack.

A "sample" here is a single frame from the ARKitScenes ``upsampling`` split,
captured by an iPhone with simultaneous wide-angle RGB and ARKit LiDAR depth.
The training task is dense **depth** prediction (in metres) given:

  * ``image``   — RGB ``wide/`` frame, native 1440 × 1920, uint8.
  * ``chm``     — low-resolution LiDAR depth ``lowres_depth/`` (192 × 256,
                  uint16 mm), bilinearly upsampled to image size and then
                  optionally degraded further by :class:`CHMCorruptor` to
                  simulate the production-time misalignment / dropout
                  regime the model is supposed to be robust to.
  * ``gt``      — high-resolution Faro laser scanner depth
                  ``highres_depth/`` (1440 × 1920, uint16 mm).

Why this layout matches the height-estimation pipeline 1-to-1
-------------------------------------------------------------
The :class:`HeightEstimationModule` consumes a 3-tuple ``(image, chm, gt)``
or, when CATT / VICReg are enabled, a 5-tuple
``(image, chm, chm_v1, chm_v2, gt)`` produced by the dataset's
``chm_contrastive_corruption`` block.  The naming is generic — "chm" and
"height" are just labels for *prompt* and *dense regression target*. For
ARKitScenes those are LiDAR depth and laser-scanner depth respectively;
the model code stays untouched.

Conventions matching :class:`SynRS3DHeightDataset`
--------------------------------------------------
* Image is read as 3-channel uint8 in **RGB** order, normalised by 255
  via :class:`Normalization` (``"8bit"``).
* Depths are read as uint16 millimetres and divided by 1000 to land
  float32 metres — same metric units as SynRS3D heights, so the loss
  scales (L1 in metres) and the per-layer probe head's output range
  carry over without retuning.
* No-return / invalid pixels in the highres GT are stored as ``0``.
  The training-time loss should set ``ignore_zero=True`` to skip them
  (zero-depth is a structural sentinel for "no LiDAR return", not an
  actual measurement of zero metres).
* Optional contrastive views (``chm_v1``, ``chm_v2``) are produced from
  the **highres GT depth** (clean substrate) by
  :class:`CHMContrastiveCorruptor` — exactly the SynRS3D / Open-Canopy
  pattern. The two views supply the CATT consistency signal.
* Optional joint augmentation (crop / spatial / color) is applied with
  :class:`Transformations` exactly like the SynRS3D dataset; every CHM
  view (main + v1 + v2) goes through the same spatial op so per-token
  alignment between views is preserved.

Train / val carve
-----------------
The download script ships ``Training/`` only (the ARKitScenes
``Validation/`` split is a separate ~3 GB download). To support a
single-disk-tree training run we add a deterministic, scene-level
carve via two kwargs:

  * ``held_out_pct``: fraction of scenes (sorted by video_id) reserved
    for validation. Default ``0.0`` (use every scene under ``split``).
  * ``role``: ``"train"`` keeps the first ``(1 - held_out_pct)`` of
    scenes; ``"val"`` keeps the last ``held_out_pct``.

Carving by ``video_id`` (not by frame) prevents leakage at the scene
level — the val frames are from videos the model never sees during
training. With 1246 complete scenes the default ``held_out_pct=0.10``
yields ≈ 1121 train scenes / 125 val scenes (~21 k / ~2.7 k frames).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.synrs3d_dataset import CHMContrastiveCorruptor, CHMCorruptor
from utils.normalisations import Normalization
from utils.prompt_normalisation import minmax_normalise_pair
from utils.transformations import Transformations

logger = logging.getLogger(__name__)


# ARKitScenes upsampling stores RGB under "wide/". Some older ARKit splits
# (3dod / raw) ship a "color/" folder — keep that as a fallback so the
# dataset can be re-pointed at a different on-disk layout without code
# changes. ``_DEPTH_DIR_GT`` and ``_DEPTH_DIR_PROMPT`` are non-negotiable.
_RGB_DIR_CANDIDATES = ("wide", "color")
_DEPTH_DIR_GT = "highres_depth"
_DEPTH_DIR_PROMPT = "lowres_depth"


class ARKitScenesDepthDataset(Dataset):
    """ARKitScenes upsampling-split dataset returning ``(image, prompt, gt)``.

    Constructor signature mirrors :class:`SynRS3DHeightDataset` so the
    same :class:`SegmentationDataModule` config surface works without
    changes (``data_dir`` positional + everything else as kwargs).

    Args:
        data_dir:                       Root containing ``Training/`` and
                                        optionally ``Validation/``.
        split:                          On-disk subdirectory to read from
                                        (``"Training"`` or
                                        ``"Validation"``). Defaults to
                                        ``"Training"`` so a default config
                                        works with only the training
                                        download present.
        held_out_pct:                   Fraction of scenes (sorted by
                                        ``video_id``) reserved for the
                                        ``"val"`` role.  ``0.0`` disables
                                        the carve (use every scene under
                                        ``split``).
        role:                           ``"train"`` keeps
                                        ``floor((1 - held_out_pct) * N)``
                                        scenes from the front; ``"val"``
                                        keeps the rest from the back;
                                        ``"all"`` ignores the carve.
        normalisation:                  Image normalisation method
                                        (``"8bit"`` for ARKitScenes uint8
                                        RGB).
        max_depth_m:                    Clip GT and prompt depths to this
                                        value in metres. ARKitScenes
                                        indoor scenes are typically
                                        < 10 m; clipping prevents stray
                                        long-range LiDAR returns from
                                        skewing the loss.
        chm_corruption:                 Dict of kwargs for
                                        :class:`CHMCorruptor`. Applied to
                                        the upsampled lowres LiDAR prompt
                                        on top of its native degradation.
                                        Set ``{}`` (or omit) to use the
                                        raw upsampled prompt.
        chm_contrastive_corruption:     Dict of kwargs for
                                        :class:`CHMContrastiveCorruptor`.
                                        When provided, ``__getitem__``
                                        returns a 5-tuple
                                        ``(image, chm, chm_v1, chm_v2,
                                        gt)`` — required by CATT and
                                        VICReg. Two views are sampled
                                        from the **highres GT depth** so
                                        the contrastive supervision sees
                                        a clean substrate.
        transforms_config:              Dict of kwargs for
                                        :class:`Transformations`.
        channels:                       Number of image channels to read
                                        (always 3 for ARKitScenes wide).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "Training",
        held_out_pct: float = 0.0,
        role: str = "all",
        normalisation: str = "8bit",
        max_depth_m: float = 1000.0,
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
            raise ValueError(
                f"held_out_pct must be in [0, 1), got {held_out_pct}"
            )
        if role not in ("train", "val", "all"):
            raise ValueError(
                f"role must be 'train', 'val' or 'all', got {role!r}"
            )
        if max_depth_m <= 0:
            raise ValueError(f"max_depth_m must be > 0, got {max_depth_m}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.held_out_pct = float(held_out_pct)
        self.role = role
        self.channels = int(channels)
        self.max_depth_m = float(max_depth_m)
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

        self.samples: list[dict] = []
        self._discover_samples()
        logger.info(
            "ARKitScenesDepthDataset: %d samples (split=%s, role=%s, "
            "held_out_pct=%.2f)",
            len(self.samples), self.split, self.role, self.held_out_pct,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_samples(self) -> None:
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"ARKitScenes split dir does not exist: {self.split_dir}"
            )

        scenes_all = sorted(
            d for d in os.listdir(self.split_dir)
            if (self.split_dir / d).is_dir()
        )
        scenes = self._apply_role_carve(scenes_all)

        for scene_id in scenes:
            scene_dir = self.split_dir / scene_id
            rgb_dir = self._resolve_rgb_dir(scene_dir)
            gt_dir = scene_dir / _DEPTH_DIR_GT
            prompt_dir = scene_dir / _DEPTH_DIR_PROMPT
            if rgb_dir is None or not gt_dir.is_dir() or not prompt_dir.is_dir():
                continue

            for rgb_file in sorted(rgb_dir.glob("*.png")):
                fname = rgb_file.name
                gt_file = gt_dir / fname
                prompt_file = prompt_dir / fname
                if gt_file.exists() and prompt_file.exists():
                    self.samples.append({
                        "image": str(rgb_file),
                        "prompt": str(prompt_file),
                        "gt": str(gt_file),
                        "scene_id": scene_id,
                    })

    def _apply_role_carve(self, scenes_all: list[str]) -> list[str]:
        """Deterministic, sorted-by-video_id train/val scene carve."""
        if self.role == "all" or self.held_out_pct == 0.0:
            return scenes_all
        n = len(scenes_all)
        n_val = max(1, int(round(self.held_out_pct * n)))
        n_train = n - n_val
        if self.role == "train":
            return scenes_all[:n_train]
        return scenes_all[n_train:]

    @staticmethod
    def _resolve_rgb_dir(scene_dir: Path) -> Path | None:
        for cand in _RGB_DIR_CANDIDATES:
            cand_path = scene_dir / cand
            if cand_path.is_dir():
                return cand_path
        return None

    # ------------------------------------------------------------------
    # PyTorch Dataset
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _read_depth(self, path: str) -> np.ndarray:
        """Read a uint16-mm depth PNG and return float32 metres, clipped."""
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"cv2.imread returned None for {path}")
        d = raw.astype(np.float32) / 1000.0
        np.clip(d, 0.0, self.max_depth_m, out=d)
        return d

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # --- read RGB ([H, W, 3] uint8 BGR -> RGB) ---
        bgr = cv2.imread(s["image"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread returned None for {s['image']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = rgb.transpose(2, 0, 1)[: self.channels]  # [3, H, W] uint8

        # --- read GT depth (clean substrate) ---
        gt_depth = self._read_depth(s["gt"])  # [H, W] float32 metres

        # --- read low-res LiDAR prompt and upsample to image resolution ---
        lowres = self._read_depth(s["prompt"])  # [192, 256] float32 metres
        prompt = cv2.resize(
            lowres, (image.shape[2], image.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )

        # --- optional per-sample min-max normalisation (BEFORE corruption) ---
        if self.minmax_normalise:
            valid_mask = prompt > 0
            prompt, gt_depth, shift, scale = minmax_normalise_pair(
                prompt, gt_depth,
                valid_mask=valid_mask,
                min_scale=self.minmax_min_scale,
            )
        else:
            shift, scale = 0.0, 1.0

        # --- corrupt the prompt ---
        chm = self.corruptor(prompt)

        # --- normalise image ([0, 1] float32) ---
        image = self.normalize.apply(image, channels=self.channels)

        # --- contrastive views from clean GT depth (CATT / VICReg) ---
        chm_v1: np.ndarray | None = None
        chm_v2: np.ndarray | None = None
        if self.contrastive_corruptor is not None:
            chm_v1 = self.contrastive_corruptor(gt_depth)
            chm_v2 = self.contrastive_corruptor(gt_depth)

        # --- joint augmentation (same spatial op on every CHM) ---
        if self.transformations is not None:
            mask_list = [chm, gt_depth]
            if chm_v1 is not None:
                mask_list.extend([chm_v1, chm_v2])
            image, transformed = self.transformations.apply(image, mask_list)
            chm, gt_depth = transformed[0], transformed[1]
            if chm_v1 is not None:
                chm_v1, chm_v2 = transformed[2], transformed[3]

        # --- to tensors ---
        image_t = torch.as_tensor(image, dtype=torch.float32)
        chm_t = torch.as_tensor(chm, dtype=torch.float32)
        gt_t = torch.as_tensor(gt_depth, dtype=torch.float32)
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
