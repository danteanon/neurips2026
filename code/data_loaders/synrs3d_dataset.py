"""
SynRS3D dataset for height estimation training.

Each sample returns (image, corrupted_chm, gt_height).  The corrupted CHM is
synthesised on-the-fly from the ground-truth nDSM and land-cover mask to
simulate temporal LiDAR mismatch.
"""

import os
import glob
import random
import logging

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import zoom

from utils.normalisations import Normalization
from utils.transformations import Transformations

logger = logging.getLogger(__name__)


class CHMCorruptor:
    """On-the-fly corruption of a ground-truth nDSM to simulate an old CHM.

    Pipeline (applied in order on a fresh copy of the GT nDSM):
      1. Random region cutout — with probability ``cutout_prob``, zero out a
         contiguous fraction of the CHM. **Class-agnostic** (does *not*
         consult ``ss_mask``). Two modes, picked per-call:

           * ``"large"`` (with prob ``cutout_one_large_prob``): a single
             rectangle whose area is sampled uniformly from
             ``cutout_area_range`` × image_area, with a random aspect ratio
             from ``cutout_aspect_ratio``. Models a large contiguous LiDAR
             gap (e.g. survey skip, cloud cover at acquisition time).
           * ``"scattered"`` (otherwise): ``N`` smaller rectangles whose
             combined nominal area lands in ``cutout_area_range``, with
             ``N`` uniform in ``cutout_n_scattered``. Models a fragmented
             coverage gap. Rectangles can overlap; realised area can be
             slightly less than nominal.

         Replaces the previous per-pixel "object_removal" step which was
         class-aware salt-and-pepper inside (tree | building) masks — that
         was visually invisible after steps 3-5 and never tested coherent
         spatial holes in LiDAR coverage. ``ss_mask`` is no longer consulted
         by the corruptor at all.

      2. Spatial misalignment — roll by a random offset up to ``±max_shift``
         pixels per axis. Default ``max_shift=24`` (3× the previous default
         of 8) to model the realistic LiDAR↔imagery temporal misregistration
         budget.
      3. Resolution degradation — bilinear downsample by ``resolution_factor``
         then upsample back to the original size.
      4. Gaussian blur — spatial smoothing (σ in pixels).
      5. Additive height noise — Gaussian, σ = ``height_noise_sigma``.
      6. Always-on baseline sensor noise — Gaussian, σ =
         ``always_on_noise_sigma``. Models residual return noise that every
         real LiDAR reading carries; breaks the sharp ``CHM == 0`` boundary
         on ground pixels so the encoder cannot gate on it.
      7. Full dropout — replace entire CHM with zeros with probability
         ``full_dropout_prob``. Applied *last* so "no LiDAR at all" remains a
         recognisable, bit-exact-zero regime distinct from the noisy steps.

    Notes
    -----
    Implementation is numpy/scipy-based and runs in DataLoader worker
    processes via ``__getitem__``. The cutout step is intentionally kept as
    a small, kornia-shaped primitive so it can be lifted to GPU
    (``LightningModule.on_after_batch_transfer`` + ``kornia.augmentation``)
    later without changing the public spec.

    Args
    ----
    cutout_prob:           Probability of applying the cutout step at all.
    cutout_area_range:     ``(low, high)`` fraction of image area to zero
                            (e.g. ``(0.10, 0.30)``).
    cutout_one_large_prob: Probability of "large" mode within cutout (rest
                            is "scattered").
    cutout_n_scattered:    ``(low, high)`` *inclusive* number of small
                            rectangles in "scattered" mode.
    cutout_aspect_ratio:   ``(low, high)`` w/h aspect ratio per rectangle,
                            sampled uniformly.
    max_shift:             Max pixel shift per axis (default 24).
    resolution_factor:     Down-then-up sample ratio (1.0 = no
                            degradation).
    gaussian_blur_sigma:   σ for the spatial blur step (0 = disabled).
    height_noise_sigma:    σ for the additive height-noise step
                            (0 = disabled).
    full_dropout_prob:     Probability of zeroing the entire CHM.
    always_on_noise_sigma: σ for the always-applied baseline noise.
    """

    def __init__(
        self,
        cutout_prob: float = 0.5,
        cutout_area_range: tuple = (0.10, 0.30),
        cutout_one_large_prob: float = 0.5,
        cutout_n_scattered: tuple = (3, 8),
        cutout_aspect_ratio: tuple = (0.3, 3.3),
        max_shift: int = 24,
        resolution_factor: float = 0.25,
        gaussian_blur_sigma: float = 0.0,
        height_noise_sigma: float = 1.5,
        full_dropout_prob: float = 0.2,
        always_on_noise_sigma: float = 0.0,
    ):
        if not 0.0 <= cutout_prob <= 1.0:
            raise ValueError(f"cutout_prob must be in [0, 1], got {cutout_prob}")
        if not 0.0 <= cutout_one_large_prob <= 1.0:
            raise ValueError(
                f"cutout_one_large_prob must be in [0, 1], got {cutout_one_large_prob}"
            )
        lo, hi = cutout_area_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"cutout_area_range must satisfy 0 ≤ lo ≤ hi ≤ 1, got {cutout_area_range}"
            )
        n_lo, n_hi = cutout_n_scattered
        if not (1 <= int(n_lo) <= int(n_hi)):
            raise ValueError(
                f"cutout_n_scattered must satisfy 1 ≤ lo ≤ hi, got {cutout_n_scattered}"
            )
        ar_lo, ar_hi = cutout_aspect_ratio
        if not (0.0 < ar_lo <= ar_hi):
            raise ValueError(
                f"cutout_aspect_ratio must satisfy 0 < lo ≤ hi, got {cutout_aspect_ratio}"
            )

        self.cutout_prob = float(cutout_prob)
        self.cutout_area_range = (float(lo), float(hi))
        self.cutout_one_large_prob = float(cutout_one_large_prob)
        self.cutout_n_scattered = (int(n_lo), int(n_hi))
        self.cutout_aspect_ratio = (float(ar_lo), float(ar_hi))
        self.max_shift = int(max_shift)
        self.resolution_factor = float(resolution_factor)
        self.gaussian_blur_sigma = float(gaussian_blur_sigma)
        self.height_noise_sigma = float(height_noise_sigma)
        self.full_dropout_prob = float(full_dropout_prob)
        self.always_on_noise_sigma = float(always_on_noise_sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_rect(self, H: int, W: int, target_area: int) -> tuple:
        """Sample a rectangle ``(y, x, h, w)`` with area ≈ ``target_area``."""
        ar_lo, ar_hi = self.cutout_aspect_ratio
        ratio = float(np.random.uniform(ar_lo, ar_hi))
        h = int(round(np.sqrt(max(target_area, 1) / ratio)))
        w = int(round(np.sqrt(max(target_area, 1) * ratio)))
        h = max(1, min(h, H))
        w = max(1, min(w, W))
        y = int(np.random.randint(0, H - h + 1))
        x = int(np.random.randint(0, W - w + 1))
        return y, x, h, w

    def _apply_cutout(self, chm: np.ndarray) -> np.ndarray:
        """Zero a region of ``chm`` in either ``"large"`` or ``"scattered"`` mode."""
        H, W = chm.shape
        target_frac = float(np.random.uniform(*self.cutout_area_range))
        target_area = int(target_frac * H * W)
        if target_area <= 0:
            return chm

        if random.random() < self.cutout_one_large_prob:
            y, x, h, w = self._sample_rect(H, W, target_area)
            chm[y:y + h, x:x + w] = 0.0
        else:
            n_lo, n_hi = self.cutout_n_scattered
            n_rects = int(np.random.randint(n_lo, n_hi + 1))
            per_rect = max(target_area // n_rects, 1)
            for _ in range(n_rects):
                y, x, h, w = self._sample_rect(H, W, per_rect)
                chm[y:y + h, x:x + w] = 0.0
        return chm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, ndsm: np.ndarray, ss_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            ndsm:    ``[H, W]`` float32 ground-truth normalised DSM.
            ss_mask: kept for backward signature compatibility but unused —
                     all corruption is now class-agnostic.

        Returns:
            ``[H, W]`` float32 corrupted CHM.
        """
        del ss_mask  # signature kept for back-compat with the dataset call site
        chm = ndsm.copy()

        # 1 — random region cutout (class-agnostic)
        if self.cutout_prob > 0 and random.random() < self.cutout_prob:
            chm = self._apply_cutout(chm)

        # 2 — spatial misalignment
        if self.max_shift > 0:
            dy = random.randint(-self.max_shift, self.max_shift)
            dx = random.randint(-self.max_shift, self.max_shift)
            chm = np.roll(chm, shift=(dy, dx), axis=(0, 1))

        # 3 — resolution degradation
        if 0 < self.resolution_factor < 1:
            h, w = chm.shape
            small = zoom(chm, self.resolution_factor, order=1)
            chm = zoom(small, (h / small.shape[0], w / small.shape[1]), order=1)
            chm = chm[:h, :w]  # trim any rounding artefacts

        # 4 — gaussian blur
        if self.gaussian_blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            chm = gaussian_filter(chm, sigma=self.gaussian_blur_sigma)

        # 5 — height noise
        if self.height_noise_sigma > 0:
            chm = chm + np.random.randn(*chm.shape).astype(np.float32) * self.height_noise_sigma
            np.clip(chm, 0, None, out=chm)

        # 6 — baseline sensor noise (always applied; clipped at 0)
        if self.always_on_noise_sigma > 0:
            chm = chm + np.random.randn(*chm.shape).astype(np.float32) * self.always_on_noise_sigma
            np.clip(chm, 0, None, out=chm)

        # 7 — full dropout
        if random.random() < self.full_dropout_prob:
            chm = np.zeros_like(chm)

        return chm.astype(np.float32)

    @classmethod
    def for_regime(cls, regime: str) -> "CHMCorruptor":
        """Return a deterministic corruptor preset for a given evaluation regime.

        Each regime isolates a single degradation axis so that paired deltas
        against ``clean`` measure exactly one effect. Notes:

        * ``"masked"`` now applies the **new** cutout step (random spatial
          rectangles, 10–30 % area) deterministically (``cutout_prob=1.0``)
          rather than the old class-aware salt-and-pepper. This is the same
          regime *name* used by ``run_post_training_eval.py`` but with the
          corruption family swapped.
        * ``"shifted"`` uses the new default ``max_shift=24`` so the regime
          tracks the training-time misalignment budget.
        """
        common_off = dict(
            cutout_prob=0.0,
            max_shift=0,
            resolution_factor=1.0,
            gaussian_blur_sigma=0.0,
            height_noise_sigma=0.0,
            full_dropout_prob=0.0,
            always_on_noise_sigma=0.0,
        )
        presets = {
            "clean":    {**common_off},
            "shifted":  {**common_off, "max_shift": 24},
            "masked":   {**common_off, "cutout_prob": 1.0,
                         "cutout_area_range": (0.10, 0.30),
                         "cutout_one_large_prob": 0.5,
                         "cutout_n_scattered": (3, 8)},
            "degraded": {**common_off, "resolution_factor": 0.25,
                         "height_noise_sigma": 1.5},
            "blurred":  {**common_off, "gaussian_blur_sigma": 3.0},
            "zero":     {**common_off, "full_dropout_prob": 1.0},
        }
        if regime not in presets:
            raise ValueError(f"Unknown regime {regime!r}. Choose from {sorted(presets)}")
        return cls(**presets[regime])


class CHMContrastiveCorruptor:
    """Alignment-preserving CHM corruption for VICReg view pairs.

    The contrastive objective requires per-token correspondence between
    the two augmented views — i.e. token ``(b, t)`` of view 1 must
    represent the same spatial location as token ``(b, t)`` of view 2.
    The training-time :class:`CHMCorruptor` includes spatial roll
    (``max_shift``), region cutout, and full dropout — all of which
    *break* per-token alignment between two independent samples and
    would silently destroy the invariance signal.

    This corruptor restricts itself to **pixel-wise / global** ops that
    leave spatial alignment intact:

      1. Multiplicative height-scale jitter — global × ``s ∈
         [s_lo, s_hi]``. Models calibration drift between LiDAR
         acquisitions.
      2. Gaussian blur with σ sampled per view from
         ``gaussian_blur_sigma_range``. Models smoothing variation.
      3. Resolution degradation — bilinear down-then-up sample by
         factor ``r ∈ [r_lo, r_hi]``. Output shape is preserved exactly
         so token positions remain aligned.
      4. Additive height noise with σ sampled per view from
         ``height_noise_sigma_range``. The most important augmentation
         — different noise realisations per view force the encoder to
         build noise-invariant token features.
      5. Optional salt-pepper sparsification with probability
         ``salt_pepper_prob`` and rate ``salt_pepper_amount``. Models
         scattered LiDAR returns being missed; pixel-wise so alignment
         is preserved.

    Each of the five augmentations uses an independently-sampled
    parameter per view — i.e. the two views see *different* noise
    realisations, blur σ, scale factor, etc. (matching VICReg's
    "different positives from same source" pattern). They share input
    pixels at every position, only their corruptions differ.

    Notes on what is **not** in here:
      * No region cutout — masks specific tokens differently per view.
      * No spatial roll / shift — translates token positions.
      * No full dropout — turns one view into a zero tensor while the
        other has signal, trivially satisfying invariance via constant
        output and destroying variance.
      * No flips / rotations — those are applied jointly later via
        ``Transformations.apply``, so both views see the same flip.

    Args:
        height_scale_range:        ``(s_lo, s_hi)`` multiplicative
                                   global gain per view. Set to
                                   ``(1.0, 1.0)`` to disable.
        gaussian_blur_sigma_range: ``(σ_lo, σ_hi)`` per-view blur σ in
                                   pixels. Set ``(0.0, 0.0)`` to disable.
        resolution_factor_range:   ``(r_lo, r_hi)`` per-view down-up
                                   sample factor. Set ``(1.0, 1.0)`` to
                                   disable. Values < 1 trigger a
                                   bilinear ``zoom`` round-trip.
        height_noise_sigma_range:  ``(σ_lo, σ_hi)`` per-view additive
                                   Gaussian noise σ. Set ``(0.0, 0.0)``
                                   to disable.
        salt_pepper_prob:          Probability of applying the
                                   salt-pepper step at all (per view).
                                   Set 0.0 to disable.
        salt_pepper_amount:        Fraction of pixels zeroed when the
                                   step fires.
    """

    def __init__(
        self,
        height_scale_range: tuple = (0.85, 1.15),
        gaussian_blur_sigma_range: tuple = (0.0, 1.5),
        resolution_factor_range: tuple = (0.5, 1.0),
        height_noise_sigma_range: tuple = (0.5, 2.0),
        salt_pepper_prob: float = 0.0,
        salt_pepper_amount: float = 0.05,
    ):
        for name, rng in (
            ("height_scale_range", height_scale_range),
            ("gaussian_blur_sigma_range", gaussian_blur_sigma_range),
            ("resolution_factor_range", resolution_factor_range),
            ("height_noise_sigma_range", height_noise_sigma_range),
        ):
            lo, hi = rng
            if not (lo <= hi):
                raise ValueError(f"{name} must satisfy lo <= hi, got {rng}")
        s_lo, _ = height_scale_range
        if s_lo <= 0:
            raise ValueError(
                f"height_scale_range lower bound must be > 0, got {height_scale_range}"
            )
        r_lo, _ = resolution_factor_range
        if r_lo <= 0:
            raise ValueError(
                f"resolution_factor_range lower bound must be > 0, got {resolution_factor_range}"
            )
        if not 0.0 <= salt_pepper_prob <= 1.0:
            raise ValueError(f"salt_pepper_prob must be in [0, 1], got {salt_pepper_prob}")
        if not 0.0 <= salt_pepper_amount <= 1.0:
            raise ValueError(f"salt_pepper_amount must be in [0, 1], got {salt_pepper_amount}")

        self.height_scale_range = tuple(map(float, height_scale_range))
        self.gaussian_blur_sigma_range = tuple(map(float, gaussian_blur_sigma_range))
        self.resolution_factor_range = tuple(map(float, resolution_factor_range))
        self.height_noise_sigma_range = tuple(map(float, height_noise_sigma_range))
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.salt_pepper_amount = float(salt_pepper_amount)

    def __call__(self, ndsm: np.ndarray) -> np.ndarray:
        """Produce one alignment-preserving augmented view.

        Args:
            ndsm: ``[H, W]`` float32 source CHM (typically the GT nDSM
                  or a copy of the main-view corrupted CHM).

        Returns:
            ``[H, W]`` float32 augmented view, same shape as input.
        """
        chm = ndsm.copy()

        s_lo, s_hi = self.height_scale_range
        if s_hi != 1.0 or s_lo != 1.0:
            chm = chm * float(np.random.uniform(s_lo, s_hi))

        b_lo, b_hi = self.gaussian_blur_sigma_range
        if b_hi > 0:
            sigma = float(np.random.uniform(b_lo, b_hi))
            if sigma > 1e-3:
                from scipy.ndimage import gaussian_filter
                chm = gaussian_filter(chm, sigma=sigma)

        r_lo, r_hi = self.resolution_factor_range
        if r_lo < 1.0 or r_hi < 1.0:
            r = float(np.random.uniform(r_lo, r_hi))
            if r < 0.999:
                h, w = chm.shape
                small = zoom(chm, r, order=1)
                chm = zoom(small, (h / small.shape[0], w / small.shape[1]), order=1)
                chm = chm[:h, :w]

        n_lo, n_hi = self.height_noise_sigma_range
        if n_hi > 0:
            sigma = float(np.random.uniform(n_lo, n_hi))
            if sigma > 1e-3:
                chm = chm + np.random.randn(*chm.shape).astype(np.float32) * sigma

        if self.salt_pepper_prob > 0 and random.random() < self.salt_pepper_prob:
            mask = np.random.rand(*chm.shape) < self.salt_pepper_amount
            chm = chm.copy()
            chm[mask] = 0.0

        np.clip(chm, 0, None, out=chm)
        return chm.astype(np.float32)


class SynRS3DHeightDataset(Dataset):
    """Dataset for SynRS3D height estimation.

    Expected directory layout (per subset)::

        data_dir/
        ├── grid_g005_high_v1/
        │   ├── opt/          ← RGB images  (uint8 TIF)
        │   ├── gt_nDSM/      ← height maps (float32 TIF)
        │   ├── gt_ss_mask/   ← land-cover  (uint8 TIF)
        │   └── train.txt     ← sample IDs (one per line)
        ├── grid_g005_mid_v1/
        │   └── ...

    Args:
        data_dir:                       Root containing the subset folders.
        subsets:                        List of subset folder names to
                                        include.
        normalisation:                  Image normalisation method
                                        (``"8bit"``, ``"90p"``, …).
        chm_corruption:                 Dict of kwargs for
                                        :class:`CHMCorruptor`.
        chm_contrastive_corruption:     Dict of kwargs for
                                        :class:`CHMContrastiveCorruptor`.
                                        When provided, ``__getitem__``
                                        returns a 5-tuple
                                        ``(image, chm, chm_v1, chm_v2,
                                        gt_height)`` for VICReg
                                        training. Two independent
                                        alignment-preserving views are
                                        sampled from the **GT nDSM**
                                        (not the main corrupted CHM) so
                                        the contrastive supervision
                                        sees a clean substrate.
        transforms_config:              Dict of kwargs for
                                        :class:`Transformations`.
        channels:                       Number of image channels to read.
    """

    def __init__(
        self,
        data_dir: str,
        subsets: list | None = None,
        normalisation: str = "8bit",
        chm_corruption: dict | None = None,
        chm_contrastive_corruption: dict | None = None,
        transforms_config: dict | None = None,
        channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.channels = channels
        self.normalize = Normalization(method=normalisation)

        self.corruptor = CHMCorruptor(**(chm_corruption or {}))
        # Optional alignment-preserving auxiliary corruptor for VICReg
        # contrastive views. Two views per sample are drawn from the GT
        # nDSM (clean substrate) using independently-sampled augmentation
        # parameters per view, then carried through the joint
        # ``Transformations.apply`` step alongside the main CHM so all
        # three CHMs (main + v1 + v2) receive the *same* spatial
        # transform (crop / flip / rotate). This preserves per-token
        # alignment between the two contrastive views, which the loss
        # requires.
        self.contrastive_corruptor: CHMContrastiveCorruptor | None = None
        if chm_contrastive_corruption is not None:
            self.contrastive_corruptor = CHMContrastiveCorruptor(
                **dict(chm_contrastive_corruption)
            )

        self.transformations = None
        if transforms_config is not None:
            self.transformations = Transformations(**dict(transforms_config))

        self.samples: list[dict] = []
        self._discover_samples(subsets)
        logger.info(f"SynRS3DHeightDataset: {len(self.samples)} samples from {subsets}")

    def _discover_samples(self, subsets: list | None):
        if subsets is None:
            subsets = sorted(
                d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            )

        for subset in subsets:
            subset_dir = os.path.join(self.data_dir, subset)
            train_txt = os.path.join(subset_dir, "train.txt")

            if os.path.exists(train_txt):
                with open(train_txt) as f:
                    ids = [line.strip() for line in f if line.strip()]
            else:
                ids = sorted(
                    os.path.splitext(f)[0]
                    for f in os.listdir(os.path.join(subset_dir, "opt"))
                    if f.endswith(".tif")
                )

            for sample_id in ids:
                opt_path = os.path.join(subset_dir, "opt", f"{sample_id}.tif")
                ndsm_path = os.path.join(subset_dir, "gt_nDSM", f"{sample_id}.tif")
                ss_path = os.path.join(subset_dir, "gt_ss_mask", f"{sample_id}.tif")

                if os.path.exists(opt_path) and os.path.exists(ndsm_path):
                    self.samples.append({
                        "opt": opt_path,
                        "ndsm": ndsm_path,
                        "ss": ss_path if os.path.exists(ss_path) else None,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_tif(path: str) -> np.ndarray:
        """Read a TIF file. Tries rasterio first, falls back to PIL."""
        try:
            with rasterio.open(path) as src:
                return src.read()  # [C, H, W]
        except Exception:
            img = np.array(Image.open(path))
            if img.ndim == 2:
                return img[np.newaxis]  # [1, H, W]
            return img.transpose(2, 0, 1)  # [C, H, W]

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # --- read ---
        image = self._read_tif(sample["opt"])[:self.channels]  # [3, H, W]
        ndsm = self._read_tif(sample["ndsm"])[0]               # [H, W] float32
        ss_mask = None
        if sample["ss"] is not None:
            ss_mask = self._read_tif(sample["ss"])[0]           # [H, W] uint8

        # --- normalise image ---
        image = self.normalize.apply(image)  # [C, H, W] float in [0, 1]

        # --- corrupt CHM (main height-prediction stream) ---
        if ss_mask is not None:
            chm = self.corruptor(ndsm, ss_mask)
        else:
            chm = self.corruptor(ndsm, np.zeros_like(ndsm, dtype=np.uint8))

        gt_height = ndsm.astype(np.float32)  # [H, W]

        # --- contrastive views (alignment-preserving, drawn from GT) ---
        # Sampled from a clean substrate (the GT nDSM) so the invariance
        # signal isn't confounded with shared cutout/dropout artefacts
        # of the main view. The two views differ only in the per-call
        # random parameters of the contrastive corruptor (different
        # noise realisations, blur σ, etc.).
        chm_v1: np.ndarray | None = None
        chm_v2: np.ndarray | None = None
        if self.contrastive_corruptor is not None:
            chm_v1 = self.contrastive_corruptor(ndsm)
            chm_v2 = self.contrastive_corruptor(ndsm)

        # --- joint augmentation (same spatial op on every CHM) ---
        # ``Transformations.apply`` runs the same crop / flip / rotate
        # on the image and on each entry of the masks list, so the two
        # contrastive views inherit identical spatial transforms as the
        # main CHM, preserving per-token alignment between v1 and v2.
        if self.transformations is not None:
            mask_list = [chm, gt_height]
            if chm_v1 is not None:
                mask_list.extend([chm_v1, chm_v2])
            image, transformed = self.transformations.apply(image, mask_list)
            chm, gt_height = transformed[0], transformed[1]
            if chm_v1 is not None:
                chm_v1, chm_v2 = transformed[2], transformed[3]

        # --- to tensors ---
        image = torch.as_tensor(image, dtype=torch.float32)
        chm = torch.as_tensor(chm, dtype=torch.float32)
        gt_height = torch.as_tensor(gt_height, dtype=torch.float32)

        if chm.ndim == 2:
            chm = chm.unsqueeze(0)        # [1, H, W]
        if gt_height.ndim == 2:
            gt_height = gt_height.unsqueeze(0)  # [1, H, W]

        if chm_v1 is None:
            return image, chm, gt_height

        chm_v1 = torch.as_tensor(chm_v1, dtype=torch.float32)
        chm_v2 = torch.as_tensor(chm_v2, dtype=torch.float32)
        if chm_v1.ndim == 2:
            chm_v1 = chm_v1.unsqueeze(0)
        if chm_v2.ndim == 2:
            chm_v2 = chm_v2.unsqueeze(0)
        return image, chm, chm_v1, chm_v2, gt_height
