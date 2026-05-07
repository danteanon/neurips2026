"""Adapter for Meta CHMv2 (DINOv3 ViT-L + DPT head).

Monocular canopy height estimation from satellite RGB.  The model has the
same backbone family as our CATT — this is the most controlled comparison
possible to demonstrate the value of cross-attention CHM prompting.

HuggingFace model: ``facebook/dinov3-vitl16-chmv2-dpt-head`` (gated, requires
HF login + acceptance of Meta's terms).

The model returns absolute canopy height in metres (float32) with
``min_depth=0.001`` and ``max_depth=96.0`` per its config.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "benchmarking"))

from eval.corruption_sweep import ModelAdapter  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-chmv2-dpt-head"


class Adapter(ModelAdapter):

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._device: Optional[torch.device] = None

    def load(self, checkpoint_path: str, device: str) -> None:
        from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessor

        self._device = torch.device(device)

        # checkpoint_path may be a local snapshot dir or empty (use HF hub)
        model_src = checkpoint_path if (
            checkpoint_path and Path(checkpoint_path).exists()
        ) else self._model_id

        self._processor = CHMv2ImageProcessor.from_pretrained(model_src)
        self._model = CHMv2ForDepthEstimation.from_pretrained(model_src)
        self._model.eval().to(self._device)

        log.info("CHMv2 (DINOv3 ViT-L + DPT) loaded on %s from %s",
                 device, model_src)

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        # CHMv2 is monocular — chm is ignored.
        # Strip any custom ndarray subclass attributes (e.g. MSIImageWrapper)
        # so the HF processor sees a plain ndarray.
        h, w = image.shape[:2]
        if image.ndim == 2:
            img_rgb = np.stack([image, image, image], axis=-1)
        else:
            img_rgb = np.asarray(image[..., :3])  # drop NIR if present

        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        inputs = self._processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        result = self._processor.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)],
        )[0]
        depth = result["predicted_depth"].cpu().numpy()
        return depth.astype(np.float32)

    @property
    def name(self) -> str:
        return "chmv2_dinov3_vitl"

    @property
    def accepts_chm(self) -> bool:
        return False
