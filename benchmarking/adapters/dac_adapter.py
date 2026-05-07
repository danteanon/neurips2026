"""Adapter for Depth Any Canopy (DAC) — Depth Anything v2 fine-tuned for CHM.

Monocular canopy height estimation from RGB.  This is the strongest
"foundation-model fine-tuned for CHM" baseline available with public
weights, and lives at the same architectural class as Depth Anything v2
(DINOv2 ViT + DPT decoder), so it makes a clean comparison against
CHMv2 (DINOv3 ViT + DPT) and against our CATT (DINOv3 ViT + cross-attn
prompt fusion + DPT).

Reference: Rege Cambrin, Corley & Garza, "Depth Any Canopy" (ECCV-W 2024).
HuggingFace: ``DarthReca/depth-any-canopy-base`` (97.5M params, DINOv2 ViT-B
encoder + DPT head, fine-tuned on filtered EarthView NEON tiles at 0.1m
RGB / 1m CHM).  A small variant (``-small``, 24.8M) is also available.

Important — image normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The training config (``configs/dataset/earthview.yaml``) specifies
**ImageNet normalisation** ``mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]`` applied to ``[0, 1]`` rescaled images.
The HF processor uploaded with the checkpoint has
``do_rescale=False, do_normalize=False`` — that's a misconfiguration
in the model card; following the training YAML is the correct path.

Important — output units / scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The model is trained with ``max_depth=1.0`` (see
``configs/model/depthany_*.yaml``) and target ``chm_uint8 / 255``
(see ``dataset/earthview_neon.py``), so the prediction is in ``[0, 1]``.

To map back to metres we need to know what one uint8 unit of NEON
CHM represents.  NEON CHM has **0.1 m vertical precision** (canonical
NEON product spec), so ``1 uint8 unit = 0.1 m`` and ``max_height = 25.5 m``
(saturating taller conifers — a known limitation of this checkpoint).

Therefore::

    height_metres = clip(prediction, 0, 1) * 255 * 0.1   # ≈ * 25.5

This is consistent with empirical least-squares calibration on DFC val
tree pixels (best pure scale ≈ 27.2; theoretical scale = 25.5).

Caveat: DAC was trained only on NEON (US, RGB).  Predictions on out-of-
domain imagery (DFC18 Houston, DFC19 Jacksonville/Omaha, Open-Canopy
SPOT France) carry a real domain gap.  Use these numbers as a baseline,
not as ground truth.
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

DEFAULT_MODEL_ID = "DarthReca/depth-any-canopy-base"

# Per the training YAML (configs/dataset/earthview.yaml), DAC was trained
# with standard ImageNet normalisation on rescaled [0, 1] images.
DAC_MEAN = (0.485, 0.456, 0.406)
DAC_STD = (0.229, 0.224, 0.225)

# Target during training was ``chm_uint8 / 255`` with NEON's 0.1 m precision
# stored as uint8 → 1 unit = 0.1 m.  See module docstring for derivation.
DAC_HEIGHT_SCALE = 25.5  # = 255 * 0.1 m/unit


class Adapter(ModelAdapter):
    """Depth Any Canopy adapter.

    The ``checkpoint_path`` argument is interpreted as either:
      * a local snapshot directory (offline mode), or
      * an HF hub id (default: ``DarthReca/depth-any-canopy-base``) when
        empty / non-existent.

    Pass ``--checkpoint DarthReca/depth-any-canopy-small`` to use the
    smaller 24.8M-param variant instead.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self._model_id = model_id
        self._model = None
        self._processor = None
        self._device: Optional[torch.device] = None

    def load(self, checkpoint_path: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = torch.device(device)

        if checkpoint_path and Path(checkpoint_path).exists():
            model_src = checkpoint_path
        elif checkpoint_path:
            # treat non-empty non-path string as an HF hub id override
            model_src = checkpoint_path
        else:
            model_src = self._model_id

        # Override the (mis-configured) saved processor to match the training
        # YAML: rescale uint8 → [0, 1] then ImageNet-normalise.
        self._processor = AutoImageProcessor.from_pretrained(model_src)
        self._processor.do_rescale = True
        self._processor.do_normalize = True
        self._processor.image_mean = list(DAC_MEAN)
        self._processor.image_std = list(DAC_STD)

        self._model = AutoModelForDepthEstimation.from_pretrained(model_src)
        self._model.eval().to(self._device)

        log.info("Depth Any Canopy loaded on %s from %s", device, model_src)

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        # DAC is monocular; chm is ignored.
        h, w = image.shape[:2]

        if image.ndim == 2:
            img_rgb = np.stack([image, image, image], axis=-1)
        else:
            # Strip ndarray subclass attrs (e.g. MSIImageWrapper) and drop NIR.
            img_rgb = np.asarray(image[..., :3])

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
        # Match training: clamp to [0, 1], then denormalise to metres.
        depth = np.clip(depth, 0.0, 1.0) * DAC_HEIGHT_SCALE
        return depth.astype(np.float32)

    @property
    def name(self) -> str:
        return "dac_dav2_b"

    @property
    def accepts_chm(self) -> bool:
        return False
