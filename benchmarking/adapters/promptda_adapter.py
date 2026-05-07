"""Adapter for Prompt Depth Anything (ViT-L / ViT-S).

The model expects:
  - image:  [B, 3, H, W] float [0,1]; ImageNet norm applied internally.
  - prompt: [B, 1, H, W] float metres; min-max norm applied internally.
  - H, W must be divisible by 14.

Output: [B, 1, H, W] float metres (after internal denormalization).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROMPTDA_ROOT = PACKAGE_ROOT / "benchmarking" / "repos" / "PromptDA"
sys.path.insert(0, str(PROMPTDA_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "benchmarking"))

from eval.corruption_sweep import ModelAdapter  # noqa: E402


def _pad_to_multiple(arr: np.ndarray, multiple: int = 14):
    """Pad (H, W, ...) or (H, W) array so both dims are divisible by *multiple*.
    Returns (padded_array, original_h, original_w)."""
    h, w = arr.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    if new_h == h and new_w == w:
        return arr, h, w
    if arr.ndim == 3:
        padded = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
        padded[:h, :w, :] = arr
    else:
        padded = np.zeros((new_h, new_w), dtype=arr.dtype)
        padded[:h, :w] = arr
    return padded, h, w


class Adapter(ModelAdapter):

    def __init__(self, model_id: str = "depth-anything/prompt-depth-anything-vitl"):
        self._model_id = model_id
        self._model = None
        self._device = None

    def load(self, checkpoint_path: str, device: str) -> None:
        self._device = torch.device(device)

        from promptda.promptda import PromptDA

        path = Path(checkpoint_path)
        if path.exists() and path.suffix == ".ckpt":
            self._model = PromptDA.from_pretrained(checkpoint_path)
        else:
            self._model = PromptDA.from_pretrained(self._model_id)

        self._model.to(self._device).eval()

    def predict(self, image: np.ndarray, chm: np.ndarray) -> np.ndarray:
        image_pad, orig_h, orig_w = _pad_to_multiple(image, 14)
        chm_pad, _, _ = _pad_to_multiple(chm, 14)

        img_t = (
            torch.from_numpy(image_pad.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )

        chm_t = (
            torch.from_numpy(chm_pad.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            pred = self._model.predict(img_t, chm_t)

        out = pred.squeeze().cpu().numpy()
        return out[:orig_h, :orig_w]

    @property
    def name(self) -> str:
        return "promptda"
